"""Generate imitation-learning datasets for cribbage.

What you get:
  - discard.npz: X_discard (N, Dd), y_discard (N,)
  - pegging.npz: X_pegging (M, Dp), y_pegging (M,)

Labels are deliberately simple:
  - discard label: kept_score +/− crib_score (depending on dealer_is_self)
  - pegging label: immediate pegging points for playing candidate card

This is meant to be the *first* data pipeline. Once this works end-to-end,
upgrade the targets (Monte Carlo / rollout / self-play RL).
"""

from __future__ import annotations
import os
from pathlib import Path
import sys

from cribbage.cribbagegame import score_hand, score_play as score_play
from cribbage.players.rule_based_player import get_full_deck


sys.path.insert(0, ".")
from crib_ai_trainer.constants import (
    TRAINING_DATA_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_DATASET_RUN_ID,
    DEFAULT_STRATEGY,
    DEFAULT_PEGGING_FEATURE_SET,
    DEFAULT_GAMES_PER_LOOP,
    DEFAULT_SEED,
    DEFAULT_USE_RANDOM_SEED,
    DEFAULT_CRIB_EV_MODE,
    DEFAULT_CRIB_MC_SAMPLES,
    DEFAULT_PEGGING_LABEL_MODE,
    DEFAULT_PEGGING_ROLLOUTS,
    DEFAULT_WIN_PROB_MODE,
    DEFAULT_WIN_PROB_ROLLOUTS,
    DEFAULT_WIN_PROB_MIN_SCORE,
    DEFAULT_PEGGING_EV_MODE,
    DEFAULT_PEGGING_EV_ROLLOUTS,
)
import argparse
from itertools import combinations
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from cribbage import cribbagegame
from cribbage.playingcards import Card, build_hand
from cribbage.strategies.pegging_strategies import (
    medium_pegging_strategy_scores,
    medium_pegging_strategy,
    get_highest_rank_card,
)
from cribbage.strategies.hand_strategies import process_dealt_hand_only_exact, exact_hand_and_min_crib
from cribbage.strategies.crib_strategies import calc_crib_min_only_given_6_cards
from cribbage.database import normalize_hand_to_str

from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.medium_player import MediumPlayer
from crib_ai_trainer.players.neural_player import (
    featurize_discard,
    featurize_pegging,
    DISCARD_FEATURE_DIM,
    get_pegging_feature_dim,
    estimate_pegging_ev_mc_for_discard,
)

from cribbage.cribbagegame import score_hand, score_play

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import random
import secrets
from itertools import combinations
import json
from datetime import datetime, timezone

RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
RANK_TO_VALUE = {
    "a": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "j": 10,
    "q": 10,
    "k": 10,
}

def remaining_rank_counts(
    known_cards: List[Card],
    all_played_cards: List[Card],
    hand: List[Card],
) -> dict[str, int]:
    counts = {r: 4 for r in RANKS}
    for c in list(known_cards) + list(all_played_cards) + list(hand):
        rank = c.get_rank().lower()
        counts[rank] = max(0, counts.get(rank, 0) - 1)
    return counts


def sample_rank_from_counts(counts: dict[str, int], rng: random.Random) -> str:
    total = sum(counts.values())
    if total <= 0:
        return "a"
    r = rng.randrange(total)
    running = 0
    for rank, count in counts.items():
        running += count
        if r < running:
            return rank
    return "a"


def exact_expected_pegging_value_ranked(
    hand: List[Card],
    table: List[Card],
    count: int,
    candidate: Card,
    known_cards: List[Card],
    all_played_cards: List[Card],
) -> float:
    """Exact expected (our pts - opp pts) by iterating remaining ranks."""
    seq_after = table + [candidate]
    our_points = float(score_play(seq_after)[0])
    counts = remaining_rank_counts(known_cards, all_played_cards, hand)
    total = sum(counts.values())
    if total == 0:
        return our_points
    acc = 0.0
    for rank, count_remain in counts.items():
        if count_remain <= 0:
            continue
        opp_value = RANK_TO_VALUE[rank]
        if count + candidate.get_value() + opp_value > 31:
            opp_points = 0.0
        else:
            opp_points = float(score_play(seq_after + [Card(f"{rank}h")])[0])
        acc += (our_points - opp_points) * float(count_remain)
    return acc / float(total)


def exact_expected_pegging_value_ranked_2ply(
    hand: List[Card],
    table: List[Card],
    count: int,
    candidate: Card,
    known_cards: List[Card],
    all_played_cards: List[Card],
) -> float:
    """Exact expected value with one opponent reply and one of our replies."""
    seq_after = table + [candidate]
    our_points = float(score_play(seq_after)[0])
    counts = remaining_rank_counts(known_cards, all_played_cards, hand)
    total = sum(counts.values())
    if total == 0:
        return our_points
    acc = 0.0
    for rank, count_remain in counts.items():
        if count_remain <= 0:
            continue
        opp_value = RANK_TO_VALUE[rank]
        if count + candidate.get_value() + opp_value > 31:
            opp_points = 0.0
            our_next = 0.0
        else:
            seq_after_opp = seq_after + [Card(f"{rank}h")]
            opp_points = float(score_play(seq_after_opp)[0])
            new_count = count + candidate.get_value() + opp_value
            playable_next = [c for c in hand if c.get_value() + new_count <= 31]
            if playable_next:
                # use mean immediate points across our playable next cards
                vals = [float(score_play(seq_after_opp + [c])[0]) for c in playable_next]
                our_next = float(sum(vals) / len(vals))
            else:
                our_next = 0.0
        acc += (our_points - opp_points + our_next) * float(count_remain)
    return acc / float(total)

def estimate_discard_value_mc(
    hand: List[Card],
    kept: List[Card],
    discards: List[Card],
    dealer_is_self: bool,
    full_deck: List[Card],
    rng: random.Random,
    n_starters: int = 16,
    n_opp_discards: int = 8,
) -> float:
    # remaining cards (46)
    hand_set = set(hand)
    remaining: List[Card] = [c for c in full_deck if c not in hand_set]
    n = len(remaining)
    if n == 0:
        return 0.0

    k_starters = min(n_starters, n)

    # sample starter indices without copying big lists repeatedly
    # if k_starters is small, randint loop is fine; if large, sample indices once.
    starter_idxs = rng.sample(range(n), k=k_starters)

    kept_plus = list(kept) + [None]  # type: ignore
    crib_cards = [discards[0], discards[1], None, None, None]  # 2 discards + 2 opp + starter

    total = 0.0
    for si in starter_idxs:
        starter = remaining[si]

        # hand pts with starter
        kept_plus[-1] = starter
        hand_pts = score_hand(kept_plus, is_crib=False)

        # crib EV: sample opponent discards excluding starter, without building rem2
        crib_total = 0.0
        for _ in range(n_opp_discards):
            # choose two distinct indices not equal to si
            i = rng.randrange(n)
            while i == si:
                i = rng.randrange(n)

            j = rng.randrange(n - 1)
            # map j into [0..n-1] excluding i
            if j >= i:
                j += 1
            while j == si:
                j = rng.randrange(n - 1)
                if j >= i:
                    j += 1

            opp1 = remaining[i]
            opp2 = remaining[j]

            crib_cards[2] = opp1
            crib_cards[3] = opp2
            crib_cards[4] = starter
            crib_total += score_hand(crib_cards, is_crib=True)

        crib_ev = crib_total / float(n_opp_discards) if n_opp_discards > 0 else 0.0

        total += hand_pts + (crib_ev if dealer_is_self else -crib_ev)

    return total / float(k_starters)

def estimate_discard_value_mc_fast_from_remaining(
    kept: List[Card],
    discards: List[Card],
    dealer_is_self: bool,
    remaining: List[Card],
    rng: random.Random,
    n_starters: int = 16,
    n_opp_discards: int = 8,
) -> float:
    n = len(remaining)
    if n == 0:
        return 0.0

    k_starters = min(n_starters, n)
    starter_idxs = rng.sample(range(n), k=k_starters)

    kept_plus = list(kept) + [None]  # type: ignore
    crib_cards = [discards[0], discards[1], None, None, None]  # type: ignore

    total = 0.0
    for si in starter_idxs:
        starter = remaining[si]

        kept_plus[-1] = starter
        hand_pts = score_hand(kept_plus, is_crib=False)

        crib_total = 0.0
        for _ in range(n_opp_discards):
            i = rng.randrange(n)
            while i == si:
                i = rng.randrange(n)

            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            while j == si:
                j = rng.randrange(n - 1)
                if j >= i:
                    j += 1

            crib_cards[2] = remaining[i]
            crib_cards[3] = remaining[j]
            crib_cards[4] = starter
            crib_total += score_hand(crib_cards, is_crib=True)

        crib_ev = crib_total / float(n_opp_discards) if n_opp_discards else 0.0
        total += hand_pts + (crib_ev if dealer_is_self else -crib_ev)

    return total / float(k_starters)


def estimate_crib_ev_mc_from_remaining(
    discards: List[Card],
    remaining: List[Card],
    rng: random.Random,
    n_samples: int = 32,
) -> float:
    """Estimate crib EV by sampling opponent discards and starter cards."""
    if n_samples <= 0 or not remaining:
        return 0.0
    total = 0.0
    n = len(remaining)
    index_of = {c: i for i, c in enumerate(remaining)}
    for _ in range(n_samples):
        starter = remaining[rng.randrange(n)]
        starter_idx = index_of[starter]
        # sample two opponent discards (distinct from starter)
        i = rng.randrange(n)
        while i == starter_idx:
            i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        while j == starter_idx:
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
        opp1 = remaining[i]
        opp2 = remaining[j]
        crib_cards = [discards[0], discards[1], opp1, opp2, starter]
        total += score_hand(crib_cards, is_crib=True)
    return total / float(n_samples)


def estimate_pegging_rollout_value(
    hand: List[Card],
    table: List[Card],
    count: int,
    candidate: Card,
    known_cards: List[Card],
    all_played_cards: List[Card],
    rng: random.Random,
    n_rollouts: int = 32,
) -> float:
    """One-step rollout: our immediate points minus expected opponent immediate points."""
    seq_after = table + [candidate]
    our_points = float(score_play(seq_after)[0])
    if n_rollouts <= 0:
        return our_points
    if n_rollouts >= 13:
        return exact_expected_pegging_value_ranked(
            hand,
            table,
            count,
            candidate,
            known_cards,
            all_played_cards,
        )
    total = 0.0
    rank_counts = remaining_rank_counts(known_cards, all_played_cards, hand)
    for _ in range(n_rollouts):
        rank = sample_rank_from_counts(rank_counts, rng)
        opp_card = Card(f"{rank}h")
        if count + candidate.get_value() + opp_card.get_value() > 31:
            opp_points = 0.0
        else:
            opp_points = float(score_play(seq_after + [opp_card])[0])
        total += (our_points - opp_points)
    return total / float(n_rollouts)


def estimate_pegging_rollout_value_2ply(
    hand: List[Card],
    table: List[Card],
    count: int,
    candidate: Card,
    known_cards: List[Card],
    all_played_cards: List[Card],
    rng: random.Random,
    n_rollouts: int = 32,
) -> float:
    """Two-step rollout: our immediate points - opponent immediate points + our next immediate points."""
    seq_after = table + [candidate]
    our_points = float(score_play(seq_after)[0])
    if n_rollouts <= 0:
        return our_points
    if n_rollouts >= 13:
        return exact_expected_pegging_value_ranked_2ply(
            hand,
            table,
            count,
            candidate,
            known_cards,
            all_played_cards,
        )
    total = 0.0
    rank_counts = remaining_rank_counts(known_cards, all_played_cards, hand)
    for _ in range(n_rollouts):
        rank = sample_rank_from_counts(rank_counts, rng)
        opp_card = Card(f"{rank}h")
        if count + candidate.get_value() + opp_card.get_value() > 31:
            opp_points = 0.0
            our_next = 0.0
        else:
            seq_after_opp = seq_after + [opp_card]
            opp_points = float(score_play(seq_after_opp)[0])
            new_count = count + candidate.get_value() + opp_card.get_value()
            playable_next = [c for c in hand if c.get_value() + new_count <= 31]
            if playable_next:
                next_card = playable_next[rng.randrange(len(playable_next))]
                our_next = float(score_play(seq_after_opp + [next_card])[0])
            else:
                our_next = 0.0
        total += (our_points - opp_points + our_next)
    return total / float(n_rollouts)


def simulate_pegging_points(
    hand_self: List[Card],
    hand_opp: List[Card],
    *,
    dealer_is_self: bool,
    rng: random.Random,
    start_table: Optional[List[Card]] = None,
    start_count: int = 0,
    start_turn: int = 0,
) -> Tuple[int, int]:
    """Simulate pegging with medium strategy for both players.

    Returns pegging points for (self, opp).
    """
    table = list(start_table) if start_table else []
    count = int(start_count)
    hands = [list(hand_self), list(hand_opp)]
    scores = [0, 0]
    turn = int(start_turn)
    passes = 0
    last_played: Optional[int] = None

    while hands[0] or hands[1]:
        playable = [c for c in hands[turn] if c.get_value() + count <= 31]
        if not playable:
            passes += 1
            if passes >= 2:
                if last_played is not None:
                    scores[last_played] += 1
                table = []
                count = 0
                passes = 0
                last_played = None
            turn = 1 - turn
            continue

        passes = 0
        card = medium_pegging_strategy(playable, count, table) or playable[0]
        hands[turn].remove(card)
        table.append(card)
        count += card.get_value()
        scores[turn] += int(score_play(table)[0])

        if count == 31:
            scores[turn] += 2
            table = []
            count = 0
            last_played = None
            turn = 1 - turn
            continue

        last_played = turn
        turn = 1 - turn

    if count > 0 and last_played is not None:
        scores[last_played] += 1

    return int(scores[0]), int(scores[1])


def estimate_win_prob_discard_rollout(
    hand: List[Card],
    kept: List[Card],
    discards: List[Card],
    dealer_is_self: bool,
    player_score: int | None,
    opponent_score: int | None,
    rng: random.Random,
    n_rollouts: int,
    min_score_for_eval: int,
) -> float:
    """Estimate win probability by simulating the rest of the round."""
    if n_rollouts <= 0:
        return 0.5
    if player_score is None:
        player_score = 0
    if opponent_score is None:
        opponent_score = 0
    if max(player_score, opponent_score) < min_score_for_eval:
        return 0.5

    full_deck = get_full_deck()
    remaining = [c for c in full_deck if c not in set(hand)]
    if len(remaining) < 6:
        return 0.5

    wins = 0
    ties = 0
    for _ in range(n_rollouts):
        opp_hand = rng.sample(remaining, 6)
        opp_discards = list(exact_hand_and_min_crib(opp_hand, dealer_is_self=not dealer_is_self))
        opp_kept = [c for c in opp_hand if c not in opp_discards]

        remaining2 = [c for c in remaining if c not in opp_hand]
        if not remaining2:
            starter = remaining[rng.randrange(len(remaining))]
        else:
            starter = remaining2[rng.randrange(len(remaining2))]

        crib = [discards[0], discards[1], opp_discards[0], opp_discards[1]]

        start_turn = 1 if dealer_is_self else 0  # non-dealer leads
        pegging_self, pegging_opp = simulate_pegging_points(
            kept,
            opp_kept,
            dealer_is_self=dealer_is_self,
            rng=rng,
            start_table=[],
            start_count=0,
            start_turn=start_turn,
        )

        self_total = player_score + pegging_self + score_hand(kept + [starter], is_crib=False)
        opp_total = opponent_score + pegging_opp + score_hand(opp_kept + [starter], is_crib=False)

        crib_score = score_hand(crib + [starter], is_crib=True)
        if dealer_is_self:
            self_total += crib_score
        else:
            opp_total += crib_score

        if self_total > opp_total:
            wins += 1
        elif self_total == opp_total:
            ties += 1

    return (wins + 0.5 * ties) / float(n_rollouts)


class LoggedData:
    pass

@dataclass
class LoggedRegressionPegData(LoggedData):
    X_pegging: List[np.ndarray] = field(default_factory=list)
    y_pegging: List[float] = field(default_factory=list)

@dataclass
class LoggedRegressionDiscardData(LoggedData):
    X_discard: List[np.ndarray] = field(default_factory=list)
    y_discard: List[float] = field(default_factory=list)
    y_discard_win: List[float] = field(default_factory=list)

@dataclass
class LoggedClassificationDiscardData(LoggedData):
    X_discard: List[np.ndarray] = field(default_factory=list)   # each is (15, D)
    y_discard: List[int] = field(default_factory=list)         # each is 0..14

@dataclass
class LoggedRankingDiscardData(LoggedData):
    X_discard: List[np.ndarray] = field(default_factory=list)  # each is (15, D)
    y_discard: List[np.ndarray] = field(default_factory=list)  # each is (15,)

@dataclass
class LoggedRegPegRegDiscardData(LoggedRegressionPegData, LoggedRegressionDiscardData):
    pass

@dataclass
class LoggedRegPegClasDiscardData(LoggedRegressionPegData, LoggedClassificationDiscardData):
    pass

@dataclass
class LoggedRegPegRankDiscardData(LoggedRegressionPegData, LoggedRankingDiscardData):
    pass


class LoggingBeginnerPlayer(BeginnerPlayer):
    """Wrap BeginnerPlayer so we can collect training data while it plays."""

    def __init__(
        self,
        name: str,
        log: LoggedData,
        discard_strategy,
        pegging_strategy,
        seed: int = 0,
        pegging_feature_set: str = "full",
        pegging_label_mode: str = "immediate",
        pegging_rollouts: int = 32,
        win_prob_mode: str = "off",
        win_prob_rollouts: int = 16,
        win_prob_min_score: int = 90,
        pegging_ev_mode: str = "off",
        pegging_ev_rollouts: int = 16,
    ):
        super().__init__(name=name)
        self._rng = random.Random(seed)
        self._full_deck = get_full_deck()
        self._log = log
        if discard_strategy == "classification":
            self._discard_strategy = self.select_crib_cards_classifier
        elif discard_strategy == "regression":
            self._discard_strategy = self.select_crib_cards_regresser
        else:
            raise ValueError(f"Unknown discard_strategy: {discard_strategy}")        
        self._pegging_strategy = pegging_strategy # not implemented yet
        self._pegging_feature_set = pegging_feature_set
        self._pegging_label_mode = pegging_label_mode
        self._pegging_rollouts = pegging_rollouts
        self._win_prob_mode = win_prob_mode
        self._win_prob_rollouts = win_prob_rollouts
        self._win_prob_min_score = win_prob_min_score
        self._pegging_ev_mode = pegging_ev_mode
        self._pegging_ev_rollouts = pegging_ev_rollouts
        self._rng_np = np.random.default_rng(seed)

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        # Extract hand and dealer info from state objects
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        return self._discard_strategy(hand, dealer_is_self, player_state.score, player_state.opponent_score)
    
    def select_crib_cards_classifier(
        self,
        hand: List[Card],
        dealer_is_self: bool,
        your_score: int | None = None,
        opponent_score: int | None = None,
    ) -> Tuple[Card, Card]:
        hand_set = set(hand)
        remaining = [c for c in self._full_deck if c not in hand_set]  # 46

        Xs: List[np.ndarray] = []
        ys: List[float] = []
        discards_list: List[Tuple[Card, Card]] = []

        for kept in combinations(hand, 4):
            kept = list(kept)
            discards = [c for c in hand if c not in kept]
            disc_t = (discards[0], discards[1])

            y = estimate_discard_value_mc_fast_from_remaining(
                kept=kept,
                discards=discards,
                dealer_is_self=dealer_is_self,
                remaining=remaining,
                rng=self._rng,
                n_starters=16,
                n_opp_discards=8,
            )

            pegging_ev = None
            if self._pegging_ev_mode == "rollout":
                pegging_ev = estimate_pegging_ev_mc_for_discard(
                    hand,
                    kept,
                    discards,
                    dealer_is_self,
                    self._rng_np,
                    self._pegging_ev_rollouts,
                )
            x = featurize_discard(kept, discards, dealer_is_self, your_score, opponent_score, pegging_ev=pegging_ev)
            Xs.append(x)
            ys.append(float(y))
            discards_list.append(disc_t)

        best_i = int(np.argmax(np.array(ys, dtype=np.float32)))

        # log ONE example for this 6-card hand:
        # X_hand is (15, D), label is best option index 0..14
        X_hand = np.stack(Xs, axis=0).astype(np.float32)  # (15, D)
        self._log.X_discard.append(X_hand) # type: ignore
        self._log.y_discard.append(best_i) # type: ignore

        return discards_list[best_i]

    def select_crib_cards_regresser(
        self,
        hand: List[Card],
        dealer_is_self: bool,
        your_score: int | None = None,
        opponent_score: int | None = None,
    ) -> Tuple[Card, Card]:
        hand_set = set(hand)
        remaining = [c for c in self._full_deck if c not in hand_set]  # 46

        best_y = float("-inf")
        best_discards: List[Tuple[Card, Card]] = []

        for kept in combinations(hand, 4):
            kept = list(kept)
            discards = [c for c in hand if c not in kept]

            y = estimate_discard_value_mc_fast_from_remaining(
                kept=kept,
                discards=discards,
                dealer_is_self=dealer_is_self,
                remaining=remaining,
                rng=self._rng,
                n_starters=16,
                n_opp_discards=8,
            )
            if y < -8: 
                print("unusual y:", y)
                print("hand:", hand)
                print("kept:", kept)
                print("discards:", discards)
                a = 1

            pegging_ev = None
            if self._pegging_ev_mode == "rollout":
                pegging_ev = estimate_pegging_ev_mc_for_discard(
                    hand,
                    kept,
                    discards,
                    dealer_is_self,
                    self._rng_np,
                    self._pegging_ev_rollouts,
                )
            x = featurize_discard(kept, discards, dealer_is_self, your_score, opponent_score, pegging_ev=pegging_ev)
            self._log.X_discard.append(x)
            self._log.y_discard.append(float(y))
            if self._win_prob_mode == "rollout":
                y_win = estimate_win_prob_discard_rollout(
                    hand=hand,
                    kept=kept,
                    discards=discards,
                    dealer_is_self=dealer_is_self,
                    player_score=your_score,
                    opponent_score=opponent_score,
                    rng=self._rng,
                    n_rollouts=self._win_prob_rollouts,
                    min_score_for_eval=self._win_prob_min_score,
                )
                self._log.y_discard_win.append(float(y_win))

            if y > best_y:
                best_y = y
                best_discards = [tuple(discards)]
            elif y == best_y:
                best_discards.append(tuple(discards))

        return best_discards[0]

    def select_card_to_play(self, player_state, round_state) -> Optional[Card]:
        """Override to log training data and pass known_cards."""
        hand = player_state.hand
        table = round_state.table_cards
        crib = round_state.crib
        count = round_state.count
        
        # table is the list of cards currently on the table (current sequence since last reset)
        playable = [c for c in hand if c + count <= 31]
        if not playable:
            return None        
        # Log *all* playable options with immediate pegging reward.
        # score_play expects sequence since reset.
        # table is the list of cards currently on the table
        # always take points if available; else play lowest that doesn't set opponent up
        best = None
        best_pts = -1
        history_since_reset = table
        for c in playable:
            sequence = history_since_reset + [c]
            pts, _ = score_play(sequence)
            if self._pegging_label_mode == "rollout2":
                y = estimate_pegging_rollout_value_2ply(
                    hand,
                    history_since_reset,
                    count,
                    c,
                    known_cards=player_state.known_cards,
                    all_played_cards=round_state.all_played_cards,
                    rng=self._rng,
                    n_rollouts=self._pegging_rollouts,
                )
            elif self._pegging_label_mode == "rollout1":
                y = estimate_pegging_rollout_value(
                    hand,
                    history_since_reset,
                    count,
                    c,
                    known_cards=player_state.known_cards,
                    all_played_cards=round_state.all_played_cards,
                    rng=self._rng,
                    n_rollouts=self._pegging_rollouts,
                )
            else:
                y = float(pts)
            # Known cards: from player_state (includes hand, table, past cards, starter)
            x = featurize_pegging(
                hand,
                history_since_reset,
                count,
                c,
                known_cards=player_state.known_cards,
                opponent_known_hand=player_state.opponent_known_hand,
                all_played_cards=round_state.all_played_cards,
                player_score=player_state.score,
                opponent_score=player_state.opponent_score,
                feature_set=self._pegging_feature_set,
            )
            self._log.X_pegging.append(x) # type: ignore
            self._log.y_pegging.append(float(y)) # type: ignore
            if (pts > best_pts) and (c + count <= 31):
                best_pts = pts
                best = c
        if best is not None:
            return best
        # otherwise play highest value        
        highest_card = playable[0]
        for c in playable:
            if c > highest_card:
                highest_card = c
        return highest_card
    


class LoggingMediumPlayer(MediumPlayer):
    """Wrap MediumPlayer so we can collect training data while it plays."""

    def __init__(
        self,
        name: str,
        log: LoggedData,
        discard_strategy,
        pegging_strategy,
        seed: int = 0,
        pegging_feature_set: str = "full",
        crib_ev_mode: str = "min",
        crib_mc_samples: int = 32,
        pegging_label_mode: str = "immediate",
        pegging_rollouts: int = 32,
        win_prob_mode: str = "off",
        win_prob_rollouts: int = 16,
        win_prob_min_score: int = 90,
        pegging_ev_mode: str = "off",
        pegging_ev_rollouts: int = 16,
    ):
        super().__init__(name=name)
        self._rng = random.Random(seed)
        self._full_deck = get_full_deck()
        self._log = log
        if discard_strategy == "classification":
            self._discard_strategy_mode = "classification"
        elif discard_strategy == "ranking":
            self._discard_strategy_mode = "ranking"
        elif discard_strategy == "regression":
            self._discard_strategy_mode = "regression"
        else:
            raise ValueError(f"Unknown discard_strategy: {discard_strategy}")        
        self._pegging_strategy = pegging_strategy
        self._pegging_feature_set = pegging_feature_set
        self._crib_ev_mode = crib_ev_mode
        self._crib_mc_samples = crib_mc_samples
        self._pegging_label_mode = pegging_label_mode
        self._pegging_rollouts = pegging_rollouts
        self._win_prob_mode = win_prob_mode
        self._win_prob_rollouts = win_prob_rollouts
        self._win_prob_min_score = win_prob_min_score
        self._pegging_ev_mode = pegging_ev_mode
        self._pegging_ev_rollouts = pegging_ev_rollouts
        self._rng_np = np.random.default_rng(seed)

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        """Override to log training data."""
        # Extract hand and dealer info from state objects
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = player_state.opponent_score
        
        if self._discard_strategy_mode == "classification":
            return self.select_crib_cards_classifier(hand, dealer_is_self, your_score, opponent_score)
        elif self._discard_strategy_mode == "ranking":
            return self.select_crib_cards_ranking(hand, dealer_is_self, your_score, opponent_score)
        else:
            return self.select_crib_cards_regresser(hand, dealer_is_self, your_score, opponent_score)

    def select_card_to_play(self, player_state, round_state) -> Optional[Card]:
        """Override to log training data and pass known_cards to play_pegging."""
        playable_cards = [c for c in player_state.hand if c.get_value() + round_state.count <= 31]
        if not playable_cards:
            return None
        # Pass known_cards to play_pegging via instance variables
        self._current_known_cards = player_state.known_cards
        self._current_hand = list(player_state.hand)
        self._current_opponent_known_hand = list(player_state.opponent_known_hand)
        self._current_all_played_cards = list(round_state.all_played_cards)
        self._current_player_score = int(player_state.score)
        self._current_opponent_score = int(player_state.opponent_score)
        self._current_pegging_feature_set = self._pegging_feature_set
        return self.play_pegging(playable_cards, round_state.count, round_state.table_cards)
    
    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        """Override to log training data."""
        if not playable:
            return None
        
        # Check if any cards are actually playable
        have_playable_cards = False
        for c in playable:
            new_count = count + c.get_value()
            if new_count <= 31:
                have_playable_cards = True
                break
        
        if not have_playable_cards:
            return None
        
        scores = medium_pegging_strategy_scores(playable, count, history_since_reset)
        
        # Get known_cards from instance variable (set in select_card_to_play)
        known_cards = getattr(self, '_current_known_cards', [])
        full_hand = getattr(self, '_current_hand', playable)
        opp_known = getattr(self, "_current_opponent_known_hand", [])
        all_played = getattr(self, "_current_all_played_cards", [])
        player_score = getattr(self, "_current_player_score", 0)
        opponent_score = getattr(self, "_current_opponent_score", 0)
        feature_set = getattr(self, "_current_pegging_feature_set", "full")
        
        # Log all playable options
        for card, score in scores.items():
            if self._pegging_label_mode == "rollout2":
                y = estimate_pegging_rollout_value_2ply(
                    full_hand,
                    history_since_reset,
                    count,
                    card,
                    known_cards=known_cards,
                    all_played_cards=all_played,
                    rng=self._rng,
                    n_rollouts=self._pegging_rollouts,
                )
            elif self._pegging_label_mode == "rollout1":
                y = estimate_pegging_rollout_value(
                    full_hand,
                    history_since_reset,
                    count,
                    card,
                    known_cards=known_cards,
                    all_played_cards=all_played,
                    rng=self._rng,
                    n_rollouts=self._pegging_rollouts,
                )
            else:
                y = score
            # Known cards: from player_state (includes hand, table, past cards, starter)
            x = featurize_pegging(
                full_hand,
                history_since_reset,
                count,
                card,
                known_cards=known_cards,
                opponent_known_hand=opp_known,
                all_played_cards=all_played,
                player_score=player_score,
                opponent_score=opponent_score,
                feature_set=feature_set,
            )
            self._log.X_pegging.append(x) # type: ignore
            self._log.y_pegging.append(float(y)) # type: ignore
        
        max_v = max(scores.values())
        highest_scoring_cards_list = [k for k, v in scores.items() if v == max_v]
        highest_scoring_card = get_highest_rank_card(highest_scoring_cards_list)
        return highest_scoring_card
    
    def select_crib_cards_classifier(self, hand, dealer_is_self, your_score=None, opponent_score=None) -> Tuple[Card, Card]:                
        # don't analyze crib, just calculate min value of the crib and use that
        full_deck = get_full_deck()
        hand_score_cache = {}
        crib_score_cache = {}        
        hand_results = process_dealt_hand_only_exact([hand, full_deck, hand_score_cache])
        df_hand = pd.DataFrame(hand_results, columns=["hand_key","min_hand_score","max_hand_score","avg_hand_score"])
        if self._crib_ev_mode == "min":
            crib_results = calc_crib_min_only_given_6_cards(hand)
            df_crib = pd.DataFrame(crib_results, columns=["hand_key","crib_key","min_crib_score","avg_crib_score"])
            df3 = pd.merge(df_hand, df_crib, on=["hand_key"])
            df3["avg_total_score"] = df3["avg_hand_score"] + (df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"])
        else:
            # Monte Carlo crib EV
            hand_set = set(hand)
            remaining = [c for c in self._full_deck if c not in hand_set]
            rows = []
            for kept in combinations(hand, 4):
                kept_list = list(kept)
                discards_list_temp = [c for c in hand if c not in kept_list]
                hand_key = normalize_hand_to_str(kept_list)
                crib_key = normalize_hand_to_str(discards_list_temp)
                crib_ev = estimate_crib_ev_mc_from_remaining(
                    discards_list_temp,
                    remaining,
                    self._rng,
                    n_samples=self._crib_mc_samples,
                )
                rows.append([hand_key, crib_key, crib_ev])
            df_crib = pd.DataFrame(rows, columns=["hand_key","crib_key","avg_crib_score"])
            df3 = pd.merge(df_hand, df_crib, on=["hand_key"])
            df3["avg_total_score"] = df3["avg_hand_score"] + (df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"])
        df3["min_total_score"] = df3["min_hand_score"] + (df3["min_crib_score"] if dealer_is_self else -df3["min_crib_score"])
        best_discards_str = df3.loc[df3["avg_total_score"] == df3["avg_total_score"].max()]["crib_key"].values[0]
        best_discards = best_discards_str.lower().replace("t", "10").split("|")
        best_discards_cards = build_hand(best_discards)
        
        Xs: List[np.ndarray] = []
        ys: List[float] = []
        discards_list: List[Tuple[Card, Card]] = []
        
        # Iterate through actual card combinations to get Card objects for featurization
        for kept in combinations(hand, 4):
            kept_list = list(kept)
            discards_list_temp = [c for c in hand if c not in kept_list]
            
            # Find matching row in df3 by converting cards to string keys
            hand_key = normalize_hand_to_str(kept_list)
            crib_key = normalize_hand_to_str(discards_list_temp)
            
            row = df3[(df3["hand_key"] == hand_key) & (df3["crib_key"] == crib_key)]
            if len(row) == 0:
                continue
            
            y = row["avg_total_score"].values[0]
            pegging_ev = None
            if self._pegging_ev_mode == "rollout":
                pegging_ev = estimate_pegging_ev_mc_for_discard(
                    hand,
                    kept_list,
                    discards_list_temp,
                    dealer_is_self,
                    self._rng_np,
                    self._pegging_ev_rollouts,
                )
            x = featurize_discard(kept_list, discards_list_temp, dealer_is_self, your_score, opponent_score, pegging_ev=pegging_ev)
            Xs.append(x)
            ys.append(float(y))
            discards_list.append((discards_list_temp[0], discards_list_temp[1]))
        
        # log ONE example for this 6-card hand:
        # X_hand is (15, D), label is best option index 0..14
        X_hand = np.stack(Xs, axis=0).astype(np.float32)  # (15, D)
        self._log.X_discard.append(X_hand) # type: ignore
        best_i = int(np.argmax(np.array(ys, dtype=np.float32)))
        self._log.y_discard.append(best_i) # type: ignore

        return tuple(best_discards_cards)
       
    def select_crib_cards_ranking(self, hand, dealer_is_self, your_score=None, opponent_score=None) -> Tuple[Card, Card]:
        full_deck = get_full_deck()
        hand_score_cache = {}
        crib_score_cache = {}
        hand_results = process_dealt_hand_only_exact([hand, full_deck, hand_score_cache])
        df_hand = pd.DataFrame(hand_results, columns=["hand_key","min_hand_score","max_hand_score","avg_hand_score"])
        if self._crib_ev_mode == "min":
            crib_results = calc_crib_min_only_given_6_cards(hand)
            df_crib = pd.DataFrame(crib_results, columns=["hand_key","crib_key","min_crib_score","avg_crib_score"])
            df3 = pd.merge(df_hand, df_crib, on=["hand_key"])
            df3["avg_total_score"] = df3["avg_hand_score"] + (df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"])
        else:
            hand_set = set(hand)
            remaining = [c for c in self._full_deck if c not in hand_set]
            rows = []
            for kept in combinations(hand, 4):
                kept_list = list(kept)
                discards_list_temp = [c for c in hand if c not in kept_list]
                hand_key = normalize_hand_to_str(kept_list)
                crib_key = normalize_hand_to_str(discards_list_temp)
                crib_ev = estimate_crib_ev_mc_from_remaining(
                    discards_list_temp,
                    remaining,
                    self._rng,
                    n_samples=self._crib_mc_samples,
                )
                rows.append([hand_key, crib_key, crib_ev])
            df_crib = pd.DataFrame(rows, columns=["hand_key","crib_key","avg_crib_score"])
            df3 = pd.merge(df_hand, df_crib, on=["hand_key"])
            df3["avg_total_score"] = df3["avg_hand_score"] + (df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"])

        Xs: List[np.ndarray] = []
        ys: List[float] = []
        discards_list: List[Tuple[Card, Card]] = []

        for kept in combinations(hand, 4):
            kept_list = list(kept)
            discards_list_temp = [c for c in hand if c not in kept_list]

            hand_key = normalize_hand_to_str(kept_list)
            crib_key = normalize_hand_to_str(discards_list_temp)

            row = df3[(df3["hand_key"] == hand_key) & (df3["crib_key"] == crib_key)]
            if len(row) == 0:
                continue

            y = row["avg_total_score"].values[0]
            pegging_ev = None
            if self._pegging_ev_mode == "rollout":
                pegging_ev = estimate_pegging_ev_mc_for_discard(
                    hand,
                    kept_list,
                    discards_list_temp,
                    dealer_is_self,
                    self._rng_np,
                    self._pegging_ev_rollouts,
                )
            x = featurize_discard(kept_list, discards_list_temp, dealer_is_self, your_score, opponent_score, pegging_ev=pegging_ev)
            Xs.append(x)
            ys.append(float(y))
            discards_list.append((discards_list_temp[0], discards_list_temp[1]))

        # Log the full 15-option set with their scores.
        X_hand = np.stack(Xs, axis=0).astype(np.float32)  # (15, D)
        y_hand = np.array(ys, dtype=np.float32)           # (15,)
        self._log.X_discard.append(X_hand)  # type: ignore
        self._log.y_discard.append(y_hand)  # type: ignore

        best_i = int(np.argmax(y_hand))
        return discards_list[best_i]


    def select_crib_cards_regresser(self, hand, dealer_is_self, your_score=None, opponent_score=None) -> Tuple[Card, Card]:                
        # don't analyze crib, just calculate min value of the crib and use that
        full_deck = get_full_deck()
        hand_score_cache = {}
        crib_score_cache = {}        
        hand_results = process_dealt_hand_only_exact([hand, full_deck, hand_score_cache])
        df_hand = pd.DataFrame(hand_results, columns=["hand_key","min_hand_score","max_hand_score","avg_hand_score"])
        if self._crib_ev_mode == "min":
            crib_results = calc_crib_min_only_given_6_cards(hand)
            df_crib = pd.DataFrame(crib_results, columns=["hand_key","crib_key","min_crib_score","avg_crib_score"])
            df3 = pd.merge(df_hand, df_crib, on=["hand_key"])
            df3["avg_total_score"] = df3["avg_hand_score"] + (df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"])
            df3["min_total_score"] = df3["min_hand_score"] + (df3["min_crib_score"] if dealer_is_self else -df3["min_crib_score"])
        else:
            hand_set = set(hand)
            remaining = [c for c in self._full_deck if c not in hand_set]
            rows = []
            for kept in combinations(hand, 4):
                kept_list = list(kept)
                discards_list_temp = [c for c in hand if c not in kept_list]
                hand_key = normalize_hand_to_str(kept_list)
                crib_key = normalize_hand_to_str(discards_list_temp)
                crib_ev = estimate_crib_ev_mc_from_remaining(
                    discards_list_temp,
                    remaining,
                    self._rng,
                    n_samples=self._crib_mc_samples,
                )
                rows.append([hand_key, crib_key, crib_ev])
            df_crib = pd.DataFrame(rows, columns=["hand_key","crib_key","avg_crib_score"])
            df3 = pd.merge(df_hand, df_crib, on=["hand_key"])
            df3["avg_total_score"] = df3["avg_hand_score"] + (df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"])

        # Iterate through actual card combinations to get Card objects for featurization
        for kept in combinations(hand, 4):
            kept_list = list(kept)
            discards_list = [c for c in hand if c not in kept_list]
            
            # Find matching row in df3 by converting cards to string keys
            hand_key = normalize_hand_to_str(kept_list)
            crib_key = normalize_hand_to_str(discards_list)
            
            row = df3[(df3["hand_key"] == hand_key) & (df3["crib_key"] == crib_key)]
            if len(row) == 0:
                continue
            
            y = row["avg_total_score"].values[0]
            pegging_ev = None
            if self._pegging_ev_mode == "rollout":
                pegging_ev = estimate_pegging_ev_mc_for_discard(
                    hand,
                    kept_list,
                    discards_list,
                    dealer_is_self,
                    self._rng_np,
                    self._pegging_ev_rollouts,
                )
            x = featurize_discard(kept_list, discards_list, dealer_is_self, your_score, opponent_score, pegging_ev=pegging_ev)
            self._log.X_discard.append(x)
            self._log.y_discard.append(float(y))
            if self._win_prob_mode == "rollout":
                y_win = estimate_win_prob_discard_rollout(
                    hand=hand,
                    kept=kept_list,
                    discards=discards_list,
                    dealer_is_self=dealer_is_self,
                    player_score=your_score,
                    opponent_score=opponent_score,
                    rng=self._rng,
                    n_rollouts=self._win_prob_rollouts,
                    min_score_for_eval=self._win_prob_min_score,
                )
                self._log.y_discard_win.append(float(y_win))

        best_discards_str = df3.loc[df3["avg_total_score"] == df3["avg_total_score"].max()]["crib_key"].values[0]
        best_discards = best_discards_str.lower().replace("t", "10").split("|")
        best_discards_cards = build_hand(best_discards)        
        return tuple(best_discards_cards)


def save_data(
    log,
    out_dir,
    cumulative_games,
    strategy,
    seed,
    pegging_feature_set: str,
    crib_ev_mode: str,
    crib_mc_samples: int,
    pegging_label_mode: str,
    pegging_rollouts: int,
    win_prob_mode: str,
    win_prob_rollouts: int,
    win_prob_min_score: int,
):
    """Save accumulated training data to disk."""
    os.makedirs(out_dir, exist_ok=True)
    
    # check that we did not use the wrong logging structure
    if log.X_discard:
        x0 = log.X_discard[0]
        if strategy == "classification":
            assert x0.shape == (15, DISCARD_FEATURE_DIM)
        elif strategy == "ranking":
            assert x0.shape == (15, DISCARD_FEATURE_DIM)
        else:
            assert x0.shape == (DISCARD_FEATURE_DIM,)
    
    y_discard_win = None
    if hasattr(log, "y_discard_win") and getattr(log, "y_discard_win"):
        try:
            y_discard_win = np.array(getattr(log, "y_discard_win"), dtype=np.float32)
        except Exception:
            y_discard_win = None

    if strategy == "classification":
        Xd = np.stack(log.X_discard).astype(np.float32) if log.X_discard else np.zeros((0, 15, DISCARD_FEATURE_DIM), np.float32)
        yd = np.array(log.y_discard, dtype=np.int64)

        assert Xd.ndim == 3
        assert Xd.shape[1] == 15
        assert Xd.shape[2] == DISCARD_FEATURE_DIM
        assert yd.ndim == 1
        assert yd.shape[0] == Xd.shape[0]
        assert yd.min() >= 0 and yd.max() < 15

    elif strategy == "regression":
        Xd = np.stack(log.X_discard).astype(np.float32) if log.X_discard else np.zeros((0, DISCARD_FEATURE_DIM), np.float32)
        yd = np.array(log.y_discard, dtype=np.float32)

        assert Xd.ndim == 2
        assert Xd.shape[1] == DISCARD_FEATURE_DIM
        assert yd.ndim == 1
        assert yd.shape[0] == Xd.shape[0]

    elif strategy == "ranking":
        Xd = np.stack(log.X_discard).astype(np.float32) if log.X_discard else np.zeros((0, 15, DISCARD_FEATURE_DIM), np.float32)
        yd = np.stack(log.y_discard).astype(np.float32) if log.y_discard else np.zeros((0, 15), np.float32)

        assert Xd.ndim == 3
        assert Xd.shape[1] == 15
        assert Xd.shape[2] == DISCARD_FEATURE_DIM
        assert yd.ndim == 2
        assert yd.shape[0] == Xd.shape[0]
        assert yd.shape[1] == 15

    pegging_dim = get_pegging_feature_dim(pegging_feature_set)
    Xp = np.stack(log.X_pegging).astype(np.float32) if log.X_pegging else np.zeros((0, pegging_dim), np.float32)
    yp = np.array(log.y_pegging, dtype=np.float32)

    out_path_discard = os.path.join(out_dir, f"discard_{cumulative_games}.npz")
    out_path_pegging = os.path.join(out_dir, f"pegging_{cumulative_games}.npz")
    logger.info(f"Saving to {out_path_discard} and {out_path_pegging}")
    if y_discard_win is not None and y_discard_win.shape[0] == yd.shape[0]:
        np.savez(out_path_discard, X=Xd, y=yd, y_win=y_discard_win)
    else:
        np.savez(out_path_discard, X=Xd, y=yd)
    np.savez(out_path_pegging, X=Xp, y=yp)
    logger.info(f"Saved discard: X={Xd.shape} y={yd.shape}")
    logger.info(f"Saved pegging: X={Xp.shape} y={yp.shape}")

    # Write/update dataset metadata for easy inspection.
    # This overwrites each time with the latest shard info.
    out_path = Path(out_dir)
    dataset_version = out_path.parent.name
    run_id = out_path.name
    dataset_meta = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "run_id": run_id,
        "strategy": strategy,
        "crib_ev_mode": crib_ev_mode,
        "crib_mc_samples": crib_mc_samples,
        "pegging_label_mode": pegging_label_mode,
        "pegging_rollouts": pegging_rollouts,
        "win_prob_mode": win_prob_mode,
        "win_prob_rollouts": win_prob_rollouts,
        "win_prob_min_score": win_prob_min_score,
        "pegging_ev_mode": pegging_ev_mode,
        "pegging_ev_rollouts": pegging_ev_rollouts,
        "has_discard_win_prob": y_discard_win is not None and (y_discard_win.shape[0] == yd.shape[0]),
        "cumulative_games": cumulative_games,
        "seed": seed,
        "discard": {
            "file": os.path.basename(out_path_discard),
            "X_shape": list(Xd.shape),
            "y_shape": list(yd.shape),
            "features": {
                "discard_features": "52 multi-hot discards",
                "kept_features": "52 multi-hot kept",
                "dealer_flag": "1 float (1.0 if dealer_is_self else 0.0)",
                "score_context": "player_score, opponent_score, score_margin, endgame_self, endgame_opp, endgame_any",
                "pegging_ev": "expected pegging points (self, opp, diff) from MC rollouts",
            "total_dim": DISCARD_FEATURE_DIM,
            },
            "label": {
                "classification": "best option index (0..14)",
                "regression": f"avg_total_score (crib_ev_mode={crib_ev_mode})",
                "ranking": f"avg_total_score (crib_ev_mode={crib_ev_mode})",
            }.get(strategy, "unknown"),
        },
        "pegging": {
            "file": os.path.basename(out_path_pegging),
            "X_shape": list(Xp.shape),
            "y_shape": list(yp.shape),
            "features": {
                "hand": "52 multi-hot",
                "table": "52 multi-hot",
                "count": "32 one-hot (0..31)",
                "candidate": "52 one-hot",
                "known_cards": "52 multi-hot",
            "total_dim": pegging_dim,
            "opponent_played": "52 multi-hot",
            "all_played": "52 multi-hot",
            "engineered": "25 scalar pegging features (15/31 flags, runs, setups, hand sizes, go-prob, scores, endgame)",
            "feature_set": pegging_feature_set,
            },
            "label": f"pegging_label_mode={pegging_label_mode}",
        },
    }
    meta_path = os.path.join(out_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2)
    logger.info(f"Saved dataset metadata -> {meta_path}")

    # Also write a human-readable summary.
    txt_path = os.path.join(out_dir, "dataset_meta.txt")
    lines = [
        f"updated_at_utc: {dataset_meta['updated_at_utc']}",
        f"dataset_version: {dataset_meta['dataset_version']}",
        f"run_id: {dataset_meta['run_id']}",
        f"strategy: {dataset_meta['strategy']}",
        f"crib_ev_mode: {dataset_meta['crib_ev_mode']}",
        f"crib_mc_samples: {dataset_meta['crib_mc_samples']}",
        f"pegging_label_mode: {dataset_meta['pegging_label_mode']}",
        f"pegging_rollouts: {dataset_meta['pegging_rollouts']}",
        f"win_prob_mode: {dataset_meta['win_prob_mode']}",
        f"win_prob_rollouts: {dataset_meta['win_prob_rollouts']}",
        f"win_prob_min_score: {dataset_meta['win_prob_min_score']}",
        f"pegging_ev_mode: {dataset_meta['pegging_ev_mode']}",
        f"pegging_ev_rollouts: {dataset_meta['pegging_ev_rollouts']}",
        f"has_discard_win_prob: {dataset_meta['has_discard_win_prob']}",
        f"cumulative_games: {dataset_meta['cumulative_games']}",
        f"seed: {dataset_meta['seed']}",
        "",
        f"discard_file: {dataset_meta['discard']['file']}",
        f"discard_X_shape: {dataset_meta['discard']['X_shape']}",
        f"discard_y_shape: {dataset_meta['discard']['y_shape']}",
        "discard_features:",
        f"  - {dataset_meta['discard']['features']['discard_features']}",
        f"  - {dataset_meta['discard']['features']['kept_features']}",
        f"  - {dataset_meta['discard']['features']['dealer_flag']}",
        f"  - {dataset_meta['discard']['features']['score_context']}",
        f"  - total_dim: {dataset_meta['discard']['features']['total_dim']}",
        f"discard_label: {dataset_meta['discard']['label']}",
        "",
        f"pegging_file: {dataset_meta['pegging']['file']}",
        f"pegging_X_shape: {dataset_meta['pegging']['X_shape']}",
        f"pegging_y_shape: {dataset_meta['pegging']['y_shape']}",
        "pegging_features:",
        f"  - {dataset_meta['pegging']['features']['hand']}",
        f"  - {dataset_meta['pegging']['features']['table']}",
        f"  - {dataset_meta['pegging']['features']['count']}",
        f"  - {dataset_meta['pegging']['features']['candidate']}",
        f"  - {dataset_meta['pegging']['features']['known_cards']}",
        f"  - {dataset_meta['pegging']['features']['opponent_played']}",
        f"  - {dataset_meta['pegging']['features']['all_played']}",
        f"  - {dataset_meta['pegging']['features']['engineered']}",
        f"  - feature_set: {dataset_meta['pegging']['features']['feature_set']}",
        f"  - total_dim: {dataset_meta['pegging']['features']['total_dim']}",
        f"pegging_label: {dataset_meta['pegging']['label']}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved dataset summary -> {txt_path}")

def _next_run_id(base_dir: str) -> str:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not existing:
        return "001"
    max_id = max(int(x) for x in existing)
    return f"{max_id + 1:03d}"

def _latest_run_id(base_dir: str) -> str | None:
    base = Path(base_dir)
    if not base.exists():
        return None
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not existing:
        return None
    max_id = max(int(x) for x in existing)
    return f"{max_id:03d}"

def _resolve_output_dir(
    base_out_dir: str,
    dataset_version: str,
    run_id: str | None,
    new_run: bool,
) -> str:
    version_dir = Path(base_out_dir) / dataset_version
    if run_id is None:
        if new_run:
            run_id = _next_run_id(str(version_dir))
        else:
            run_id = _latest_run_id(str(version_dir)) or "001"
    return str(version_dir / run_id)


def get_cumulative_game_count(out_dir):
    """Get the cumulative game count from existing files."""
    out_dir_path = Path(out_dir)
    existing_discard = sorted(out_dir_path.glob("discard_*.npz"))
    
    cumulative_games = 0
    if existing_discard:
        # Extract numbers from filenames and find max
        for f in existing_discard:
            try:
                num = int(f.stem.split('_')[1])
                cumulative_games = max(cumulative_games, num)
            except (ValueError, IndexError):
                pass
    return cumulative_games

def play_one_game(players) -> None:
    game = cribbagegame.CribbageGame(players=players, copy_players=False)
    # Some engines have game.play(), some run rounds internally.
    final_pegging_score = game.start()

def generate_il_data(
    games,
    out_dir,
    seed,
    strategy,
    pegging_feature_set: str = "full",
    crib_ev_mode: str = "mc",
    crib_mc_samples: int = 32,
    pegging_label_mode: str = "immediate",
    pegging_rollouts: int = 32,
    win_prob_mode: str = "off",
    win_prob_rollouts: int = 16,
    win_prob_min_score: int = 90,
    pegging_ev_mode: str = "off",
    pegging_ev_rollouts: int = 16,
) -> int:
    if seed is None:
        seed = secrets.randbits(32)
        logger.info(f"No seed provided, using random seed={seed}")
    if games < 0:
        logger.info(f"Generating IL data forever into {out_dir} using 2 medium players")
    else:
        logger.info(f"Generating IL data for {games} games into {out_dir} using 2 medium players")
    rng = np.random.default_rng(seed)
    
    # Get starting cumulative count
    cumulative_games = get_cumulative_game_count(out_dir)
    save_interval = 2000
    
    if strategy == "classification":
        log = LoggedRegPegClasDiscardData()
        p1 = LoggingMediumPlayer(
            "teacher1",
            log,
            discard_strategy="classification",
            pegging_strategy="regression",
            seed=seed or 0,
            pegging_feature_set=pegging_feature_set,
            crib_ev_mode=crib_ev_mode,
            crib_mc_samples=crib_mc_samples,
            pegging_label_mode=pegging_label_mode,
            pegging_rollouts=pegging_rollouts,
            win_prob_mode=win_prob_mode,
            win_prob_rollouts=win_prob_rollouts,
            win_prob_min_score=win_prob_min_score,
            pegging_ev_mode=pegging_ev_mode,
            pegging_ev_rollouts=pegging_ev_rollouts,
        )
        p2 = LoggingMediumPlayer(
            "teacher2",
            log,
            discard_strategy="classification",
            pegging_strategy="regression",
            seed=seed or 0,
            pegging_feature_set=pegging_feature_set,
            crib_ev_mode=crib_ev_mode,
            crib_mc_samples=crib_mc_samples,
            pegging_label_mode=pegging_label_mode,
            pegging_rollouts=pegging_rollouts,
            win_prob_mode=win_prob_mode,
            win_prob_rollouts=win_prob_rollouts,
            win_prob_min_score=win_prob_min_score,
            pegging_ev_mode=pegging_ev_mode,
            pegging_ev_rollouts=pegging_ev_rollouts,
        )
    elif strategy == "ranking":
        log = LoggedRegPegRankDiscardData()
        p1 = LoggingMediumPlayer(
            "teacher1",
            log,
            discard_strategy="ranking",
            pegging_strategy="regression",
            seed=seed or 0,
            pegging_feature_set=pegging_feature_set,
            crib_ev_mode=crib_ev_mode,
            crib_mc_samples=crib_mc_samples,
            pegging_label_mode=pegging_label_mode,
            pegging_rollouts=pegging_rollouts,
            win_prob_mode=win_prob_mode,
            win_prob_rollouts=win_prob_rollouts,
            win_prob_min_score=win_prob_min_score,
            pegging_ev_mode=pegging_ev_mode,
            pegging_ev_rollouts=pegging_ev_rollouts,
        )
        p2 = LoggingMediumPlayer(
            "teacher2",
            log,
            discard_strategy="ranking",
            pegging_strategy="regression",
            seed=seed or 0,
            pegging_feature_set=pegging_feature_set,
            crib_ev_mode=crib_ev_mode,
            crib_mc_samples=crib_mc_samples,
            pegging_label_mode=pegging_label_mode,
            pegging_rollouts=pegging_rollouts,
            win_prob_mode=win_prob_mode,
            win_prob_rollouts=win_prob_rollouts,
            win_prob_min_score=win_prob_min_score,
            pegging_ev_mode=pegging_ev_mode,
            pegging_ev_rollouts=pegging_ev_rollouts,
        )
    elif strategy == "regression":  
        log = LoggedRegPegRegDiscardData()
        p1 = LoggingMediumPlayer(
            "teacher1",
            log,
            discard_strategy="regression",
            pegging_strategy="regression",
            seed=seed or 0,
            pegging_feature_set=pegging_feature_set,
            crib_ev_mode=crib_ev_mode,
            crib_mc_samples=crib_mc_samples,
            pegging_label_mode=pegging_label_mode,
            pegging_rollouts=pegging_rollouts,
            win_prob_mode=win_prob_mode,
            win_prob_rollouts=win_prob_rollouts,
            win_prob_min_score=win_prob_min_score,
            pegging_ev_mode=pegging_ev_mode,
            pegging_ev_rollouts=pegging_ev_rollouts,
        )
        p2 = LoggingMediumPlayer(
            "teacher2",
            log,
            discard_strategy="regression",
            pegging_strategy="regression",
            seed=seed or 0,
            pegging_feature_set=pegging_feature_set,
            crib_ev_mode=crib_ev_mode,
            crib_mc_samples=crib_mc_samples,
            pegging_label_mode=pegging_label_mode,
            pegging_rollouts=pegging_rollouts,
            win_prob_mode=win_prob_mode,
            win_prob_rollouts=win_prob_rollouts,
            win_prob_min_score=win_prob_min_score,
        )
    
    games_since_save = 0

    i = 0
    while True:
        if i % 100 == 0:
            if games < 0:
                logger.info(f"Playing games {i} - {i + 100}/∞")
            else:
                logger.info(f"Playing games {i} - {i + 100}/{games}")
        
        # If your engine uses RNG/Deck seeding, set it here.
        # Some engines read global RNG; we at least randomize player order sometimes.
        if (i % 2) == 1:
            players = [p2, p1]
        else:
            players = [p1, p2]
        play_one_game(players)
        games_since_save += 1
        
        # Save every save_interval games if total games > save_interval
        if games_since_save >= save_interval:
            cumulative_games += games_since_save
            logger.info(f"Reached {save_interval} games, saving checkpoint at {cumulative_games} total games")
            save_data(
                log,
                out_dir,
                cumulative_games,
                strategy,
                seed,
                pegging_feature_set,
                crib_ev_mode,
                crib_mc_samples,
                pegging_label_mode,
                pegging_rollouts,
                win_prob_mode,
                win_prob_rollouts,
                win_prob_min_score,
            )
            
            # Clear the logs to save memory
            log.X_discard.clear()
            log.y_discard.clear()
            if hasattr(log, "y_discard_win"):
                log.y_discard_win.clear()
            log.X_pegging.clear()
            log.y_pegging.clear()
            games_since_save = 0
    
        i += 1
        if games >= 0 and i >= games:
            break

    # Save any remaining data
    if games_since_save > 0:
        cumulative_games += games_since_save
        logger.info(f"Saving final data at {cumulative_games} total games")
        save_data(
            log,
            out_dir,
            cumulative_games,
            strategy,
            seed,
            pegging_feature_set,
            crib_ev_mode,
            crib_mc_samples,
            pegging_label_mode,
            pegging_rollouts,
            win_prob_mode,
            win_prob_rollouts,
            win_prob_min_score,
        )
    
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--games", type=int, default=2000)
    ap.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES_PER_LOOP,
        help="Number of games to simulate. Use -1 to run forever.",
    )
    default_out_dir = TRAINING_DATA_DIR
    ap.add_argument("--out_dir", type=str, default=default_out_dir)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument(
        "--run_id",
        type=str,
        default=DEFAULT_DATASET_RUN_ID or None,
        help="Run id folder (e.g., 001). Omit to append to latest run unless --new_run is set.",
    )
    ap.add_argument(
        "--new_run",
        action="store_true",
        help="Create a new run folder even if one already exists for this dataset_version.",
    )
    default_seed = None if DEFAULT_USE_RANDOM_SEED else DEFAULT_SEED
    ap.add_argument("--seed", type=int, default=default_seed, help="Random seed. Omit to use a random seed.")
    ap.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    ap.add_argument(
        "--pegging_feature_set",
        type=str,
        default=DEFAULT_PEGGING_FEATURE_SET,
        choices=["basic", "full"],
        help="Which pegging feature set to use.",
    )
    ap.add_argument(
        "--crib_ev_mode",
        type=str,
        default=DEFAULT_CRIB_EV_MODE,
        choices=["min", "mc"],
        help="How to estimate crib EV for discard labels.",
    )
    ap.add_argument(
        "--crib_mc_samples",
        type=int,
        default=DEFAULT_CRIB_MC_SAMPLES,
        help="Number of Monte Carlo samples for crib EV when crib_ev_mode=mc.",
    )
    ap.add_argument(
        "--pegging_label_mode",
        type=str,
        default=DEFAULT_PEGGING_LABEL_MODE,
        choices=["immediate", "rollout1", "rollout2"],
        help="How to label pegging data.",
    )
    ap.add_argument(
        "--pegging_rollouts",
        type=int,
        default=DEFAULT_PEGGING_ROLLOUTS,
        help="Number of rollouts for pegging_label_mode=rollout1.",
    )
    ap.add_argument(
        "--win_prob_mode",
        type=str,
        default=DEFAULT_WIN_PROB_MODE,
        choices=["off", "rollout"],
        help="Win-probability label mode for discard (rollout or off).",
    )
    ap.add_argument(
        "--win_prob_rollouts",
        type=int,
        default=DEFAULT_WIN_PROB_ROLLOUTS,
        help="Number of rollouts for win-probability estimation.",
    )
    ap.add_argument(
        "--win_prob_min_score",
        type=int,
        default=DEFAULT_WIN_PROB_MIN_SCORE,
        help="Only estimate win-prob when max(score) >= this threshold (else label 0.5).",
    )
    ap.add_argument(
        "--pegging_ev_mode",
        type=str,
        default=DEFAULT_PEGGING_EV_MODE,
        choices=["off", "rollout"],
        help="Whether to add pegging EV features to discard (rollout or off).",
    )
    ap.add_argument(
        "--pegging_ev_rollouts",
        type=int,
        default=DEFAULT_PEGGING_EV_ROLLOUTS,
        help="Number of rollouts for pegging EV estimation.",
    )
    args = ap.parse_args()
    resolved_out_dir = _resolve_output_dir(args.out_dir, args.dataset_version, args.run_id, args.new_run)
    generate_il_data(
        args.games,
        resolved_out_dir,
        args.seed,
        args.strategy,
        args.pegging_feature_set,
        args.crib_ev_mode,
        args.crib_mc_samples,
        args.pegging_label_mode,
        args.pegging_rollouts,
        args.win_prob_mode,
        args.win_prob_rollouts,
        args.win_prob_min_score,
        args.pegging_ev_mode,
        args.pegging_ev_rollouts,
    )

# python .\scripts\generate_il_data.py
# python .\scripts\generate_il_data.py --games -1 --out_dir "il_datasets" --dataset_version "discard_v2" --strategy regression

# .\.venv\Scripts\python.exe .\scripts\generate_il_data.py --games 4000 --out_dir "il_datasets" --dataset_version "discard_v2" --strategy regression


# .\.venv\Scripts\python.exe .\scripts\train_linear_models.py --data_dir "il_datasets\discard_v2\001" --models_dir "models" --model_version "discard_v2" --run_id 003 --discard_loss regression --epochs 5 --eval_samples 2048 --lr 0.0001 --l2 0.001 --batch_size 1024

# .\.venv\Scripts\python.exe .\scripts\benchmark_2_players.py --players NeuralRegressionPlayer,beginner --games 200 --models_dir "models\discard_v2\003"
