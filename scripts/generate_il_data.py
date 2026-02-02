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
from crib_ai_trainer.constants import TRAINING_DATA_DIR
import argparse
from itertools import combinations
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from cribbage import cribbagegame
from cribbage.playingcards import Card, build_hand
from cribbage.strategies.pegging_strategies import medium_pegging_strategy_scores, get_highest_rank_card
from cribbage.strategies.hand_strategies import process_dealt_hand_only_exact
from cribbage.strategies.crib_strategies import calc_crib_min_only_given_6_cards
from cribbage.database import normalize_hand_to_str

from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.medium_player import MediumPlayer
from crib_ai_trainer.players.neural_player import featurize_discard, featurize_pegging

from cribbage.cribbagegame import score_hand, score_play

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import random
from itertools import combinations

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

@dataclass
class LoggedClassificationDiscardData(LoggedData):
    X_discard: List[np.ndarray] = field(default_factory=list)   # each is (15, D)
    y_discard: List[int] = field(default_factory=list)         # each is 0..14

@dataclass
class LoggedRegPegRegDiscardData(LoggedRegressionPegData, LoggedRegressionDiscardData):
    pass

@dataclass
class LoggedRegPegClasDiscardData(LoggedRegressionPegData, LoggedClassificationDiscardData):
    pass



class LoggingBeginnerPlayer(BeginnerPlayer):
    """Wrap BeginnerPlayer so we can collect training data while it plays."""

    def __init__(self, name: str, log: LoggedData, discard_strategy, pegging_strategy, seed: int = 0, ):
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

    def select_crib_cards(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        # return self.select_crib_cards_classifier(hand, dealer_is_self)
        return self._discard_strategy(hand, dealer_is_self)
    
    def select_crib_cards_classifier(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
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

            x = featurize_discard(kept, discards, dealer_is_self)  # (105,)
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

    def select_crib_cards_regresser(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
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

            x = featurize_discard(kept, discards, dealer_is_self)  # or drop hand param
            self._log.X_discard.append(x)
            self._log.y_discard.append(float(y))

            if y > best_y:
                best_y = y
                best_discards = [tuple(discards)]
            elif y == best_y:
                best_discards.append(tuple(discards))

        return best_discards[0]

    def select_card_to_play(self, hand: List[Card], table, crib, count: int, game_state):
        # table is the list of cards currently on the table (current sequence since last reset)
        # Note: Ideally known_cards should include hand + table + past_table_cards + starter_card,
        # but we only have access to hand + table here. The game engine would need to be modified
        # to pass past_table_cards and starter_card for complete information.
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
            pts = score_play(sequence)
            y = pts
            # Known cards: hand + current table sequence
            # TODO: Enhance to include past_table_cards + starter_card when available
            x = featurize_pegging(hand, history_since_reset, count, c, known_cards=game_state.known_cards)
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

    def __init__(self, name: str, log: LoggedData, discard_strategy, pegging_strategy, seed: int = 0, ):
        super().__init__(name=name)
        self._rng = random.Random(seed)
        self._full_deck = get_full_deck()
        self._log = log
        if discard_strategy == "classification":
            self._discard_strategy_mode = "classification"
        elif discard_strategy == "regression":
            self._discard_strategy_mode = "regression"
        else:
            raise ValueError(f"Unknown discard_strategy: {discard_strategy}")        
        self._pegging_strategy = pegging_strategy

    def select_crib_cards(self, hand: List[Card], dealer_is_self: bool, your_score=None, opponent_score=None) -> Tuple[Card, Card]:
        """Override to log training data."""
        if self._discard_strategy_mode == "classification":
            return self.select_crib_cards_classifier(hand, dealer_is_self, game_state)
        else:
            return self.select_crib_cards_regresser(hand, dealer_is_self, game_state)

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card], game_state) -> Optional[Card]:
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
        
        # Log all playable options
        for card, score in scores.items():
            y = score
            # Known cards: playable hand + current table sequence
            # TODO: Enhance to include past_table_cards + starter_card when available
            known = playable + history_since_reset
            x = featurize_pegging(playable, history_since_reset, count, card, known_cards=known)
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
        crib_results = calc_crib_min_only_given_6_cards(hand)
        df_crib = pd.DataFrame(crib_results, columns=["hand_key","crib_key","min_crib_score","avg_crib_score"])
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
            x = featurize_discard(kept_list, discards_list_temp, dealer_is_self)  # (105,)
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
       

    def select_crib_cards_regresser(self, hand, dealer_is_self, your_score=None, opponent_score=None) -> Tuple[Card, Card]:                
        # don't analyze crib, just calculate min value of the crib and use that
        full_deck = get_full_deck()
        hand_score_cache = {}
        crib_score_cache = {}        
        hand_results = process_dealt_hand_only_exact([hand, full_deck, hand_score_cache])
        df_hand = pd.DataFrame(hand_results, columns=["hand_key","min_hand_score","max_hand_score","avg_hand_score"])
        crib_results = calc_crib_min_only_given_6_cards(hand)
        df_crib = pd.DataFrame(crib_results, columns=["hand_key","crib_key","min_crib_score","avg_crib_score"])
        df3 = pd.merge(df_hand, df_crib, on=["hand_key"])        
        df3["avg_total_score"] = df3["avg_hand_score"] + (df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"])
        df3["min_total_score"] = df3["min_hand_score"] + (df3["min_crib_score"] if dealer_is_self else -df3["min_crib_score"])

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
            x = featurize_discard(kept_list, discards_list, dealer_is_self)
            self._log.X_discard.append(x)
            self._log.y_discard.append(float(y))

        best_discards_str = df3.loc[df3["avg_total_score"] == df3["avg_total_score"].max()]["crib_key"].values[0]
        best_discards = best_discards_str.lower().replace("t", "10").split("|")
        best_discards_cards = build_hand(best_discards)        
        return tuple(best_discards_cards)


def play_one_game(players) -> None:
    game = cribbagegame.CribbageGame(players=players, copy_players=False)
    # Some engines have game.play(), some run rounds internally.
    final_pegging_score = game.start()


def generate_il_data(games, out_dir, seed, strategy) -> int:
    logger.info(f"Generating IL data for {games} games into {out_dir} using 2 medium players")
    rng = np.random.default_rng(seed)
    # log = LoggedData()
    # todo - this probably needs to be dynamic
    if strategy == "classification":
        log = LoggedRegPegClasDiscardData()
        p1 = LoggingMediumPlayer("teacher1", log, discard_strategy="classification", pegging_strategy="regression")
        p2 = LoggingMediumPlayer("teacher2", log, discard_strategy="classification", pegging_strategy="regression")
    elif strategy == "regression":  
        log = LoggedRegPegRegDiscardData()
        p1 = LoggingMediumPlayer("teacher1", log, discard_strategy="regression", pegging_strategy="regression")
        p2 = LoggingMediumPlayer("teacher2", log, discard_strategy="regression", pegging_strategy="regression")
    

    for i in range(games):
        if i % 100 == 0:
            logger.info(f"Playing games {i} - {i + 100}/{games}")
        # If your engine uses RNG/Deck seeding, set it here.
        # Some engines read global RNG; we at least randomize player order sometimes.
        if (i % 2) == 1:
            players = [p2, p1]
        else:
            players = [p1, p2]
        play_one_game(players)
    
    logger.info(f"Saving data to {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    # check that we did not use the wrong logging structure
    if log.X_discard:
        x0 = log.X_discard[0]
        if strategy == "classification":
            assert x0.shape == (15, 105)
        else:
            assert x0.shape == (105,)
    if strategy == "classification":
        Xd = np.stack(log.X_discard).astype(np.float32) if log.X_discard else np.zeros((0, 15, 105), np.float32)
        yd = np.array(log.y_discard, dtype=np.int64)

        assert Xd.ndim == 3
        assert Xd.shape[1] == 15
        assert Xd.shape[2] == 105
        assert yd.ndim == 1
        assert yd.shape[0] == Xd.shape[0]
        assert yd.min() >= 0 and yd.max() < 15

    elif strategy == "regression":
        Xd = np.stack(log.X_discard).astype(np.float32) if log.X_discard else np.zeros((0, 105), np.float32)
        yd = np.array(log.y_discard, dtype=np.float32)

        assert Xd.ndim == 2
        assert Xd.shape[1] == 105
        assert yd.ndim == 1
        assert yd.shape[0] == Xd.shape[0]

    Xp = np.stack(log.X_pegging).astype(np.float32) if log.X_pegging else np.zeros((0, 188), np.float32)
    yp = np.array(log.y_pegging, dtype=np.float32)

    out_dir_path = Path(out_dir)
    existing = sorted(out_dir_path.glob("discard_*.npz"))
    next_i = len(existing) + 1
    out_path_discard = os.path.join(out_dir, f"discard_{next_i:05d}.npz")
    existing = sorted(out_dir_path.glob("discard_*.npz"))
    next_i = len(existing) + 1
    out_path_pegging = os.path.join(out_dir, f"pegging_{next_i:05d}.npz")
    logger.info(f"Saving to {out_path_discard} and {out_path_pegging}")
    np.savez(out_path_discard, X=Xd, y=yd)
    np.savez(out_path_pegging, X=Xp, y=yp)
    logger.info(f"Saved discard: X={Xd.shape} y={yd.shape}")
    logger.info(f"Saved pegging: X={Xp.shape} y={yp.shape}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--games", type=int, default=200)
    default_out_dir = TRAINING_DATA_DIR
    ap.add_argument("--out_dir", type=str, default=default_out_dir)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", type=str, default="classification")
    args = ap.parse_args()
    generate_il_data(args.games, args.out_dir, args.seed, args.strategy)

# python .\scripts\generate_il_data.py --games 20
#  python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py
# python scripts/benchmark_2_players.py --players neural,beginner
# python scripts/benchmark_2_players.py --players neural,beginner
