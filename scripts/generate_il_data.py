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

from cribbage.cribbagegame import score_hand, score_play as score_pegging_play

from crib_ai_trainer.constants import TRAINING_DATA_DIR

sys.path.insert(0, ".")

import argparse
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from cribbage import cribbagegame
from cribbage.playingcards import Card

from crib_ai_trainer.players.rule_based_player import ReasonablePlayer, get_full_deck
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


@dataclass
class LoggedData:
    X_discard: List[np.ndarray] = field(default_factory=list)
    y_discard: List[float] = field(default_factory=list)
    X_pegging: List[np.ndarray] = field(default_factory=list)
    y_pegging: List[float] = field(default_factory=list)


class LoggingReasonablePlayer(ReasonablePlayer):
    """Wrap ReasonablePlayer so we can collect training data while it plays."""

    def __init__(self, name: str, log: LoggedData, seed: int = 0):
        super().__init__(name=name)
        self._rng = random.Random(seed)
        self._full_deck = get_full_deck()
        self._log = log

    def select_crib_cards(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
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

            x = featurize_discard(kept, discards, dealer_is_self)  # or drop hand param
            self._log.X_discard.append(x)
            self._log.y_discard.append(float(y))

            if y > best_y:
                best_y = y
                best_discards = [tuple(discards)]
            elif y == best_y:
                best_discards.append(tuple(discards))

        return best_discards[0]

    def select_card_to_play(self, hand: List[Card], table, crib, count: int):
        # table is the list of cards currently on the table
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
            pts = score_pegging_play(sequence)
            y = pts
            x = featurize_pegging(hand, history_since_reset, count, c)
            self._log.X_pegging.append(x)
            self._log.y_pegging.append(float(y))
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
    


def play_one_game(players) -> None:
    game = cribbagegame.CribbageGame(players=players)
    # Some engines have game.play(), some run rounds internally.
    final_pegging_score = game.start()


def generate_il_data(games, out_dir, seed) -> int:
    logger.info(f"Generating IL data for {games} games into {out_dir} using 2 reasonable players")
    rng = np.random.default_rng(seed)
    log = LoggedData()

    # Two logging reasonable players self-play
    p1 = LoggingReasonablePlayer("teacher1", log)
    p2 = LoggingReasonablePlayer("teacher2", log)

    for i in range(games):
        if i % 100 == 0:
            logger.info(f"Playing game {i}/{games}")
        # If your engine uses RNG/Deck seeding, set it here.
        # Some engines read global RNG; we at least randomize player order sometimes.
        if (i % 2) == 1:
            players = [p2, p1]
        else:
            players = [p1, p2]
        play_one_game(players)
    
    logger.info(f"Saving data to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    Xd = np.stack(log.X_discard).astype(np.float32) if log.X_discard else np.zeros((0, 105), np.float32)
    yd = np.array(log.y_discard, dtype=np.float32)

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
    ap.add_argument("--games", type=int, default=2000)
    default_out_dir = TRAINING_DATA_DIR
    ap.add_argument("--out_dir", type=str, default=default_out_dir)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate_il_data(args.games, args.out_dir, args.seed)

#  python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20
# python scripts/benchmark_2_players.py --players neural,reasonable --games 500 --models_dir models