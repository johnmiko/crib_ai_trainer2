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
from pathlib import Path
import sys

sys.path.insert(0, ".")

import argparse
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from cribbage import cribbagegame
from cribbage.playingcards import Card

from crib_ai_trainer.players.rule_based_player import ReasonablePlayer
from crib_ai_trainer.players.neural_player import featurize_discard, featurize_pegging

from cribbage.cribbagegame import score_hand, score_play

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoggedData:
    X_discard: List[np.ndarray] = field(default_factory=list)
    y_discard: List[float] = field(default_factory=list)
    X_pegging: List[np.ndarray] = field(default_factory=list)
    y_pegging: List[float] = field(default_factory=list)


class LoggingReasonablePlayer(ReasonablePlayer):
    """Wrap ReasonablePlayer so we can collect training data while it plays."""

    def __init__(self, name: str, log: LoggedData):
        super().__init__(name=name)
        self._log = log

    def select_crib_cards(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        # Log *all* discard options with heuristic label.
        for kept in combinations(hand, 4):
            kept = list(kept)
            discards = [c for c in hand if c not in kept]

            # same neutral starter hack used in your ReasonablePlayer
            kept_score = score_hand(kept, is_crib=False)
            crib_score = score_hand(discards, is_crib=True)
            y = kept_score + crib_score if dealer_is_self else kept_score - crib_score

            x = featurize_discard(hand, kept, discards, dealer_is_self)
            self._log.X_discard.append(x)
            self._log.y_discard.append(float(y))

        return super().select_crib_cards(hand, dealer_is_self)

    def select_card_to_play(self, hand: List[Card], table, crib, count: int):
        # table is the list of cards currently on the table
        playable = [c for c in hand if c + count <= 31]
        if not playable:
            return None

        # Log *all* playable options with immediate pegging reward.
        # score_play expects sequence since reset.
        history_since_reset = list(table)
        for cand in playable:
            seq = history_since_reset + [cand]
            y = score_play(seq)
            x = featurize_pegging(hand, history_since_reset, count, cand)
            self._log.X_pegging.append(x)
            self._log.y_pegging.append(float(y))

        return super().select_card_to_play(hand, table, crib, count)


def play_one_game(players) -> None:
    game = cribbagegame.CribbageGame(players=players)
    # Some engines have game.play(), some run rounds internally.
    final_pegging_score = game.start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out_dir", type=str, default="/scripts/il_datasets/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logger.info(f"Generating IL data for {args.games} games into {args.out_dir}")
    rng = np.random.default_rng(args.seed)
    log = LoggedData()

    # Two logging reasonable players self-play
    p1 = LoggingReasonablePlayer("teacher1", log)
    p2 = LoggingReasonablePlayer("teacher2", log)

    for i in range(args.games):
        if i % 100 == 0:
            logger.info(f"Playing game {i}/{args.games}")
        # If your engine uses RNG/Deck seeding, set it here.
        # Some engines read global RNG; we at least randomize player order sometimes.
        if (i % 2) == 1:
            players = [p2, p1]
        else:
            players = [p1, p2]
        play_one_game(players)

    # Stack and save
    import os
    logger.info(f"Saving data to {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    Xd = np.stack(log.X_discard).astype(np.float32) if log.X_discard else np.zeros((0, 105), np.float32)
    yd = np.array(log.y_discard, dtype=np.float32)

    Xp = np.stack(log.X_pegging).astype(np.float32) if log.X_pegging else np.zeros((0, 188), np.float32)
    yp = np.array(log.y_pegging, dtype=np.float32)

    out_dir_path = Path(args.out_dir)
    existing = sorted(out_dir_path.glob("discard_*.npz"))
    next_i = len(existing) + 1
    out_path_discard = os.path.join(args.out_dir, f"discard_{next_i:05d}.npz")
    existing = sorted(out_dir_path.glob("discard_*.npz"))
    next_i = len(existing) + 1
    out_path_pegging = os.path.join(args.out_dir, f"pegging_{next_i:05d}.npz")
    np.savez(out_path_discard, X=Xd, y=yd)
    np.savez(out_path_pegging, X=Xp, y=yp)
    logger.info(f"Saved discard: X={Xd.shape} y={yd.shape}")
    logger.info(f"Saved pegging: X={Xp.shape} y={yp.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#  python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20