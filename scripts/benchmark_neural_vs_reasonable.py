"""Benchmark trained NeuralPlayer vs ReasonablePlayer.

Usage:
  python benchmark_neural_vs_reasonable.py --games 500 --models_dir models
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

import numpy as np

from cribbage import cribbagegame

from crib_ai_trainer.players.rule_based_player import ReasonablePlayer
from crib_ai_trainer.players.neural_player import LinearValueModel, NeuralPlayer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_scores(game) -> tuple[int, int]:
    # Try a few common patterns
    if hasattr(game, "scores"):
        s = getattr(game, "scores")
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            return int(s[0]), int(s[1])
    if hasattr(game, "players"):
        players = getattr(game, "players")
        if len(players) >= 2:
            p0, p1 = players[0], players[1]
            if hasattr(p0, "score") and hasattr(p1, "score"):
                return int(p0.score), int(p1.score)
    raise RuntimeError("Can't read final scores from game. Add a get_scores() mapping for your engine.")


def play_game(p0, p1) -> tuple[int, int]:
    game = cribbagegame.CribbageGame(players=[p0, p1])
    final_pegging_scores = game.start()
    return (final_pegging_scores[0], final_pegging_scores[1])
    # return get_scores(game)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    return float(center - half), float(center + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logger.info("Loading models from %s", args.models_dir)
    discard_model = LinearValueModel.load_npz(f"{args.models_dir}/discard_linear.npz")
    pegging_model = LinearValueModel.load_npz(f"{args.models_dir}/pegging_linear.npz")

    print("discard |w|", float(np.linalg.norm(discard_model.w)), "b", float(discard_model.b))
    print("pegging  |w|", float(np.linalg.norm(pegging_model.w)), "b", float(pegging_model.b))
    breakpoint()
    rng = np.random.default_rng(args.seed)

    wins = 0
    diffs = []
    for i in range(args.games):
        if (i % 100) == 0:
            logger.info(f"Playing game {i+1}/{args.games}")
        # Alternate seats because cribbage has dealer advantage
        if i % 2 == 0:
            p0 = NeuralPlayer(discard_model, pegging_model, name="neural")
            # p1 = NeuralPlayer(discard_model, pegging_model, name="neural")
            p1 = ReasonablePlayer(name="reasonable")
            s0, s1 = play_game(p0, p1)
            diff = s0 - s1
            if diff > 0:
                wins += 1
        else:
            p0 = ReasonablePlayer(name="reasonable")
            # p0 = NeuralPlayer(discard_model, pegging_model, name="neural")
            p1 = NeuralPlayer(discard_model, pegging_model, name="neural")
            s0, s1 = play_game(p0, p1)
            diff = s1 - s0
            if diff > 0:
                wins += 1
        diffs.append(diff)

    winrate = wins / args.games
    lo, hi = wilson_ci(wins, args.games)
    avg_diff = float(np.mean(diffs)) if diffs else 0.0

    print(f"games={args.games}")
    print(f"wins={wins} winrate={winrate:.3f} (95% CI {lo:.3f}..{hi:.3f})")
    print(f"avg point diff (neural - reasonable): {avg_diff:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python scripts/benchmark_neural_vs_reasonable.py --games 500 --models_dir models