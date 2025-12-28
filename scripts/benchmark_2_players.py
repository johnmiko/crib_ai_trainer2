"""Benchmark trained NeuralPlayer vs ReasonablePlayer.

Usage:
  python benchmark_neural_vs_reasonable.py --games 500 --models_dir models
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import os

from crib_ai_trainer.constants import MODELS_DIR

sys.path.insert(0, ".")


from cribbage import cribbagegame
from crib_ai_trainer.players.random_player import RandomPlayer
from crib_ai_trainer.players.rule_based_player import ReasonablePlayer, basic_pegging_strategy
from crib_ai_trainer.players.neural_player import LinearValueModel, NeuralDiscardPlayer, NeuralPegPlayer, NeuralPlayer
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


def benchmark_2_players(args) -> int:
    logger.info("Loading models from %s", args.models_dir)
    discard_model = LinearValueModel.load_npz(f"{args.models_dir}/discard_linear.npz")
    pegging_model = LinearValueModel.load_npz(f"{args.models_dir}/pegging_linear.npz")
    # print("discard |w|", float(np.linalg.norm(discard_model.w)), "b", float(discard_model.b))
    # print("pegging  |w|", float(np.linalg.norm(pegging_model.w)), "b", float(pegging_model.b))
    # discard and pegging weights after 4000 games
    # discard |w| 3.507629156112671 b 1.5574564933776855
    # pegging  |w| 1.011649489402771 b 0.2532176971435547
    # breakpoint()
    rng = np.random.default_rng(args.seed)

    wins = 0
    diffs = []
    def player_factory(name: str):
        if name == "neural":
            return NeuralPlayer(discard_model, pegging_model, name="neural")
        elif name == "reasonable":
            return ReasonablePlayer(name="reasonable")
        elif name == "random":            
            return RandomPlayer(name="random", seed=args.seed)
        else:
            raise ValueError(f"Unknown player type: {name}")
    
    player_names = args.players.split(",")
    if len(player_names) != 2:
        raise ValueError("Must specify exactly two players via --players")
    # temp debugging
    # o pegging scored 184/500 and discard_model only scored 72/500
    # p0 = NeuralDiscardPlayer(discard_model, pegging_model, name="neural_discard")
    # p0 = NeuralPegPlayer(discard_model, pegging_model, name="neural_peg")
    # p1 = player_factory("reasonable")
    # player_names = [p0.name, p1.name]
    # temp end

    p0 = player_factory(player_names[0])
    p1 = player_factory(player_names[1])



    for i in range(args.games):
        if (i % 100) == 0:
            logger.info(f"Playing game {i}/{args.games}")
        # Alternate seats because cribbage has dealer advantage
        if i % 2 == 0:
            s0, s1 = play_game(p0, p1)
            diff = s0 - s1
            if diff > 0:
                wins += 1
        else:
            s0, s1 = play_game(p1, p0)
            diff = s1 - s0
            if diff > 0:
                wins += 1
        diffs.append(diff)

    winrate = wins / args.games
    lo, hi = wilson_ci(wins, args.games)    
    file_list = os.listdir("il_datasets")
    estimated_training_games = len(file_list) * 2000 / 2
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    output_str = f"{player_names[0]} vs {player_names[1]} after {estimated_training_games} training games wins={wins}/{args.games} winrate={winrate:.3f} (95% CI {lo:.3f} - {hi:.3f}) avg point diff {avg_diff:.2f}\n"
    with open("benchmark_results.txt", "a") as f:
        f.write(output_str)
    print(output_str)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=str, default="neural,reasonable")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    benchmark_2_players(args)

# python scripts/benchmark_2_players.py --players neural,reasonable --games 500 --models_dir models
# python scripts/benchmark_2_players.py --players neural,random --games 500 --models_dir models