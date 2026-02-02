"""Benchmark trained NeuralPlayer vs ReasonablePlayer.

Usage:
  python benchmark_neural_vs_reasonable.py --games 500 --models_dir models
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import os

sys.path.insert(0, ".")
from cribbage.utils import play_multiple_games
from crib_ai_trainer.constants import MODELS_DIR, TRAINING_DATA_DIR

from cribbage.players.random_player import RandomPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.beginner_player import BeginnerPlayer
from crib_ai_trainer.players.neural_player import LinearDiscardClassifier, LinearValueModel, NeuralClassificationPlayer, NeuralRegressionPlayer
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



def benchmark_2_players(args) -> int:
    logger.info("Loading models from %s", args.models_dir)    
    pegging_model = LinearValueModel.load_npz(f"{args.models_dir}/pegging_linear.npz")
    # print("discard |w|", float(np.linalg.norm(discard_model.w)), "b", float(discard_model.b))
    # print("pegging  |w|", float(np.linalg.norm(pegging_model.w)), "b", float(pegging_model.b))
    # discard and pegging weights after 4000 games
    # discard |w| 3.507629156112671 b 1.5574564933776855
    # pegging  |w| 1.011649489402771 b 0.2532176971435547
    # breakpoint()
    rng = np.random.default_rng(args.seed)

    def player_factory(name: str):
        if name == "NeuralClassificationPlayer":
            discard_model = LinearDiscardClassifier.load_npz(f"{args.models_dir}/discard_linear.npz")
            return NeuralClassificationPlayer(discard_model, pegging_model, name=name)
        elif name == "NeuralRegressionPlayer":
            discard_model = LinearValueModel.load_npz(f"{args.models_dir}/discard_linear.npz")
            return NeuralRegressionPlayer(discard_model, pegging_model, name=name)
        elif name == "beginner":
            return BeginnerPlayer(name=name)
        elif name == "random":            
            return RandomPlayer(name=name, seed=args.seed)
        elif name == "medium":
            return MediumPlayer(name=name)
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
    # {"wins":wins, "diffs": diffs, "winrate": winrate, "ci_lo": lo, "ci_hi": hi} 
    results = play_multiple_games(args.games, p0=p0, p1=p1)
    wins, diffs, winrate, lo, hi = results["wins"], results["diffs"], results["winrate"], results["ci_lo"], results["ci_hi"]
    file_list = os.listdir(TRAINING_DATA_DIR) # todo need to be able to pass in
    logger.info(f"file_list: {file_list}")
    estimated_training_games = len(file_list) * 2000 / 2
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    output_str = f"{player_names[0]} vs {player_names[1]} after {estimated_training_games} training games wins={wins}/{args.games} winrate={winrate*100:.3f} (95% CI {lo*100:.3f} - {hi*100:.3f}) avg point diff {avg_diff:.2f}\n"
    with open("benchmark_results.txt", "a") as f:
        f.write(output_str)
    print(output_str)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=str, default="NeuralClassificationPlayer,beginner")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto_random_benchmark", default=True)
    args = ap.parse_args()
    logger.info(f"models dir: {args.models_dir}")
    benchmark_2_players(args)

# python scripts/benchmark_2_players.py 
# python scripts/benchmark_2_players.py --players NeuralClassificationPlayer,medium --games 500
# python scripts/benchmark_2_players.py --players NeuralClassificationPlayer,beginner --games 500
# python scripts/benchmark_2_players.py --players NeuralRegressionPlayer,random --games 500
# python scripts/benchmark_2_players.py --players NeuralRegressionPlayer,medium --games 500