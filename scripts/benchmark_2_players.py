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
from crib_ai_trainer.players.neural_player import (
    LinearDiscardClassifier,
    LinearValueModel,
    NeuralClassificationPlayer,
    NeuralRegressionPlayer,
    NeuralDiscardOnlyPlayer,
    NeuralPegOnlyPlayer,
)
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

    def base_player_factory(name: str):
        if name == "beginner":
            return BeginnerPlayer(name=name)
        if name == "random":
            return RandomPlayer(name=name, seed=args.seed)
        if name == "medium":
            return MediumPlayer(name=name)
        raise ValueError(f"Unknown fallback player type: {name}")

    def resolve_model_tag() -> str:
        if args.model_tag:
            return args.model_tag
        # Default to models_dir basename (e.g., "ranking", "regression", "classification")
        return os.path.basename(os.path.normpath(args.models_dir))

    model_tag = resolve_model_tag()

    def player_factory(name: str):
        if name == "NeuralClassificationPlayer":
            discard_model = LinearDiscardClassifier.load_npz(f"{args.models_dir}/discard_linear.npz")
            return NeuralClassificationPlayer(discard_model, pegging_model, name=f"{name}:{model_tag}")
        if name == "NeuralRegressionPlayer":
            discard_model = LinearValueModel.load_npz(f"{args.models_dir}/discard_linear.npz")
            return NeuralRegressionPlayer(discard_model, pegging_model, name=f"{name}:{model_tag}")
        if name == "NeuralDiscardOnlyPlayer":
            discard_model = LinearDiscardClassifier.load_npz(f"{args.models_dir}/discard_linear.npz")
            fallback = base_player_factory(args.fallback_player)
            return NeuralDiscardOnlyPlayer(discard_model, fallback, name=f"{name}:{model_tag}")
        if name == "NeuralPegOnlyPlayer":
            fallback = base_player_factory(args.fallback_player)
            return NeuralPegOnlyPlayer(pegging_model, fallback, name=f"{name}:{model_tag}")
        return base_player_factory(name)
    
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
    
    # Count actual training games from discard file names (which are cumulative)
    from pathlib import Path
    data_dir = Path(args.data_dir)
    discard_files = sorted(data_dir.glob("discard_*.npz"))
    if args.max_shards is not None:
        if args.max_shards <= 0:
            raise ValueError("--max_shards must be > 0 if provided")
        discard_files = discard_files[: args.max_shards]
    
    if discard_files:
        # Get the highest cumulative game count from filenames
        max_games = 0
        for f in discard_files:
            try:
                # Extract number from filename like "discard_2000.npz"
                num = int(f.stem.split('_')[1])
                max_games = max(max_games, num)
            except (ValueError, IndexError):
                pass
        estimated_training_games = max_games
    else:
        estimated_training_games = 0
    
    logger.info(f"Estimated training games from files: {estimated_training_games}")
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    output_str = f"{player_names[0]} vs {player_names[1]} after {estimated_training_games} training games wins={wins}/{args.games} winrate={winrate*100:.1f}% (95% CI {lo*100:.1f}% - {hi*100:.1f})% avg point diff {avg_diff:.2f}\n"
    with open("benchmark_results.txt", "a") as f:
        f.write(output_str)
    print(output_str)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=str, default="NeuralClassificationPlayer,beginner")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fallback_player", type=str, default="beginner")
    ap.add_argument("--model_tag", type=str, default=None)
    ap.add_argument("--auto_random_benchmark", default=True)
    args = ap.parse_args()
    logger.info(f"models dir: {args.models_dir}")
    benchmark_2_players(args)

# python scripts/benchmark_2_players.py 
# python scripts/benchmark_2_players.py --players NeuralClassificationPlayer,medium --games 500
# python scripts/benchmark_2_players.py --players NeuralClassificationPlayer,beginner --games 500
# python scripts/benchmark_2_players.py --players NeuralRegressionPlayer,random --games 500
# python scripts/benchmark_2_players.py --players NeuralRegressionPlayer,medium --games 500
