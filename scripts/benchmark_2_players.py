"""Benchmark trained NeuralPlayer vs ReasonablePlayer.

Usage:
  python benchmark_neural_vs_reasonable.py --games 500 --models_dir models
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import os
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from cribbage.utils import play_multiple_games
from crib_ai_trainer.constants import (
    MODELS_DIR,
    TRAINING_DATA_DIR,
    DEFAULT_BENCHMARK_PLAYERS,
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_MAX_SHARDS,
    DEFAULT_SEED,
    DEFAULT_FALLBACK_PLAYER,
    DEFAULT_MODEL_TAG,
    DEFAULT_DISCARD_FEATURE_SET,
    DEFAULT_PEGGING_MODEL_FEATURE_SET,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODEL_RUN_ID,
)

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


def _find_latest_run_id(version_dir: Path) -> str | None:
    if not version_dir.exists():
        return None
    run_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return None
    run_id = max(int(p.name) for p in run_dirs)
    return f"{run_id:03d}"


def _resolve_models_dir(args) -> str:
    base = Path(args.models_dir)

    # If the user passed an explicit run folder with a model, use it directly.
    if (base / "model_meta.json").exists() or (base / "discard_linear.npz").exists():
        return str(base)

    # Prefer explicit model version subdir if it exists.
    version_dir = base / args.model_version if args.model_version else base
    if not version_dir.exists():
        version_dir = base

    if args.model_run_id:
        return str(version_dir / args.model_run_id)

    if args.latest_model:
        latest = _find_latest_run_id(version_dir)
        if latest:
            return str(version_dir / latest)

    # Fallback: return version_dir itself
    return str(version_dir)



def benchmark_2_players(
    args,
    players_override: str | None = None,
    fallback_override: str | None = None,
) -> int:
    args.models_dir = _resolve_models_dir(args)
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
        # Try to read model_meta.json to derive a stable version label.
        meta_path = os.path.join(args.models_dir, "model_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                version = meta.get("model_version")
                run_id = meta.get("run_id")
                if version and run_id:
                    return f"{version}-{run_id}"
                if version:
                    return str(version)
            except Exception:
                pass
        # Default to models_dir basename (e.g., "ranking", "regression", "classification")
        return os.path.basename(os.path.normpath(args.models_dir))

    # Load model metadata if available to align feature sets automatically.
    meta_path = os.path.join(args.models_dir, "model_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            args.discard_feature_set = meta.get("discard_feature_set", args.discard_feature_set)
            args.pegging_feature_set = meta.get("pegging_feature_set", args.pegging_feature_set)
        except Exception:
            pass

    model_tag = resolve_model_tag()

    fallback_player_name = fallback_override or args.fallback_player

    def player_factory(name: str):
        if name == "NeuralClassificationPlayer":
            discard_model = LinearDiscardClassifier.load_npz(f"{args.models_dir}/discard_linear.npz")
            return NeuralClassificationPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "NeuralRegressionPlayer":
            discard_model = LinearValueModel.load_npz(f"{args.models_dir}/discard_linear.npz")
            return NeuralRegressionPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "NeuralDiscardOnlyPlayer":
            discard_model = LinearDiscardClassifier.load_npz(f"{args.models_dir}/discard_linear.npz")
            fallback = base_player_factory(fallback_player_name)
            return NeuralDiscardOnlyPlayer(
                discard_model,
                fallback,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
            )
        if name == "NeuralPegOnlyPlayer":
            fallback = base_player_factory(fallback_player_name)
            return NeuralPegOnlyPlayer(
                pegging_model,
                fallback,
                name=f"{name}:{model_tag}",
                pegging_feature_set=args.pegging_feature_set,
            )
        return base_player_factory(name)
    
    players_value = players_override or args.players
    player_names = players_value.split(",")
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
    games_to_play = args.benchmark_games

    results = play_multiple_games(games_to_play, p0=p0, p1=p1)
    wins, diffs, winrate = results["wins"], results["diffs"], results["winrate"]
    
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
    if diffs and len(diffs) > 1:
        std_diff = float(np.std(diffs, ddof=1))
        se_diff = std_diff / float(np.sqrt(len(diffs)))
        diff_ci_lo = avg_diff - 1.96 * se_diff
        diff_ci_hi = avg_diff + 1.96 * se_diff
    else:
        diff_ci_lo = avg_diff
        diff_ci_hi = avg_diff
    display_names = []
    for name in player_names:
        if name.startswith("Neural") and name.endswith("Player") and "Regression" in name:
            tag = model_tag
            # Prefer version like "discard_v3-001" -> V3.001
            version_digits = []
            if "discard_v" in tag:
                try:
                    v_part = tag.split("discard_v", 1)[1]
                    v_num = "".join(c for c in v_part.split("-", 1)[0] if c.isdigit())
                    run_num = ""
                    if "-" in tag:
                        run_num = "".join(c for c in tag.split("-", 1)[1] if c.isdigit())
                    if v_num:
                        version_digits.append(f"V{v_num}")
                    if run_num:
                        version_digits.append(run_num)
                except Exception:
                    version_digits = []
            if version_digits:
                display_names.append(f"Neural{'.'.join(version_digits)}Player")
            else:
                version_label = "".join(c for c in tag if c.isdigit())
                if version_label:
                    display_names.append(f"NeuralV{version_label}Player")
                else:
                    display_names.append(f"Neural{tag.capitalize()}Player")
        else:
            display_names.append(name)
    model_dir_label = os.path.basename(os.path.normpath(args.models_dir))
    output_str = (
        f"{display_names[0]} vs {display_names[1]} [{model_dir_label}] after {estimated_training_games} training games "
        f"avg point diff {avg_diff:.2f} (95% CI {diff_ci_lo:.2f} - {diff_ci_hi:.2f}) "
        f"wins={wins}/{games_to_play} winrate={winrate*100:.1f}%\n"
    )
    with open("benchmark_results.txt", "a") as f:
        f.write(output_str)
    print(output_str)

    # Structured logging for experiments
    experiment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "players": player_names,
        "display_players": display_names,
        "models_dir": args.models_dir,
        "data_dir": str(data_dir),
        "benchmark_games": games_to_play,
        "wins": wins,
        "winrate": winrate,
        "avg_point_diff": avg_diff,
        "avg_point_diff_ci_lo": diff_ci_lo,
        "avg_point_diff_ci_hi": diff_ci_hi,
        "estimated_training_games": estimated_training_games,
        "discard_feature_set": args.discard_feature_set,
        "pegging_feature_set": args.pegging_feature_set,
        "model_tag": model_tag,
        "seed": args.seed,
    }
    experiments_path = "experiments.jsonl"
    with open(experiments_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(experiment) + "\n")
    logger.info(f"Appended experiment -> {experiments_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=str, default=DEFAULT_BENCHMARK_PLAYERS)
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR, help="Base models dir or explicit run dir")
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--model_run_id", type=str, default=DEFAULT_MODEL_RUN_ID or None, help="Explicit run id (e.g., 014)")
    ap.add_argument("--latest_model", action="store_true", default=True)
    ap.add_argument("--no_latest_model", dest="latest_model", action="store_false")
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--fallback_player", type=str, default=DEFAULT_FALLBACK_PLAYER)
    ap.add_argument("--model_tag", type=str, default=DEFAULT_MODEL_TAG or None)
    ap.add_argument("--discard_feature_set", type=str, default=DEFAULT_DISCARD_FEATURE_SET, choices=["base", "engineered_no_scores", "full"])
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--auto_mixed_benchmarks", action="store_true", default=True)
    ap.add_argument(
        "--no_auto_mixed_benchmarks",
        dest="auto_mixed_benchmarks",
        action="store_false",
    )
    args = ap.parse_args()
    logger.info(f"models dir: {args.models_dir}")
    benchmark_2_players(args)
    if args.auto_mixed_benchmarks:
        logger.info("Running discard-only and pegging-only benchmarks vs beginner fallback")
        benchmark_2_players(
            args,
            players_override="NeuralDiscardOnlyPlayer,beginner",
            fallback_override="beginner",
        )
        benchmark_2_players(
            args,
            players_override="NeuralPegOnlyPlayer,beginner",
            fallback_override="beginner",
        )

# python scripts/benchmark_2_players.py
# python scripts/benchmark_2_players.py --players NeuralRegressionPlayer,beginner --games 200 --models_dir "models/ranking" --data_dir "il_datasets/medium_discard_ranking" --max_shards 6 --fallback_player beginner

# .\.venv\Scripts\python.exe .\scripts\generate_il_data.py --games 4000 --out_dir "il_datasets" --dataset_version "discard_v2" --run_id 001 --strategy regression

# .\.venv\Scripts\python.exe .\scripts\train_linear_models.py --data_dir "il_datasets\discard_v3\001" --models_dir "models" --model_version "discard_v3" --run_id 002 --discard_loss regression --epochs 5 --eval_samples 2048 --lr 0.00005 --l2 0.001 --batch_size 2048

# .\.venv\Scripts\python.exe .\scripts\benchmark_2_players.py --players NeuralRegressionPlayer,beginner --benchmark_games 200 --models_dir "models\regression\" --data_dir "il_datasets\discard_v3\001"
# .\.venv\Scripts\python.exe .\scripts\benchmark_2_players.py
