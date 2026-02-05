"""Benchmark trained NeuralPlayer vs ReasonablePlayer.

Usage:
  python benchmark_neural_vs_reasonable.py --games 500 --models_dir models
"""

from __future__ import annotations

import sys
import numpy as np
import os
import json
import multiprocessing as mp
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from cribbage.utils import play_multiple_games
from cribbage.players.random_player import RandomPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.hard_player import HardPlayer
from crib_ai_trainer.players.neural_player import (
    LinearDiscardClassifier,
    LinearValueModel,
    NeuralClassificationPlayer,
    NeuralRegressionPlayer,
    NeuralDiscardOnlyPlayer,
    NeuralPegOnlyPlayer,
    MLPValueModel,
)
from utils import build_benchmark_parser
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



def _build_player_factory(args, fallback_override: str | None):
    args.models_dir = _resolve_models_dir(args)
    logger.info("Loading models from %s", args.models_dir)
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
        if name == "hard":
            return HardPlayer(name=name)
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
    model_type = "linear"
    mlp_hidden = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            args.discard_feature_set = meta.get("discard_feature_set", args.discard_feature_set)
            args.pegging_feature_set = meta.get("pegging_feature_set", args.pegging_feature_set)
            model_type = meta.get("model_type", model_type)
            mlp_hidden = meta.get("mlp_hidden", None)
        except Exception:
            pass
    else:
        # Fallback: infer model type from files if metadata is missing.
        if os.path.exists(os.path.join(args.models_dir, "discard_mlp.pt")):
            model_type = "mlp"

    fallback_player_name = fallback_override or args.fallback_player

    def _load_discard_model():
        if model_type == "mlp":
            return MLPValueModel.load_pt(f"{args.models_dir}/discard_mlp.pt")
        return LinearValueModel.load_npz(f"{args.models_dir}/discard_linear.npz")

    def _load_pegging_model():
        if model_type == "mlp":
            return MLPValueModel.load_pt(f"{args.models_dir}/pegging_mlp.pt")
        return LinearValueModel.load_npz(f"{args.models_dir}/pegging_linear.npz")

    model_tag = resolve_model_tag()
    pegging_model = _load_pegging_model()

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
            discard_model = _load_discard_model()
            return NeuralRegressionPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "NeuralDiscardOnlyPlayer":
            discard_model = _load_discard_model()
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
    
    return player_factory, model_tag, model_type


def _estimate_training_games(data_dir: Path, max_shards: int | None) -> int:
    discard_files = sorted(data_dir.glob("discard_*.npz"))
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("--max_shards must be > 0 if provided")
        discard_files = discard_files[: max_shards]
    if discard_files:
        max_games = 0
        for f in discard_files:
            try:
                num = int(f.stem.split("_")[1])
                max_games = max(max_games, num)
            except (ValueError, IndexError):
                pass
        return max_games
    return 0


def _compute_win_ci(winrate: float, total_games: int) -> tuple[float, float]:
    if total_games <= 1:
        return winrate, winrate
    se = float(np.sqrt(winrate * (1.0 - winrate) / total_games))
    lo = max(0.0, winrate - 1.96 * se)
    hi = min(1.0, winrate + 1.96 * se)
    return lo, hi


def _benchmark_single(
    args,
    players_override: str | None = None,
    fallback_override: str | None = None,
    games_override: int | None = None,
) -> dict:
    player_factory, model_tag, model_type = _build_player_factory(args, fallback_override)

    players_value = players_override or args.players
    player_names = players_value.split(",")
    if len(player_names) != 2:
        raise ValueError("Must specify exactly two players via --players")

    p0 = player_factory(player_names[0])
    p1 = player_factory(player_names[1])
    games_to_play = games_override or args.benchmark_games

    results = play_multiple_games(games_to_play, p0=p0, p1=p1)
    wins = results["wins"]
    diffs = results["diffs"]
    winrate = results["winrate"]
    win_ci_lo = results.get("ci_lo", winrate)
    win_ci_hi = results.get("ci_hi", winrate)
    diff_ci_lo = results.get("diff_ci_lo")
    diff_ci_hi = results.get("diff_ci_hi")

    # Count actual training games from discard file names (which are cumulative)
    data_dir = Path(args.data_dir)
    estimated_training_games = _estimate_training_games(data_dir, args.max_shards)

    logger.info(f"Estimated training games from files: {estimated_training_games}")
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    if diff_ci_lo is None or diff_ci_hi is None:
        if diffs and len(diffs) > 1:
            std_diff = float(np.std(diffs, ddof=1))
            se_diff = std_diff / float(np.sqrt(len(diffs)))
            diff_ci_lo = avg_diff - 1.96 * se_diff
            diff_ci_hi = avg_diff + 1.96 * se_diff
        else:
            diff_ci_lo = avg_diff
            diff_ci_hi = avg_diff
    display_names = []
    model_prefix = "MLP" if model_type == "mlp" else "Linear"
    for name in player_names:
        if name.startswith("Neural") and name.endswith("Player") and "Regression" in name:
            tag = model_tag
            # Prefer version like "discard_v6-008" -> V6.008 (for linear) or V6 (for mlp)
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
                    if run_num and model_type != "mlp":
                        version_digits.append(run_num)
                except Exception:
                    version_digits = []
            if version_digits:
                if model_type == "mlp":
                    display_names.append(f"{model_prefix}{''.join(version_digits)}")
                else:
                    display_names.append(f"{model_prefix}V{'.'.join(version_digits[0:])}Player")
            else:
                version_label = "".join(c for c in tag if c.isdigit())
                if version_label:
                    if model_type == "mlp":
                        display_names.append(f"{model_prefix}V{version_label}")
                    else:
                        display_names.append(f"{model_prefix}V{version_label}Player")
                else:
                    if model_type == "mlp":
                        display_names.append(f"{model_prefix}{tag.capitalize()}")
                    else:
                        display_names.append(f"{model_prefix}{tag.capitalize()}Player")
        else:
            display_names.append(name)
    return {
        "wins": wins,
        "diffs": diffs,
        "winrate": winrate,
        "win_ci_lo": win_ci_lo,
        "win_ci_hi": win_ci_hi,
        "diff_ci_lo": diff_ci_lo,
        "diff_ci_hi": diff_ci_hi,
        "avg_diff": avg_diff,
        "player_names": player_names,
        "display_names": display_names,
        "model_type": model_type,
        "model_tag": model_tag,
        "models_dir": args.models_dir,
        "data_dir": str(data_dir),
        "estimated_training_games": estimated_training_games,
        "discard_feature_set": args.discard_feature_set,
        "pegging_feature_set": args.pegging_feature_set,
        "seed": args.seed,
        "games_to_play": games_to_play,
    }


def _benchmark_worker(args_tuple) -> dict:
    args_dict, players_override, fallback_override, games_override, worker_id = args_tuple
    args = SimpleNamespace(**args_dict)
    base_seed = args.seed or 0
    args.seed = int(base_seed) + worker_id
    result = _benchmark_single(args, players_override, fallback_override, games_override)
    result["worker_id"] = worker_id
    return result


def benchmark_2_players(
    args,
    players_override: str | None = None,
    fallback_override: str | None = None,
) -> int:
    if args.seed is None:
        args.seed = 42
    if args.benchmark_workers < 1:
        args.benchmark_workers = 1

    if args.benchmark_workers > 1:
        total_games = args.benchmark_games
        if total_games <= 0:
            raise ValueError("--benchmark_games must be > 0.")

        if args.benchmark_games_per_worker is not None:
            per_worker_games = args.benchmark_games_per_worker
            if per_worker_games <= 0:
                raise ValueError("benchmark_games_per_worker must be > 0 when using multiple workers.")
            total_games = per_worker_games * args.benchmark_workers
            games_per_worker = [per_worker_games] * args.benchmark_workers
        else:
            base = total_games // args.benchmark_workers
            remainder = total_games % args.benchmark_workers
            if base == 0:
                raise ValueError(
                    "benchmark_games is smaller than benchmark_workers. "
                    "Reduce workers or increase games."
                )
            games_per_worker = [
                base + (1 if i < remainder else 0) for i in range(args.benchmark_workers)
            ]

        logger.info(
            "Benchmarking with %d workers for %d total games (%s per worker)",
            args.benchmark_workers,
            total_games,
            ",".join(str(g) for g in games_per_worker),
        )

        args_dict = vars(args).copy()
        ctx = mp.get_context("spawn")
        results = []
        with ctx.Pool(processes=args.benchmark_workers) as pool:
            for idx, result in enumerate(
                pool.imap_unordered(
                    _benchmark_worker,
                    [
                        (args_dict, players_override, fallback_override, games_per_worker[worker_id], worker_id)
                        for worker_id in range(args.benchmark_workers)
                    ],
                ),
                start=1,
            ):
                results.append(result)
                logger.info(
                    "Benchmark worker %s finished (%d/%d)",
                    str(result.get("worker_id")),
                    idx,
                    args.benchmark_workers,
                )

        wins = int(sum(r["wins"] for r in results))
        diffs = []
        for r in results:
            diffs.extend(r["diffs"])
        winrate = float(wins) / float(total_games) if total_games else 0.0
        win_ci_lo, win_ci_hi = _compute_win_ci(winrate, total_games)
        avg_diff = float(np.mean(diffs)) if diffs else 0.0
        if diffs and len(diffs) > 1:
            std_diff = float(np.std(diffs, ddof=1))
            se_diff = std_diff / float(np.sqrt(len(diffs)))
            diff_ci_lo = avg_diff - 1.96 * se_diff
            diff_ci_hi = avg_diff + 1.96 * se_diff
        else:
            diff_ci_lo = avg_diff
            diff_ci_hi = avg_diff

        first = results[0]
        player_names = first["player_names"]
        display_names = first["display_names"]
        model_type = first["model_type"]
        model_tag = first["model_tag"]
        model_dir_label = os.path.basename(os.path.normpath(first["models_dir"]))
        estimated_training_games = first["estimated_training_games"]
        data_dir = first["data_dir"]
        discard_feature_set = first["discard_feature_set"]
        pegging_feature_set = first["pegging_feature_set"]

        output_str = (
            f"{display_names[0]} vs {display_names[1]} [{model_dir_label}] after {estimated_training_games} training games "
            f"avg point diff {avg_diff:.2f} (95% CI {diff_ci_lo:.2f} - {diff_ci_hi:.2f}) "
            f"wins={wins}/{total_games} winrate={winrate*100:.2f}% "
            f"(95% CI {win_ci_lo*100:.2f}% - {win_ci_hi*100:.2f}%)\n"
        )
        if getattr(args, "no_benchmark_write", False):
            logger.info("Skipping benchmark_results.txt write (no_benchmark_write=True).")
        else:
            with open("benchmark_results.txt", "a") as f:
                f.write(output_str)
        print(output_str)

        experiment = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "players": player_names,
            "display_players": display_names,
            "models_dir": first["models_dir"],
            "data_dir": data_dir,
            "benchmark_games": total_games,
            "wins": wins,
            "winrate": winrate,
            "avg_point_diff": avg_diff,
            "avg_point_diff_ci_lo": diff_ci_lo,
            "avg_point_diff_ci_hi": diff_ci_hi,
            "winrate_ci_lo": win_ci_lo,
            "winrate_ci_hi": win_ci_hi,
            "estimated_training_games": estimated_training_games,
            "discard_feature_set": discard_feature_set,
            "pegging_feature_set": pegging_feature_set,
            "model_tag": model_tag,
            "seed": args.seed,
        }
        experiments_path = "experiments.jsonl"
        if getattr(args, "no_benchmark_write", False):
            logger.info("Skipping experiments.jsonl write (no_benchmark_write=True).")
        else:
            with open(experiments_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(experiment) + "\n")
            logger.info(f"Appended experiment -> {experiments_path}")
        return 0

    single = _benchmark_single(args, players_override, fallback_override)
    model_dir_label = os.path.basename(os.path.normpath(single["models_dir"]))
    output_str = (
        f"{single['display_names'][0]} vs {single['display_names'][1]} [{model_dir_label}] "
        f"after {single['estimated_training_games']} training games "
        f"avg point diff {single['avg_diff']:.2f} (95% CI {single['diff_ci_lo']:.2f} - {single['diff_ci_hi']:.2f}) "
        f"wins={single['wins']}/{single['games_to_play']} winrate={single['winrate']*100:.2f}% "
        f"(95% CI {single['win_ci_lo']*100:.2f}% - {single['win_ci_hi']*100:.2f}%)\n"
    )
    if getattr(args, "no_benchmark_write", False):
        logger.info("Skipping benchmark_results.txt write (no_benchmark_write=True).")
    else:
        with open("benchmark_results.txt", "a") as f:
            f.write(output_str)
    print(output_str)

    experiment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "players": single["player_names"],
        "display_players": single["display_names"],
        "models_dir": single["models_dir"],
        "data_dir": single["data_dir"],
        "benchmark_games": single["games_to_play"],
        "wins": single["wins"],
        "winrate": single["winrate"],
        "avg_point_diff": single["avg_diff"],
        "avg_point_diff_ci_lo": single["diff_ci_lo"],
        "avg_point_diff_ci_hi": single["diff_ci_hi"],
        "winrate_ci_lo": single["win_ci_lo"],
        "winrate_ci_hi": single["win_ci_hi"],
        "estimated_training_games": single["estimated_training_games"],
        "discard_feature_set": single["discard_feature_set"],
        "pegging_feature_set": single["pegging_feature_set"],
        "model_tag": single["model_tag"],
        "seed": single["seed"],
    }
    experiments_path = "experiments.jsonl"
    if getattr(args, "no_benchmark_write", False):
        logger.info("Skipping experiments.jsonl write (no_benchmark_write=True).")
    else:
        with open(experiments_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment) + "\n")
        logger.info(f"Appended experiment -> {experiments_path}")
    return 0


if __name__ == "__main__":
    args = build_benchmark_parser().parse_args()
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
