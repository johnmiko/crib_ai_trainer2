"""Train multiple MLP sizes and benchmark against medium.

This script trains several MLP models and then benchmarks each against
the medium player, plus the latest existing "medium" model as a baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, ".")

from crib_ai_trainer.constants import (
    TRAINING_DATA_DIR,
    MODELS_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_DATASET_RUN_ID,
    DEFAULT_MODEL_VERSION,
    DEFAULT_DISCARD_LOSS,
    DEFAULT_DISCARD_FEATURE_SET,
    DEFAULT_PEGGING_MODEL_FEATURE_SET,
    DEFAULT_LR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_L2,
    DEFAULT_SEED,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_MAX_SHARDS,
    DEFAULT_RANK_PAIRS_PER_HAND,
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_BENCHMARK_WORKERS,
    DEFAULT_BENCHMARK_GAMES_PER_WORKER,
    DEFAULT_PEGGING_DATA_DIR,
)
from scripts.generate_il_data import _resolve_output_dir
from scripts.train_linear_models import train_linear_models, _resolve_models_dir
from scripts.benchmark_2_players import benchmark_2_players


def _find_latest_run_id(version_dir: Path) -> str | None:
    if not version_dir.exists():
        return None
    run_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return None
    run_id = max(int(p.name) for p in run_dirs)
    return f"{run_id:03d}"


def _resolve_variant_dir(base_models_dir: str, model_version: str, label: str) -> str:
    base = Path(base_models_dir)
    version_dir = base / model_version if model_version else base
    latest_run = _find_latest_run_id(version_dir) or "001"
    return str(version_dir / f"{latest_run}_{label}")


def _resolve_dataset_dir(base_dir: str, version: str, run_id: str | None) -> str:
    base = Path(base_dir)
    has_shards = bool(list(base.glob("discard_*.npz"))) or bool(list(base.glob("pegging_*.npz")))
    if has_shards and run_id is not None:
        return str(base)
    return _resolve_output_dir(base_dir, version, run_id, new_run=False)


def _train_mlp(args, dataset_dir: str, hidden: str, models_dir: str) -> str:
    train_args = argparse.Namespace(
        data_dir=dataset_dir,
        extra_data_dir=None,
        extra_ratio=0.0,
        pegging_data_dir=args.pegging_data_dir,
        models_dir=models_dir,
        model_version=args.model_version,
        run_id=None,
        discard_loss=args.discard_loss,
        discard_feature_set=args.discard_feature_set,
        pegging_feature_set=args.pegging_feature_set,
        model_type="mlp",
        mlp_hidden=hidden,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2=args.l2,
        seed=args.seed,
        eval_samples=args.eval_samples,
        max_shards=args.max_shards,
        rank_pairs_per_hand=args.rank_pairs_per_hand,
    )
    train_linear_models(train_args)
    return models_dir


def _benchmark_model(args, models_dir: str, label: str) -> None:
    bench_args = argparse.Namespace(
        players="NeuralRegressionPlayer,beginner",
        benchmark_games=args.benchmark_games,
        benchmark_workers=args.benchmark_workers,
        benchmark_games_per_worker=args.benchmark_games_per_worker,
        models_dir=models_dir,
        model_version=args.model_version,
        model_run_id=None,
        latest_model=False,
        data_dir=args.data_dir,
        max_shards=args.max_shards,
        seed=args.seed,
        fallback_player="beginner",
        model_tag=label,
        discard_feature_set=args.discard_feature_set,
        pegging_feature_set=args.pegging_feature_set,
        auto_mixed_benchmarks=False,
        games=args.benchmark_games,
    )
    benchmark_2_players(bench_args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--dataset_run_id", type=str, default=DEFAULT_DATASET_RUN_ID or None)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--discard_loss", type=str, default=DEFAULT_DISCARD_LOSS, choices=["classification", "regression", "ranking"])
    ap.add_argument("--discard_feature_set", type=str, default=DEFAULT_DISCARD_FEATURE_SET, choices=["base", "engineered_no_scores", "full"])
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    ap.add_argument("--benchmark_games", type=int, default=3000)
    ap.add_argument("--benchmark_workers", type=int, default=DEFAULT_BENCHMARK_WORKERS)
    ap.add_argument("--benchmark_games_per_worker", type=int, default=DEFAULT_BENCHMARK_GAMES_PER_WORKER)
    ap.add_argument("--pegging_data_dir", type=str, default=DEFAULT_PEGGING_DATA_DIR)
    ap.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Skip training and only benchmark existing model dirs.",
    )
    ap.add_argument(
        "--benchmark_dirs",
        type=str,
        default="",
        help="Semicolon-separated label=path pairs for benchmarking only.",
    )
    ap.add_argument("--train_workers", type=int, default=0, help="0 means one worker per model variant.")
    ap.add_argument(
        "--mlp_variants",
        type=str,
        default="small=128,64;medium=256,128;large=512,256;xl=1024,512;small3=128,64,32;mini3=64,32,16",
        help="Semicolon-separated label=hidden_sizes pairs.",
    )
    args = ap.parse_args()

    dataset_dir = _resolve_dataset_dir(args.data_dir, args.dataset_version, args.dataset_run_id)
    print(f"Dataset dir: {dataset_dir}")

    variants: dict[str, str] = {}
    for part in [p.strip() for p in args.mlp_variants.split(";") if p.strip()]:
        if "=" not in part:
            raise SystemExit(f"Invalid --mlp_variants entry: {part!r}")
        label, hidden = part.split("=", 1)
        variants[label.strip()] = hidden.strip()

    benchmark_dirs: dict[str, str] = {}
    if args.benchmark_dirs.strip():
        for part in [p.strip() for p in args.benchmark_dirs.split(";") if p.strip()]:
            if "=" not in part:
                raise SystemExit(f"Invalid --benchmark_dirs entry: {part!r}")
            label, path = part.split("=", 1)
            benchmark_dirs[label.strip()] = path.strip()
    elif args.benchmark_only:
        base = Path(args.models_dir)
        version_dir = base / args.model_version if args.model_version else base
        for label in variants:
            benchmark_dirs[label] = _resolve_variant_dir(args.models_dir, args.model_version, label)

    trained_dirs: dict[str, str] = {}
    if not args.benchmark_only:
        variant_jobs: list[tuple[str, str, str]] = []
        for label, hidden in variants.items():
            model_dir = _resolve_variant_dir(args.models_dir, args.model_version, label)
            variant_jobs.append((label, hidden, model_dir))

        train_workers = args.train_workers or len(variant_jobs)
        if train_workers <= 1 or len(variant_jobs) <= 1:
            for label, hidden, model_dir in variant_jobs:
                print(f"Training {label} MLP...")
                _train_mlp(args, dataset_dir, hidden, model_dir)
                print(f"{label} model dir: {model_dir}")
                trained_dirs[label] = model_dir
        else:
            print(f"Training {len(variant_jobs)} MLPs with {train_workers} workers...")
            with ProcessPoolExecutor(max_workers=train_workers) as pool:
                future_map = {
                    pool.submit(_train_mlp, args, dataset_dir, hidden, model_dir): (label, model_dir)
                    for label, hidden, model_dir in variant_jobs
                }
                for future in as_completed(future_map):
                    label, model_dir = future_map[future]
                    future.result()
                    print(f"{label} model dir: {model_dir}")
                    trained_dirs[label] = model_dir

    if args.benchmark_only:
        for label, model_dir in benchmark_dirs.items():
            print(f"Benchmark: {label} vs beginner")
            _benchmark_model(args, model_dir, label)
    else:
        for label, model_dir in trained_dirs.items():
            print(f"Benchmark: {label} vs beginner")
            _benchmark_model(args, model_dir, label)
