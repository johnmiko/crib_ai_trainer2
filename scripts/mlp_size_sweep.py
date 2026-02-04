"""Train small/large MLPs and benchmark against medium.

This script trains two new MLP models (small + large) and then benchmarks
small, medium (latest existing), and large against the medium player.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _resolve_dataset_dir(base_dir: str, version: str, run_id: str | None) -> str:
    base = Path(base_dir)
    has_shards = bool(list(base.glob("discard_*.npz"))) or bool(list(base.glob("pegging_*.npz")))
    if has_shards and run_id is not None:
        return str(base)
    return _resolve_output_dir(base_dir, version, run_id, new_run=False)


def _train_mlp(args, dataset_dir: str, hidden: str) -> str:
    models_dir = _resolve_models_dir(args.models_dir, args.model_version, None)
    train_args = argparse.Namespace(
        data_dir=dataset_dir,
        extra_data_dir=None,
        extra_ratio=0.0,
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
        players="NeuralRegressionPlayer,medium",
        benchmark_games=args.benchmark_games,
        models_dir=models_dir,
        model_version=args.model_version,
        model_run_id=None,
        latest_model=False,
        data_dir=args.data_dir,
        max_shards=args.max_shards,
        seed=args.seed,
        fallback_player="medium",
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
    ap.add_argument("--mlp_small", type=str, default="128,64")
    ap.add_argument("--mlp_large", type=str, default="512,256")
    args = ap.parse_args()

    dataset_dir = _resolve_dataset_dir(args.data_dir, args.dataset_version, args.dataset_run_id)
    print(f"Dataset dir: {dataset_dir}")

    print("Training small MLP...")
    small_dir = _train_mlp(args, dataset_dir, args.mlp_small)
    print(f"Small model dir: {small_dir}")

    print("Training large MLP...")
    large_dir = _train_mlp(args, dataset_dir, args.mlp_large)
    print(f"Large model dir: {large_dir}")

    # Medium model: latest existing run under model_version
    version_dir = Path(args.models_dir) / args.model_version
    latest_run = _find_latest_run_id(version_dir)
    medium_dir = str(version_dir / latest_run) if latest_run else str(version_dir)
    print(f"Medium model dir (latest): {medium_dir}")

    print("Benchmark: medium vs medium (latest existing)")
    _benchmark_model(args, medium_dir, "medium")
    print("Benchmark: small vs medium")
    _benchmark_model(args, small_dir, "small")
    print("Benchmark: large vs medium")
    _benchmark_model(args, large_dir, "large")
