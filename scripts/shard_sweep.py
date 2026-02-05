"""Train models over decreasing shard counts and benchmark each.

Goal: find how many shards are needed before performance stops improving.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import (
    MODELS_DIR,
    DEFAULT_MODEL_VERSION,
    DEFAULT_DISCARD_LOSS,
    DEFAULT_DISCARD_FEATURE_SET,
    DEFAULT_PEGGING_MODEL_FEATURE_SET,
    DEFAULT_MODEL_TYPE,
    DEFAULT_MLP_HIDDEN,
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
    DEFAULT_BENCHMARK_PLAYERS,
    DEFAULT_FALLBACK_PLAYER,
    DEFAULT_MAX_BUFFER_GAMES,
)
from scripts.train_models import train_models
from scripts.benchmark_2_players import benchmark_2_players


def _find_latest_run_id(version_dir: Path) -> str:
    if not version_dir.exists():
        raise SystemExit(f"Missing model version dir: {version_dir}")
    run_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        raise SystemExit(f"No numeric run folders found in {version_dir}")
    run_id = max(int(p.name) for p in run_dirs)
    return f"{run_id:03d}"


def _count_discard_shards(data_dir: Path) -> int:
    shards = sorted(data_dir.glob("discard_*.npz"))
    if not shards:
        raise SystemExit(f"No discard shards found in {data_dir}")
    return len(shards)


def _resolve_data_dir(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Data dir not found: {p}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Dataset shard directory (e.g., il_datasets/xxx/001).")
    ap.add_argument("--pegging_data_dir", type=str, default=None, help="Optional pegging shard dir.")
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--discard_loss", type=str, default=DEFAULT_DISCARD_LOSS, choices=["classification", "regression", "ranking"])
    ap.add_argument("--discard_feature_set", type=str, default=DEFAULT_DISCARD_FEATURE_SET, choices=["base", "engineered_no_scores", "engineered_no_scores_pev", "full", "full_pev"])
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["linear", "mlp", "gbt", "rf"])
    ap.add_argument("--mlp_hidden", type=str, default=DEFAULT_MLP_HIDDEN)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--torch_threads", type=int, default=8)
    ap.add_argument("--parallel_heads", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    ap.add_argument("--start_shards", type=int, default=None, help="Start from this max_shards (defaults to all).")
    ap.add_argument("--min_shards", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument("--benchmark_workers", type=int, default=DEFAULT_BENCHMARK_WORKERS)
    ap.add_argument("--players", type=str, default=DEFAULT_BENCHMARK_PLAYERS)
    ap.add_argument("--fallback_player", type=str, default=DEFAULT_FALLBACK_PLAYER)
    ap.add_argument("--max_buffer_games", type=int, default=DEFAULT_MAX_BUFFER_GAMES)
    args = ap.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    pegging_data_dir = Path(args.pegging_data_dir) if args.pegging_data_dir else data_dir

    total_shards = _count_discard_shards(data_dir)
    start_shards = args.start_shards or total_shards
    if start_shards > total_shards:
        raise SystemExit(f"--start_shards={start_shards} exceeds available shards ({total_shards}).")
    if args.min_shards <= 0:
        raise SystemExit("--min_shards must be > 0.")
    if args.step <= 0:
        raise SystemExit("--step must be > 0.")

    version_dir = Path(args.models_dir) / args.model_version
    base_run_id = _find_latest_run_id(version_dir)

    for max_shards in range(start_shards, args.min_shards - 1, -args.step):
        model_dir = version_dir / f"{base_run_id}_{max_shards}shards"
        print(f"\n=== Shard sweep: max_shards={max_shards} -> {model_dir} ===")

        train_args = SimpleNamespace(
            data_dir=str(data_dir),
            pegging_data_dir=str(pegging_data_dir),
            extra_data_dir=None,
            extra_ratio=0.0,
            models_dir=str(model_dir),
            model_version=args.model_version,
            run_id=None,
            discard_loss=args.discard_loss,
            discard_feature_set=args.discard_feature_set,
            pegging_feature_set=args.pegging_feature_set,
            model_type=args.model_type,
            mlp_hidden=args.mlp_hidden,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            seed=args.seed,
            torch_threads=args.torch_threads,
            parallel_heads=args.parallel_heads,
            eval_samples=args.eval_samples,
            max_shards=max_shards,
            rank_pairs_per_hand=args.rank_pairs_per_hand,
            max_train_samples=None,
        )
        train_models(train_args)

        bench_args = SimpleNamespace(
            players=args.players,
            benchmark_games=args.benchmark_games,
            benchmark_workers=args.benchmark_workers,
            max_buffer_games=args.max_buffer_games,
            models_dir=str(model_dir),
            model_version=args.model_version,
            model_run_id=None,
            latest_model=False,
            data_dir=str(data_dir),
            max_shards=max_shards,
            seed=args.seed,
            fallback_player=args.fallback_player,
            model_tag=f"{args.model_version}-{model_dir.name}",
            discard_feature_set=args.discard_feature_set,
            pegging_feature_set=args.pegging_feature_set,
            auto_mixed_benchmarks=False,
            games=args.benchmark_games,
            no_benchmark_write=False,
            benchmark_output_path="shard_size_benchmark.txt",
            experiments_output_path=None,
        )
        benchmark_2_players(bench_args)

# Script summary: train on decreasing shard counts and benchmark each size.
