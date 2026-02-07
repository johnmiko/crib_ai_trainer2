"""Train models over decreasing shard counts and benchmark each.

Goal: find how many shards are needed before performance stops improving.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import sys
sys.path.insert(0, ".")

import numpy as np

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
from crib_ai_trainer.players.neural_player import (
    get_discard_feature_indices,
    get_pegging_feature_indices,
    LinearValueModel,
    MLPValueModel,
)


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


def _write_model_meta(model_dir: Path, *, model_version: str, model_type: str, mlp_hidden: str,
                      discard_feature_set: str, pegging_feature_set: str, num_shards_used: int) -> None:
    meta = {
        "model_version": model_version,
        "model_type": model_type,
        "mlp_hidden": mlp_hidden,
        "discard_feature_set": discard_feature_set,
        "pegging_feature_set": pegging_feature_set,
        "num_shards_used": num_shards_used,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_incremental_models(model_dir: Path, model_type: str, discard_dim: int, pegging_dim: int, mlp_hidden: str):
    if model_type == "mlp":
        discard_path = model_dir / "discard_mlp.pt"
        pegging_path = model_dir / "pegging_mlp.pt"
        if not discard_path.exists() or not pegging_path.exists():
            raise SystemExit(f"Missing MLP model files in {model_dir}")
        discard_model = MLPValueModel.load_pt(str(discard_path))
        pegging_model = MLPValueModel.load_pt(str(pegging_path))
        return discard_model, pegging_model
    if model_type == "linear":
        discard_path = model_dir / "discard_linear.npz"
        pegging_path = model_dir / "pegging_linear.npz"
        if not discard_path.exists() or not pegging_path.exists():
            raise SystemExit(f"Missing linear model files in {model_dir}")
        discard_model = LinearValueModel.load_npz(str(discard_path))
        pegging_model = LinearValueModel.load_npz(str(pegging_path))
        return discard_model, pegging_model
    raise SystemExit(f"Incremental sweep only supports model_type=mlp or linear (got {model_type}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Dataset shard directory (e.g., datasets/xxx).")
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
    ap.add_argument("--end_shards", type=int, default=None, help="Only for incremental: final shard count.")
    ap.add_argument("--incremental", action="store_true", default=True, help="Incrementally train on new shards instead of retraining.")
    ap.add_argument("--incremental_epochs", type=int, default=None, help="Epochs per new shard (defaults to --epochs).")
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

    if args.incremental:
        if args.discard_loss != "regression":
            raise SystemExit("--incremental requires discard_loss=regression.")
        if args.model_type not in {"mlp", "linear"}:
            raise SystemExit("--incremental only supports model_type=mlp or linear.")
        end_shards = args.end_shards or total_shards
        if end_shards < start_shards:
            raise SystemExit("--end_shards must be >= --start_shards for incremental sweep.")
        if args.incremental_epochs is not None and args.incremental_epochs <= 0:
            raise SystemExit("--incremental_epochs must be > 0 if provided.")
        inc_epochs = args.incremental_epochs or args.epochs
        discard_feature_indices = get_discard_feature_indices(args.discard_feature_set)
        pegging_feature_indices = get_pegging_feature_indices(args.pegging_feature_set)
        discard_shards = sorted(data_dir.glob("discard_*.npz"))
        pegging_shards = sorted(pegging_data_dir.glob("pegging_*.npz"))
        if len(discard_shards) != len(pegging_shards):
            min_count = min(len(discard_shards), len(pegging_shards))
            discard_shards = discard_shards[:min_count]
            pegging_shards = pegging_shards[:min_count]
            total_shards = min_count
            if start_shards > total_shards:
                start_shards = total_shards
            if end_shards > total_shards:
                end_shards = total_shards
        prev_model_dir = None
        prev_shards = 0
        if start_shards <= 0:
            start_shards = 1
        if end_shards <= 0:
            raise SystemExit("--end_shards must be > 0.")
        for max_shards in range(start_shards, end_shards + 1, args.step):
            model_dir = version_dir / f"{base_run_id}_inc_{max_shards}shards"
            print(f"\n=== Incremental shard sweep: max_shards={max_shards} -> {model_dir} ===")
            if prev_model_dir is None:
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
            else:
                discard_model, pegging_model = _load_incremental_models(
                    Path(prev_model_dir),
                    args.model_type,
                    int(len(discard_feature_indices)),
                    int(len(pegging_feature_indices)),
                    args.mlp_hidden,
                )
                for shard_idx in range(prev_shards, max_shards):
                    d_path = discard_shards[shard_idx]
                    p_path = pegging_shards[shard_idx]
                    with np.load(d_path) as d:
                        Xd = d["X"].astype(np.float32)[:, discard_feature_indices]
                        yd = d["y"].astype(np.float32)
                    with np.load(p_path) as p:
                        Xp = p["X"].astype(np.float32)[:, pegging_feature_indices]
                        yp = p["y"].astype(np.float32)
                    discard_model.fit_mse(
                        Xd,
                        yd,
                        lr=args.lr,
                        epochs=inc_epochs,
                        batch_size=args.batch_size,
                        l2=args.l2,
                        seed=args.seed,
                    )
                    pegging_model.fit_mse(
                        Xp,
                        yp,
                        lr=args.lr,
                        epochs=inc_epochs,
                        batch_size=args.batch_size,
                        l2=args.l2,
                        seed=args.seed,
                    )
                model_dir.mkdir(parents=True, exist_ok=True)
                if args.model_type == "mlp":
                    discard_model.save_pt(str(Path(model_dir) / "discard_mlp.pt"))
                    pegging_model.save_pt(str(Path(model_dir) / "pegging_mlp.pt"))
                else:
                    discard_model.save_npz(str(Path(model_dir) / "discard_linear.npz"))
                    pegging_model.save_npz(str(Path(model_dir) / "pegging_linear.npz"))
                _write_model_meta(
                    Path(model_dir),
                    model_version=args.model_version,
                    model_type=args.model_type,
                    mlp_hidden=args.mlp_hidden,
                    discard_feature_set=args.discard_feature_set,
                    pegging_feature_set=args.pegging_feature_set,
                    num_shards_used=max_shards,
                )
            prev_model_dir = str(model_dir)
            prev_shards = max_shards

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
                benchmark_output_path="text/shard_size_benchmark.txt",
                experiments_output_path=None,
            )
            benchmark_2_players(bench_args)
    else:
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
                benchmark_output_path="text/shard_size_benchmark.txt",
                experiments_output_path=None,
            )
            benchmark_2_players(bench_args)

# Script summary: train on decreasing shard counts and benchmark each size.
