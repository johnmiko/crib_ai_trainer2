"""Train multiple MLP sizes and benchmark against medium.

This script trains several MLP models and then benchmarks each against
the medium player, plus the latest existing "medium" model as a baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, ".")

from crib_ai_trainer.constants import (
    TRAINING_DATA_DIR,
    MODELS_DIR,
    DEFAULT_DATASET_VERSION,
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
    DEFAULT_PEGGING_DATA_DIR,
    DEFAULT_MLP_HIDDEN,
)
from scripts.generate_il_data import _resolve_output_dir
from scripts.train_models import train_models, _resolve_models_dir
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


def _resolve_dataset_dir(base_dir: str, version: str) -> str:
    base = Path(base_dir)
    has_shards = bool(list(base.glob("discard_*.npz"))) or bool(list(base.glob("pegging_*.npz")))
    if has_shards:
        return str(base)
    return _resolve_output_dir(base_dir, version)


@dataclass(frozen=True)
class VariantConfig:
    label: str
    model_type: str
    mlp_hidden: str | None = None
    rnn_hidden: int | None = None
    transformer: tuple[int, int, int, int, float] | None = None
    discard_model_type: str | None = None
    discard_mlp_hidden: str | None = None
    discard_rnn_hidden: int | None = None
    discard_transformer: tuple[int, int, int, int, float] | None = None


def _train_variant(
    args,
    dataset_dir: str,
    pegging_dataset_dir: str,
    variant: VariantConfig,
    models_dir: str,
) -> str:
    if variant.model_type in {"gru", "lstm", "transformer"} and args.pegging_feature_set != "full_seq":
        raise SystemExit("pegging_feature_set must be full_seq for GRU/LSTM/transformer models.")
    if variant.model_type == "mlp" and not variant.mlp_hidden and not args.discard_only:
        raise SystemExit(f"MLP variant {variant.label} is missing hidden sizes.")
    mlp_hidden = variant.mlp_hidden or args.mlp_hidden
    discard_model_type = variant.discard_model_type or args.discard_model_type or args.model_type
    discard_mlp_hidden = variant.discard_mlp_hidden or mlp_hidden
    train_args = argparse.Namespace(
        data_dir=dataset_dir,
        extra_data_dir=None,
        extra_ratio=0.0,
        pegging_data_dir=pegging_dataset_dir,
        models_dir=models_dir,
        model_version=args.model_version,
        run_id=None,
        discard_loss=args.discard_loss,
        discard_feature_set=args.discard_feature_set,
        pegging_feature_set=args.pegging_feature_set,
        model_type=args.model_type,
        mlp_hidden=mlp_hidden,
        discard_mlp_hidden=discard_mlp_hidden,
        pegging_mlp_hidden=mlp_hidden,
        discard_model_type=discard_model_type,
        pegging_model_type=variant.model_type,
        pegging_rnn_hidden=variant.rnn_hidden or args.pegging_rnn_hidden,
        pegging_transformer_d_model=(variant.transformer[0] if variant.transformer else args.pegging_transformer_d_model),
        pegging_transformer_heads=(variant.transformer[1] if variant.transformer else args.pegging_transformer_heads),
        pegging_transformer_layers=(variant.transformer[2] if variant.transformer else args.pegging_transformer_layers),
        pegging_transformer_ff_dim=(variant.transformer[3] if variant.transformer else args.pegging_transformer_ff_dim),
        pegging_transformer_dropout=(variant.transformer[4] if variant.transformer else args.pegging_transformer_dropout),
        discard_rnn_hidden=variant.discard_rnn_hidden or args.discard_rnn_hidden,
        discard_transformer_d_model=(
            variant.discard_transformer[0] if variant.discard_transformer else args.discard_transformer_d_model
        ),
        discard_transformer_heads=(
            variant.discard_transformer[1] if variant.discard_transformer else args.discard_transformer_heads
        ),
        discard_transformer_layers=(
            variant.discard_transformer[2] if variant.discard_transformer else args.discard_transformer_layers
        ),
        discard_transformer_ff_dim=(
            variant.discard_transformer[3] if variant.discard_transformer else args.discard_transformer_ff_dim
        ),
        discard_transformer_dropout=(
            variant.discard_transformer[4] if variant.discard_transformer else args.discard_transformer_dropout
        ),
        discard_only=args.discard_only,
        pegging_only=args.pegging_only,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2=args.l2,
        seed=args.seed,
        torch_threads=args.torch_threads,
        parallel_heads=args.parallel_heads,
        eval_samples=args.eval_samples,
        max_shards=args.max_shards,
        rank_pairs_per_hand=args.rank_pairs_per_hand,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    train_models(train_args)
    return models_dir


def _benchmark_model(args, models_dir: str, label: str, data_dir: str) -> None:
    model_tag = f"{args.model_version}-{Path(models_dir).name}"
    players = args.players
    bench_args = argparse.Namespace(
        players=players,
        benchmark_games=args.benchmark_games,
        benchmark_workers=args.benchmark_workers,
        max_buffer_games=args.max_buffer_games,
        models_dir=models_dir,
        model_version=args.model_version,
        model_run_id=None,
        latest_model=False,
        data_dir=data_dir,
        max_shards=args.max_shards,
        seed=args.seed,
        fallback_player="beginner",
        model_tag=model_tag,
        discard_feature_set=args.discard_feature_set,
        pegging_feature_set=args.pegging_feature_set,
        auto_mixed_benchmarks=False,
        games=args.benchmark_games,
    )
    if args.benchmark_parts == "combined":
        benchmark_2_players(bench_args)
        return
    parts = [p.strip() for p in args.players.split(",") if p.strip()]
    opponent = parts[1] if len(parts) >= 2 else "beginner"
    if args.benchmark_parts in {"discard", "both"}:
        bench_args.players = f"NeuralDiscardOnlyPlayer,{opponent}"
        benchmark_2_players(bench_args)
    if args.benchmark_parts in {"pegging", "both"}:
        bench_args.players = f"NeuralPegOnlyPlayer,{opponent}"
        benchmark_2_players(bench_args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument(
        "--pegging_data_dir",
        type=str,
        default=DEFAULT_PEGGING_DATA_DIR,
        help="Base dir for pegging shards (can be a dataset dir or root datasets dir).",
    )
    ap.add_argument(
        "--pegging_dataset_version",
        type=str,
        default=None,
        help="Optional dataset version for pegging data (defaults to --dataset_version).",
    )
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--discard_loss", type=str, default=DEFAULT_DISCARD_LOSS, choices=["classification", "regression", "ranking"])
    ap.add_argument("--discard_feature_set", type=str, default=DEFAULT_DISCARD_FEATURE_SET, choices=["base", "engineered_no_scores", "full", "full_pev"])
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full", "full_seq"])
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Stop after N shards without loss improvement.",
    )
    ap.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum loss improvement to reset early stopping.",
    )
    ap.add_argument(
        "--benchmark_parts",
        type=str,
        default="combined",
        choices=["combined", "discard", "pegging", "both"],
        help="Which parts to benchmark (combined/discard-only/pegging-only/both).",
    )
    ap.add_argument("--benchmark_games", type=int, default=3000)
    ap.add_argument("--benchmark_workers", type=int, default=DEFAULT_BENCHMARK_WORKERS)
    ap.add_argument("--players", type=str, default="AIPlayer,beginner")
    ap.add_argument("--mlp_hidden", type=str, default=DEFAULT_MLP_HIDDEN, help="Default MLP sizes for non-MLP variants.")
    ap.add_argument("--model_type", type=str, default="mlp", choices=["linear", "mlp", "gbt", "rf"])
    ap.add_argument("--discard_model_type", type=str, default=None, choices=["linear", "mlp", "gbt", "rf", "gru", "lstm", "transformer"])
    ap.add_argument("--pegging_rnn_hidden", type=int, default=64, help="Default GRU/LSTM hidden size.")
    ap.add_argument("--discard_rnn_hidden", type=int, default=64, help="Hidden size for GRU/LSTM discard model.")
    ap.add_argument("--pegging_transformer_d_model", type=int, default=128, help="Transformer d_model for pegging.")
    ap.add_argument("--pegging_transformer_heads", type=int, default=4, help="Transformer num heads for pegging.")
    ap.add_argument("--pegging_transformer_layers", type=int, default=2, help="Transformer layers for pegging.")
    ap.add_argument("--pegging_transformer_ff_dim", type=int, default=256, help="Transformer FFN dim for pegging.")
    ap.add_argument("--pegging_transformer_dropout", type=float, default=0.1, help="Transformer dropout for pegging.")
    ap.add_argument("--discard_transformer_d_model", type=int, default=128, help="Transformer d_model for discard.")
    ap.add_argument("--discard_transformer_heads", type=int, default=4, help="Transformer num heads for discard.")
    ap.add_argument("--discard_transformer_layers", type=int, default=2, help="Transformer layers for discard.")
    ap.add_argument("--discard_transformer_ff_dim", type=int, default=256, help="Transformer FFN dim for discard.")
    ap.add_argument("--discard_transformer_dropout", type=float, default=0.1, help="Transformer dropout for discard.")
    ap.add_argument("--torch_threads", type=int, default=8, help="Torch CPU thread count (intra/inter-op).")
    ap.add_argument(
        "--parallel_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train discard and pegging heads in parallel.",
    )
    ap.add_argument("--max_buffer_games", type=int, default=500)
    ap.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Skip training and only benchmark existing model dirs.",
    )
    ap.add_argument(
        "--run_latest_benchmark",
        action="store_true",
        help="Benchmark the latest run_id_* variant folders for the configured model_version.",
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
        default="",
        help="Semicolon-separated label=hidden_sizes pairs.",
    )
    ap.add_argument(
        "--discard_mlp_variants",
        type=str,
        default="",
        help="Semicolon-separated discard label=hidden_sizes pairs.",
    )
    ap.add_argument(
        "--rnn_variants",
        type=str,
        default="",
        help="Semicolon-separated label=gru:128 or label=lstm:256 entries.",
    )
    ap.add_argument(
        "--transformer_variants",
        type=str,
        default="",
        help="Semicolon-separated label=d_model,heads,layers,ff_dim,dropout entries.",
    )
    ap.add_argument(
        "--discard_rnn_variants",
        type=str,
        default="",
        help="Semicolon-separated discard label=gru:128 or label=lstm:256 entries.",
    )
    ap.add_argument(
        "--discard_transformer_variants",
        type=str,
        default="",
        help="Semicolon-separated discard label=d_model,heads,layers,ff_dim,dropout entries.",
    )
    ap.add_argument(
        "--custom_sizes",
        type=str,
        default="",
        help="Semicolon-separated sizes or label=sizes (e.g. 128,64;small3=128,64,32).",
    )
    ap.add_argument(
        "--pegging_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only the pegging model for each variant.",
    )
    ap.add_argument(
        "--discard_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only the discard model for each variant.",
    )
    args = ap.parse_args()

    if args.pegging_only and args.discard_only:
        raise SystemExit("--pegging_only and --discard_only are mutually exclusive.")

    dataset_dir = _resolve_dataset_dir(args.data_dir, args.dataset_version)
    pegging_version = args.pegging_dataset_version or args.dataset_version
    pegging_dataset_dir = _resolve_dataset_dir(args.pegging_data_dir, pegging_version)
    print(f"Dataset dir: {dataset_dir}")
    print(f"Pegging dataset dir: {pegging_dataset_dir}")

    variants: list[VariantConfig] = []

    def _add_variant(v: VariantConfig) -> None:
        if any(existing.label == v.label for existing in variants):
            raise SystemExit(f"Duplicate model label: {v.label}")
        variants.append(v)

    for part in [p.strip() for p in args.mlp_variants.split(";") if p.strip()]:
        if "=" in part:
            label, hidden = part.split("=", 1)
            label = label.strip()
            hidden = hidden.strip()
        else:
            hidden = part
            label = part.replace(",", "x").replace(" ", "")
        _add_variant(VariantConfig(label=label, model_type="mlp", mlp_hidden=hidden))
    for part in [p.strip() for p in args.discard_mlp_variants.split(";") if p.strip()]:
        if "=" in part:
            label, hidden = part.split("=", 1)
            label = label.strip()
            hidden = hidden.strip()
        else:
            hidden = part
            label = part.replace(",", "x").replace(" ", "")
        _add_variant(
            VariantConfig(
                label=label,
                model_type=args.model_type,
                discard_model_type="mlp",
                discard_mlp_hidden=hidden,
            )
        )
    if args.custom_sizes.strip():
        for part in [p.strip() for p in args.custom_sizes.split(";") if p.strip()]:
            if "=" in part:
                label, hidden = part.split("=", 1)
                label = label.strip()
                hidden = hidden.strip()
            else:
                hidden = part
                label = part.replace(",", "x").replace(" ", "")
            _add_variant(VariantConfig(label=label, model_type="mlp", mlp_hidden=hidden))

    if args.rnn_variants.strip():
        for part in [p.strip() for p in args.rnn_variants.split(";") if p.strip()]:
            if "=" not in part or ":" not in part:
                raise SystemExit(f"Invalid --rnn_variants entry: {part!r}")
            label, spec = part.split("=", 1)
            model_type, hidden = spec.split(":", 1)
            model_type = model_type.strip()
            if model_type not in {"gru", "lstm"}:
                raise SystemExit(f"Invalid RNN model type {model_type!r} in {part!r}")
            _add_variant(
                VariantConfig(
                    label=label.strip(),
                    model_type=model_type,
                    rnn_hidden=int(hidden.strip()),
                )
            )
    if args.discard_rnn_variants.strip():
        for part in [p.strip() for p in args.discard_rnn_variants.split(";") if p.strip()]:
            if "=" not in part or ":" not in part:
                raise SystemExit(f"Invalid --discard_rnn_variants entry: {part!r}")
            label, spec = part.split("=", 1)
            model_type, hidden = spec.split(":", 1)
            model_type = model_type.strip()
            if model_type not in {"gru", "lstm"}:
                raise SystemExit(f"Invalid discard RNN model type {model_type!r} in {part!r}")
            _add_variant(
                VariantConfig(
                    label=label.strip(),
                    model_type=args.model_type,
                    discard_model_type=model_type,
                    discard_rnn_hidden=int(hidden.strip()),
                )
            )

    if args.transformer_variants.strip():
        for part in [p.strip() for p in args.transformer_variants.split(";") if p.strip()]:
            if "=" not in part:
                raise SystemExit(f"Invalid --transformer_variants entry: {part!r}")
            label, spec = part.split("=", 1)
            parts = [p.strip() for p in spec.split(",") if p.strip()]
            if len(parts) != 5:
                raise SystemExit(
                    f"Transformer variants must be d_model,heads,layers,ff_dim,dropout (got {spec!r})"
                )
            d_model, heads, layers, ff_dim = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
            dropout = float(parts[4])
            _add_variant(
                VariantConfig(
                    label=label.strip(),
                    model_type="transformer",
                    transformer=(d_model, heads, layers, ff_dim, dropout),
                )
            )
    if args.discard_transformer_variants.strip():
        for part in [p.strip() for p in args.discard_transformer_variants.split(";") if p.strip()]:
            if "=" not in part:
                raise SystemExit(f"Invalid --discard_transformer_variants entry: {part!r}")
            label, spec = part.split("=", 1)
            parts = [p.strip() for p in spec.split(",") if p.strip()]
            if len(parts) != 5:
                raise SystemExit(
                    f"Discard transformer variants must be d_model,heads,layers,ff_dim,dropout (got {spec!r})"
                )
            d_model, heads, layers, ff_dim = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
            dropout = float(parts[4])
            _add_variant(
                VariantConfig(
                    label=label.strip(),
                    model_type=args.model_type,
                    discard_model_type="transformer",
                    discard_transformer=(d_model, heads, layers, ff_dim, dropout),
                )
            )

    if not variants:
        raise SystemExit("No model variants specified.")
    if any(v.model_type in {"gru", "lstm", "transformer"} for v in variants) and args.pegging_feature_set != "full_seq":
        raise SystemExit("GRU/LSTM/transformer variants require --pegging_feature_set full_seq.")

    benchmark_dirs: dict[str, str] = {}
    if args.benchmark_dirs.strip():
        for part in [p.strip() for p in args.benchmark_dirs.split(";") if p.strip()]:
            if "=" not in part:
                raise SystemExit(f"Invalid --benchmark_dirs entry: {part!r}")
            label, path = part.split("=", 1)
            benchmark_dirs[label.strip()] = path.strip()
    elif args.benchmark_only or args.run_latest_benchmark:
        base = Path(args.models_dir)
        version_dir = base / args.model_version if args.model_version else base
        latest_run = _find_latest_run_id(version_dir)
        if latest_run is None:
            raise SystemExit(f"No run folders found under {version_dir}")
        for label in (v.label for v in variants):
            path = version_dir / f"{latest_run}_{label}"
            if not path.exists():
                raise SystemExit(f"Missing model dir for {label}: {path}")
            benchmark_dirs[label] = str(path)

    trained_dirs: dict[str, str] = {}
    if args.run_latest_benchmark:
        args.benchmark_only = True

    if not args.benchmark_only:
        variant_jobs: list[tuple[VariantConfig, str]] = []
        for variant in variants:
            model_dir = _resolve_variant_dir(args.models_dir, args.model_version, variant.label)
            variant_jobs.append((variant, model_dir))

        train_workers = args.train_workers or len(variant_jobs)
        if train_workers <= 1 or len(variant_jobs) <= 1:
            for variant, model_dir in variant_jobs:
                print(f"Training {variant.label} ({variant.model_type})...")
                _train_variant(args, dataset_dir, pegging_dataset_dir, variant, model_dir)
                print(f"{variant.label} model dir: {model_dir}")
                trained_dirs[variant.label] = model_dir
        else:
            print(f"Training {len(variant_jobs)} variants with {train_workers} workers...")
            with ProcessPoolExecutor(max_workers=train_workers) as pool:
                future_map = {
                    pool.submit(
                        _train_variant,
                        args,
                        dataset_dir,
                        pegging_dataset_dir,
                        variant,
                        model_dir,
                    ): (variant.label, model_dir)
                    for variant, model_dir in variant_jobs
                }
                for future in as_completed(future_map):
                    label, model_dir = future_map[future]
                    future.result()
                    print(f"{label} model dir: {model_dir}")
                    trained_dirs[label] = model_dir

    if args.benchmark_only:
        for label, model_dir in benchmark_dirs.items():
            print(f"Benchmark: {label} vs beginner")
            _benchmark_model(args, model_dir, label, dataset_dir)
    else:
        for label, model_dir in trained_dirs.items():
            print(f"Benchmark: {label} vs beginner")
            _benchmark_model(args, model_dir, label, dataset_dir)

# Script summary: train multiple MLP sizes and benchmark each against a baseline opponent.
