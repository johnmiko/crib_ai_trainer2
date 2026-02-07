"""Train models from sharded datasets.

Usage:
  python scripts/train_models.py --data_dir datasets --models_dir models --epochs 20
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from crib_ai_trainer.constants import (
    MODELS_DIR,
    TRAINING_DATA_DIR,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODEL_RUN_ID,
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
    DEFAULT_MODEL_TYPE,
    DEFAULT_MLP_HIDDEN,
)
from crib_ai_trainer.players.neural_player import (
    LinearDiscardClassifier,
    LinearValueModel,
    get_discard_feature_indices,
    get_pegging_feature_indices,
    MLPValueModel,
    GBTValueModel,
    RandomForestValueModel,
)
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _next_run_id(base_dir: str) -> str:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not existing:
        return "001"
    max_id = max(int(x) for x in existing)
    return f"{max_id + 1:03d}"

def _resolve_models_dir(base_models_dir: str, model_version: str, run_id: str | None) -> str:
    version_dir = Path(base_models_dir) / model_version
    if run_id is None:
        run_id = _next_run_id(str(version_dir))
    return str(version_dir / run_id)

def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return (128, 64)
    return tuple(int(p) for p in parts)


def _max_games_from_shards(shards: list[Path], prefix: str) -> int:
    if not shards:
        raise SystemExit(f"No {prefix} shards provided.")
    max_games = None
    for path in shards:
        stem = path.stem
        if not stem.startswith(f"{prefix}_"):
            raise SystemExit(f"Unexpected shard name '{path.name}' for prefix '{prefix}'.")
        parts = stem.split("_", 1)
        if len(parts) != 2:
            raise SystemExit(f"Invalid shard name '{path.name}'.")
        try:
            games = int(parts[1])
        except ValueError:
            raise SystemExit(f"Invalid shard suffix in '{path.name}'.")
        if max_games is None or games > max_games:
            max_games = games
    if max_games is None:
        raise SystemExit(f"Unable to parse game counts from {prefix} shards.")
    return max_games


def _load_incremental_models(model_dir: Path, model_type: str):
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
    raise SystemExit(f"--incremental only supports model_type=mlp or linear (got {model_type}).")


def train_models(args) -> int:
    if args.torch_threads is not None:
        import torch

        target_threads = int(args.torch_threads)
        if torch.get_num_threads() != target_threads:
            torch.set_num_threads(target_threads)
        if torch.get_num_interop_threads() != target_threads:
            torch.set_num_interop_threads(target_threads)
    if args.lr <= 0 or args.lr > 0.05:
        raise SystemExit(
            f"Invalid --lr={args.lr}. For stability with engineered features, use 0 < lr <= 0.05 "
            f"(recommended 0.0001–0.005). Larger values can explode and produce NaNs."
        )
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0.")
    if args.l2 < 0 or args.l2 > 0.1:
        raise SystemExit("--l2 must be in [0, 0.1]. Larger values dominate the loss and stall learning.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0.")
    if args.rank_pairs_per_hand <= 0:
        raise SystemExit("--rank_pairs_per_hand must be > 0.")
    data_dir = Path(args.data_dir)
    pegging_data_dir = Path(args.pegging_data_dir) if getattr(args, "pegging_data_dir", None) else data_dir
    extra_data_dir = Path(args.extra_data_dir) if args.extra_data_dir else None
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading training data from {data_dir}")
    discard_shards = sorted(data_dir.glob("discard_*.npz"))
    pegging_shards = sorted(pegging_data_dir.glob("pegging_*.npz"))
    if not discard_shards:
        raise SystemExit(f"No discard shards in {data_dir} (expected discard_*.npz)")
    if not pegging_shards:
        raise SystemExit(f"No pegging shards in {pegging_data_dir} (expected pegging_*.npz)")
    extra_discard_shards = []
    extra_pegging_shards = []
    if extra_data_dir:
        extra_discard_shards = sorted(extra_data_dir.glob("discard_*.npz"))
        extra_pegging_shards = sorted(extra_data_dir.glob("pegging_*.npz"))
        if not extra_discard_shards or not extra_pegging_shards:
            raise SystemExit(f"No extra shards found in {extra_data_dir}")
    if args.max_shards is not None:
        if args.max_shards <= 0:
            raise SystemExit("--max_shards must be > 0 if provided")
        discard_shards = discard_shards[: args.max_shards]
        pegging_shards = pegging_shards[: args.max_shards]
        logger.info(f"Limiting training to first {args.max_shards} shard(s)")

    discard_games_used = _max_games_from_shards(discard_shards, "discard")
    pegging_games_used = _max_games_from_shards(pegging_shards, "pegging")

    # init models
    discard_mode = args.discard_loss
    if discard_mode is None:
        if "classification" in args.data_dir:
            discard_mode = "classification"
        elif "ranking" in args.data_dir:
            discard_mode = "ranking"
        else:
            discard_mode = "regression"

    discard_feature_indices = get_discard_feature_indices(args.discard_feature_set)
    pegging_feature_indices = get_pegging_feature_indices(args.pegging_feature_set)
    model_type = args.model_type
    mlp_hidden = _parse_hidden_sizes(args.mlp_hidden)
    incremental = bool(getattr(args, "incremental", False))
    incremental_from = getattr(args, "incremental_from", None)
    incremental_start_shard = getattr(args, "incremental_start_shard", 0)
    incremental_epochs = getattr(args, "incremental_epochs", None)

    if discard_mode in {"classification", "ranking"} and model_type != "linear":
        raise SystemExit("Only linear models are supported for classification/ranking at the moment.")
    if model_type in {"gbt", "rf"} and discard_mode != "regression":
        raise SystemExit("GBT/RF models are only supported for regression.")
    if model_type in {"gbt", "rf"} and args.max_train_samples is None and args.max_shards is None:
        raise SystemExit(
            "GBT/RF training requires --max_train_samples or --max_shards to limit memory use."
        )
    if incremental:
        if discard_mode != "regression":
            raise SystemExit("--incremental requires discard_loss=regression.")
        if model_type not in {"mlp", "linear"}:
            raise SystemExit("--incremental only supports model_type=mlp or linear.")
        if args.max_train_samples is not None:
            raise SystemExit("--incremental does not support --max_train_samples.")
        if incremental_from is None:
            raise SystemExit("--incremental requires --incremental_from.")
        if len(discard_shards) != len(pegging_shards):
            raise SystemExit(
                f"Discard/pegging shard counts differ ({len(discard_shards)} vs {len(pegging_shards)}). "
                "Make them match before incremental training."
            )
        if incremental_start_shard < 0:
            raise SystemExit("--incremental_start_shard must be >= 0.")
        if incremental_start_shard >= len(discard_shards):
            raise SystemExit(
                f"--incremental_start_shard={incremental_start_shard} exceeds shard count ({len(discard_shards)})."
            )
        if incremental_epochs is not None and incremental_epochs <= 0:
            raise SystemExit("--incremental_epochs must be > 0 if provided.")

    if discard_mode == "classification":
        with np.load(discard_shards[0]) as d0:
            discard_model = LinearDiscardClassifier(int(len(discard_feature_indices)))
    else:
        with np.load(discard_shards[0]) as d0:
            if model_type == "mlp":
                discard_model = MLPValueModel(int(len(discard_feature_indices)), mlp_hidden, seed=args.seed or 0)
            elif model_type == "gbt":
                discard_model = GBTValueModel(seed=args.seed or 0, max_iter=int(args.epochs))
            elif model_type == "rf":
                discard_model = RandomForestValueModel(seed=args.seed or 0, n_estimators=int(args.epochs))
            else:
                discard_model = LinearValueModel(int(len(discard_feature_indices)))
    with np.load(pegging_shards[0]) as p0:
        if model_type == "mlp":
            pegging_model = MLPValueModel(int(len(pegging_feature_indices)), mlp_hidden, seed=args.seed or 0)
        elif model_type == "gbt":
            pegging_model = GBTValueModel(seed=args.seed or 0, max_iter=int(args.epochs))
        elif model_type == "rf":
            pegging_model = RandomForestValueModel(seed=args.seed or 0, n_estimators=int(args.epochs))
        else:
            pegging_model = LinearValueModel(int(len(pegging_feature_indices)))

    last_discard_loss = None
    last_pegging_loss = None
    eval_metrics = {}

    rng = np.random.default_rng(args.seed)
    extra_ratio = float(args.extra_ratio)
    if extra_ratio < 0.0 or extra_ratio > 1.0:
        raise SystemExit("--extra_ratio must be between 0 and 1.")

    def _load_all_discard() -> tuple[np.ndarray, np.ndarray]:
        Xs = []
        ys = []
        for shard in discard_shards:
            with np.load(shard) as d:
                Xd = d["X"].astype(np.float32)
                yd = d["y"].astype(np.float32)
            Xd = Xd[:, discard_feature_indices]
            Xs.append(Xd)
            ys.append(yd)
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        if args.max_train_samples is not None:
            if args.max_train_samples <= 0:
                raise SystemExit("--max_train_samples must be > 0 if provided.")
            if X_all.shape[0] < args.max_train_samples:
                raise SystemExit(
                    f"--max_train_samples={args.max_train_samples} exceeds available discard samples ({X_all.shape[0]})."
                )
            idx = rng.choice(X_all.shape[0], size=args.max_train_samples, replace=False)
            X_all = X_all[idx]
            y_all = y_all[idx]
        return X_all, y_all

    def _load_all_pegging() -> tuple[np.ndarray, np.ndarray]:
        Xs = []
        ys = []
        for shard in pegging_shards:
            with np.load(shard) as p:
                Xp = p["X"].astype(np.float32)
                yp = p["y"].astype(np.float32)
            Xp = Xp[:, pegging_feature_indices]
            Xs.append(Xp)
            ys.append(yp)
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        if args.max_train_samples is not None:
            if args.max_train_samples <= 0:
                raise SystemExit("--max_train_samples must be > 0 if provided.")
            if X_all.shape[0] < args.max_train_samples:
                raise SystemExit(
                    f"--max_train_samples={args.max_train_samples} exceeds available pegging samples ({X_all.shape[0]})."
                )
            idx = rng.choice(X_all.shape[0], size=args.max_train_samples, replace=False)
            X_all = X_all[idx]
            y_all = y_all[idx]
        return X_all, y_all

    if incremental:
        inc_epochs = incremental_epochs or args.epochs
        discard_model, pegging_model = _load_incremental_models(Path(incremental_from), model_type)
        extra_idx = 0
        for shard_idx in range(incremental_start_shard, len(discard_shards)):
            d_path = discard_shards[shard_idx]
            p_path = pegging_shards[shard_idx]
            with np.load(d_path) as d:
                Xd = d["X"].astype(np.float32)
                yd = d["y"].astype(np.float32)
            Xd = Xd[:, discard_feature_indices]
            with np.load(p_path) as p:
                Xp = p["X"].astype(np.float32)
                yp = p["y"].astype(np.float32)
            Xp = Xp[:, pegging_feature_indices]

            def _train_discard():
                return discard_model.fit_mse(
                    Xd,
                    yd,
                    lr=args.lr,
                    epochs=inc_epochs,
                    batch_size=args.batch_size,
                    l2=args.l2,
                    seed=args.seed,
                )

            def _train_pegging():
                return pegging_model.fit_mse(
                    Xp,
                    yp,
                    lr=args.lr,
                    epochs=inc_epochs,
                    batch_size=args.batch_size,
                    l2=args.l2,
                    seed=args.seed,
                )

            if args.parallel_heads:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_discard = pool.submit(_train_discard)
                    f_pegging = pool.submit(_train_pegging)
                    discard_losses = f_discard.result()
                    pegging_losses = f_pegging.result()
            else:
                discard_losses = _train_discard()
                pegging_losses = _train_pegging()
            last_discard_loss = float(discard_losses[-1]) if discard_losses else last_discard_loss
            last_pegging_loss = float(pegging_losses[-1]) if pegging_losses else last_pegging_loss

            if extra_data_dir is not None and extra_ratio > 0.0 and rng.random() < extra_ratio:
                d_path = extra_discard_shards[extra_idx % len(extra_discard_shards)]
                p_path = extra_pegging_shards[extra_idx % len(extra_pegging_shards)]
                extra_idx += 1
                with np.load(d_path) as d:
                    Xd = d["X"].astype(np.float32)
                    yd = d["y"].astype(np.float32)
                Xd = Xd[:, discard_feature_indices]
                with np.load(p_path) as p:
                    Xp = p["X"].astype(np.float32)
                    yp = p["y"].astype(np.float32)
                Xp = Xp[:, pegging_feature_indices]

                def _train_discard_extra():
                    return discard_model.fit_mse(
                        Xd,
                        yd,
                        lr=args.lr,
                        epochs=inc_epochs,
                        batch_size=args.batch_size,
                        l2=args.l2,
                        seed=args.seed,
                    )

                def _train_pegging_extra():
                    return pegging_model.fit_mse(
                        Xp,
                        yp,
                        lr=args.lr,
                        epochs=inc_epochs,
                        batch_size=args.batch_size,
                        l2=args.l2,
                        seed=args.seed,
                    )

                if args.parallel_heads:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        f_discard = pool.submit(_train_discard_extra)
                        f_pegging = pool.submit(_train_pegging_extra)
                        discard_losses = f_discard.result()
                        pegging_losses = f_pegging.result()
                else:
                    discard_losses = _train_discard_extra()
                    pegging_losses = _train_pegging_extra()
                last_discard_loss = float(discard_losses[-1]) if discard_losses else last_discard_loss
                last_pegging_loss = float(pegging_losses[-1]) if pegging_losses else last_pegging_loss
    elif model_type in {"gbt", "rf"}:
        print(f"Training {model_type} models on {len(discard_shards)} shard(s) (full in-memory fit).")
        Xd_all, yd_all = _load_all_discard()
        Xp_all, yp_all = _load_all_pegging()

        def _train_discard():
            discard_model.fit(Xd_all, yd_all)  # type: ignore[attr-defined]
            pred = discard_model.predict_batch(Xd_all)  # type: ignore[attr-defined]
            return [float(np.mean((pred - yd_all) ** 2))]

        def _train_pegging():
            pegging_model.fit(Xp_all, yp_all)  # type: ignore[attr-defined]
            pred = pegging_model.predict_batch(Xp_all)  # type: ignore[attr-defined]
            return [float(np.mean((pred - yp_all) ** 2))]

        if args.parallel_heads:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_discard = pool.submit(_train_discard)
                f_pegging = pool.submit(_train_pegging)
                discard_losses = f_discard.result()
                pegging_losses = f_pegging.result()
        else:
            discard_losses = _train_discard()
            pegging_losses = _train_pegging()

        last_discard_loss = float(discard_losses[-1]) if discard_losses else last_discard_loss
        last_pegging_loss = float(pegging_losses[-1]) if pegging_losses else last_pegging_loss
    else:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            extra_idx = 0
            primary_idx = 0
            steps = len(discard_shards)
            for _ in range(steps):
                use_extra = extra_data_dir is not None and rng.random() < extra_ratio
                if use_extra:
                    d_path = extra_discard_shards[extra_idx % len(extra_discard_shards)]
                    p_path = extra_pegging_shards[extra_idx % len(extra_pegging_shards)]
                    extra_idx += 1
                else:
                    d_path = discard_shards[primary_idx % len(discard_shards)]
                    p_path = pegging_shards[primary_idx % len(pegging_shards)]
                    primary_idx += 1
                logger.debug(f"  Training on shard {d_path.name} and {p_path.name}")
                with np.load(d_path) as d:
                    if discard_mode == "classification":
                        Xd = d["X"].astype(np.int64)
                        yd = d["y"].astype(np.int64)
                    else:
                        Xd = d["X"].astype(np.float32)
                        yd = d["y"].astype(np.float32)
                if discard_mode in {"classification", "ranking"}:
                    Xd = Xd[..., discard_feature_indices]
                else:
                    Xd = Xd[:, discard_feature_indices]
                # print("discard X dim", Xd.shape, "y range", float(yd.min()), float(yd.mean()), float(yd.max()))

                with np.load(p_path) as p:
                    Xp = p["X"].astype(np.float32)
                    yp = p["y"].astype(np.float32)
                Xp = Xp[:, pegging_feature_indices]
                def _train_discard():
                    if discard_mode == "classification":
                        logger.debug(f"Training discard model {discard_model}")
                        return discard_model.fit_ce( # type: ignore
                            Xd, yd, lr=args.lr, epochs=args.epochs,
                            batch_size=args.batch_size, l2=args.l2, seed=args.seed
                        )
                    if discard_mode == "ranking":
                        return discard_model.fit_rank_pairwise( # type: ignore
                            Xd, yd,
                            lr=args.lr,
                            epochs=1,
                            batch_size=args.batch_size,
                            l2=args.l2,
                            seed=args.seed,
                            pairs_per_hand=args.rank_pairs_per_hand,
                        )
                    return discard_model.fit_mse( # type: ignore
                        Xd, yd,
                        lr=args.lr,
                        epochs=1,
                        batch_size=args.batch_size,
                        l2=args.l2,
                        seed=args.seed,
                    )

                def _train_pegging():
                    return pegging_model.fit_mse(
                        Xp, yp,
                        lr=args.lr,
                        epochs=1,
                        batch_size=args.batch_size,
                        l2=args.l2,
                        seed=args.seed,
                    )

                if args.parallel_heads:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        f_discard = pool.submit(_train_discard)
                        f_pegging = pool.submit(_train_pegging)
                        discard_losses = f_discard.result()
                        pegging_losses = f_pegging.result()
                else:
                    discard_losses = _train_discard()
                    pegging_losses = _train_pegging()

                last_discard_loss = float(discard_losses[-1]) if discard_losses else last_discard_loss
                last_pegging_loss = float(pegging_losses[-1]) if pegging_losses else last_pegging_loss
                if last_pegging_loss is not None and not np.isfinite(last_pegging_loss):
                    raise SystemExit(
                        "Pegging loss became NaN/inf. Try a smaller --lr (e.g., 5e-5), "
                        "a larger --batch_size (e.g., 2048+), or increase --l2."
                    )

    if model_type == "mlp":
        discard_path = models_dir / "discard_mlp.pt"
        pegging_path = models_dir / "pegging_mlp.pt"
        discard_model.save_pt(str(discard_path))
        pegging_model.save_pt(str(pegging_path))
    elif model_type == "gbt":
        discard_path = models_dir / "discard_gbt.pkl"
        pegging_path = models_dir / "pegging_gbt.pkl"
        discard_model.save_joblib(str(discard_path))  # type: ignore[attr-defined]
        pegging_model.save_joblib(str(pegging_path))  # type: ignore[attr-defined]
    elif model_type == "rf":
        discard_path = models_dir / "discard_rf.pkl"
        pegging_path = models_dir / "pegging_rf.pkl"
        discard_model.save_joblib(str(discard_path))  # type: ignore[attr-defined]
        pegging_model.save_joblib(str(pegging_path))  # type: ignore[attr-defined]
    else:
        discard_path = models_dir / "discard_linear.npz"
        pegging_path = models_dir / "pegging_linear.npz"
        discard_model.save_npz(str(discard_path))
        pegging_model.save_npz(str(pegging_path))
    # “last loss” = the mean squared error (MSE) from the final training step that ran.
    print(f"Saved discard model -> {discard_path}")
    if last_discard_loss is not None:
        print(f"  last loss={last_discard_loss:.6f}")
    print(f"Saved pegging model -> {pegging_path}")
    if last_pegging_loss is not None:
        print(f"  last loss={last_pegging_loss:.6f}")

    if args.eval_samples > 0:
        print(f"Running quick eval on up to {args.eval_samples} samples...")
        eval_discard_path = discard_shards[-1]
        eval_pegging_path = pegging_shards[-1]
        with np.load(eval_discard_path) as d:
            Xd = d["X"]
            yd = d["y"]
        with np.load(eval_pegging_path) as p:
            Xp = p["X"]
            yp = p["y"]

        n_d = min(args.eval_samples, Xd.shape[0])
        n_p = min(args.eval_samples, Xp.shape[0])

        if discard_mode == "classification":
            Xd_eval = Xd[:n_d].astype(np.float32)
            Xd_eval = Xd_eval[..., discard_feature_indices]
            yd_eval = yd[:n_d].astype(np.int64)
            scores = np.tensordot(Xd_eval, discard_model.w, axes=([2], [0])) + discard_model.b
            preds = np.argmax(scores, axis=1)
            acc = float(np.mean(preds == yd_eval)) if n_d > 0 else 0.0
            print(f"  discard classifier top-1 acc: {acc:.3f} on {n_d} samples")
            eval_metrics["discard_classifier_top1_acc"] = acc
        elif discard_mode == "ranking":
            Xd_eval = Xd[:n_d].astype(np.float32)
            Xd_eval = Xd_eval[..., discard_feature_indices]
            yd_eval = yd[:n_d].astype(np.float32)
            # report average margin between top-1 and top-2 for model vs target
            scores = np.tensordot(Xd_eval, discard_model.w, axes=([2], [0])) + discard_model.b
            model_margins = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
            target_margins = np.sort(yd_eval, axis=1)[:, -1] - np.sort(yd_eval, axis=1)[:, -2]
            avg_model_margin = float(np.mean(model_margins))
            avg_target_margin = float(np.mean(target_margins))
            print(f"  discard ranker avg model margin: {avg_model_margin:.3f}")
            print(f"  discard ranker avg target margin: {avg_target_margin:.3f}")
            eval_metrics["discard_ranker_avg_model_margin"] = avg_model_margin
            eval_metrics["discard_ranker_avg_target_margin"] = avg_target_margin
        else:
            Xd_eval = Xd[:n_d].astype(np.float32)
            Xd_eval = Xd_eval[:, discard_feature_indices]
            yd_eval = yd[:n_d].astype(np.float32)
            pred = discard_model.predict_batch(Xd_eval)
            mse = float(np.mean((pred - yd_eval) ** 2)) if n_d > 0 else 0.0
            print(f"  discard regressor MSE: {mse:.4f} on {n_d} samples")
            eval_metrics["discard_regressor_mse"] = mse

        Xp_eval = Xp[:n_p].astype(np.float32)
        Xp_eval = Xp_eval[:, pegging_feature_indices]
        yp_eval = yp[:n_p].astype(np.float32)
        pred_p = pegging_model.predict_batch(Xp_eval)
        mse_p = float(np.mean((pred_p - yp_eval) ** 2)) if n_p > 0 else 0.0
        print(f"  pegging regressor MSE: {mse_p:.4f} on {n_p} samples")
        eval_metrics["pegging_regressor_mse"] = mse_p

    # Write model metadata for easy inspection.
    model_path = Path(models_dir)
    model_version = model_path.parent.name
    run_id = model_path.name
    model_meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "run_id": run_id,
        "data_dir": str(data_dir),
        "pegging_data_dir": str(pegging_data_dir),
        "extra_data_dir": str(extra_data_dir) if extra_data_dir else None,
        "extra_ratio": extra_ratio if extra_data_dir else 0.0,
        "incremental": incremental,
        "incremental_from": incremental_from,
        "incremental_start_shard": incremental_start_shard if incremental else None,
        "incremental_epochs": incremental_epochs if incremental else None,
        "models_dir": str(models_dir),
        "model_type": model_type,
        "discard_loss": discard_mode,
        "discard_feature_set": args.discard_feature_set,
        "pegging_feature_set": args.pegging_feature_set,
        "discard_feature_dim": int(len(discard_feature_indices)),
        "pegging_feature_dim": int(len(pegging_feature_indices)),
        "mlp_hidden": list(mlp_hidden),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "l2": args.l2,
        "seed": args.seed,
        "max_shards": args.max_shards,
        "num_shards_used": len(discard_shards),
        "discard_games_used": discard_games_used,
        "pegging_games_used": pegging_games_used,
        "training_games_used": min(discard_games_used, pegging_games_used),
        "last_discard_loss": last_discard_loss,
        "last_pegging_loss": last_pegging_loss,
        "eval_metrics": eval_metrics,
        "discard_model_file": "discard_linear.npz",
        "pegging_model_file": "pegging_linear.npz",
    }
    if model_type == "mlp":
        model_meta["discard_model_file"] = "discard_mlp.pt"
        model_meta["pegging_model_file"] = "pegging_mlp.pt"
    elif model_type == "gbt":
        model_meta["discard_model_file"] = "discard_gbt.pkl"
        model_meta["pegging_model_file"] = "pegging_gbt.pkl"
    elif model_type == "rf":
        model_meta["discard_model_file"] = "discard_rf.pkl"
        model_meta["pegging_model_file"] = "pegging_rf.pkl"
    meta_path = models_dir / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)
    logger.info(f"Saved model metadata -> {meta_path}")

    # Also write a human-readable summary.
    txt_path = models_dir / "model_meta.txt"
    lines = [
        f"trained_at_utc: {model_meta['trained_at_utc']}",
        f"model_version: {model_meta['model_version']}",
        f"run_id: {model_meta['run_id']}",
        f"data_dir: {model_meta['data_dir']}",
        f"models_dir: {model_meta['models_dir']}",
        f"incremental: {model_meta['incremental']}",
        f"incremental_from: {model_meta['incremental_from']}",
        f"incremental_start_shard: {model_meta['incremental_start_shard']}",
        f"incremental_epochs: {model_meta['incremental_epochs']}",
        f"model_type: {model_meta['model_type']}",
        f"discard_loss: {model_meta['discard_loss']}",
        f"discard_feature_set: {model_meta['discard_feature_set']}",
        f"pegging_feature_set: {model_meta['pegging_feature_set']}",
        f"discard_feature_dim: {model_meta['discard_feature_dim']}",
        f"pegging_feature_dim: {model_meta['pegging_feature_dim']}",
        f"mlp_hidden: {model_meta['mlp_hidden']}",
        f"epochs: {model_meta['epochs']}",
        f"lr: {model_meta['lr']}",
        f"batch_size: {model_meta['batch_size']}",
        f"l2: {model_meta['l2']}",
        f"seed: {model_meta['seed']}",
        f"max_shards: {model_meta['max_shards']}",
        f"num_shards_used: {model_meta['num_shards_used']}",
        f"discard_games_used: {model_meta['discard_games_used']}",
        f"pegging_games_used: {model_meta['pegging_games_used']}",
        f"training_games_used: {model_meta['training_games_used']}",
        f"last_discard_loss: {model_meta['last_discard_loss']}",
        f"last_pegging_loss: {model_meta['last_pegging_loss']}",
        f"discard_model_file: {model_meta['discard_model_file']}",
        f"pegging_model_file: {model_meta['pegging_model_file']}",
        "eval_metrics:",
    ]
    for k, v in model_meta["eval_metrics"].items():
        lines.append(f"  - {k}: {v}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved model summary -> {txt_path}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--pegging_data_dir", type=str, default=None, help="Optional dataset dir for pegging shards.")
    ap.add_argument("--extra_data_dir", type=str, default=None, help="Optional extra dataset to mix in (e.g., self-play).")
    ap.add_argument("--extra_ratio", type=float, default=0.0, help="Fraction of batches to sample from extra_data_dir.")
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--run_id", type=str, default=DEFAULT_MODEL_RUN_ID or None, help="Run id folder (e.g., 001). Omit to auto-increment.")
    ap.add_argument("--discard_loss", type=str, default=DEFAULT_DISCARD_LOSS, choices=["classification", "regression", "ranking"])
    ap.add_argument(
        "--discard_feature_set",
        type=str,
        default=DEFAULT_DISCARD_FEATURE_SET,
        choices=["base", "engineered_no_scores", "engineered_no_scores_pev", "full", "full_pev"],
    )
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["linear", "mlp", "gbt", "rf"])
    ap.add_argument("--mlp_hidden", type=str, default=DEFAULT_MLP_HIDDEN, help="Comma-separated hidden sizes, e.g. 128,64")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--torch_threads",
        type=int,
        default=8,
        help="Torch CPU thread count (intra/inter-op).",
    )
    ap.add_argument(
        "--parallel_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train discard and pegging heads in parallel.",
    )
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For GBT/RF only: cap total training samples to control memory.",
    )
    ap.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train by loading an existing model and continuing on later shards.",
    )
    ap.add_argument(
        "--incremental_from",
        type=str,
        default=None,
        help="Model directory to load when using --incremental.",
    )
    ap.add_argument(
        "--incremental_start_shard",
        type=int,
        default=0,
        help="0-based shard index to start training from when using --incremental.",
    )
    ap.add_argument(
        "--incremental_epochs",
        type=int,
        default=None,
        help="Epochs per shard for --incremental (defaults to --epochs).",
    )
    args = ap.parse_args()
    args.models_dir = _resolve_models_dir(args.models_dir, args.model_version, args.run_id)
    train_models(args)

# python .\scripts\train_models.py
# .\.venv\Scripts\python.exe .\scripts\train_models.py --data_dir "datasets/discard_v3" --models_dir "models" --model_version "discard_v3" --discard_loss regression --epochs 5 --eval_samples 2048 --lr 0.00005 --batch_size 2048 --l2 0.001

# Script summary: train discard/pegging value models from IL datasets and write model artifacts.
