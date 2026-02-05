"""Train models from sharded datasets.

Usage:
  python scripts/train_models.py --data_dir il_datasets --models_dir models --epochs 20
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path

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


def train_models(args) -> int:
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

    if discard_mode in {"classification", "ranking"} and model_type != "linear":
        raise SystemExit("Only linear models are supported for classification/ranking at the moment.")

    if discard_mode == "classification":
        with np.load(discard_shards[0]) as d0:
            discard_model = LinearDiscardClassifier(int(len(discard_feature_indices)))
    else:
        with np.load(discard_shards[0]) as d0:
            if model_type == "mlp":
                discard_model = MLPValueModel(int(len(discard_feature_indices)), mlp_hidden, seed=args.seed or 0)
            else:
                discard_model = LinearValueModel(int(len(discard_feature_indices)))
    with np.load(pegging_shards[0]) as p0:
        if model_type == "mlp":
            pegging_model = MLPValueModel(int(len(pegging_feature_indices)), mlp_hidden, seed=args.seed or 0)
        else:
            pegging_model = LinearValueModel(int(len(pegging_feature_indices)))

    last_discard_loss = None
    last_pegging_loss = None
    eval_metrics = {}

    rng = np.random.default_rng(args.seed)
    extra_ratio = float(args.extra_ratio)
    if extra_ratio < 0.0 or extra_ratio > 1.0:
        raise SystemExit("--extra_ratio must be between 0 and 1.")

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
            if discard_mode == "classification":
                logger.debug(f"Training discard model {discard_model}")
                discard_losses = discard_model.fit_ce( # type: ignore
                Xd, yd, lr=args.lr, epochs=args.epochs,
                batch_size=args.batch_size, l2=args.l2, seed=args.seed
            )
            elif discard_mode == "ranking":
                discard_losses = discard_model.fit_rank_pairwise( # type: ignore
                    Xd, yd,
                    lr=args.lr,
                    epochs=1,
                    batch_size=args.batch_size,
                    l2=args.l2,
                    seed=args.seed,
                    pairs_per_hand=args.rank_pairs_per_hand,
                )
            else:
                discard_losses = discard_model.fit_mse( # type: ignore
                    Xd, yd,
                    lr=args.lr,
                    epochs=1,
                    batch_size=args.batch_size,
                    l2=args.l2,
                    seed=args.seed,
                )
            pegging_losses = pegging_model.fit_mse(
                Xp, yp,
                lr=args.lr,
                epochs=1,
                batch_size=args.batch_size,
                l2=args.l2,
                seed=args.seed,
            )

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
        "last_discard_loss": last_discard_loss,
        "last_pegging_loss": last_pegging_loss,
        "eval_metrics": eval_metrics,
        "discard_model_file": "discard_linear.npz",
        "pegging_model_file": "pegging_linear.npz",
    }
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
    ap.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["linear", "mlp"])
    ap.add_argument("--mlp_hidden", type=str, default=DEFAULT_MLP_HIDDEN, help="Comma-separated hidden sizes, e.g. 128,64")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    args = ap.parse_args()
    args.models_dir = _resolve_models_dir(args.models_dir, args.model_version, args.run_id)
    train_models(args)

# python .\scripts\train_models.py
# .\.venv\Scripts\python.exe .\scripts\train_models.py --data_dir "il_datasets/discard_v3/001" --models_dir "models" --model_version "discard_v3" --discard_loss regression --epochs 5 --eval_samples 2048 --lr 0.00005 --batch_size 2048 --l2 0.001
