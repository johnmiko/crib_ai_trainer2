"""Train LinearValueModel models from sharded datasets.

Usage:
  python scripts/train_linear_models.py --data_dir il_datasets --models_dir models --epochs 20
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import numpy as np
import json
from datetime import datetime, timezone

sys.path.insert(0, ".")
from crib_ai_trainer.constants import MODELS_DIR, TRAINING_DATA_DIR
from crib_ai_trainer.players.neural_player import LinearDiscardClassifier, LinearValueModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_linear_models(args) -> int:
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading training data from {data_dir}")
    discard_shards = sorted(data_dir.glob("discard_*.npz"))
    pegging_shards = sorted(data_dir.glob("pegging_*.npz"))
    if not discard_shards:
        raise SystemExit(f"No discard shards in {data_dir} (expected discard_*.npz)")
    if not pegging_shards:
        raise SystemExit(f"No pegging shards in {data_dir} (expected pegging_*.npz)")
    if len(discard_shards) != len(pegging_shards):
        raise SystemExit(
            f"Shard count mismatch: {len(discard_shards)} discard vs {len(pegging_shards)} pegging"
        )
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

    if discard_mode == "classification":
        with np.load(discard_shards[0]) as d0:
            discard_model = LinearDiscardClassifier(int(d0["X"].shape[2]))
    else:
        with np.load(discard_shards[0]) as d0:
            if d0["X"].ndim == 3:
                discard_model = LinearValueModel(int(d0["X"].shape[2]))
            else:
                discard_model = LinearValueModel(int(d0["X"].shape[1]))
    with np.load(pegging_shards[0]) as p0:
        pegging_model = LinearValueModel(int(p0["X"].shape[1]))

    last_discard_loss = None
    last_pegging_loss = None
    eval_metrics = {}

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for d_path, p_path in zip(discard_shards, pegging_shards):
            logger.debug(f"  Training on shard {d_path.name} and {p_path.name}")
            with np.load(d_path) as d:
                if discard_mode == "classification":
                    Xd = d["X"].astype(np.int64)
                    yd = d["y"].astype(np.int64)
                else:
                    Xd = d["X"].astype(np.float32)
                    yd = d["y"].astype(np.float32)
            # print("discard X dim", Xd.shape, "y range", float(yd.min()), float(yd.mean()), float(yd.max()))

            with np.load(p_path) as p:
                Xp = p["X"].astype(np.float32)
                yp = p["y"].astype(np.float32)
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
            yd_eval = yd[:n_d].astype(np.int64)
            scores = np.tensordot(Xd_eval, discard_model.w, axes=([2], [0])) + discard_model.b
            preds = np.argmax(scores, axis=1)
            acc = float(np.mean(preds == yd_eval)) if n_d > 0 else 0.0
            print(f"  discard classifier top-1 acc: {acc:.3f} on {n_d} samples")
            eval_metrics["discard_classifier_top1_acc"] = acc
        elif discard_mode == "ranking":
            Xd_eval = Xd[:n_d].astype(np.float32)
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
            yd_eval = yd[:n_d].astype(np.float32)
            pred = discard_model.predict_batch(Xd_eval)
            mse = float(np.mean((pred - yd_eval) ** 2)) if n_d > 0 else 0.0
            print(f"  discard regressor MSE: {mse:.4f} on {n_d} samples")
            eval_metrics["discard_regressor_mse"] = mse

        Xp_eval = Xp[:n_p].astype(np.float32)
        yp_eval = yp[:n_p].astype(np.float32)
        pred_p = pegging_model.predict_batch(Xp_eval)
        mse_p = float(np.mean((pred_p - yp_eval) ** 2)) if n_p > 0 else 0.0
        print(f"  pegging regressor MSE: {mse_p:.4f} on {n_p} samples")
        eval_metrics["pegging_regressor_mse"] = mse_p

    # Write model metadata for easy inspection.
    model_meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "models_dir": str(models_dir),
        "discard_loss": discard_mode,
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

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--discard_loss", type=str, default=None, choices=["classification", "regression", "ranking"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_samples", type=int, default=2048)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--rank_pairs_per_hand", type=int, default=20)
    args = ap.parse_args()
    train_linear_models(args)

# python .\scripts\train_linear_models.py
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --models_dir models --epochs 20
# train over the whole dataset 20 times
