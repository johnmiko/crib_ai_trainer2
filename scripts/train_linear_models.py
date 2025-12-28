"""Train LinearValueModel models from sharded datasets.

Usage:
  python scripts/train_linear_models.py --data_dir il_datasets --out_dir models --epochs 20
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import numpy as np

sys.path.insert(0, ".")
from crib_ai_trainer.players.neural_player import LinearValueModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # init models
    with np.load(discard_shards[0]) as d0:
        discard_model = LinearValueModel(int(d0["X"].shape[1]))
    with np.load(pegging_shards[0]) as p0:
        pegging_model = LinearValueModel(int(p0["X"].shape[1]))

    last_discard_loss = None
    last_pegging_loss = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for d_path, p_path in zip(discard_shards, pegging_shards):
            with np.load(d_path) as d:
                Xd = d["X"].astype(np.float32)
                yd = d["y"].astype(np.float32)

            with np.load(p_path) as p:
                Xp = p["X"].astype(np.float32)
                yp = p["y"].astype(np.float32)

            discard_losses = discard_model.fit_mse(
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

    discard_path = out_dir / "discard_linear.npz"
    pegging_path = out_dir / "pegging_linear.npz"
    discard_model.save_npz(str(discard_path))
    pegging_model.save_npz(str(pegging_path))
    # “last loss” = the mean squared error (MSE) from the final training step that ran.
    print(f"Saved discard model -> {discard_path}")
    if last_discard_loss is not None:
        print(f"  last loss={last_discard_loss:.6f}")
    print(f"Saved pegging model -> {pegging_path}")
    if last_pegging_loss is not None:
        print(f"  last loss={last_pegging_loss:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20
# train over the whole dataset 20 times