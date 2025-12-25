import os
import numpy as np
from logging import getLogger
from crib_ai_trainer2.training.trainer import Trainer, TrainConfig

logger = getLogger(__name__)

def export_best_model_npz(out_path: str) -> None:
    cfg = TrainConfig(num_training_games=1, benchmark_games=5)
    t = Trainer(cfg)
    t._rank_models(cfg.benchmark_games)
    best = t.models.get(t.best_model_name)
    # Example: export perceptron if best
    if hasattr(best, "export_npz"):
        best.export_npz(out_path)  # type: ignore
        logger.info(f"Exported model to {out_path}")
    else:
        # export a dummy linear policy weights
        w = np.zeros((1,), dtype=np.float32)
        b = np.zeros((1,), dtype=np.float32)
        np.savez(out_path, w=w, b=b)
        logger.info(f"Exported placeholder model to {out_path}")
