from __future__ import annotations
from logging import getLogger
from .training.trainer import Trainer, TrainConfig

logger = getLogger(__name__)

def main():
    cfg = TrainConfig(num_training_games=5, benchmark_games=25, run_indefinitely=False)
    t = Trainer(cfg)
    t._rank_models(cfg.benchmark_games)

if __name__ == "__main__":
    main()
