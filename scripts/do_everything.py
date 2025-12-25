import os
from logging import getLogger
from crib_ai_trainer2.training.trainer import Trainer, TrainConfig

logger = getLogger(__name__)

def main():
    cfg = TrainConfig(num_training_games=10, benchmark_games=50, run_indefinitely=False)
    t = Trainer(cfg)
    t.train()

if __name__ == "__main__":
    main()
