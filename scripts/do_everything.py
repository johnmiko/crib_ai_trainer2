
import os
import sys
from logging import getLogger
# Ensure project root is on sys.path for absolute imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from crib_ai_trainer2.training.trainer import Trainer, TrainConfig

logger = getLogger(__name__)

def main():
    cfg = TrainConfig(num_training_games=10, benchmark_games=50, run_indefinitely=False)
    t = Trainer(cfg)
    t.train()

if __name__ == "__main__":
    main()
