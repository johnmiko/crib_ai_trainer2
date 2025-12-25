
import os
import sys
from logging import getLogger
# Ensure project root is on sys.path for absolute imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from crib_ai_trainer2.training.trainer import Trainer, TrainConfig

logger = getLogger(__name__)


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def main():
    logger.info("Initializing training configuration...")
    cfg = TrainConfig(num_training_games=20, benchmark_games=50, run_indefinitely=False)
    logger.info("Creating Trainer instance...")
    t = Trainer(cfg)
    logger.info("Starting training...")
    t.train()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
