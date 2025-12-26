
import os
import sys
from logging import getLogger
# Ensure project root is on sys.path for absolute imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from crib_ai_trainer.training.trainer import Trainer, TrainConfig

logger = getLogger(__name__)


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def main():
    logger.info("Initializing training configuration...")
    cfg = TrainConfig(num_training_games=100, benchmark_games=100, run_indefinitely=False)
    logger.info("Creating Trainer instance...")
    t = Trainer(cfg)
    # Only train non-neural models
    non_neural = [k for k in t.models if not (k.startswith('nn_') or k == 'neural')]
    t.cfg.include_models = non_neural
    logger.info(f"Training non-neural models: {non_neural}")
    t.train()
    logger.info("Non-neural training finished.")

    # Now run neural network training script
    logger.info("Starting neural network training via train_neural_configs.py...")
    import subprocess
    script_path = os.path.join(ROOT, 'scripts', 'train_neural_configs.py')
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        logger.error(f"train_neural_configs.py failed with exit code {result.returncode}")
    else:
        logger.info("Neural network training finished.")

if __name__ == "__main__":
    main()
