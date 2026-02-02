#  python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20
# python scripts/benchmark_2_players.py --players neural,random --games 500 --models_dir models
import sys
import subprocess
sys.path.insert(0, ".")
from crib_ai_trainer.constants import MODELS_DIR, TRAINING_DATA_DIR
from scripts.benchmark_2_players import benchmark_2_players
from scripts.train_linear_models import train_linear_models

import argparse

from scripts.generate_il_data import generate_il_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--training_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--strategy", type=str, default="classification")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--players", type=str, default="neural,beginner")
    ap.add_argument("--benchmark_games", type=int, default=500)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)    
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=8192)
    args = ap.parse_args()
    generate_il_data(args.games, args.training_dir, args.seed, args.strategy) 
    train_linear_models(args)
    benchmark_2_players(args)
    args.players = "NeuralClassificationPlayer,beginner"
    benchmark_2_players(args)

