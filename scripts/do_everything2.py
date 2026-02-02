#  python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20
# python scripts/benchmark_2_players.py --players neural,random --games 500 --models_dir models
import sys
import subprocess

from crib_ai_trainer.constants import MODELS_DIR, TRAINING_DATA_DIR
from scripts.benchmark_2_players import benchmark_2_players
from scripts.train_linear_models import train_linear_models
sys.path.insert(0, ".")
import argparse

from scripts.generate_il_data import generate_il_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--training_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--players", type=str, default="neural,medium")
    ap.add_argument("--benchmark_games", type=int, default=500)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    args = ap.parse_args()
    generate_il_data(args.games, args.training_dir, args.seed)    
    train_linear_models(args)
    benchmark_2_players(args)
    args.players = "neural,reasonable"
    benchmark_2_players(args)

