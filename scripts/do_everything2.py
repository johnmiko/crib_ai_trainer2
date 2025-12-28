#  python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20
# python scripts/benchmark_2_players.py --players neural,random --games 500 --models_dir models
import sys
import subprocess
sys.path.insert(0, ".")
import argparse

from scripts.generate_il_data import generate_il_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out_dir", type=str, default="C:\\Users\\johnm\\ccode\\crib_ai_trainer2\\il_datasets\\")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate_il_data(args.games, "C:\\Users\\johnm\\ccode\\crib_ai_trainer2\\il_datasets\\", args.seed)
    subprocess.run([
    sys.executable,
    "python",
    "scripts/train_linear_models.py",
    "--data_dir", "C:\\Users\\johnm\\ccode\\crib_ai_trainer2\\il_datasets\\",
    "--out_dir", "C:\\Users\\johnm\\ccode\\crib_ai_trainer2\\models\\",
    "--epochs", "20",
    ])    
    subprocess.run([
    sys.executable,
    "scripts/benchmark_2_players.py",
    "--players", "neural,random",
    "--games", "500",
    "--models_dir", "C:\\Users\\johnm\\ccode\\crib_ai_trainer2\\models\\",
    ])
    subprocess.run([
    sys.executable,
    "scripts/benchmark_2_players.py",
    "--players", "neural,reasonable",
    "--games", "500",
    "--models_dir", "C:\\Users\\johnm\\ccode\\crib_ai_trainer2\\models\\",
    ])

