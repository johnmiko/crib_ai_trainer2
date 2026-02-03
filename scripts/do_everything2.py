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
    ap.add_argument("--games", type=int, default=2000, help="Games per loop")
    ap.add_argument("--loops", type=int, default=1, help="Number of generate→train→benchmark cycles. Use -1 to loop forever.")
    ap.add_argument("--training_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default="discard_v1")
    ap.add_argument("--dataset_run_id", type=str, default=None)
    ap.add_argument("--strategy", type=str, default="classification")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--players", type=str, default="NeuralClassificationPlayer,beginner")
    ap.add_argument("--benchmark_games", type=int, default=500)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default="discard_v1")
    ap.add_argument("--model_run_id", type=str, default=None)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=8192)
    args = ap.parse_args()
    if args.data_dir is None:
        args.data_dir = args.training_dir
    if args.loops == 0 or args.loops < -1:
        raise SystemExit("--loops must be >= 1 or -1 for infinite")
    i = 0
    while True:
        i += 1
        if args.loops == -1:
            print(f"\n=== Loop {i}/∞ ===")
        else:
            print(f"\n=== Loop {i}/{args.loops} ===")
        # Resolve dataset run folder
        from scripts.generate_il_data import _resolve_output_dir
        dataset_dir = _resolve_output_dir(args.training_dir, args.dataset_version, args.dataset_run_id)
        generate_il_data(args.games, dataset_dir, args.seed, args.strategy)

        # Resolve model run folder
        from scripts.train_linear_models import _resolve_models_dir
        args.data_dir = dataset_dir
        args.models_dir = _resolve_models_dir(args.models_dir, args.model_version, args.model_run_id)
        train_linear_models(args)
        benchmark_2_players(args)
        args.players = "NeuralClassificationPlayer,beginner"
        benchmark_2_players(args)
        if args.loops != -1 and i >= args.loops:
            break

# python .\scripts\do_everything2.py
# python .\scripts\do_everything2.py --games 2000 --loops -1 --strategy ranking --training_dir "il_datasets/medium_discard_ranking" --data_dir "il_datasets/medium_discard_ranking" --models_dir "models/ranking" --players NeuralRegressionPlayer,beginner --benchmark_games 200
