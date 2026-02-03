# python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20
# python .\scripts\benchmark_2_players.py --players neural,random --games 500 --models_dir models
import sys

sys.path.insert(0, ".")

import argparse
from pathlib import Path

from crib_ai_trainer.constants import (
    MODELS_DIR,
    TRAINING_DATA_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_DATASET_RUN_ID,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODEL_RUN_ID,
    DEFAULT_STRATEGY,
    DEFAULT_DISCARD_LOSS,
    DEFAULT_PEGGING_FEATURE_SET,
    DEFAULT_DISCARD_FEATURE_SET,
    DEFAULT_PEGGING_MODEL_FEATURE_SET,
    DEFAULT_GAMES_PER_LOOP,
    DEFAULT_LOOPS,
    DEFAULT_EPOCHS,
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_LR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_L2,
    DEFAULT_MAX_SHARDS,
    DEFAULT_FALLBACK_PLAYER,
    DEFAULT_RANK_PAIRS_PER_HAND,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_MODEL_TAG,
    DEFAULT_SEED,
    DEFAULT_USE_RANDOM_SEED,
)
from scripts.benchmark_2_players import benchmark_2_players
from scripts.generate_il_data import generate_il_data, _resolve_output_dir
from scripts.train_linear_models import train_linear_models, _resolve_models_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--il_games", type=int, default=DEFAULT_GAMES_PER_LOOP, help="Games per loop for IL data generation")
    ap.add_argument(
        "--loops",
        type=int,
        default=DEFAULT_LOOPS,
        help="Number of generate->train->benchmark cycles. Use -1 to loop forever.",
    )
    ap.add_argument("--training_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--dataset_run_id", type=str, default=DEFAULT_DATASET_RUN_ID or None)
    ap.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    ap.add_argument(
        "--pegging_feature_set",
        type=str,
        default=DEFAULT_PEGGING_FEATURE_SET,
        choices=["basic", "full"],
    )
    default_seed = None if DEFAULT_USE_RANDOM_SEED else DEFAULT_SEED
    ap.add_argument("--seed", type=int, default=default_seed)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--players", type=str, default="NeuralRegressionPlayer,beginner")
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument(
        "--benchmark_mode",
        type=str,
        default="all",
        choices=["all", "full"],
        help="Benchmark all three (full/discard/pegging) or only the full model.",
    )
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--model_run_id", type=str, default=DEFAULT_MODEL_RUN_ID or None)
    ap.add_argument("--discard_loss", type=str, default=DEFAULT_DISCARD_LOSS, choices=["classification", "regression", "ranking"])
    ap.add_argument("--discard_feature_set", type=str, default=DEFAULT_DISCARD_FEATURE_SET, choices=["base", "engineered_no_scores", "full"])
    ap.add_argument("--pegging_model_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--fallback_player", type=str, default=DEFAULT_FALLBACK_PLAYER)
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--model_tag", type=str, default=DEFAULT_MODEL_TAG or None)
    args = ap.parse_args()

    if args.data_dir is None:
        args.data_dir = args.training_dir
    if args.loops == 0 or args.loops < -1:
        raise SystemExit("--loops must be >= 1 or -1 for infinite")

    base_models_dir = args.models_dir
    # Resolve dataset directory. If training_dir points to a run folder but a different
    # dataset_version is provided, prefer the versioned path.
    training_path = Path(args.training_dir)
    has_shards = bool(list(training_path.glob("discard_*.npz"))) or bool(
        list(training_path.glob("pegging_*.npz"))
    )
    if has_shards and (args.dataset_run_id is not None):
        dataset_dir = str(training_path)
    else:
        base_out_dir = args.training_dir
        if has_shards and args.dataset_version not in training_path.parts:
            # training_dir is a run folder (e.g., discard_v2/001). Use its parent parent as base.
            base_out_dir = str(training_path.parent.parent)
        dataset_dir = _resolve_output_dir(
            base_out_dir,
            args.dataset_version,
            args.dataset_run_id,
            new_run=False,
        )
    args.data_dir = dataset_dir

    data_pegging_feature_set = args.pegging_feature_set

    i = 0
    while True:
        i += 1
        if args.loops == -1:
            print(f"\n=== Loop {i}/infinity ===")
        else:
            print(f"\n=== Loop {i}/{args.loops} ===")

        print(f"dataset_dir: {dataset_dir}", flush=True)
        print(f"next_model_version: {args.model_version}", flush=True)
        print("step: generate_il_data", flush=True)
        generate_il_data(
            args.il_games,
            dataset_dir,
            args.seed,
            args.strategy,
            data_pegging_feature_set,
        )

        print("step: train_linear_models", flush=True)
        args.pegging_feature_set = args.pegging_model_feature_set
        args.models_dir = _resolve_models_dir(base_models_dir, args.model_version, args.model_run_id)
        print(f"models_dir: {args.models_dir}", flush=True)
        train_linear_models(args)

        print("step: benchmark", flush=True)
        args.games = args.benchmark_games
        args.benchmark_games = args.benchmark_games
        args.discard_feature_set = args.discard_feature_set
        args.pegging_feature_set = args.pegging_model_feature_set
        args.players = "NeuralRegressionPlayer,beginner"
        benchmark_2_players(args)
        if args.benchmark_mode == "all":
            args.players = "NeuralDiscardOnlyPlayer,beginner"
            benchmark_2_players(args)
            args.players = "NeuralPegOnlyPlayer,beginner"
            benchmark_2_players(args)
        args.pegging_feature_set = data_pegging_feature_set

        if args.loops != -1 and i >= args.loops:
            break

# python .\scripts\do_everything2.py
# below does the full feature set
# python .\scripts\do_everything2.py --il_games 2000 --loops -1 --dataset_version "discard_v3" --model_version "discard_v3" --strategy regression --discard_loss regression --benchmark_games 1000 --benchmark_mode full
# this does the engineered no scores feature set
# .\.venv\Scripts\python.exe .\scripts\do_everything2.py --il_games 2000 --loops -1 --dataset_version "discard_v3" --model_version "discard_v3" --strategy regression --discard_loss regression --benchmark_games 1000 --benchmark_mode full --discard_feature_set engineered_no_scores --pegging_model_feature_set full_no_scores

