# python .\scripts\generate_il_data.py --games 2000 --out_dir "il_datasets/"
# python .\scripts\train_linear_models.py --data_dir "il_datasets/" --out_dir models --epochs 20
# python .\scripts\benchmark_2_players.py --players neural,random --games 500 --models_dir models
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

from pathlib import Path

from scripts.benchmark_2_players import benchmark_2_players
from scripts.generate_il_data import generate_il_data, _resolve_output_dir
from scripts.train_linear_models import train_linear_models, _resolve_models_dir
from utils import build_do_everything_parser


if __name__ == "__main__":
    args = build_do_everything_parser().parse_args()

    if args.data_dir is None:
        args.data_dir = args.training_dir
    if args.loops <= 0:
        raise SystemExit("--loops must be >= 1")

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

    def _format_elapsed(seconds: float) -> str:
        total = int(round(seconds))
        minutes = total // 60
        secs = total % 60
        return f"{minutes}m {secs}s"

    def _log_step_timing(step_name: str, start_ts: datetime, end_ts: datetime, elapsed_s: float) -> None:
        print(f"{step_name} start: {start_ts.isoformat(timespec='seconds')}", flush=True)
        print(f"{step_name} end:   {end_ts.isoformat(timespec='seconds')}", flush=True)
        print(f"{step_name} elapsed: {_format_elapsed(elapsed_s)}", flush=True)

    i = 0
    while True:
        i += 1
        print(f"\n=== Loop {i}/{args.loops} ===")

        print(f"dataset_dir: {dataset_dir}", flush=True)
        print(f"next_model_version: {args.model_version}", flush=True)
        print("step: generate_il_data", flush=True)
        _t0 = time.perf_counter()
        _start = datetime.now()
        generate_il_data(
            args.il_games,
            dataset_dir,
            args.seed,
            args.strategy,
            data_pegging_feature_set,
            args.crib_ev_mode,
            args.crib_mc_samples,
            args.pegging_label_mode,
            args.pegging_rollouts,
            args.pegging_ev_mode,
            args.pegging_ev_rollouts,
            args.win_prob_mode,
            args.win_prob_rollouts,
            args.win_prob_min_score,
            args.il_workers,
            args.il_games_per_worker,
        )
        _end = datetime.now()
        _log_step_timing("generate_il_data", _start, _end, time.perf_counter() - _t0)

        print("step: train_linear_models", flush=True)
        args.pegging_feature_set = args.pegging_model_feature_set
        args.models_dir = _resolve_models_dir(base_models_dir, args.model_version, args.model_run_id)
        print(f"models_dir: {args.models_dir}", flush=True)
        _t0 = time.perf_counter()
        _start = datetime.now()
        train_linear_models(args)
        _end = datetime.now()
        _log_step_timing("train_linear_models", _start, _end, time.perf_counter() - _t0)

        print("step: benchmark", flush=True)
        args.games = args.benchmark_games
        args.benchmark_games = args.benchmark_games
        args.discard_feature_set = args.discard_feature_set
        args.pegging_feature_set = args.pegging_model_feature_set
        _t0 = time.perf_counter()
        _start = datetime.now()
        benchmark_2_players(args)
        if args.benchmark_mode == "all":
            parts = [p.strip() for p in args.players.split(",") if p.strip()]
            opponent = parts[1] if len(parts) >= 2 else "beginner"
            args.players = f"NeuralDiscardOnlyPlayer,{opponent}"
            benchmark_2_players(args)
            args.players = f"NeuralPegOnlyPlayer,{opponent}"
            benchmark_2_players(args)
        args.pegging_feature_set = data_pegging_feature_set
        _end = datetime.now()
        _log_step_timing("benchmark", _start, _end, time.perf_counter() - _t0)

        if i >= args.loops:
            break

# python .\scripts\do_everything2.py
# below does the full feature set
# python .\scripts\do_everything2.py --il_games 2000 --loops 100 --dataset_version "discard_v3" --model_version "discard_v3" --strategy regression --discard_loss regression --benchmark_games 1000 --benchmark_mode full
# this does the engineered no scores feature set
# .\.venv\Scripts\python.exe .\scripts\do_everything2.py --il_games 2000 --loops 100 --dataset_version "discard_v3" --model_version "discard_v3" --strategy regression --discard_loss regression --benchmark_games 1000 --benchmark_mode full --discard_feature_set engineered_no_scores --pegging_model_feature_set full_no_scores
