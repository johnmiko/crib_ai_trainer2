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
    _overall_t0 = time.perf_counter()
    _overall_start = datetime.now()
    args = build_do_everything_parser().parse_args()

    if args.smoke:
        args.loops = 1
        args.il_games = 1
        args.benchmark_games = 1
        args.il_workers = 1
        args.il_games_per_worker = 1
        args.benchmark_workers = 1
        args.benchmark_games_per_worker = 1
        args.no_benchmark_write = True

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

    def _log_step_start(step_name: str, start_ts: datetime) -> None:
        print(f"{step_name} start: {start_ts.isoformat(timespec='seconds')}", flush=True)

    def _log_step_end(step_name: str, end_ts: datetime, elapsed_s: float) -> None:
        print(f"{step_name} end:   {end_ts.isoformat(timespec='seconds')}", flush=True)
        print(f"{step_name} elapsed: {_format_elapsed(elapsed_s)}", flush=True)

    i = 0
    while True:
        i += 1
        print(f"\n=== Loop {i}/{args.loops} ===")
        _loop_t0 = time.perf_counter()
        _loop_start = datetime.now()

        print(f"dataset_dir: {dataset_dir}", flush=True)
        print(f"next_model_version: {args.model_version}", flush=True)
        print("step: generate_il_data", flush=True)
        _t0 = time.perf_counter()
        _start = datetime.now()
        _log_step_start("generate_il_data", _start)
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
            not args.skip_pegging_data,
        )
        _end = datetime.now()
        _log_step_end("generate_il_data", _end, time.perf_counter() - _t0)

        print("step: train_linear_models", flush=True)
        args.pegging_feature_set = args.pegging_model_feature_set
        args.models_dir = _resolve_models_dir(base_models_dir, args.model_version, args.model_run_id)
        args.pegging_data_dir = args.pegging_data_dir or args.data_dir
        print(f"models_dir: {args.models_dir}", flush=True)
        _t0 = time.perf_counter()
        _start = datetime.now()
        _log_step_start("train_linear_models", _start)
        train_linear_models(args)
        _end = datetime.now()
        _log_step_end("train_linear_models", _end, time.perf_counter() - _t0)

        print("step: benchmark", flush=True)
        args.games = args.benchmark_games
        args.benchmark_games = args.benchmark_games
        args.discard_feature_set = args.discard_feature_set
        args.pegging_feature_set = args.pegging_model_feature_set
        _orig_seed = args.seed
        if _orig_seed is None:
            args.seed = 42
        _t0 = time.perf_counter()
        _start = datetime.now()
        _log_step_start("benchmark", _start)
        benchmark_2_players(args)
        if args.benchmark_mode == "all":
            parts = [p.strip() for p in args.players.split(",") if p.strip()]
            opponent = parts[1] if len(parts) >= 2 else "beginner"
            args.players = f"NeuralDiscardOnlyPlayer,{opponent}"
            benchmark_2_players(args)
            args.players = f"NeuralPegOnlyPlayer,{opponent}"
            benchmark_2_players(args)
        args.pegging_feature_set = data_pegging_feature_set
        args.seed = _orig_seed
        _end = datetime.now()
        _log_step_end("benchmark", _end, time.perf_counter() - _t0)

        _loop_end = datetime.now()
        _loop_elapsed = time.perf_counter() - _loop_t0
        print(f"loop {i} start: {_loop_start.isoformat(timespec='seconds')}", flush=True)
        print(f"loop {i} end:   {_loop_end.isoformat(timespec='seconds')}", flush=True)
        print(f"loop {i} elapsed: {_format_elapsed(_loop_elapsed)}", flush=True)

        if i >= args.loops:
            break

    _overall_end = datetime.now()
    print(f"do_everything start: {_overall_start.isoformat(timespec='seconds')}", flush=True)
    print(f"do_everything end:   {_overall_end.isoformat(timespec='seconds')}", flush=True)
    print(f"do_everything elapsed: {_format_elapsed(time.perf_counter() - _overall_t0)}", flush=True)

# python .\scripts\do_everything2.py
# below does the full feature set
# python .\scripts\do_everything2.py --il_games 2000 --loops 100 --dataset_version "discard_v3" --model_version "discard_v3" --strategy regression --discard_loss regression --benchmark_games 1000 --benchmark_mode full
# this does the engineered no scores feature set
# .\.venv\Scripts\python.exe .\scripts\do_everything2.py --il_games 2000 --loops 100 --dataset_version "discard_v3" --model_version "discard_v3" --strategy regression --discard_loss regression --benchmark_games 1000 --benchmark_mode full --discard_feature_set engineered_no_scores --pegging_model_feature_set full_no_scores
