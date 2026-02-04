import json
import time
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from scripts.generate_il_data import generate_il_data
from scripts.do_everything2 import benchmark_2_players, generate_il_data as _gen_il, train_linear_models
from scripts.generate_il_data import _resolve_output_dir
from scripts.train_linear_models import _resolve_models_dir
from utils import build_do_everything_parser


def _run_do_everything_smoke(training_dir: str, models_dir: str) -> None:
    args = build_do_everything_parser().parse_args([])
    args.smoke = True
    args.loops = 1
    args.il_games = 1
    args.benchmark_games = 1
    args.il_workers = 1
    args.il_games_per_worker = 1
    args.benchmark_workers = 1
    args.benchmark_games_per_worker = 1
    args.no_benchmark_write = True
    args.training_dir = training_dir
    args.models_dir = models_dir
    args.dataset_version = "discard_smoke"
    args.model_version = "discard_smoke"

    if args.data_dir is None:
        args.data_dir = args.training_dir

    base_models_dir = args.models_dir
    training_path = Path(args.training_dir)
    has_shards = bool(list(training_path.glob("discard_*.npz"))) or bool(
        list(training_path.glob("pegging_*.npz"))
    )
    if has_shards and (args.dataset_run_id is not None):
        dataset_dir = str(training_path)
    else:
        base_out_dir = args.training_dir
        if has_shards and args.dataset_version not in training_path.parts:
            base_out_dir = str(training_path.parent.parent)
        dataset_dir = _resolve_output_dir(
            base_out_dir,
            args.dataset_version,
            args.dataset_run_id,
            new_run=False,
        )
    args.data_dir = dataset_dir
    data_pegging_feature_set = args.pegging_feature_set

    _gen_il(
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

    args.pegging_feature_set = args.pegging_model_feature_set
    args.models_dir = _resolve_models_dir(base_models_dir, args.model_version, args.model_run_id)
    train_linear_models(args)

    args.games = args.benchmark_games
    args.benchmark_games = args.benchmark_games
    args.discard_feature_set = args.discard_feature_set
    args.pegging_feature_set = args.pegging_model_feature_set
    benchmark_2_players(args)
    if args.benchmark_mode == "all":
        parts = [p.strip() for p in args.players.split(",") if p.strip()]
        opponent = parts[1] if len(parts) >= 2 else "beginner"
        args.players = f"NeuralDiscardOnlyPlayer,{opponent}"
        benchmark_2_players(args)
        args.players = f"NeuralPegOnlyPlayer,{opponent}"
        benchmark_2_players(args)


def test_do_everything_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        training_dir = str(Path(tmpdir) / "il_datasets")
        models_dir = str(Path(tmpdir) / "models")
        _run_do_everything_smoke(training_dir, models_dir)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_timing(games: int, label: str) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        start = time.perf_counter()
        generate_il_data(
            games=games,
            out_dir=tmpdir,
            seed=0,
            strategy="regression",
            pegging_feature_set="full",
            crib_ev_mode="min",
            crib_mc_samples=0,
            pegging_label_mode="immediate",
            pegging_rollouts=0,
            win_prob_mode="off",
            win_prob_rollouts=0,
            win_prob_min_score=0,
            pegging_ev_mode="off",
            pegging_ev_rollouts=0,
            workers=1,
            games_per_worker=None,
        )
        elapsed = time.perf_counter() - start

    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {"seconds": float(elapsed), "timestamp_utc": timestamp}

    base = Path(__file__).parent
    baseline_path = base / f"il_timing_baseline_{label}.json"
    latest_path = base / f"il_timing_latest_{label}.json"

    if not baseline_path.exists():
        _write_json(baseline_path, payload)
    _write_json(latest_path, payload)
    return float(elapsed)


def test_generate_il_data_timing_smoke():
    elapsed = _run_timing(1, "1")
    assert elapsed > 0.0


def test_generate_il_data_timing_10_games():
    elapsed = _run_timing(10, "10")
    assert elapsed > 0.0
