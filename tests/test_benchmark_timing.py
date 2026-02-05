import json
import time
from pathlib import Path

from scripts.benchmark_2_players import benchmark_2_players
from utils import build_benchmark_parser


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_benchmark_timing() -> float:
    args = build_benchmark_parser().parse_args([])
    args.players = "beginner,medium"
    args.benchmark_games = 20
    args.benchmark_workers = 1
    args.max_buffer_games = 500
    args.no_benchmark_write = True

    start = time.perf_counter()
    benchmark_2_players(args)
    return time.perf_counter() - start


def test_benchmark_2_players_timing():
    elapsed = _run_benchmark_timing()

    base = Path(__file__).parent
    baseline_path = base / "benchmark_timing_baseline.json"
    latest_path = base / "benchmark_timing_latest.json"

    payload = {"seconds": float(elapsed)}
    _write_json(latest_path, payload)

    if not baseline_path.exists():
        raise AssertionError(f"Baseline missing. Latest seconds={elapsed:.6f}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_seconds = float(baseline["seconds"])
    assert elapsed <= baseline_seconds
