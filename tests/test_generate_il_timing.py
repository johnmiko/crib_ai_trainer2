import json
import time
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from scripts.generate_il_data import generate_il_data


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
