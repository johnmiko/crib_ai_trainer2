from __future__ import annotations

import runpy
import sys
from pathlib import Path
import numpy as np
import logging

from scripts.generate_il_data import generate_il_data
from scripts.generate_il_data import _load_db_stats_full
from cribbage.constants import HAND_CRIB_DB_PATH
import sqlite3

logger = logging.getLogger(__name__)


def run_script(path: Path, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(path)] + argv
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv


def test_generate_il_data_creates_npz_files(tmp_path: Path) -> None:
    logger.info(tmp_path)
    out_dir = tmp_path / "datasets"    

    # small + deterministic
    generate_il_data(50, str(out_dir), 0, "regression", "full", "mc", 32, "immediate", 32, "off", 16, 90, "off", 16, 1, True, True, 500, "hard")

    discard_files = sorted(out_dir.glob("discard_*.npz"))
    pegging_files = sorted(out_dir.glob("pegging_*.npz"))
    assert discard_files, "No discard_*.npz files created"
    assert pegging_files, "No pegging_*.npz files created"
    discard = discard_files[0]
    pegging = pegging_files[0]

    with np.load(discard) as d:
        assert "X" in d and "y" in d
        X = d["X"]
        y = d["y"]
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.dtype in (np.float32, np.float64)

    with np.load(pegging) as p:
        assert "X" in p and "y" in p
        X = p["X"]
        y = p["y"]
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.dtype in (np.float32, np.float64)

# def test_mc():
#         y = estimate_discard_value_mc_fast_from_remaining(
#         kept=kept,
#         discards=discards,
#         dealer_is_self=dealer_is_self,
#         remaining=remaining,
#         rng=self._rng,
#         n_starters=16,
#         n_opp_discards=8,
#     )
def test_specific_hand_discard_is_correct():
    hand = ["ks", "jc", "7s", "8d", "10s", "kh"]
    discards = ["ks","7s"]
    kept = ["jc","8d","10s","kh"]
    # score_hand(kept)


def test_load_db_stats_full_schema_and_values() -> None:
    if not HAND_CRIB_DB_PATH or not Path(HAND_CRIB_DB_PATH).exists():
        import pytest

        pytest.skip("HAND_CRIB_DB_PATH not set or DB missing.")

    conn = sqlite3.connect(HAND_CRIB_DB_PATH)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(hand1)")
    hand_cols = {row[1] for row in cur.fetchall()}
    required_hand_cols = {"min_score", "max_score", "avg_score"}
    assert required_hand_cols.issubset(hand_cols), (
        "hand1 table missing required columns: "
        f"{sorted(required_hand_cols - hand_cols)}"
    )

    cur.execute("PRAGMA table_info(crib1)")
    crib_cols = {row[1] for row in cur.fetchall()}
    required_crib_cols = {"min_score", "avg_score"}
    assert required_crib_cols.issubset(crib_cols), (
        "crib1 table missing required columns: "
        f"{sorted(required_crib_cols - crib_cols)}"
    )

    conn.close()

    hand_stats, crib_stats = _load_db_stats_full()
    assert hand_stats, "hand_stats should not be empty"
    assert crib_stats, "crib_stats should not be empty"

    hand_vals = next(iter(hand_stats.values()))
    assert len(hand_vals) == 3
    assert all(isinstance(v, float) for v in hand_vals)

    crib_vals = next(iter(crib_stats.values()))
    assert len(crib_vals) == 2
    assert all(isinstance(v, float) for v in crib_vals)
