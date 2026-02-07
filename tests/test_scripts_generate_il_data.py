from __future__ import annotations

import runpy
import sys
from pathlib import Path
import numpy as np
import logging

from scripts.generate_il_data import generate_il_data

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
    generate_il_data(50, str(out_dir), 0)    

    discard = out_dir / "discard_00001.npz"
    pegging = out_dir / "pegging_00001.npz"
    assert discard.exists()
    assert pegging.exists()

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