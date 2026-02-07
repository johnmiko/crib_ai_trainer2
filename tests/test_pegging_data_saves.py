from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

import scripts.generate_il_data as g


def test_pegging_data_saves(tmp_path: Path) -> None:
    out_dir = tmp_path / "pegging_only"
    captured: Dict[str, Any] = {}
    original_save = g.save_data

    def _capture_save(*args, **kwargs):  # type: ignore[no-untyped-def]
        log = args[0]
        captured["X_pegging"] = list(getattr(log, "X_pegging", []))
        captured["y_pegging"] = list(getattr(log, "y_pegging", []))
        return original_save(*args, **kwargs)

    g.save_data = _capture_save  # type: ignore[assignment]
    try:
        g.generate_il_data(
            games=1,
            out_dir=str(out_dir),
            seed=123,
            strategy="regression",
            pegging_feature_set="full_seq",
            crib_ev_mode="mc",
            crib_mc_samples=4,
            pegging_label_mode="immediate",
            pegging_rollouts=1,
            win_prob_mode="off",
            win_prob_rollouts=1,
            win_prob_min_score=90,
            pegging_ev_mode="off",
            pegging_ev_rollouts=1,
            workers=1,
            save_pegging=True,
            save_discard=False,
            max_buffer_games=1,
            teacher_player="hard",
        )
    finally:
        g.save_data = original_save  # type: ignore[assignment]

    saved = list(Path(out_dir).glob("pegging_*.npz"))
    assert saved, "Expected a pegging shard to be written."
    data = np.load(saved[0])
    X_saved = data["X"]
    y_saved = data["y"]

    X_logged = np.stack(captured.get("X_pegging", [])).astype(np.float32)
    y_logged = np.array(captured.get("y_pegging", []), dtype=np.float32)

    assert X_saved.shape == X_logged.shape
    assert y_saved.shape == y_logged.shape
    np.testing.assert_allclose(X_saved, X_logged, rtol=0, atol=0)
    np.testing.assert_allclose(y_saved, y_logged, rtol=0, atol=0)
