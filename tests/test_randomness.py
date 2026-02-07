import argparse
from pathlib import Path

from scripts.benchmark_2_players import benchmark_2_players


def _run_benchmark(data_dir: str, models_dir: str) -> None:
    args = argparse.Namespace(
        players="beginner,medium",
        benchmark_games=2,
        benchmark_workers=1,
        max_buffer_games=500,
        models_dir=models_dir,
        model_version="seed_test",
        model_run_id=None,
        latest_model=False,
        data_dir=data_dir,
        max_shards=None,
        seed=67,
        fallback_player="beginner",
        model_tag="seed_test",
        discard_feature_set="full",
        pegging_feature_set="full",
        auto_mixed_benchmarks=False,
        games=2,
        no_benchmark_write=True,
    )
    benchmark_2_players(args)


def test_benchmark_seed_stability(capsys, tmp_path: Path) -> None:
    # Run the same 2-game beginner vs medium benchmark twice with a fixed seed.
    # We assert stdout/stderr match exactly to catch RNG sources that aren't
    # being seeded or reset consistently.
    data_dir = str(tmp_path / "datasets")
    models_dir = str(tmp_path / "models")

    _run_benchmark(data_dir, models_dir)
    first = capsys.readouterr()

    _run_benchmark(data_dir, models_dir)
    second = capsys.readouterr()

    assert first.out == second.out
    assert first.err == second.err
