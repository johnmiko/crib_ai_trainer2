"""Self-play loop with frozen-best acceptance rule.

Workflow:
1) Generate self-play data using latest accepted model.
2) Train a new model mixing teacher + self-play data.
3) Benchmark new vs frozen best and vs beginner.
4) Accept only if strict criteria are met.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import (
    TRAINING_DATA_DIR,
    MODELS_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_MODEL_VERSION,
    DEFAULT_GAMES_PER_LOOP,
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_DISCARD_FEATURE_SET,
    DEFAULT_PEGGING_MODEL_FEATURE_SET,
    DEFAULT_MODEL_TYPE,
    DEFAULT_MLP_HIDDEN,
)

from cribbage.utils import play_multiple_games
from cribbage.players.beginner_player import BeginnerPlayer

from scripts.generate_self_play_data import generate_self_play_data, _resolve_models_dir, _resolve_output_dir
from scripts.train_linear_models import train_linear_models
from crib_ai_trainer.players.neural_player import (
    NeuralRegressionPlayer,
    LinearValueModel,
    MLPValueModel,
)


def _load_meta(models_dir: Path) -> dict:
    meta_path = models_dir / "model_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_value_model(models_dir: Path, model_type: str):
    if model_type == "mlp":
        return MLPValueModel.load_pt(str(models_dir / "discard_mlp.pt")), MLPValueModel.load_pt(str(models_dir / "pegging_mlp.pt"))
    return LinearValueModel.load_npz(str(models_dir / "discard_linear.npz")), LinearValueModel.load_npz(str(models_dir / "pegging_linear.npz"))


def _make_player(models_dir: Path, name_override: str | None = None) -> NeuralRegressionPlayer:
    meta = _load_meta(models_dir)
    model_type = meta.get("model_type", "linear")
    discard_feature_set = meta.get("discard_feature_set", "full")
    pegging_feature_set = meta.get("pegging_feature_set", "full")
    discard_model, pegging_model = _load_value_model(models_dir, model_type)
    name = name_override or f"selfplay:{models_dir.name}"
    return NeuralRegressionPlayer(
        discard_model,
        pegging_model,
        name=name,
        discard_feature_set=discard_feature_set,
        pegging_feature_set=pegging_feature_set,
    )


def _evaluate(p0, p1, games: int) -> dict:
    return play_multiple_games(games, p0=p0, p1=p1)


def _get_best_path(best_file: Path, models_dir: Path, model_version: str) -> Path:
    if best_file.exists():
        return Path(best_file.read_text(encoding="utf-8").strip())
    # Always resolve under model_version (avoid picking the base models dir)
    version_dir = models_dir / model_version
    if not version_dir.exists():
        return Path(_resolve_models_dir(str(models_dir), model_version, None))
    run_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return Path(_resolve_models_dir(str(models_dir), model_version, None))
    run_id = max(int(p.name) for p in run_dirs)
    return version_dir / f"{run_id:03d}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--teacher_dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--selfplay_dataset_version", type=str, default="selfplay_v1")
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_LOOP)
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument("--selfplay_ratio", type=float, default=0.3)
    ap.add_argument("--best_file", type=str, default="best_model.txt")
    ap.add_argument("--loops", type=int, default=-1)
    ap.add_argument("--discard_feature_set", type=str, default=DEFAULT_DISCARD_FEATURE_SET, choices=["base", "engineered_no_scores", "full"])
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["linear", "mlp"])
    ap.add_argument("--mlp_hidden", type=str, default=DEFAULT_MLP_HIDDEN)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    best_file = Path(args.best_file)
    teacher_dir = _resolve_output_dir(TRAINING_DATA_DIR, args.teacher_dataset_version, None, new_run=False)
    selfplay_dir = _resolve_output_dir(TRAINING_DATA_DIR, args.selfplay_dataset_version, None, new_run=False)

    i = 0
    while True:
        i += 1
        print(f"\n=== Self-play loop {i} ===")
        best_path = _get_best_path(best_file, models_dir, args.model_version)
        print(f"Frozen best: {best_path}")
        print(f"Best model run: {best_path.name} (version: {best_path.parent.name})")

        # 1) Generate self-play data
        generate_self_play_data(
            args.games,
            selfplay_dir,
            str(best_path),
            str(best_path),  # mirror vs frozen best
            None,
            "regression",
            args.pegging_feature_set,
            "mc",
            32,
            "rollout2",
            32,
        )

        # 2) Train new model mixing teacher + self-play data
        train_args = argparse.Namespace(
            data_dir=teacher_dir,
            extra_data_dir=selfplay_dir,
            extra_ratio=args.selfplay_ratio,
            models_dir=args.models_dir,
            model_version=args.model_version,
            run_id=None,
            discard_loss="regression",
            discard_feature_set=args.discard_feature_set,
            pegging_feature_set=args.pegging_feature_set,
            model_type=args.model_type,
            mlp_hidden=args.mlp_hidden,
            lr=0.00005,
            epochs=5,
            batch_size=2048,
            l2=0.001,
            seed=0,
            eval_samples=2048,
            max_shards=None,
            rank_pairs_per_hand=20,
        )
        train_linear_models(train_args)
        new_path = Path(train_args.models_dir)
        print(f"New candidate: {new_path}")

        # 3) Benchmarks
        best_player = _make_player(best_path, name_override=f"selfplay:best:{best_path.name}")
        new_player = _make_player(new_path, name_override=f"selfplay:new:{new_path.name}")

        print("Benchmark: BEST vs BEGINNER")
        best_vs_beginner = _evaluate(best_player, BeginnerPlayer(name="beginner"), args.benchmark_games)
        print("Benchmark: NEW vs BEGINNER")
        new_vs_beginner = _evaluate(new_player, BeginnerPlayer(name="beginner"), args.benchmark_games)
        print("Benchmark: NEW vs BEST")
        new_vs_best = _evaluate(new_player, best_player, args.benchmark_games)

        # 4) Acceptance: frozen + strict
        accept = (
            new_vs_best["winrate"] > 0.5
            and float(sum(new_vs_best["diffs"]) / len(new_vs_best["diffs"])) > 0.0
            and new_vs_beginner["winrate"] >= best_vs_beginner["winrate"]
        )

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "best_model": str(best_path),
            "new_model": str(new_path),
            "best_vs_beginner": best_vs_beginner,
            "new_vs_beginner": new_vs_beginner,
            "new_vs_best": new_vs_best,
            "accepted": accept,
        }
        Path("selfplay_experiments.jsonl").write_text("", encoding="utf-8") if not Path("selfplay_experiments.jsonl").exists() else None
        with open("selfplay_experiments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if accept:
            best_file.write_text(str(new_path), encoding="utf-8")
            print("Accepted new model.")
        else:
            print("Rejected new model.")

        def _avg_diff(result: dict) -> float:
            diffs = result.get("diffs", [])
            if not diffs:
                return 0.0
            return float(sum(diffs) / len(diffs))

        def _wins(result: dict) -> int:
            return int(result.get("wins", 0))

        def _games(result: dict) -> int:
            return int(result.get("games", args.benchmark_games))

        print("Loop summary:")
        print(f"- Results log: {Path('selfplay_experiments.jsonl').resolve()}")
        print(f"- Best model: {best_path}")
        print(f"- New model: {new_path}")
        print(
            f"- Best vs beginner: wins={_wins(best_vs_beginner)}/{_games(best_vs_beginner)} "
            f"winrate={best_vs_beginner['winrate']:.3f} avg_diff={_avg_diff(best_vs_beginner):.2f}"
        )
        print(
            f"- New vs beginner: wins={_wins(new_vs_beginner)}/{_games(new_vs_beginner)} "
            f"winrate={new_vs_beginner['winrate']:.3f} avg_diff={_avg_diff(new_vs_beginner):.2f}"
        )
        print(
            f"- New vs best: wins={_wins(new_vs_best)}/{_games(new_vs_best)} "
            f"winrate={new_vs_best['winrate']:.3f} avg_diff={_avg_diff(new_vs_best):.2f}"
        )
        print(f"- Accepted: {accept}")

        if args.loops != -1 and i >= args.loops:
            break
