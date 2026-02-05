"""Self-play loop with frozen-best acceptance rule.

Workflow:
1) Generate self-play data using latest accepted model.
2) Train a new model mixing teacher + self-play data.
3) Benchmark new vs frozen best and vs beginner.
4) Accept only if strict criteria are met.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import argparse

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import (
    TRAINING_DATA_DIR,
)

from cribbage.utils import play_multiple_games
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.medium_player import MediumPlayer

from scripts.generate_self_play_data import generate_self_play_data, _resolve_models_dir, _resolve_output_dir
from scripts.train_linear_models import train_linear_models
from crib_ai_trainer.players.neural_player import (
    NeuralRegressionPlayer,
    LinearValueModel,
    MLPValueModel,
)
from utils import build_self_play_loop_parser


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


def _avg_diff(result: dict) -> float:
    diffs = result.get("diffs", [])
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def _winrate(result: dict | None) -> float:
    if result is None:
        return 0.0
    return float(result.get("winrate", 0.0))


def _load_best_record(best_file: Path) -> dict:
    if not best_file.exists():
        return {}
    raw = best_file.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "path" in data:
            return data
    except Exception:
        pass
    # Backward compatibility: plain path
    return {"path": raw, "benchmarks": {}}


def _write_best_record(best_file: Path, record: dict) -> None:
    best_file.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _get_best_path(best_file: Path, models_dir: Path, model_version: str) -> Path:
    record = _load_best_record(best_file)
    if record.get("path"):
        return Path(str(record["path"]).strip())
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
    args = build_self_play_loop_parser().parse_args()

    models_dir = Path(args.models_dir)
    best_file = Path(args.best_file)
    teacher_dir = _resolve_output_dir(TRAINING_DATA_DIR, args.teacher_dataset_version, None, new_run=False)
    selfplay_dir = _resolve_output_dir(TRAINING_DATA_DIR, args.selfplay_dataset_version, None, new_run=False)

    i = 0
    while True:
        i += 1
        print(f"\n=== Self-play loop {i} ===")
        best_record = _load_best_record(best_file)
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
            args.pegging_ev_mode,
            args.pegging_ev_rollouts,
            args.win_prob_mode,
            args.win_prob_rollouts,
            args.win_prob_min_score,
        )

        # 2) Train new model mixing teacher + self-play data
        # Prefer saving new models under the same version as the current best.
        model_version_for_training = args.model_version
        if best_path.parent.name and best_path.parent.name != args.model_version:
            model_version_for_training = best_path.parent.name
            print(f"Using model_version={model_version_for_training} to keep accepted models in the same series.")

        train_args = argparse.Namespace(
            data_dir=teacher_dir,
            extra_data_dir=selfplay_dir,
            extra_ratio=args.selfplay_ratio,
            models_dir=args.models_dir,
            model_version=model_version_for_training,
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

        print("Benchmark: NEW vs BEST")
        new_vs_best = _evaluate(new_player, best_player, args.benchmark_games)
        print(
            f"  -> wins={new_vs_best['wins']}/{args.benchmark_games} "
            f"winrate={new_vs_best['winrate']:.3f} avg_diff={_avg_diff(new_vs_best):.2f}"
        )

        label = "beginner" if args.benchmark_opponent == "beginner" else "medium"
        best_vs_medium = None
        new_vs_medium = None

        if new_vs_best["winrate"] > 0.5:
            # Best vs opponent (cached)
            cached = (best_record.get("benchmarks") or {}).get(label, {})
            if cached and cached.get("benchmark_games") == args.benchmark_games:
                best_vs_medium = cached
                print(f"Using cached BEST vs {label.upper()} results.")
            else:
                if args.benchmark_opponent == "beginner":
                    print("Benchmark: BEST vs BEGINNER")
                    best_vs_medium = _evaluate(best_player, BeginnerPlayer(name="beginner"), args.benchmark_games)
                else:
                    print("Benchmark: BEST vs MEDIUM")
                    best_vs_medium = _evaluate(best_player, MediumPlayer(name="medium"), args.benchmark_games)
                print(
                    f"  -> wins={best_vs_medium['wins']}/{args.benchmark_games} "
                    f"winrate={best_vs_medium['winrate']:.3f} avg_diff={_avg_diff(best_vs_medium):.2f}"
                )
                best_record.setdefault("benchmarks", {})[label] = {
                    **best_vs_medium,
                    "benchmark_games": args.benchmark_games,
                }
                _write_best_record(best_file, best_record)

            if args.benchmark_opponent == "beginner":
                print("Benchmark: NEW vs BEGINNER")
                new_vs_medium = _evaluate(new_player, BeginnerPlayer(name="beginner"), args.benchmark_games)
            else:
                print("Benchmark: NEW vs MEDIUM")
                new_vs_medium = _evaluate(new_player, MediumPlayer(name="medium"), args.benchmark_games)
            print(
                f"  -> wins={new_vs_medium['wins']}/{args.benchmark_games} "
                f"winrate={new_vs_medium['winrate']:.3f} avg_diff={_avg_diff(new_vs_medium):.2f}"
            )

        # 4) Acceptance: frozen + strict
        accept = False
        if new_vs_best["winrate"] > 0.5 and best_vs_medium is not None and new_vs_medium is not None:
            accept = new_vs_medium["winrate"] >= best_vs_medium["winrate"]

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "best_model": str(best_path),
            "new_model": str(new_path),
            "best_vs_medium": best_vs_medium,
            "new_vs_medium": new_vs_medium,
            "new_vs_best": new_vs_best,
            "accepted": accept,
        }
        Path("selfplay_experiments.jsonl").write_text("", encoding="utf-8") if not Path("selfplay_experiments.jsonl").exists() else None
        with open("selfplay_experiments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if accept:
            new_record = {
                "path": str(new_path),
                "benchmarks": {},
            }
            if new_vs_medium is not None:
                new_record["benchmarks"][label] = {
                    **new_vs_medium,
                    "benchmark_games": args.benchmark_games,
                }
            _write_best_record(best_file, new_record)
            print("Accepted new model.")
        else:
            print("Rejected new model.")

        def _wins(result: dict | None) -> int:
            if result is None:
                return 0
            return int(result.get("wins", 0))

        def _games(result: dict | None) -> int:
            if result is None:
                return 0
            return int(result.get("games", args.benchmark_games))

        print("Loop summary:")
        print(f"- Results log: {Path('selfplay_experiments.jsonl').resolve()}")
        print(f"- Best model: {best_path}")
        print(f"- New model: {new_path}")
        if best_vs_medium is not None:
            print(
                f"- Best vs {label}: wins={_wins(best_vs_medium)}/{_games(best_vs_medium)} "
                f"winrate={best_vs_medium['winrate']:.3f} avg_diff={_avg_diff(best_vs_medium):.2f}"
            )
        else:
            print(f"- Best vs {label}: skipped")
        if new_vs_medium is not None:
            print(
                f"- New vs {label}: wins={_wins(new_vs_medium)}/{_games(new_vs_medium)} "
                f"winrate={new_vs_medium['winrate']:.3f} avg_diff={_avg_diff(new_vs_medium):.2f}"
            )
        else:
            print(f"- New vs {label}: skipped")
        print(
            f"- New vs best: wins={_wins(new_vs_best)}/{_games(new_vs_best)} "
            f"winrate={new_vs_best['winrate']:.3f} avg_diff={_avg_diff(new_vs_best):.2f}"
        )
        print(f"- Accepted: {accept}")

        summary_line = (
            f"new after {args.games} self play games, {args.benchmark_games} benchmark games vs best "
            f"[{best_path.name}] wins={_wins(new_vs_best)}/{_games(new_vs_best)} "
            f"winrate={new_vs_best['winrate']:.3f} avg_diff={_avg_diff(new_vs_best):.2f}, "
            f"vs {label}: wins={_wins(new_vs_medium)}/{_games(new_vs_medium)} "
            f"winrate={_winrate(new_vs_medium):.3f} avg_diff={_avg_diff(new_vs_medium) if new_vs_medium else 0.0:.2f}, "
            f"Best vs {label}: wins={_wins(best_vs_medium)}/{_games(best_vs_medium)} "
            f"winrate={_winrate(best_vs_medium):.3f} avg_diff={_avg_diff(best_vs_medium) if best_vs_medium else 0.0:.2f}\n"
        )
        with open("selfplay_results.txt", "a", encoding="utf-8") as f:
            f.write(summary_line)

        if args.loops != -1 and i >= args.loops:
            break
