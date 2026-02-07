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

from cribbage.utils import play_multiple_games, wilson_ci, mean_ci
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.hard_player import HardPlayer

from scripts.generate_self_play_data import generate_self_play_data, _resolve_models_dir, _resolve_output_dir
import multiprocessing as mp
from scripts.train_models import train_models
from crib_ai_trainer.players.neural_player import (
    AIPlayer,
    LinearValueModel,
    MLPValueModel,
    GBTValueModel,
    RandomForestValueModel,
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
    if model_type == "gbt":
        return GBTValueModel.load_joblib(str(models_dir / "discard_gbt.pkl")), GBTValueModel.load_joblib(str(models_dir / "pegging_gbt.pkl"))
    if model_type == "rf":
        return RandomForestValueModel.load_joblib(str(models_dir / "discard_rf.pkl")), RandomForestValueModel.load_joblib(str(models_dir / "pegging_rf.pkl"))
    return LinearValueModel.load_npz(str(models_dir / "discard_linear.npz")), LinearValueModel.load_npz(str(models_dir / "pegging_linear.npz"))


def _make_player(models_dir: Path, name_override: str | None = None) -> AIPlayer:
    meta = _load_meta(models_dir)
    model_type = meta.get("model_type", "linear")
    discard_feature_set = meta.get("discard_feature_set", "full")
    pegging_feature_set = meta.get("pegging_feature_set", "full")
    discard_model, pegging_model = _load_value_model(models_dir, model_type)
    name = name_override or f"selfplay:{models_dir.name}"
    return AIPlayer(
        discard_model,
        pegging_model,
        name=name,
        discard_feature_set=discard_feature_set,
        pegging_feature_set=pegging_feature_set,
    )


def _split_games(total_games: int, workers: int) -> list[int]:
    if total_games <= 0:
        raise SystemExit("--benchmark_games must be > 0.")
    if workers <= 0:
        raise SystemExit("--benchmark_workers must be >= 1.")
    workers = min(workers, total_games)
    base, rem = divmod(total_games, workers)
    tasks = [base] * workers
    for i in range(rem):
        tasks[i] += 1
    return tasks


def _build_player_from_spec(spec: dict):
    kind = spec["type"]
    if kind == "ai":
        return _make_player(Path(spec["path"]), name_override=spec.get("name"))
    if kind == "beginner":
        return BeginnerPlayer(name="beginner")
    if kind == "medium":
        return MediumPlayer(name="medium")
    if kind == "hard":
        return HardPlayer(name="hard")
    raise SystemExit(f"Unknown player spec: {spec}")


def _benchmark_worker(args: tuple[dict, dict, int, int | None, int]) -> dict:
    p0_spec, p1_spec, games, seed, start_index = args
    worker_seed = None if seed is None else int(seed) + int(start_index)
    p0 = _build_player_from_spec(p0_spec)
    p1 = _build_player_from_spec(p1_spec)
    result = play_multiple_games(games, p0=p0, p1=p1, seed=worker_seed)
    return result


def _evaluate_multi(p0_spec: dict, p1_spec: dict, games: int, seed: int | None, workers: int) -> dict:
    tasks = _split_games(games, workers)
    if len(tasks) == 1:
        return _benchmark_worker((p0_spec, p1_spec, tasks[0], seed, 0))

    start_indices = []
    acc = 0
    for t in tasks:
        start_indices.append(acc)
        acc += t

    args_list = [
        (p0_spec, p1_spec, tasks[i], seed, start_indices[i]) for i in range(len(tasks))
    ]

    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=len(tasks)) as pool:
        for result in pool.imap_unordered(_benchmark_worker, args_list):
            results.append(result)

    wins = int(sum(r["wins"] for r in results))
    ties = int(sum(r.get("ties", 0) for r in results))
    diffs = []
    for r in results:
        diffs.extend(r["diffs"])
    denom = max(1, games - ties)
    winrate = float(wins) / float(denom)
    win_lo, win_hi = wilson_ci(wins, denom)
    diff_lo, diff_hi = mean_ci(diffs)
    return {
        "wins": wins,
        "diffs": diffs,
        "winrate": winrate,
        "ci_lo": win_lo,
        "ci_hi": win_hi,
        "diff_ci_lo": diff_lo,
        "diff_ci_hi": diff_hi,
        "ties": ties,
    }


def _avg_diff(result: dict) -> float:
    diffs = result.get("diffs", [])
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def _winrate(result: dict | None) -> float:
    if result is None:
        return 0.0
    return float(result.get("winrate", 0.0))


def _load_best_map(best_file: Path) -> dict:
    if not best_file.exists():
        raise SystemExit(f"Best model file not found: {best_file}")
    raw = best_file.read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit(f"Best model file is empty: {best_file}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit(f"Best model file must be a JSON object: {best_file}")
    return data


def _write_best_record(best_file: Path, model_version: str, record: dict) -> None:
    data = _load_best_map(best_file)
    data[model_version] = record
    best_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_best_record(best_file: Path, model_version: str) -> dict:
    data = _load_best_map(best_file)
    record = data.get(model_version)
    if not isinstance(record, dict):
        raise SystemExit(f"Best model file missing entry for {model_version}: {best_file}")
    if not record.get("path"):
        raise SystemExit(f"Best model entry missing path for {model_version}: {best_file}")
    return record


def _get_best_path(best_file: Path, models_dir: Path, model_version: str) -> Path:
    record = _get_best_record(best_file, model_version)
    return Path(str(record["path"]).strip())


def _next_run_id(version_dir: Path) -> str:
    version_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return "001"
    run_id = max(int(p.name) for p in run_dirs)
    return f"{run_id + 1:03d}"


def _next_candidate_id(best_path: Path) -> str:
    if not best_path.name.isdigit():
        raise SystemExit(f"Best model path must end with numeric run id (got {best_path})")
    return f"{int(best_path.name) + 1:03d}"


def _selfplay_version_from_model_version(model_version: str) -> str:
    if not model_version.startswith("discard_v"):
        if model_version.startswith("selfplayv"):
            return model_version
        raise SystemExit(f"model_version must start with 'discard_v' (got {model_version!r})")
    suffix = model_version.split("discard_v", 1)[1]
    if not suffix.isdigit():
        raise SystemExit(f"model_version must be like 'discard_v7' (got {model_version!r})")
    return f"selfplayv{suffix}"


if __name__ == "__main__":
    args = build_self_play_loop_parser().parse_args()
    if args.smoke:
        args.loops = 1
        args.games = 1
        args.selfplay_workers = 1
        args.benchmark_games = 2
        args.benchmark_workers = 1
    if not args.best_file:
        args.best_file = "text/best_models_selfplay.json"
    if not args.selfplay_dataset_version:
        args.selfplay_dataset_version = _selfplay_version_from_model_version(args.model_version)

    models_dir = Path(args.models_dir)
    best_file = Path(args.best_file)
    teacher_dir = _resolve_output_dir(TRAINING_DATA_DIR, args.teacher_dataset_version)
    selfplay_dir = _resolve_output_dir(TRAINING_DATA_DIR, args.selfplay_dataset_version)

    i = 0
    no_improve_streak = 0
    last_selfplay_shards_used = None
    while True:
        i += 1
        print(f"\n=== Self-play loop {i} ===")
        best_record = _get_best_record(best_file, args.model_version)
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
            args.selfplay_workers,
        )

        # 2) Train new model mixing teacher + self-play data
        # Prefer saving new models under the same version as the current best.
        model_version_for_training = args.model_version
        if best_path.parent.name and best_path.parent.name != args.model_version:
            model_version_for_training = best_path.parent.name
            print(f"Using model_version={model_version_for_training} to keep accepted models in the same series.")

        selfplay_version = _selfplay_version_from_model_version(model_version_for_training)
        base_models_dir = Path(args.models_dir)
        if base_models_dir.name == "regression":
            models_root = base_models_dir / selfplay_version
        else:
            models_root = base_models_dir / "regression" / selfplay_version
        run_id = _next_candidate_id(best_path)
        models_dir = models_root / run_id

        if args.incremental:
            if last_selfplay_shards_used is None:
                if "selfplay_data_dir" not in best_record or "selfplay_shards_used" not in best_record:
                    shards = sorted(Path(selfplay_dir).glob("discard_*.npz"))
                    if not shards:
                        raise SystemExit(f"No self-play discard shards found in {selfplay_dir}")
                    best_record["selfplay_data_dir"] = str(selfplay_dir)
                    best_record["selfplay_shards_used"] = len(shards)
                    _write_best_record(best_file, args.model_version, best_record)
                if Path(best_record["selfplay_data_dir"]) != Path(selfplay_dir):
                    raise SystemExit(
                        f"best_model selfplay_data_dir={best_record['selfplay_data_dir']} "
                        f"does not match current selfplay_dir={selfplay_dir}."
                    )
                last_selfplay_shards_used = int(best_record["selfplay_shards_used"])
            selfplay_shards = sorted(Path(selfplay_dir).glob("discard_*.npz"))
            if not selfplay_shards:
                raise SystemExit(f"No self-play discard shards found in {selfplay_dir}")
            if last_selfplay_shards_used >= len(selfplay_shards):
                raise SystemExit(
                    f"No new self-play shards to train (have {len(selfplay_shards)}, "
                    f"already used {last_selfplay_shards_used})."
                )

            teacher_ratio = 1.0 - float(args.selfplay_ratio)
            if teacher_ratio < 0.0 or teacher_ratio > 1.0:
                raise SystemExit("--selfplay_ratio must be between 0 and 1.")
            best_record["selfplay_data_dir"] = str(selfplay_dir)
            best_record["selfplay_shards_used"] = len(selfplay_shards)
            _write_best_record(best_file, args.model_version, best_record)

        train_args = argparse.Namespace(
            data_dir=selfplay_dir if args.incremental else teacher_dir,
            extra_data_dir=teacher_dir if args.incremental else selfplay_dir,
            extra_ratio=(1.0 - float(args.selfplay_ratio)) if args.incremental else args.selfplay_ratio,
            models_dir=str(models_dir),
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
            torch_threads=8,
            parallel_heads=True,
            eval_samples=2048,
            max_shards=None,
            rank_pairs_per_hand=20,
            incremental=args.incremental,
            incremental_from=str(best_path) if args.incremental else None,
            incremental_start_shard=last_selfplay_shards_used if args.incremental else 0,
            incremental_epochs=args.incremental_epochs,
            max_train_samples=None,
        )
        train_models(train_args)
        new_path = Path(train_args.models_dir)
        print(f"New candidate: {new_path}")

        # 3) Benchmarks
        if args.benchmark_seed is None:
            args.benchmark_seed = int(datetime.now(timezone.utc).timestamp()) % 2_000_000_000
        print(f"Benchmark seed: {args.benchmark_seed}")
        print("Benchmark: NEW vs BEST")
        new_vs_best = _evaluate_multi(
            {"type": "ai", "path": str(new_path), "name": f"selfplay:new:{new_path.name}"},
            {"type": "ai", "path": str(best_path), "name": f"selfplay:best:{best_path.name}"},
            args.benchmark_games,
            args.benchmark_seed,
            args.benchmark_workers,
        )
        print(
            f"  -> wins={new_vs_best['wins']}/{args.benchmark_games} "
            f"winrate={new_vs_best['winrate']:.3f} avg_diff={_avg_diff(new_vs_best):.2f}"
        )

        if args.benchmark_opponent == "beginner":
            label = "beginner"
        elif args.benchmark_opponent == "medium":
            label = "medium"
        else:
            label = "hard"
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
                    best_vs_medium = _evaluate_multi(
                        {"type": "ai", "path": str(best_path), "name": f"selfplay:best:{best_path.name}"},
                        {"type": "beginner"},
                        args.benchmark_games,
                        args.benchmark_seed,
                        args.benchmark_workers,
                    )
                elif args.benchmark_opponent == "medium":
                    print("Benchmark: BEST vs MEDIUM")
                    best_vs_medium = _evaluate_multi(
                        {"type": "ai", "path": str(best_path), "name": f"selfplay:best:{best_path.name}"},
                        {"type": "medium"},
                        args.benchmark_games,
                        args.benchmark_seed,
                        args.benchmark_workers,
                    )
                else:
                    print("Benchmark: BEST vs HARD")
                    best_vs_medium = _evaluate_multi(
                        {"type": "ai", "path": str(best_path), "name": f"selfplay:best:{best_path.name}"},
                        {"type": "hard"},
                        args.benchmark_games,
                        args.benchmark_seed,
                        args.benchmark_workers,
                    )
                print(
                    f"  -> wins={best_vs_medium['wins']}/{args.benchmark_games} "
                    f"winrate={best_vs_medium['winrate']:.3f} avg_diff={_avg_diff(best_vs_medium):.2f}"
                )
                best_record.setdefault("benchmarks", {})[label] = {
                    **best_vs_medium,
                    "benchmark_games": args.benchmark_games,
                }
                _write_best_record(best_file, args.model_version, best_record)

            if args.benchmark_opponent == "beginner":
                print("Benchmark: NEW vs BEGINNER")
                new_vs_medium = _evaluate_multi(
                    {"type": "ai", "path": str(new_path), "name": f"selfplay:new:{new_path.name}"},
                    {"type": "beginner"},
                    args.benchmark_games,
                    args.benchmark_seed,
                    args.benchmark_workers,
                )
            elif args.benchmark_opponent == "medium":
                print("Benchmark: NEW vs MEDIUM")
                new_vs_medium = _evaluate_multi(
                    {"type": "ai", "path": str(new_path), "name": f"selfplay:new:{new_path.name}"},
                    {"type": "medium"},
                    args.benchmark_games,
                    args.benchmark_seed,
                    args.benchmark_workers,
                )
            else:
                print("Benchmark: NEW vs HARD")
                new_vs_medium = _evaluate_multi(
                    {"type": "ai", "path": str(new_path), "name": f"selfplay:new:{new_path.name}"},
                    {"type": "hard"},
                    args.benchmark_games,
                    args.benchmark_seed,
                    args.benchmark_workers,
                )
            print(
                f"  -> wins={new_vs_medium['wins']}/{args.benchmark_games} "
                f"winrate={new_vs_medium['winrate']:.3f} avg_diff={_avg_diff(new_vs_medium):.2f}"
            )

        # 4) Acceptance: winrate only
        accept = False
        if best_vs_medium is not None and new_vs_medium is not None:
            accept = new_vs_medium["winrate"] > best_vs_medium["winrate"]

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "best_model": str(best_path),
            "new_model": str(new_path),
            "best_vs_medium": best_vs_medium,
            "new_vs_medium": new_vs_medium,
            "new_vs_best": new_vs_best,
            "accepted": accept,
        }
        if not args.smoke:
            results_path = Path("text/selfplay_experiments.jsonl")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text("", encoding="utf-8") if not results_path.exists() else None
            with open(results_path, "a", encoding="utf-8") as f:
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
            if not args.smoke:
                _write_best_record(best_file, args.model_version, new_record)
            print("Accepted new model.")
            no_improve_streak = 0
        else:
            print("Rejected new model.")
            no_improve_streak += 1

        def _wins(result: dict | None) -> int:
            if result is None:
                return 0
            return int(result.get("wins", 0))

        def _games(result: dict | None) -> int:
            if result is None:
                return 0
            return int(result.get("games", args.benchmark_games))

        print("Loop summary:")
        print(f"- Results log: {Path('text/selfplay_experiments.jsonl').resolve()}")
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

        best_label = best_path.name
        if not best_label.isdigit():
            raise SystemExit(f"Best model path must end with numeric run id (got {best_path})")
        summary_line = (
            f"{new_path.name} after {args.games} self play games, {args.benchmark_games} benchmark games vs best "
            f"[{best_label}] wins={_wins(new_vs_best)}/{_games(new_vs_best)} "
            f"winrate={new_vs_best['winrate']:.3f} avg_diff={_avg_diff(new_vs_best):.2f}, "
            f"vs {label}: wins={_wins(new_vs_medium)}/{_games(new_vs_medium)} "
            f"winrate={_winrate(new_vs_medium):.3f} avg_diff={_avg_diff(new_vs_medium):.2f}, "
            f"Best vs {label}: wins={_wins(best_vs_medium)}/{_games(best_vs_medium)} "
            f"winrate={_winrate(best_vs_medium):.3f} avg_diff={_avg_diff(best_vs_medium):.2f}\n"
        )
        if not args.smoke:
            Path("text").mkdir(parents=True, exist_ok=True)
            with open("text/selfplay_results.txt", "a", encoding="utf-8") as f:
                f.write(summary_line)

        if no_improve_streak >= args.max_no_improve:
            print(f"Stopping after {no_improve_streak} non-improving loops.")
            break

        if args.loops != -1 and i >= args.loops:
            break

# Script summary: generate self-play data, train a new model, benchmark it, and accept only winrate improvements.
