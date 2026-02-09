"""Evaluate trained models on held-out IL data and optionally benchmark vs a baseline.

This script is intended for model selection after IL training. It reports:
  - discard/pegging validation losses (or accuracy for classifier)
  - optional winrate vs a baseline opponent (default: medium)
  - combined rank using loss rank + winrate rank
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from crib_ai_trainer.players.neural_player import (
    LinearDiscardClassifier,
    LinearValueModel,
    MLPValueModel,
    PeggingRNNValueModel,
    PeggingTransformerValueModel,
    GBTValueModel,
    RandomForestValueModel,
    get_discard_feature_indices,
    get_pegging_feature_indices,
)
from cribbage.utils import play_multiple_games
from cribbage.players.random_player import RandomPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.hard_player import HardPlayer
from crib_ai_trainer.players.neural_player import (
    AIPlayer,
    NeuralDiscardOnlyPlayer,
    NeuralPegOnlyPlayer,
)


@dataclass(frozen=True)
class ModelEvalResult:
    label: str
    model_dir: str
    discard_loss: str
    discard_model_type: str | None
    pegging_model_type: str | None
    discard_only: bool
    pegging_only: bool
    discard_metric: dict[str, float]
    pegging_metric: dict[str, float]
    winrate: float | None
    avg_diff: float | None
    benchmark_games: int | None


def _read_queue_models(args) -> list[str]:
    if args.queue_models:
        return [p.strip() for p in args.queue_models.split(",") if p.strip()]
    if args.queue_file:
        queue_path = Path(args.queue_file)
        if not queue_path.exists():
            raise SystemExit(f"--queue_file not found: {queue_path}")
        return [line.strip() for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.scan_version:
        version_dir = Path(args.models_dir) / args.model_version
        if not version_dir.exists():
            raise SystemExit(f"Model version dir not found: {version_dir}")
        if args.scan_subdir:
            version_dir = version_dir / args.scan_subdir
            if not version_dir.exists():
                raise SystemExit(f"Model subdir not found: {version_dir}")
        return [str(p) for p in version_dir.iterdir() if p.is_dir()]
    raise SystemExit("Provide --queue_models, --queue_file, or --scan_version.")


def _load_meta(model_dir: str) -> dict[str, Any]:
    meta_path = Path(model_dir) / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing model_meta.json in {model_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_eval_dirs(args, meta: dict[str, Any]) -> tuple[Path, Path]:
    data_dir = Path(args.eval_data_dir) if args.eval_data_dir else Path(meta.get("data_dir", ""))
    if not data_dir:
        raise SystemExit("No eval data dir provided and model_meta.json missing data_dir.")
    pegging_dir = Path(args.eval_pegging_data_dir) if args.eval_pegging_data_dir else Path(meta.get("pegging_data_dir", data_dir))
    return data_dir, pegging_dir


def _read_eval_pegging_feature_set(pegging_dir: Path) -> str | None:
    meta_path = pegging_dir / "dataset_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pegging = meta.get("pegging", {})
    features = pegging.get("features", {})
    return features.get("feature_set")


def _select_shard(shards: list[Path], pick: str) -> Path:
    if not shards:
        raise SystemExit("No shards found for evaluation.")
    if pick == "first":
        return shards[0]
    if pick == "all":
        return shards[-1]
    # default: last
    return shards[-1]


def _load_discard_model(model_dir: str, meta: dict[str, Any]):
    discard_model_type = meta.get("discard_model_type") or meta.get("model_type")
    discard_loss = meta.get("discard_loss", "regression")
    discard_file = meta.get("discard_model_file")
    if discard_loss in {"classification", "ranking"}:
        if discard_model_type != "linear":
            raise SystemExit("Classification/ranking discard must be linear.")
        path = Path(model_dir) / (discard_file or "discard_linear.npz")
        return LinearDiscardClassifier.load_npz(str(path))
    path = Path(model_dir) / (discard_file or "discard_linear.npz")
    if discard_model_type == "mlp":
        return MLPValueModel.load_pt(str(path))
    if discard_model_type in {"gru", "lstm"}:
        return PeggingRNNValueModel.load_pt(str(path))
    if discard_model_type == "transformer":
        return PeggingTransformerValueModel.load_pt(str(path))
    if discard_model_type == "gbt":
        return GBTValueModel.load_joblib(str(path))
    if discard_model_type == "rf":
        return RandomForestValueModel.load_joblib(str(path))
    return LinearValueModel.load_npz(str(path))


def _load_pegging_model(model_dir: str, meta: dict[str, Any]):
    pegging_model_type = meta.get("pegging_model_type") or meta.get("model_type")
    pegging_file = meta.get("pegging_model_file")
    path = Path(model_dir) / (pegging_file or "pegging_linear.npz")
    if pegging_model_type == "mlp":
        return MLPValueModel.load_pt(str(path))
    if pegging_model_type in {"gru", "lstm"}:
        return PeggingRNNValueModel.load_pt(str(path))
    if pegging_model_type == "transformer":
        return PeggingTransformerValueModel.load_pt(str(path))
    if pegging_model_type == "gbt":
        return GBTValueModel.load_joblib(str(path))
    if pegging_model_type == "rf":
        return RandomForestValueModel.load_joblib(str(path))
    return LinearValueModel.load_npz(str(path))


def _eval_discard(model, meta: dict[str, Any], shard: Path, eval_samples: int) -> dict[str, float]:
    discard_feature_set = meta.get("discard_feature_set", "full")
    discard_loss = meta.get("discard_loss", "regression")
    discard_feature_indices = get_discard_feature_indices(discard_feature_set)

    with np.load(shard) as d:
        Xd = d["X"].astype(np.float32)
        yd = d["y"]
    n = min(eval_samples, Xd.shape[0]) if eval_samples > 0 else Xd.shape[0]
    Xd = Xd[:n]
    yd = yd[:n]
    if discard_loss == "classification":
        Xd_eval = Xd.astype(np.float32)
        Xd_eval = Xd_eval[..., discard_feature_indices]
        yd_eval = yd.astype(np.int64)
        scores = np.tensordot(Xd_eval, model.w, axes=([2], [0])) + model.b
        preds = np.argmax(scores, axis=1)
        acc = float(np.mean(preds == yd_eval)) if n > 0 else 0.0
        return {"discard_classifier_top1_acc": acc}
    if discard_loss == "ranking":
        Xd_eval = Xd.astype(np.float32)
        Xd_eval = Xd_eval[..., discard_feature_indices]
        yd_eval = yd.astype(np.float32)
        scores = np.tensordot(Xd_eval, model.w, axes=([2], [0])) + model.b
        model_margins = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
        target_margins = np.sort(yd_eval, axis=1)[:, -1] - np.sort(yd_eval, axis=1)[:, -2]
        return {
            "discard_ranker_avg_model_margin": float(np.mean(model_margins)) if n > 0 else 0.0,
            "discard_ranker_avg_target_margin": float(np.mean(target_margins)) if n > 0 else 0.0,
        }
    Xd_eval = Xd.astype(np.float32)
    Xd_eval = Xd_eval[:, discard_feature_indices]
    yd_eval = yd.astype(np.float32)
    pred = model.predict_batch(Xd_eval)
    mse = float(np.mean((pred - yd_eval) ** 2)) if n > 0 else 0.0
    return {"discard_regressor_mse": mse}


def _eval_pegging(model, meta: dict[str, Any], shard: Path, eval_samples: int) -> dict[str, float]:
    pegging_feature_set = meta.get("pegging_feature_set", "full")
    pegging_feature_indices = get_pegging_feature_indices(pegging_feature_set)
    with np.load(shard) as p:
        Xp = p["X"].astype(np.float32)
        yp = p["y"].astype(np.float32)
    n = min(eval_samples, Xp.shape[0]) if eval_samples > 0 else Xp.shape[0]
    Xp_eval = Xp[:n]
    yp_eval = yp[:n]
    is_seq_model = hasattr(model, "seq_len") and hasattr(model, "step_dim") and hasattr(model, "static_dim")
    if not is_seq_model:
        if len(pegging_feature_indices) > 0 and max(pegging_feature_indices) < Xp_eval.shape[1]:
            Xp_eval = Xp_eval[:, pegging_feature_indices]
    else:
        expected = int(model.static_dim) + int(model.seq_len) * int(model.step_dim)
        if Xp_eval.shape[1] != expected:
            raise ValueError(
                f"Pegging eval shard dim {Xp_eval.shape[1]} does not match seq model "
                f"expected {expected}. Use full_seq eval shards or set --skip_mismatch."
            )
    pred = model.predict_batch(Xp_eval)
    mse = float(np.mean((pred - yp_eval) ** 2)) if n > 0 else 0.0
    return {"pegging_regressor_mse": mse}


def _make_opponent(name: str, seed: int) -> Any:
    if name == "beginner":
        return BeginnerPlayer(name="beginner")
    if name == "medium":
        return MediumPlayer(name="medium")
    if name == "hard":
        return HardPlayer(name="hard")
    if name == "random":
        return RandomPlayer(name="random", seed=seed)
    raise SystemExit(f"Unknown benchmark opponent: {name}")


def _maybe_benchmark(args, model_dir: str, meta: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    if not args.run_benchmark:
        return None, None, None
    discard_only = bool(meta.get("discard_only", False))
    pegging_only = bool(meta.get("pegging_only", False))
    discard_model = None
    pegging_model = None
    if not pegging_only:
        discard_model = _load_discard_model(model_dir, meta)
    if not discard_only:
        pegging_model = _load_pegging_model(model_dir, meta)

    discard_feature_set = meta.get("discard_feature_set", "full")
    pegging_feature_set = meta.get("pegging_feature_set", "full")

    if discard_only:
        fallback = _make_opponent(args.benchmark_opponent, args.seed)
        player = NeuralDiscardOnlyPlayer(
            discard_model,
            fallback,
            name="neural_discard_only",
            discard_feature_set=discard_feature_set,
        )
    elif pegging_only:
        fallback = _make_opponent(args.benchmark_opponent, args.seed)
        player = NeuralPegOnlyPlayer(
            pegging_model,
            fallback,
            name="neural_peg_only",
            pegging_feature_set=pegging_feature_set,
        )
    else:
        player = AIPlayer(
            discard_model,
            pegging_model,
            name="neural",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )

    opponent = _make_opponent(args.benchmark_opponent, args.seed + 123)
    results = play_multiple_games(
        args.benchmark_games,
        p0=player,
        p1=opponent,
        seed=args.seed,
        fast_mode=True,
        copy_players=False,
    )
    return float(results.get("winrate", 0.0)), float(np.mean(results.get("diffs", [0.0]))), args.benchmark_games


def _rank_models(results: list[ModelEvalResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        loss_score = None
        if "discard_regressor_mse" in r.discard_metric:
            loss_score = r.discard_metric["discard_regressor_mse"]
            if "pegging_regressor_mse" in r.pegging_metric:
                loss_score = 0.5 * (loss_score + r.pegging_metric["pegging_regressor_mse"])
        elif "pegging_regressor_mse" in r.pegging_metric:
            loss_score = r.pegging_metric["pegging_regressor_mse"]
        row = {
            "label": r.label,
            "model_dir": r.model_dir,
            "discard_only": r.discard_only,
            "pegging_only": r.pegging_only,
            "discard_loss": r.discard_loss,
            "discard_model_type": r.discard_model_type,
            "pegging_model_type": r.pegging_model_type,
            "discard_metric": r.discard_metric,
            "pegging_metric": r.pegging_metric,
            "loss_score": loss_score,
            "winrate": r.winrate,
            "avg_diff": r.avg_diff,
            "benchmark_games": r.benchmark_games,
        }
        rows.append(row)
    loss_ranked = [r for r in rows if r["loss_score"] is not None]
    loss_ranked.sort(key=lambda x: x["loss_score"])
    for idx, r in enumerate(loss_ranked, start=1):
        r["loss_rank"] = idx
    win_ranked = [r for r in rows if r["winrate"] is not None]
    win_ranked.sort(key=lambda x: x["winrate"], reverse=True)
    for idx, r in enumerate(win_ranked, start=1):
        r["win_rank"] = idx
    for r in rows:
        if r.get("loss_rank") is not None and r.get("win_rank") is not None:
            r["combined_rank"] = int(r["loss_rank"]) + int(r["win_rank"])
        elif r.get("loss_rank") is not None:
            r["combined_rank"] = int(r["loss_rank"])
        else:
            r["combined_rank"] = None
    rows.sort(key=lambda x: (x["combined_rank"] is None, x["combined_rank"] or 1_000_000))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models_dir", type=str, default="models", help="Base models directory.")
    ap.add_argument("--model_version", type=str, default="", help="Model version to scan (empty for base).")
    ap.add_argument("--queue_models", type=str, default="", help="Comma-separated list of model dirs to scan.")
    ap.add_argument("--queue_file", type=str, default="", help="File with one model dir per line.")
    ap.add_argument("--scan_version", action="store_true", help="Scan all model dirs under models_dir/model_version.")
    ap.add_argument(
        "--scan_subdir",
        type=str,
        default="",
        choices=["", "discard", "pegging"],
        help="Optional subdir under model_version to scan (discard or pegging).",
    )
    ap.add_argument("--eval_data_dir", type=str, default="", help="Discard eval dataset dir (if evaluating discard).")
    ap.add_argument("--eval_pegging_data_dir", type=str, default="", help="Pegging eval dataset dir (if evaluating pegging).")
    ap.add_argument(
        "--eval_part",
        type=str,
        default="both",
        choices=["both", "discard", "pegging"],
        help="Evaluate discard only, pegging only, or both.",
    )
    ap.add_argument("--eval_samples", type=int, default=2048, help="Max samples per eval shard.")
    ap.add_argument("--eval_shard_pick", type=str, default="last", choices=["last", "first", "all"], help="Which shard(s) to evaluate.")
    ap.add_argument("--max_shards", type=int, default=None, help="Max shards to scan per model.")
    ap.add_argument("--skip_mismatch", action="store_true", help="Skip models whose eval shard dims don't match.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--run_benchmark", action="store_true", help="Run benchmark_2_players for each model.")
    ap.add_argument("--benchmark_opponent", type=str, default="medium", choices=["beginner", "medium", "hard"], help="Opponent for benchmark.")
    ap.add_argument("--benchmark_games", type=int, default=2000, help="Benchmark games per model.")
    ap.add_argument("--benchmark_workers", type=int, default=10, help="Benchmark worker processes.")
    ap.add_argument("--max_buffer_games", type=int, default=500, help="Max games per worker batch.")
    ap.add_argument("--benchmark_output_path", type=str, default=None, help="Text output path for benchmark results.")
    ap.add_argument("--experiments_output_path", type=str, default=None, help="JSONL output path for experiments.")
    ap.add_argument("--no_benchmark_write", action="store_true", help="Skip writing benchmark outputs.")
    ap.add_argument("--output_jsonl", type=str, default="text/model_selection_report.jsonl", help="Report JSONL path.")
    ap.add_argument("--output_txt", type=str, default="text/model_selection_report.txt", help="Report text path.")
    args = ap.parse_args()

    model_dirs = _read_queue_models(args)
    results: list[ModelEvalResult] = []
    skipped: list[dict[str, str]] = []
    for model_dir in model_dirs:
        try:
            meta = _load_meta(model_dir)
        except FileNotFoundError as exc:
            msg = str(exc)
            print(msg)
            skipped.append({"model_dir": model_dir, "reason": msg})
            continue
        discard_only = bool(meta.get("discard_only", False))
        pegging_only = bool(meta.get("pegging_only", False))
        discard_loss = meta.get("discard_loss", "regression")
        discard_model_type = meta.get("discard_model_type") or meta.get("model_type")
        pegging_model_type = meta.get("pegging_model_type") or meta.get("model_type")

        eval_data_dir, eval_pegging_dir = _resolve_eval_dirs(args, meta)
        discard_shards = sorted(Path(eval_data_dir).glob("discard_*.npz"))
        pegging_shards = sorted(Path(eval_pegging_dir).glob("pegging_*.npz"))
        if args.max_shards:
            discard_shards = discard_shards[: args.max_shards]
            pegging_shards = pegging_shards[: args.max_shards]
        discard_metric = {}
        pegging_metric = {}

        if args.eval_part in {"both", "discard"} and not pegging_only:
            try:
                discard_model = _load_discard_model(model_dir, meta)
                discard_shard = _select_shard(discard_shards, args.eval_shard_pick)
                discard_metric = _eval_discard(discard_model, meta, discard_shard, args.eval_samples)
            except (ValueError, RuntimeError) as exc:
                if args.skip_mismatch:
                    msg = f"discard eval skipped: {exc}"
                    print(f"Skipping {model_dir}: {msg}")
                    skipped.append({"model_dir": model_dir, "reason": msg})
                else:
                    raise
        if args.eval_part in {"both", "pegging"} and not discard_only:
            try:
                pegging_model = _load_pegging_model(model_dir, meta)
                pegging_shard = _select_shard(pegging_shards, args.eval_shard_pick)
                eval_feature_set = _read_eval_pegging_feature_set(eval_pegging_dir)
                is_seq_model = hasattr(pegging_model, "seq_len") and hasattr(pegging_model, "step_dim")
                if is_seq_model and eval_feature_set not in (None, "full_seq"):
                    raise ValueError(
                        f"Eval pegging feature_set={eval_feature_set!r} but seq model requires full_seq."
                    )
                pegging_metric = _eval_pegging(pegging_model, meta, pegging_shard, args.eval_samples)
            except (ValueError, RuntimeError) as exc:
                if args.skip_mismatch:
                    msg = f"pegging eval skipped: {exc}"
                    print(f"Skipping {model_dir}: {msg}")
                    skipped.append({"model_dir": model_dir, "reason": msg})
                else:
                    raise

        winrate, avg_diff, bench_games = _maybe_benchmark(args, model_dir, meta)
        label = Path(model_dir).name
        results.append(
            ModelEvalResult(
                label=label,
                model_dir=model_dir,
                discard_loss=discard_loss,
                discard_model_type=discard_model_type,
                pegging_model_type=pegging_model_type,
                discard_only=discard_only,
                pegging_only=pegging_only,
                discard_metric=discard_metric,
                pegging_metric=pegging_metric,
                winrate=winrate,
                avg_diff=avg_diff,
                benchmark_games=bench_games,
            )
        )

    ranked = _rank_models(results)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": ranked,
            "skipped": skipped,
        }
        f.write(json.dumps(payload, indent=2))

    lines = []
    lines.append("Model Selection Report")
    lines.append(f"created_at_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    header = [
        "rank",
        "label",
        "loss_score",
        "discard_metric",
        "pegging_metric",
        "winrate",
    ]
    lines.append(" | ".join(header))
    lines.append("-" * 120)
    for row in ranked:
        loss_score = row.get("loss_score")
        loss_str = f"{loss_score:.6f}" if isinstance(loss_score, float) else "n/a"
        winrate = row.get("winrate")
        win_str = f"{winrate:.3f}" if isinstance(winrate, float) else "n/a"
        lines.append(
            " | ".join(
                [
                    str(row.get("combined_rank") or "n/a"),
                    row.get("label", ""),
                    loss_str,
                    json.dumps(row.get("discard_metric", {})),
                    json.dumps(row.get("pegging_metric", {})),
                    win_str,
                ]
            )
        )
    with open(args.output_txt, "w", encoding="utf-8") as f:
        if skipped:
            lines.append("")
            lines.append("Skipped Models")
            lines.append("-" * 120)
            for item in skipped:
                lines.append(f"{item.get('model_dir')}: {item.get('reason')}")
        f.write("\n".join(lines) + "\n")
    print(f"Wrote report -> {args.output_txt}")
    print(f"Wrote report -> {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
