"""Benchmark trained NeuralPlayer vs ReasonablePlayer.

Usage:
  python benchmark_neural_vs_reasonable.py --games 500 --models_dir models
"""

from __future__ import annotations

import sys
import numpy as np
import os
import json
import multiprocessing as mp
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from cribbage.utils import play_multiple_games
from cribbage.players.random_player import RandomPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.hard_player import HardPlayer
from crib_ai_trainer.players.neural_player import (
    LinearDiscardClassifier,
    LinearValueModel,
    NeuralClassificationPlayer,
    AIPlayer,
    MLPPlayer,
    GBTPlayer,
    RandomForestPlayer,
    NeuralDiscardOnlyPlayer,
    NeuralPegOnlyPlayer,
    MLPValueModel,
    PeggingRNNValueModel,
    GBTValueModel,
    RandomForestValueModel,
)
from utils import build_benchmark_parser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_scores(game) -> tuple[int, int]:
    # Try a few common patterns
    if hasattr(game, "scores"):
        s = getattr(game, "scores")
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            return int(s[0]), int(s[1])
    if hasattr(game, "players"):
        players = getattr(game, "players")
        if len(players) >= 2:
            p0, p1 = players[0], players[1]
            if hasattr(p0, "score") and hasattr(p1, "score"):
                return int(p0.score), int(p1.score)
    raise RuntimeError("Can't read final scores from game. Add a get_scores() mapping for your engine.")


def _find_latest_run_id(version_dir: Path) -> str | None:
    if not version_dir.exists():
        return None
    run_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return None
    run_id = max(int(p.name) for p in run_dirs)
    return f"{run_id:03d}"


def _resolve_models_dir(args) -> str:
    base = Path(args.models_dir)

    # If the user passed an explicit run folder with a model, use it directly.
    if (base / "model_meta.json").exists() or (base / "discard_linear.npz").exists():
        return str(base)

    # Prefer explicit model version subdir if it exists.
    version_dir = base / args.model_version if args.model_version else base
    if not version_dir.exists():
        version_dir = base

    if args.model_run_id:
        return str(version_dir / args.model_run_id)

    if args.latest_model:
        latest = _find_latest_run_id(version_dir)
        if latest:
            return str(version_dir / latest)

    # Fallback: return version_dir itself
    return str(version_dir)



def _build_player_factory(args, fallback_override: str | None):
    args.models_dir = _resolve_models_dir(args)
    logger.debug("Loading models from %s", args.models_dir)
    # print("discard |w|", float(np.linalg.norm(discard_model.w)), "b", float(discard_model.b))
    # print("pegging  |w|", float(np.linalg.norm(pegging_model.w)), "b", float(pegging_model.b))
    # discard and pegging weights after 4000 games
    # discard |w| 3.507629156112671 b 1.5574564933776855
    # pegging  |w| 1.011649489402771 b 0.2532176971435547
    # breakpoint()
    rng = np.random.default_rng(args.seed)

    def base_player_factory(name: str):
        if name == "beginner":
            return BeginnerPlayer(name=name)
        if name == "random":
            return RandomPlayer(name=name, seed=args.seed)
        if name == "medium":
            return MediumPlayer(name=name)
        if name == "hard":
            return HardPlayer(name=name)
        raise ValueError(f"Unknown fallback player type: {name}")

    def resolve_model_tag() -> str:
        if args.model_tag:
            return args.model_tag
        # Try to read model_meta.json to derive a stable version label.
        meta_path = os.path.join(args.models_dir, "model_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                version = meta.get("model_version")
                run_id = meta.get("run_id")
                if version and run_id:
                    return f"{version}-{run_id}"
                if version:
                    return str(version)
            except Exception:
                pass
        # Default to models_dir basename (e.g., "ranking", "regression", "classification")
        return os.path.basename(os.path.normpath(args.models_dir))

    # Load model metadata to align feature sets and model files.
    meta_path = os.path.join(args.models_dir, "model_meta.json")
    model_type = "linear"
    mlp_hidden = None
    discard_model_file = None
    pegging_model_file = None
    size_suffix = ""
    if not os.path.exists(meta_path):
        raise SystemExit(f"Expected model_meta.json at {meta_path} but it does not exist.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    args.discard_feature_set = meta.get("discard_feature_set", args.discard_feature_set)
    args.pegging_feature_set = meta.get("pegging_feature_set", args.pegging_feature_set)
    model_type = meta.get("model_type", model_type)
    discard_model_type = meta.get("discard_model_type", model_type)
    pegging_model_type = meta.get("pegging_model_type", model_type)
    discard_only = bool(meta.get("discard_only", False))
    pegging_only = bool(meta.get("pegging_only", False))
    mlp_hidden = meta.get("mlp_hidden", None)
    discard_model_file = meta.get("discard_model_file")
    pegging_model_file = meta.get("pegging_model_file")
    if discard_only and pegging_only:
        raise SystemExit("model_meta.json cannot set both discard_only and pegging_only.")
    if not discard_only and not pegging_only:
        if discard_model_file is None or pegging_model_file is None:
            raise SystemExit("model_meta.json missing discard_model_file or pegging_model_file.")
    if discard_only and discard_model_file is None:
        raise SystemExit("model_meta.json missing discard_model_file for discard_only model.")
    if pegging_only and pegging_model_file is None:
        raise SystemExit("model_meta.json missing pegging_model_file for pegging_only model.")
    if model_type == "mlp":
        def _coerce_hidden(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return tuple(int(v) for v in value)
            if isinstance(value, str):
                parts = [p.strip() for p in value.split(",") if p.strip()]
                if not parts:
                    return None
                return tuple(int(p) for p in parts)
            return None

        discard_hidden = _coerce_hidden(meta.get("discard_mlp_hidden"))
        pegging_hidden = _coerce_hidden(meta.get("pegging_mlp_hidden"))
        fallback_hidden = _coerce_hidden(meta.get("mlp_hidden"))
        if discard_hidden is None:
            discard_hidden = fallback_hidden
        if pegging_hidden is None:
            pegging_hidden = fallback_hidden
        if discard_hidden is not None and pegging_hidden is not None and discard_hidden != pegging_hidden:
            d = "x".join(str(int(v)) for v in discard_hidden)
            p = "x".join(str(int(v)) for v in pegging_hidden)
            size_suffix = f"[d{d}_p{p}]"

    fallback_player_name = fallback_override or args.fallback_player

    def _load_discard_model():
        if discard_model_file is None:
            raise SystemExit("discard model file missing for this models_dir.")
        if discard_model_file is not None:
            path = f"{args.models_dir}/{discard_model_file}"
            if discard_model_type == "mlp":
                return MLPValueModel.load_pt(path)
            if discard_model_type == "gbt":
                return GBTValueModel.load_joblib(path)
            if discard_model_type == "rf":
                return RandomForestValueModel.load_joblib(path)
            return LinearValueModel.load_npz(path)
        if discard_model_type == "mlp":
            return MLPValueModel.load_pt(f"{args.models_dir}/discard_mlp.pt")
        if discard_model_type == "gbt":
            return GBTValueModel.load_joblib(f"{args.models_dir}/discard_gbt.pkl")
        if discard_model_type == "rf":
            return RandomForestValueModel.load_joblib(f"{args.models_dir}/discard_rf.pkl")
        return LinearValueModel.load_npz(f"{args.models_dir}/discard_linear.npz")

    def _load_pegging_model():
        if pegging_model_file is None:
            raise SystemExit("pegging model file missing for this models_dir.")
        if pegging_model_file is not None:
            path = f"{args.models_dir}/{pegging_model_file}"
            if pegging_model_type == "mlp":
                return MLPValueModel.load_pt(path)
            if pegging_model_type == "gru" or pegging_model_type == "lstm":
                return PeggingRNNValueModel.load_pt(path)
            if pegging_model_type == "gbt":
                return GBTValueModel.load_joblib(path)
            if pegging_model_type == "rf":
                return RandomForestValueModel.load_joblib(path)
            return LinearValueModel.load_npz(path)
        if pegging_model_type == "mlp":
            return MLPValueModel.load_pt(f"{args.models_dir}/pegging_mlp.pt")
        if pegging_model_type == "gru":
            return PeggingRNNValueModel.load_pt(f"{args.models_dir}/pegging_gru.pt")
        if pegging_model_type == "lstm":
            return PeggingRNNValueModel.load_pt(f"{args.models_dir}/pegging_lstm.pt")
        if pegging_model_type == "gbt":
            return GBTValueModel.load_joblib(f"{args.models_dir}/pegging_gbt.pkl")
        if pegging_model_type == "rf":
            return RandomForestValueModel.load_joblib(f"{args.models_dir}/pegging_rf.pkl")
        return LinearValueModel.load_npz(f"{args.models_dir}/pegging_linear.npz")

    model_tag = resolve_model_tag()
    def player_factory(name: str):
        if name == "NeuralClassificationPlayer":
            pegging_model = _load_pegging_model()
            discard_model = LinearDiscardClassifier.load_npz(f"{args.models_dir}/discard_linear.npz")
            return NeuralClassificationPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "AIPlayer":
            pegging_model = _load_pegging_model()
            discard_model = _load_discard_model()
            return AIPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "MLPPlayer":
            if model_type != "mlp":
                raise ValueError(f"MLPPlayer requires model_type=mlp, got {model_type}.")
            pegging_model = _load_pegging_model()
            discard_model = _load_discard_model()
            return MLPPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "GBTPlayer":
            if model_type != "gbt":
                raise ValueError(f"GBTPlayer requires model_type=gbt, got {model_type}.")
            pegging_model = _load_pegging_model()
            discard_model = _load_discard_model()
            return GBTPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "RandomForestPlayer":
            if model_type != "rf":
                raise ValueError(f"RandomForestPlayer requires model_type=rf, got {model_type}.")
            pegging_model = _load_pegging_model()
            discard_model = _load_discard_model()
            return RandomForestPlayer(
                discard_model,
                pegging_model,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
                pegging_feature_set=args.pegging_feature_set,
            )
        if name == "NeuralDiscardOnlyPlayer":
            discard_model = _load_discard_model()
            fallback = base_player_factory(fallback_player_name)
            return NeuralDiscardOnlyPlayer(
                discard_model,
                fallback,
                name=f"{name}:{model_tag}",
                discard_feature_set=args.discard_feature_set,
            )
        if name == "NeuralPegOnlyPlayer":
            pegging_model = _load_pegging_model()
            fallback = base_player_factory(fallback_player_name)
            return NeuralPegOnlyPlayer(
                pegging_model,
                fallback,
                name=f"{name}:{model_tag}",
                pegging_feature_set=args.pegging_feature_set,
            )
        return base_player_factory(name)
    
    return player_factory, model_tag, discard_model_type, size_suffix


def _read_training_games_from_meta(models_dir: str) -> tuple[int | str, int | str, int | str]:
    meta_path = Path(models_dir) / "model_meta.json"
    if not meta_path.exists():
        raise ValueError(f"Expected model_meta.json at {meta_path} but it does not exist.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "discard_games_used" in meta:
        discard_games = int(meta["discard_games_used"])
    else:
        discard_games = "unknown"
    if "pegging_games_used" in meta:
        pegging_games = int(meta["pegging_games_used"])
    else:
        pegging_games = "unknown"
    if "training_games_used" in meta:
        training_games = int(meta["training_games_used"])
    else:
        training_games = "unknown"
    return discard_games, pegging_games, training_games


def _compute_win_ci(winrate: float, total_games: int) -> tuple[float, float]:
    if total_games <= 1:
        return winrate, winrate
    se = float(np.sqrt(winrate * (1.0 - winrate) / total_games))
    lo = max(0.0, winrate - 1.96 * se)
    hi = min(1.0, winrate + 1.96 * se)
    return lo, hi


def _benchmark_single(
    args,
    players_override: str | None = None,
    fallback_override: str | None = None,
    games_override: int | None = None,
) -> dict:
    player_factory, model_tag, model_type, size_suffix = _build_player_factory(args, fallback_override)

    players_value = players_override or args.players
    player_names = [p.strip() for p in players_value.split(",") if p.strip()]
    if len(player_names) != 2:
        raise ValueError("Must specify exactly two players via --players")

    p0 = player_factory(player_names[0])
    p1 = player_factory(player_names[1])
    games_to_play = games_override or args.benchmark_games

    results = play_multiple_games(
        games_to_play,
        p0=p0,
        p1=p1,
        seed=args.seed,
        fast_mode=True,
        copy_players=False,
    )
    wins = results["wins"]
    diffs = results["diffs"]
    winrate = results["winrate"]
    win_ci_lo = results.get("ci_lo", winrate)
    win_ci_hi = results.get("ci_hi", winrate)
    diff_ci_lo = results.get("diff_ci_lo")
    diff_ci_hi = results.get("diff_ci_hi")

    is_neural = any(
        name in {"AIPlayer", "MLPPlayer", "GBTPlayer", "RandomForestPlayer", "NeuralDiscardOnlyPlayer", "NeuralPegOnlyPlayer"}
        for name in player_names
    )
    if is_neural:
        discard_games_used, pegging_games_used, estimated_training_games = _read_training_games_from_meta(args.models_dir)
        logger.debug(
            "Training games from model_meta: discard=%s pegging=%s min=%s",
            discard_games_used,
            pegging_games_used,
            estimated_training_games,
        )
    else:
        discard_games_used = 0
        pegging_games_used = 0
        estimated_training_games = 0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    if diff_ci_lo is None or diff_ci_hi is None:
        if diffs and len(diffs) > 1:
            std_diff = float(np.std(diffs, ddof=1))
            se_diff = std_diff / float(np.sqrt(len(diffs)))
            diff_ci_lo = avg_diff - 1.96 * se_diff
            diff_ci_hi = avg_diff + 1.96 * se_diff
        else:
            diff_ci_lo = avg_diff
            diff_ci_hi = avg_diff
    display_names = []
    if model_type == "mlp":
        model_prefix = "MLP"
    elif model_type == "gbt":
        model_prefix = "GBT"
    elif model_type == "rf":
        model_prefix = "RF"
    elif model_type == "gru":
        model_prefix = "GRU"
    elif model_type == "lstm":
        model_prefix = "LSTM"
    else:
        model_prefix = "Linear"
    for name in player_names:
        if name in {
            "AIPlayer",
            "MLPPlayer",
            "GBTPlayer",
            "RandomForestPlayer",
            "NeuralDiscardOnlyPlayer",
            "NeuralPegOnlyPlayer",
        }:
            tag = model_tag
            # Prefer version like "discard_v6-008" -> V6.008 (for linear) or V6 (for mlp)
            version_digits = []
            if "discard_v" in tag:
                try:
                    v_part = tag.split("discard_v", 1)[1]
                    v_num = "".join(c for c in v_part.split("-", 1)[0] if c.isdigit())
                    run_num = ""
                    if "-" in tag:
                        run_num = "".join(c for c in tag.split("-", 1)[1] if c.isdigit())
                    if v_num:
                        version_digits.append(f"V{v_num}")
                    if run_num and model_type != "mlp":
                        version_digits.append(run_num)
                except Exception:
                    version_digits = []
            if version_digits:
                if model_type == "mlp":
                    label = f"{model_prefix}{''.join(version_digits)}{size_suffix}"
                else:
                    label = f"{model_prefix}V{'.'.join(version_digits[0:])}Player"
            else:
                version_label = "".join(c for c in tag if c.isdigit())
                if version_label:
                    if model_type == "mlp":
                        label = f"{model_prefix}V{version_label}{size_suffix}"
                    else:
                        label = f"{model_prefix}V{version_label}Player"
                else:
                    if model_type == "mlp":
                        label = f"{model_prefix}{tag.capitalize()}{size_suffix}"
                    else:
                        label = f"{model_prefix}{tag.capitalize()}Player"
            if name == "NeuralDiscardOnlyPlayer":
                label = f"{label}-DiscardOnly"
            elif name == "NeuralPegOnlyPlayer":
                label = f"{label}-PegOnly"
            display_names.append(label)
        else:
            display_names.append(name)
    return {
        "wins": wins,
        "diffs": diffs,
        "winrate": winrate,
        "win_ci_lo": win_ci_lo,
        "win_ci_hi": win_ci_hi,
        "diff_ci_lo": diff_ci_lo,
        "diff_ci_hi": diff_ci_hi,
        "avg_diff": avg_diff,
        "player_names": player_names,
        "display_names": display_names,
        "model_type": model_type,
        "model_tag": model_tag,
        "models_dir": args.models_dir,
        "estimated_training_games": estimated_training_games,
        "discard_games_used": discard_games_used,
        "pegging_games_used": pegging_games_used,
        "discard_feature_set": args.discard_feature_set,
        "pegging_feature_set": args.pegging_feature_set,
        "seed": args.seed,
        "games_to_play": games_to_play,
    }


def _benchmark_worker(args_tuple) -> dict:
    args_dict, players_override, fallback_override, games_override, task_id = args_tuple
    args = SimpleNamespace(**args_dict)
    base_seed = args.seed or 0
    args.seed = int(base_seed) + task_id
    result = _benchmark_single(args, players_override, fallback_override, games_override)
    result["worker_id"] = task_id
    return result


def benchmark_2_players(
    args,
    players_override: str | None = None,
    fallback_override: str | None = None,
) -> int:
    if args.seed is None:
        args.seed = 67
    if args.benchmark_workers < 1:
        raise ValueError("benchmark_workers must be >= 1.")
    if args.max_buffer_games is None or args.max_buffer_games <= 0:
        raise ValueError("max_buffer_games must be > 0.")

    total_games = args.benchmark_games
    if total_games <= 0:
        raise ValueError("--benchmark_games must be > 0.")

    if args.benchmark_workers > 1 or total_games > args.max_buffer_games:
        chunk_size = min(args.max_buffer_games, total_games)
        tasks = []
        remaining = total_games
        while remaining > 0:
            chunk_games = min(chunk_size, remaining)
            tasks.append(chunk_games)
            remaining -= chunk_games
        workers = min(args.benchmark_workers, len(tasks))
        if total_games < workers:
            workers = total_games
        if workers <= 0:
            raise ValueError("--benchmark_workers must be >= 1 for the requested games.")

        if len(set(tasks)) == 1:
            task_summary = f"{tasks[0]} per task"
        else:
            task_summary = ",".join(str(g) for g in tasks) + " per task"
        logger.info(
            "Benchmarking with %d workers for %d total games (%s)",
            workers,
            total_games,
            task_summary,
        )

        args_dict = vars(args).copy()
        ctx = mp.get_context("spawn")
        results = []
        with ctx.Pool(processes=workers) as pool:
            for idx, result in enumerate(
                pool.imap_unordered(
                    _benchmark_worker,
                    [
                        (args_dict, players_override, fallback_override, tasks[task_id], task_id)
                        for task_id in range(len(tasks))
                    ],
                ),
                start=1,
            ):
                results.append(result)
                logger.debug(
                    "Benchmark worker %s finished (%d/%d)",
                    str(result.get("worker_id")),
                    idx,
                    len(tasks),
                )

        wins = int(sum(r["wins"] for r in results))
        diffs = []
        for r in results:
            diffs.extend(r["diffs"])
        winrate = float(wins) / float(total_games) if total_games else 0.0
        win_ci_lo, win_ci_hi = _compute_win_ci(winrate, total_games)
        avg_diff = float(np.mean(diffs)) if diffs else 0.0
        if diffs and len(diffs) > 1:
            std_diff = float(np.std(diffs, ddof=1))
            se_diff = std_diff / float(np.sqrt(len(diffs)))
            diff_ci_lo = avg_diff - 1.96 * se_diff
            diff_ci_hi = avg_diff + 1.96 * se_diff
        else:
            diff_ci_lo = avg_diff
            diff_ci_hi = avg_diff

        first = results[0]
        player_names = first["player_names"]
        display_names = first["display_names"]
        model_type = first["model_type"]
        model_tag = first["model_tag"]
        model_dir_label = os.path.basename(os.path.normpath(first["models_dir"]))
        if model_tag and "-" in model_tag:
            model_dir_label = model_tag.split("-", 1)[1]
        estimated_training_games = first["estimated_training_games"]
        discard_games_used = first.get("discard_games_used", 0)
        pegging_games_used = first.get("pegging_games_used", 0)
        discard_feature_set = first["discard_feature_set"]
        pegging_feature_set = first["pegging_feature_set"]

        is_neural = any(name in {"AIPlayer", "MLPPlayer", "GBTPlayer", "RandomForestPlayer"} for name in player_names)
        if is_neural:
            training_label = (
                estimated_training_games
                if isinstance(estimated_training_games, str)
                else str(estimated_training_games)
            )
            output_str = (
                f"{display_names[0]}[{model_dir_label}] vs {display_names[1]} after {training_label} training games "
                f"avg point diff {avg_diff:.2f} (95% CI {diff_ci_lo:.2f} - {diff_ci_hi:.2f}) "
                f"wins={wins}/{total_games} winrate={winrate*100:.2f}% "
                f"(95% CI {win_ci_lo*100:.2f}% - {win_ci_hi*100:.2f}%)\n"
            )
        else:
            output_str = (
                f"{display_names[0]} vs {display_names[1]} "
                f"avg point diff {avg_diff:.2f} (95% CI {diff_ci_lo:.2f} - {diff_ci_hi:.2f}) "
                f"wins={wins}/{total_games} winrate={winrate*100:.2f}% "
                f"(95% CI {win_ci_lo*100:.2f}% - {win_ci_hi*100:.2f}%)\n"
            )
        output_path = getattr(args, "benchmark_output_path", None) or "text/benchmark_results.txt"
        if getattr(args, "no_benchmark_write", False):
            logger.info("Skipping text/benchmark_results.txt write (no_benchmark_write=True).")
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a") as f:
                f.write(output_str)
        print(output_str)

        experiment = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "players": player_names,
            "display_players": display_names,
            "models_dir": first["models_dir"],
            "benchmark_games": total_games,
            "wins": wins,
            "winrate": winrate,
            "avg_point_diff": avg_diff,
            "avg_point_diff_ci_lo": diff_ci_lo,
            "avg_point_diff_ci_hi": diff_ci_hi,
            "winrate_ci_lo": win_ci_lo,
            "winrate_ci_hi": win_ci_hi,
            "estimated_training_games": estimated_training_games,
            "discard_games_used": discard_games_used,
            "pegging_games_used": pegging_games_used,
            "discard_feature_set": discard_feature_set,
            "pegging_feature_set": pegging_feature_set,
            "model_tag": model_tag,
            "seed": args.seed,
        }
        experiments_path = getattr(args, "experiments_output_path", None) or "text/experiments.jsonl"
        if getattr(args, "no_benchmark_write", False):
            logger.info("Skipping text/experiments.jsonl write (no_benchmark_write=True).")
        else:
            Path(experiments_path).parent.mkdir(parents=True, exist_ok=True)
            with open(experiments_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(experiment) + "\n")
            logger.info(f"Appended experiment -> {experiments_path}")
        return 0

    if total_games < args.benchmark_workers:
        raise ValueError("benchmark_games is smaller than benchmark_workers. Reduce workers or increase games.")

    single = _benchmark_single(args, players_override, fallback_override)
    model_dir_label = os.path.basename(os.path.normpath(single["models_dir"]))
    if single["model_tag"] and "-" in single["model_tag"]:
        model_dir_label = single["model_tag"].split("-", 1)[1]
    is_neural = any(
        name in {"AIPlayer", "MLPPlayer", "GBTPlayer", "RandomForestPlayer", "NeuralDiscardOnlyPlayer", "NeuralPegOnlyPlayer"}
        for name in single["player_names"]
    )
    if is_neural:
        training_label = (
            single["estimated_training_games"]
            if isinstance(single["estimated_training_games"], str)
            else str(single["estimated_training_games"])
        )
        output_str = (
            f"{single['display_names'][0]}[{model_dir_label}] vs {single['display_names'][1]} "
            f"after {training_label} training games "
            f"avg point diff {single['avg_diff']:.2f} (95% CI {single['diff_ci_lo']:.2f} - {single['diff_ci_hi']:.2f}) "
            f"wins={single['wins']}/{single['games_to_play']} winrate={single['winrate']*100:.2f}% "
            f"(95% CI {single['win_ci_lo']*100:.2f}% - {single['win_ci_hi']*100:.2f}%)\n"
        )
    else:
        output_str = (
            f"{single['display_names'][0]} vs {single['display_names'][1]} "
            f"avg point diff {single['avg_diff']:.2f} (95% CI {single['diff_ci_lo']:.2f} - {single['diff_ci_hi']:.2f}) "
            f"wins={single['wins']}/{single['games_to_play']} winrate={single['winrate']*100:.2f}% "
            f"(95% CI {single['win_ci_lo']*100:.2f}% - {single['win_ci_hi']*100:.2f}%)\n"
        )
    output_path = getattr(args, "benchmark_output_path", None) or "text/benchmark_results.txt"
    if getattr(args, "no_benchmark_write", False):
        logger.info("Skipping text/benchmark_results.txt write (no_benchmark_write=True).")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(output_str)
    print(output_str)

    experiment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "players": single["player_names"],
        "display_players": single["display_names"],
        "models_dir": single["models_dir"],
        "benchmark_games": single["games_to_play"],
        "wins": single["wins"],
        "winrate": single["winrate"],
        "avg_point_diff": single["avg_diff"],
        "avg_point_diff_ci_lo": single["diff_ci_lo"],
        "avg_point_diff_ci_hi": single["diff_ci_hi"],
        "winrate_ci_lo": single["win_ci_lo"],
        "winrate_ci_hi": single["win_ci_hi"],
        "estimated_training_games": single["estimated_training_games"],
        "discard_games_used": single.get("discard_games_used", 0),
        "pegging_games_used": single.get("pegging_games_used", 0),
        "discard_feature_set": single["discard_feature_set"],
        "pegging_feature_set": single["pegging_feature_set"],
        "model_tag": single["model_tag"],
        "seed": single["seed"],
    }
    experiments_path = getattr(args, "experiments_output_path", None) or "text/experiments.jsonl"
    if getattr(args, "no_benchmark_write", False):
        logger.info("Skipping text/experiments.jsonl write (no_benchmark_write=True).")
    else:
        Path(experiments_path).parent.mkdir(parents=True, exist_ok=True)
        with open(experiments_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment) + "\n")
        logger.info(f"Appended experiment -> {experiments_path}")
    return 0


if __name__ == "__main__":
    args = build_benchmark_parser().parse_args()
    logger.info(f"models dir: {args.models_dir}")
    if args.queue_models and args.queue_file:
        raise SystemExit("--queue_models and --queue_file cannot both be set.")
    queue = []
    if args.queue_models:
        queue = [p.strip() for p in args.queue_models.split(",") if p.strip()]
        if not queue:
            raise SystemExit("--queue_models must include at least one models_dir.")
    elif args.queue_file:
        queue_path = Path(args.queue_file)
        if not queue_path.exists():
            raise SystemExit(f"--queue_file not found: {queue_path}")
        queue = [line.strip() for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not queue:
            raise SystemExit("--queue_file did not contain any models_dir entries.")
    else:
        queue = [args.models_dir]

    for idx, models_dir in enumerate(queue, start=1):
        if len(queue) > 1:
            logger.info(f"Queue {idx}/{len(queue)}: {models_dir}")
        args.models_dir = models_dir
        if not args.only_mixed_benchmarks:
            benchmark_2_players(args)
        if args.auto_mixed_benchmarks or args.only_mixed_benchmarks:
            logger.info("Running discard-only and pegging-only benchmarks vs beginner fallback")
            benchmark_2_players(
                args,
                players_override="NeuralDiscardOnlyPlayer,beginner",
                fallback_override="beginner",
            )
            benchmark_2_players(
                args,
                players_override="NeuralPegOnlyPlayer,beginner",
                fallback_override="beginner",
            )

# python scripts/benchmark_2_players.py
# python scripts/benchmark_2_players.py --players AIPlayer,beginner --games 200 --models_dir "models/ranking" --max_shards 6 --fallback_player beginner

# .\.venv\Scripts\python.exe .\scripts\generate_il_data.py --games 4000 --out_dir "datasets" --dataset_version "discard_v2" --strategy regression

# .\.venv\Scripts\python.exe .\scripts\train_models.py --data_dir "datasets\discard_v3" --models_dir "models" --model_version "discard_v3" --run_id 002 --discard_loss regression --epochs 5 --eval_samples 2048 --lr 0.00005 --l2 0.001 --batch_size 2048

# .\.venv\Scripts\python.exe .\scripts\benchmark_2_players.py --players AIPlayer,beginner --benchmark_games 200 --models_dir "models\regression\"
# .\.venv\Scripts\python.exe .\scripts\benchmark_2_players.py

# Script summary: benchmark two players and log results to text/jsonl outputs.
