"""Automate generate -> train -> benchmark loops with simple winrate stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import (
    TRAINING_DATA_DIR,
    MODELS_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_MODEL_VERSION,
    DEFAULT_DISCARD_LOSS,
    DEFAULT_DISCARD_FEATURE_SET,
    DEFAULT_PEGGING_MODEL_FEATURE_SET,
    DEFAULT_LR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_L2,
    DEFAULT_SEED,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_MAX_SHARDS,
    DEFAULT_RANK_PAIRS_PER_HAND,
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_BENCHMARK_WORKERS,
    DEFAULT_PEGGING_DATA_DIR,
    DEFAULT_GAMES,
)
from scripts.generate_il_data import generate_il_data, _resolve_output_dir
from scripts.train_models import train_models, _resolve_models_dir
from scripts.benchmark_2_players import benchmark_2_players


def _read_last_jsonl(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Expected benchmark log at {path} but it does not exist.")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"Expected benchmark log entries in {path} but it is empty.")
    return json.loads(lines[-1])


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _log(path: Path, line: str) -> None:
    _append_line(path, line)
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--dataset_run_id", type=str, default=None)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--discard_loss", type=str, default=DEFAULT_DISCARD_LOSS, choices=["classification", "regression", "ranking"])
    ap.add_argument("--discard_feature_set", type=str, default=DEFAULT_DISCARD_FEATURE_SET, choices=["base", "engineered_no_scores", "full"])
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--model_type", type=str, default="mlp", choices=["linear", "mlp", "gbt", "rf"])
    ap.add_argument("--mlp_hidden", type=str, default="128,64")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument("--benchmark_workers", type=int, default=DEFAULT_BENCHMARK_WORKERS)
    ap.add_argument("--benchmark_opponent", type=str, default="beginner", choices=["beginner", "medium"])
    ap.add_argument("--benchmark_seed", type=int, default=67)
    ap.add_argument("--benchmark_output_path", type=str, default="autopilot_benchmark.txt")
    ap.add_argument("--experiments_output_path", type=str, default="autopilot_experiments.jsonl")
    ap.add_argument("--results_output_path", type=str, default="autopilot_results.txt")
    ap.add_argument("--log_path", type=str, default="autopilot_log.txt")
    ap.add_argument("--pegging_data_dir", type=str, default=DEFAULT_PEGGING_DATA_DIR)
    ap.add_argument("--torch_threads", type=int, default=8)
    ap.add_argument("--parallel_heads", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_buffer_games", type=int, default=500)
    ap.add_argument("--loops", type=int, default=20)
    ap.add_argument("--max_no_improve", type=int, default=3)
    ap.add_argument("--generate_il", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--generate_il_each_loop", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--il_games", type=int, default=DEFAULT_GAMES)
    ap.add_argument("--il_workers", type=int, default=10)
    ap.add_argument("--win_prob_mode", type=str, default="off", choices=["off", "rollout"])
    ap.add_argument("--win_prob_rollouts", type=int, default=16)
    ap.add_argument("--win_prob_min_score", type=int, default=90)
    ap.add_argument("--pegging_ev_mode", type=str, default="off", choices=["off", "rollout"])
    ap.add_argument("--pegging_ev_rollouts", type=int, default=16)
    args = ap.parse_args()

    results_path = Path(args.results_output_path)
    experiments_path = Path(args.experiments_output_path)
    log_path = Path(args.log_path)

    best_winrate = None
    no_improve = 0

    dataset_dir = None
    for loop_idx in range(1, args.loops + 1):
        _log(log_path, f"=== Autopilot loop {loop_idx} ===")
        if args.generate_il and (args.generate_il_each_loop or loop_idx == 1):
            dataset_dir = _resolve_output_dir(args.data_dir, args.dataset_version, args.dataset_run_id, new_run=True)
            _log(log_path, f"Generating IL data into {dataset_dir} (games={args.il_games})")
            generate_il_data(
                args.il_games,
                dataset_dir,
                args.seed,
                "regression",
                args.pegging_feature_set,
                "mc",
                32,
                "rollout2",
                32,
                args.win_prob_mode,
                args.win_prob_rollouts,
                args.win_prob_min_score,
                args.pegging_ev_mode,
                args.pegging_ev_rollouts,
                args.il_workers,
                True,
                args.max_buffer_games,
            )
        if dataset_dir is None:
            dataset_dir = _resolve_output_dir(args.data_dir, args.dataset_version, args.dataset_run_id, new_run=False)

        model_dir = _resolve_models_dir(args.models_dir, args.model_version, None)
        _log(log_path, f"Training model in {model_dir}")
        train_args = argparse.Namespace(
            data_dir=dataset_dir,
            extra_data_dir=None,
            extra_ratio=0.0,
            pegging_data_dir=args.pegging_data_dir,
            models_dir=model_dir,
            model_version=args.model_version,
            run_id=None,
            discard_loss=args.discard_loss,
            discard_feature_set=args.discard_feature_set,
            pegging_feature_set=args.pegging_feature_set,
            model_type=args.model_type,
            mlp_hidden=args.mlp_hidden,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            seed=args.seed,
            torch_threads=args.torch_threads,
            parallel_heads=args.parallel_heads,
            eval_samples=args.eval_samples,
            max_shards=args.max_shards,
            rank_pairs_per_hand=args.rank_pairs_per_hand,
        )
        train_models(train_args)

        opponent = "beginner" if args.benchmark_opponent == "beginner" else "medium"
        players = f"AIPlayer,{opponent}"
        _log(log_path, f"Benchmarking {players} for {args.benchmark_games} games (seed={args.benchmark_seed})")
        bench_args = argparse.Namespace(
            players=players,
            benchmark_games=args.benchmark_games,
            benchmark_workers=args.benchmark_workers,
            max_buffer_games=args.max_buffer_games,
            models_dir=model_dir,
            model_version=args.model_version,
            model_run_id=None,
            latest_model=False,
            data_dir=dataset_dir,
            max_shards=args.max_shards,
            seed=args.benchmark_seed,
            fallback_player="beginner",
            model_tag=None,
            discard_feature_set=args.discard_feature_set,
            pegging_feature_set=args.pegging_feature_set,
            auto_mixed_benchmarks=False,
            games=args.benchmark_games,
            benchmark_output_path=args.benchmark_output_path,
            experiments_output_path=str(experiments_path),
        )
        benchmark_2_players(bench_args)
        last = _read_last_jsonl(experiments_path)
        winrate = float(last["winrate"])

        summary = f"loop={loop_idx} model_dir={model_dir} winrate={winrate:.4f}"
        _append_line(results_path, summary)
        _log(log_path, summary)

        if best_winrate is None or winrate > best_winrate:
            best_winrate = winrate
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.max_no_improve:
            _log(log_path, f"Stopping after {no_improve} non-improving loops.")
            break

# Script summary: automate generate/train/benchmark loops and stop after repeated non-improvement.
