import argparse

from crib_ai_trainer.constants import (
    MODELS_DIR,
    TRAINING_DATA_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODEL_RUN_ID,
    DEFAULT_STRATEGY,
    DEFAULT_DISCARD_LOSS,
    DEFAULT_PEGGING_FEATURE_SET,
    DEFAULT_DISCARD_FEATURE_SET,
    DEFAULT_PEGGING_MODEL_FEATURE_SET,
    DEFAULT_PEGGING_DATA_DIR,
    DEFAULT_GAMES_PER_LOOP,
    DEFAULT_LOOPS,
    DEFAULT_IL_WORKERS,
    DEFAULT_IL_GAMES_PER_WORKER,
    DEFAULT_EPOCHS,
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_BENCHMARK_WORKERS,
    DEFAULT_BENCHMARK_GAMES_PER_WORKER,
    DEFAULT_BENCHMARK_PLAYERS,
    DEFAULT_LR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_L2,
    DEFAULT_MAX_SHARDS,
    DEFAULT_FALLBACK_PLAYER,
    DEFAULT_RANK_PAIRS_PER_HAND,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_MODEL_TAG,
    DEFAULT_SEED,
    DEFAULT_USE_RANDOM_SEED,
    DEFAULT_MODEL_TYPE,
    DEFAULT_MLP_HIDDEN,
    DEFAULT_CRIB_EV_MODE,
    DEFAULT_CRIB_MC_SAMPLES,
    DEFAULT_PEGGING_LABEL_MODE,
    DEFAULT_PEGGING_ROLLOUTS,
    DEFAULT_PEGGING_EV_MODE,
    DEFAULT_PEGGING_EV_ROLLOUTS,
    DEFAULT_WIN_PROB_MODE,
    DEFAULT_WIN_PROB_ROLLOUTS,
    DEFAULT_WIN_PROB_MIN_SCORE,
    DEFAULT_MAX_BUFFER_GAMES,
)


def _default_seed() -> int | None:
    return None if DEFAULT_USE_RANDOM_SEED else DEFAULT_SEED


def coerce_int_args(args: argparse.Namespace, names: list[str]) -> None:
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            continue
        if isinstance(value, int):
            continue
        try:
            setattr(args, name, int(value))
        except Exception as exc:
            raise ValueError(f"{name} must be an int, got {value!r}") from exc


def build_generate_il_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES_PER_LOOP,
        help="Number of games to simulate. Use -1 to run forever.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_IL_WORKERS,
        help="Number of worker processes for IL generation.",
    )
    ap.add_argument("--out_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--seed", type=int, default=_default_seed(), help="Random seed. Omit to use a random seed.")
    ap.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    ap.add_argument(
        "--teacher_player",
        type=str,
        default="hard",
        choices=["medium", "hard"],
        help="Which teacher player to use for IL generation.",
    )
    ap.add_argument(
        "--skip_pegging_data",
        action="store_true",
        help="Generate only discard data (skip pegging logging and files).",
    )
    ap.add_argument(
        "--pegging_feature_set",
        type=str,
        default=DEFAULT_PEGGING_FEATURE_SET,
        choices=["basic", "full"],
        help="Which pegging feature set to use.",
    )
    ap.add_argument(
        "--crib_ev_mode",
        type=str,
        default=DEFAULT_CRIB_EV_MODE,
        choices=["min", "mc"],
        help="How to estimate crib EV for discard labels.",
    )
    ap.add_argument(
        "--crib_mc_samples",
        type=int,
        default=DEFAULT_CRIB_MC_SAMPLES,
        help="Number of Monte Carlo samples for crib EV when crib_ev_mode=mc.",
    )
    ap.add_argument(
        "--pegging_label_mode",
        type=str,
        default=DEFAULT_PEGGING_LABEL_MODE,
        choices=["immediate", "rollout1", "rollout2"],
        help="Pegging label mode.",
    )
    ap.add_argument(
        "--pegging_rollouts",
        type=int,
        default=DEFAULT_PEGGING_ROLLOUTS,
        help="Number of rollouts for pegging_label_mode=rollout1.",
    )
    ap.add_argument(
        "--win_prob_mode",
        type=str,
        default=DEFAULT_WIN_PROB_MODE,
        choices=["off", "rollout"],
        help="Win-probability label mode for discard (rollout or off).",
    )
    ap.add_argument(
        "--win_prob_rollouts",
        type=int,
        default=DEFAULT_WIN_PROB_ROLLOUTS,
        help="Number of rollouts for win-probability estimation.",
    )
    ap.add_argument(
        "--win_prob_min_score",
        type=int,
        default=DEFAULT_WIN_PROB_MIN_SCORE,
        help="Only estimate win-prob when max(score) >= this threshold (else label 0.5).",
    )
    ap.add_argument(
        "--pegging_ev_mode",
        type=str,
        default=DEFAULT_PEGGING_EV_MODE,
        choices=["off", "rollout"],
        help="Whether to add pegging EV features to discard (rollout or off).",
    )
    ap.add_argument(
        "--pegging_ev_rollouts",
        type=int,
        default=DEFAULT_PEGGING_EV_ROLLOUTS,
        help="Number of rollouts for pegging EV estimation.",
    )
    ap.add_argument(
        "--max_buffer_games",
        type=int,
        default=DEFAULT_MAX_BUFFER_GAMES,
        help="Max games to buffer in memory before saving when using multi-worker IL generation.",
    )
    return ap


def build_generate_self_play_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_LOOP)
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes for self-play generation.")
    ap.add_argument("--out_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--models_dir", type=str, required=True)
    ap.add_argument("--model_version", type=str, default=None)
    ap.add_argument("--model_run_id", type=str, default=None)
    ap.add_argument("--opponent_models_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_FEATURE_SET, choices=["basic", "full"])
    ap.add_argument("--crib_ev_mode", type=str, default=DEFAULT_CRIB_EV_MODE, choices=["min", "mc"])
    ap.add_argument("--crib_mc_samples", type=int, default=DEFAULT_CRIB_MC_SAMPLES)
    ap.add_argument("--pegging_label_mode", type=str, default=DEFAULT_PEGGING_LABEL_MODE, choices=["immediate", "rollout1", "rollout2"])
    ap.add_argument("--pegging_rollouts", type=int, default=DEFAULT_PEGGING_ROLLOUTS)
    ap.add_argument("--pegging_ev_mode", type=str, default=DEFAULT_PEGGING_EV_MODE, choices=["off", "rollout"])
    ap.add_argument("--pegging_ev_rollouts", type=int, default=DEFAULT_PEGGING_EV_ROLLOUTS)
    ap.add_argument("--win_prob_mode", type=str, default=DEFAULT_WIN_PROB_MODE, choices=["off", "rollout"])
    ap.add_argument("--win_prob_rollouts", type=int, default=DEFAULT_WIN_PROB_ROLLOUTS)
    ap.add_argument("--win_prob_min_score", type=int, default=DEFAULT_WIN_PROB_MIN_SCORE)
    ap.add_argument("--max_buffer_games", type=int, default=DEFAULT_MAX_BUFFER_GAMES)
    return ap


def build_benchmark_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=str, default=DEFAULT_BENCHMARK_PLAYERS)
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument(
        "--benchmark_workers",
        type=int,
        default=DEFAULT_BENCHMARK_WORKERS,
        help="Number of worker processes for benchmarking.",
    )
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR, help="Base models dir or explicit run dir")
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--model_run_id", type=str, default=DEFAULT_MODEL_RUN_ID or None, help="Explicit run id (e.g., 014)")
    ap.add_argument("--latest_model", action="store_true", default=True)
    ap.add_argument("--no_latest_model", dest="latest_model", action="store_false")
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--fallback_player", type=str, default=DEFAULT_FALLBACK_PLAYER)
    ap.add_argument("--model_tag", type=str, default=DEFAULT_MODEL_TAG or None)
    ap.add_argument(
        "--discard_feature_set",
        type=str,
        default=DEFAULT_DISCARD_FEATURE_SET,
        choices=["base", "engineered_no_scores", "engineered_no_scores_pev", "full", "full_pev"],
    )
    ap.add_argument(
        "--pegging_feature_set",
        type=str,
        default=DEFAULT_PEGGING_MODEL_FEATURE_SET,
        choices=["base", "full_no_scores", "full", "full_seq"],
    )
    ap.add_argument("--auto_mixed_benchmarks", action="store_true", default=True)
    ap.add_argument("--no_auto_mixed_benchmarks", dest="auto_mixed_benchmarks", action="store_false")
    ap.add_argument("--only_mixed_benchmarks", action="store_true", default=False, help="Run only discard/pegging-only benchmarks.")
    ap.add_argument("--max_buffer_games", type=int, default=DEFAULT_MAX_BUFFER_GAMES)
    ap.add_argument("--benchmark_output_path", type=str, default=None)
    ap.add_argument("--experiments_output_path", type=str, default=None)
    ap.add_argument("--queue_models", type=str, default=None, help="Comma-separated list of model dirs to benchmark.")
    ap.add_argument("--queue_file", type=str, default=None, help="Path to a file with one models_dir per line.")
    return ap


def build_do_everything_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke cycle (loops=1, games=1) and skip text/benchmark_results.txt writes.",
    )
    ap.add_argument("--il_games", type=int, default=DEFAULT_GAMES_PER_LOOP, help="Games per loop for IL data generation")
    ap.add_argument("--il_workers", type=int, default=DEFAULT_IL_WORKERS, help="Workers for IL data generation")
    ap.add_argument(
        "--loops",
        type=int,
        default=DEFAULT_LOOPS,
        help="Number of generate->train->benchmark cycles.",
    )
    ap.add_argument("--training_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--pegging_data_dir", type=str, default=DEFAULT_PEGGING_DATA_DIR, help="Dataset dir for pegging shards.")
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_FEATURE_SET, choices=["basic", "full"])
    ap.add_argument("--seed", type=int, default=_default_seed())
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--players", type=str, default=DEFAULT_BENCHMARK_PLAYERS)
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument("--benchmark_workers", type=int, default=DEFAULT_BENCHMARK_WORKERS)
    ap.add_argument(
        "--benchmark_mode",
        type=str,
        default="single",
        choices=["all", "single"],
        help="Benchmark all three (full/discard/pegging) or only the combined model.",
    )
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--model_run_id", type=str, default=DEFAULT_MODEL_RUN_ID or None)
    ap.add_argument("--discard_loss", type=str, default=DEFAULT_DISCARD_LOSS, choices=["classification", "regression", "ranking"])
    ap.add_argument(
        "--discard_feature_set",
        type=str,
        default=DEFAULT_DISCARD_FEATURE_SET,
        choices=["base", "engineered_no_scores", "engineered_no_scores_pev", "full", "full_pev"],
    )
    ap.add_argument("--pegging_model_feature_set", type=str, default=DEFAULT_PEGGING_MODEL_FEATURE_SET, choices=["base", "full_no_scores", "full"])
    ap.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["linear", "mlp", "gbt", "rf"])
    ap.add_argument("--mlp_hidden", type=str, default=DEFAULT_MLP_HIDDEN)
    ap.add_argument("--discard_mlp_hidden", type=str, default=None, help="Override MLP sizes for discard head only.")
    ap.add_argument("--pegging_mlp_hidden", type=str, default=None, help="Override MLP sizes for pegging head only.")
    ap.add_argument("--crib_ev_mode", type=str, default=DEFAULT_CRIB_EV_MODE, choices=["min", "mc"])
    ap.add_argument("--crib_mc_samples", type=int, default=DEFAULT_CRIB_MC_SAMPLES)
    ap.add_argument("--pegging_label_mode", type=str, default=DEFAULT_PEGGING_LABEL_MODE, choices=["immediate", "rollout1"])
    ap.add_argument("--pegging_rollouts", type=int, default=DEFAULT_PEGGING_ROLLOUTS)
    ap.add_argument("--pegging_ev_mode", type=str, default=DEFAULT_PEGGING_EV_MODE, choices=["off", "rollout"])
    ap.add_argument("--pegging_ev_rollouts", type=int, default=DEFAULT_PEGGING_EV_ROLLOUTS)
    ap.add_argument("--win_prob_mode", type=str, default=DEFAULT_WIN_PROB_MODE, choices=["off", "rollout"])
    ap.add_argument("--win_prob_rollouts", type=int, default=DEFAULT_WIN_PROB_ROLLOUTS)
    ap.add_argument("--win_prob_min_score", type=int, default=DEFAULT_WIN_PROB_MIN_SCORE)
    ap.add_argument("--max_buffer_games", type=int, default=DEFAULT_MAX_BUFFER_GAMES)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--l2", type=float, default=DEFAULT_L2)
    ap.add_argument("--torch_threads", type=int, default=8, help="Torch CPU thread count (intra/inter-op).")
    ap.add_argument(
        "--parallel_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train discard and pegging heads in parallel.",
    )
    ap.add_argument("--max_shards", type=int, default=(DEFAULT_MAX_SHARDS or None))
    ap.add_argument("--fallback_player", type=str, default=DEFAULT_FALLBACK_PLAYER)
    ap.add_argument("--rank_pairs_per_hand", type=int, default=DEFAULT_RANK_PAIRS_PER_HAND)
    ap.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--model_tag", type=str, default=DEFAULT_MODEL_TAG or None)
    ap.add_argument("--extra_data_dir", type=str, default=None, help="Optional extra dataset to mix in (e.g., self-play).")
    ap.add_argument("--extra_ratio", type=float, default=0.0, help="Fraction of batches to sample from extra_data_dir.")
    ap.add_argument("--incremental", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--incremental_from", type=str, default=None)
    ap.add_argument("--incremental_start_shard", type=int, default=0)
    ap.add_argument("--incremental_epochs", type=int, default=None)
    ap.add_argument(
        "--skip_pegging_data",
        action="store_true",
        default=False,
        help="Generate only discard data (skip pegging logging and files).",
    )
    return ap


def build_self_play_loop_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--teacher_dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--selfplay_dataset_version", type=str, default=None)
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_LOOP)
    ap.add_argument("--selfplay_workers", type=int, default=1, help="Worker processes for self-play generation.")
    ap.add_argument("--benchmark_games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    ap.add_argument("--benchmark_workers", type=int, default=DEFAULT_BENCHMARK_WORKERS)
    ap.add_argument("--benchmark_seed", type=int, default=None, help="Seed used for self-play benchmarks (random if omitted).")
    ap.add_argument("--benchmark_opponent", type=str, default="medium", choices=["medium", "beginner", "hard"])
    ap.add_argument("--selfplay_ratio", type=float, default=0.3)
    ap.add_argument("--max_no_improve", type=int, default=3, help="Stop after this many non-improving loops.")
    ap.add_argument("--best_file", type=str, default=None)
    ap.add_argument("--loops", type=int, default=-1)
    ap.add_argument(
        "--discard_feature_set",
        type=str,
        default=DEFAULT_DISCARD_FEATURE_SET,
        choices=["base", "engineered_no_scores", "engineered_no_scores_pev", "full", "full_pev"],
    )
    ap.add_argument(
        "--pegging_feature_set",
        type=str,
        default=DEFAULT_PEGGING_MODEL_FEATURE_SET,
        choices=["base", "full_no_scores", "full", "full_seq"],
    )
    ap.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["linear", "mlp", "gbt", "rf"])
    ap.add_argument("--mlp_hidden", type=str, default=DEFAULT_MLP_HIDDEN)
    ap.add_argument("--discard_mlp_hidden", type=str, default=None, help="Override MLP sizes for discard head only.")
    ap.add_argument("--pegging_mlp_hidden", type=str, default=None, help="Override MLP sizes for pegging head only.")
    ap.add_argument("--pegging_ev_mode", type=str, default=DEFAULT_PEGGING_EV_MODE, choices=["off", "rollout"])
    ap.add_argument("--pegging_ev_rollouts", type=int, default=DEFAULT_PEGGING_EV_ROLLOUTS)
    ap.add_argument("--win_prob_mode", type=str, default=DEFAULT_WIN_PROB_MODE, choices=["off", "rollout"])
    ap.add_argument("--win_prob_rollouts", type=int, default=DEFAULT_WIN_PROB_ROLLOUTS)
    ap.add_argument("--win_prob_min_score", type=int, default=DEFAULT_WIN_PROB_MIN_SCORE)
    ap.add_argument("--incremental", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--incremental_epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="Run 1 loop with tiny settings and skip all writes.")
    return ap
