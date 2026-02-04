import os
from dotenv import load_dotenv

load_dotenv(override=True)

RANK_VALUE = {**{i: i for i in range(1, 10)}, 10: 10, 11: 10, 12: 10, 13: 10}


def _getenv_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


TRAINING_DATA_DIR = _getenv_str("TRAINING_DATA_DIR", "il_datasets")
MODELS_DIR = _getenv_str("MODELS_DIR", "models")

# Shared defaults for scripts
DEFAULT_DATASET_VERSION = _getenv_str("DATASET_VERSION", "discard_v3")
DEFAULT_DATASET_RUN_ID = _getenv_str("DATASET_RUN_ID", "")
DEFAULT_MODEL_VERSION = _getenv_str("MODEL_VERSION", "discard_v3")
DEFAULT_MODEL_RUN_ID = _getenv_str("MODEL_RUN_ID", "")
DEFAULT_STRATEGY = _getenv_str("STRATEGY", "regression")
DEFAULT_DISCARD_LOSS = _getenv_str("DISCARD_LOSS", "regression")
DEFAULT_PEGGING_FEATURE_SET = _getenv_str("PEGGING_FEATURE_SET", "full")
DEFAULT_DISCARD_FEATURE_SET = _getenv_str("DISCARD_FEATURE_SET", "full")
DEFAULT_PEGGING_MODEL_FEATURE_SET = _getenv_str("PEGGING_MODEL_FEATURE_SET", "full")
DEFAULT_MODEL_TYPE = _getenv_str("MODEL_TYPE", "linear")
DEFAULT_MLP_HIDDEN = _getenv_str("MLP_HIDDEN", "128,64")
DEFAULT_CRIB_EV_MODE = _getenv_str("CRIB_EV_MODE", "mc")
DEFAULT_CRIB_MC_SAMPLES = _getenv_int("CRIB_MC_SAMPLES", 32)
DEFAULT_PEGGING_LABEL_MODE = _getenv_str("PEGGING_LABEL_MODE", "rollout2")
DEFAULT_PEGGING_ROLLOUTS = _getenv_int("PEGGING_ROLLOUTS", 32)

DEFAULT_GAMES_PER_LOOP = _getenv_int("GAMES_PER_LOOP", 2000)
DEFAULT_LOOPS = _getenv_int("LOOPS", -1)
DEFAULT_EPOCHS = _getenv_int("EPOCHS", 5)
DEFAULT_BENCHMARK_GAMES = _getenv_int("BENCHMARK_GAMES", 200)
DEFAULT_EVAL_SAMPLES = _getenv_int("EVAL_SAMPLES", 2048)
DEFAULT_MAX_SHARDS = _getenv_int("MAX_SHARDS", 0)
DEFAULT_RANK_PAIRS_PER_HAND = _getenv_int("RANK_PAIRS_PER_HAND", 20)

DEFAULT_LR = _getenv_float("LR", 0.00005)
DEFAULT_L2 = _getenv_float("L2", 0.001)
DEFAULT_BATCH_SIZE = _getenv_int("BATCH_SIZE", 2048)

DEFAULT_BENCHMARK_PLAYERS = _getenv_str("BENCHMARK_PLAYERS", "NeuralRegressionPlayer,medium")
DEFAULT_FALLBACK_PLAYER = _getenv_str("FALLBACK_PLAYER", "medium")
DEFAULT_MODEL_TAG = _getenv_str("MODEL_TAG", "")

DEFAULT_SEED = _getenv_int("SEED", 0)
DEFAULT_USE_RANDOM_SEED = _getenv_bool("USE_RANDOM_SEED", True)
