import os
from dotenv import load_dotenv

load_dotenv(override=True)

RANK_VALUE = {**{i: i for i in range(1, 10)}, 10: 10, 11: 10, 12: 10, 13: 10}

TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", "il_datasets")
MODELS_DIR = os.getenv("MODELS_DIR", "models")