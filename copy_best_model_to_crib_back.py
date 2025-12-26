import os
import shutil
from logging import getLogger

logger = getLogger(__name__)

BEST_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'trained_models', 'best_model.txt')
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
# Save directly to the absolute crib_back directory
CRIB_BACK_DIR = r'C:\Users\johnm\ccode\crib_back'


def get_best_model_name():
    try:
        with open(BEST_MODEL_FILE, 'r') as f:
            model_name = f.read().strip()
        logger.info(f"Best model: {model_name}")
        return model_name
    except Exception as e:
        logger.error(f"Failed to read best model: {e}")
        return None


def copy_best_model_to_crib_back():
    model_name = get_best_model_name()
    if not model_name:
        logger.error("No best model found.")
        return

    # Find model weights (pt file) and config (json file)
    pt_file = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.pt")
    json_file = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.json")

    if not os.path.exists(pt_file):
        logger.error(f"Weights file not found: {pt_file}")
        return
    if not os.path.exists(json_file):
        logger.error(f"Config file not found: {json_file}")
        return

    if not os.path.exists(CRIB_BACK_DIR):
        os.makedirs(CRIB_BACK_DIR)

    dest_pt = os.path.join(CRIB_BACK_DIR, "best-ai.pt")
    dest_json = os.path.join(CRIB_BACK_DIR, "best-ai.json")

    shutil.copy2(pt_file, dest_pt)
    shutil.copy2(json_file, dest_json)
    logger.info(f"Copied {pt_file} and {json_file} to {CRIB_BACK_DIR} as best-ai.pt and best-ai.json")


if __name__ == "__main__":
    copy_best_model_to_crib_back()
