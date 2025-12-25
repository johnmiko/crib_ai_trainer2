import os
import json
import subprocess
from datetime import datetime
from logging import getLogger

logger = getLogger(__name__)

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def save_model_artifact(model_name: str, version: int, model_data: dict, metrics: dict, config: dict, base_dir: str = "trained_models"):
    # Directory: trained_models/{model_name}/v{version}/
    out_dir = os.path.join(base_dir, model_name, f"v{version}")
    os.makedirs(out_dir, exist_ok=True)
    # Save best.json
    artifact = {
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": get_git_commit_hash(),
        "config": config,
        "metrics": metrics,
        "model": model_data,
    }
    best_path = os.path.join(out_dir, "best.json")
    with open(best_path, "w") as f:
        json.dump(artifact, f, indent=2)
    logger.info(f"Saved model artifact to {best_path}")
    # Save snapshot to history
    hist_dir = os.path.join(out_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)
    snap_path = os.path.join(hist_dir, f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json")
    with open(snap_path, "w") as f:
        json.dump(artifact, f, indent=2)
    logger.info(f"Saved model snapshot to {snap_path}")
