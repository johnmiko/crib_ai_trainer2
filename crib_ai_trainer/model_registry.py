import torch
from crib_ai_trainer.players.rule_based_player import ReasonablePlayer, DifficultReasonablePlayer
from crib_ai_trainer.players.old_neural_player import NeuralPlayer
from models.neural_config import NeuralNetConfig
import os

def load_best_model():
    """Load the best model (neural or rule-based) as a PlayerInterface."""
    best_file = os.path.join("trained_models", "best_model.txt")
    if not os.path.exists(best_file):
        return ReasonablePlayer()
    with open(best_file, "r") as f:
        name = f.read().strip()
    if name.startswith("neural_"):
        # Remove 'neural_' prefix if present
        model_name = name[len("neural_"):] if name.startswith("neural_") else name
        config_path = os.path.join("trained_models", f"{model_name}.json")
        weights_path = os.path.join("trained_models", f"{model_name}.pt")
        if os.path.exists(config_path) and os.path.exists(weights_path):
            config = NeuralNetConfig.load(config_path)
            import torch.nn as nn
            class FlexibleNN(nn.Module):
                def __init__(self, config: NeuralNetConfig):
                    super().__init__()
                    layers = []
                    input_dim = 208  # D_TOTAL, hardcoded for now
                    for i in range(config.depth):
                        h = config.hidden_sizes[min(i, len(config.hidden_sizes)-1)]
                        layers.append(nn.Linear(input_dim, h))
                        layers.append(nn.ReLU())
                        input_dim = h
                    layers.append(nn.Linear(input_dim, 52))
                    self.net = nn.Sequential(*layers)
                def forward(self, x):
                    return self.net(x)
            model = FlexibleNN(config)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()
            return NeuralPlayer(model)
        else:
            return ReasonablePlayer()
    elif name == "reasonable":
        return ReasonablePlayer()
    elif name == "difficult":
        return DifficultReasonablePlayer()
    else:
        # fallback
        return ReasonablePlayer()
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
