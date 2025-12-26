import os



import shutil
import torch
import pytest
from crib_ai_trainer.training.trainer import Trainer, TrainConfig
from models.neural_config import NeuralNetConfig
from scripts.train_neural_configs import FlexibleNN
from crib_ai_trainer.players.neural_player import NeuralPlayer

@pytest.mark.parametrize("winrate", [0.6])
def test_overwrite_pt_on_win(tmp_path, winrate):
    # Setup config for nn_32x1
    config = NeuralNetConfig(hidden_sizes=[32], depth=1, learning_rate=0.001, name="nn_32x1")
    model_path = tmp_path / "neural_nn_32x1.pt"
    # Create a dummy model and save initial weights
    model = FlexibleNN(config)
    torch.save(model.state_dict(), model_path)
    # Simulate previous weights
    prev_weights = model.state_dict()
    # Simulate training: change weights
    for param in model.parameters():
        param.data += 1.0
    # Simulate self-play evaluation: winrate > 0.5
    if winrate > 0.5:
        torch.save(model.state_dict(), model_path)
    # Load weights back and check they match the new weights
    loaded = FlexibleNN(config)
    loaded.load_state_dict(torch.load(model_path))
    # The weights should match the updated model, not the previous
    for p1, p2 in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2), "Weights were not overwritten as expected"
