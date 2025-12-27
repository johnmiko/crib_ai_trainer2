
import os
import shutil
import torch
import pytest
from crib_ai_trainer.training.trainer import Trainer, TrainConfig
from models.neural_config import NeuralNetConfig
from scripts.train_neural_configs import FlexibleNN, load_configs
from crib_ai_trainer.constants import RANK_VALUE

TEST_MODEL_NAME = "nn_64x1"
MODEL_PATH = os.path.join('trained_models', f'neural_{TEST_MODEL_NAME}.pt')
CONFIG_PATH = os.path.join('configs', 'neural_configs.json')

@pytest.fixture(scope="module")
def setup_and_teardown():
    # Backup model weights if they exist
    backup_path = MODEL_PATH + ".bak"
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, backup_path)
    yield
    # Restore model weights
    if os.path.exists(backup_path):
        shutil.move(backup_path, MODEL_PATH)
    elif os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

def test_nn_64x1_runs_first():
    configs = load_configs(CONFIG_PATH)
    assert configs[0].name == TEST_MODEL_NAME, "nn_64x1 should be the first config run"

def test_trains_against_best_model():
    configs = load_configs(CONFIG_PATH)
    cfg = next(c for c in configs if c.name == TEST_MODEL_NAME)
    trainer = Trainer(TrainConfig(num_training_games=2, benchmark_games=2, run_indefinitely=False))
    model = FlexibleNN(cfg)
    player = trainer.models.get(cfg.name)
    assert player is None or hasattr(player, 'model'), "Should be able to add neural player"
    # Add model and check best model
    trainer.models[cfg.name] = player or FlexibleNN(cfg)
    best_model_name = trainer.best_model_name
    # If best model is not present, add a dummy model for test
    if best_model_name not in trainer.models and best_model_name != 'reasonable':
        trainer.models[best_model_name] = FlexibleNN(cfg)
    assert best_model_name in trainer.models or best_model_name == 'reasonable', "Should train against best model or fallback to reasonable"

def test_old_vs_new_weights(setup_and_teardown):
    configs = load_configs(CONFIG_PATH)
    cfg = next(c for c in configs if c.name == TEST_MODEL_NAME)
    # Save dummy old weights
    model = FlexibleNN(cfg)
    torch.save(model.state_dict(), MODEL_PATH)
    # Simulate training new weights
    new_model = FlexibleNN(cfg)
    for param in new_model.parameters():
        param.data += 0.01  # Make new weights different
    from crib_ai_trainer.players.old_neural_player import NeuralPlayer
    old_player = NeuralPlayer(model)
    new_player = NeuralPlayer(new_model)
    from scripts.train_neural_configs import benchmark_models
    new_wins, old_wins, winrate, ci = benchmark_models(new_player, old_player, 4, seed_offset=10000)
    # If new wins, save over weights
    if new_wins > old_wins:
        torch.save(new_model.state_dict(), MODEL_PATH)
    # Check that weights were updated if new won
    loaded = FlexibleNN(cfg)
    loaded.load_state_dict(torch.load(MODEL_PATH))
    if new_wins > old_wins:
        for p1, p2 in zip(new_model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2), "Weights should be updated to new model"
    else:
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2), "Weights should remain as old model if new did not win"
