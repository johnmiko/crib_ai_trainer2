import os
import sys
import tempfile
import shutil
import pytest
from unittest import mock

def test_train_neural_configs_runs():
    # Patch file writing and torch.save to avoid saving any files
    minimal_json = '[{"hidden_sizes": [4], "depth": 1, "learning_rate": 0.01, "entropy_coef": 0.0, "regularization": null, "name": "test"}]'
    def fake_open(file, *args, **kwargs):
        if file.endswith('neural_configs.json'):
            return mock.mock_open(read_data=minimal_json)()
        return mock.mock_open()()
    with mock.patch("builtins.open", fake_open), \
         mock.patch("torch.save"), \
         mock.patch("os.path.exists", return_value=False), \
         mock.patch("os.makedirs"), \
         mock.patch("builtins.print") as mock_print:
        # Patch sys.argv to simulate running as script
        sys_argv_backup = sys.argv
        sys.argv = ["train_neural_configs.py"]
        # Patch config path to use a temp file
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_neural_configs", os.path.join(os.path.dirname(__file__), "../../scripts/train_neural_configs.py"))
        train_mod = importlib.util.module_from_spec(spec)
        # Patch load_configs to only load a single config
        def fake_load_configs(_):
            from models.neural_config import NeuralNetConfig
            return [NeuralNetConfig(hidden_sizes=[4], depth=1, learning_rate=0.01, name="test")]  # minimal config
        setattr(train_mod, "load_configs", fake_load_configs)
        # Patch Trainer to only run 1 round
        from crib_ai_trainer.training.trainer import Trainer, TrainConfig
        orig_train = Trainer.train
        def one_round_train(self):
            self.cfg.num_training_games = 1
            self.cfg.benchmark_games = 1
            self.cfg.run_indefinitely = False
            return orig_train(self)
        with mock.patch("crib_ai_trainer.training.trainer.Trainer.train", new=one_round_train):
            # Actually run main()
            spec.loader.exec_module(train_mod)
            train_mod.main()
        sys.argv = sys_argv_backup
        # Check that the script ran and printed something
        assert mock_print.called
