
import os
import sys
import json
# Ensure project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crib_ai_trainer.training.trainer import Trainer, TrainConfig
from models.neural_config import NeuralNetConfig
from crib_ai_trainer.players.neural_player import NeuralPlayer
import torch.nn as nn
from crib_ai_trainer.features import D_TOTAL

def load_configs(config_path):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    return [NeuralNetConfig(**cfg) for cfg in configs]

class FlexibleNN(nn.Module):
    def __init__(self, config: NeuralNetConfig):
        super().__init__()
        layers = []
        input_dim = D_TOTAL
        for i in range(config.depth):
            h = config.hidden_sizes[min(i, len(config.hidden_sizes)-1)]
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 52))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def main():
    config_path = os.path.join('configs', 'neural_configs.json')
    configs = load_configs(config_path)
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    trainer = Trainer(TrainConfig(num_training_games=2000, benchmark_games=1500, run_indefinitely=True))
    # Remove default 'neural' if present
    if 'neural' in trainer.models:
        del trainer.models['neural']
    import torch
    import torch
    from crib_ai_trainer.game import CribbageGame
    round_num = 1
    while True:
        print(f"=== Full training+benchmark round {round_num} ===")
        for cfg in configs:
            print(f"Training neural config: {cfg.name}")
            model = FlexibleNN(cfg)
            model_path = os.path.join('trained_models', f'neural_{cfg.name}.pt')
            json_path = os.path.join('trained_models', f'neural_{cfg.name}.json')
            # Load current best model
            best_model_name = "reasonable"
            best_model_path = os.path.join('trained_models', 'best_model.txt')
            if os.path.exists(best_model_path):
                with open(best_model_path, 'r') as f:
                    best_model_name = f.read().strip()
            # Load best model for benchmarking
            from crib_ai_trainer.players.rule_based_player import RuleBasedPlayer
            from crib_ai_trainer.players.neural_player import NeuralPlayer
            best_player = None
            if best_model_name.startswith('neural_'):
                best_cfg_name = best_model_name.replace('neural_', '')
                best_cfg = next((c for c in configs if c.name == best_cfg_name), None)
                if best_cfg:
                    best_model = FlexibleNN(best_cfg)
                    best_model_file = os.path.join('trained_models', f'neural_{best_cfg_name}.pt')
                    if os.path.exists(best_model_file):
                        best_model.load_state_dict(torch.load(best_model_file))
                        best_player = NeuralPlayer(best_model)
            if best_player is None:
                best_player = RuleBasedPlayer()

            # Load previous weights for this config if available (for self-play eval)
            prev_model = FlexibleNN(cfg)
            prev_weights_loaded = False
            if os.path.exists(model_path):
                try:
                    prev_model.load_state_dict(torch.load(model_path))
                    print(f"Loaded previous weights from {model_path} for self-play evaluation")
                    prev_weights_loaded = True
                except Exception as e:
                    print(f"Could not load previous weights for {cfg.name}: {e}")
            player = NeuralPlayer(model)
            trainer.models[cfg.name] = player
            trainer.neural_config = cfg
            trainer.cfg.include_models = [cfg.name]
            # Train against the current best model
            trainer.models[best_model_name] = best_player
            trainer.cfg.exclude_models = [m for m in trainer.models if m not in [cfg.name, best_model_name]]
            trainer.train()



            print(f"Finished trainer.train() for {cfg.name}")
            # Self-play evaluation: new model vs previous version
            if prev_weights_loaded:
                import numpy as np
                print(f"Self-play evaluation: {cfg.name} (new) vs {cfg.name} (previous)")
                prev_player = NeuralPlayer(prev_model)
                self_wins = 0
                prev_wins = 0
                benchmark_games = trainer.cfg.benchmark_games
                for i in range(benchmark_games):
                    if i % 2 == 0:
                        game = CribbageGame(player, prev_player, seed=10000+i)
                        s0, s1 = game.play_game()
                    else:
                        game = CribbageGame(prev_player, player, seed=10000+i)
                        s1, s0 = game.play_game()
                    if s0 > s1:
                        self_wins += 1
                    else:
                        prev_wins += 1
                n = benchmark_games
                winrate = self_wins / n
                z = 1.96  # 95% CI
                ci = z * np.sqrt(winrate * (1 - winrate) / n) if n > 0 else 0.0
                print(f"{cfg.name} (new) winrate: {winrate:.3f} ± {ci:.3f} vs previous version")
                if (winrate - ci) > 0.5:
                    try:
                        torch.save(model.state_dict(), model_path)
                        print(f"New model {cfg.name} outperformed its previous version (winrate - ci = {winrate - ci:.3f} > 0.5). Saved model weights to {model_path}")
                    except Exception as e:
                        print(f"Could not save model weights for {cfg.name} after self-play: {e}")
                else:
                    print(f"New model {cfg.name} did not outperform its previous version with sufficient confidence. Skipping benchmark against best model.")
                    print(f"Finished training {cfg.name}\n")
                    continue

            # Benchmark new model vs current best
            print(f"Benchmarking {cfg.name} vs current best ({best_model_name})...")
            import numpy as np
            new_wins = 0
            best_wins = 0
            benchmark_games = trainer.cfg.benchmark_games
            for i in range(benchmark_games):
                if i % 2 == 0:
                    game = CribbageGame(player, best_player, seed=i)
                    s0, s1 = game.play_game()
                else:
                    game = CribbageGame(best_player, player, seed=i)
                    s1, s0 = game.play_game()
                if s0 > s1:
                    new_wins += 1
                else:
                    best_wins += 1
            n = benchmark_games
            winrate = new_wins / n
            z = 1.96  # 95% CI
            ci = z * np.sqrt(winrate * (1 - winrate) / n) if n > 0 else 0.0
            print(f"{cfg.name} winrate vs {best_model_name}: {winrate:.3f} ± {ci:.3f}")
            if (winrate - ci) > 0.5:
                try:
                    torch.save(model.state_dict(), model_path)
                    print(f"New model {cfg.name} outperformed best (winrate - ci = {winrate - ci:.3f} > 0.5). Saved model weights to {model_path}")
                    # Update best_model.txt
                    with open(best_model_path, 'w') as f:
                        f.write(f"neural_{cfg.name}")
                    print(f"Updated best_model.txt to neural_{cfg.name}")
                except Exception as e:
                    print(f"Could not save model weights or update best_model.txt for {cfg.name}: {e}")
            else:
                print(f"Best model retained ({best_model_name}); {cfg.name} did not outperform best with sufficient confidence.")
            print(f"Finished training {cfg.name}\n")
        if not trainer.cfg.run_indefinitely:
            break
        round_num += 1

if __name__ == "__main__":
    main()