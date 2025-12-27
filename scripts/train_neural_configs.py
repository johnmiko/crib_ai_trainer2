from cribbage.cribbagegame import CribbageGame
import numpy as np

def benchmark_models(player_a, player_b, num_games, seed_offset=0):
    """Run num_games of player_a vs player_b, alternating dealer. Returns (a_wins, b_wins, winrate, ci)."""
    a_wins = 0
    b_wins = 0
    for i in range(num_games):
        if i % 2 == 0:
            game = CribbageGame([player_a, player_b], seed=seed_offset + i)
            s0, s1 = game.start()
        else:
            game = CribbageGame([player_b, player_a], seed=seed_offset + i)
            s1, s0 = game.start()
        if s0 > s1:
            a_wins += 1
        else:
            b_wins += 1
    n = num_games
    winrate = a_wins / n
    z = 1.96  # 95% CI
    ci = z * np.sqrt(winrate * (1 - winrate) / n) if n > 0 else 0.0
    return a_wins, b_wins, winrate, ci

import os
import sys
import json
# Ensure project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crib_ai_trainer.training.trainer import Trainer, TrainConfig
from models.neural_config import NeuralNetConfig
from crib_ai_trainer.players.rule_based_player import ReasonablePlayer
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
    trainer = Trainer(TrainConfig(num_training_games=500, benchmark_games=500, run_indefinitely=True))
    # Remove default 'neural' if present
    if 'neural' in trainer.models:
        del trainer.models['neural']
    import torch
    import torch
    from crib_ai_trainer.game import CribbageGame
    round_num = 1
    from collections import deque
    configs_deque = deque(configs)
    while True:
        print(f"=== Full training+benchmark round {round_num} ===")
        for _ in range(len(configs_deque)):
            cfg = configs_deque[0]
            import logging
            logger = logging.getLogger(__name__)
            print(f"Training neural config: {cfg.name}")
            model = FlexibleNN(cfg)
            model_path = os.path.join('trained_models', f'neural_{cfg.name}.pt')
            json_path = os.path.join('trained_models', f'neural_{cfg.name}.json')
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
            # Only train against previous version of itself (for training, not for self-play eval)
            trainer.models[cfg.name + "_old"] = NeuralPlayer(prev_model) if prev_weights_loaded else None
            # Train ONLY against the old version of itself (self-play)
            if prev_weights_loaded:
                trainer.cfg.exclude_models = [m for m in trainer.models if m not in [cfg.name, cfg.name + "_old"]]
                trainer.train()
            else:
                # If no previous weights, just train against reasonable
                trainer.cfg.exclude_models = [m for m in trainer.models if m not in [cfg.name, "reasonable"]]
                trainer.train()



            print(f"Finished trainer.train() for {cfg.name}")
            # Self-play evaluation: new model vs previous version
            do_benchmark = True
            if prev_weights_loaded:
                logger.info(f"=== SELF-PLAY BENCHMARK: {cfg.name} (new) vs {cfg.name} (old) ===")
                prev_player = NeuralPlayer(prev_model)
                self_wins, prev_wins, winrate, ci = benchmark_models(player, prev_player, trainer.cfg.benchmark_games, seed_offset=10000)
                logger.info(f"SELF-PLAY RESULT: {cfg.name} (new) winrate: {winrate:.3f} ± {ci:.3f} vs previous version (W={self_wins}, L={prev_wins}, N={trainer.cfg.benchmark_games})")
                if winrate > 0.5:
                    try:
                        torch.save(model.state_dict(), model_path)
                        logger.info(f"SELF-PLAY: New model {cfg.name} outperformed its previous version (winrate = {winrate:.3f} > 0.5). Saved model weights to {model_path}")
                    except Exception as e:
                        logger.error(f"SELF-PLAY: Could not save model weights for {cfg.name} after self-play: {e}")
                else:
                    logger.info(f"SELF-PLAY: New model {cfg.name} did not outperform its previous version. Skipping benchmark against best model.")
                    logger.info(f"SELF-PLAY: Finished training {cfg.name}\n")
                    do_benchmark = False
            else:
                logger.info(f"SELF-PLAY: No previous weights found for {cfg.name}, skipping self-play benchmark.")
            # Only if new weights beat old weights, benchmark against best model
            if do_benchmark:
                # Reload the current best model from best_model.txt
                best_model_name = "reasonable"
                best_model_path = os.path.join('trained_models', 'best_model.txt')
                if os.path.exists(best_model_path):
                    with open(best_model_path, 'r') as f:
                        best_model_name = f.read().strip()
                from crib_ai_trainer.players.rule_based_player import ReasonablePlayer
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
                    best_player = ReasonablePlayer()
                # ...existing code for benchmarking against best...

            # Benchmark new model vs current best
            print(f"Benchmarking {cfg.name} vs current best ({best_model_name})...")
            new_wins, best_wins, winrate, ci = benchmark_models(player, best_player, trainer.cfg.benchmark_games)
            print(f"{cfg.name} winrate vs {best_model_name}: {winrate:.3f} ± {ci:.3f}")
            if (winrate) > 0.5:
                try:
                    torch.save(model.state_dict(), model_path)
                    print(f"New model {cfg.name} outperformed best (winrate = {winrate:.3f} > 0.5). Saved model weights to {model_path}")
                    # Update best_model.txt
                    with open(best_model_path, 'w') as f:
                        f.write(f"neural_{cfg.name}")
                    print(f"Updated best_model.txt to neural_{cfg.name}")
                except Exception as e:
                    print(f"Could not save model weights or update best_model.txt for {cfg.name}: {e}")
            else:
                print(f"Best model retained ({best_model_name}); {cfg.name} did not outperform best with sufficient confidence.")
            print(f"Finished training {cfg.name}\n")
            # Rotate configs for next round
            configs_deque.rotate(-1)
        if not trainer.cfg.run_indefinitely:
            break
        round_num += 1

if __name__ == "__main__":
    main()