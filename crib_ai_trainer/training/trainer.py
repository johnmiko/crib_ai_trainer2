from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import numpy as np
import torch
from logging import getLogger
from cribbage.cribbagegame import CribbageGame
from crib_ai_trainer.players.random_player import RandomPlayer
from crib_ai_trainer.players.rule_based_player import RuleBasedPlayer

from crib_ai_trainer.players.mcts_player import ISMCTSPlayer
from crib_ai_trainer.features import D_TOTAL, encode_state
from models.perceptron import SimplePerceptron, PerceptronConfig
from models.neural_config import NeuralNetConfig
from cribbage.scoring import RANK_VALUE

logger = getLogger(__name__)

@dataclass
class TrainConfig:
    num_training_games: int = 10
    benchmark_games: int = 50
    run_indefinitely: bool = False
    max_seconds: Optional[int] = None
    include_models: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None
    max_underperformance: float = 0.35
    reward_win_bonus: int = 30

class Trainer:
    BEST_MODEL_FILE = "trained_models/best_model.txt"

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.models: Dict[str, object] = {}
        self.best_model_name: str = self._load_best_model()
        self._init_models()

    def _load_best_model(self) -> str:
        try:
            with open(self.BEST_MODEL_FILE, "r") as f:
                name = f.read().strip()
                if name:
                    logger.info(f"Loaded best model from file: {name}")
                    return name
        except Exception:
            pass
        return "reasonable"

    def _init_models(self) -> None:
        self.models["random"] = RandomPlayer()
        self.models["reasonable"] = RuleBasedPlayer()
        from crib_ai_trainer.players.rule_based_player import DifficultRuleBasedPlayer
        self.models["difficult"] = DifficultRuleBasedPlayer()
        from crib_ai_trainer.players.mcts_player import ISMCTSPlayer
        import os
        mcts_path = os.path.join('trained_models', 'is_mcts.json')
        self._mcts_path = mcts_path
        self.models["is_mcts"] = ISMCTSPlayer.load(mcts_path, name="is_mcts", seed=None)
        # single perceptron for both discard and pegging
        from models.perceptron_io import load_perceptron, save_perceptron
        self._save_perceptron = save_perceptron
        # Only load perceptron if it is the best model or is being trained or trained against
        perceptron_needed = False
        if self.best_model_name == "perceptron":
            perceptron_needed = True
        elif self.cfg.include_models and "perceptron" in self.cfg.include_models:
            perceptron_needed = True
        elif self.cfg.exclude_models and "perceptron" not in self.cfg.exclude_models:
            perceptron_needed = True
        if perceptron_needed:
            self.models["perceptron"] = load_perceptron()
        from crib_ai_trainer.players.cfr_player import CFRPlayer
        import os
        cfr_path = os.path.join('trained_models', 'cfr.json')
        self._cfr_path = cfr_path
        self.models["cfr"] = CFRPlayer(iterations=500, load_path=cfr_path)
        from crib_ai_trainer.players.neural_player import NeuralPlayer
        import torch.nn as nn
        # Example config, can be loaded or set dynamically
        self.neural_config = NeuralNetConfig(
            hidden_sizes=[128],
            depth=1,
            learning_rate=1e-3,
            entropy_coef=0.0,
            regularization=None,
            name="default"
        )
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
        self.models["neural"] = NeuralPlayer(FlexibleNN(self.neural_config))

    def train(self) -> None:
        logger.info("Starting training loop...")
        start_time = time.time()
        round_num = 1
        while True:
            logger.info(f"=== Training round {round_num} ===")
            for name in list(self.models.keys()):
                if self._is_excluded(name):
                    logger.info(f"Skipping model: {name}")
                    continue
                self._train_model(name)
            # Only save/update models that are being trained
            include = set(self.cfg.include_models) if self.cfg.include_models else set(self.models.keys())
            # Save perceptron weights if being trained
            if "perceptron" in self.models and hasattr(self, "_save_perceptron") and "perceptron" in include:
                self._save_perceptron(self.models["perceptron"])
            # Save CFR tables if being trained
            if "cfr" in self.models and hasattr(self.models["cfr"], "save") and "cfr" in include:
                self.models["cfr"].save(self._cfr_path)
            # Save ISMCTS parameters if being trained. Always save at least once so the file exists.
            if "is_mcts" in self.models and hasattr(self.models["is_mcts"], "save") and "is_mcts" in include:
                import os
                import shutil
                from cribbage.cribbagegame import CribbageGame
                mcts_model = self.models["is_mcts"]
                mcts_path = self._mcts_path
                # If the file does not exist, save the current model to create it
                if not os.path.exists(mcts_path):
                    mcts_model.save(mcts_path)
                    logger.info(f"ISMCTS model parameters saved for the first time to {mcts_path}")
                # Load old model for comparison
                from crib_ai_trainer.players.mcts_player import ISMCTSPlayer
                old_model = ISMCTSPlayer.load(mcts_path, name="is_mcts", seed=None)
                # Benchmark new vs old
                new_wins = 0
                old_wins = 0
                benchmark_games = self.cfg.benchmark_games
                for i in range(benchmark_games):
                    if i % 2 == 0:
                        game = CribbageGame(mcts_model, old_model, seed=i)
                        s0, s1 = game.play_game()
                    else:
                        game = CribbageGame(old_model, mcts_model, seed=i)
                        s1, s0 = game.play_game()
                    if s0 > s1:
                        new_wins += 1
                    else:
                        old_wins += 1
                if new_wins > old_wins:
                    mcts_model.save(mcts_path)
                    logger.info(f"Updated ISMCTS model parameters (new: {new_wins}, old: {old_wins})")
                else:
                    logger.info(f"Retained old ISMCTS model parameters (new: {new_wins}, old: {old_wins})")
            logger.info("Benchmarking models...")
            self._rank_models(self.cfg.benchmark_games)
            if not self.cfg.run_indefinitely:
                logger.info("Training loop complete.")
                break
            if self.cfg.max_seconds and (time.time() - start_time) > self.cfg.max_seconds:
                logger.info("Max time reached, stopping training loop.")
                break
            round_num += 1

    def _is_excluded(self, name: str) -> bool:
        # manually exclude certain models that don't need training
        if name in ["random", "perceptron", "cfr"]:
            return True
        if self.cfg.include_models and name not in self.cfg.include_models:
            return True
        if self.cfg.exclude_models and name in self.cfg.exclude_models:
            return True
        return False

    def _train_model(self, name: str) -> None:
        # Always train against the current best model (unless self-play)
        opponent_name = self.best_model_name if self.best_model_name != name else "reasonable"
        # If the best model is not available (e.g., just started), fallback to reasonable
        if opponent_name not in self.models:
            opponent_name = "reasonable"
        logger.info(f"Training model: {name} vs {opponent_name}")
        player = self.models[name]
        opponent = self.models[opponent_name]
        # If CFR, run internal training iterations to update regrets
        if name == "cfr":
            cfr = player
            from cribbage.playingcards import Card
            def legal_actions_fn(count: int):
                playable = []
                for r in range(1,14):
                    if count + RANK_VALUE[r] <= 31:
                        playable.append(Card('H', r))
                return playable
            def opponent_policy_fn(*args, **kwargs):
                return None
            def hand_sampler_fn():
                import random
                return [Card('H', random.randint(1,13)) for _ in range(6)]
            cfr.train_pegging(legal_actions_fn, opponent_policy_fn)
            cfr.train_discard(hand_sampler_fn)
        # If perceptron, do imitation learning from reasonable player
        if name == "perceptron":
            import torch
            import torch.optim as optim
            # Use reasonable player as teacher
            teacher = self.models["reasonable"]
            perceptron = player
            # Generate imitation data for both discard and pegging
            from cribbage.playingcards import Card
            X = []
            y = []
            for _ in range(50):
                # Discard imitation
                hand = [Card('H', i+1) for i in range(6)]
                starter = Card('S', 5)
                seen = []
                count = 0
                history = []
                action = teacher.choose_discard(hand, dealer_is_self=True)[0].to_index()
                X.append(encode_state(hand, starter, seen, count, history))
                y.append(action)
            for _ in range(50):
                # Pegging imitation
                hand = [Card('H', i+1) for i in range(4)]
                starter = Card('S', 5)
                seen = []
                count = 0
                history = []
                action = teacher.play_pegging(hand, count, history).to_index()
                X.append(encode_state(hand, starter, seen, count, history))
                y.append(action)
            import numpy as np
            X = np.array(X)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            optimizer = optim.Adam(perceptron.parameters(), lr=0.01)
            for _ in range(20):
                optimizer.zero_grad()
                logits = perceptron(X)
                loss = torch.nn.functional.nll_loss(logits, y)
                loss.backward()
                optimizer.step()

        w = 0
        l = 0
        total_reward = 0
        for _ in range(self.cfg.num_training_games):
            game = CribbageGame(player, opponent)
            s0, s1 = game.play_game()
            # Reward shaping: primary = final point diff, bonus for win/loss
            reward = (s0 - s1)
            if s0 >= 121 and s0 > s1:
                w += 1
                reward += self.cfg.reward_win_bonus
            else:
                l += 1
                reward -= self.cfg.reward_win_bonus
            total_reward += reward
        logger.info(f"Training results vs {opponent_name}: W={w} L={l} | Total reward: {total_reward}")
        # minimal: if wins exceed losses, consider improved (placeholder for proper update)
        # In real training, we'd update torch models with gradients here.

    def _rank_models(self, games: int) -> None:
        import numpy as np
        names = list(self.models.keys())
        # Load best model from file, default to 'reasonable'
        best = self._load_best_model()
        best_score = 0.0
        for name in names:
            if self._is_excluded(name):
                logger.info(f"Skipping model: {name}")
                continue
            if name == best:
                continue
            w = 0
            results = []
            for g in range(games):
                # Alternate dealer and swap positions for fairness
                if g % 2 == 0:
                    game = CribbageGame(self.models[name], self.models[best], seed=g)
                    s0, s1 = game.play_game()
                    win = s0 > s1
                else:
                    game = CribbageGame(self.models[best], self.models[name], seed=g)
                    s1, s0 = game.play_game()
                    win = s0 > s1
                w += int(win)
                results.append(int(win))
            winrate = w / games
            n = games
            phat = winrate
            z = 1.96  # 95% CI
            ci = z * np.sqrt(phat * (1 - phat) / n)
            logger.info(f"Benchmark {name} vs {best}: {winrate:.2f} ± {ci:.2f}")
            # Save neural config and results for any neural config model
            if hasattr(self, 'neural_config') and (name == getattr(self.neural_config, 'name', None) or name.startswith('nn_')):
                from models.neural_config import NeuralNetConfig
                config = self.neural_config
                results_dict = {
                    "winrate": winrate,
                    "ci": ci,
                    "games": games,
                    "opponent": best
                }
                config.save(f"trained_models/neural_{name}.json", results=results_dict)
            if (winrate - ci) > 0.5:
                logger.info(f"{name} beats {best} with winrate {winrate:.2f}! Now considered best.")
                best = name
                best_score = winrate
                # Save best model name to file, include neural variant if applicable
                try:
                    with open(self.BEST_MODEL_FILE, "w") as f:
                        if hasattr(self, 'neural_config') and (name == getattr(self.neural_config, 'name', None) or name.startswith('nn_')):
                            f.write(f"neural_{name}")
                        else:
                            f.write(best)
                except Exception as e:
                    logger.warning(f"Could not save best model: {e}")
        self.best_model_name = best
        logger.info(f"Best model now: {self.best_model_name}")
