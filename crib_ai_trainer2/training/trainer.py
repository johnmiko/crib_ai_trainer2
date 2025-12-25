from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import numpy as np
import torch
from logging import getLogger
from ..game import CribbageGame
from ..players.random_player import RandomPlayer
from ..players.rule_based_player import RuleBasedPlayer
from ..players.mcts_player import ISMCTSPlayer
from ..features import D_TOTAL, encode_state
from ...models.perceptron import SimplePerceptron, PerceptronConfig

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

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.models: Dict[str, object] = {}
        self.best_model_name: str = "reasonable"
        self._init_models()

    def _init_models(self) -> None:
        self.models["random"] = RandomPlayer()
        self.models["reasonable"] = RuleBasedPlayer()
        self.models["is_mcts"] = ISMCTSPlayer(simulations=500)
        # simple perceptron for discard (2-card select) and pegging (1-card select) as separate heads
        self.models["perceptron_discard"] = SimplePerceptron(PerceptronConfig(input_dim=D_TOTAL, output_dim=52))
        self.models["perceptron_pegging"] = SimplePerceptron(PerceptronConfig(input_dim=D_TOTAL, output_dim=52))
        from ..players.cfr_player import CFRPlayer
        self.models["cfr"] = CFRPlayer(iterations=500)

    def train(self) -> None:
        start_time = time.time()
        while True:
            for name in list(self.models.keys()):
                if self._is_excluded(name):
                    continue
                self._train_model(name)
            # benchmarking among models
            self._rank_models(self.cfg.benchmark_games)
            if not self.cfg.run_indefinitely:
                break
            if self.cfg.max_seconds and (time.time() - start_time) > self.cfg.max_seconds:
                logger.info("Max time reached, stopping training loop.")
                break

    def _is_excluded(self, name: str) -> bool:
        if self.cfg.include_models and name not in self.cfg.include_models:
            return True
        if self.cfg.exclude_models and name in self.cfg.exclude_models:
            return True
        return False

    def _train_model(self, name: str) -> None:
        logger.info(f"Training model: {name}")
        # default train against best model
        opponent_name = self.best_model_name if self.best_model_name != name else "reasonable"
        player = self.models[name]
        opponent = self.models[opponent_name]
        # If CFR, run internal training iterations to update regrets
        if name == "cfr":
            cfr = player
            from ..cards import Card
            def legal_actions_fn(count: int):
                # abstract legal actions as ranks 1..13 that keep count <=31
                playable = []
                for r in range(1,14):
                    if count + RANK_VALUE[r] <= 31:
                        playable.append(Card('H', r))
                return playable
            def opponent_policy_fn(*args, **kwargs):
                return None
            def hand_sampler_fn():
                # sample a random 6-card hand (ranks only)
                import random
                return [Card('H', random.randint(1,13)) for _ in range(6)]
            cfr.train_pegging(legal_actions_fn, opponent_policy_fn)
            cfr.train_discard(hand_sampler_fn)
        w = 0
        l = 0
        for _ in range(self.cfg.num_training_games):
            game = CribbageGame(player, opponent)
            s0, s1 = game.play_game()
            if s0 >= 121 and s0 > s1:
                w += 1
            else:
                l += 1
        logger.info(f"Training results: W={w} L={l}")
        # minimal: if wins exceed losses, consider improved (placeholder for proper update)
        # In real training, we'd update torch models with gradients here.

    def _rank_models(self, games: int) -> None:
        names = list(self.models.keys())
        scores = {n: 0.0 for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                w = 0
                for _ in range(games):
                    game = CribbageGame(self.models[a], self.models[b])
                    s0, s1 = game.play_game()
                    w += 1 if s0 > s1 else 0
                winrate = w / games
                scores[a] += winrate
                scores[b] += (1 - winrate)
                logger.info(f"Benchmark {a} vs {b}: {winrate:.2f}")
        self.best_model_name = max(scores.keys(), key=lambda n: scores[n])
        logger.info(f"Best model now: {self.best_model_name}")
