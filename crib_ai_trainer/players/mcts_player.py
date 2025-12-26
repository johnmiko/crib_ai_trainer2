from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import random
from logging import getLogger
from cribbage.playingcards import Card, Deck
from cribbage.playingcards import Deck
from cribbage.cribbagegame import score_play as score_pegging_play

SUITS = Deck.SUITS
RANKS = Deck.RANKS

logger = getLogger(__name__)

class ISMCTSPlayer:
    def save(self, path: str):
        import json
        try:
            data = {
                'simulations': self.simulations,
                'belief_samples': self.belief_samples
            }
            with open(path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved ISMCTS parameters to {path}")
        except Exception as e:
            logger.error(f"Failed to save ISMCTS parameters to {path}: {e}")

    @classmethod
    def load(cls, path: str, name: str = "is_mcts", seed: int = None):
        import json
        import os
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded ISMCTS parameters from {path}")
                return cls(name=name, simulations=data.get('simulations', 1000), seed=seed, belief_samples=data.get('belief_samples', 10))
            except Exception as e:
                logger.error(f"Failed to load ISMCTS parameters from {path}: {e}")
        else:
            logger.info(f"No ISMCTS parameters found at {path}, using defaults.")
        return cls(name=name, seed=seed)
    def __init__(self, name: str = "is_mcts", simulations: int = 1000, seed: int | None = None, belief_samples: int = 10):
        self.name = name
        self.simulations = simulations
        self.belief_samples = belief_samples
        self._rng = random.Random(seed)

    def select_crib_cards(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        # Monte Carlo: sample starters and opponent discards; choose pair maximizing expected outcome
        n = len(hand)
        best_pair = (hand[0], hand[1])
        best_score = -1e9
        for i in range(n):
            for j in range(i + 1, n):
                kept = [hand[k] for k in range(n) if k not in (i, j)]
                total = 0.0
                for _ in range(max(50, self.simulations // n)):
                    # sample a starter from remaining deck uniformly
                    starter = Card(rank=self._rng.choice(list(RANKS.values())), suit=self._rng.choice(list(SUITS.values())))
                    # sample opponent discards (belief sampling)
                    # For v1, just random legal discards
                    total += self._estimate_hand_value(kept, starter)
                avg = total / max(1, max(50, self.simulations // n))
                if avg > best_score:
                    best_score = avg
                    best_pair = (hand[i], hand[j])
        return best_pair

    def _estimate_hand_value(self, kept: List[Card], starter: Card) -> float:
        v = 0.0
        s = sum(c.rank['value'] for c in kept) + starter.rank['value']
        v += -abs(15 - (s % 15)) * 0.1
        v += sum(1 for c in kept if c.rank['value'] <= 5) * 0.2
        return v

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card],
                     my_hand: Optional[List[Card]] = None, known_cards: Optional[List[Card]] = None) -> Optional[Card]:
        # IS-MCTS: simulate opponent hidden hand consistent with history; UCT to select action
        if not playable:
            return None
        root_children = {c: {"N": 0, "W": 0.0} for c in playable}
        for _ in range(self.simulations):
            # Belief sample: sample a possible opponent hand consistent with known cards and history
            opp_hand = self._sample_opponent_hand_belief(my_hand, known_cards, history_since_reset)
            # select
            c = self._uct_select(root_children, count, history_since_reset)
            reward = self._simulate_play(c, count, history_since_reset, opp_hand)
            rc = root_children[c]
            rc["N"] += 1
            rc["W"] += reward
        # choose best average reward
        best = max(playable, key=lambda c: root_children[c]["W"] / max(1, root_children[c]["N"]))
        return best

    def _uct_select(self, children: Dict[Card, Dict[str, float]], count: int, history: List[Card]) -> Card:
        total_N = sum(v["N"] for v in children.values()) + 1
        def uct(v):
            N = v["N"]
            W = v["W"]
            return (W / max(1, N)) + math.sqrt(2.0 * math.log(total_N) / max(1, N))
        return max(children.keys(), key=lambda c: uct(children[c]))

    def _simulate_play(self, card: Card, count: int, history: List[Card], opp_hand: Optional[List[Card]]) -> float:
        # rollout: immediate points + heuristic future pegging potential
        pts = score_pegging_play(history, card, count)
        new_count = count + card.rank['value']
        # heuristic: prefer keeping count <= 21 to avoid opponent 10 to 31
        future_val = -max(0, new_count - 21) * 0.05
        # Optionally, simulate opponent response (not implemented in v1)
        return pts + future_val

    def _sample_opponent_hand_belief(self, my_hand: Optional[List[Card]], known_cards: Optional[List[Card]], history: Optional[List[Card]]) -> List[Card]:
        # Sample a random 4-card hand for opponent, consistent with known cards and history (cards played by opponent)
        all_cards = [Card(rank=RANKS[rank], suit=SUITS[suit]) for suit in SUITS for rank in RANKS]
        exclude = set((c.suit, c.rank) for c in (known_cards or []) + (my_hand or []))
        # Remove cards played by both players in pegging history
        if history:
            exclude.update((c.suit, c.rank) for c in history)
        candidates = [c for c in all_cards if (c.suit, c.rank) not in exclude]
        # If not enough, return as many as possible
        return self._rng.sample(candidates, min(4, len(candidates)))
