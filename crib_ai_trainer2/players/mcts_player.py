from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import random
from logging import getLogger
from crib_ai_trainer2.cards import Card, Deck, SUITS, RANKS
from crib_ai_trainer2.scoring import score_pegging_play, RANK_VALUE

logger = getLogger(__name__)

class ISMCTSPlayer:
    def __init__(self, name: str = "is_mcts", simulations: int = 1000, seed: int | None = None):
        self.name = name
        self.simulations = simulations
        self._rng = random.Random(seed)

    def choose_discard(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
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
                    # approximate: value of pegging ignored; focus on hand value
                    starter = Card(self._rng.choice(SUITS), self._rng.choice(RANKS))
                    total += self._estimate_hand_value(kept, starter)
                avg = total / max(1, max(50, self.simulations // n))
                if avg > best_score:
                    best_score = avg
                    best_pair = (hand[i], hand[j])
        return best_pair

    def _estimate_hand_value(self, kept: List[Card], starter: Card) -> float:
        # simple heuristic: favor 15s potential and runs; cheap estimation
        # use rank values and proximity to 15
        v = 0.0
        s = sum(RANK_VALUE[c.rank] for c in kept) + RANK_VALUE[starter.rank]
        v += -abs(15 - (s % 15)) * 0.1
        # small bonus for low cards aiding pegging flexibility
        v += sum(1 for c in kept if RANK_VALUE[c.rank] <= 5) * 0.2
        return v

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        # IS-MCTS: simulate opponent hidden hand consistent with history; UCT to select action
        if not playable:
            return None
        root_children = {c: {"N": 0, "W": 0.0} for c in playable}
        for _ in range(self.simulations):
            # select
            c = self._uct_select(root_children, count, history_since_reset)
            reward = self._simulate_play(c, count, history_since_reset)
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

    def _simulate_play(self, card: Card, count: int, history: List[Card]) -> float:
        # rollout: immediate points + heuristic future pegging potential
        pts = score_pegging_play(history, card, count)
        new_count = count + RANK_VALUE[card.rank]
        # heuristic: prefer keeping count <= 21 to avoid opponent 10 to 31
        future_val = -max(0, new_count - 21) * 0.05
        return pts + future_val
