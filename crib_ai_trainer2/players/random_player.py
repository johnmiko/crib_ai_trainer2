from typing import List, Tuple, Optional
import random
from logging import getLogger
from ..cards import Card

logger = getLogger(__name__)

class RandomPlayer:
    def __init__(self, name: str = "random", seed: int | None = None):
        self.name = name
        self._rng = random.Random(seed)

    def choose_discard(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        return tuple(self._rng.sample(hand, 2))  # type: ignore

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        return self._rng.choice(playable) if playable else None
