from dataclasses import dataclass
from typing import List, Tuple
import random
from logging import getLogger

logger = getLogger(__name__)


@dataclass(frozen=True)
class Card:
    suit: str
    rank: int

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def to_index(self) -> int:
        s = SUITES.index(self.suit)
        r = self.rank - 1
        return s * 13 + r

class Deck:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.cards: List[Card] = [Card(s, r) for s in SUITES for r in RANKS]

    def shuffle(self) -> None:
        self._rng.shuffle(self.cards)

    def deal(self, n: int) -> List[Card]:
        hand = self.cards[:n]
        self.cards = self.cards[n:]
        return hand

    def cut(self) -> Card:
        return self.deal(1)[0]

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.cards = [Card(s, r) for s in SUITES for r in RANKS]
        self.shuffle()
