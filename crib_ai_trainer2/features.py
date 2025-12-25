from typing import List
import numpy as np
from logging import getLogger
from .cards import Card

logger = getLogger(__name__)

D_HAND = 52
D_SEEN = 52
D_STARTER = 52
D_COUNT = 32
K_HISTORY = 8
D_HISTORY = K_HISTORY * 52
D_TOTAL = D_HAND + D_SEEN + D_STARTER + D_COUNT + D_HISTORY

def one_hot_card(card: Card) -> np.ndarray:
    v = np.zeros(52, dtype=np.float32)
    v[card.to_index()] = 1.0
    return v

def multi_hot_cards(cards: List[Card]) -> np.ndarray:
    v = np.zeros(52, dtype=np.float32)
    for c in cards:
        v[c.to_index()] = 1.0
    return v

def one_hot_count(count: int) -> np.ndarray:
    v = np.zeros(32, dtype=np.float32)
    v[min(max(count, 0), 31)] = 1.0
    return v

def history_since_reset(history: List[Card], k: int = K_HISTORY) -> np.ndarray:
    v = np.zeros(k * 52, dtype=np.float32)
    # last k cards; each slot is 52 one-hot
    h = history[-k:]
    start = k - len(h)
    for i, c in enumerate(h):
        v[(start + i) * 52 + c.to_index()] = 1.0
    return v

def encode_state(hand: List[Card], starter: Card | None, seen: List[Card], count: int, history_reset: List[Card]) -> np.ndarray:
    return np.concatenate([
        multi_hot_cards(hand),
        multi_hot_cards(seen),
        one_hot_card(starter) if starter is not None else np.zeros(52, dtype=np.float32),
        one_hot_count(count),
        history_since_reset(history_reset, K_HISTORY),
    ])
