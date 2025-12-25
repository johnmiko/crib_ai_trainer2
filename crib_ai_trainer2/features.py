from typing import List
import numpy as np
from logging import getLogger
from crib_ai_trainer2.cards import Card

logger = getLogger(__name__)

"""
Feature encoding for Cribbage AI:
    - 52-bit multi-hot for hand
    - 52-bit multi-hot for seen/played
    - 52-bit one-hot for starter
    - 32 one-hot for count
    - K x 52 one-hot for history since last reset (K=8)
"""
D_HAND = 52
D_SEEN = 52
D_STARTER = 52
D_COUNT = 32
K_HISTORY = 8
D_HISTORY = K_HISTORY * 52
D_TOTAL = D_HAND + D_SEEN + D_STARTER + D_COUNT + D_HISTORY

def one_hot_card(card: Card) -> np.ndarray:
    v = np.zeros(52, dtype=np.float32)
    if card is not None:
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
    """
    Returns a feature vector for the current state.
    hand: cards in hand (multi-hot, 52)
    seen: cards already played/seen (multi-hot, 52)
    starter: starter card (one-hot, 52)
    count: current pegging count (one-hot, 32)
    history_reset: last K cards played since reset (K x 52 one-hot)
    """
    # Defensive: ensure all arrays are correct length
    hand_vec = multi_hot_cards(hand)
    seen_vec = multi_hot_cards(seen)
    starter_vec = one_hot_card(starter) if starter is not None else np.zeros(52, dtype=np.float32)
    count_vec = one_hot_count(count)
    hist_vec = history_since_reset(history_reset, K_HISTORY)
    assert hand_vec.shape == (52,)
    assert seen_vec.shape == (52,)
    assert starter_vec.shape == (52,)
    assert count_vec.shape == (32,)
    assert hist_vec.shape == (K_HISTORY * 52,)
    return np.concatenate([
        hand_vec,
        seen_vec,
        starter_vec,
        count_vec,
        hist_vec,
    ])
