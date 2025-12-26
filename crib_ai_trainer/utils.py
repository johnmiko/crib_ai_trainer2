from itertools import combinations
from typing import List, Tuple, Optional
from logging import getLogger
from cribbage.playingcards import Card
from itertools import combinations


logger = getLogger(__name__)

def fill_hand(possible_hands: list[tuple[list[Card], list[Card]]]) -> list[tuple[list[Card], list[Card]]]:
    """
    Given a list of (hand, crib) pairs, return all possible 5-card hands for each hand and crib.
    Returns a list of tuples: (hand_5, crib_5)
    """    
    from crib_ai_trainer.constants import SUITES
    filled = []
    deck = [Card(suit, rank) for suit in SUITES for rank in range(1, 14)]
    for hand, crib in possible_hands:
        hand_set = set(hand + crib)
        remaining = [card for card in deck if card not in hand_set]
        # For each 4-card hand, add each remaining card to make a 5-card hand
        hand_5s = [list(hand) + [card] for card in remaining] if len(hand) == 4 else []
        crib_5s = [list(crib) + [card] for card in remaining] if len(crib) == 4 else []
        for h5 in hand_5s:
            for c5 in crib_5s:
                filled.append((h5, c5))
    return filled






def get_possible_hands(hand: list[Card]) -> list[tuple[list[Card], list[Card]]]:
    """
    Given a 6-card hand, return all possible (cards_to_keep, crib_cards) pairs,
    where cards_to_keep is a list of 4 cards and crib_cards is the 2 cards put in the crib.
    """
    if len(hand) != 6:
        raise ValueError("Hand must have exactly 6 cards")
    all_combos = []
    for kept in combinations(hand, 4):
        crib = [c for c in hand if c not in kept]
        all_combos.append((list(kept), crib))
    return all_combos
