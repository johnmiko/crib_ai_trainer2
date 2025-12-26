from typing import List, Tuple, Optional
from logging import getLogger
from crib_ai_trainer.cards import Card
from cribbage.playingcards import Card
from cribbage.cribbagegame import score_hand, score_play as score_pegging_play
from cribbage.playingcards import Deck
from itertools import combinations

logger = getLogger(__name__)


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


class RuleBasedPlayer:
    def __init__(self, name: str = "reasonable"):
        self.name = name

    def choose_discard(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        # choose discard maximizing expected hand score after cut (approx: without starter)
        best_discards = []
        best_score = -1
        n = len(hand)
        for i in range(n):
            for j in range(i + 1, n):
                kept = [hand[k] for k in range(n) if k not in (i, j)]
                # approximate by using a neutral starter (no extra points)
                score = score_hand(kept, kept[0], is_crib=False) - score_hand([hand[i], hand[j]], kept[0], is_crib=True)
                if score > best_score:
                    best_score = score
                    best_discards = [(hand[i], hand[j])]
                elif score == best_score:
                    best_discards.append((hand[i], hand[j]))
        return best_discards[0]  # type: ignore

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        # always take points if available; else play lowest that doesn't set opponent up
        best = None
        best_pts = -1
        for c in playable:
            pts = score_pegging_play(history_since_reset, c, count)
            if pts > best_pts:
                best_pts = pts
                best = c
        if best is not None:
            return best
        # otherwise play lowest value
        return sorted(playable, key=lambda c: c.rank['value'])[0] if playable else None


class DifficultRuleBasedPlayer:
    def __init__(self, name: str = "difficult"):
        self.name = name

    def choose_discard(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        # choose discard maximizing expected hand score after cut, considering all possible starters if tied
        from crib_ai_trainer.cards import Card
        from crib_ai_trainer.scoring import score_hand
        best_discards = []
        best_score = -1
        n = len(hand)
        deck = [Card(rank=Deck.RANKS[rank], suit=Deck.SUITS[suit]) for suit in Deck.SUITS for rank in Deck.RANKS]
        hand_set = set(hand)
        for i in range(n):
            for j in range(i + 1, n):
                kept = [hand[k] for k in range(n) if k not in (i, j)]
                # approximate by using a neutral starter (no extra points)
                score = score_hand(kept, kept[0], is_crib=False) - score_hand([hand[i], hand[j]], kept[0], is_crib=True)
                if score > best_score:
                    best_score = score
                    best_discards = [(hand[i], hand[j], kept)]
                elif score == best_score:
                    best_discards.append((hand[i], hand[j], kept))
        # If only one best, return it
        if len(best_discards) == 1:
            return best_discards[0][0], best_discards[0][1]
        # Otherwise, evaluate all possible starters for each best discard
        max_expected = -1
        best_pair = None
        for discard1, discard2, kept in best_discards:
            starters = [c for c in deck if c not in hand_set and c not in [discard1, discard2]]
            total = 0
            for starter in starters:
                total += score_hand(kept, starter, is_crib=False)
            expected = total / len(starters) if starters else 0
            if expected > max_expected:
                max_expected = expected
                best_pair = (discard1, discard2)
        return best_pair  # type: ignore

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        # Use same logic as RuleBasedPlayer for pegging
        best = None
        best_pts = -1
        for c in playable:
            pts = score_pegging_play(history_since_reset, c, count)
            if pts > best_pts:
                best_pts = pts
                best = c
        if best is not None:
            return best
        return sorted(playable, key=lambda c: c.rank['value'])[0] if playable else None
