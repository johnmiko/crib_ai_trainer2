from crib_ai_trainer.players.rule_based_player import get_possible_hands
from cribbage.playingcards import Card
import pytest



def test_get_possible_hands_correctness():
    hand = [Card('H', 1), Card('H', 2), Card('C', 3), Card('D', 4), Card('S', 5), Card('H', 6)]
    possible_hands = get_possible_hands(hand)
    assert len(possible_hands) == 15  # C(6,4) = 15
    seen = set()
    for kept, crib in possible_hands:
        assert len(kept) == 4
        assert len(crib) == 2
        assert set(kept).union(set(crib)) == set(hand)
        assert tuple(sorted((tuple(sorted((c.suit, c.rank) for c in kept)),
                             tuple(sorted((c.suit, c.rank) for c in crib))))) not in seen
        seen.add(tuple(sorted((tuple(sorted((c.suit, c.rank) for c in kept)),
                               tuple(sorted((c.suit, c.rank) for c in crib))))))