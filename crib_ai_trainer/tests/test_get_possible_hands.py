import pytest
from crib_ai_trainer.players.rule_based_player import get_possible_hands
from crib_ai_trainer.cards import Card

def test_get_possible_hands_basic():
    hand = [Card('H', 1), Card('H', 2), Card('H', 3), Card('H', 4), Card('H', 5), Card('H', 6)]
    possible_hands = get_possible_hands(hand)
    # There are 52 cards in the deck, 6 in hand, so 46 possible new hands
    assert len(possible_hands) == 46
    # Each hand should have 7 cards
    for h in possible_hands:
        assert len(h) == 7
        # The original 6 cards must be present in each hand
        for c in hand:
            assert c in h
        # The 7th card must not be in the original hand
        extra = [c for c in h if c not in hand]
        assert len(extra) == 1
        assert extra[0] not in hand

def test_get_possible_hands_error():
    with pytest.raises(ValueError):
        get_possible_hands([Card('H', 1)])
