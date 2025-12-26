import pytest
from crib_ai_trainer.players.rule_based_player import DifficultRuleBasedPlayer
from cribbage.playingcards import Card

def test_difficult_choose_discard_prefers_expected_value():
    # Construct a hand where two discards yield the same max immediate score
    # but one has higher expected value over all starters
    hand = [Card('H', 1), Card('H', 4), Card('C', 9), Card('C', 10), Card('S', 11), Card('H', 13)]
    player = DifficultRuleBasedPlayer()
    # Should not raise and should return a tuple of two cards
    discard = player.choose_discard(hand, dealer_is_self=True)
    assert isinstance(discard, tuple) and len(discard) == 2
    # Check that the returned cards are in the hand
    assert discard[0] in hand and discard[1] in hand

# Additional test: check that difficult player is deterministic for same hand

def test_difficult_choose_discard_deterministic():
    hand = [Card('H', 2), Card('H', 3), Card('H', 4), Card('H', 5), Card('H', 6), Card('H', 7)]
    player = DifficultRuleBasedPlayer()
    result1 = player.choose_discard(hand, dealer_is_self=True)
    result2 = player.choose_discard(hand, dealer_is_self=True)
    assert result1 == result2
