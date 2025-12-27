import pytest

from crib_ai_trainer.players.rule_based_player import RuleBasedPlayer
from cribbage.playingcards import Card, CardFactory

@pytest.fixture
def player():    
    return RuleBasedPlayer()

def test_select_crib_cards(player):    
    hand = CardFactory.create_hand_from_strs(['ah', 'ad', 'as', '2h', '2d', 'ac'])
    discards = player.select_crib_cards(hand, dealer_is_self=True)
    assert discards == (Card('two', 'hearts'), Card('two', 'diamonds'))    