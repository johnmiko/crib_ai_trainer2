import pytest

from cribbage.players.rule_based_player import BeginnerPlayer
from cribbage.playingcards import Card, build_hand

@pytest.fixture
def player():    
    return BeginnerPlayer()

def test_select_crib_cards(player): 
    hand = build_hand(["ah","ad","2h","2d","ac","as"])
    discards = player.select_crib_cards(hand, dealer_is_self=True)
    assert discards == (Card('2h'), Card('2d'))    

def test_select_card_to_play(player):
    hand = build_hand(["5h","7d","9c","js"])
    playable = [Card('5h'), Card('7d'), Card('9c')]
    card = player.select_card_to_play(playable, count=15, table=[], crib=None)
    assert card == Card('5h')  # plays 5 for 20

    playable = [Card('7d'), Card('9c')]
    card = player.select_card_to_play(playable, count=20, table=[], crib=None)
    assert card == Card('7d')  # plays 7 for 27

    playable = [Card('9c')]
    card = player.select_card_to_play(playable, count=27, table=[], crib=None)
    assert card == None  # cannot play without going over 31