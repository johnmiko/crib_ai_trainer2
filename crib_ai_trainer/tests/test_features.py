from cribbage.playingcards import Card, Deck
from crib_ai_trainer.features import multi_hot_cards



def test_multi_hot_cards():
    # hand = [Card('H', 1), Card('D', 13), Card('S', 7)]
    # hand = [9♥, 2♠, 9♣, 3♣, 4♦, 7♦]
    # hand = [Card('H', 9), Card('S', 2), Card('C', 9), Card('C', 3), Card('D', 4), Card('D', 7)]
    hand = [Card(Deck.SUITS["hearts"],9), Card(Deck.SUITS["spades"],2), Card(Deck.SUITS["clubs"],9), Card(Deck.SUITS["clubs"],3), Card(Deck.SUITS["diamonds"],4), Card(Deck.SUITS["diamonds"],7)]
    vec = multi_hot_cards(hand)
    assert vec.sum() == 3.0
    assert vec[0] == 1.0  # AH
    assert vec[12] == 1.0  # KD
    assert vec[32 + 6] == 1.0  # 7S