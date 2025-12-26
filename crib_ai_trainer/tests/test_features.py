from cribbage.playingcards import Card, Deck
from crib_ai_trainer.features import multi_hot_cards



def test_multi_hot_cards():
    hand = [Card(Deck.RANKS["ace"], Deck.SUITS["hearts"]), Card(Deck.RANKS["king"], Deck.SUITS["diamonds"]), Card(Deck.RANKS["seven"], Deck.SUITS["spades"]),Card(Deck.RANKS["ace"], Deck.SUITS["spades"]), Card(Deck.RANKS["ace"], Deck.SUITS["diamonds"]),Card(Deck.RANKS["ace"], Deck.SUITS["clubs"])]
    vec = multi_hot_cards(hand)
    assert vec.sum() == 6.0
    assert vec[0] == 1.0  # AH
    assert vec[12] == 0.0  # KD
    assert vec[32 + 6] == 0.0  # 7S