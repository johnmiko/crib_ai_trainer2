from cribbage.playingcards import Card, Deck
import numpy as np
from crib_ai_trainer.features import multi_hot_cards



def test_multi_hot_cards():
    hand = [Card('ah'), Card('kd'), Card('7s'), Card('as'), Card('ad'), Card('ac')]
    vec = multi_hot_cards(hand)
    assert vec.sum() == 6.0
    expected_vec = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0]
    assert np.array_equal(vec, expected_vec)