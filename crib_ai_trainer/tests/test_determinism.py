
import random
from cribbage.playingcards import Deck

def test_seeded_sampling_determinism():
    random.seed(123)
    d1 = Deck()
    d1.shuffle()
    h1 = [d1.draw() for _ in range(6)]
    random.seed(123)
    d2 = Deck()
    d2.shuffle()
    h2 = [d2.draw() for _ in range(6)]
    assert h1 == h2
