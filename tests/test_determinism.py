
import random
from cribbage.playingcards import Deck

def test_seeded_sampling_determinism():    
    d1 = Deck(123)
    d1.shuffle()
    h1 = [d1.draw() for _ in range(6)]    
    d2 = Deck(123)
    d2.shuffle()
    h2 = [d2.draw() for _ in range(6)]
    assert h1 == h2
