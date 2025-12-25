from crib_ai_trainer2.cards import Deck

def test_seeded_sampling_determinism():
    d1 = Deck(seed=123)
    d1.shuffle()
    h1 = d1.deal(6)
    d2 = Deck(seed=123)
    d2.shuffle()
    h2 = d2.deal(6)
    assert h1 == h2
