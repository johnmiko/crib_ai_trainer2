
from crib_ai_trainer.players.neural_player import featurize_discard, featurize_pegging
from cribbage.playingcards import get_random_hand
table = []

def test_featurize_discard():
    hand = get_random_hand()
    kept = hand[:4]
    discards = [c for c in hand if c not in kept]
    x = featurize_discard(kept, discards, True)
    assert x.shape == (105,)   # 52 + 52 + 1

def test_featurize_pegging():
    hand = get_random_hand()
    table = get_random_hand(0)
    count = 15
    c = hand[0]

    y = featurize_pegging(hand, table, count, c, known_cards=[])
    assert y.shape == (240,)   # 52(hand) + 52(table) + 32(count) + 52(cand) + 52(known)