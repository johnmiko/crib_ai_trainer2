import pytest
from crib_ai_trainer2.cards import Card
from crib_ai_trainer2.scoring import score_hand, score_pegging_play

# Basic scoring tests

def test_fifteens_pairs_runs_flush_nobs():
    hand = [Card('H', 5), Card('D', 5), Card('S', 5), Card('C', 5)]
    starter = Card('H', 10)
    points = score_hand(hand, starter, is_crib=False)
    # 4 of a kind = 12, multiple fifteens, no flush (mixed suits), nobs 0
    assert points >= 12

def test_pegging_scoring_basic():
    seq = [Card('H', 5), Card('D', 5)]
    count = 10
    new = Card('S', 5)
    pts = score_pegging_play(seq, new, count)
    # should get pair points (3 of a kind = 6)
    assert pts >= 6
