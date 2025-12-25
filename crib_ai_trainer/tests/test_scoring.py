from crib_ai_trainer.cards import Card
from crib_ai_trainer.scoring import score_hand, score_fifteens, score_pairs, score_runs, score_nobs

def test_flush_scoring_variations():
    # Hand flush, starter matches (5 points)
    hand = [Card('H', 2), Card('H', 3), Card('H', 4), Card('H', 5)]
    starter = Card('H', 6)
    assert score_hand(hand, starter, is_crib=False) - (
        score_fifteens(hand + [starter]) + score_pairs(hand + [starter]) + score_runs(hand + [starter]) + score_nobs(hand, starter)
    ) == 5

    # Hand flush, starter does not match (4 points)
    hand = [Card('S', 2), Card('S', 3), Card('S', 4), Card('S', 5)]
    starter = Card('H', 6)
    assert score_hand(hand, starter, is_crib=False) - (
        score_fifteens(hand + [starter]) + score_pairs(hand + [starter]) + score_runs(hand + [starter]) + score_nobs(hand, starter)
    ) == 4

    # Crib flush, starter matches (5 points)
    hand = [Card('D', 2), Card('D', 3), Card('D', 4), Card('D', 5)]
    starter = Card('D', 6)
    assert score_hand(hand, starter, is_crib=True) - (
        score_fifteens(hand + [starter]) + score_pairs(hand + [starter]) + score_runs(hand + [starter]) + score_nobs(hand, starter)
    ) == 5

    # Crib flush, starter does not match (0 points)
    hand = [Card('C', 2), Card('C', 3), Card('C', 4), Card('C', 5)]
    starter = Card('H', 6)
    assert score_hand(hand, starter, is_crib=True) - (
        score_fifteens(hand + [starter]) + score_pairs(hand + [starter]) + score_runs(hand + [starter]) + score_nobs(hand, starter)
    ) == 0
import pytest
from crib_ai_trainer.cards import Card
from crib_ai_trainer.scoring import score_hand, score_pegging_play

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
