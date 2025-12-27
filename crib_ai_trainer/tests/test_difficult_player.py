# test_difficult_player.py
import pytest

from crib_ai_trainer.players.rule_based_player import DifficultRuleBasedPlayer, generate_hand_ranges, get_full_deck, remaining_deck
from cribbage.playingcards import Deck, Card
from itertools import combinations
from math import comb
from typing import Iterable, List, Tuple

def test_remaining_deck_size():
    full = get_full_deck()
    assert len(full) == 52
    hand = full[:6]
    rem = remaining_deck(full, hand)
    assert len(rem) == 46
    assert all(c not in hand for c in rem)


def test_generate_possible_starters_counts():
    hand = [Card(Deck.RANKS["ace"], Deck.SUITS["hearts"]),
            Card(Deck.RANKS["four"], Deck.SUITS["hearts"]),
              Card(Deck.RANKS["nine"], Deck.SUITS["clubs"]),
              Card(Deck.RANKS["ten"], Deck.SUITS["clubs"]), 
              Card(Deck.RANKS["jack"], Deck.SUITS["spades"]),
              Card(Deck.RANKS["king"], Deck.SUITS["hearts"])]
    hand_scores = generate_hand_ranges(hand)    
    breakpoint()
    assert False


def test_generate_possible_cribs_counts():
    full = get_full_deck()
    hand = full[:6]
    discards = hand[:2]
    cribs = list(generate_possible_cribs(discards, full, hand))
    # opponent chooses 2 from remaining 46
    assert len(cribs) == comb(46, 2)
    # each crib has 4 cards
    assert all(len(crib) == 4 for crib, _ in cribs)
    # probabilities should sum to ~1
    total_p = sum(p for _, p in cribs)
    assert pytest.approx(total_p, rel=1e-9) == 1.0