import types
from crib_ai_trainer.game import CribbageGame, PlayerInterface
from crib_ai_trainer.cards import Card
from crib_ai_trainer.gamestate import GameState
from crib_ai_trainer.scoring import score_hand, score_fifteens, score_pairs, score_runs, score_nobs

class DummyPlayer(PlayerInterface):
    def __init__(self, hand):
        self.name = "dummy"
        self._hand = hand
    def choose_discard(self, hand, dealer_is_self):
        # Always discard the first two
        return hand[0], hand[1]
    def play_pegging(self, playable, count, history):
        return playable[0] if playable else None

def test_basic_gameplay_works_as_expected():
    # Player 1: H1-H6, Player 2: D1-D6
    hand1 = [Card('H', i) for i in range(1, 7)]
    hand2 = [Card('D', i) for i in range(1, 7)]
    p0 = DummyPlayer(hand1)
    p1 = DummyPlayer(hand2)
    game = CribbageGame(p0, p1)
    # Monkeypatch hands and dealer
    game.state.hands = [hand1.copy(), hand2.copy()]
    game.state.dealer = 1  # Player 1 is dealer (0-based)
    # Check initial state  
    expected_state = GameState( 
        hands=[hand1, hand2],
        crib=[],
        starter=None,
        played=[[], []],
        scores=[0, 0],
        dealer=1,
        count=0,
        history_since_reset=[],
        round_num=0,
        round_history=[]
    )

    game.discard_to_crib_phase()
    expected_state.crib = [hand1[0], hand1[1], hand2[0], hand2[1]]
    expected_state.hands = [
        [Card('H', i) for i in range(3, 7)],
        [Card('D', i) for i in range(3, 7)],
    ]    
    assert game.state == expected_state
    # Proceed to cut and pegging phase    
    game.state.starter = Card('S', 7)
    game.state.played = [[], []]
    game.state.count = 0
    game.state.history_since_reset = []
    game.pegging_phase()
    expected_state.count = 21  # All cards played
    expected_state.played = [
        [Card('H', i) for i in range(3, 7)],
        [Card('D', i) for i in range(3, 7)],
    ]
    expected_state.round_history = [
        [Card('H', 3), Card('D', 3), Card('H', 4), Card('D', 4), Card('H', 5), Card('D', 5), Card('H', 6), Card('D', 6)],
    ]
    assert game.state == expected_state

    
    


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

def test_score_pegging_play_15_scores_2_points():
    seq = [Card('H', 5), Card('D', 5)]
    count = 10
    new = Card('S', 5)
    pts = score_pegging_play(seq, new, count)
    assert pts == 2
