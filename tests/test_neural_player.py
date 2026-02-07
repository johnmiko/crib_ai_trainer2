
import sqlite3
import pytest

from crib_ai_trainer.players.neural_player import (
    featurize_discard,
    featurize_pegging,
    DISCARD_FEATURE_DIM,
    get_pegging_feature_dim,
)
from cribbage.constants import HAND_CRIB_DB_PATH
from cribbage.playingcards import get_random_hand
table = []

def test_featurize_discard():
    if not HAND_CRIB_DB_PATH:
        pytest.skip("HAND_CRIB_DB_PATH not set.")
    conn = sqlite3.connect(HAND_CRIB_DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(hand1)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()
    required = {"min_hand_score", "max_hand_score", "avg_hand_score"}
    if not required.issubset(cols):
        pytest.skip("hand1 table missing min/max/avg columns.")

    hand = get_random_hand()
    kept = hand[:4]
    discards = [c for c in hand if c not in kept]
    x = featurize_discard(kept, discards, True)
    assert x.shape == (DISCARD_FEATURE_DIM,)

def test_featurize_pegging():
    hand = get_random_hand()
    table = get_random_hand(0)
    count = 15
    c = hand[0]

    y = featurize_pegging(hand, table, count, c, known_cards=[])
    assert y.shape == (get_pegging_feature_dim("full"),)
