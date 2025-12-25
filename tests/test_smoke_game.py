import sys
import os
print(1)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(12)
from crib_ai_trainer2.game import CribbageGame
from crib_ai_trainer2.players.random_player import RandomPlayer
print(3)
def test_smoke_game_valid_score():
    print(4)
    g = CribbageGame(RandomPlayer(), RandomPlayer())
    s0, s1 = g.play_game()
    assert (s0 >= 0 and s1 >= 0) and (max(s0, s1) >= 121)
