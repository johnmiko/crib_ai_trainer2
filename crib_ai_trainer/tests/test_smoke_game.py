from crib_ai_trainer.game import CribbageGame
from crib_ai_trainer.players.random_player import RandomPlayer

def test_smoke_game_valid_score():
    g = CribbageGame(RandomPlayer(), RandomPlayer())
    s0, s1 = g.play_game()
    assert (s0 >= 0 and s1 >= 0) and (max(s0, s1) >= 121)
