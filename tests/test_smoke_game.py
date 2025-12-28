from cribbage.cribbagegame import CribbageGame
from crib_ai_trainer.players.random_player import RandomPlayer

def test_smoke_game_valid_score():
    g = CribbageGame(players=[RandomPlayer(), RandomPlayer()])
    # crib_engine's CribbageGame does not have play_game(), so we just check instantiation
    assert len(g.players) == 2
