import unittest

import pytest

from cribbage import cribbagegame
from cribbage.player import RandomPlayer
from cribbage.playingcards import Card, Deck

from crib_ai_trainer.players.neural_player import LinearValueModel, NeuralPlayer
from crib_ai_trainer.players.rule_based_player import ReasonablePlayer


@pytest.fixture
def setUp():
        players = [ReasonablePlayer(), NeuralPlayer(LinearValueModel(105), LinearValueModel(188))]
        game = cribbagegame.CribbageGame(players=players)
        round = cribbagegame.CribbageRound(game, dealer=game.players[0])
        return game, round

class TestCribbageRound():    
    def test_get_crib(self, setUp):
        game, round = setUp
        round._deal()
        round._populate_crib()

    def test_cut(self, setUp):
        game, round = setUp
        round._cut()

    def test_get_table_value(self, setUp):
        game, round = setUp
        round.table = []
        total = round.get_table_value(0)
        assert total == 0
        round.table = [Card('7h')]
        total = round.get_table_value(0)
        assert total == 7

    def test_play_round(self, setUp):
        game, round = setUp
        round.play()


if __name__ == '__main__':
    unittest.main()
