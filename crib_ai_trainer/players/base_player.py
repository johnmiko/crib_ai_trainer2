from abc import ABC
from abc import abstractmethod

class Player(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def select_crib_cards(self, hand, dealer_is_self, game_state=None):
        return NotImplemented

    @abstractmethod
    def select_card_to_play(self, hand, table, crib, count, game_state=None):
        # pegging decision
        return NotImplemented