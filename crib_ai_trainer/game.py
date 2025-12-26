from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from logging import getLogger
from cribbage.playingcards import Card, Deck
from cribbage.cribbagegame import score_hand, score_play as score_pegging_play
from crib_ai_trainer.constants import RANK_VALUE
from crib_ai_trainer.gamestate import GameState

logger = getLogger(__name__)

@dataclass
class PlayerInterface:
    name: str
    def select_crib_cards(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        raise NotImplementedError
    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        raise NotImplementedError

# class CribbageGame:
#     def __init__(self, p0: PlayerInterface, p1: PlayerInterface, seed: int | None = None):
#         self.p0 = p0
#         self.p1 = p1
#         self.deck = Deck(seed)
#         self.deck.shuffle()
#         # Initialize canonical GameState
#         self.state = GameState(
#             hands=[[], []],
#             crib=[],
#             starter=None,
#             played=[[], []],
#             scores=[0, 0],
#             dealer=0,
#             count=0,
#             history_since_reset=[],
#             round_num=0,
#             round_history=[]
#         )

#     def play_game(self) -> Tuple[int, int]:
#         # play until someone reaches 121
#         while max(self.state.scores) < 121:
#             self.play_round()
#         return self.state.scores[0], self.state.scores[1]

#     def discard_to_crib_phase(self) -> None:
#         # players discard 2 cards each to crib
#         self.state.crib = []
#         # players discard 2
#         d0 = self.p0.select_crib_cards(self.state.hands[0], dealer_is_self=(self.state.dealer == 0))
#         d1 = self.p1.select_crib_cards(self.state.hands[1], dealer_is_self=(self.state.dealer == 1))
#         for c in d0:
#             self.state.hands[0].remove(c)
#             self.state.crib.append(c)
#         for c in d1:
#             self.state.hands[1].remove(c)
#             self.state.crib.append(c)

#     def play_round(self) -> None:
#         self.deck.reset()
#         self.deck.shuffle()
#         self.state.hands = [self.deck.deal(6), self.deck.deal(6)]
#         self.discard_to_crib_phase()
#         self.state.starter = self.deck.cut()
#         self.state.played = [[], []]
#         self.state.count = 0
#         self.state.history_since_reset = []
#         self.pegging_phase()
#         # count hands
#         self.state.scores[0] += score_hand(self.state.hands[0], self.state.starter, is_crib=False)
#         self.state.scores[1] += score_hand(self.state.hands[1], self.state.starter, is_crib=False)
#         # count crib (dealer's)
#         self.state.scores[self.state.dealer] += score_hand(self.state.crib, self.state.starter, is_crib=True)
#         # alternate dealer
#         self.state.dealer = 1 - self.state.dealer
#         self.state.round_num += 1

#     def pegging_phase(self) -> None:
#         # pegging: play cards without exceeding 31, go logic, reset on 31 or both go
#         self.state.round_history.append([])
#         hands = self.state.hands
#         count = 0
#         history_reset: List[Card] = []
#         played_out = [False, False]
#         turn = 1 - self.state.dealer  # pone starts
#         passes = [False, False]
#         while not all(played_out):
#             playable = [c for c in hands[turn] if count + RANK_VALUE[c.rank] <= 31]
#             if playable:
#                 card = (self.p0 if turn == 0 else self.p1).play_pegging(playable, count, history_reset)
#                 if card is None or card not in playable:
#                     raise ValueError(f"Player {turn} played invalid card {card} at count {count}")
#                 hands[turn].remove(card)
#                 count += RANK_VALUE[card.rank]
#                 points = score_pegging_play(history_reset, card, count - RANK_VALUE[card.rank])
#                 self.state.scores[turn] += points
#                 history_reset.append(card)
#                 self.state.round_history[-1].append(card)
#                 passes = [False, False]
#                 if count == 31:
#                     # last card bonus to current player
#                     self.state.scores[turn] += 1
#                     count = 0
#                     history_reset = []
#                 played_out[turn] = len(hands[turn]) == 0
#             else:
#                 # go
#                 passes[turn] = True
#                 if all(passes):
#                     # last card bonus to last player who played
#                     if history_reset:
#                         last_player = 1 - turn
#                         self.state.scores[last_player] += 1
#                     count = 0
#                     history_reset = []
#                     passes = [False, False]
#                 else:
#                     # switch turn
#                     pass
#             turn = 1 - turn
