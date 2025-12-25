from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from logging import getLogger
from .cards import Card, Deck
from .scoring import score_hand, score_pegging_play, RANK_VALUE

logger = getLogger(__name__)

@dataclass
class PlayerInterface:
    name: str
    def choose_discard(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        raise NotImplementedError
    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        raise NotImplementedError

class CribbageGame:
    def __init__(self, p0: PlayerInterface, p1: PlayerInterface, seed: int | None = None):
        self.p0 = p0
        self.p1 = p1
        self.scores = [0, 0]
        self.dealer = 0
        self.deck = Deck(seed)
        self.deck.shuffle()

    def play_game(self) -> Tuple[int, int]:
        # play until someone reaches 121
        while max(self.scores) < 121:
            self.play_round()
        return self.scores[0], self.scores[1]

    def play_round(self) -> None:
        # new shuffled deck each round
        self.deck.reset()
        self.deck.shuffle()
        # deal 6 each
        hands = [self.deck.deal(6), self.deck.deal(6)]
        crib: List[Card] = []
        # players discard 2
        d0 = self.p0.choose_discard(hands[0], dealer_is_self=(self.dealer == 0))
        d1 = self.p1.choose_discard(hands[1], dealer_is_self=(self.dealer == 1))
        for c in d0:
            hands[0].remove(c)
            crib.append(c)
        for c in d1:
            hands[1].remove(c)
            crib.append(c)
        # cut starter
        starter = self.deck.cut()
        # pegging phase
        self.pegging_phase(hands, starter)
        # count hands
        self.scores[0] += score_hand(hands[0], starter, is_crib=False)
        self.scores[1] += score_hand(hands[1], starter, is_crib=False)
        # count crib (dealer's)
        self.scores[self.dealer] += score_hand(crib, starter, is_crib=True)
        # alternate dealer
        self.dealer = 1 - self.dealer

    def pegging_phase(self, hands: List[List[Card]], starter: Card) -> None:
        # pegging: play cards without exceeding 31, go logic, reset on 31 or both go
        count = 0
        history_reset: List[Card] = []
        played_out = [False, False]
        turn = 1 - self.dealer  # pone starts
        passes = [False, False]
        while not all(played_out):
            playable = [c for c in hands[turn] if count + RANK_VALUE[c.rank] <= 31]
            if playable:
                card = (self.p0 if turn == 0 else self.p1).play_pegging(playable, count, history_reset)
                if card is None or card not in playable:
                    # default: play lowest legal
                    card = sorted(playable, key=lambda c: RANK_VALUE[c.rank])[0]
                hands[turn].remove(card)
                count += RANK_VALUE[card.rank]
                points = score_pegging_play(history_reset, card, count - RANK_VALUE[card.rank])
                self.scores[turn] += points
                history_reset.append(card)
                passes = [False, False]
                if count == 31:
                    # last card bonus to current player
                    self.scores[turn] += 1
                    count = 0
                    history_reset = []
                played_out[turn] = len(hands[turn]) == 0
            else:
                # go
                passes[turn] = True
                if all(passes):
                    # last card bonus to last player who played
                    if history_reset:
                        last_player = 1 - turn
                        self.scores[last_player] += 1
                    count = 0
                    history_reset = []
                    passes = [False, False]
                else:
                    # switch turn
                    pass
            turn = 1 - turn
