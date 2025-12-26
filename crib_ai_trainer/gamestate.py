from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any
import copy
import json
import hashlib
from cribbage.playingcards import Card

@dataclass
class GameState:
    hands: List[List[Card]]
    crib: List[Card]
    starter: Optional[Card]
    played: List[List[Card]]
    scores: List[int]
    dealer: int
    count: int
    history_since_reset: List[Card]
    round_num: int = 0
    round_history: List[List[Card]] = field(default_factory=list)
    # round_history: list of lists, each sublist contains cards played in that round

    def clone(self) -> GameState:
        return copy.deepcopy(self)

    def serialize(self) -> str:
        # Use asdict and custom encoder for Card
        def card_encoder(obj: Any):
            if isinstance(obj, Card):
                return {'suit': obj.suit, 'rank': obj.rank}
            raise TypeError(f"Type {type(obj)} not serializable")
        return json.dumps(asdict(self), default=card_encoder, sort_keys=True)

    def hash(self) -> str:
        s = self.serialize()
        return hashlib.sha256(s.encode('utf-8')).hexdigest()
    
    def __eq__(self, other):
        if not isinstance(other, GameState):
            return NotImplemented
        return (
            self.hands == other.hands and
            self.crib == other.crib and
            self.starter == other.starter and
            self.played == other.played and
            self.scores == other.scores and
            self.dealer == other.dealer and
            self.count == other.count and
            self.history_since_reset == other.history_since_reset and
            self.round_num == other.round_num and
            self.round_history == other.round_history
        )
