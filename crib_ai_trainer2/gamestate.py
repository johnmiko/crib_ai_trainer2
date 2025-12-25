from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any
import copy
import json
import hashlib
from .cards import Card

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
    # Add more fields as needed for full state

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
