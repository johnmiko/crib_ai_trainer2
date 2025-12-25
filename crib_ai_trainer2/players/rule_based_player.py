from typing import List, Tuple, Optional
from logging import getLogger
from crib_ai_trainer2.cards import Card
from crib_ai_trainer2.scoring import score_hand, RANK_VALUE, score_pegging_play

logger = getLogger(__name__)

class RuleBasedPlayer:
    def __init__(self, name: str = "reasonable"):
        self.name = name

    def choose_discard(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        # choose discard maximizing expected hand score after cut (approx: without starter)
        best = None
        best_score = -1
        n = len(hand)
        for i in range(n):
            for j in range(i + 1, n):
                kept = [hand[k] for k in range(n) if k not in (i, j)]
                # approximate by using a neutral starter (no extra points)
                score = score_hand(kept, kept[0], is_crib=False) - score_hand([hand[i], hand[j]], kept[0], is_crib=True)
                if score > best_score:
                    best_score = score
                    best = (hand[i], hand[j])
        return best  # type: ignore

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        # always take points if available; else play lowest that doesn't set opponent up
        best = None
        best_pts = -1
        for c in playable:
            pts = score_pegging_play(history_since_reset, c, count)
            if pts > best_pts:
                best_pts = pts
                best = c
        if best is not None:
            return best
        # otherwise play lowest value
        return sorted(playable, key=lambda c: RANK_VALUE[c.rank])[0] if playable else None
