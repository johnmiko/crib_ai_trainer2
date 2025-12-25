from typing import List, Sequence
from logging import getLogger
from crib_ai_trainer2.cards import Card

logger = getLogger(__name__)

# Utility for rank value in fifteens and runs (Ace=1)
RANK_VALUE = {**{i: i for i in range(1, 10)}, 10: 10, 11: 10, 12: 10, 13: 10}

def _sum_values(cards: Sequence[Card]) -> int:
    return sum(RANK_VALUE[c.rank] for c in cards)

def score_fifteens(cards: Sequence[Card]) -> int:
    # all combos summing to 15
    points = 0
    n = len(cards)
    for mask in range(1, 1 << n):
        subset = [cards[i] for i in range(n) if (mask >> i) & 1]
        if _sum_values(subset) == 15:
            points += 2
    return points

def score_pairs(cards: Sequence[Card]) -> int:
    counts = {}
    for c in cards:
        counts[c.rank] = counts.get(c.rank, 0) + 1
    points = 0
    for k in counts.values():
        if k >= 2:
            # 2->2, 3->6, 4->12
            points += {2: 2, 3: 6, 4: 12}.get(k, 0)
    return points

def score_runs(cards: Sequence[Card]) -> int:
    # runs length >=3; account for multiple cards of same rank via multiplicity
    # approach: count ranks, find maximal contiguous sequences, multiply by product of multiplicities
    rank_counts = {r: 0 for r in range(1, 14)}
    for c in cards:
        rank_counts[c.rank] += 1
    points = 0
    # search for runs
    r = 1
    while r <= 13:
        if rank_counts[r] == 0:
            r += 1
            continue
        start = r
        mult = rank_counts[r]
        length = 1
        r += 1
        while r <= 13 and rank_counts[r] > 0:
            mult *= rank_counts[r]
            length += 1
            r += 1
        if length >= 3:
            points += length * mult
    return points

def score_flush(cards: Sequence[Card], starter: Card, is_crib: bool) -> int:
    # hand flush: 4 for hand if all same suit; with starter: 5
    suits = [c.suit for c in cards]
    if len(set(suits)) == 1:
        if starter.suit == suits[0]:
            return 5
        # crib requires starter match to count at all
        return 4 if not is_crib else 0
    return 0

def score_nobs(cards: Sequence[Card], starter: Card) -> int:
    # jack of hand matching starter suit
    return 1 if any(c.rank == 11 and c.suit == starter.suit for c in cards) else 0

def score_hand(hand: Sequence[Card], starter: Card, is_crib: bool) -> int:
    all_cards = list(hand) + [starter]
    return (
        score_fifteens(all_cards)
        + score_pairs(all_cards)
        + score_runs(all_cards)
        + score_flush(hand, starter, is_crib)
        + score_nobs(hand, starter)
    )

# Pegging scoring utilities

def score_pegging_play(sequence_since_reset: List[Card], new_card: Card, count: int) -> int:
    # returns points for playing new_card (pairs, runs, 15/31, last card handled by caller)
    points = 0
    new_count = count + RANK_VALUE[new_card.rank]
    if new_count == 15:
        points += 2
    if new_count == 31:
        points += 2
    # pair/run detection on recent tail including new_card
    tail = sequence_since_reset + [new_card]
    # pairs: check last k cards of same rank
    k = 1
    r = new_card.rank
    i = len(tail) - 2
    while i >= 0 and tail[i].rank == r:
        k += 1
        i -= 1
    if k in (2, 3, 4):
        points += {2: 2, 3: 6, 4: 12}[k]
    # runs: longest run ending at tail
    # try lengths from 7 down to 3
    for L in range(min(7, len(tail)), 2, -1):
        window = tail[-L:]
        ranks = [c.rank for c in window]
        # normalize multiplicities: runs require distinct ranks
        if len(set(ranks)) != len(ranks):
            continue
        sr = sorted(ranks)
        ok = all(sr[i] + 1 == sr[i + 1] for i in range(len(sr) - 1))
        if ok:
            points += L
            break
    return points
