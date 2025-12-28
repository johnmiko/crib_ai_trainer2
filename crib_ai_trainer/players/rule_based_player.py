from typing import List, Tuple, Optional
from logging import getLogger
from cribbage.playingcards import Card
from cribbage.cribbagegame import score_hand, score_play as score_pegging_play
from cribbage.playingcards import Deck
from itertools import combinations

from matplotlib.pylab import f
from numpy import full

logger = getLogger(__name__)


def get_possible_hands(hand: list[Card]) -> list[tuple[list[Card], list[Card]]]:
    """
    Given a 6-card hand, return all possible (cards_to_keep, crib_cards) pairs,
    where cards_to_keep is a list of 4 cards and crib_cards is the 2 cards put in the crib.
    """
    if len(hand) != 6:
        raise ValueError("Hand must have exactly 6 cards")
    all_combos = []
    for kept in combinations(hand, 4):
        crib = [c for c in hand if c not in kept]
        all_combos.append((list(kept), crib))
    return all_combos

def basic_pegging_strategy(playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
    # always take points if available; else play highest card
    best = None
    best_pts = -1
    for c in playable:
        sequence = history_since_reset + [c]
        pts = score_pegging_play(sequence)
        if (pts > best_pts) and (c + count <= 31):
            best_pts = pts
            best = c
    if best is not None:
        return best
    # otherwise play highest value        
    highest_card = playable[0]
    for c in playable:
        if c > highest_card:
            highest_card = c
    return highest_card

def basic_crib_strategy(hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
    # Counts points in hand and then adds/subtracts points in crib to decide
    best_discards: List[Tuple[Card, Card]] = []
    best_score = float("-inf")

    for kept in combinations(hand, 4):  # all 6-choose-4 possible hands
        kept = list(kept)
        starter = kept[0]  # same "neutral starter" hack as before

        # the 2 not in kept are the discards
        discards = [c for c in hand if c not in kept]
        assert len(discards) == 2

        kept_score = score_hand(kept, is_crib=False)
        crib_score = score_hand(discards, is_crib=True)

        # if you want to actually use dealer_is_self:
        score = kept_score + crib_score if dealer_is_self else kept_score - crib_score
        if score > best_score:
            best_score = score
            best_discards = [tuple(discards)]
        elif score == best_score:
            best_discards.append(tuple(discards))

    return best_discards[0]  # type: ignore


class BeginnerPlayer:
    def __init__(self, name: str = "beginner"):
        self.name = name

    def select_crib_cards(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        return basic_crib_strategy(hand, dealer_is_self)

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        return basic_pegging_strategy(playable, count, history_since_reset)

    def select_card_to_play(self, hand: List[Card], table, crib, count: int):
        # table is the list of cards currently on the table
        playable_cards = [c for c in hand if c + count <= 31]
        if not playable_cards:
            return None
        return self.play_pegging(playable_cards, count, table)

class DifficultPlayer(BeginnerPlayer):
    def __init__(self, name: str = "difficult"):
        super().__init__(name=name)

    def select_crib_cards(self, hand: List["Card"], dealer_is_self: bool) -> Tuple["Card", "Card"]:
        """
        Choose discard using expected value over:
          - all possible starters for the kept hand
          - all possible opponent discards + starters for the crib
        """
        assert len(hand) == 6
        full_deck = get_full_deck()

        best_discards: List[Tuple["Card", "Card"]] = []
        best_score = float("-inf")

        crib_ranges = generate_crib_ranges(hand)
        hand_ranges = generate_hand_ranges(hand)

        return best_discards[0]  # tie-break by first


from itertools import combinations
from math import comb
from typing import Iterable, List, Tuple
# from your_module import Deck, Card, score_hand


def get_full_deck() -> List["Card"]:
    return [Card(f"{r}{s}") for r in Deck.RANKS for s in Deck.SUITS]


def remaining_deck(full_deck: List["Card"], exclude: Iterable["Card"]) -> List["Card"]:
    exclude_set = set(exclude)
    return [c for c in full_deck if c not in exclude_set]


def generate_possible_starters(
    full_deck: List["Card"],
    known_hand: List["Card"],
):
    """
    All possible starter cards given the known 6-card hand.
    Each starter is any card not in known_hand, all equally likely.
    """
    starters = remaining_deck(full_deck, known_hand)
    return starters


def generate_crib_ranges(
    known_hand: List["Card"],
):
    """
    Crib = our 2 discards + opponent’s 2 random discards
    Opponent’s discards are 2 random cards from the remaining deck (not in known_hand).
    Returns (crib_cards, probability) for all possible cribs (before starter).
    """
    full_deck = get_full_deck()
    pool = remaining_deck(full_deck, known_hand)
    n = len(pool)
    if n < 2:
        return []
    # total = comb(n, 3)
    # p = 1.0 / total
    all_hand_scores = {}
    for hand in combinations(known_hand, 2):
        hand_score = {hand: {"min": 0, "max": 0, "average": 0.0}}
        scores = []
        for opp_discards_and_starter in combinations(pool, 3):        
            possible_hand = hand + opp_discards_and_starter
            scores.append(score_hand(possible_hand, is_crib=True))
        hand_score[hand]["min"] = min(scores)
        hand_score[hand]["max"] = max(scores)
        hand_score[hand]["average"] = sum(scores) / len(scores)
        all_hand_scores.update(hand_score)
    return all_hand_scores

def generate_hand_ranges(
    known_hand: List["Card"],
):
    """
    Crib = our 2 discards + opponent’s 2 random discards
    Opponent’s discards are 2 random cards from the remaining deck (not in known_hand).
    Returns (crib_cards, probability) for all possible cribs (before starter).
    """
    full_deck = get_full_deck()
    pool = remaining_deck(full_deck, known_hand)
    n = len(pool)
    if n < 2:
        return []
    # total = comb(n, 3)
    # p = 1.0 / total
    all_hand_scores = {}
    for hand in combinations(known_hand, 4):
        crib_cards = remaining_deck(known_hand, hand)
        hand_score = {hand: {"min": 0, "max": 0, "average": 0.0, "crib_average": 0.0}}
        scores = []
        for starter in combinations(pool, 1):        
            possible_hand = list(hand) + list(starter)
            scores.append(score_hand(possible_hand, is_crib=False))
            possible_opponent_discards = remaining_deck(get_full_deck(), list(known_hand) + list(starter))
            crib_scores = []
            for opp_discards in combinations(possible_opponent_discards, 2):
                possible_crib = list(crib_cards) + list(opp_discards)
                crib_scores.append(score_hand(possible_crib + list(starter), is_crib=True))
        hand_score[hand]["min"] = min(scores)
        hand_score[hand]["max"] = max(scores)
        hand_score[hand]["average"] = sum(scores) / len(scores)
        hand_score[hand]["crib_average"] = sum(crib_scores) / len(crib_scores) if crib_scores else 0.0
        all_hand_scores.update(hand_score)
    return all_hand_scores


def expected_kept_score(
    kept: List["Card"],
    full_deck: List["Card"],
    known_hand: List["Card"],
) -> float:
    """
    E[hand score] over all possible starters, given we keep these 4 cards.
    Starters are any card not in the known 6-card hand.
    """
    starters = generate_possible_starters(full_deck, known_hand)
    if not starters:
        return 0.0
    ev = 0.0
    return ev


def expected_crib_score(
    discards: List["Card"],
    full_deck: List["Card"],
    known_hand: List["Card"],
    dealer_is_self: bool,
) -> float:
    """
    E[crib score] over:
      - all possible opponent discards (2 from remaining deck),
      - all possible starters consistent with known_hand + those discards.
    """
    ev = 0.0
    pool = remaining_deck(full_deck, known_hand)
    n = len(pool)
    if n < 2:
        return 0.0

    total_opp = comb(n, 2)
    p_opp = 1.0 / total_opp

    for opp_discards in combinations(pool, 2):
        crib = list(discards) + list(opp_discards)
        # starter cannot be in known_hand or in opp_discards
        starter_pool = remaining_deck(full_deck, list(known_hand) + list(opp_discards))
        if not starter_pool:
            continue
        p_starter = 1.0 / len(starter_pool)
        for starter in starter_pool:
            s = score_hand(crib + [starter], is_crib=True)
            ev += p_opp * p_starter * s

    return ev if dealer_is_self else -ev
