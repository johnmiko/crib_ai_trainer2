from itertools import combinations
from functools import lru_cache
import math
import numpy as np
from typing import List, Tuple

from cribbage.playingcards import Card
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.cribbagegame import score_play
from cribbage.scoring import HasPairTripleQuad, HasStraight_DuringPlay
from cribbage.players.rule_based_player import get_full_deck
from cribbage.strategies.pegging_strategies import medium_pegging_strategy
from cribbage.strategies.hand_strategies import exact_hand_and_min_crib

from crib_ai_trainer.features import multi_hot_cards

RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
RANK_TO_I = {r: i for i, r in enumerate(RANKS)}
TENS_RANKS = {"10", "j", "q", "k"}
_SUITS = ["h", "d", "c", "s"]
SUIT_TO_I = {s: i for i, s in enumerate(_SUITS)}
_FULL_DECK = get_full_deck()

# Base discard features (52 discards + 52 kept + 1 dealer flag)
BASE_DISCARD_FEATURE_DIM = 105

# Engineered discard features count:
# 13 kept rank counts + 13 discard rank counts +
# 2 value sums + 2 tens counts + 2 fives counts +
# 3 kept pair/trip/quad + 3 discard pair/trip/quad +
# 3 run counts (3/4/5) + 1 run max +
# 2 flush flags (kept/discard) + 1 nobs + 2 fifteen counts +
# 3 pegging EV features (self/opp/diff) +
# 2 scores + 1 score margin + 3 endgame flags
ENGINEERED_DISCARD_NO_SCORE_BASE_DIM = 47
ENGINEERED_DISCARD_PEGGING_EV_DIM = 3
ENGINEERED_DISCARD_NO_SCORE_DIM = ENGINEERED_DISCARD_NO_SCORE_BASE_DIM + ENGINEERED_DISCARD_PEGGING_EV_DIM
ENGINEERED_DISCARD_SCORE_DIM = 6
ENGINEERED_DISCARD_FEATURE_DIM = ENGINEERED_DISCARD_NO_SCORE_DIM + ENGINEERED_DISCARD_SCORE_DIM

DISCARD_FEATURE_DIM = BASE_DISCARD_FEATURE_DIM + ENGINEERED_DISCARD_FEATURE_DIM

# Pegging features
PEGGING_BASE_FEATURE_DIM = 240  # hand(52) + table(52) + count(32) + candidate(52) + known(52)
PEGGING_ENGINEERED_NO_SCORE_DIM = 32
PEGGING_ENGINEERED_SCORE_DIM = 6
PEGGING_ENGINEERED_FEATURE_DIM = PEGGING_ENGINEERED_NO_SCORE_DIM + PEGGING_ENGINEERED_SCORE_DIM
PEGGING_FULL_FEATURE_DIM = PEGGING_BASE_FEATURE_DIM + 52 + 52 + PEGGING_ENGINEERED_FEATURE_DIM  # + opp_played + all_played
PEGGING_SEQ_LEN = 8
PEGGING_SEQ_STEP_DIM = 84  # card(52) + count(32)
PEGGING_SEQ_FEATURE_DIM = PEGGING_SEQ_LEN * PEGGING_SEQ_STEP_DIM
PEGGING_FULL_SEQ_FEATURE_DIM = PEGGING_FULL_FEATURE_DIM + PEGGING_SEQ_FEATURE_DIM

def get_pegging_feature_dim(feature_set: str) -> int:
    if feature_set == "basic":
        return PEGGING_BASE_FEATURE_DIM
    if feature_set == "full":
        return PEGGING_FULL_FEATURE_DIM
    if feature_set == "full_seq":
        return PEGGING_FULL_SEQ_FEATURE_DIM
    if feature_set == "full_no_scores":
        return PEGGING_BASE_FEATURE_DIM + 52 + 52 + PEGGING_ENGINEERED_NO_SCORE_DIM
    raise ValueError(f"Unknown pegging feature_set: {feature_set}")


def get_discard_feature_indices(feature_set: str) -> np.ndarray:
    if feature_set == "base":
        return np.arange(BASE_DISCARD_FEATURE_DIM, dtype=np.int64)
    if feature_set == "engineered_no_scores":
        end = BASE_DISCARD_FEATURE_DIM + ENGINEERED_DISCARD_NO_SCORE_BASE_DIM
        return np.arange(end, dtype=np.int64)
    if feature_set == "engineered_no_scores_pev":
        end = BASE_DISCARD_FEATURE_DIM + ENGINEERED_DISCARD_NO_SCORE_DIM
        return np.arange(end, dtype=np.int64)
    if feature_set == "full":
        end = BASE_DISCARD_FEATURE_DIM + ENGINEERED_DISCARD_NO_SCORE_BASE_DIM + ENGINEERED_DISCARD_SCORE_DIM
        return np.arange(end, dtype=np.int64)
    if feature_set == "full_pev":
        return np.arange(DISCARD_FEATURE_DIM, dtype=np.int64)
    raise ValueError(f"Unknown discard feature_set: {feature_set}")


def get_pegging_feature_indices(feature_set: str) -> np.ndarray:
    if feature_set == "base":
        return np.arange(PEGGING_BASE_FEATURE_DIM, dtype=np.int64)
    if feature_set == "full_no_scores":
        end = PEGGING_BASE_FEATURE_DIM + 52 + 52 + PEGGING_ENGINEERED_NO_SCORE_DIM
        return np.arange(end, dtype=np.int64)
    if feature_set == "full_seq":
        return np.arange(PEGGING_FULL_SEQ_FEATURE_DIM, dtype=np.int64)
    if feature_set == "full":
        return np.arange(PEGGING_FULL_FEATURE_DIM, dtype=np.int64)
    raise ValueError(f"Unknown pegging feature_set: {feature_set}")


def _rank_counts_key(cards: List[Card]) -> tuple[int, ...]:
    return tuple(sorted(RANK_TO_I[c.get_rank().lower()] for c in cards))


@lru_cache(maxsize=100_000)
def _rank_counts_cached(key: tuple[int, ...]) -> np.ndarray:
    counts = np.zeros(13, dtype=np.float32)
    for idx in key:
        counts[idx] += 1.0
    return counts


def _rank_counts(cards: List[Card]) -> np.ndarray:
    return _rank_counts_cached(_rank_counts_key(cards))


def _value_sum(cards: List[Card]) -> float:
    return float(sum(c.get_value() for c in cards))


def _count_tens(cards: List[Card]) -> float:
    return float(sum(1 for c in cards if c.get_rank().lower() in TENS_RANKS))


def _count_fives(cards: List[Card]) -> float:
    return float(sum(1 for c in cards if c.get_rank().lower() == "5"))


def _pair_trip_quad_counts(cards: List[Card]) -> Tuple[float, float, float]:
    counts = _rank_counts(cards)
    pair_count = float(np.sum(counts * (counts - 1.0) / 2.0))
    trip_count = float(np.sum(counts * (counts - 1.0) * (counts - 2.0) / 6.0))
    quad_count = float(np.sum(counts * (counts - 1.0) * (counts - 2.0) * (counts - 3.0) / 24.0))
    return pair_count, trip_count, quad_count


def _run_counts(cards: List[Card]) -> Tuple[float, float, float, float]:
    counts = _rank_counts(cards)
    run3 = 0.0
    run4 = 0.0
    run5 = 0.0
    run_max = 0.0
    # Runs are counted by multiplicity of rank counts.
    for start in range(0, 13):
        if start + 3 <= 13:
            c = counts[start:start + 3]
            if np.all(c > 0):
                run3 += float(np.prod(c))
                run_max = max(run_max, 3.0)
        if start + 4 <= 13:
            c = counts[start:start + 4]
            if np.all(c > 0):
                run4 += float(np.prod(c))
                run_max = max(run_max, 4.0)
        if start + 5 <= 13:
            c = counts[start:start + 5]
            if np.all(c > 0):
                run5 += float(np.prod(c))
                run_max = max(run_max, 5.0)
    return run3, run4, run5, run_max


@lru_cache(maxsize=100_000)
def _count_fifteens_cached(values_key: tuple[int, ...]) -> float:
    values = list(values_key)
    total = 0
    for r in range(2, len(values) + 1):
        for combo in combinations(values, r):
            if sum(combo) == 15:
                total += 1
    return float(total)


def _count_fifteens(cards: List[Card]) -> float:
    values_key = tuple(sorted(c.get_value() for c in cards))
    return _count_fifteens_cached(values_key)


def _all_same_suit(cards: List[Card]) -> float:
    if not cards:
        return 0.0
    suit = cards[0].get_suit()
    return 1.0 if all(c.get_suit() == suit for c in cards) else 0.0


def _has_nobs(cards: List[Card]) -> float:
    return 1.0 if any(c.get_rank().lower() == "j" for c in cards) else 0.0


def _remaining_rank_counts_for_pegging(known_cards: List[Card]) -> dict[str, int]:
    counts = {r: 4 for r in RANKS}
    for c in known_cards:
        rank = c.get_rank().lower()
        counts[rank] = max(0, counts.get(rank, 0) - 1)
    return counts


def _sample_rank_from_counts_np(counts: dict[str, int], rng: np.random.Generator) -> str:
    total = sum(counts.values())
    if total <= 0:
        return "a"
    r = int(rng.integers(0, total))
    running = 0
    for rank, count in counts.items():
        running += count
        if r < running:
            return rank
    return "a"


def _sample_hand_ranks(
    counts: dict[str, int],
    rng: np.random.Generator,
    n_cards: int,
) -> List[str]:
    local = dict(counts)
    ranks: List[str] = []
    for _ in range(n_cards):
        total = sum(local.values())
        if total <= 0:
            break
        rank = _sample_rank_from_counts_np(local, rng)
        ranks.append(rank)
        local[rank] = max(0, local[rank] - 1)
    return ranks


def _build_rank_cards(ranks: List[str]) -> List[Card]:
    used: dict[str, int] = {r: 0 for r in RANKS}
    cards: List[Card] = []
    for rank in ranks:
        idx = used[rank]
        suit = _SUITS[idx % len(_SUITS)]
        used[rank] += 1
        cards.append(Card(f"{rank}{suit}"))
    return cards

def _score_context_features(player_score: int | None, opponent_score: int | None) -> np.ndarray:
    if player_score is None:
        player_score = 0
    if opponent_score is None:
        opponent_score = 0
    score_margin = float(player_score - opponent_score)
    endgame_self = 1.0 if player_score >= 110 else 0.0
    endgame_opp = 1.0 if opponent_score >= 110 else 0.0
    endgame_any = 1.0 if max(player_score, opponent_score) >= 110 else 0.0
    return np.array(
        [
            float(player_score),
            float(opponent_score),
            score_margin,
            endgame_self,
            endgame_opp,
            endgame_any,
        ],
        dtype=np.float32,
    )


def _simulate_pegging_points(
    hand_self: List[Card],
    hand_opp: List[Card],
    *,
    dealer_is_self: bool,
) -> Tuple[int, int]:
    table: List[Card] = []
    count = 0
    hands = [list(hand_self), list(hand_opp)]
    scores = [0, 0]
    turn = 1 if dealer_is_self else 0  # non-dealer leads
    passes = 0
    last_played: Optional[int] = None

    while hands[0] or hands[1]:
        playable = [c for c in hands[turn] if c.get_value() + count <= 31]
        if not playable:
            passes += 1
            if passes >= 2:
                if last_played is not None:
                    scores[last_played] += 1
                table = []
                count = 0
                passes = 0
                last_played = None
            turn = 1 - turn
            continue

        passes = 0
        card = medium_pegging_strategy(playable, count, table) or playable[0]
        hands[turn].remove(card)
        table.append(card)
        count += card.get_value()
        scores[turn] += int(score_play(table)[0])

        if count == 31:
            scores[turn] += 2
            table = []
            count = 0
            last_played = None
            turn = 1 - turn
            continue

        last_played = turn
        turn = 1 - turn

    if count > 0 and last_played is not None:
        scores[last_played] += 1

    return int(scores[0]), int(scores[1])


def estimate_pegging_ev_mc_for_discard(
    hand: List[Card],
    kept: List[Card],
    discards: List[Card],
    dealer_is_self: bool,
    rng: np.random.Generator,
    n_rollouts: int,
) -> Tuple[float, float, float]:
    if n_rollouts <= 0:
        return 0.0, 0.0, 0.0
    rank_counts = _remaining_rank_counts_for_pegging(hand)
    if sum(rank_counts.values()) < 6:
        return 0.0, 0.0, 0.0

    total_self = 0.0
    total_opp = 0.0
    for _ in range(n_rollouts):
        opp_ranks = _sample_hand_ranks(rank_counts, rng, 6)
        opp_hand = _build_rank_cards(opp_ranks)
        opp_discards = list(exact_hand_and_min_crib(opp_hand, dealer_is_self=not dealer_is_self))
        opp_kept = [c for c in opp_hand if c not in opp_discards]
        peg_self, peg_opp = _simulate_pegging_points(
            kept,
            opp_kept,
            dealer_is_self=dealer_is_self,
        )
        total_self += float(peg_self)
        total_opp += float(peg_opp)

    avg_self = total_self / float(n_rollouts)
    avg_opp = total_opp / float(n_rollouts)
    return avg_self, avg_opp, avg_self - avg_opp


def featurize_discard(
    kept: List[Card],
    discards: List[Card],
    dealer_is_self: bool,
    player_score: int | None = None,
    opponent_score: int | None = None,
    pegging_ev: Tuple[float, float, float] | None = None,
) -> np.ndarray:
    disc_vec = multi_hot_cards(discards)    # (52,)
    kept_vec = multi_hot_cards(kept)          # (52,)
    dealer_vec = np.array([1.0 if dealer_is_self else 0.0], dtype=np.float32)

    score_context = _score_context_features(player_score, opponent_score)

    if pegging_ev is None:
        pegging_ev = (0.0, 0.0, 0.0)

    engineered = np.concatenate([
        _rank_counts(kept),                       # 13
        _rank_counts(discards),                   # 13
        np.array([_value_sum(kept)], dtype=np.float32),
        np.array([_value_sum(discards)], dtype=np.float32),
        np.array([_count_tens(kept)], dtype=np.float32),
        np.array([_count_tens(discards)], dtype=np.float32),
        np.array([_count_fives(kept)], dtype=np.float32),
        np.array([_count_fives(discards)], dtype=np.float32),
        np.array(_pair_trip_quad_counts(kept), dtype=np.float32),
        np.array(_pair_trip_quad_counts(discards), dtype=np.float32),
        np.array(_run_counts(kept), dtype=np.float32),
        np.array([_all_same_suit(kept)], dtype=np.float32),
        np.array([_all_same_suit(discards)], dtype=np.float32),
        np.array([_has_nobs(kept)], dtype=np.float32),
        np.array([_count_fifteens(kept)], dtype=np.float32),
        np.array([_count_fifteens(discards)], dtype=np.float32),
        np.array([pegging_ev[0]], dtype=np.float32),
        np.array([pegging_ev[1]], dtype=np.float32),
        np.array([pegging_ev[2]], dtype=np.float32),
        score_context,
    ])

    out = np.concatenate([
        disc_vec,
        kept_vec,
        dealer_vec,
        engineered,
    ])
    assert out.shape[0] == DISCARD_FEATURE_DIM, f"discard features dim {out.shape[0]} != {DISCARD_FEATURE_DIM}"
    return out

def one_hot_count(count: int) -> np.ndarray:
    v = np.zeros(32, dtype=np.float32)
    v[count] = 1.0
    return v


def featurize_pegging(
    hand: List[Card],
    table: List[Card],
    count: int,
    candidate: Card,
    known_cards: List[Card] = None,
    opponent_known_hand: List[Card] = None,
    all_played_cards: List[Card] = None,
    player_score: int | None = None,
    opponent_score: int | None = None,
    feature_set: str = "full",
    unseen_value_counts: np.ndarray | None = None,
    unseen_count: int | None = None,
) -> np.ndarray:
    if known_cards is None:
        known_cards = []
    if opponent_known_hand is None:
        opponent_known_hand = []
    if all_played_cards is None:
        all_played_cards = []
    if player_score is None:
        player_score = 0
    if opponent_score is None:
        opponent_score = 0
    
    def _pegging_sequence_features(table_cards: List[Card]) -> np.ndarray:
        seq_cards = table_cards[-PEGGING_SEQ_LEN:]
        steps: List[np.ndarray] = []
        count_so_far = 0
        for c in seq_cards:
            card_vec = np.zeros(52, dtype=np.float32)
            card_vec[c.to_index()] = 1.0
            count_vec = one_hot_count(count_so_far)
            steps.append(np.concatenate([card_vec, count_vec]))
            count_so_far += c.get_value()
        if len(steps) < PEGGING_SEQ_LEN:
            pad = PEGGING_SEQ_LEN - len(steps)
            steps.extend([np.zeros(PEGGING_SEQ_STEP_DIM, dtype=np.float32) for _ in range(pad)])
        return np.concatenate(steps).astype(np.float32, copy=False)

    hand_vec = multi_hot_cards(hand)           # (52,)
    table_vec = multi_hot_cards(table)         # (52,)
    count_vec = one_hot_count(count)           # (32,)
    known_vec = multi_hot_cards(known_cards)  # (52,)
    opp_played_vec = multi_hot_cards(opponent_known_hand)  # (52,)
    all_played_vec = multi_hot_cards(all_played_cards)     # (52,)

    cand_vec = np.zeros(52, dtype=np.float32)
    cand_vec[candidate.to_index()] = 1.0

    # Engineered pegging features (scalars)
    new_count = count + candidate.get_value()
    remaining_to_31 = 31 - new_count
    makes_15 = 1.0 if new_count == 15 else 0.0
    makes_31 = 1.0 if new_count == 31 else 0.0

    seq_after = table + [candidate]
    immediate_points = float(score_play(seq_after)[0])
    pair_points = float(HasPairTripleQuad().check(seq_after)[0])
    run_length = float(HasStraight_DuringPlay().check(seq_after)[0])

    # Run setup features based on last two cards after our play
    run_setup_gap1 = 0.0
    run_setup_gap2 = 0.0
    run_setup_any = 0.0
    opponent_pair_setup = 0.0

    if len(seq_after) >= 1:
        last_rank = seq_after[-1].get_rank().lower()
        # Opponent can score a pair if they play same rank and stay <=31
        same_rank_card = Card(f"{last_rank}h")
        if new_count + same_rank_card.get_value() <= 31:
            opponent_pair_setup = 1.0

    if len(seq_after) >= 2:
        r1 = RANK_TO_I[seq_after[-1].get_rank().lower()] + 1
        r2 = RANK_TO_I[seq_after[-2].get_rank().lower()] + 1
        gap = abs(r1 - r2)
        if gap == 1:
            # Opponent can play r1-1 or r2+1
            candidates = []
            low = min(r1, r2) - 1
            high = max(r1, r2) + 1
            if 1 <= low <= 13:
                candidates.append(low)
            if 1 <= high <= 13:
                candidates.append(high)
            for rv in candidates:
                rank_str = RANKS[rv - 1]
                c = Card(f"{rank_str}h")
                if new_count + c.get_value() <= 31:
                    run_setup_gap1 += 1.0
        elif gap == 2:
            # Opponent can play the middle rank
            mid = min(r1, r2) + 1
            rank_str = RANKS[mid - 1]
            c = Card(f"{rank_str}h")
            if new_count + c.get_value() <= 31:
                run_setup_gap2 = 1.0

    # Count how many ranks could create a run of 3+ for opponent next
    for rv in range(1, 14):
        rank_str = RANKS[rv - 1]
        c = Card(f"{rank_str}h")
        if new_count + c.get_value() > 31:
            continue
        run_len = HasStraight_DuringPlay().check(seq_after + [c])[0]
        if run_len >= 3:
            run_setup_any += 1.0

    our_hand_count = float(len(hand))
    opp_hand_count_est = float(max(0, 4 - len(opponent_known_hand)))
    table_len = float(len(table))
    opp_played_count = float(len(opponent_known_hand))
    known_cards_count = float(len(known_cards))

    # Opponent "go" danger: estimate if opponent has any playable card.
    unseen_suit_counts = None
    if unseen_value_counts is None or unseen_count is None:
        known_set = set(known_cards) | set(all_played_cards) | set(hand)
        unseen = [c for c in _FULL_DECK if c not in known_set]
        unseen_count = len(unseen)
        unseen_value_counts = np.zeros(11, dtype=np.int32)
        for c in unseen:
            unseen_value_counts[c.get_value()] += 1
        unseen_rank_counts = np.zeros(13, dtype=np.int32)
        unseen_suit_counts = np.zeros(4, dtype=np.int32)
        for c in unseen:
            unseen_rank_counts[RANK_TO_I[c.get_rank().lower()]] += 1
            unseen_suit_counts[SUIT_TO_I[c.get_suit()]] += 1
    else:
        unseen_rank_counts = None

    max_val = 10 if remaining_to_31 >= 10 else remaining_to_31
    if max_val < 1:
        playable_unseen_count = 0
    else:
        playable_unseen_count = int(unseen_value_counts[1 : max_val + 1].sum())
    opp_can_play_prob = float(playable_unseen_count / max(1, unseen_count))
    opp_playable_count = float(playable_unseen_count)
    unseen_count = float(unseen_count)

    if unseen_rank_counts is None or unseen_suit_counts is None:
        known_set = set(known_cards) | set(all_played_cards) | set(hand)
        unseen = [c for c in _FULL_DECK if c not in known_set]
        unseen_rank_counts = np.zeros(13, dtype=np.int32)
        unseen_suit_counts = np.zeros(4, dtype=np.int32)
        for c in unseen:
            unseen_rank_counts[RANK_TO_I[c.get_rank().lower()]] += 1
            unseen_suit_counts[SUIT_TO_I[c.get_suit()]] += 1

    def _response_counts(table_state: List[Card], count_state: int) -> dict[str, float]:
        resp_any = 0.0
        resp_15 = 0.0
        resp_31 = 0.0
        resp_pair = 0.0
        resp_run = 0.0
        for rv in range(1, 14):
            count_cards = float(unseen_rank_counts[rv - 1])
            if count_cards <= 0.0:
                continue
            rank_str = RANKS[rv - 1]
            c = Card(f"{rank_str}h")
            new_count = count_state + c.get_value()
            if new_count > 31:
                continue
            seq = table_state + [c]
            immediate_points = float(score_play(seq)[0])
            pair_points = float(HasPairTripleQuad().check(seq)[0])
            run_len = float(HasStraight_DuringPlay().check(seq)[0])
            if immediate_points > 0.0:
                resp_any += count_cards
            if new_count == 15:
                resp_15 += count_cards
            if new_count == 31:
                resp_31 += count_cards
            if pair_points > 0.0:
                resp_pair += count_cards
            if run_len >= 3.0:
                resp_run += count_cards
        return {
            "any": resp_any,
            "r15": resp_15,
            "r31": resp_31,
            "pair": resp_pair,
            "run": resp_run,
        }

    seq_after = table + [candidate]
    new_count = count + candidate.get_value()
    resp_counts = _response_counts(seq_after, new_count)
    denom = max(1.0, unseen_count)
    opp_resp_any_prob = resp_counts["any"] / denom
    opp_resp_15_prob = resp_counts["r15"] / denom
    opp_resp_31_prob = resp_counts["r31"] / denom
    opp_resp_pair_prob = resp_counts["pair"] / denom
    opp_resp_run_prob = resp_counts["run"] / denom

    opp_skipped_15 = 0.0
    opp_skipped_31 = 0.0
    opp_skipped_pair = 0.0
    opp_skipped_run = 0.0
    if len(table) >= 1:
        last_card = table[-1]
        prev_table = table[:-1]
        prev_count = count - last_card.get_value()
        if prev_count >= 0:
            last_points = float(score_play(prev_table + [last_card])[0])
            if last_points <= 0.0:
                prev_resp = _response_counts(prev_table, prev_count)
                if prev_resp["r15"] > 0.0:
                    opp_skipped_15 = 1.0
                if prev_resp["r31"] > 0.0:
                    opp_skipped_31 = 1.0
                if prev_resp["pair"] > 0.0:
                    opp_skipped_pair = 1.0
                if prev_resp["run"] > 0.0:
                    opp_skipped_run = 1.0

    def _prob_at_least_k(total: int, success: int, draws: int, k: int) -> float:
        if total <= 0 or success <= 0 or draws <= 0:
            return 0.0
        if draws > total or k > draws or k > success:
            return 0.0
        denom = math.comb(total, draws)
        if denom <= 0:
            return 0.0
        prob = 0.0
        max_t = min(success, draws)
        for t in range(k, max_t + 1):
            prob += math.comb(success, t) * math.comb(total - success, draws - t)
        return float(prob / denom)

    def _max_group_prob(counts: np.ndarray, draws: int, k: int, total: int) -> float:
        if draws <= 0 or total <= 0:
            return 0.0
        best = 0.0
        for c in counts:
            prob = _prob_at_least_k(total, int(c), draws, k)
            if prob > best:
                best = prob
        return best

    opp_unknown_cards = int(max(0, int(opp_hand_count_est)))
    unseen_total = int(unseen_count)
    opp_flush_belief = _max_group_prob(unseen_suit_counts, opp_unknown_cards, 4, unseen_total)
    opp_trip_belief = _max_group_prob(unseen_rank_counts, opp_unknown_cards, 3, unseen_total)
    opp_quad_belief = _max_group_prob(unseen_rank_counts, opp_unknown_cards, 4, unseen_total)
    tens_count = int(
        unseen_rank_counts[RANK_TO_I["10"]]
        + unseen_rank_counts[RANK_TO_I["j"]]
        + unseen_rank_counts[RANK_TO_I["q"]]
        + unseen_rank_counts[RANK_TO_I["k"]]
    )
    if opp_unknown_cards <= 0 or unseen_total <= 0 or tens_count < opp_unknown_cards:
        opp_only_tens_belief = 0.0
    else:
        opp_only_tens_belief = float(
            math.comb(tens_count, opp_unknown_cards) / math.comb(unseen_total, opp_unknown_cards)
        )

    score_context = _score_context_features(player_score, opponent_score)

    engineered = np.array(
        [
            float(new_count),
            float(remaining_to_31),
            makes_15,
            makes_31,
            immediate_points,
            pair_points,
            run_length,
            run_setup_gap1,
            run_setup_gap2,
            run_setup_any,
            opponent_pair_setup,
            our_hand_count,
            opp_hand_count_est,
            table_len,
            opp_played_count,
            known_cards_count,
            opp_can_play_prob,
            opp_playable_count,
            unseen_count,
            opp_resp_any_prob,
            opp_resp_15_prob,
            opp_resp_31_prob,
            opp_resp_pair_prob,
            opp_resp_run_prob,
            opp_skipped_15,
            opp_skipped_31,
            opp_skipped_pair,
            opp_skipped_run,
            opp_flush_belief,
            opp_trip_belief,
            opp_quad_belief,
            opp_only_tens_belief,
        ],
        dtype=np.float32,
    )
    engineered = np.concatenate([engineered, score_context])

    base = np.concatenate([
        hand_vec,
        table_vec,
        count_vec,
        cand_vec,
        known_vec,
    ])
    if feature_set == "basic":
        assert base.shape[0] == PEGGING_BASE_FEATURE_DIM, f"pegging features dim {base.shape[0]} != {PEGGING_BASE_FEATURE_DIM}"
        return base
    if feature_set == "full_no_scores":
        engineered_no_scores = engineered[:PEGGING_ENGINEERED_NO_SCORE_DIM]
        out = np.concatenate([
            base,
            opp_played_vec,
            all_played_vec,
            engineered_no_scores,
        ])
        expected = PEGGING_BASE_FEATURE_DIM + 52 + 52 + PEGGING_ENGINEERED_NO_SCORE_DIM
        assert out.shape[0] == expected, f"pegging features dim {out.shape[0]} != {expected}"
        return out
    if feature_set == "full":
        out = np.concatenate([
            base,
            opp_played_vec,
            all_played_vec,
            engineered,
        ])
        assert out.shape[0] == PEGGING_FULL_FEATURE_DIM, f"pegging features dim {out.shape[0]} != {PEGGING_FULL_FEATURE_DIM}"
        return out
    if feature_set == "full_seq":
        out = np.concatenate([
            base,
            opp_played_vec,
            all_played_vec,
            engineered,
        ])
        seq = _pegging_sequence_features(table)
        out = np.concatenate([out, seq])
        assert out.shape[0] == PEGGING_FULL_SEQ_FEATURE_DIM, f"pegging features dim {out.shape[0]} != {PEGGING_FULL_SEQ_FEATURE_DIM}"
        return out
    raise ValueError(f"Unknown pegging feature_set: {feature_set}")


class LinearDiscardClassifier:
    """
    Scores each of 15 discard options with a linear model:
      score_i = w·x_i + b
    Train with softmax cross-entropy on the best option label (0..14).

    Expects X shape (N, 15, D), y shape (N,)
    """
    def __init__(self, n_features: int):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

    def predict_scores(self, X15: np.ndarray) -> np.ndarray:
        # X15: (15, D) -> (15,)
        return (X15 @ self.w + self.b).astype(np.float32)

    def fit_ce(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.05,
        epochs: int = 10,
        batch_size: int = 256,
        l2: float = 0.0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> List[float]:
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (N,15,D), got {X.shape}")
        if X.shape[1] != 15:
            raise ValueError(f"X must have 15 options, got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (N,), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must share N, got {X.shape[0]} and {y.shape[0]}")
        if X.shape[2] != self.w.shape[0]:
            raise ValueError(f"X has D={X.shape[2]} but model expects {self.w.shape[0]}")

        rng = np.random.default_rng(seed)
        N = X.shape[0]
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)

        losses: List[float] = []

        for _ in range(epochs):
            idx = np.arange(N)
            if shuffle:
                rng.shuffle(idx)

            epoch_loss = 0.0
            n_seen = 0

            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]  # (B,15,D)
                yb = y[batch_idx]  # (B,)

                # scores: (B,15)
                scores = np.tensordot(Xb, self.w, axes=([2], [0])) + self.b

                # stable softmax
                scores = scores - scores.max(axis=1, keepdims=True)
                exp_scores = np.exp(scores)
                probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (B,15)

                # CE loss
                p_true = probs[np.arange(len(batch_idx)), yb]
                batch_loss = float(-np.mean(np.log(p_true + 1e-12)))
                epoch_loss += batch_loss * len(batch_idx)
                n_seen += len(batch_idx)

                # gradient wrt scores: probs - one_hot(y)
                grad_scores = probs
                grad_scores[np.arange(len(batch_idx)), yb] -= 1.0
                grad_scores /= float(len(batch_idx))  # mean

                # grad_w: sum over options and batch
                # (B,15,D) weighted by (B,15) -> (D,)
                grad_w = np.tensordot(grad_scores, Xb, axes=([0, 1], [0, 1])).astype(np.float32)
                grad_b = float(np.sum(grad_scores))

                if l2 > 0.0:
                    grad_w += 2.0 * l2 * self.w

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            losses.append(epoch_loss / max(1, n_seen))

        return losses

    def save_npz(self, path: str) -> None:
        np.savez(path, w=self.w, b=np.array([self.b], dtype=np.float32))

    @classmethod
    def load_npz(cls, path: str) -> "LinearDiscardClassifier":
        data = np.load(path)
        w = data["w"].astype(np.float32)
        b = float(data["b"].reshape(-1)[0])
        m = cls(int(w.shape[0]))
        m.w = w
        m.b = b
        return m


class LinearValueModel:
    # used for regression
    def __init__(self, n_features):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x) + self.b)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict for a batch.

        Args:
            X: shape (N, D)
        Returns:
            y_hat: shape (N,)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N,D), got {X.shape}")
        return (X @ self.w + self.b).astype(np.float32)

    def fit_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.05,
        epochs: int = 10,
        batch_size: int = 4096,
        l2: float = 0.0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> List[float]:
        """Train with mean squared error using minibatch gradient descent.

        This is intentionally simple so you can swap in PyTorch later.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N,D), got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (N,), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must share N, got {X.shape[0]} and {y.shape[0]}")
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(f"X has D={X.shape[1]} but model expects {self.w.shape[0]}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        rng = np.random.default_rng(seed)
        N = X.shape[0]
        losses: List[float] = []

        # Ensure float32 for speed and stable updates
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        for _ in range(epochs):
            idx = np.arange(N)
            if shuffle:
                rng.shuffle(idx)

            epoch_loss = 0.0
            n_seen = 0
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]

                pred = Xb @ self.w + self.b  # (B,)
                err = pred - yb
                # MSE
                batch_loss = float(np.mean(err * err))
                epoch_loss += batch_loss * len(batch_idx)
                n_seen += len(batch_idx)

                # d/dw mean((Xw+b-y)^2) = 2/B * X^T (pred - y)
                grad_w = (2.0 / len(batch_idx)) * (Xb.T @ err)
                grad_b = float((2.0 / len(batch_idx)) * np.sum(err))

                if l2 > 0.0:
                    grad_w += 2.0 * l2 * self.w

                self.w -= lr * grad_w.astype(np.float32)
                self.b -= lr * grad_b

            losses.append(epoch_loss / max(1, n_seen))
        return losses

    def fit_rank_pairwise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.05,
        epochs: int = 5,
        batch_size: int = 128,
        l2: float = 0.0,
        seed: int = 0,
        shuffle: bool = True,
        pairs_per_hand: int = 20,
    ) -> List[float]:
        """Train a linear ranker with pairwise logistic loss.

        X: shape (N, 15, D)
        y: shape (N, 15) scores (higher is better)
        """
        if X.ndim != 3 or X.shape[1] != 15:
            raise ValueError(f"X must be (N,15,D), got {X.shape}")
        if y.ndim != 2 or y.shape[1] != 15:
            raise ValueError(f"y must be (N,15), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must share N, got {X.shape[0]} and {y.shape[0]}")
        if X.shape[2] != self.w.shape[0]:
            raise ValueError(f"X has D={X.shape[2]} but model expects {self.w.shape[0]}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if pairs_per_hand <= 0:
            raise ValueError("pairs_per_hand must be > 0")

        rng = np.random.default_rng(seed)
        N = X.shape[0]
        losses: List[float] = []

        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        for _ in range(epochs):
            idx = np.arange(N)
            if shuffle:
                rng.shuffle(idx)

            epoch_loss = 0.0
            n_seen = 0

            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]  # (B,15,D)
                yb = y[batch_idx]  # (B,15)

                grad_w = np.zeros_like(self.w)
                grad_b = 0.0
                batch_loss = 0.0

                for i in range(Xb.shape[0]):
                    Xi = Xb[i]  # (15,D)
                    yi = yb[i]  # (15,)
                    scores = Xi @ self.w + self.b  # (15,)

                    # sample pairs where yi differs
                    for _ in range(pairs_per_hand):
                        a = rng.integers(0, 15)
                        b = rng.integers(0, 15)
                        if a == b:
                            continue
                        if yi[a] == yi[b]:
                            continue
                        s = 1.0 if yi[a] > yi[b] else -1.0
                        diff = scores[a] - scores[b]
                        # logistic loss: log(1+exp(-s*diff))
                        z = -s * diff
                        loss = np.log1p(np.exp(z))
                        batch_loss += float(loss)
                        # grad for diff: -s * sigmoid(z)
                        sigmoid = 1.0 / (1.0 + np.exp(-z))
                        g = -s * sigmoid
                        grad_w += g * (Xi[a] - Xi[b])
                        grad_b += g

                if l2 > 0.0:
                    grad_w += 2.0 * l2 * self.w

                # normalize by number of hands in batch
                if Xb.shape[0] > 0:
                    grad_w /= float(Xb.shape[0])
                    grad_b /= float(Xb.shape[0])
                    batch_loss /= float(Xb.shape[0])

                self.w -= lr * grad_w.astype(np.float32)
                self.b -= lr * grad_b

                epoch_loss += batch_loss * len(batch_idx)
                n_seen += len(batch_idx)

            losses.append(epoch_loss / max(1, n_seen))

        return losses
    def save_npz(self, path: str) -> None:
        np.savez(path, w=self.w, b=np.array([self.b], dtype=np.float32))

    @classmethod
    def load_npz(cls, path: str) -> "LinearValueModel":
        data = np.load(path)
        w = data["w"].astype(np.float32)
        b = float(data["b"].reshape(-1)[0])
        m = cls(int(w.shape[0]))
        m.w = w
        m.b = b
        return m


class MLPValueModel:
    """Small MLP regressor using PyTorch for nonlinear value prediction."""

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (128, 64), seed: int | None = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(0 if seed is None else int(seed))
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

    def predict(self, x: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = self.model(t).squeeze(0).squeeze(0).item()
        return float(y)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            y = self.model(t).squeeze(1).cpu().numpy()
        return y.astype(np.float32, copy=False)

    def fit_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.001,
        epochs: int = 5,
        batch_size: int = 1024,
        l2: float = 0.0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> List[float]:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(0 if seed is None else int(seed))
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        N = X.shape[0]

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
        loss_fn = nn.MSELoss()

        losses: List[float] = []
        for _ in range(epochs):
            idx = np.arange(N)
            if shuffle:
                np.random.default_rng(seed).shuffle(idx)
            epoch_loss = 0.0
            n_seen = 0
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                xb = torch.tensor(X[batch_idx], dtype=torch.float32)
                yb = torch.tensor(y[batch_idx], dtype=torch.float32)
                optimizer.zero_grad()
                pred = self.model(xb).squeeze(1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_idx)
                n_seen += len(batch_idx)
            losses.append(epoch_loss / max(1, n_seen))
        return losses

    def save_pt(self, path: str) -> None:
        import torch

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_sizes": self.hidden_sizes,
            },
            path,
        )

    @classmethod
    def load_pt(cls, path: str) -> "MLPValueModel":
        import torch

        data = torch.load(path, map_location="cpu")
        m = cls(int(data["input_dim"]), tuple(int(h) for h in data["hidden_sizes"]))
        m.model.load_state_dict(data["state_dict"])
        m.model.eval()
        return m


class PeggingRNNValueModel:
    """GRU/LSTM regressor over pegging sequences plus static features."""

    def __init__(
        self,
        static_dim: int,
        *,
        rnn_type: str = "gru",
        rnn_hidden: int = 64,
        head_hidden: Tuple[int, ...] = (128, 64),
        seq_len: int = PEGGING_SEQ_LEN,
        step_dim: int = PEGGING_SEQ_STEP_DIM,
        seed: int | None = 0,
    ):
        import torch
        import torch.nn as nn

        torch.manual_seed(0 if seed is None else int(seed))
        rnn_type = rnn_type.lower()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        self.rnn_type = rnn_type
        self.seq_len = int(seq_len)
        self.step_dim = int(step_dim)
        self.static_dim = int(static_dim)
        self.rnn_hidden = int(rnn_hidden)
        self.head_hidden = tuple(int(h) for h in head_hidden)

        if rnn_type == "gru":
            self.rnn = nn.GRU(self.step_dim, self.rnn_hidden, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.step_dim, self.rnn_hidden, batch_first=True)

        head_layers = []
        prev = self.rnn_hidden + self.static_dim
        for h in self.head_hidden:
            head_layers.append(nn.Linear(prev, h))
            head_layers.append(nn.ReLU())
            prev = h
        head_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head_layers)

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        static = X[:, : self.static_dim]
        seq_flat = X[:, self.static_dim :]
        expected = self.seq_len * self.step_dim
        if seq_flat.shape[1] != expected:
            raise ValueError(f"Expected seq_flat dim {expected}, got {seq_flat.shape[1]}")
        seq = seq_flat.reshape(-1, self.seq_len, self.step_dim)
        return static, seq

    def predict(self, x: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            X = x.astype(np.float32, copy=False).reshape(1, -1)
            static, seq = self._split(X)
            t_static = torch.tensor(static, dtype=torch.float32)
            t_seq = torch.tensor(seq, dtype=torch.float32)
            if self.rnn_type == "gru":
                _, h = self.rnn(t_seq)
            else:
                _, (h, _) = self.rnn(t_seq)
            h_last = h.squeeze(0)
            feats = torch.cat([t_static, h_last], dim=1)
            y = self.head(feats).squeeze(1).item()
        return float(y)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            X = X.astype(np.float32, copy=False)
            static, seq = self._split(X)
            t_static = torch.tensor(static, dtype=torch.float32)
            t_seq = torch.tensor(seq, dtype=torch.float32)
            if self.rnn_type == "gru":
                _, h = self.rnn(t_seq)
            else:
                _, (h, _) = self.rnn(t_seq)
            h_last = h.squeeze(0)
            feats = torch.cat([t_static, h_last], dim=1)
            y = self.head(feats).squeeze(1).cpu().numpy()
        return y.astype(np.float32, copy=False)

    def fit_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.001,
        epochs: int = 5,
        batch_size: int = 1024,
        l2: float = 0.0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> List[float]:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(0 if seed is None else int(seed))
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        N = X.shape[0]

        optimizer = optim.Adam(
            list(self.rnn.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=l2,
        )
        loss_fn = nn.MSELoss()

        losses: List[float] = []
        for _ in range(epochs):
            idx = np.arange(N)
            if shuffle:
                np.random.default_rng(seed).shuffle(idx)
            epoch_loss = 0.0
            n_seen = 0
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                static, seq = self._split(Xb)
                t_static = torch.tensor(static, dtype=torch.float32)
                t_seq = torch.tensor(seq, dtype=torch.float32)
                t_y = torch.tensor(yb, dtype=torch.float32)
                optimizer.zero_grad()
                if self.rnn_type == "gru":
                    _, h = self.rnn(t_seq)
                else:
                    _, (h, _) = self.rnn(t_seq)
                h_last = h.squeeze(0)
                feats = torch.cat([t_static, h_last], dim=1)
                pred = self.head(feats).squeeze(1)
                loss = loss_fn(pred, t_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_idx)
                n_seen += len(batch_idx)
            losses.append(epoch_loss / max(1, n_seen))
        return losses

    def save_pt(self, path: str) -> None:
        import torch

        torch.save(
            {
                "rnn_state": self.rnn.state_dict(),
                "head_state": self.head.state_dict(),
                "rnn_type": self.rnn_type,
                "rnn_hidden": self.rnn_hidden,
                "head_hidden": self.head_hidden,
                "static_dim": self.static_dim,
                "seq_len": self.seq_len,
                "step_dim": self.step_dim,
            },
            path,
        )

    @classmethod
    def load_pt(cls, path: str) -> "PeggingRNNValueModel":
        import torch

        data = torch.load(path, map_location="cpu")
        m = cls(
            int(data["static_dim"]),
            rnn_type=str(data["rnn_type"]),
            rnn_hidden=int(data["rnn_hidden"]),
            head_hidden=tuple(int(h) for h in data["head_hidden"]),
            seq_len=int(data["seq_len"]),
            step_dim=int(data["step_dim"]),
        )
        m.rnn.load_state_dict(data["rnn_state"])
        m.head.load_state_dict(data["head_state"])
        m.rnn.eval()
        m.head.eval()
        return m


class PeggingTransformerValueModel:
    """Transformer encoder regressor over pegging sequences plus static features."""

    def __init__(
        self,
        static_dim: int,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        head_hidden: Tuple[int, ...] = (128, 64),
        seq_len: int = PEGGING_SEQ_LEN,
        step_dim: int = PEGGING_SEQ_STEP_DIM,
        seed: int | None = 0,
    ):
        import torch
        import torch.nn as nn

        torch.manual_seed(0 if seed is None else int(seed))
        self.static_dim = int(static_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.seq_len = int(seq_len)
        self.step_dim = int(step_dim)
        self.head_hidden = tuple(int(h) for h in head_hidden)

        self.seq_proj = nn.Linear(self.step_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        head_layers = []
        prev = self.d_model + self.static_dim
        for h in self.head_hidden:
            head_layers.append(nn.Linear(prev, h))
            head_layers.append(nn.ReLU())
            prev = h
        head_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head_layers)

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        static = X[:, : self.static_dim]
        seq_flat = X[:, self.static_dim :]
        expected = self.seq_len * self.step_dim
        if seq_flat.shape[1] != expected:
            raise ValueError(f"Expected seq_flat dim {expected}, got {seq_flat.shape[1]}")
        seq = seq_flat.reshape(-1, self.seq_len, self.step_dim)
        return static, seq

    def _encode(self, seq: "torch.Tensor") -> "torch.Tensor":
        # seq: (B, T, step_dim)
        x = self.seq_proj(seq)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)
        return x.mean(dim=1)

    def predict(self, x: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            X = x.astype(np.float32, copy=False).reshape(1, -1)
            static, seq = self._split(X)
            t_static = torch.tensor(static, dtype=torch.float32)
            t_seq = torch.tensor(seq, dtype=torch.float32)
            pooled = self._encode(t_seq)
            feats = torch.cat([t_static, pooled], dim=1)
            y = self.head(feats).squeeze(1).item()
        return float(y)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            X = X.astype(np.float32, copy=False)
            static, seq = self._split(X)
            t_static = torch.tensor(static, dtype=torch.float32)
            t_seq = torch.tensor(seq, dtype=torch.float32)
            pooled = self._encode(t_seq)
            feats = torch.cat([t_static, pooled], dim=1)
            y = self.head(feats).squeeze(1).cpu().numpy()
        return y.astype(np.float32, copy=False)

    def fit_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.001,
        epochs: int = 5,
        batch_size: int = 1024,
        l2: float = 0.0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> List[float]:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(0 if seed is None else int(seed))
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        N = X.shape[0]

        optimizer = optim.Adam(
            list(self.seq_proj.parameters()) + list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=l2,
        )
        loss_fn = nn.MSELoss()

        losses: List[float] = []
        for _ in range(epochs):
            idx = np.arange(N)
            if shuffle:
                np.random.default_rng(seed).shuffle(idx)
            epoch_loss = 0.0
            n_seen = 0
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                static, seq = self._split(Xb)
                t_static = torch.tensor(static, dtype=torch.float32)
                t_seq = torch.tensor(seq, dtype=torch.float32)
                t_y = torch.tensor(yb, dtype=torch.float32)
                optimizer.zero_grad()
                pooled = self._encode(t_seq)
                feats = torch.cat([t_static, pooled], dim=1)
                pred = self.head(feats).squeeze(1)
                loss = loss_fn(pred, t_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_idx)
                n_seen += len(batch_idx)
            losses.append(epoch_loss / max(1, n_seen))
        return losses

    def save_pt(self, path: str) -> None:
        import torch

        torch.save(
            {
                "seq_proj": self.seq_proj.state_dict(),
                "encoder": self.encoder.state_dict(),
                "head": self.head.state_dict(),
                "static_dim": self.static_dim,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "head_hidden": self.head_hidden,
                "seq_len": self.seq_len,
                "step_dim": self.step_dim,
            },
            path,
        )

    @classmethod
    def load_pt(cls, path: str) -> "PeggingTransformerValueModel":
        import torch

        data = torch.load(path, map_location="cpu")
        m = cls(
            int(data["static_dim"]),
            d_model=int(data["d_model"]),
            nhead=int(data["nhead"]),
            num_layers=int(data["num_layers"]),
            dim_feedforward=int(data["dim_feedforward"]),
            dropout=float(data["dropout"]),
            head_hidden=tuple(int(h) for h in data["head_hidden"]),
            seq_len=int(data["seq_len"]),
            step_dim=int(data["step_dim"]),
        )
        m.seq_proj.load_state_dict(data["seq_proj"])
        m.encoder.load_state_dict(data["encoder"])
        m.head.load_state_dict(data["head"])
        m.seq_proj.eval()
        m.encoder.eval()
        m.head.eval()
        return m


class GBTValueModel:
    """Gradient-boosted tree regressor using scikit-learn."""

    def __init__(self, *, seed: int = 0, max_iter: int = 100):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model = HistGradientBoostingRegressor(
            max_iter=int(max_iter),
            random_state=int(seed),
            validation_fraction=None,
            early_stopping=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, x: np.ndarray) -> float:
        return float(self.model.predict(x.reshape(1, -1))[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32, copy=False)

    def save_joblib(self, path: str) -> None:
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load_joblib(cls, path: str) -> "GBTValueModel":
        import joblib

        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


class RandomForestValueModel:
    """Random forest regressor using scikit-learn."""

    def __init__(self, *, seed: int = 0, n_estimators: int = 200):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(seed),
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, x: np.ndarray) -> float:
        return float(self.model.predict(x.reshape(1, -1))[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32, copy=False)

    def save_joblib(self, path: str) -> None:
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load_joblib(cls, path: str) -> "RandomForestValueModel":
        import joblib

        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj

def select_discard_with_model(discard_model, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
    return select_discard_with_model_with_scores(discard_model, hand, dealer_is_self, None, None)

def select_discard_with_model_with_scores(
    discard_model,
    hand: List[Card],
    dealer_is_self: bool,
    player_score: int | None,
    opponent_score: int | None,
    feature_indices: np.ndarray | None = None,
    discard_feature_set: str | None = None,
) -> Tuple[Card, Card]:
    if hasattr(discard_model, "predict_scores"):
        Xs: List[np.ndarray] = []
        discards_list: List[Tuple[Card, Card]] = []
        rng = np.random.default_rng()
        for kept in combinations(hand, 4):
            kept = list(kept)
            discards = [c for c in hand if c not in kept]
            discards_list.append((discards[0], discards[1]))
            pegging_ev = None
            if discard_feature_set in {"engineered_no_scores_pev", "full_pev"}:
                pegging_ev = estimate_pegging_ev_mc_for_discard(
                    hand,
                    kept,
                    discards,
                    dealer_is_self,
                    rng,
                    n_rollouts=8,
                )
            x = featurize_discard(kept, discards, dealer_is_self, player_score, opponent_score, pegging_ev=pegging_ev)
            if feature_indices is not None:
                x = x[feature_indices]
            Xs.append(x)
        X15 = np.stack(Xs, axis=0).astype(np.float32)  # (15, D)
        scores = discard_model.predict_scores(X15)  # (15,)
        best_i = int(np.argmax(scores))
        return discards_list[best_i]

    best, best_v = None, float("-inf")
    rng = np.random.default_rng()
    for kept in combinations(hand, 4):
        kept = list(kept)
        discards = [c for c in hand if c not in kept]
        pegging_ev = None
        if discard_feature_set in {"engineered_no_scores_pev", "full_pev"}:
            pegging_ev = estimate_pegging_ev_mc_for_discard(
                hand,
                kept,
                discards,
                dealer_is_self,
                rng,
                n_rollouts=8,
            )
        x = featurize_discard(
            kept,
            discards,
            dealer_is_self,
            player_score,
            opponent_score,
            pegging_ev=pegging_ev,
        )  # np array
        if feature_indices is not None:
            x = x[feature_indices]
        v = float(discard_model.predict(x))
        if v > best_v:
            best_v, best = v, tuple(discards)
    return best

def regression_pegging_strategy(
    pegging_model,
    hand,
    table,
    crib,
    count,
    past_table_cards=None,
    starter_card=None,
    known_cards=None,
    opponent_known_hand=None,
    all_played_cards=None,
    player_score: int = 0,
    opponent_score: int = 0,
    feature_set: str = "full",
    feature_indices: np.ndarray | None = None,
):
    if past_table_cards is None:
        past_table_cards = []
    
    playable = [c for c in hand if c + count <= 31]
    if not playable:
        return None
    best, best_v = None, float("-inf")
    if known_cards is None:
        # Known cards include: hand, current table sequence, past table cards, and starter
        known = hand + table + past_table_cards
        if starter_card is not None:
            known = known + [starter_card]
    else:
        known = known_cards

    known_set = set(known) | set(all_played_cards or []) | set(hand)
    unseen = [c for c in _FULL_DECK if c not in known_set]
    unseen_value_counts = np.zeros(11, dtype=np.int32)
    for c in unseen:
        unseen_value_counts[c.get_value()] += 1
    unseen_count = len(unseen)

    for c in playable:
        x = featurize_pegging(
            hand,
            table,
            count,
            c,
            known_cards=known,
            opponent_known_hand=opponent_known_hand,
            all_played_cards=all_played_cards,
            player_score=player_score,
            opponent_score=opponent_score,
            feature_set=feature_set,
            unseen_value_counts=unseen_value_counts,
            unseen_count=unseen_count,
        )  # np array
        if feature_indices is not None:
            x = x[feature_indices]
        v = float(pegging_model.predict(x))
        if v > best_v:
            best_v, best = v, c
    return best

class AIPlayer:
    def __init__(
        self,
        discard_model,
        pegging_model,
        name="neural",
        discard_feature_set: str = "full",
        pegging_feature_set: str = "full",
    ):
        self.name = name
        self.discard_model = discard_model
        self.pegging_model = pegging_model
        self.pegging_feature_set = pegging_feature_set
        self.discard_feature_set = discard_feature_set
        self.discard_feature_indices = get_discard_feature_indices(discard_feature_set)
        self.pegging_feature_indices = get_pegging_feature_indices(pegging_feature_set)

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        # Extract hand and dealer info from state objects
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = None  # Not available in state
        return self.select_crib_cards_regressor(hand, dealer_is_self, your_score, opponent_score) # type: ignore

    def select_crib_cards_regressor(self, hand, dealer_is_self, your_score=None, opponent_score=None) -> Tuple[Card, Card]:
        return select_discard_with_model_with_scores(
            self.discard_model,
            hand,
            dealer_is_self,
            your_score,
            opponent_score,
            self.discard_feature_indices,
            self.discard_feature_set,
        )

    def select_card_to_play(self, player_state, round_state):
        hand = player_state.hand
        table = round_state.table_cards
        crib = round_state.crib
        count = round_state.count
        best = regression_pegging_strategy(
            self.pegging_model,
            hand,
            table,
            crib,
            count,
            known_cards=player_state.known_cards,
            opponent_known_hand=player_state.opponent_known_hand,
            all_played_cards=round_state.all_played_cards,
            player_score=player_state.score,
            opponent_score=player_state.opponent_score,
            feature_set=self.pegging_feature_set,
            feature_indices=self.pegging_feature_indices,
        )
        return best


class MLPPlayer(AIPlayer):
    pass


class GBTPlayer(AIPlayer):
    pass


class RandomForestPlayer(AIPlayer):
    pass

class NeuralClassificationPlayer:
    def __init__(
        self,
        discard_model,
        pegging_model,
        name="neural",
        discard_feature_set: str = "full",
        pegging_feature_set: str = "full",
    ):
        self.name = name
        self.discard_model = discard_model
        self.pegging_model = pegging_model
        self.pegging_feature_set = pegging_feature_set
        self.discard_feature_set = discard_feature_set
        self.discard_feature_indices = get_discard_feature_indices(discard_feature_set)
        self.pegging_feature_indices = get_pegging_feature_indices(pegging_feature_set)

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        # Extract hand and dealer info from state objects
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = None  # Not available in state
        return self.select_crib_cards_classification(hand, dealer_is_self, your_score, opponent_score) # type: ignore


    def select_crib_cards_classification(self, hand: List[Card], dealer_is_self: bool, your_score=None, opponent_score=None) -> Tuple[Card, Card]:
        return select_discard_with_model_with_scores(
            self.discard_model,
            hand,
            dealer_is_self,
            your_score,
            opponent_score,
            self.discard_feature_indices,
            self.discard_feature_set,
        )

    def select_card_to_play(self, player_state, round_state):
        hand = player_state.hand
        table = round_state.table_cards
        crib = round_state.crib
        count = round_state.count
        best = regression_pegging_strategy(
            self.pegging_model,
            hand,
            table,
            crib,
            count,
            known_cards=player_state.known_cards,
            opponent_known_hand=player_state.opponent_known_hand,
            all_played_cards=round_state.all_played_cards,
            player_score=player_state.score,
            opponent_score=player_state.opponent_score,
            feature_set=self.pegging_feature_set,
            feature_indices=self.pegging_feature_indices,
        )
        return best

class NeuralDiscardPlayer(BeginnerPlayer):
    def __init__(self, discard_model, pegging_model, name="neural"):
        self.name = name
        self.discard_model = discard_model
        self.pegging_model = pegging_model

    def select_crib_cards(self, hand, dealer_is_self):
        best, best_v = None, float("-inf")
        for kept in combinations(hand, 4):
            kept = list(kept)
            discards = [c for c in hand if c not in kept]
            x = featurize_discard(kept, discards, dealer_is_self, None, None)  # np array
            v = float(self.discard_model.predict(x))
            if v > best_v:
                best_v, best = v, tuple(discards)
        return best
    

class NeuralPegPlayer(BeginnerPlayer):
    def __init__(self, discard_model, pegging_model, name="neural"):
        self.name = name
        self.discard_model = discard_model
        self.pegging_model = pegging_model

    def select_card_to_play(self, hand, table, crib, count, past_table_cards=None, starter_card=None):
        if past_table_cards is None:
            past_table_cards = []
        
        playable = [c for c in hand if c + count <= 31]
        if not playable:
            return None
        best, best_v = None, float("-inf")
        for c in playable:
            # Known cards include: hand, current table sequence, past table cards, and starter
            known = hand + table + past_table_cards
            if starter_card is not None:
                known = known + [starter_card]
            x = featurize_pegging(
                hand,
                table,
                count,
                c,
                known_cards=known,
                player_score=0,
                opponent_score=0,
            )  # np array
            v = float(self.pegging_model.predict(x))
            if v > best_v:
                best_v, best = v, c
        return best


class NeuralDiscardOnlyPlayer:
    """Use a neural discard model, but fall back to another player's pegging."""
    def __init__(self, discard_model, pegging_fallback, name="neural_discard_only", discard_feature_set: str = "full"):
        self.name = name
        self.discard_model = discard_model
        self.pegging_fallback = pegging_fallback
        self.discard_feature_indices = get_discard_feature_indices(discard_feature_set)
        self.discard_feature_set = discard_feature_set

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        return select_discard_with_model_with_scores(
            self.discard_model,
            player_state.hand,
            player_state.is_dealer,
            player_state.score,
            player_state.opponent_score,
            self.discard_feature_indices,
            self.discard_feature_set,
        )

    def select_card_to_play(self, player_state, round_state):
        return self.pegging_fallback.select_card_to_play(player_state, round_state)


class NeuralPegOnlyPlayer:
    """Use a neural pegging model, but fall back to another player's discard."""
    def __init__(
        self,
        pegging_model,
        discard_fallback,
        name="neural_peg_only",
        pegging_feature_set: str = "full",
    ):
        self.name = name
        self.pegging_model = pegging_model
        self.discard_fallback = discard_fallback
        self.pegging_feature_set = pegging_feature_set
        self.pegging_feature_indices = get_pegging_feature_indices(pegging_feature_set)

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        return self.discard_fallback.select_crib_cards(player_state, round_state)

    def select_card_to_play(self, player_state, round_state):
        hand = player_state.hand
        table = round_state.table_cards
        crib = round_state.crib
        count = round_state.count
        return regression_pegging_strategy(
            self.pegging_model,
            hand,
            table,
            crib,
            count,
            known_cards=player_state.known_cards,
            opponent_known_hand=player_state.opponent_known_hand,
            all_played_cards=round_state.all_played_cards,
            player_score=player_state.score,
            opponent_score=player_state.opponent_score,
            feature_set=self.pegging_feature_set,
            feature_indices=self.pegging_feature_indices,
        )
