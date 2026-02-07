"""Analyze discard label margins to see how brittle the decision is.

This script reconstructs 6-card hands from classification datasets and
computes the gap between the best and second-best discard according to
the same scoring used by the MediumPlayer discard strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple
from itertools import combinations

import numpy as np
import pandas as pd

from cribbage.playingcards import Card
from cribbage.players.rule_based_player import get_full_deck
from cribbage.strategies.hand_strategies import process_dealt_hand_only_exact
from cribbage.strategies.crib_strategies import calc_crib_min_only_given_6_cards
from cribbage.database import normalize_hand_to_str


RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
SUITS = ["h", "d", "c", "s"]


def card_from_index(i: int) -> Card:
    suit = SUITS[i // 13]
    rank = RANKS[i % 13]
    return Card(f"{rank}{suit}")


def decode_hand_from_option(x_option: np.ndarray) -> Tuple[List[Card], bool]:
    """Decode a hand and dealer flag from a single discard feature vector.

    We only use the first 105 dims (52 discards + 52 kept + 1 dealer flag).
    """
    if x_option.shape[0] < 105:
        raise ValueError(f"Expected discard feature length >= 105, got {x_option.shape[0]}")
    disc_vec = x_option[:52]
    kept_vec = x_option[52:104]
    dealer_flag = bool(round(float(x_option[104])))

    disc_idx = np.where(disc_vec > 0.5)[0].tolist()
    kept_idx = np.where(kept_vec > 0.5)[0].tolist()
    if len(disc_idx) != 2 or len(kept_idx) != 4:
        raise ValueError(f"Unexpected discard/kept sizes: {len(disc_idx)} / {len(kept_idx)}")

    hand_cards = [card_from_index(i) for i in disc_idx + kept_idx]
    return hand_cards, dealer_flag


def score_discard_options(hand: List[Card], dealer_is_self: bool, full_deck: List[Card]) -> List[float]:
    """Compute avg_total_score for each discard option for the given 6-card hand."""
    hand_score_cache = {}
    crib_score_cache = {}

    hand_results = process_dealt_hand_only_exact([hand, full_deck, hand_score_cache])
    df_hand = pd.DataFrame(hand_results, columns=["hand_key", "min_hand_score", "max_hand_score", "avg_hand_score"])
    crib_results = calc_crib_min_only_given_6_cards(hand)
    df_crib = pd.DataFrame(crib_results, columns=["hand_key", "crib_key", "min_crib_score", "avg_crib_score"])
    df3 = pd.merge(df_hand, df_crib, on=["hand_key"])
    df3["avg_total_score"] = df3["avg_hand_score"] + (
        df3["avg_crib_score"] if dealer_is_self else -df3["avg_crib_score"]
    )

    scores: List[float] = []
    for kept in combinations(hand, 4):
        kept_list = list(kept)
        discards_list = [c for c in hand if c not in kept_list]
        hand_key = normalize_hand_to_str(kept_list)
        crib_key = normalize_hand_to_str(discards_list)
        row = df3[(df3["hand_key"] == hand_key) & (df3["crib_key"] == crib_key)]
        if len(row) == 0:
            # Shouldn't happen, but avoid crashing on a single mismatch.
            scores.append(float("nan"))
            continue
        scores.append(float(row["avg_total_score"].values[0]))
    return scores


def iterate_discard_samples(
    data_dir: Path,
    max_shards: int | None,
    max_hands: int,
    seed: int,
) -> Iterable[Tuple[List[Card], bool]]:
    discard_shards = sorted(data_dir.glob("discard_*.npz"))
    if not discard_shards:
        raise SystemExit(f"No discard shards in {data_dir} (expected discard_*.npz)")
    if max_shards is not None:
        discard_shards = discard_shards[:max_shards]

    rng = np.random.default_rng(seed)
    remaining = max_hands

    for shard in discard_shards:
        with np.load(shard) as d:
            X = d["X"]
        if X.ndim != 3 or X.shape[1] != 15 or X.shape[2] != 105:
            raise SystemExit(
                f"Expected classification discard data (N,15,105), got {X.shape} in {shard}"
            )

        n = X.shape[0]
        if n == 0:
            continue

        take = min(remaining, n)
        idx = rng.choice(n, size=take, replace=False)
        for i in idx:
            hand, dealer_is_self = decode_hand_from_option(X[i, 0])
            yield hand, dealer_is_self

        remaining -= take
        if remaining <= 0:
            break


def analyze(data_dir: Path, max_shards: int | None, max_hands: int, seed: int) -> None:
    full_deck = get_full_deck()
    margins: List[float] = []
    top2_spread: List[float] = []
    close_0_1 = 0
    close_0_5 = 0
    close_1_0 = 0
    total = 0
    nan_scores = 0

    for hand, dealer_is_self in iterate_discard_samples(data_dir, max_shards, max_hands, seed):
        scores = score_discard_options(hand, dealer_is_self, full_deck)
        scores = [s for s in scores if not np.isnan(s)]
        if len(scores) < 2:
            nan_scores += 1
            continue
        scores_sorted = sorted(scores, reverse=True)
        margin = scores_sorted[0] - scores_sorted[1]
        margins.append(margin)
        top2_spread.append(scores_sorted[0] - scores_sorted[1])
        if margin <= 0.1:
            close_0_1 += 1
        if margin <= 0.5:
            close_0_5 += 1
        if margin <= 1.0:
            close_1_0 += 1
        total += 1

    if total == 0:
        print("No valid hands analyzed.")
        return

    margins_np = np.array(margins, dtype=np.float32)
    print(f"Hands analyzed: {total}")
    print(f"Hands skipped (NaN scores): {nan_scores}")
    print(f"Margin mean: {float(margins_np.mean()):.3f}")
    print(f"Margin median: {float(np.median(margins_np)):.3f}")
    print(f"Margin 25/75 pct: {float(np.percentile(margins_np,25)):.3f} / {float(np.percentile(margins_np,75)):.3f}")
    print(f"Margin <= 0.1: {close_0_1} ({close_0_1/total*100:.1f}%)")
    print(f"Margin <= 0.5: {close_0_5} ({close_0_5/total*100:.1f}%)")
    print(f"Margin <= 1.0: {close_1_0} ({close_1_0/total*100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="datasets")
    ap.add_argument("--max_shards", type=int, default=1)
    ap.add_argument("--max_hands", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    analyze(Path(args.data_dir), args.max_shards, args.max_hands, args.seed)
