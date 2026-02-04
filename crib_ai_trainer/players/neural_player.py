from itertools import combinations
import numpy as np
from typing import List, Tuple

from cribbage.playingcards import Card
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.cribbagegame import score_play
from cribbage.scoring import HasPairTripleQuad, HasStraight_DuringPlay
from cribbage.players.rule_based_player import get_full_deck

from crib_ai_trainer.features import multi_hot_cards

RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
RANK_TO_I = {r: i for i, r in enumerate(RANKS)}
TENS_RANKS = {"10", "j", "q", "k"}

# Base discard features (52 discards + 52 kept + 1 dealer flag)
BASE_DISCARD_FEATURE_DIM = 105

# Engineered discard features count:
# 13 kept rank counts + 13 discard rank counts +
# 2 value sums + 2 tens counts + 2 fives counts +
# 3 kept pair/trip/quad + 3 discard pair/trip/quad +
# 3 run counts (3/4/5) + 1 run max +
# 2 flush flags (kept/discard) + 1 nobs + 2 fifteen counts +
# 2 scores + 1 score margin + 3 endgame flags
ENGINEERED_DISCARD_NO_SCORE_DIM = 47
ENGINEERED_DISCARD_SCORE_DIM = 6
ENGINEERED_DISCARD_FEATURE_DIM = ENGINEERED_DISCARD_NO_SCORE_DIM + ENGINEERED_DISCARD_SCORE_DIM

DISCARD_FEATURE_DIM = BASE_DISCARD_FEATURE_DIM + ENGINEERED_DISCARD_FEATURE_DIM

# Pegging features
PEGGING_BASE_FEATURE_DIM = 240  # hand(52) + table(52) + count(32) + candidate(52) + known(52)
PEGGING_ENGINEERED_NO_SCORE_DIM = 20
PEGGING_ENGINEERED_SCORE_DIM = 5
PEGGING_ENGINEERED_FEATURE_DIM = PEGGING_ENGINEERED_NO_SCORE_DIM + PEGGING_ENGINEERED_SCORE_DIM
PEGGING_FULL_FEATURE_DIM = PEGGING_BASE_FEATURE_DIM + 52 + 52 + PEGGING_ENGINEERED_FEATURE_DIM  # + opp_played + all_played

def get_pegging_feature_dim(feature_set: str) -> int:
    if feature_set == "basic":
        return PEGGING_BASE_FEATURE_DIM
    if feature_set == "full":
        return PEGGING_FULL_FEATURE_DIM
    if feature_set == "full_no_scores":
        return PEGGING_BASE_FEATURE_DIM + 52 + 52 + PEGGING_ENGINEERED_NO_SCORE_DIM
    raise ValueError(f"Unknown pegging feature_set: {feature_set}")


def get_discard_feature_indices(feature_set: str) -> np.ndarray:
    if feature_set == "base":
        return np.arange(BASE_DISCARD_FEATURE_DIM, dtype=np.int64)
    if feature_set == "engineered_no_scores":
        end = BASE_DISCARD_FEATURE_DIM + ENGINEERED_DISCARD_NO_SCORE_DIM
        return np.arange(end, dtype=np.int64)
    if feature_set == "full":
        return np.arange(DISCARD_FEATURE_DIM, dtype=np.int64)
    raise ValueError(f"Unknown discard feature_set: {feature_set}")


def get_pegging_feature_indices(feature_set: str) -> np.ndarray:
    if feature_set == "base":
        return np.arange(PEGGING_BASE_FEATURE_DIM, dtype=np.int64)
    if feature_set == "full_no_scores":
        end = PEGGING_BASE_FEATURE_DIM + 52 + 52 + PEGGING_ENGINEERED_NO_SCORE_DIM
        return np.arange(end, dtype=np.int64)
    if feature_set == "full":
        return np.arange(PEGGING_FULL_FEATURE_DIM, dtype=np.int64)
    raise ValueError(f"Unknown pegging feature_set: {feature_set}")


def _rank_counts(cards: List[Card]) -> np.ndarray:
    counts = np.zeros(13, dtype=np.float32)
    for c in cards:
        counts[RANK_TO_I[c.get_rank().lower()]] += 1.0
    return counts


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


def _count_fifteens(cards: List[Card]) -> float:
    values = [c.get_value() for c in cards]
    total = 0
    for r in range(2, len(values) + 1):
        for combo in combinations(values, r):
            if sum(combo) == 15:
                total += 1
    return float(total)


def _all_same_suit(cards: List[Card]) -> float:
    if not cards:
        return 0.0
    suit = cards[0].get_suit()
    return 1.0 if all(c.get_suit() == suit for c in cards) else 0.0


def _has_nobs(cards: List[Card]) -> float:
    return 1.0 if any(c.get_rank().lower() == "j" for c in cards) else 0.0

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


def featurize_discard(
    kept: List[Card],
    discards: List[Card],
    dealer_is_self: bool,
    player_score: int | None = None,
    opponent_score: int | None = None,
) -> np.ndarray:
    disc_vec = multi_hot_cards(discards)    # (52,)
    kept_vec = multi_hot_cards(kept)          # (52,)
    dealer_vec = np.array([1.0 if dealer_is_self else 0.0], dtype=np.float32)

    score_context = _score_context_features(player_score, opponent_score)

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
    full_deck = get_full_deck()
    known_set = set(known_cards) | set(all_played_cards) | set(hand)
    unseen = [c for c in full_deck if c not in known_set]
    playable_unseen = [c for c in unseen if c.get_value() <= remaining_to_31]
    opp_can_play_prob = float(len(playable_unseen) / max(1, len(unseen)))
    opp_playable_count = float(len(playable_unseen))
    unseen_count = float(len(unseen))

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

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (128, 64), seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
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

        torch.manual_seed(seed)
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

def select_discard_with_model(discard_model, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
    return select_discard_with_model_with_scores(discard_model, hand, dealer_is_self, None, None)

def select_discard_with_model_with_scores(
    discard_model,
    hand: List[Card],
    dealer_is_self: bool,
    player_score: int | None,
    opponent_score: int | None,
    feature_indices: np.ndarray | None = None,
) -> Tuple[Card, Card]:
    if hasattr(discard_model, "predict_scores"):
        Xs: List[np.ndarray] = []
        discards_list: List[Tuple[Card, Card]] = []
        for kept in combinations(hand, 4):
            kept = list(kept)
            discards = [c for c in hand if c not in kept]
            discards_list.append((discards[0], discards[1]))
            x = featurize_discard(kept, discards, dealer_is_self, player_score, opponent_score)
            if feature_indices is not None:
                x = x[feature_indices]
            Xs.append(x)
        X15 = np.stack(Xs, axis=0).astype(np.float32)  # (15, D)
        scores = discard_model.predict_scores(X15)  # (15,)
        best_i = int(np.argmax(scores))
        return discards_list[best_i]

    best, best_v = None, float("-inf")
    for kept in combinations(hand, 4):
        kept = list(kept)
        discards = [c for c in hand if c not in kept]
        x = featurize_discard(kept, discards, dealer_is_self, player_score, opponent_score)  # np array
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
    for c in playable:
        if known_cards is None:
            # Known cards include: hand, current table sequence, past table cards, and starter
            known = hand + table + past_table_cards
            if starter_card is not None:
                known = known + [starter_card]
        else:
            known = known_cards
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
        )  # np array
        if feature_indices is not None:
            x = x[feature_indices]
        v = float(pegging_model.predict(x))
        if v > best_v:
            best_v, best = v, c
    return best

class NeuralRegressionPlayer:
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

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        return select_discard_with_model_with_scores(
            self.discard_model,
            player_state.hand,
            player_state.is_dealer,
            player_state.score,
            player_state.opponent_score,
            self.discard_feature_indices,
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
