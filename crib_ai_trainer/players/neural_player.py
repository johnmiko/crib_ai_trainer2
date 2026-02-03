from itertools import combinations
import numpy as np
from typing import List, Tuple

from cribbage.playingcards import Card
from cribbage.players.beginner_player import BeginnerPlayer

from crib_ai_trainer.features import multi_hot_cards


def featurize_discard(
    kept: List[Card],
    discards: List[Card],
    dealer_is_self: bool,
) -> np.ndarray:
    disc_vec = multi_hot_cards(discards)    # (52,)
    kept_vec = multi_hot_cards(kept)          # (52,)
    dealer_vec = np.array([1.0 if dealer_is_self else 0.0], dtype=np.float32)

    return np.concatenate([
        disc_vec,
        kept_vec,
        dealer_vec,
    ])

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
) -> np.ndarray:
    if known_cards is None:
        known_cards = []
    
    hand_vec = multi_hot_cards(hand)           # (52,)
    table_vec = multi_hot_cards(table)         # (52,)
    count_vec = one_hot_count(count)           # (32,)
    known_vec = multi_hot_cards(known_cards)  # (52,)

    cand_vec = np.zeros(52, dtype=np.float32)
    cand_vec[candidate.to_index()] = 1.0

    return np.concatenate([
        hand_vec,
        table_vec,
        count_vec,
        cand_vec,
        known_vec
    ])


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

def select_discard_with_model(discard_model, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
    if hasattr(discard_model, "predict_scores"):
        Xs: List[np.ndarray] = []
        discards_list: List[Tuple[Card, Card]] = []
        for kept in combinations(hand, 4):
            kept = list(kept)
            discards = [c for c in hand if c not in kept]
            discards_list.append((discards[0], discards[1]))
            Xs.append(featurize_discard(kept, discards, dealer_is_self))
        X15 = np.stack(Xs, axis=0).astype(np.float32)  # (15, D)
        scores = discard_model.predict_scores(X15)  # (15,)
        best_i = int(np.argmax(scores))
        return discards_list[best_i]

    best, best_v = None, float("-inf")
    for kept in combinations(hand, 4):
        kept = list(kept)
        discards = [c for c in hand if c not in kept]
        x = featurize_discard(kept, discards, dealer_is_self)  # np array
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
        x = featurize_pegging(hand, table, count, c, known_cards=known)  # np array
        v = float(pegging_model.predict(x))
        if v > best_v:
            best_v, best = v, c
    return best

class NeuralRegressionPlayer:
    def __init__(self, discard_model, pegging_model, name="neural"):
        self.name = name
        self.discard_model = discard_model
        self.pegging_model = pegging_model

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        # Extract hand and dealer info from state objects
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = None  # Not available in state
        return self.select_crib_cards_regressor(hand, dealer_is_self, your_score, opponent_score) # type: ignore

    def select_crib_cards_regressor(self, hand, dealer_is_self, your_score=None, opponent_score=None) -> Tuple[Card, Card]:
        return select_discard_with_model(self.discard_model, hand, dealer_is_self)

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
        )
        return best

class NeuralClassificationPlayer:
    def __init__(self, discard_model, pegging_model, name="neural"):
        self.name = name
        self.discard_model = discard_model
        self.pegging_model = pegging_model

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        # Extract hand and dealer info from state objects
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = None  # Not available in state
        return self.select_crib_cards_classification(hand, dealer_is_self, your_score, opponent_score) # type: ignore


    def select_crib_cards_classification(self, hand: List[Card], dealer_is_self: bool, your_score=None, opponent_score=None) -> Tuple[Card, Card]:
        return select_discard_with_model(self.discard_model, hand, dealer_is_self)

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
            x = featurize_discard(kept, discards, dealer_is_self)  # np array
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
            x = featurize_pegging(hand, table, count, c, known_cards=known)  # np array
            v = float(self.pegging_model.predict(x))
            if v > best_v:
                best_v, best = v, c
        return best


class NeuralDiscardOnlyPlayer:
    """Use a neural discard model, but fall back to another player's pegging."""
    def __init__(self, discard_model, pegging_fallback, name="neural_discard_only"):
        self.name = name
        self.discard_model = discard_model
        self.pegging_fallback = pegging_fallback

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        return select_discard_with_model(self.discard_model, player_state.hand, player_state.is_dealer)

    def select_card_to_play(self, player_state, round_state):
        return self.pegging_fallback.select_card_to_play(player_state, round_state)


class NeuralPegOnlyPlayer:
    """Use a neural pegging model, but fall back to another player's discard."""
    def __init__(self, pegging_model, discard_fallback, name="neural_peg_only"):
        self.name = name
        self.pegging_model = pegging_model
        self.discard_fallback = discard_fallback

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
        )
