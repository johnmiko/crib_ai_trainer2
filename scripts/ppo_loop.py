"""PPO self-play loop: full games vs frozen best, point-diff reward."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import multiprocessing as mp

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import MODELS_DIR
from crib_ai_trainer.players.neural_player import (
    AIPlayer,
    MLPValueModel,
    NeuralPegOnlyPlayer,
    NeuralDiscardOnlyPlayer,
    PeggingRNNValueModel,
    PeggingTransformerValueModel,
    featurize_discard,
    featurize_pegging,
    get_discard_feature_indices,
    get_pegging_feature_indices,
)
from cribbage.players.hard_player import HardPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.beginner_player import BeginnerPlayer
from scripts.loop_utils import read_queue_models, resolve_model_dir
from cribbage.utils import play_game
from cribbage.cribbagegame import CribbageGame
from cribbage.training_game import TrainingGame


@dataclass
class PPODecision:
    candidate_features: np.ndarray
    action_idx: int
    old_log_prob: float
    advantage: float


def _compute_reward(diff: float, reward_mode: str, winrate_weight: float) -> float:
    if diff > 0:
        win = 1.0
    elif diff < 0:
        win = 0.0
    else:
        win = 0.5
    if reward_mode == "mixed":
        return (winrate_weight * win) + ((1.0 - winrate_weight) * diff)
    return diff


class PolicyMLP:
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], seed: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(int(seed))
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

    def load_from_value_model(self, value_model: MLPValueModel) -> None:
        self.model.load_state_dict(value_model.model.state_dict())

    def logits(self, X: np.ndarray):
        import torch

        t = torch.tensor(X, dtype=torch.float32)
        return self.model(t).squeeze(1)


class PPOPlayer:
    def __init__(
        self,
        discard_policy: PolicyMLP | None,
        pegging_policy: PolicyMLP | None,
        *,
        name: str,
        discard_feature_set: str,
        pegging_feature_set: str,
        rng: np.random.Generator,
        deterministic: bool,
        training_mode: str = "full",
        discard_fallback=None,
        pegging_fallback=None,
    ):
        self.name = name
        self.discard_policy = discard_policy
        self.pegging_policy = pegging_policy
        self.discard_feature_set = discard_feature_set
        self.pegging_feature_set = pegging_feature_set
        self.discard_feature_indices = get_discard_feature_indices(discard_feature_set)
        self.pegging_feature_indices = get_pegging_feature_indices(pegging_feature_set)
        self.rng = rng
        self.deterministic = deterministic
        self.training_mode = training_mode
        self.discard_fallback = discard_fallback
        self.pegging_fallback = pegging_fallback
        self._discard_logs: list[tuple[np.ndarray, int, float]] = []
        self._pegging_logs: list[tuple[np.ndarray, int, float]] = []

    def reset_logs(self) -> None:
        self._discard_logs = []
        self._pegging_logs = []

    def pop_logs(self) -> tuple[list[tuple[np.ndarray, int, float]], list[tuple[np.ndarray, int, float]]]:
        logs = (self._discard_logs, self._pegging_logs)
        self._discard_logs = []
        self._pegging_logs = []
        return logs

    def select_crib_cards(self, player_state, round_state):
        if self.training_mode == "pegging_only" and self.discard_fallback is not None:
            return self.discard_fallback.select_crib_cards(player_state, round_state)
        if self.discard_policy is None:
            raise SystemExit("Discard policy is not available for this PPO run.")
        hand = list(player_state.hand)
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = getattr(player_state, "opponent_score", None)
        candidates = []
        discards_list = []
        for discards in combinations(hand, 2):
            discards = list(discards)
            kept = [c for c in hand if c not in discards]
            feats = featurize_discard(
                kept,
                discards,
                dealer_is_self,
                player_score=your_score,
                opponent_score=opponent_score,
                pegging_ev=None,
            )
            feats = feats[self.discard_feature_indices]
            candidates.append(feats)
            discards_list.append(discards)
        X = np.stack(candidates).astype(np.float32)
        logits = self.discard_policy.logits(X).detach().cpu().numpy()
        probs = _softmax(logits)
        if self.deterministic:
            idx = int(np.argmax(probs))
        else:
            idx = int(self.rng.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[idx]))
        if self.training_mode != "pegging_only":
            self._discard_logs.append((X, idx, log_prob))
        return tuple(discards_list[idx])

    def select_card_to_play(self, player_state, round_state):
        if self.training_mode == "discard_only" and self.pegging_fallback is not None:
            return self.pegging_fallback.select_card_to_play(player_state, round_state)
        if self.training_mode == "discard_only":
            return None
        if self.pegging_policy is None:
            raise SystemExit("Pegging policy is not available for this PPO run.")
        hand = list(player_state.hand)
        table = round_state.table_cards
        count = round_state.count
        legal = []
        feats_list = []
        for c in hand:
            if count + c.value <= 31:
                feats = featurize_pegging(
                    hand,
                    table,
                    count,
                    c,
                    known_cards=player_state.known_cards,
                    opponent_known_hand=player_state.opponent_known_hand,
                    all_played_cards=round_state.all_played_cards,
                    player_score=player_state.score,
                    opponent_score=getattr(player_state, "opponent_score", None),
                    feature_set=self.pegging_feature_set,
                    unseen_value_counts=getattr(round_state, "unseen_value_counts", None),
                    unseen_count=getattr(round_state, "unseen_count", None),
                )
                feats = feats[self.pegging_feature_indices]
                legal.append(c)
                feats_list.append(feats)
        if not legal:
            return None
        X = np.stack(feats_list).astype(np.float32)
        logits = self.pegging_policy.logits(X).detach().cpu().numpy()
        probs = _softmax(logits)
        if self.deterministic:
            idx = int(np.argmax(probs))
        else:
            idx = int(self.rng.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[idx]))
        if self.training_mode != "discard_only":
            self._pegging_logs.append((X, idx, log_prob))
        return legal[idx]


def _softmax(logits: np.ndarray) -> np.ndarray:
    max_logit = float(np.max(logits))
    exps = np.exp(logits - max_logit)
    denom = float(np.sum(exps))
    if denom <= 0:
        raise SystemExit("Softmax denominator is zero.")
    return exps / denom


def _ensure_unique_names(p0, p1):
    if getattr(p0, "name", None) == getattr(p1, "name", None):
        p0.name = f"{p0.name}_0"
        p1.name = f"{p1.name}_1"
    return p0, p1


def _build_opponent(name: str):
    if name == "hard":
        return HardPlayer(name="hard")
    if name == "medium":
        return MediumPlayer(name="medium")
    if name == "beginner":
        return BeginnerPlayer(name="beginner")
    raise SystemExit(f"Unsupported opponent: {name}")


class _NullDiscardFallback:
    def select_crib_cards(self, player_state, round_state):
        raise SystemExit("Discard fallback was called unexpectedly.")


def _make_pegging_player(pegging_model, pegging_feature_set: str):
    return NeuralPegOnlyPlayer(
        pegging_model=pegging_model,
        discard_fallback=_NullDiscardFallback(),
        name="pegging_only_fallback",
        pegging_feature_set=pegging_feature_set,
    )


def _resolve_play_mode(training_mode: str) -> str:
    if training_mode == "discard_only":
        return "full"
    return training_mode


def _resolve_best_dir(args, best_file: Path) -> Path:
    if args.output_root and args.best_run_id:
        return Path(args.output_root) / args.best_run_id
    if args.discard_model_dir:
        base = resolve_model_dir(args.discard_model_dir)
        ppo_dir = base / "ppo"
        if ppo_dir.exists():
            return resolve_model_dir(str(ppo_dir))
        return base
    if args.pegging_model_dir:
        base = resolve_model_dir(args.pegging_model_dir)
        ppo_dir = base / "ppo"
        if ppo_dir.exists():
            return resolve_model_dir(str(ppo_dir))
        return base
    return _get_best_dir(best_file, args.model_version, args.best_run_id)


def _play_n_hands(p0, p1, seed: int | None, hands: int, training_mode: str) -> float:
    p0, p1 = _ensure_unique_names(p0, p1)
    play_mode = _resolve_play_mode(training_mode)
    if play_mode == "full":
        game = CribbageGame(players=[p0, p1], seed=seed, copy_players=False, fast_mode=True)
    else:
        game = TrainingGame(players=[p0, p1], seed=seed, copy_players=False, fast_mode=True, training_mode=play_mode)
    max_score = max(121, 121 * int(hands))
    game.MAX_SCORE = max_score
    game.board.max_score = max_score
    game_score = [0, 0]
    for _ in range(hands):
        game_score = game.play_round(game_score, seed=game.round_seed)
    return float(game_score[0] - game_score[1])


def _next_run_id(base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return "001"
    max_id = max(int(p.name) for p in run_dirs)
    return f"{max_id + 1:03d}"


def _load_best_map(best_file: Path) -> dict:
    if not best_file.exists():
        raise SystemExit(f"Expected best_file at {best_file} but it does not exist.")
    data = json.loads(best_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{best_file} must contain a JSON object.")
    return data


def _get_best_dir(best_file: Path, model_version: str, best_run_id: str | None) -> Path:
    if best_run_id is not None:
        return Path(MODELS_DIR) / "ppo" / model_version / best_run_id
    best_map = _load_best_map(best_file)
    record = best_map.get(model_version)
    if record is None or "path" not in record:
        raise SystemExit(f"Missing entry for '{model_version}' in {best_file}.")
    return Path(record["path"])


def _write_best_dir(best_file: Path, model_version: str, run_dir: Path) -> None:
    best_map = _load_best_map(best_file) if best_file.exists() else {}
    best_map[model_version] = {"path": str(run_dir)}
    best_file.parent.mkdir(parents=True, exist_ok=True)
    best_file.write_text(json.dumps(best_map, indent=2), encoding="utf-8")


def _load_value_models(best_dir: Path, training_mode: str) -> tuple[MLPValueModel | None, MLPValueModel | None, dict]:
    meta_path = best_dir / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Expected model_meta.json at {meta_path} but it does not exist.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    discard_file = meta.get("discard_model_file")
    pegging_file = meta.get("pegging_model_file")
    if not discard_file and not pegging_file:
        raise SystemExit(f"model_meta.json missing discard_model_file/pegging_model_file at {meta_path}.")

    discard_model = None
    if discard_file:
        discard_path = best_dir / discard_file
        if not discard_path.exists():
            raise SystemExit(f"Missing discard model file in {best_dir}: {discard_path}")
        discard_model = MLPValueModel.load_pt(str(discard_path))

    pegging_model = None
    if pegging_file:
        pegging_path = best_dir / pegging_file
        if not pegging_path.exists():
            raise SystemExit(f"Missing pegging model file in {best_dir}: {pegging_path}")
        pegging_model_type = meta.get("pegging_model_type") or meta.get("model_type") or "mlp"
        if pegging_model_type == "mlp":
            pegging_model = MLPValueModel.load_pt(str(pegging_path))
        elif pegging_model_type in {"gru", "lstm"}:
            pegging_model = PeggingRNNValueModel.load_pt(str(pegging_path))
        elif pegging_model_type == "transformer":
            pegging_model = PeggingTransformerValueModel.load_pt(str(pegging_path))
        else:
            raise SystemExit(f"Unsupported pegging_model_type={pegging_model_type} in {best_dir}.")

    if training_mode != "pegging_only" and discard_model is None:
        raise SystemExit(f"Expected discard model in {best_dir} for training_mode={training_mode}.")
    if training_mode != "discard_only" and pegging_model is None:
        raise SystemExit(f"Expected pegging model in {best_dir} for training_mode={training_mode}.")
    return discard_model, pegging_model, meta


def _build_policies(
    discard_dir: Path,
    pegging_dir: Path,
    seed: int,
    training_mode: str,
) -> tuple[PolicyMLP | None, PolicyMLP | None, dict, dict | None]:
    discard_model, _, discard_meta = _load_value_models(discard_dir, training_mode)
    _, pegging_model, pegging_meta = _load_value_models(pegging_dir, training_mode)
    meta = dict(discard_meta)
    if pegging_meta:
        for key in (
            "pegging_feature_set",
            "pegging_feature_dim",
            "pegging_model_type",
            "pegging_model_file",
            "pegging_mlp_hidden",
        ):
            if key in pegging_meta:
                meta[key] = pegging_meta[key]
    pegging_model_type = meta.get("pegging_model_type") or meta.get("model_type") or "mlp"
    if training_mode != "discard_only" and pegging_model_type != "mlp":
        raise SystemExit(f"PPO currently supports only MLP pegging models. Got {pegging_model_type} in {pegging_dir}.")
    if pegging_meta:
        for key in (
            "pegging_feature_set",
            "pegging_feature_dim",
            "pegging_model_type",
            "pegging_model_file",
            "pegging_mlp_hidden",
        ):
            if key in pegging_meta:
                meta[key] = pegging_meta[key]
    hidden = tuple(int(h) for h in meta.get("mlp_hidden", []))
    if not hidden:
        alt = meta.get("pegging_mlp_hidden")
        if isinstance(alt, (list, tuple)):
            hidden = tuple(int(h) for h in alt)
    if not hidden:
        raise SystemExit(f"model_meta.json missing mlp_hidden at {discard_dir}.")
    discard_dim = int(meta.get("discard_feature_dim", 0))
    pegging_dim = int(meta.get("pegging_feature_dim", 0))
    if training_mode != "pegging_only" and discard_dim <= 0:
        raise SystemExit(f"model_meta.json missing discard feature dims at {discard_dir}.")
    if training_mode != "discard_only" and pegging_dim <= 0:
        raise SystemExit(f"model_meta.json missing pegging feature dims at {pegging_dir}.")
    discard_policy = None
    if training_mode != "pegging_only":
        discard_policy = PolicyMLP(discard_dim, hidden, seed)
        discard_policy.load_from_value_model(discard_model)
    pegging_policy = None
    if training_mode != "discard_only":
        pegging_policy = PolicyMLP(pegging_dim, hidden, seed + 1)
        pegging_policy.load_from_value_model(pegging_model)
    return discard_policy, pegging_policy, meta, pegging_meta


def _split_games(total: int, workers: int) -> list[int]:
    if workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if total < 0:
        raise SystemExit("--games_per_iteration must be >= 0.")
    q, r = divmod(total, workers)
    return [(q + 1) if i < r else q for i in range(workers)]


def _policy_state(policy: PolicyMLP) -> dict:
    return {
        "state_dict": policy.model.state_dict(),
        "input_dim": policy.input_dim,
        "hidden_sizes": policy.hidden_sizes,
    }


def _load_policy_from_state(state: dict) -> PolicyMLP:
    import torch

    policy = PolicyMLP(int(state["input_dim"]), tuple(int(h) for h in state["hidden_sizes"]), seed=0)
    policy.model.load_state_dict(state["state_dict"])
    policy.model.eval()
    return policy


def _collect_games_worker(args_tuple: tuple) -> tuple[list[PPODecision], list[PPODecision]]:
    (
        discard_state,
        pegging_state,
        discard_feature_set,
        pegging_feature_set,
        best_dir_str,
        games,
        seed,
        worker_idx,
        hands_per_game,
        training_mode,
        opponent_name,
        discard_dir_str,
        pegging_dir_str,
        discard_pegging_fallback,
        reward_mode,
        winrate_weight,
    ) = args_tuple
    if games <= 0:
        return [], []
    rng = np.random.default_rng(int(seed) + int(worker_idx))
    discard_policy = _load_policy_from_state(discard_state) if discard_state is not None else None
    pegging_policy = _load_policy_from_state(pegging_state) if pegging_state is not None else None
    best_dir = Path(best_dir_str)
    discard_dir = Path(discard_dir_str)
    pegging_dir = Path(pegging_dir_str)
    best_discard, _, _ = _load_value_models(discard_dir, training_mode)
    _, best_pegging, _ = _load_value_models(pegging_dir, training_mode)
    base_opponent = _build_opponent(opponent_name)
    peg_fb = None
    if training_mode == "discard_only":
        if discard_pegging_fallback:
            peg_fb = _build_opponent(discard_pegging_fallback)
        else:
            if best_pegging is None:
                raise SystemExit(f"Expected pegging model in {pegging_dir} for discard_only pegging.")
            peg_fb = _make_pegging_player(best_pegging, pegging_feature_set)
    if training_mode == "pegging_only":
        opponent = NeuralPegOnlyPlayer(
            pegging_model=best_pegging,
            discard_fallback=base_opponent,
            name=f"best:{best_dir.name}",
            pegging_feature_set=pegging_feature_set,
        )
    elif training_mode == "discard_only":
        opponent = AIPlayer(
            best_discard,
            best_pegging,
            name=f"best:{best_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )
    else:
        opponent = AIPlayer(
            best_discard,
            best_pegging,
            name=f"best:{best_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )
    candidate = PPOPlayer(
        discard_policy,
        pegging_policy,
        name="candidate",
        discard_feature_set=discard_feature_set,
        pegging_feature_set=pegging_feature_set,
        rng=rng,
        deterministic=False,
        training_mode=training_mode,
        discard_fallback=base_opponent if training_mode == "pegging_only" else None,
        pegging_fallback=peg_fb,
    )
    _ensure_unique_names(candidate, opponent)
    discard_samples: list[PPODecision] = []
    pegging_samples: list[PPODecision] = []
    base_seed = int(seed) + int(worker_idx) * 100000
    for i in range(games):
        candidate.reset_logs()
        game_seed = base_seed + i
        if hands_per_game is None:
            if i % 2 == 0:
                s0, s1 = play_game(
                    candidate,
                    opponent,
                    seed=game_seed,
                    fast_mode=True,
                    copy_players=False,
                    training_mode=_resolve_play_mode(training_mode),
                )
                diff = float(s0 - s1)
            else:
                s0, s1 = play_game(
                    opponent,
                    candidate,
                    seed=game_seed,
                    fast_mode=True,
                    copy_players=False,
                    training_mode=_resolve_play_mode(training_mode),
                )
                diff = float(s1 - s0)
        else:
            if i % 2 == 0:
                diff = _play_n_hands(candidate, opponent, game_seed, hands_per_game, training_mode)
            else:
                diff = -_play_n_hands(opponent, candidate, game_seed, hands_per_game, training_mode)
        disc_logs, peg_logs = candidate.pop_logs()
        reward = _compute_reward(diff, reward_mode, winrate_weight)
        if training_mode != "pegging_only":
            for feats, idx, logp in disc_logs:
                discard_samples.append(PPODecision(feats, idx, logp, reward))
        if training_mode != "discard_only":
            for feats, idx, logp in peg_logs:
                pegging_samples.append(PPODecision(feats, idx, logp, reward))
    return discard_samples, pegging_samples


def _evaluate_worker(args_tuple: tuple) -> tuple[str, int, float, int]:
    (
        discard_state,
        pegging_state,
        discard_feature_set,
        pegging_feature_set,
        best_dir_str,
        player_kind,
        games,
        seed,
        worker_idx,
        training_mode,
        opponent_name,
        discard_dir_str,
        pegging_dir_str,
        discard_pegging_fallback,
        reward_mode,
        winrate_weight,
    ) = args_tuple
    if games <= 0:
        return player_kind, 0, 0.0, 0
    rng = np.random.default_rng(int(seed) + int(worker_idx))
    discard_policy = _load_policy_from_state(discard_state) if discard_state is not None else None
    pegging_policy = _load_policy_from_state(pegging_state) if pegging_state is not None else None
    best_dir = Path(best_dir_str)
    discard_dir = Path(discard_dir_str)
    pegging_dir = Path(pegging_dir_str)
    best_discard, _, _ = _load_value_models(discard_dir, training_mode)
    _, best_pegging, _ = _load_value_models(pegging_dir, training_mode)
    base_opponent = _build_opponent(opponent_name)
    peg_fb = None
    if training_mode == "discard_only":
        if discard_pegging_fallback:
            peg_fb = _build_opponent(discard_pegging_fallback)
        else:
            if best_pegging is None:
                raise SystemExit(f"Expected pegging model in {pegging_dir} for discard_only pegging.")
            peg_fb = _make_pegging_player(best_pegging, pegging_feature_set)
    if training_mode == "pegging_only":
        opponent = NeuralPegOnlyPlayer(
            pegging_model=best_pegging,
            discard_fallback=base_opponent,
            name=f"best:{best_dir.name}",
            pegging_feature_set=pegging_feature_set,
        )
    elif training_mode == "discard_only":
        opponent = AIPlayer(
            best_discard,
            best_pegging,
            name=f"best:{best_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )
    else:
        opponent = AIPlayer(
            best_discard,
            best_pegging,
            name=f"best:{best_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )
    player_name = "candidate"
    if player_kind == "best":
        discard_policy = _load_policy_from_state(discard_state) if discard_state is not None else None
        pegging_policy = _load_policy_from_state(pegging_state) if pegging_state is not None else None
        player_name = "best"
    candidate = PPOPlayer(
        discard_policy,
        pegging_policy,
        name=player_name,
        discard_feature_set=discard_feature_set,
        pegging_feature_set=pegging_feature_set,
        rng=rng,
        deterministic=True,
        training_mode=training_mode,
        discard_fallback=base_opponent if training_mode == "pegging_only" else None,
        pegging_fallback=peg_fb,
    )
    _ensure_unique_names(candidate, opponent)
    wins = 0
    diff_sum = 0.0
    base_seed = int(seed) + int(worker_idx) * 100000
    for i in range(games):
        game_seed = base_seed + i
        if i % 2 == 0:
            s0, s1 = play_game(
                candidate,
                opponent,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=_resolve_play_mode(training_mode),
            )
            diff = float(s0 - s1)
        else:
            s0, s1 = play_game(
                opponent,
                candidate,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=_resolve_play_mode(training_mode),
            )
            diff = float(s1 - s0)
        if diff > 0:
            wins += 1
        diff_sum += diff
    return player_kind, wins, diff_sum, games


def _evaluate_head_to_head(
    best_state: dict | None,
    cand_state: dict | None,
    discard_feature_set: str,
    pegging_feature_set: str,
    best_dir: Path,
    discard_dir: Path,
    pegging_dir: Path,
    games: int,
    seed: int,
    training_mode: str,
    opponent_name: str,
    discard_pegging_fallback: str | None,
) -> dict:
    if games <= 0:
        return {"wins": 0, "games": 0, "winrate": 0.0, "avg_diff": 0.0}
    best_discard, _, _ = _load_value_models(discard_dir, training_mode)
    _, best_pegging, _ = _load_value_models(pegging_dir, training_mode)
    base_opponent = _build_opponent(opponent_name)
    peg_fb = None
    if training_mode == "discard_only":
        if discard_pegging_fallback:
            peg_fb = _build_opponent(discard_pegging_fallback)
        else:
            if best_pegging is None:
                raise SystemExit(f"Expected pegging model in {pegging_dir} for discard_only pegging.")
            peg_fb = _make_pegging_player(best_pegging, pegging_feature_set)

    def _make_player(state: dict | None, name: str):
        discard_policy = _load_policy_from_state(state) if state is not None else None
        pegging_policy = _load_policy_from_state(state) if state is not None else None
        return PPOPlayer(
            discard_policy,
            pegging_policy,
            name=name,
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
            rng=np.random.default_rng(0),
            deterministic=True,
            training_mode=training_mode,
            discard_fallback=base_opponent if training_mode == "pegging_only" else None,
            pegging_fallback=peg_fb,
        )

    best_player = _make_player(best_state, "best")
    cand_player = _make_player(cand_state, "candidate")
    wins = 0
    diffs = []
    for i in range(games):
        game_seed = int(seed) + i
        if i % 2 == 0:
            s0, s1 = play_game(
                cand_player,
                best_player,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=_resolve_play_mode(training_mode),
            )
            diff = float(s0 - s1)
        else:
            s0, s1 = play_game(
                best_player,
                cand_player,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=_resolve_play_mode(training_mode),
            )
            diff = float(s1 - s0)
        if diff > 0:
            wins += 1
        diffs.append(diff)
    winrate = wins / games if games else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"wins": wins, "games": games, "winrate": winrate, "avg_diff": avg_diff}


def _normalize_advantages(samples: list[PPODecision]) -> None:
    if not samples:
        raise SystemExit("No samples collected for PPO update.")
    adv = np.array([s.advantage for s in samples], dtype=np.float32)
    std = float(np.std(adv))
    if std <= 0:
        raise SystemExit("All advantages are identical; increase games_per_iteration.")
    mean = float(np.mean(adv))
    for s in samples:
        s.advantage = (s.advantage - mean) / std


def _ppo_update(policy: PolicyMLP, samples: list[PPODecision], *, lr: float, epochs: int, batch_size: int, clip: float, entropy_coef: float) -> None:
    import torch
    import torch.optim as optim

    if not samples:
        raise SystemExit("No samples provided for PPO update.")
    optimizer = optim.Adam(policy.model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        for start in range(0, len(samples), batch_size):
            batch_idx = idx[start:start + batch_size]
            if batch_idx.size == 0:
                continue
            loss_sum = 0.0
            for j in batch_idx:
                s = samples[int(j)]
                feats = torch.tensor(s.candidate_features, dtype=torch.float32)
                logits = policy.model(feats).squeeze(1)
                log_probs = torch.log_softmax(logits, dim=0)
                probs = torch.softmax(logits, dim=0)
                new_log_prob = log_probs[s.action_idx]
                ratio = torch.exp(new_log_prob - torch.tensor(float(s.old_log_prob)))
                adv = torch.tensor(float(s.advantage))
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                surrogate = torch.min(ratio * adv, clipped * adv)
                entropy = -(probs * log_probs).sum()
                loss_sum = loss_sum - surrogate - (entropy_coef * entropy)
            loss = loss_sum / float(len(batch_idx))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _evaluate(candidate: PPOPlayer, opponent: AIPlayer, games: int, seed: int, training_mode: str) -> dict:
    candidate.deterministic = True
    wins = 0
    diffs = []
    for i in range(games):
        game_seed = int(seed) + i
        if i % 2 == 0:
            s0, s1 = play_game(
                candidate,
                opponent,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=training_mode,
            )
            diff = float(s0 - s1)
        else:
            s0, s1 = play_game(
                opponent,
                candidate,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=training_mode,
            )
            diff = float(s1 - s0)
        if diff > 0:
            wins += 1
        diffs.append(diff)
    candidate.deterministic = False
    winrate = wins / games if games else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"wins": wins, "games": games, "winrate": winrate, "avg_diff": avg_diff}


def _save_candidate(run_dir: Path, discard_policy: PolicyMLP | None, pegging_policy: PolicyMLP | None, meta: dict) -> None:
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    if discard_policy is not None:
        discard_path = run_dir / "discard_policy.pt"
        torch.save(
            {
                "state_dict": discard_policy.model.state_dict(),
                "input_dim": discard_policy.input_dim,
                "hidden_sizes": discard_policy.hidden_sizes,
            },
            discard_path,
        )
        meta["discard_model_file"] = "discard_policy.pt"
    if pegging_policy is not None:
        pegging_path = run_dir / "pegging_policy.pt"
        torch.save(
            {
                "state_dict": pegging_policy.model.state_dict(),
                "input_dim": pegging_policy.input_dim,
                "hidden_sizes": pegging_policy.hidden_sizes,
            },
            pegging_path,
        )
        meta["pegging_model_file"] = "pegging_policy.pt"
    meta_path = run_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models_dir", type=str, default=str(Path(MODELS_DIR) / "ppo"), help="Base models dir for PPO runs.")
    ap.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="If set, save PPO runs under this directory (run_id subfolders).",
    )
    ap.add_argument("--model_version", type=str, required=True, help="Model version label.")
    ap.add_argument(
        "--start_model_dir",
        type=str,
        default=None,
        help="If set, use this model directory as the starting/best model.",
    )
    ap.add_argument("--queue_models", type=str, default=None, help="Comma-separated list of model dirs to run.")
    ap.add_argument("--queue_file", type=str, default=None, help="File containing one model dir per line.")
    ap.add_argument("--best_file", type=str, default="text/best_models_ppo.json", help="Best-model JSON mapping.")
    ap.add_argument("--best_run_id", type=str, default=None, help="Override best_file and use this run id.")
    ap.add_argument("--games_per_iteration", type=int, default=200, help="Training games per loop.")
    ap.add_argument("--hands_per_game", type=int, default=None, help="Use N hands per game for training only.")
    ap.add_argument("--workers", type=int, default=10, help="Worker processes for data collection.")
    ap.add_argument("--eval_games", type=int, default=200, help="Eval games per loop.")
    ap.add_argument("--eval_workers", type=int, default=10, help="Worker processes for eval.")
    ap.add_argument(
        "--training_mode",
        type=str,
        default="full",
        choices=["full", "discard_only", "pegging_only"],
        help="Play full games or isolate discard/pegging outcomes.",
    )
    ap.add_argument("--discard_model_dir", type=str, default=None, help="Use this directory for the discard model.")
    ap.add_argument("--pegging_model_dir", type=str, default=None, help="Use this directory for the pegging model.")
    ap.add_argument(
        "--discard_pegging_fallback",
        type=str,
        default=None,
        choices=["hard", "medium", "beginner"],
        help="If set, use this opponent's pegging during discard-only training instead of the pegging model.",
    )
    ap.add_argument("--opponent", type=str, default="hard", choices=["hard", "medium", "beginner"])
    ap.add_argument("--loops", type=int, default=100, help="Number of PPO loops.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    ap.add_argument("--ppo_epochs", type=int, default=2, help="PPO epochs per loop.")
    ap.add_argument("--batch_size", type=int, default=64, help="PPO batch size.")
    ap.add_argument("--clip", type=float, default=0.2, help="PPO clip ratio.")
    ap.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    ap.add_argument("--accept_margin", type=float, default=0.0, help="Required improvement margin to accept.")
    ap.add_argument(
        "--accept_metric",
        type=str,
        default="avg_diff",
        choices=["avg_diff", "winrate"],
        help="Metric used to accept/reject candidates.",
    )
    ap.add_argument(
        "--reward_mode",
        type=str,
        default="point_diff",
        choices=["point_diff", "mixed"],
        help="Reward used for PPO advantages.",
    )
    ap.add_argument("--winrate_weight", type=float, default=0.7, help="Winrate weight for mixed reward.")
    ap.add_argument(
        "--plateau_window",
        type=int,
        default=5,
        help="If set, stop when there are zero accepted models in the last N loops.",
    )
    ap.add_argument("--plateau_winrate_step", type=float, default=0.1, help="Winrate weight step per plateau window.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = ap.parse_args()

    loops = args.loops
    if loops == 0:
        raise SystemExit("--loops must be non-zero.")
    if args.games_per_iteration <= 0:
        raise SystemExit("--games_per_iteration must be > 0.")
    if args.eval_games <= 0:
        raise SystemExit("--eval_games must be > 0.")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if args.eval_workers <= 0:
        raise SystemExit("--eval_workers must be > 0.")
    if args.hands_per_game is not None and args.hands_per_game <= 0:
        raise SystemExit("--hands_per_game must be > 0 if provided.")
    if args.reward_mode == "mixed" and (args.winrate_weight < 0.0 or args.winrate_weight > 1.0):
        raise SystemExit("--winrate_weight must be in [0, 1].")

    queue = read_queue_models(args.queue_models, args.queue_file)
    if queue:
        for model_dir in queue:
            run_args = argparse.Namespace(**vars(args))
            if args.training_mode == "pegging_only":
                run_args.pegging_model_dir = model_dir
            else:
                run_args.discard_model_dir = model_dir
            if run_args.output_root is None:
                run_args.output_root = str(Path(model_dir) / "ppo")
            _run_single(run_args)
        return 0

    return _run_single(args)


def _run_single(args: argparse.Namespace) -> int:
    loops = args.loops
    seed = int(args.seed) if args.seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))
    print(f"Using random seed: {seed}")
    best_file = Path(args.best_file)
    if args.output_root:
        base_models_dir = Path(args.output_root)
    elif args.discard_model_dir:
        root = Path(args.discard_model_dir)
        base_models_dir = root if root.name.lower() == "ppo" else root / "ppo"
    elif args.pegging_model_dir:
        root = Path(args.pegging_model_dir)
        base_models_dir = root if root.name.lower() == "ppo" else root / "ppo"
    else:
        base_models_dir = Path(args.models_dir) / args.model_version
    base_models_dir.mkdir(parents=True, exist_ok=True)

    if args.start_model_dir:
        start_dir = Path(args.start_model_dir)
        if not start_dir.exists():
            raise SystemExit(f"--start_model_dir not found: {start_dir}")
        _write_best_dir(best_file, args.model_version, start_dir)

    best_dir = _resolve_best_dir(args, best_file)
    if not best_dir.exists():
        raise SystemExit(f"Best model dir not found: {best_dir}")
    print(f"Current best model: {best_dir}")
    loop_total = "infinite" if loops < 0 else str(loops)
    loop_idx = 0
    accept_history: list[bool] = []
    winrate_weight = float(args.winrate_weight)
    while loops < 0 or loop_idx < loops:
        loop_idx += 1
        print(f"=== PPO loop {loop_idx}/{loop_total} ===")

        # best_dir is updated only on acceptance

        discard_source_dir = resolve_model_dir(args.discard_model_dir) if args.discard_model_dir else best_dir
        pegging_source_dir = resolve_model_dir(args.pegging_model_dir) if args.pegging_model_dir else best_dir
        if args.discard_model_dir and not discard_source_dir.exists():
            raise SystemExit(f"--discard_model_dir not found: {discard_source_dir}")
        if args.pegging_model_dir and not pegging_source_dir.exists():
            raise SystemExit(f"--pegging_model_dir not found: {pegging_source_dir}")

        run_training_mode = args.training_mode
        if args.discard_model_dir and not args.pegging_model_dir:
            run_training_mode = "discard_only"
            if args.training_mode != "discard_only":
                print("Info: using training_mode=discard_only to update only discard while playing full games.")

        discard_policy, pegging_policy, meta, pegging_meta = _build_policies(
            discard_source_dir,
            pegging_source_dir,
            seed + loop_idx * 10,
            run_training_mode,
        )
        discard_feature_set = meta.get("discard_feature_set", "full")
        pegging_feature_set = meta.get("pegging_feature_set", "full")

        best_discard_state = _policy_state(discard_policy) if discard_policy is not None else None
        best_pegging_state = _policy_state(pegging_policy) if pegging_policy is not None else None
        collect_counts = _split_games(args.games_per_iteration, args.workers)
        collect_tasks = []
        for idx, count in enumerate(collect_counts):
            if count <= 0:
                continue
            collect_tasks.append(
                (
                    best_discard_state,
                    best_pegging_state,
                    discard_feature_set,
                    pegging_feature_set,
                    str(best_dir),
                    int(count),
                    seed + loop_idx * 1000,
                    idx,
                    args.hands_per_game,
                    run_training_mode,
                    args.opponent,
                    str(discard_source_dir),
                    str(pegging_source_dir),
                    args.discard_pegging_fallback,
                    args.reward_mode,
                    winrate_weight,
                )
            )
        ctx = mp.get_context("spawn")
        discard_samples: list[PPODecision] = []
        pegging_samples: list[PPODecision] = []
        with ctx.Pool(processes=args.workers) as pool:
            for d_s, p_s in pool.imap_unordered(_collect_games_worker, collect_tasks):
                discard_samples.extend(d_s)
                pegging_samples.extend(p_s)
        if run_training_mode != "pegging_only":
            _normalize_advantages(discard_samples)
            _ppo_update(
                discard_policy,
                discard_samples,
                lr=args.lr,
                epochs=args.ppo_epochs,
                batch_size=args.batch_size,
                clip=args.clip,
                entropy_coef=args.entropy_coef,
            )
        if run_training_mode != "discard_only":
            _normalize_advantages(pegging_samples)
            _ppo_update(
                pegging_policy,
                pegging_samples,
                lr=args.lr,
                epochs=args.ppo_epochs,
                batch_size=args.batch_size,
                clip=args.clip,
                entropy_coef=args.entropy_coef,
            )

        eval_seed = seed + loop_idx * 2000
        eval_counts = _split_games(args.eval_games, args.eval_workers)
        eval_tasks = []
        cand_discard_state = _policy_state(discard_policy) if discard_policy is not None else None
        cand_pegging_state = _policy_state(pegging_policy) if pegging_policy is not None else None
        for idx, count in enumerate(eval_counts):
            if count <= 0:
                continue
            eval_tasks.append(
                (
                    cand_discard_state,
                    cand_pegging_state,
                    discard_feature_set,
                    pegging_feature_set,
                    str(best_dir),
                    "candidate",
                    int(count),
                    eval_seed,
                    idx,
                    run_training_mode,
                    args.opponent,
                    str(discard_source_dir),
                    str(pegging_source_dir),
                    args.discard_pegging_fallback,
                    args.reward_mode,
                    winrate_weight,
                )
            )
            eval_tasks.append(
                (
                    best_discard_state,
                    best_pegging_state,
                    discard_feature_set,
                    pegging_feature_set,
                    str(best_dir),
                    "best",
                    int(count),
                    eval_seed,
                    idx + 10000,
                    run_training_mode,
                    args.opponent,
                    str(discard_source_dir),
                    str(pegging_source_dir),
                    args.discard_pegging_fallback,
                    args.reward_mode,
                    winrate_weight,
                )
            )
        wins = 0
        total_games = 0
        diff_sum = 0.0
        best_wins = 0
        best_games = 0
        best_diff_sum = 0.0
        with ctx.Pool(processes=args.eval_workers) as pool:
            for kind, w, d_sum, g in pool.imap_unordered(_evaluate_worker, eval_tasks):
                if kind == "candidate":
                    wins += int(w)
                    diff_sum += float(d_sum)
                    total_games += int(g)
                else:
                    best_wins += int(w)
                    best_diff_sum += float(d_sum)
                    best_games += int(g)
        if total_games <= 0:
            raise SystemExit("No eval games were played.")
        eval_result = {
            "wins": wins,
            "games": total_games,
            "winrate": wins / total_games,
            "avg_diff": diff_sum / total_games,
        }
        best_eval = {
            "wins": best_wins,
            "games": best_games,
            "winrate": (best_wins / best_games) if best_games else 0.0,
            "avg_diff": (best_diff_sum / best_games) if best_games else 0.0,
        }
        h2h_eval = _evaluate_head_to_head(
            best_discard_state,
            cand_discard_state,
            discard_feature_set,
            pegging_feature_set,
            best_dir,
            discard_source_dir,
            pegging_source_dir,
            eval_games,
            eval_seed,
            run_training_mode,
            args.opponent,
            args.discard_pegging_fallback,
        )
        print(
            f"eval: wins={eval_result['wins']}/{eval_result['games']} "
            f"winrate={eval_result['winrate']:.3f} avg_diff={eval_result['avg_diff']:.2f}"
        )
        print(
            f"h2h: wins={h2h_eval['wins']}/{h2h_eval['games']} "
            f"winrate={h2h_eval['winrate']:.3f} avg_diff={h2h_eval['avg_diff']:.2f}"
        )

        if args.accept_metric == "winrate":
            cand_metric = eval_result["winrate"]
            best_metric = best_eval["winrate"]
        else:
            cand_metric = eval_result["avg_diff"]
            best_metric = best_eval["avg_diff"]

        accepted = (
            h2h_eval["winrate"] > 0.5
            and cand_metric > (best_metric + float(args.accept_margin))
        )
        if accepted:
            run_id = _next_run_id(base_models_dir)
            run_dir = base_models_dir / run_id
            out_meta = {
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_version": args.model_version,
                "run_id": run_id,
                "source_best": str(best_dir),
                "model_type": "mlp",
                "training_mode": run_training_mode,
                "discard_feature_set": discard_feature_set,
                "pegging_feature_set": pegging_feature_set,
                "discard_feature_dim": discard_policy.input_dim if discard_policy is not None else 0,
                "pegging_feature_dim": pegging_policy.input_dim if pegging_policy is not None else 0,
                "mlp_hidden": list(discard_policy.hidden_sizes) if discard_policy is not None else (
                    list(pegging_policy.hidden_sizes) if pegging_policy is not None else []
                ),
                "lr": args.lr,
                "ppo_epochs": args.ppo_epochs,
                "batch_size": args.batch_size,
                "clip": args.clip,
                "entropy_coef": args.entropy_coef,
                "games_per_iteration": args.games_per_iteration,
                "eval_games": args.eval_games,
                "eval_avg_diff": eval_result["avg_diff"],
                "eval_winrate": eval_result["winrate"],
            }
            _save_candidate(run_dir, discard_policy, pegging_policy, out_meta)
            _write_best_dir(best_file, args.model_version, run_dir)
            best_dir = run_dir
            print(f"Accepted candidate -> {run_dir}")
        else:
            print("Rejected candidate (did not improve).")

        accept_history.append(accepted)
        if args.plateau_window is not None and args.plateau_window > 0 and len(accept_history) >= args.plateau_window:
            window = accept_history[-args.plateau_window :]
            if not any(window):
                if args.reward_mode == "mixed" and winrate_weight < 1.0 - 1e-9:
                    new_weight = min(1.0, winrate_weight + float(args.plateau_winrate_step))
                    print(
                        f"Plateau detected (0/{args.plateau_window} accepted). "
                        f"Increasing winrate_weight {winrate_weight:.2f} -> {new_weight:.2f}."
                    )
                    winrate_weight = new_weight
                    accept_history = []
                else:
                    print(f"Stopping on plateau: 0/{args.plateau_window} accepted in last window.")
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Script summary: PPO self-play loop vs frozen best, full games, point-diff reward.
