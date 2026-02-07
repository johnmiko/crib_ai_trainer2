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
    featurize_discard,
    featurize_pegging,
    get_discard_feature_indices,
    get_pegging_feature_indices,
)
from cribbage.utils import play_game
from cribbage.cribbagegame import CribbageGame


@dataclass
class PPODecision:
    candidate_features: np.ndarray
    action_idx: int
    old_log_prob: float
    advantage: float


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
        discard_policy: PolicyMLP,
        pegging_policy: PolicyMLP,
        *,
        name: str,
        discard_feature_set: str,
        pegging_feature_set: str,
        rng: np.random.Generator,
        deterministic: bool,
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
        self._discard_logs.append((X, idx, log_prob))
        return tuple(discards_list[idx])

    def select_card_to_play(self, player_state, round_state):
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


def _play_n_hands(p0, p1, seed: int | None, hands: int) -> float:
    p0, p1 = _ensure_unique_names(p0, p1)
    game = CribbageGame(players=[p0, p1], seed=seed, copy_players=False, fast_mode=True)
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


def _load_value_models(best_dir: Path) -> tuple[MLPValueModel, MLPValueModel, dict]:
    meta_path = best_dir / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Expected model_meta.json at {meta_path} but it does not exist.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    discard_file = meta.get("discard_model_file")
    pegging_file = meta.get("pegging_model_file")
    if not discard_file or not pegging_file:
        raise SystemExit(f"model_meta.json missing discard_model_file/pegging_model_file at {meta_path}.")
    discard_path = best_dir / discard_file
    pegging_path = best_dir / pegging_file
    if not discard_path.exists() or not pegging_path.exists():
        raise SystemExit(f"Missing model files in {best_dir}.")
    discard_model = MLPValueModel.load_pt(str(discard_path))
    pegging_model = MLPValueModel.load_pt(str(pegging_path))
    return discard_model, pegging_model, meta


def _build_policies(best_dir: Path, seed: int) -> tuple[PolicyMLP, PolicyMLP, dict]:
    discard_model, pegging_model, meta = _load_value_models(best_dir)
    hidden = tuple(int(h) for h in meta.get("mlp_hidden", []))
    if not hidden:
        raise SystemExit(f"model_meta.json missing mlp_hidden at {best_dir}.")
    discard_dim = int(meta.get("discard_feature_dim", 0))
    pegging_dim = int(meta.get("pegging_feature_dim", 0))
    if discard_dim <= 0 or pegging_dim <= 0:
        raise SystemExit(f"model_meta.json missing feature dims at {best_dir}.")
    discard_policy = PolicyMLP(discard_dim, hidden, seed)
    pegging_policy = PolicyMLP(pegging_dim, hidden, seed + 1)
    discard_policy.load_from_value_model(discard_model)
    pegging_policy.load_from_value_model(pegging_model)
    return discard_policy, pegging_policy, meta


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
    ) = args_tuple
    if games <= 0:
        return [], []
    rng = np.random.default_rng(int(seed) + int(worker_idx))
    discard_policy = _load_policy_from_state(discard_state)
    pegging_policy = _load_policy_from_state(pegging_state)
    best_dir = Path(best_dir_str)
    best_discard, best_pegging, _ = _load_value_models(best_dir)
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
                s0, s1 = play_game(candidate, opponent, seed=game_seed, fast_mode=True, copy_players=False)
                diff = float(s0 - s1)
            else:
                s0, s1 = play_game(opponent, candidate, seed=game_seed, fast_mode=True, copy_players=False)
                diff = float(s1 - s0)
        else:
            if i % 2 == 0:
                diff = _play_n_hands(candidate, opponent, game_seed, hands_per_game)
            else:
                diff = -_play_n_hands(opponent, candidate, game_seed, hands_per_game)
        disc_logs, peg_logs = candidate.pop_logs()
        for feats, idx, logp in disc_logs:
            discard_samples.append(PPODecision(feats, idx, logp, diff))
        for feats, idx, logp in peg_logs:
            pegging_samples.append(PPODecision(feats, idx, logp, diff))
    return discard_samples, pegging_samples


def _evaluate_worker(args_tuple: tuple) -> tuple[int, float, int]:
    (
        discard_state,
        pegging_state,
        discard_feature_set,
        pegging_feature_set,
        best_dir_str,
        games,
        seed,
        worker_idx,
    ) = args_tuple
    if games <= 0:
        return 0, 0.0, 0
    rng = np.random.default_rng(int(seed) + int(worker_idx))
    discard_policy = _load_policy_from_state(discard_state)
    pegging_policy = _load_policy_from_state(pegging_state)
    best_dir = Path(best_dir_str)
    best_discard, best_pegging, _ = _load_value_models(best_dir)
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
        deterministic=True,
    )
    _ensure_unique_names(candidate, opponent)
    wins = 0
    diff_sum = 0.0
    base_seed = int(seed) + int(worker_idx) * 100000
    for i in range(games):
        game_seed = base_seed + i
        if i % 2 == 0:
            s0, s1 = play_game(candidate, opponent, seed=game_seed, fast_mode=True, copy_players=False)
            diff = float(s0 - s1)
        else:
            s0, s1 = play_game(opponent, candidate, seed=game_seed, fast_mode=True, copy_players=False)
            diff = float(s1 - s0)
        if diff > 0:
            wins += 1
        diff_sum += diff
    return wins, diff_sum, games


def _collect_games(
    candidate: PPOPlayer,
    opponent: AIPlayer,
    games: int,
    seed: int,
) -> tuple[list[PPODecision], list[PPODecision]]:
    discard_samples: list[PPODecision] = []
    pegging_samples: list[PPODecision] = []
    for i in range(games):
        candidate.reset_logs()
        game_seed = int(seed) + i
        if i % 2 == 0:
            s0, s1 = play_game(candidate, opponent, seed=game_seed, fast_mode=True, copy_players=False)
            diff = float(s0 - s1)
        else:
            s0, s1 = play_game(opponent, candidate, seed=game_seed, fast_mode=True, copy_players=False)
            diff = float(s1 - s0)
        disc_logs, peg_logs = candidate.pop_logs()
        for feats, idx, logp in disc_logs:
            discard_samples.append(PPODecision(feats, idx, logp, diff))
        for feats, idx, logp in peg_logs:
            pegging_samples.append(PPODecision(feats, idx, logp, diff))
    return discard_samples, pegging_samples


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


def _evaluate(candidate: PPOPlayer, opponent: AIPlayer, games: int, seed: int) -> dict:
    candidate.deterministic = True
    wins = 0
    diffs = []
    for i in range(games):
        game_seed = int(seed) + i
        if i % 2 == 0:
            s0, s1 = play_game(candidate, opponent, seed=game_seed, fast_mode=True, copy_players=False)
            diff = float(s0 - s1)
        else:
            s0, s1 = play_game(opponent, candidate, seed=game_seed, fast_mode=True, copy_players=False)
            diff = float(s1 - s0)
        if diff > 0:
            wins += 1
        diffs.append(diff)
    candidate.deterministic = False
    winrate = wins / games if games else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"wins": wins, "games": games, "winrate": winrate, "avg_diff": avg_diff}


def _save_candidate(run_dir: Path, discard_policy: PolicyMLP, pegging_policy: PolicyMLP, meta: dict) -> None:
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    discard_path = run_dir / "discard_policy.pt"
    pegging_path = run_dir / "pegging_policy.pt"
    torch.save(
        {
            "state_dict": discard_policy.model.state_dict(),
            "input_dim": discard_policy.input_dim,
            "hidden_sizes": discard_policy.hidden_sizes,
        },
        discard_path,
    )
    torch.save(
        {
            "state_dict": pegging_policy.model.state_dict(),
            "input_dim": pegging_policy.input_dim,
            "hidden_sizes": pegging_policy.hidden_sizes,
        },
        pegging_path,
    )
    meta_path = run_dir / "model_meta.json"
    meta["discard_model_file"] = "discard_policy.pt"
    meta["pegging_model_file"] = "pegging_policy.pt"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default=str(Path(MODELS_DIR) / "ppo"))
    ap.add_argument("--model_version", type=str, required=True)
    ap.add_argument("--best_file", type=str, default="text/best_models_ppo.json")
    ap.add_argument("--best_run_id", type=str, default=None, help="Override best_file and use this run id.")
    ap.add_argument("--games_per_iteration", type=int, default=200)
    ap.add_argument("--hands_per_game", type=int, default=None, help="Use N hands per game for training only.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_games", type=int, default=200)
    ap.add_argument("--eval_workers", type=int, default=10)
    ap.add_argument("--loops", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ppo_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--accept_margin", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
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

    seed = int(args.seed) if args.seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))
    print(f"Using random seed: {seed}")
    best_file = Path(args.best_file)
    base_models_dir = Path(args.models_dir) / args.model_version
    base_models_dir.mkdir(parents=True, exist_ok=True)

    loop_total = "infinite" if loops < 0 else str(loops)
    loop_idx = 0
    while loops < 0 or loop_idx < loops:
        loop_idx += 1
        print(f"=== PPO loop {loop_idx}/{loop_total} ===")

        best_dir = _get_best_dir(best_file, args.model_version, args.best_run_id)
        if not best_dir.exists():
            raise SystemExit(f"Best model dir not found: {best_dir}")

        discard_policy, pegging_policy, meta = _build_policies(best_dir, seed + loop_idx * 10)
        discard_feature_set = meta.get("discard_feature_set", "full")
        pegging_feature_set = meta.get("pegging_feature_set", "full")

        discard_state = _policy_state(discard_policy)
        pegging_state = _policy_state(pegging_policy)
        collect_counts = _split_games(args.games_per_iteration, args.workers)
        collect_tasks = []
        for idx, count in enumerate(collect_counts):
            if count <= 0:
                continue
            collect_tasks.append(
                (
                    discard_state,
                    pegging_state,
                    discard_feature_set,
                    pegging_feature_set,
                    str(best_dir),
                    int(count),
                    seed + loop_idx * 1000,
                    idx,
                    args.hands_per_game,
                )
            )
        ctx = mp.get_context("spawn")
        discard_samples: list[PPODecision] = []
        pegging_samples: list[PPODecision] = []
        with ctx.Pool(processes=args.workers) as pool:
            for d_s, p_s in pool.imap_unordered(_collect_games_worker, collect_tasks):
                discard_samples.extend(d_s)
                pegging_samples.extend(p_s)
        _normalize_advantages(discard_samples)
        _normalize_advantages(pegging_samples)

        _ppo_update(
            discard_policy,
            discard_samples,
            lr=args.lr,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            clip=args.clip,
            entropy_coef=args.entropy_coef,
        )
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
        discard_state = _policy_state(discard_policy)
        pegging_state = _policy_state(pegging_policy)
        for idx, count in enumerate(eval_counts):
            if count <= 0:
                continue
            eval_tasks.append(
                (
                    discard_state,
                    pegging_state,
                    discard_feature_set,
                    pegging_feature_set,
                    str(best_dir),
                    int(count),
                    eval_seed,
                    idx,
                )
            )
        wins = 0
        total_games = 0
        diff_sum = 0.0
        with ctx.Pool(processes=args.eval_workers) as pool:
            for w, d_sum, g in pool.imap_unordered(_evaluate_worker, eval_tasks):
                wins += int(w)
                diff_sum += float(d_sum)
                total_games += int(g)
        if total_games <= 0:
            raise SystemExit("No eval games were played.")
        eval_result = {
            "wins": wins,
            "games": total_games,
            "winrate": wins / total_games,
            "avg_diff": diff_sum / total_games,
        }
        print(
            f"eval: wins={eval_result['wins']}/{eval_result['games']} "
            f"winrate={eval_result['winrate']:.3f} avg_diff={eval_result['avg_diff']:.2f}"
        )

        if eval_result["avg_diff"] >= float(args.accept_margin):
            run_id = _next_run_id(base_models_dir)
            run_dir = base_models_dir / run_id
            out_meta = {
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_version": args.model_version,
                "run_id": run_id,
                "source_best": str(best_dir),
                "model_type": "mlp",
                "discard_feature_set": discard_feature_set,
                "pegging_feature_set": pegging_feature_set,
                "discard_feature_dim": discard_policy.input_dim,
                "pegging_feature_dim": pegging_policy.input_dim,
                "mlp_hidden": list(discard_policy.hidden_sizes),
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
            print(f"Accepted candidate -> {run_dir}")
        else:
            print("Rejected candidate (did not improve).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Script summary: PPO self-play loop vs frozen best, full games, point-diff reward.
