"""RL-lite loop: play games, label decisions by final point diff, and train MLPs."""
from __future__ import annotations

import argparse
import json
import re
import random
import shutil
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import multiprocessing as mp

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import MODELS_DIR
from crib_ai_trainer.players.neural_player import (
    MLPValueModel,
    PeggingRNNValueModel,
    PeggingTransformerValueModel,
    AIPlayer,
    featurize_discard,
    featurize_pegging,
    get_discard_feature_indices,
    get_pegging_feature_indices,
    NeuralPegOnlyPlayer,
    NeuralDiscardOnlyPlayer,
)
from scripts.loop_utils import read_queue_models, resolve_model_dir
from cribbage.players.hard_player import HardPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.utils import play_game
from cribbage.cribbagegame import CribbageGame
from cribbage.training_game import TrainingGame


class RLLoggingPlayer(AIPlayer):
    def __init__(self, *args, discard_fallback=None, pegging_fallback=None, training_mode: str = "full", **kwargs):
        super().__init__(*args, **kwargs)
        self._discard_features: list[np.ndarray] = []
        self._pegging_features: list[np.ndarray] = []
        self.discard_fallback = discard_fallback
        self.pegging_fallback = pegging_fallback
        self.training_mode = training_mode

    def reset_logs(self) -> None:
        self._discard_features = []
        self._pegging_features = []

    def get_logged_features(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self._discard_features, self._pegging_features

    def select_crib_cards(self, player_state, round_state):
        if self.training_mode == "pegging_only" and self.discard_fallback is not None:
            return self.discard_fallback.select_crib_cards(player_state, round_state)
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = getattr(player_state, "opponent_score", None)
        discards = self.select_crib_cards_regressor(
            hand, dealer_is_self, your_score, opponent_score
        )
        kept = [c for c in hand if c not in discards]
        feats = featurize_discard(
            kept,
            discards,
            dealer_is_self,
            player_score=your_score,
            opponent_score=opponent_score,
            pegging_ev=None,
        )
        if self.training_mode != "pegging_only":
            self._discard_features.append(feats)
        return discards

    def select_card_to_play(self, player_state, round_state):
        if self.training_mode == "discard_only" and self.pegging_fallback is not None:
            return self.pegging_fallback.select_card_to_play(player_state, round_state)
        hand = player_state.hand
        table = round_state.table_cards
        count = round_state.count
        best = super().select_card_to_play(player_state, round_state)
        if best is None:
            return best
        feats = featurize_pegging(
            hand,
            table,
            count,
            best,
            known_cards=player_state.known_cards,
            opponent_known_hand=player_state.opponent_known_hand,
            all_played_cards=round_state.all_played_cards,
            player_score=player_state.score,
            opponent_score=getattr(player_state, "opponent_score", None),
            feature_set=self.pegging_feature_set,
            unseen_value_counts=getattr(round_state, "unseen_value_counts", None),
            unseen_count=getattr(round_state, "unseen_count", None),
        )
        if self.training_mode != "discard_only":
            self._pegging_features.append(feats)
        return best


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


def _next_run_id(base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return "001"
    max_id = max(int(p.name) for p in run_dirs)
    return f"{max_id + 1:03d}"


def _load_best_models(best_dir: Path) -> tuple[object | None, object | None, dict]:
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
        elif pegging_model_type == "gru":
            pegging_model = PeggingRNNValueModel.load_pt(str(pegging_path))
        elif pegging_model_type == "transformer":
            pegging_model = PeggingTransformerValueModel.load_pt(str(pegging_path))
        else:
            raise SystemExit(f"Unsupported pegging_model_type={pegging_model_type} in {meta_path}.")
    return discard_model, pegging_model, meta


def _merge_meta(discard_meta: dict, pegging_meta: dict | None) -> dict:
    meta = dict(discard_meta)
    if not pegging_meta:
        return meta
    for key in (
        "pegging_feature_set",
        "pegging_feature_dim",
        "pegging_model_type",
        "pegging_model_file",
        "pegging_mlp_hidden",
        "pegging_rnn_hidden",
        "pegging_transformer_d_model",
        "pegging_transformer_heads",
        "pegging_transformer_layers",
        "pegging_transformer_ff_dim",
        "pegging_transformer_dropout",
    ):
        if key in pegging_meta:
            meta[key] = pegging_meta[key]
    return meta


def _infer_version_from_path(path: Path, default: str) -> str:
    for part in path.parts:
        m = re.fullmatch(r"v(\d+)", part.lower())
        if m:
            return f"v{m.group(1)}"
        m = re.search(r"discard_v(\d+)", part.lower())
        if m:
            return f"v{m.group(1)}"
    return default




def _save_models(model_dir: Path, discard_model, pegging_model, meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    discard_file = meta.get("discard_model_file")
    pegging_file = meta.get("pegging_model_file")
    if discard_file and discard_model is not None:
        discard_path = model_dir / discard_file
        discard_model.save_pt(str(discard_path))
    if pegging_file and pegging_model is not None:
        pegging_path = model_dir / pegging_file
        pegging_model.save_pt(str(pegging_path))
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))


def _evaluate(player: AIPlayer, opponent, games: int, seed: int, training_mode: str) -> dict:
    _ensure_unique_names(player, opponent)
    wins = 0
    diffs = []
    for i in range(games):
        game_seed = int(seed) + i
        if i % 2 == 0:
            s0, s1 = play_game(
                player,
                opponent,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=_resolve_play_mode(training_mode),
            )
            diff = s0 - s1
        else:
            s0, s1 = play_game(
                opponent,
                player,
                seed=game_seed,
                fast_mode=True,
                copy_players=False,
                training_mode=_resolve_play_mode(training_mode),
            )
            diff = s1 - s0
        if diff > 0:
            wins += 1
        diffs.append(diff)
    winrate = wins / games if games else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"wins": wins, "games": games, "winrate": winrate, "avg_diff": avg_diff}


def _ensure_unique_names(p0, p1):
    if getattr(p0, "name", None) == getattr(p1, "name", None):
        p0.name = f"{p0.name}_0"
        p1.name = f"{p1.name}_1"
    return p0, p1


def _resolve_play_mode(training_mode: str) -> str:
    if training_mode == "discard_only":
        return "full"
    return training_mode


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


def _compute_target(diff: float, target_mode: str, winrate_weight: float) -> float:
    if diff > 0:
        win = 1.0
    elif diff < 0:
        win = 0.0
    else:
        win = 0.5
    if target_mode == "point_diff":
        return diff
    if target_mode == "winrate":
        return win
    if target_mode == "mixed":
        return (winrate_weight * win) + ((1.0 - winrate_weight) * diff)
    # score_aware
    if diff < 0:
        return win
    return (winrate_weight * win) + ((1.0 - winrate_weight) * diff)


def _build_opponent(name: str):
    if name == "hard":
        return HardPlayer(name="hard")
    if name == "medium":
        return MediumPlayer(name="medium")
    if name == "beginner":
        return BeginnerPlayer(name="beginner")
    raise SystemExit(f"Unsupported opponent: {name}")


def _collect_training_batch(args_tuple: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        source_dir_str,
        discard_feature_set,
        pegging_feature_set,
        games,
        seed,
        target_mode,
        winrate_weight,
        hands_per_game,
        baseline_mode,
        opponent_name,
        training_mode,
        pegging_fallback_name,
    ) = args_tuple
    source_dir = Path(source_dir_str)
    discard_model, pegging_model, _ = _load_best_models(source_dir)
    discard_enabled = training_mode != "pegging_only"
    pegging_enabled = training_mode != "discard_only"

    if pegging_enabled and pegging_model is None:
        raise SystemExit(f"Expected pegging model in {source_dir} for training_mode={training_mode}.")
    if discard_enabled and discard_model is None:
        raise SystemExit(f"Expected discard model in {source_dir} for training_mode={training_mode}.")
    pegging_fallback = None
    if training_mode == "discard_only":
        if pegging_fallback_name:
            pegging_fallback = _build_opponent(pegging_fallback_name)
        else:
            if pegging_model is None:
                raise SystemExit(f"Expected pegging model in {source_dir} for discard_only pegging.")
            pegging_fallback = _make_pegging_player(pegging_model, pegging_feature_set)

    learner = RLLoggingPlayer(
        discard_model,
        pegging_model,
        name=f"learner:{source_dir.name}",
        discard_feature_set=discard_feature_set,
        pegging_feature_set=pegging_feature_set,
        discard_fallback=_build_opponent(opponent_name) if training_mode == "pegging_only" else None,
        pegging_fallback=pegging_fallback,
        training_mode=training_mode,
    )
    opponent = _build_opponent(opponent_name)
    baseline_opponent = HardPlayer(name="hard")

    Xd_list: list[np.ndarray] = []
    Xp_list: list[np.ndarray] = []
    yd_list: list[float] = []
    yp_list: list[float] = []

    for game_idx in range(games):
        learner.reset_logs()
        game_seed = int(seed) + game_idx
        if hands_per_game is None:
            if game_idx % 2 == 0:
                s0, s1 = play_game(
                    learner,
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
                    learner,
                    seed=game_seed,
                    fast_mode=True,
                    copy_players=False,
                    training_mode=_resolve_play_mode(training_mode),
                )
                diff = float(s1 - s0)
        else:
            if game_idx % 2 == 0:
                diff = _play_n_hands(learner, opponent, game_seed, hands_per_game, training_mode)
            else:
                diff = -_play_n_hands(opponent, learner, game_seed, hands_per_game, training_mode)
        if baseline_mode == "hard_vs_hard":
            if hands_per_game is None:
                b0_player = baseline_opponent
                b1_player = HardPlayer(name="hard")
                _ensure_unique_names(b0_player, b1_player)
                b0, b1 = play_game(
                    b0_player,
                    b1_player,
                    seed=game_seed,
                    fast_mode=True,
                    copy_players=False,
                    training_mode=_resolve_play_mode(training_mode),
                )
                baseline_diff = float(b0 - b1)
            else:
                baseline_diff = _play_n_hands(
                    baseline_opponent,
                    HardPlayer(name="hard"),
                    game_seed,
                    hands_per_game,
                    training_mode,
                )
            diff = diff - baseline_diff
        target = _compute_target(diff, target_mode, winrate_weight)
        d_feats, p_feats = learner.get_logged_features()
        if discard_enabled:
            Xd_list.extend(d_feats)
            yd_list.extend([target] * len(d_feats))
        if pegging_enabled:
            Xp_list.extend(p_feats)
            yp_list.extend([target] * len(p_feats))

    if discard_enabled and not Xd_list:
        raise SystemExit("No discard training data collected from games.")
    if pegging_enabled and not Xp_list:
        raise SystemExit("No pegging training data collected from games.")

    if discard_enabled:
        Xd = np.stack(Xd_list).astype(np.float32, copy=False)
        yd = np.array(yd_list, dtype=np.float32)
    else:
        Xd = np.empty((0, 0), dtype=np.float32)
        yd = np.empty((0,), dtype=np.float32)
    if pegging_enabled:
        Xp = np.stack(Xp_list).astype(np.float32, copy=False)
        yp = np.array(yp_list, dtype=np.float32)
    else:
        Xp = np.empty((0, 0), dtype=np.float32)
        yp = np.empty((0,), dtype=np.float32)

    if discard_enabled:
        discard_idx = get_discard_feature_indices(discard_feature_set)
        Xd = Xd[:, discard_idx]
    if pegging_enabled:
        pegging_idx = get_pegging_feature_indices(pegging_feature_set)
        Xp = Xp[:, pegging_idx]
    return Xd, yd, Xp, yp


def _load_best_map(best_file: Path) -> dict:
    if not best_file.exists():
        raise SystemExit(f"Best model file not found: {best_file}")
    raw = best_file.read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit(f"Best model file is empty: {best_file}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit(f"Best model file must be a JSON object: {best_file}")
    return data


def _get_best_dir(best_file: Path, model_version: str) -> Path:
    data = _load_best_map(best_file)
    entry = data.get(model_version)
    if not isinstance(entry, dict) or not entry.get("path"):
        raise SystemExit(f"Best model entry missing path for {model_version} in {best_file}")
    return Path(str(entry["path"]).strip())


def _write_best_dir(best_file: Path, model_version: str, best_dir: Path) -> None:
    data = _load_best_map(best_file)
    entry = data.get(model_version)
    if not isinstance(entry, dict):
        entry = {}
    entry["path"] = str(best_dir)
    data[model_version] = entry
    best_file.write_text(json.dumps(data, indent=2), encoding="utf-8")




def _evaluate_hands(player: AIPlayer, opponent, games: int, seed: int, hands: int, training_mode: str) -> dict:
    _ensure_unique_names(player, opponent)
    wins = 0
    diffs = []
    for i in range(games):
        game_seed = int(seed) + i
        if i % 2 == 0:
            diff = _play_n_hands(player, opponent, game_seed, hands, training_mode)
        else:
            diff = -_play_n_hands(opponent, player, game_seed, hands, training_mode)
        if diff > 0:
            wins += 1
        diffs.append(diff)
    winrate = wins / games if games else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"wins": wins, "games": games, "winrate": winrate, "avg_diff": avg_diff}


def _run_single(args: argparse.Namespace) -> int:
    base_seed = args.seed
    if base_seed is None:
        base_seed = random.SystemRandom().randint(1, 2_000_000_000)
        print(f"Using random seed: {base_seed}")

    if args.output_root:
        base_dir = Path(args.output_root)
    elif args.discard_model_dir:
        discard_root = Path(args.discard_model_dir)
        if discard_root.name.lower() == "rl":
            base_dir = discard_root
        else:
            base_dir = discard_root / "rl"
    else:
        base_dir = Path(args.models_dir) / args.model_version
    base_dir.mkdir(parents=True, exist_ok=True)
    default_best_file = Path("text/best_models_selfplay.json")
    best_file = Path(args.best_file) if args.best_file else default_best_file
    if args.save_data_dir is None:
        if args.discard_model_dir:
            discard_root = resolve_model_dir(args.discard_model_dir)
            label = discard_root.name
            parent_name = discard_root.parent.name if discard_root.parent else ""
            if parent_name and parent_name != "rl":
                label = parent_name
            version = _infer_version_from_path(discard_root, args.model_version)
            args.save_data_dir = f"datasets/{version}/{label}/rl"
        else:
            args.save_data_dir = f"datasets/{args.model_version}/rl"

    if args.start_model_dir:
        best_dir = Path(args.start_model_dir)
        if not best_dir.exists():
            raise SystemExit(f"--start_model_dir not found: {best_dir}")
        if not best_file.exists():
            best_file.parent.mkdir(parents=True, exist_ok=True)
            best_file.write_text(json.dumps({args.model_version: {"path": str(best_dir)}}, indent=2), encoding="utf-8")
        else:
            _write_best_dir(best_file, args.model_version, best_dir)
    elif args.best_run_id:
        best_dir = base_dir / args.best_run_id
    elif args.discard_model_dir:
        best_dir = resolve_model_dir(args.discard_model_dir)
    elif args.pegging_model_dir:
        best_dir = resolve_model_dir(args.pegging_model_dir)
    else:
        try:
            best_dir = _get_best_dir(best_file, args.model_version)
        except SystemExit:
            best_dir = None
        if best_dir is None or not best_dir.exists():
            raise SystemExit(f"Best model path not found. Provide --start_model_dir or --discard_model_dir, or update {best_file}.")

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

    print(f"Current best model: {best_dir}")
    loop_idx = 0
    max_loops = None if args.loops < 0 else args.loops

    header_printed = False
    accept_history: list[bool] = []
    winrate_weight = float(args.winrate_weight)
    while True:
        loop_idx += 1
        loop_label = "infinite" if max_loops is None else str(max_loops)

        discard_model, _, discard_meta = _load_best_models(discard_source_dir)
        _, pegging_model, pegging_meta = _load_best_models(pegging_source_dir)
        meta = _merge_meta(discard_meta, pegging_meta)
        discard_feature_set = meta.get("discard_feature_set", "full")
        pegging_feature_set = meta.get("pegging_feature_set", "full")
        discard_enabled = run_training_mode != "pegging_only"
        pegging_enabled = run_training_mode != "discard_only"

        if pegging_enabled and pegging_model is None:
            raise SystemExit(f"Expected pegging model in {pegging_source_dir} for training_mode={run_training_mode}.")
        if discard_enabled and discard_model is None:
            raise SystemExit(f"Expected discard model in {discard_source_dir} for training_mode={run_training_mode}.")

        candidate_dir = base_dir / "_candidate"
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        _save_models(candidate_dir, discard_model, pegging_model, meta)

        opponent = _build_opponent(args.opponent)
        pegging_fallback = None
        if run_training_mode == "discard_only":
            if args.discard_pegging_fallback:
                pegging_fallback = _build_opponent(args.discard_pegging_fallback)
            else:
                if pegging_model is None:
                    raise SystemExit(f"Expected pegging model in {pegging_source_dir} for discard_only pegging.")
                pegging_fallback = _make_pegging_player(pegging_model, pegging_feature_set)
        learner = RLLoggingPlayer(
            discard_model,
            pegging_model,
            name=f"learner:{candidate_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
            discard_fallback=opponent if run_training_mode == "pegging_only" else None,
            pegging_fallback=pegging_fallback,
            training_mode=run_training_mode,
        )

        Xd_list: list[np.ndarray] = []
        Xp_list: list[np.ndarray] = []
        yd_list: list[float] = []
        yp_list: list[float] = []

        if args.target_mode in {"mixed", "score_aware"}:
            if winrate_weight < 0.0 or winrate_weight > 1.0:
                raise SystemExit("--winrate_weight must be in [0, 1].")

        games_total = int(args.games_per_iteration)
        if args.workers == 1:
            Xd, yd, Xp, yp = _collect_training_batch(
                (
                    str(candidate_dir),
                    discard_feature_set,
                    pegging_feature_set,
                    games_total,
                    int(base_seed) + (loop_idx * 100_000),
                    args.target_mode,
                    winrate_weight,
                    args.hands_per_game,
                    args.baseline_mode,
                    args.opponent,
                    run_training_mode,
                    args.discard_pegging_fallback,
                )
            )
        else:
            chunk = games_total // args.workers
            remainder = games_total % args.workers
            tasks = []
            seed_base = int(base_seed) + (loop_idx * 100_000)
            for i in range(args.workers):
                n_games = chunk + (1 if i < remainder else 0)
                if n_games == 0:
                    continue
                tasks.append(
                    (
                        str(candidate_dir),
                        discard_feature_set,
                        pegging_feature_set,
                        n_games,
                        seed_base + (i * 10_000),
                        args.target_mode,
                        winrate_weight,
                        args.hands_per_game,
                        args.baseline_mode,
                        args.opponent,
                        run_training_mode,
                        args.discard_pegging_fallback,
                    )
                )
            if not tasks:
                raise SystemExit("No training tasks created for workers.")
            ctx = mp.get_context("spawn")
            results = []
            with ctx.Pool(processes=min(args.workers, len(tasks))) as pool:
                for res in pool.imap_unordered(_collect_training_batch, tasks):
                    results.append(res)
            Xd = np.concatenate([r[0] for r in results], axis=0).astype(np.float32, copy=False)
            yd = np.concatenate([r[1] for r in results], axis=0).astype(np.float32, copy=False)
            Xp = np.concatenate([r[2] for r in results], axis=0).astype(np.float32, copy=False)
            yp = np.concatenate([r[3] for r in results], axis=0).astype(np.float32, copy=False)

        if args.save_data:
            save_dir = Path(args.save_data_dir)
            if args.target_mode == "winrate":
                save_dir = save_dir / "winrate"
            save_dir.mkdir(parents=True, exist_ok=True)
            pattern = "pegging_*.npz" if pegging_enabled else "discard_*.npz"
            existing_ids = sorted(
                int(p.stem.split("_")[1])
                for p in save_dir.glob(pattern)
                if p.stem.split("_")[1].isdigit()
            )
            next_id = (existing_ids[-1] + 1) if existing_ids else 1
            shard_id = f"{next_id:03d}"
            if discard_enabled:
                discard_path = save_dir / f"discard_{shard_id}.npz"
                np.savez(discard_path, X=Xd, y=yd)
            if pegging_enabled:
                pegging_path = save_dir / f"pegging_{shard_id}.npz"
                np.savez(pegging_path, X=Xp, y=yp)
            if args.max_saved_shards is not None:
                if args.max_saved_shards <= 0:
                    raise SystemExit("--max_saved_shards must be > 0 if provided.")
                if discard_enabled:
                    discard_shards = sorted(save_dir.glob("discard_*.npz"))
                    excess = len(discard_shards) - args.max_saved_shards
                    if excess > 0:
                        for shard in discard_shards[:excess]:
                            shard.unlink()
                if pegging_enabled:
                    pegging_shards = sorted(save_dir.glob("pegging_*.npz"))
                    excess = len(pegging_shards) - args.max_saved_shards
                    if excess > 0:
                        for shard in pegging_shards[:excess]:
                            shard.unlink()

        if args.train_from_saved_shards:
            if args.max_train_shards is not None and args.max_train_shards <= 0:
                raise SystemExit("--max_train_shards must be > 0 if provided.")
            train_dir = Path(args.save_data_dir)
            if args.target_mode == "winrate":
                train_dir = train_dir / "winrate"
            if discard_enabled:
                discard_shards = sorted(train_dir.glob("discard_*.npz"))
                if args.max_train_shards is not None:
                    discard_shards = discard_shards[-args.max_train_shards :]
                if not discard_shards:
                    raise SystemExit(f"No discard shards found in {train_dir} for training.")
                Xd_list = []
                yd_list = []
                for path in discard_shards:
                    data = np.load(path)
                    Xd_list.append(data["X"])
                    yd_list.append(data["y"])
                Xd = np.concatenate(Xd_list, axis=0).astype(np.float32, copy=False)
                yd = np.concatenate(yd_list, axis=0).astype(np.float32, copy=False)
            if pegging_enabled:
                pegging_shards = sorted(train_dir.glob("pegging_*.npz"))
                if args.max_train_shards is not None:
                    pegging_shards = pegging_shards[-args.max_train_shards :]
                if not pegging_shards:
                    raise SystemExit(f"No pegging shards found in {train_dir} for training.")
                Xp_list = []
                yp_list = []
                for path in pegging_shards:
                    data = np.load(path)
                    Xp_list.append(data["X"])
                    yp_list.append(data["y"])
                Xp = np.concatenate(Xp_list, axis=0).astype(np.float32, copy=False)
                yp = np.concatenate(yp_list, axis=0).astype(np.float32, copy=False)

        if discard_enabled and discard_model is not None:
            discard_model.fit_mse(Xd, yd, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, l2=args.l2, seed=base_seed)
        if pegging_enabled and pegging_model is not None:
            pegging_model.fit_mse(Xp, yp, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, l2=args.l2, seed=base_seed)

        candidate_dir = base_dir / "_candidate"
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)

        cand_meta = dict(meta)
        cand_meta["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
        cand_meta["trained_from"] = str(discard_source_dir)
        cand_meta["training_games_used"] = int(args.games_per_iteration)
        cand_meta["discard_games_used"] = int(args.games_per_iteration) if discard_enabled else 0
        cand_meta["pegging_games_used"] = int(args.games_per_iteration) if pegging_enabled else 0
        _save_models(candidate_dir, discard_model, pegging_model, cand_meta)

        def _load_model_for_eval(model_dir: Path, model_file: str | None, model_kind: str):
            if not model_file:
                return None
            path = model_dir / model_file
            if model_kind == "discard":
                return MLPValueModel.load_pt(str(path))
            pegging_model_type = cand_meta.get("pegging_model_type") or cand_meta.get("model_type") or "mlp"
            if pegging_model_type == "mlp":
                return MLPValueModel.load_pt(str(path))
            if pegging_model_type == "gru":
                return PeggingRNNValueModel.load_pt(str(path))
            if pegging_model_type == "transformer":
                return PeggingTransformerValueModel.load_pt(str(path))
            raise SystemExit(f"Unsupported pegging_model_type={pegging_model_type} in {model_dir}.")

        best_discard = _load_model_for_eval(discard_source_dir, cand_meta.get("discard_model_file"), "discard")
        best_pegging = _load_model_for_eval(pegging_source_dir, cand_meta.get("pegging_model_file"), "pegging")
        cand_discard = _load_model_for_eval(candidate_dir, cand_meta.get("discard_model_file"), "discard")
        cand_pegging = best_pegging

        if run_training_mode == "pegging_only":
            best_player = NeuralPegOnlyPlayer(
                pegging_model=best_pegging,
                discard_fallback=opponent,
                name=f"best:{best_dir.name}",
                pegging_feature_set=pegging_feature_set,
            )
            cand_player = NeuralPegOnlyPlayer(
                pegging_model=cand_pegging,
                discard_fallback=opponent,
                name=f"cand:{candidate_dir.name}",
                pegging_feature_set=pegging_feature_set,
            )
        elif run_training_mode == "discard_only":
            if args.discard_pegging_fallback:
                peg_fallback = _build_opponent(args.discard_pegging_fallback)
            else:
                if best_pegging is None:
                    raise SystemExit(f"Expected pegging model in {pegging_source_dir} for discard_only evaluation.")
                peg_fallback = _make_pegging_player(best_pegging, pegging_feature_set)
            best_player = NeuralDiscardOnlyPlayer(
                discard_model=best_discard,
                pegging_fallback=peg_fallback,
                name=f"best:{discard_source_dir.name}",
                discard_feature_set=discard_feature_set,
            )
            cand_player = NeuralDiscardOnlyPlayer(
                discard_model=cand_discard,
                pegging_fallback=peg_fallback,
                name=f"cand:{candidate_dir.name}",
                discard_feature_set=discard_feature_set,
            )
        else:
            best_player = AIPlayer(
                discard_model=best_discard,
                pegging_model=best_pegging,
                name=f"best:{best_dir.name}",
                discard_feature_set=discard_feature_set,
                pegging_feature_set=pegging_feature_set,
            )
            cand_player = AIPlayer(
                discard_model=cand_discard,
                pegging_model=cand_pegging,
                name=f"cand:{candidate_dir.name}",
                discard_feature_set=discard_feature_set,
                pegging_feature_set=pegging_feature_set,
            )

        eval_seed = int(base_seed) + (loop_idx * 1_000_000)
        eval_games = args.eval_games if args.eval_games is not None else args.games_per_iteration
        if args.eval_hands_per_game is None:
            best_eval = _evaluate(best_player, opponent, eval_games, eval_seed, run_training_mode)
            cand_eval = _evaluate(cand_player, opponent, eval_games, eval_seed, run_training_mode)
            if args.baseline_mode == "hard_vs_hard":
                baseline_eval = _evaluate(
                    HardPlayer(name="hard"),
                    HardPlayer(name="hard"),
                    eval_games,
                    eval_seed,
                    run_training_mode,
                )
            else:
                baseline_eval = None
        else:
            best_eval = _evaluate_hands(best_player, opponent, eval_games, eval_seed, args.eval_hands_per_game, run_training_mode)
            cand_eval = _evaluate_hands(cand_player, opponent, eval_games, eval_seed, args.eval_hands_per_game, run_training_mode)
            if args.baseline_mode == "hard_vs_hard":
                baseline_eval = _evaluate_hands(
                    HardPlayer(name="hard"),
                    HardPlayer(name="hard"),
                    eval_games,
                    eval_seed,
                    args.eval_hands_per_game,
                    run_training_mode,
                )
            else:
                baseline_eval = None
        best_adj = best_eval["avg_diff"]
        cand_adj = cand_eval["avg_diff"]
        if baseline_eval is not None:
            best_adj = best_adj - baseline_eval["avg_diff"]
            cand_adj = cand_adj - baseline_eval["avg_diff"]
        if args.accept_metric == "winrate":
            best_metric = best_eval["winrate"]
            cand_metric = cand_eval["winrate"]
        else:
            best_metric = best_adj
            cand_metric = cand_adj
        def _fmt(val: str, width: int, align: str = "left") -> str:
            if align == "right":
                return str(val).rjust(width)
            return str(val).ljust(width)

        cols = [
            ("loop", 12, "left"),
            ("best_w/g", 12, "right"),
            ("best_wr", 7, "right"),
            ("best_diff", 9, "right"),
        ]
        if baseline_eval is not None:
            cols += [
                ("base_diff", 9, "right"),
                ("best_adj", 9, "right"),
            ]
        cols += [
            ("cand_w/g", 12, "right"),
            ("cand_wr", 7, "right"),
            ("cand_diff", 9, "right"),
        ]
        if baseline_eval is not None:
            cols.append(("cand_adj", 9, "right"))
        cols.append(("result", 14, "left"))

        if not header_printed:
            header = " | ".join(_fmt(name, width, "left") for name, width, _ in cols)
            print(header)
            header_printed = True
        values = [
            f"{loop_idx}/{loop_label}",
            f"{best_eval['wins']}/{best_eval['games']}",
            f"{best_eval['winrate']:.3f}",
            f"{best_eval['avg_diff']:.2f}",
        ]
        if baseline_eval is not None:
            values += [
                f"{baseline_eval['avg_diff']:.2f}",
                f"{best_adj:.2f}",
            ]
        values += [
            f"{cand_eval['wins']}/{cand_eval['games']}",
            f"{cand_eval['winrate']:.3f}",
            f"{cand_eval['avg_diff']:.2f}",
        ]
        if baseline_eval is not None:
            values.append(f"{cand_adj:.2f}")

        accepted = cand_metric > (best_metric + args.accept_margin)
        result_text = "rejected"
        if accepted:
            new_run_id = _next_run_id(base_dir)
            new_best_dir = base_dir / new_run_id
            result_text = f"accepted -> {new_run_id}"
            shutil.copytree(candidate_dir, new_best_dir)
            best_dir = new_best_dir
            _write_best_dir(best_file, args.model_version, best_dir)
        values.append(result_text)
        eval_line = " | ".join(
            _fmt(val, width, align)
            for (val, (name, width, align)) in zip(values, cols)
        )
        print(eval_line)

        accept_history.append(accepted)
        if args.plateau_window is not None and args.plateau_window > 0 and len(accept_history) >= args.plateau_window:
            window = accept_history[-args.plateau_window :]
            if not any(window):
                if winrate_weight < 1.0 - 1e-9:
                    new_weight = min(1.0, winrate_weight + float(args.plateau_winrate_step))
                    print(
                        f"Plateau detected (0/{args.plateau_window} accepted). "
                        f"Increasing winrate_weight {winrate_weight:.2f} -> {new_weight:.2f}."
                    )
                    winrate_weight = new_weight
                    accept_history = []
                else:
                    print(
                        f"Stopping on plateau at winrate_weight=1.0: 0/{args.plateau_window} accepted in last window."
                    )
                    break

        if max_loops is not None and loop_idx >= max_loops:
            break

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR, help="Base models directory.")
    ap.add_argument("--model_version", type=str, default="selfplayv8", help="Model version label for saving and lookup.")
    ap.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="If set, save RL runs under this directory (run_id subfolders).",
    )
    ap.add_argument(
        "--start_model_dir",
        type=str,
        default=None,
        help="If set, use this model directory as the starting/best model.",
    )
    ap.add_argument("--queue_models", type=str, default=None, help="Comma-separated list of model dirs to run.")
    ap.add_argument("--queue_file", type=str, default=None, help="File containing one model dir per line.")
    ap.add_argument("--best_file", type=str, default=None, help="Path to best-model JSON mapping.")
    ap.add_argument("--best_run_id", type=str, default=None, help="Explicit run id to start from.")
    ap.add_argument("--discard_model_dir", type=str, default=None, help="Use this directory for the discard model.")
    ap.add_argument("--pegging_model_dir", type=str, default=None, help="Use this directory for the pegging model.")
    ap.add_argument("--games_per_iteration", type=int, default=200, help="Training games per loop.")
    ap.add_argument("--workers", type=int, default=10, help="Worker processes for data collection.")
    ap.add_argument("--loops", type=int, default=-1, help="Number of loops (-1 for infinite).")
    ap.add_argument("--eval_games", type=int, default=None, help="Eval games per loop (defaults to games_per_iteration).")
    ap.add_argument("--accept_margin", type=float, default=0.0, help="Required improvement margin for acceptance.")
    ap.add_argument(
        "--accept_metric",
        type=str,
        default="avg_diff",
        choices=["avg_diff", "winrate"],
        help="Metric used to accept/reject candidates.",
    )
    ap.add_argument(
        "--plateau_window",
        type=int,
        default=5,
        help="If set, use plateau windows to increase winrate_weight and eventually stop.",
    )
    ap.add_argument(
        "--plateau_winrate_step",
        type=float,
        default=0.1,
        help="Increase winrate_weight by this amount after a plateau window.",
    )
    ap.add_argument("--hands_per_game", type=int, default=None, help="If set, play this many hands per game for training instead of full games.")
    ap.add_argument("--eval_hands_per_game", type=int, default=None, help="If set, play this many hands per eval game instead of full games.")
    ap.add_argument(
        "--training_mode",
        type=str,
        default="full",
        choices=["full", "discard_only", "pegging_only"],
        help="Play full games or isolate discard/pegging outcomes.",
    )
    ap.add_argument(
        "--discard_pegging_fallback",
        type=str,
        default=None,
        choices=["hard", "medium", "beginner"],
        help="If set, use this opponent's pegging during discard-only training instead of the pegging model.",
    )
    ap.add_argument("--baseline_mode", type=str, default="none", choices=["none", "hard_vs_hard"], help="Baseline adjustment mode.")
    ap.add_argument("--opponent", type=str, default="hard", choices=["hard", "medium", "beginner"], help="Opponent policy.")
    ap.add_argument("--save_data_dir", type=str, default=None, help="Directory for saved training shards.")
    ap.add_argument("--save_data", action=argparse.BooleanOptionalAction, default=True, help="Save training shards.")
    ap.add_argument("--max_saved_shards", type=int, default=500, help="Cap on saved shard count.")
    ap.add_argument("--train_from_saved_shards", action=argparse.BooleanOptionalAction, default=True, help="Train from saved shards.")
    ap.add_argument("--max_train_shards", type=int, default=5, help="Max shards used for training.")
    ap.add_argument(
        "--target_mode",
        type=str,
        default="point_diff",
        choices=["point_diff", "winrate", "mixed", "score_aware"],
    )
    ap.add_argument("--winrate_weight", type=float, default=0.7, help="Weight for winrate in mixed/score-aware targets.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    ap.add_argument("--epochs", type=int, default=3, help="Training epochs per loop.")
    ap.add_argument("--batch_size", type=int, default=1024, help="Training batch size.")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 regularization.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = ap.parse_args()

    if args.games_per_iteration <= 0:
        raise SystemExit("--games_per_iteration must be > 0.")
    if args.hands_per_game is not None and args.hands_per_game <= 0:
        raise SystemExit("--hands_per_game must be > 0 if provided.")
    if args.eval_hands_per_game is not None and args.eval_hands_per_game <= 0:
        raise SystemExit("--eval_hands_per_game must be > 0 if provided.")
    if args.eval_games is not None and args.eval_games <= 0:
        raise SystemExit("--eval_games must be > 0 if provided.")
    if args.accept_margin < 0:
        raise SystemExit("--accept_margin must be >= 0.")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if args.loops == 0:
        raise SystemExit("--loops must be != 0.")
    queue = read_queue_models(args.queue_models, args.queue_file)
    if queue:
        for model_dir in queue:
            run_args = argparse.Namespace(**vars(args))
            run_args.start_model_dir = model_dir
            if run_args.output_root is None:
                run_args.output_root = str(Path(model_dir) / "rl")
            if run_args.model_version == "selfplayv8":
                run_args.model_version = Path(model_dir).name
            _run_single(run_args)
    else:
        _run_single(args)

# Script summary: play games vs the chosen opponent, label decisions by final point diff, and update the best MLP if it improves.
