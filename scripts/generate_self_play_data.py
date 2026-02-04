"""Generate self-play datasets using a neural model policy.

This generates *on-policy* data (model plays both sides) while labels still
come from crib EV (discard) + pegging rollout targets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from itertools import combinations
from typing import List, Tuple

import numpy as np

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import (
    TRAINING_DATA_DIR,
    DEFAULT_DATASET_VERSION,
    DEFAULT_DATASET_RUN_ID,
    DEFAULT_STRATEGY,
    DEFAULT_PEGGING_FEATURE_SET,
    DEFAULT_GAMES_PER_LOOP,
    DEFAULT_SEED,
    DEFAULT_USE_RANDOM_SEED,
    DEFAULT_CRIB_EV_MODE,
    DEFAULT_CRIB_MC_SAMPLES,
    DEFAULT_PEGGING_LABEL_MODE,
    DEFAULT_PEGGING_ROLLOUTS,
    DEFAULT_PEGGING_EV_MODE,
    DEFAULT_PEGGING_EV_ROLLOUTS,
    DEFAULT_WIN_PROB_MODE,
    DEFAULT_WIN_PROB_ROLLOUTS,
    DEFAULT_WIN_PROB_MIN_SCORE,
)

from cribbage.cribbagegame import score_hand, score_play
from cribbage.players.rule_based_player import get_full_deck
from cribbage.playingcards import Card
from cribbage.strategies.hand_strategies import process_dealt_hand_only_exact
from cribbage.strategies.crib_strategies import calc_crib_min_only_given_6_cards
from cribbage.database import normalize_hand_to_str

from crib_ai_trainer.players.neural_player import (
    featurize_discard,
    featurize_pegging,
    get_discard_feature_indices,
    get_pegging_feature_indices,
    select_discard_with_model_with_scores,
    estimate_pegging_ev_mc_for_discard,
    regression_pegging_strategy,
    LinearValueModel,
    LinearDiscardClassifier,
    MLPValueModel,
)

from scripts.generate_il_data import (
    LoggedRegPegRegDiscardData,
    save_data,
    get_cumulative_game_count,
    estimate_crib_ev_mc_from_remaining,
    estimate_pegging_rollout_value,
    estimate_pegging_rollout_value_2ply,
    _resolve_output_dir,
)

from cribbage import cribbagegame

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_latest_run_id(version_dir: Path) -> str | None:
    if not version_dir.exists():
        return None
    run_dirs = [p for p in version_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return None
    run_id = max(int(p.name) for p in run_dirs)
    return f"{run_id:03d}"


def _resolve_models_dir(base_models_dir: str, model_version: str | None, run_id: str | None) -> str:
    base = Path(base_models_dir)
    if (base / "model_meta.json").exists():
        return str(base)
    version_dir = base / model_version if model_version else base
    if run_id:
        return str(version_dir / run_id)
    latest = _find_latest_run_id(version_dir)
    if latest:
        return str(version_dir / latest)
    return str(version_dir)


def _load_models(models_dir: str):
    meta_path = os.path.join(models_dir, "model_meta.json")
    model_type = "linear"
    discard_feature_set = "full"
    pegging_feature_set = "full"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_type = meta.get("model_type", model_type)
        discard_feature_set = meta.get("discard_feature_set", discard_feature_set)
        pegging_feature_set = meta.get("pegging_feature_set", pegging_feature_set)

    if model_type == "mlp":
        discard_model = MLPValueModel.load_pt(os.path.join(models_dir, "discard_mlp.pt"))
        pegging_model = MLPValueModel.load_pt(os.path.join(models_dir, "pegging_mlp.pt"))
    else:
        discard_model = LinearValueModel.load_npz(os.path.join(models_dir, "discard_linear.npz"))
        pegging_model = LinearValueModel.load_npz(os.path.join(models_dir, "pegging_linear.npz"))

    return discard_model, pegging_model, discard_feature_set, pegging_feature_set


class LoggingNeuralPlayer:
    """Use a neural policy, but log training data with crib/rollout labels."""

    def __init__(
        self,
        name: str,
        log: LoggedRegPegRegDiscardData,
        discard_model,
        pegging_model,
        discard_feature_set: str,
        pegging_feature_set: str,
        crib_ev_mode: str,
        crib_mc_samples: int,
        pegging_label_mode: str,
        pegging_rollouts: int,
        seed: int = 0,
    ):
        self.name = name
        self._log = log
        self._rng = random.Random(seed)
        self._rng_np = np.random.default_rng(seed)
        self._full_deck = get_full_deck()

        self.discard_model = discard_model
        self.pegging_model = pegging_model
        self.discard_feature_set = discard_feature_set
        self.discard_feature_indices = get_discard_feature_indices(discard_feature_set)
        self.pegging_feature_indices = get_pegging_feature_indices(pegging_feature_set)
        self.pegging_feature_set = pegging_feature_set

        self._crib_ev_mode = crib_ev_mode
        self._crib_mc_samples = crib_mc_samples
        self._pegging_label_mode = pegging_label_mode
        self._pegging_rollouts = pegging_rollouts

    def select_crib_cards(self, player_state, round_state) -> Tuple[Card, Card]:
        hand = player_state.hand
        dealer_is_self = player_state.is_dealer
        your_score = player_state.score
        opponent_score = player_state.opponent_score

        # Compute exact hand scores for all kept combos.
        full_deck = get_full_deck()
        hand_score_cache = {}
        hand_results = process_dealt_hand_only_exact([hand, full_deck, hand_score_cache])
        df_hand = np.array(hand_results, dtype=object)

        if self._crib_ev_mode == "min":
            crib_results = calc_crib_min_only_given_6_cards(hand)
            df_crib = np.array(crib_results, dtype=object)
            crib_map = { (r[0], r[1]): float(r[3]) for r in df_crib }  # avg_crib_score
        else:
            hand_set = set(hand)
            remaining = [c for c in self._full_deck if c not in hand_set]
            crib_map = {}
            for kept in combinations(hand, 4):
                kept_list = list(kept)
                discards_list_temp = [c for c in hand if c not in kept_list]
                hand_key = normalize_hand_to_str(kept_list)
                crib_key = normalize_hand_to_str(discards_list_temp)
                crib_ev = estimate_crib_ev_mc_from_remaining(
                    discards_list_temp,
                    remaining,
                    self._rng,
                    n_samples=self._crib_mc_samples,
                )
                crib_map[(hand_key, crib_key)] = float(crib_ev)

        # Build target labels per option
        for kept in combinations(hand, 4):
            kept_list = list(kept)
            discards_list = [c for c in hand if c not in kept_list]
            hand_key = normalize_hand_to_str(kept_list)
            crib_key = normalize_hand_to_str(discards_list)

            # avg_hand_score in df_hand: [hand_key, min, max, avg]
            avg_hand = None
            for row in df_hand:
                if row[0] == hand_key:
                    avg_hand = float(row[3])
                    break
            if avg_hand is None:
                continue

            avg_crib = crib_map.get((hand_key, crib_key), 0.0)
            y = avg_hand + (avg_crib if dealer_is_self else -avg_crib)

            pegging_ev = None
            if self.discard_feature_set in {"engineered_no_scores_pev", "full_pev"}:
                pegging_ev = estimate_pegging_ev_mc_for_discard(
                    hand,
                    kept_list,
                    discards_list,
                    dealer_is_self,
                    self._rng_np,
                    n_rollouts=8,
                )
            x = featurize_discard(
                kept_list,
                discards_list,
                dealer_is_self,
                your_score,
                opponent_score,
                pegging_ev=pegging_ev,
            )
            self._log.X_discard.append(x)
            self._log.y_discard.append(float(y))

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
        count = round_state.count
        known_cards = player_state.known_cards
        all_played = round_state.all_played_cards

        playable = [c for c in hand if c.get_value() + count <= 31]
        if not playable:
            return None

        # Log all options with rollout labels
        for c in playable:
            if self._pegging_label_mode == "rollout2":
                y = estimate_pegging_rollout_value_2ply(
                    hand,
                    table,
                    count,
                    c,
                    known_cards,
                    all_played,
                    self._rng,
                    n_rollouts=self._pegging_rollouts,
                )
            elif self._pegging_label_mode == "rollout1":
                y = estimate_pegging_rollout_value(
                    hand,
                    table,
                    count,
                    c,
                    known_cards,
                    all_played,
                    self._rng,
                    n_rollouts=self._pegging_rollouts,
                )
            else:
                y = float(score_play(table + [c])[0])

            x = featurize_pegging(
                hand,
                table,
                count,
                c,
                known_cards=known_cards,
                opponent_known_hand=player_state.opponent_known_hand,
                all_played_cards=all_played,
                player_score=player_state.score,
                opponent_score=player_state.opponent_score,
                feature_set=self.pegging_feature_set,
            )
            self._log.X_pegging.append(x)
            self._log.y_pegging.append(float(y))

        return regression_pegging_strategy(
            self.pegging_model,
            hand,
            table,
            round_state.crib,
            count,
            known_cards=known_cards,
            opponent_known_hand=player_state.opponent_known_hand,
            all_played_cards=all_played,
            player_score=player_state.score,
            opponent_score=player_state.opponent_score,
            feature_set=self.pegging_feature_set,
            feature_indices=self.pegging_feature_indices,
        )


def play_one_game(players) -> None:
    game = cribbagegame.CribbageGame(players=players, copy_players=False)
    game.start()


def generate_self_play_data(
    games: int,
    out_dir: str,
    models_dir: str,
    opponent_models_dir: str | None,
    seed: int | None,
    strategy: str,
    pegging_feature_set: str,
    crib_ev_mode: str,
    crib_mc_samples: int,
    pegging_label_mode: str,
    pegging_rollouts: int,
    pegging_ev_mode: str = DEFAULT_PEGGING_EV_MODE,
    pegging_ev_rollouts: int = DEFAULT_PEGGING_EV_ROLLOUTS,
    win_prob_mode: str = DEFAULT_WIN_PROB_MODE,
    win_prob_rollouts: int = DEFAULT_WIN_PROB_ROLLOUTS,
    win_prob_min_score: int = DEFAULT_WIN_PROB_MIN_SCORE,
) -> int:
    if seed is None:
        seed = random.randint(1, 2**31 - 1)
        logger.info("No seed provided, using random seed=%s", seed)

    discard_model, pegging_model, discard_feature_set, pegging_feature_set_model = _load_models(models_dir)
    if opponent_models_dir:
        opp_discard, opp_pegging, opp_discard_fs, opp_pegging_fs = _load_models(opponent_models_dir)
    else:
        opp_discard, opp_pegging, opp_discard_fs, opp_pegging_fs = discard_model, pegging_model, discard_feature_set, pegging_feature_set_model

    log = LoggedRegPegRegDiscardData()
    p1 = LoggingNeuralPlayer(
        "selfplay1",
        log,
        discard_model,
        pegging_model,
        discard_feature_set,
        pegging_feature_set_model,
        crib_ev_mode,
        crib_mc_samples,
        pegging_label_mode,
        pegging_rollouts,
        seed=seed,
    )
    p2 = LoggingNeuralPlayer(
        "selfplay2",
        log,
        opp_discard,
        opp_pegging,
        opp_discard_fs,
        opp_pegging_fs,
        crib_ev_mode,
        crib_mc_samples,
        pegging_label_mode,
        pegging_rollouts,
        seed=seed + 1,
    )

    cumulative_games = get_cumulative_game_count(out_dir)
    save_interval = 2000
    games_since_save = 0

    for i in range(games):
        if i % 100 == 0:
            logger.info("Playing games %d - %d/%d", i, min(i + 100, games), games)
        players = [p1, p2] if (i % 2 == 0) else [p2, p1]
        play_one_game(players)
        games_since_save += 1

        if games_since_save >= save_interval:
            cumulative_games += games_since_save
            save_data(
                log,
                out_dir,
                cumulative_games,
                strategy,
                seed,
                pegging_feature_set,
                crib_ev_mode,
                crib_mc_samples,
                pegging_label_mode,
                pegging_rollouts,
                pegging_ev_mode,
                pegging_ev_rollouts,
                win_prob_mode,
                win_prob_rollouts,
                win_prob_min_score,
            )
            log.X_discard.clear()
            log.y_discard.clear()
            log.X_pegging.clear()
            log.y_pegging.clear()
            games_since_save = 0

    if games_since_save > 0:
        cumulative_games += games_since_save
        save_data(
            log,
            out_dir,
            cumulative_games,
            strategy,
            seed,
            pegging_feature_set,
            crib_ev_mode,
            crib_mc_samples,
            pegging_label_mode,
            pegging_rollouts,
            pegging_ev_mode,
            pegging_ev_rollouts,
            win_prob_mode,
            win_prob_rollouts,
            win_prob_min_score,
        )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_LOOP)
    ap.add_argument("--out_dir", type=str, default=TRAINING_DATA_DIR)
    ap.add_argument("--dataset_version", type=str, default=DEFAULT_DATASET_VERSION)
    ap.add_argument("--run_id", type=str, default=DEFAULT_DATASET_RUN_ID or None)
    ap.add_argument("--models_dir", type=str, required=True)
    ap.add_argument("--model_version", type=str, default=None)
    ap.add_argument("--model_run_id", type=str, default=None)
    ap.add_argument("--opponent_models_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    ap.add_argument("--pegging_feature_set", type=str, default=DEFAULT_PEGGING_FEATURE_SET, choices=["basic", "full"])
    ap.add_argument("--crib_ev_mode", type=str, default=DEFAULT_CRIB_EV_MODE, choices=["min", "mc"])
    ap.add_argument("--crib_mc_samples", type=int, default=DEFAULT_CRIB_MC_SAMPLES)
    ap.add_argument("--pegging_label_mode", type=str, default=DEFAULT_PEGGING_LABEL_MODE, choices=["immediate", "rollout1", "rollout2"])
    ap.add_argument("--pegging_rollouts", type=int, default=DEFAULT_PEGGING_ROLLOUTS)
    ap.add_argument("--pegging_ev_mode", type=str, default=DEFAULT_PEGGING_EV_MODE, choices=["off", "rollout"])
    ap.add_argument("--pegging_ev_rollouts", type=int, default=DEFAULT_PEGGING_EV_ROLLOUTS)
    ap.add_argument("--win_prob_mode", type=str, default=DEFAULT_WIN_PROB_MODE, choices=["off", "rollout"])
    ap.add_argument("--win_prob_rollouts", type=int, default=DEFAULT_WIN_PROB_ROLLOUTS)
    ap.add_argument("--win_prob_min_score", type=int, default=DEFAULT_WIN_PROB_MIN_SCORE)
    args = ap.parse_args()

    resolved_out_dir = _resolve_output_dir(args.out_dir, args.dataset_version, args.run_id, new_run=False)
    models_dir = _resolve_models_dir(args.models_dir, args.model_version, args.model_run_id)
    generate_self_play_data(
        args.games,
        resolved_out_dir,
        models_dir,
        args.opponent_models_dir,
        args.seed,
        args.strategy,
        args.pegging_feature_set,
        args.crib_ev_mode,
        args.crib_mc_samples,
        args.pegging_label_mode,
        args.pegging_rollouts,
        args.pegging_ev_mode,
        args.pegging_ev_rollouts,
        args.win_prob_mode,
        args.win_prob_rollouts,
        args.win_prob_min_score,
    )
