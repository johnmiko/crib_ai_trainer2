"""RL-lite loop: play games, label decisions by final point diff, and train MLPs."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import multiprocessing as mp

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import MODELS_DIR
from crib_ai_trainer.players.neural_player import (
    MLPValueModel,
    AIPlayer,
    featurize_discard,
    featurize_pegging,
    get_discard_feature_indices,
    get_pegging_feature_indices,
)
from cribbage.players.hard_player import HardPlayer
from cribbage.utils import play_game
from cribbage.cribbagegame import CribbageGame


class RLLoggingPlayer(AIPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._discard_features: list[np.ndarray] = []
        self._pegging_features: list[np.ndarray] = []

    def reset_logs(self) -> None:
        self._discard_features = []
        self._pegging_features = []

    def get_logged_features(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self._discard_features, self._pegging_features

    def select_crib_cards(self, player_state, round_state):
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
        self._discard_features.append(feats)
        return discards

    def select_card_to_play(self, player_state, round_state):
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
        self._pegging_features.append(feats)
        return best


def _next_run_id(base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return "001"
    max_id = max(int(p.name) for p in run_dirs)
    return f"{max_id + 1:03d}"


def _load_best_models(best_dir: Path) -> tuple[MLPValueModel, MLPValueModel, dict]:
    meta_path = best_dir / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Expected model_meta.json at {meta_path} but it does not exist.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model_type") != "mlp":
        raise SystemExit(f"Expected model_type=mlp in {meta_path}, got {meta.get('model_type')}.")
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


def _save_models(model_dir: Path, discard_model: MLPValueModel, pegging_model: MLPValueModel, meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    discard_path = model_dir / meta["discard_model_file"]
    pegging_path = model_dir / meta["pegging_model_file"]
    discard_model.save_pt(str(discard_path))
    pegging_model.save_pt(str(pegging_path))
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))


def _evaluate(player: AIPlayer, opponent, games: int, seed: int) -> dict:
    wins = 0
    diffs = []
    for i in range(games):
        game_seed = int(seed) + i
        if i % 2 == 0:
            s0, s1 = play_game(player, opponent, seed=game_seed, fast_mode=True, copy_players=False)
            diff = s0 - s1
        else:
            s0, s1 = play_game(opponent, player, seed=game_seed, fast_mode=True, copy_players=False)
            diff = s1 - s0
        if diff > 0:
            wins += 1
        diffs.append(diff)
    winrate = wins / games if games else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"wins": wins, "games": games, "winrate": winrate, "avg_diff": avg_diff}


def _play_n_hands(p0, p1, seed: int | None, hands: int) -> float:
    game = CribbageGame(players=[p0, p1], seed=seed, copy_players=False, fast_mode=True)
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


def _collect_training_batch(args_tuple: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        best_dir_str,
        discard_feature_set,
        pegging_feature_set,
        games,
        seed,
        target_mode,
        winrate_weight,
        hands_per_game,
    ) = args_tuple
    best_dir = Path(best_dir_str)
    discard_model, pegging_model, _ = _load_best_models(best_dir)
    learner = RLLoggingPlayer(
        discard_model,
        pegging_model,
        name=f"learner:{best_dir.name}",
        discard_feature_set=discard_feature_set,
        pegging_feature_set=pegging_feature_set,
    )
    opponent = HardPlayer(name="hard")

    Xd_list: list[np.ndarray] = []
    Xp_list: list[np.ndarray] = []
    yd_list: list[float] = []
    yp_list: list[float] = []

    for game_idx in range(games):
        learner.reset_logs()
        game_seed = int(seed) + game_idx
        if hands_per_game is None:
            if game_idx % 2 == 0:
                s0, s1 = play_game(learner, opponent, seed=game_seed, fast_mode=True, copy_players=False)
                diff = float(s0 - s1)
            else:
                s0, s1 = play_game(opponent, learner, seed=game_seed, fast_mode=True, copy_players=False)
                diff = float(s1 - s0)
        else:
            if game_idx % 2 == 0:
                diff = _play_n_hands(learner, opponent, game_seed, hands_per_game)
            else:
                diff = -_play_n_hands(opponent, learner, game_seed, hands_per_game)
        target = _compute_target(diff, target_mode, winrate_weight)
        d_feats, p_feats = learner.get_logged_features()
        Xd_list.extend(d_feats)
        Xp_list.extend(p_feats)
        yd_list.extend([target] * len(d_feats))
        yp_list.extend([target] * len(p_feats))

    if not Xd_list or not Xp_list:
        raise SystemExit("No training data collected from games.")

    Xd = np.stack(Xd_list).astype(np.float32, copy=False)
    yd = np.array(yd_list, dtype=np.float32)
    Xp = np.stack(Xp_list).astype(np.float32, copy=False)
    yp = np.array(yp_list, dtype=np.float32)

    discard_idx = get_discard_feature_indices(discard_feature_set)
    pegging_idx = get_pegging_feature_indices(pegging_feature_set)
    Xd = Xd[:, discard_idx]
    Xp = Xp[:, pegging_idx]
    return Xd, yd, Xp, yp


def _evaluate_hands(player: AIPlayer, opponent, games: int, seed: int, hands: int) -> dict:
    wins = 0
    diffs = []
    for i in range(games):
        game_seed = int(seed) + i
        if i % 2 == 0:
            diff = _play_n_hands(player, opponent, game_seed, hands)
        else:
            diff = -_play_n_hands(opponent, player, game_seed, hands)
        if diff > 0:
            wins += 1
        diffs.append(diff)
    winrate = wins / games if games else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"wins": wins, "games": games, "winrate": winrate, "avg_diff": avg_diff}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default="selfplayv8")
    ap.add_argument("--best_file", type=str, default=None)
    ap.add_argument("--best_run_id", type=str, default=None)
    ap.add_argument("--games_per_iteration", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--loops", type=int, default=-1)
    ap.add_argument("--eval_games", type=int, default=None)
    ap.add_argument("--accept_margin", type=float, default=0.0)
    ap.add_argument("--hands_per_game", type=int, default=None, help="If set, play this many hands per game for training instead of full games.")
    ap.add_argument("--eval_hands_per_game", type=int, default=None, help="If set, play this many hands per eval game instead of full games.")
    ap.add_argument("--save_data_dir", type=str, default="il_datasets/selfplayv8/rl")
    ap.add_argument("--save_data", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_saved_shards", type=int, default=500)
    ap.add_argument("--train_from_saved_shards", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_train_shards", type=int, default=500)
    ap.add_argument(
        "--target_mode",
        type=str,
        default="point_diff",
        choices=["point_diff", "winrate", "mixed", "score_aware"],
    )
    ap.add_argument("--winrate_weight", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
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

    base_seed = args.seed
    if base_seed is None:
        base_seed = random.SystemRandom().randint(1, 2_000_000_000)
        print(f"Using random seed: {base_seed}")

    base_dir = Path(args.models_dir) / args.model_version
    default_best_file = Path(f"text/best_model_{args.model_version}.txt")
    best_file = Path(args.best_file) if args.best_file else default_best_file

    if args.best_run_id:
        best_dir = base_dir / args.best_run_id
    else:
        if not best_file.exists():
            raise SystemExit(f"Expected best_file at {best_file} but it does not exist.")
        best_dir = Path(best_file.read_text(encoding="utf-8").strip())
        if not best_dir.exists():
            raise SystemExit(f"Best model path from {best_file} does not exist: {best_dir}")

    loop_idx = 0
    max_loops = None if args.loops < 0 else args.loops

    while True:
        loop_idx += 1
        loop_label = "infinite" if max_loops is None else str(max_loops)
        print(f"=== RL loop {loop_idx}/{loop_label} ===")

        discard_model, pegging_model, meta = _load_best_models(best_dir)
        discard_feature_set = meta.get("discard_feature_set", "full")
        pegging_feature_set = meta.get("pegging_feature_set", "full")

        learner = RLLoggingPlayer(
            discard_model,
            pegging_model,
            name=f"learner:{best_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )
        opponent = HardPlayer(name="hard")

        Xd_list: list[np.ndarray] = []
        Xp_list: list[np.ndarray] = []
        yd_list: list[float] = []
        yp_list: list[float] = []

        if args.target_mode in {"mixed", "score_aware"}:
            if args.winrate_weight < 0.0 or args.winrate_weight > 1.0:
                raise SystemExit("--winrate_weight must be in [0, 1].")

        games_total = int(args.games_per_iteration)
        if args.workers == 1:
            Xd, yd, Xp, yp = _collect_training_batch(
                (
                    str(best_dir),
                    discard_feature_set,
                    pegging_feature_set,
                    games_total,
                    int(base_seed) + (loop_idx * 100_000),
                    args.target_mode,
                    args.winrate_weight,
                    args.hands_per_game,
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
                        str(best_dir),
                        discard_feature_set,
                        pegging_feature_set,
                        n_games,
                        seed_base + (i * 10_000),
                        args.target_mode,
                        args.winrate_weight,
                        args.hands_per_game,
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
            existing_ids = sorted(
                int(p.stem.split("_")[1])
                for p in save_dir.glob("discard_*.npz")
                if p.stem.split("_")[1].isdigit()
            )
            next_id = (existing_ids[-1] + 1) if existing_ids else 1
            shard_id = f"{next_id:03d}"
            discard_path = save_dir / f"discard_{shard_id}.npz"
            pegging_path = save_dir / f"pegging_{shard_id}.npz"
            np.savez(discard_path, X=Xd, y=yd)
            np.savez(pegging_path, X=Xp, y=yp)
            if args.max_saved_shards is not None:
                if args.max_saved_shards <= 0:
                    raise SystemExit("--max_saved_shards must be > 0 if provided.")
                discard_shards = sorted(save_dir.glob("discard_*.npz"))
                pegging_shards = sorted(save_dir.glob("pegging_*.npz"))
                excess = len(discard_shards) - args.max_saved_shards
                if excess > 0:
                    for shard in discard_shards[:excess]:
                        shard.unlink()
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
            discard_shards = sorted(train_dir.glob("discard_*.npz"))
            pegging_shards = sorted(train_dir.glob("pegging_*.npz"))
            if args.max_train_shards is not None:
                discard_shards = discard_shards[-args.max_train_shards :]
                pegging_shards = pegging_shards[-args.max_train_shards :]
            if not discard_shards or not pegging_shards:
                raise SystemExit(f"No saved shards found in {train_dir} for training.")
            Xd_list = []
            yd_list = []
            for path in discard_shards:
                data = np.load(path)
                Xd_list.append(data["X"])
                yd_list.append(data["y"])
            Xp_list = []
            yp_list = []
            for path in pegging_shards:
                data = np.load(path)
                Xp_list.append(data["X"])
                yp_list.append(data["y"])
            Xd = np.concatenate(Xd_list, axis=0).astype(np.float32, copy=False)
            yd = np.concatenate(yd_list, axis=0).astype(np.float32, copy=False)
            Xp = np.concatenate(Xp_list, axis=0).astype(np.float32, copy=False)
            yp = np.concatenate(yp_list, axis=0).astype(np.float32, copy=False)

        discard_model.fit_mse(Xd, yd, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, l2=args.l2, seed=base_seed)
        pegging_model.fit_mse(Xp, yp, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, l2=args.l2, seed=base_seed)

        candidate_dir = base_dir / "_candidate"
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)

        cand_meta = dict(meta)
        cand_meta["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
        cand_meta["trained_from"] = str(best_dir)
        cand_meta["training_games_used"] = int(args.games_per_iteration)
        cand_meta["discard_games_used"] = int(args.games_per_iteration)
        cand_meta["pegging_games_used"] = int(args.games_per_iteration)
        _save_models(candidate_dir, discard_model, pegging_model, cand_meta)

        best_player = AIPlayer(
            discard_model=MLPValueModel.load_pt(str(best_dir / cand_meta["discard_model_file"])),
            pegging_model=MLPValueModel.load_pt(str(best_dir / cand_meta["pegging_model_file"])),
            name=f"best:{best_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )
        cand_player = AIPlayer(
            discard_model=MLPValueModel.load_pt(str(candidate_dir / cand_meta["discard_model_file"])),
            pegging_model=MLPValueModel.load_pt(str(candidate_dir / cand_meta["pegging_model_file"])),
            name=f"cand:{candidate_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )

        eval_seed = int(base_seed) + (loop_idx * 1_000_000)
        eval_games = args.eval_games if args.eval_games is not None else args.games_per_iteration
        if args.eval_hands_per_game is None:
            best_eval = _evaluate(best_player, opponent, eval_games, eval_seed)
            cand_eval = _evaluate(cand_player, opponent, eval_games, eval_seed)
        else:
            best_eval = _evaluate_hands(best_player, opponent, eval_games, eval_seed, args.eval_hands_per_game)
            cand_eval = _evaluate_hands(cand_player, opponent, eval_games, eval_seed, args.eval_hands_per_game)
        print(
            "eval | "
            f"best {best_eval['wins']}/{best_eval['games']} wr={best_eval['winrate']:.3f} diff={best_eval['avg_diff']:.2f} | "
            f"cand {cand_eval['wins']}/{cand_eval['games']} wr={cand_eval['winrate']:.3f} diff={cand_eval['avg_diff']:.2f}"
        )

        if cand_eval["avg_diff"] >= (best_eval["avg_diff"] + args.accept_margin):
            new_run_id = _next_run_id(base_dir)
            new_best_dir = base_dir / new_run_id
            print(f"Accepted candidate (higher avg_diff). Saving as {new_run_id}.")
            shutil.copytree(candidate_dir, new_best_dir)
            best_dir = new_best_dir
            best_file.write_text(str(best_dir))
        else:
            print("Rejected candidate (did not improve).")

        if max_loops is not None and loop_idx >= max_loops:
            break

# Script summary: play games vs hard, label decisions by final point diff, and update the best MLP if it improves.
