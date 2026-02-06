"""Evolutionary loop: perturb best MLP, evaluate vs fixed opponents, accept if better."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, ".")

from crib_ai_trainer.constants import MODELS_DIR, DEFAULT_MODEL_VERSION
from crib_ai_trainer.players.neural_player import MLPValueModel, AIPlayer
from cribbage.players.beginner_player import BeginnerPlayer
from cribbage.players.medium_player import MediumPlayer
from cribbage.players.hard_player import HardPlayer
from cribbage.utils import play_multiple_games


def _load_meta(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Expected model_meta.json at {path} but it does not exist.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mlp_models(model_dir: Path) -> tuple[MLPValueModel, MLPValueModel, dict]:
    meta_path = model_dir / "model_meta.json"
    meta = _load_meta(meta_path)
    if meta.get("model_type") != "mlp":
        raise SystemExit(f"Expected model_type=mlp in {meta_path}, got {meta.get('model_type')}.")
    discard_file = meta.get("discard_model_file")
    pegging_file = meta.get("pegging_model_file")
    if not discard_file or not pegging_file:
        raise SystemExit(f"model_meta.json missing discard_model_file/pegging_model_file at {meta_path}.")
    discard_path = model_dir / discard_file
    pegging_path = model_dir / pegging_file
    if not discard_path.exists() or not pegging_path.exists():
        raise SystemExit(f"Missing model files in {model_dir}.")
    discard_model = MLPValueModel.load_pt(str(discard_path))
    pegging_model = MLPValueModel.load_pt(str(pegging_path))
    return discard_model, pegging_model, meta


def _save_mlp_models(model_dir: Path, discard_model: MLPValueModel, pegging_model: MLPValueModel, meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    discard_path = model_dir / meta["discard_model_file"]
    pegging_path = model_dir / meta["pegging_model_file"]
    discard_model.save_pt(str(discard_path))
    pegging_model.save_pt(str(pegging_path))
    meta_path = model_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))


def _perturb_mlp(model: MLPValueModel, std: float, seed: int) -> MLPValueModel:
    import torch

    torch.manual_seed(int(seed))
    new_model = MLPValueModel(model.input_dim, model.hidden_sizes)
    new_model.model.load_state_dict(model.model.state_dict())
    for param in new_model.model.parameters():
        if param.dtype.is_floating_point:
            noise = torch.randn_like(param) * float(std)
            param.data.add_(noise)
    new_model.model.eval()
    return new_model


def _build_opponent(name: str):
    if name == "beginner":
        return BeginnerPlayer(name=name)
    if name == "medium":
        return MediumPlayer(name=name)
    if name == "hard":
        return HardPlayer(name=name)
    raise SystemExit(f"Unknown opponent: {name}")


def _next_run_id(base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        return "001"
    max_id = max(int(p.name) for p in run_dirs)
    return f"{max_id + 1:03d}"


def _evaluate(model: AIPlayer, opponent_name: str, games: int, seed: int) -> dict:
    opponent = _build_opponent(opponent_name)
    result = play_multiple_games(
        games,
        p0=model,
        p1=opponent,
        seed=seed,
        fast_mode=True,
        copy_players=False,
    )
    return {
        "wins": int(result["wins"]),
        "winrate": float(result["winrate"]),
        "games": int(games),
        "opponent": opponent_name,
    }


def _format_eval(label: str, evals: list[dict]) -> str:
    parts = []
    total_wins = 0
    total_games = 0
    for e in evals:
        total_wins += e["wins"]
        total_games += e["games"]
        parts.append(f"{e['opponent']}: {e['wins']}/{e['games']} ({e['winrate']:.3f})")
    total_wr = (float(total_wins) / float(total_games)) if total_games else 0.0
    return f"{label} total {total_wins}/{total_games} ({total_wr:.3f}) | " + ", ".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default=MODELS_DIR)
    ap.add_argument("--model_version", type=str, default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--best_run_id", type=str, required=True)
    ap.add_argument("--candidate_dir_name", type=str, default="_candidate")
    ap.add_argument("--best_file", type=str, default=None)
    ap.add_argument("--games_per_opponent", type=int, default=10)
    ap.add_argument("--opponents", type=str, default="beginner,medium,hard")
    ap.add_argument("--perturb_std", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loops", type=int, default=-1)
    args = ap.parse_args()

    if args.games_per_opponent <= 0:
        raise SystemExit("--games_per_opponent must be > 0.")
    if args.perturb_std <= 0:
        raise SystemExit("--perturb_std must be > 0.")
    if args.loops == 0:
        raise SystemExit("--loops must be != 0.")
    if args.loops < 0:
        total_loops = None
    else:
        total_loops = args.loops

    base_seed = args.seed
    if base_seed is None:
        base_seed = random.SystemRandom().randint(1, 2_000_000_000)
        print(f"Using random seed: {base_seed}")

    base_dir = Path(args.models_dir) / args.model_version
    best_dir = base_dir / args.best_run_id
    candidate_dir = base_dir / args.candidate_dir_name
    best_file = Path(args.best_file) if args.best_file else None

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    if not opponents:
        raise SystemExit("--opponents must include at least one opponent.")

    loop_idx = 0
    while True:
        loop_idx += 1
        loop_label = str(total_loops) if total_loops is not None else "infinite"
        print(f"=== Evo loop {loop_idx}/{loop_label} ===")
        discard_model, pegging_model, meta = _load_mlp_models(best_dir)
        discard_feature_set = meta.get("discard_feature_set", "full")
        pegging_feature_set = meta.get("pegging_feature_set", "full")

        discard_mut = _perturb_mlp(discard_model, args.perturb_std, base_seed + loop_idx * 10 + 1)
        pegging_mut = _perturb_mlp(pegging_model, args.perturb_std, base_seed + loop_idx * 10 + 2)

        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        cand_meta = dict(meta)
        cand_meta["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
        cand_meta["mutation_std"] = float(args.perturb_std)
        cand_meta["mutation_seed"] = int(base_seed)
        cand_meta["mutated_from"] = str(best_dir)

        _save_mlp_models(candidate_dir, discard_mut, pegging_mut, cand_meta)

        best_player = AIPlayer(
            discard_model,
            pegging_model,
            name=f"best:{best_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )
        cand_player = AIPlayer(
            discard_mut,
            pegging_mut,
            name=f"cand:{candidate_dir.name}",
            discard_feature_set=discard_feature_set,
            pegging_feature_set=pegging_feature_set,
        )

        best_evals = []
        cand_evals = []
        for idx, opp in enumerate(opponents):
            opp_seed = int(base_seed) + (loop_idx * 1000) + (idx * 100)
            best_evals.append(_evaluate(best_player, opp, args.games_per_opponent, opp_seed))
            cand_evals.append(_evaluate(cand_player, opp, args.games_per_opponent, opp_seed))

        best_summary = _format_eval("best", best_evals)
        cand_summary = _format_eval("cand", cand_evals)
        print(best_summary)
        print(cand_summary)

        best_total = sum(e["wins"] for e in best_evals)
        cand_total = sum(e["wins"] for e in cand_evals)
        if cand_total > best_total:
            new_run_id = _next_run_id(base_dir)
            new_best_dir = base_dir / new_run_id
            print(f"Accepted candidate (higher total wins). Saving as {new_run_id}.")
            if new_best_dir.exists():
                shutil.rmtree(new_best_dir)
            shutil.copytree(candidate_dir, new_best_dir)
            best_dir = new_best_dir
            if best_file is not None:
                best_file.write_text(str(best_dir))
        else:
            print("Rejected candidate (did not improve).")
        if total_loops is not None and loop_idx >= total_loops:
            break

# Script summary: perturb best MLP, evaluate vs fixed opponents, and accept if winrate improves.
