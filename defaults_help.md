# Defaults Help

This project reads default settings from `.env` via `crib_ai_trainer/constants.py`.
If you run a script with no flags, it will use the values in `.env`.

Edit this file to change the defaults:
`C:\Users\johnm\ccode\crib_ai_trainer2\.env`

## Key Defaults (from .env)
These affect all four scripts:
- `TRAINING_DATA_DIR` and `MODELS_DIR`: base folders for data and models
- `DATASET_VERSION`, `DATASET_RUN_ID`: where new IL data is stored
- `MODEL_VERSION`, `MODEL_RUN_ID`: where new models are stored
- `STRATEGY` and `DISCARD_LOSS`: classification vs regression vs ranking
- `PEGGING_FEATURE_SET`: feature set used when **generating datasets** (`basic` or `full`)
- `DISCARD_FEATURE_SET`: feature subset used when **training/benchmarking models**
- `PEGGING_MODEL_FEATURE_SET`: pegging feature subset used when **training/benchmarking models**
- `GAMES_PER_LOOP`, `LOOPS`: how many games per loop and loop count
- `EPOCHS`, `LR`, `L2`, `BATCH_SIZE`: training hyperparameters
- `BENCHMARK_GAMES`, `BENCHMARK_PLAYERS`, `FALLBACK_PLAYER`: benchmark defaults
- `EVAL_SAMPLES`: quick eval sample size
- `MAX_SHARDS`: limit shard count (0 = all)
- `RANK_PAIRS_PER_HAND`: ranking loss pairs per hand
- `SEED`, `USE_RANDOM_SEED`: RNG behavior
- `MODEL_TYPE`: `linear` or `mlp`
- `MLP_HIDDEN`: comma-separated hidden sizes for the MLP (e.g., `128,64`)
- `CRIB_EV_MODE`: `min` or `mc` for discard label crib EV
- `CRIB_MC_SAMPLES`: number of MC samples for crib EV (when `CRIB_EV_MODE=mc`)
- `PEGGING_LABEL_MODE`: `immediate`, `rollout1`, or `rollout2` for pegging labels
- `PEGGING_ROLLOUTS`: number of rollouts for pegging labels (when `PEGGING_LABEL_MODE=rollout1`)

## Script Quick Usage (uses defaults)
Run any of these without flags and they will use `.env`:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_il_data.py
```

```powershell
.\.venv\Scripts\python.exe .\scripts\train_linear_models.py
```

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_2_players.py
```

Benchmark results are also logged in structured form to:
`C:\Users\johnm\ccode\crib_ai_trainer2\experiments.jsonl`

```powershell
.\.venv\Scripts\python.exe .\scripts\do_everything2.py
```

If you want to override just one setting temporarily, pass it as a flag (e.g., `--games 2000`).
