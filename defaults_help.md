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
- `PEGGING_FEATURE_SET`: `basic` or `full`
- `GAMES_PER_LOOP`, `LOOPS`: how many games per loop and loop count
- `EPOCHS`, `LR`, `L2`, `BATCH_SIZE`: training hyperparameters
- `BENCHMARK_GAMES`, `BENCHMARK_PLAYERS`, `FALLBACK_PLAYER`: benchmark defaults
- `EVAL_SAMPLES`: quick eval sample size
- `MAX_SHARDS`: limit shard count (0 = all)
- `RANK_PAIRS_PER_HAND`: ranking loss pairs per hand
- `SEED`, `USE_RANDOM_SEED`: RNG behavior

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

```powershell
.\.venv\Scripts\python.exe .\scripts\do_everything2.py
```

If you want to override just one setting temporarily, pass it as a flag (e.g., `--games 2000`).
