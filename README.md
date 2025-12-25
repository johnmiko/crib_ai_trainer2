# crib_ai_trainer2

A flexible, test-driven cribbage AI training framework.

Features:
- Full cribbage rules: dealing, discard to crib, pegging, scoring.
- Multiple AIs: Random, Rule-Based, IS-MCTS (belief sampling), Simple Perceptron.
- Training: self-play, benchmarking against old versions, rankings report.
- Feature encoding: 52-bit multi-hot for hand/seen/starter, 32 one-hot count, K×52 history.
- Export: model.npz (arrays `w`, `b`, float32) for use in external repos.

## Quickstart

1. Create and activate venv, install deps

```powershell
Push-Location "C:\Users\johnm\ccode\crib_ai_trainer2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run tests

```powershell
pytest -q
```

3. Train all models (Ctrl-C to stop)

```powershell
python scripts/do_everything.py
```

4. Rank models

```powershell
python -m crib_ai_trainer2.ranking
```

Logs are emitted via `logging.getLogger(__name__)`.
