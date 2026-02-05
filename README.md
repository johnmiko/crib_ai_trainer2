# crib_ai_trainer2

- Trying to solve the game of cribbage as an AI problem
- The purpose of this repo is to build a cribbage AI that is very difficult to beat, so that I can play it myself on my phone

## Strategies / Models
- Random Player - Makes discard and pegging plays randomly
- Beginner Player
    - Discard strategy - calculate hand points of the 4 cards kept, and the 2 cards put in the crib
    - Pegging strategy - play highest scoring pegging play, if points are equal or 0, play highest card

## Current Commands
- Generate IL data (default): `python .\scripts\generate_il_data.py`
- Generate regression data forever: `python .\scripts\generate_il_data.py --games -1 --out_dir "il_datasets/medium_discard_regression" --strategy regression`
- Train ranking model (current config): `python .\scripts\train_models.py --data_dir "il_datasets/medium_discard_ranking" --models_dir "models/ranking" --discard_loss ranking --epochs 6 --eval_samples 2048 --max_shards 6 --rank_pairs_per_hand 20`
- Benchmark (current config): `python scripts/benchmark_2_players.py --players NeuralDiscardOnlyPlayer,beginner --games 200 --models_dir "models/ranking" --data_dir "il_datasets/medium_discard_ranking" --max_shards 6 --fallback_player beginner`
- Full loop (forever): `python .\scripts\do_everything2.py --games 2000 --loops -1 --strategy ranking --training_dir "il_datasets/medium_discard_ranking" --data_dir "il_datasets/medium_discard_ranking" --models_dir "models/ranking" --players NeuralRegressionPlayer,beginner --benchmark_games 200`

# Results / History
2026-02-03
Results (1000 hands)

Margin mean: 1.299
Margin median: 0.655
Margin 25/75 pct: 0.117 / 1.902
Margin ≤ 0.1: 24.3%
Margin ≤ 0.5: 45.5%
Margin ≤ 1.0: 55.8%
Interpretation
Label margins are small (many near ties), which makes classification brittle.

This confirms the same story at scale: nearly half the hands have a top‑1 vs top‑2 gap under 0.5 points, which makes strict classification a pretty brittle target even when the teacher is good.

If you want, the next logical step is to train a discard model that predicts a score for each option (regression ranking) but keep the same medium‑player scoring (no Monte Carlo). That would let the model treat near‑ties as near‑ties.

2025-12-27
Trained a simple multinomial logistic regression model using 2 "beginner" players to create the training data. Results
- linear regression vs beginner after 12000.0 training games wins=45/500 winrate=0.090 (95% CI 0.068 - 0.118) avg point diff -17.88
- linear regression vs random after 12000.0 training games wins=330ish/500 (forgot to record exact statistics)
We can see some learning occurred as the winrate against a player making random decisions was about 66%, but against the beginner player winrate was only 9%. Independently replacing the pegging strategy and the discard strategy the results were
- regression discard only vs beginner after 12000.0 training games wins=72/500 winrate=0.144 (95% CI 0.116 - 0.177) avg point diff -14.14
- regression pegging only vs beginner after 12000.0 training games wins=184/500 winrate=0.368 (95% CI 0.327 - 0.411) avg point diff -2.02
This shows that the AI was able to somewhat learn how to peg but was unable to learn how to choose the correct cards to discard using a simple regression strategy trained from the beginner player. After creating a new discard dataset using monte carlo simulation the results did not improve. When looking at the details of a specific hand and calculating the actual exact statistical values of hand [ad, 2h, 4d, 6h, 7h, ks], the best cards to put in your own crib is [7h, 6h] giving average hand + crib value of 9.15. Model predicts [7h, 6h] value as 8.6, but it predicted that [ks, ad] discard is 9.6, where the actual value is 7.61. This causes the AI to lose 1-2 points per hand which is a lot in crib. Next steps are to switch to a classification strategy for discarding instead of regression
