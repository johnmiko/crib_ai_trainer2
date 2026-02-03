# Experiments Log

This file is a running log of experiments for the cribbage AI trainer.
Keep entries short, reproducible, and focused on results.

---

## Template

**Date:** YYYY-MM-DD  
**Goal:**  
**Dataset:**  
**Model / Loss:**  
**Training Cmd:**  
**Benchmark Cmd:**  
**Results (avg point diff primary):**  
**Notes / Next Steps:**  

---

## 2026-02-03

**Goal:** Understand discard difficulty and compare discard/pegging isolation.  
**Dataset:** `il_datasets/medium_discard_classification` (5 shards, ~106k games)  
**Model / Loss:** Linear (classification)  
**Training Cmd:** `python .\scripts\train_linear_models.py --data_dir "il_datasets/medium_discard_classification" --models_dir "models/classification" --epochs 5 --eval_samples 2048 --max_shards 5`  
**Benchmark Cmd:** `python .\scripts\benchmark_2_players.py --players NeuralClassificationPlayer,beginner --games 200 --models_dir "models/classification" --data_dir "il_datasets/medium_discard_classification" --max_shards 5`  
**Results (avg point diff primary):** ~ -28.47  
**Notes / Next Steps:** Pegging stronger than discard; discard appears to be bottleneck.

---

## 2026-02-03

**Goal:** Regression discard with medium-player labels.  
**Dataset:** `il_datasets/medium_discard_regression` (~4k games)  
**Model / Loss:** Linear (regression)  
**Training Cmd:** `python .\scripts\train_linear_models.py --data_dir "il_datasets/medium_discard_regression" --models_dir "models/regression" --epochs 5 --eval_samples 2048 --max_shards 3`  
**Benchmark Cmd:** `python .\scripts\benchmark_2_players.py --players NeuralRegressionPlayer,beginner --games 200 --models_dir "models/regression" --data_dir "il_datasets/medium_discard_regression" --max_shards 3`  
**Results (avg point diff primary):** ~ -25.89  
**Notes / Next Steps:** Slight improvement vs classification; discard still weak.

---

## 2026-02-03

**Goal:** Ranking loss for discard.  
**Dataset:** `il_datasets/medium_discard_ranking` (~8k games)  
**Model / Loss:** Linear (pairwise ranking)  
**Training Cmd:** `python .\scripts\train_linear_models.py --data_dir "il_datasets/medium_discard_ranking" --models_dir "models/ranking" --discard_loss ranking --epochs 6 --eval_samples 2048 --max_shards 6 --rank_pairs_per_hand 20`  
**Benchmark Cmd:** `python .\scripts\benchmark_2_players.py --players NeuralRegressionPlayer,beginner --games 200 --models_dir "models/ranking" --data_dir "il_datasets/medium_discard_ranking" --max_shards 6`  
**Results (avg point diff primary):** ~ -28.75  
**Notes / Next Steps:** Ranking did not beat regression at this scale; discard remains bottleneck.
