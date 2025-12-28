import os
import json
import matplotlib.pyplot as plt
from logging import getLogger

logger = getLogger(__name__)

def save_ranking_report(ranking_data, out_path):
    with open(out_path, "w") as f:
        json.dump(ranking_data, f, indent=2)
    logger.info(f"Saved ranking report to {out_path}")

def plot_model_performance(history, out_path_prefix):
    # history: list of dicts with keys 'model', 'winrate', 'games', 'timestamp'
    models = sorted(set(h['model'] for h in history))
    for model in models:
        times = [h['timestamp'] for h in history if h['model'] == model]
        winrates = [h['winrate'] for h in history if h['model'] == model]
        plt.plot(times, winrates, label=model)
    plt.xlabel('Time')
    plt.ylabel('Winrate')
    plt.title('Model Winrate Over Time')
    plt.legend()
    plt.tight_layout()
    out_path = out_path_prefix + "_winrate.png"
    plt.savefig(out_path)
    logger.info(f"Saved winrate plot to {out_path}")
    plt.close()
