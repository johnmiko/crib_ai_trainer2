"""
Benchmark the 'reasonable' opponent against the 'best' model.
Run with: .venv\Scripts\Activate.ps1; python -m scripts.benchmark_reasonable_vs_best [num_games]
"""
import logging
from crib_ai_trainer.players.rule_based_player import ReasonablePlayer, DifficultReasonablePlayer
from crib_ai_trainer.model_registry import load_best_model
from crib_ai_trainer.game import CribbageGame

def main(num_games=100):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Benchmarking 'reasonable' opponent vs 'best' model for {num_games} games...")

    reasonable = ReasonablePlayer()
    difficult = DifficultReasonablePlayer()
    best_model = load_best_model()
    # Optionally benchmark difficult as well
    logger.info(f"Benchmarking 'difficult' opponent vs 'best' model for {num_games} games...")
    wins_difficult = {"difficult": 0, "best": 0}
    for i in range(num_games):
        if i % 2 == 0:
            p0, p1 = difficult, best_model
        else:
            p0, p1 = best_model, difficult
        game = CribbageGame(p0, p1)
        s0, s1 = game.play_game()
        if s0 > s1:
            winner = "difficult" if i % 2 == 0 else "best"
        else:
            winner = "best" if i % 2 == 0 else "difficult"
        wins_difficult[winner] += 1
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i+1} games for difficult...")
    logger.info(f"Difficult wins: {wins_difficult['difficult']} / {num_games} ({wins_difficult['difficult']/num_games:.1%})")
    logger.info(f"Best model wins vs difficult: {wins_difficult['best']} / {num_games} ({wins_difficult['best']/num_games:.1%})")

    wins = {"reasonable": 0, "best": 0}
    for i in range(num_games):
        # Alternate dealer for fairness
        if i % 2 == 0:
            p0, p1 = reasonable, best_model
        else:
            p0, p1 = best_model, reasonable
        game = CribbageGame(p0, p1)
        s0, s1 = game.play_game()
        if s0 > s1:
            winner = "reasonable" if i % 2 == 0 else "best"
        else:
            winner = "best" if i % 2 == 0 else "reasonable"
        wins[winner] += 1
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i+1} games...")

    logger.info("Benchmark complete.")
    logger.info(f"Reasonable wins: {wins['reasonable']} / {num_games} ({wins['reasonable']/num_games:.1%})")
    logger.info(f"Best model wins: {wins['best']} / {num_games} ({wins['best']/num_games:.1%})")

if __name__ == "__main__":
    import sys
    num_games = 500
    if len(sys.argv) > 1:
        try:
            num_games = int(sys.argv[1])
        except ValueError:
            print("Usage: python -m scripts.benchmark_reasonable_vs_best [num_games]")
            sys.exit(1)
    main(num_games)
