import os
import torch
from models.perceptron import SimplePerceptron, PerceptronConfig
from crib_ai_trainer.features import D_TOTAL

MODEL_PATH = os.path.join('trained_models', 'perceptron.pt')


def load_perceptron():
    from logging import getLogger
    logger = getLogger(__name__)
    model = SimplePerceptron(PerceptronConfig(input_dim=D_TOTAL, output_dim=52))
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            logger.info(f"Loaded perceptron weights from {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load perceptron weights: {e}")
    return model

def save_perceptron(model):
    from logging import getLogger
    logger = getLogger(__name__)
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        logger.info(f"Saved perceptron weights to {MODEL_PATH}")
    except Exception as e:
        logger.warning(f"Could not save perceptron weights: {e}")
