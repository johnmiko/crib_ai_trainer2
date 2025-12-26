from dataclasses import dataclass, asdict
from typing import List, Optional
import json

@dataclass
class NeuralNetConfig:
    hidden_sizes: List[int]
    depth: int
    learning_rate: float
    entropy_coef: float = 0.0
    regularization: Optional[float] = None
    name: Optional[str] = None

    def save(self, path: str, results: dict = None):
        import os
        from logging import getLogger
        logger = getLogger(__name__)
        data = asdict(self)
        if results:
            data['results'] = results
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"Created directory for saving neural config: {dir_name}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_name}: {e}")
                return
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved neural config and results to {path}")
        except Exception as e:
            logger.error(f"Failed to save neural config to {path}: {e}")

    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return NeuralNetConfig(**{k: v for k, v in data.items() if k != 'results'})
