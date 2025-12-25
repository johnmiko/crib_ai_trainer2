from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class PerceptronConfig:
    input_dim: int
    output_dim: int

class SimplePerceptron(nn.Module):
    def __init__(self, cfg: PerceptronConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.input_dim, cfg.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax over actions (logits)
        return torch.log_softmax(self.linear(x), dim=-1)

    def export_npz(self, path: str) -> None:
        w = self.linear.weight.detach().cpu().numpy().astype(np.float32)
        b = self.linear.bias.detach().cpu().numpy().astype(np.float32)
        np.savez(path, w=w, b=b)

    def predict_action(self, x: torch.Tensor) -> int:
        # Returns the action index with highest probability
        logits = self.forward(x)
        return int(torch.argmax(logits, dim=-1).item())
