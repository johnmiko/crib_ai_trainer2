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
        self.name = "perceptron"

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

    # --- Player interface ---
    def choose_discard(self, hand, dealer_is_self):
        # Select 2 cards to discard from hand (length 6)
        import numpy as np
        from crib_ai_trainer.features import encode_state
        best_pair = (hand[0], hand[1])
        best_score = -float('inf')
        for i in range(len(hand)):
            for j in range(i+1, len(hand)):
                kept = [hand[k] for k in range(len(hand)) if k not in (i, j)]
                # Dummy values for starter/seen/count/history
                state = encode_state(kept, None, [], 0, [])
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                score = self.forward(x)[0].max().item()
                if score > best_score:
                    best_score = score
                    best_pair = (hand[i], hand[j])
        return best_pair

    def play_pegging(self, playable, count, history_since_reset):
        # Select a card to play from playable
        import numpy as np
        from crib_ai_trainer.features import encode_state
        if not playable:
            return None
        # Use dummy hand/starter/seen for encoding
        hand = playable
        state = encode_state(hand, None, [], count, history_since_reset)
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x)[0].detach().numpy()
        # Mask out non-playable actions
        playable_idxs = [c.to_index() for c in playable]
        mask = np.full_like(logits, -np.inf)
        for idx in playable_idxs:
            mask[idx] = logits[idx]
        best_idx = int(np.argmax(mask))
        for c in playable:
            if c.to_index() == best_idx:
                return c
        return playable[0]
