import torch
from crib_ai_trainer.cards import Card
from crib_ai_trainer.features import encode_state, D_TOTAL
from crib_ai_trainer.players.rule_based_player import RuleBasedPlayer

class NeuralPlayer:
    def __init__(self, model):
        self.model = model
        self.teacher = RuleBasedPlayer()  # fallback for invalid actions

    def choose_discard(self, hand, dealer_is_self):
        starter = Card('S', 5)  # dummy starter for encoding
        seen = []
        count = 0
        history = []
        x = encode_state(hand, starter, seen, count, history)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            action_idx = torch.argmax(logits, dim=1).item()
        # Map index to card in hand
        for card in hand:
            if card.to_index() == action_idx:
                # Return as tuple for compatibility
                return (card, hand[0] if card != hand[0] else hand[1])
        # Fallback to rule-based if not found
        return self.teacher.choose_discard(hand, dealer_is_self)

    def play_pegging(self, playable, count, history_since_reset):
        starter = Card('S', 5)  # dummy starter for encoding
        seen = []
        x = encode_state(playable, starter, seen, count, history_since_reset)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            action_idx = torch.argmax(logits, dim=1).item()
        for card in playable:
            if card.to_index() == action_idx:
                return card
        # Fallback to rule-based if not found
        return self.teacher.play_pegging(playable, count, history_since_reset)
