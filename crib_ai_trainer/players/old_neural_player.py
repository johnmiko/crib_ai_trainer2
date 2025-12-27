import torch
from cribbage.playingcards import Card, Deck
from crib_ai_trainer.features import encode_state, D_TOTAL
from crib_ai_trainer.players.rule_based_player import ReasonablePlayer

class NeuralPlayer:
    def __init__(self, model):
        self.model = model
        self.teacher = ReasonablePlayer()  # fallback for invalid actions

    def select_crib_cards(self, hand, dealer_is_self):
        starter = Card(rank=Deck.RANKS['five'], suit=Deck.SUITS['spades'])
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
        return self.teacher.select_crib_cards(hand, dealer_is_self)
    
    def select_card_to_play(self, hand, table, crib, count):
        return self.play_pegging(hand, count, history_since_reset=table)

    def play_pegging(self, playable, count, history_since_reset):
        starter = Card(rank=Deck.RANKS['five'], suit=Deck.SUITS['spades'])
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
