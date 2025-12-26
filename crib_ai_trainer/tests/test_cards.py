
def cards_are_generated_correctly():
    from crib_ai_trainer.cards import Card
    card = Card('H', 5)
    assert str(card) == '5H'
    assert card.to_index() == 17  # (Hearts is index 1, so 1*13 + (5-1) = 17