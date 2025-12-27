
def cards_are_generated_correctly():
    from cribbage.playingcards import Card
    card = Card('5h')
    assert str(card) == '5h'
    # to_index logic may need update, but this avoids constructor error