from itertools import combinations
from typing import List
# adjust imports to your project
from playingCards import Deck, Card
from scoring import score_hand


def get_full_deck() -> List[Card]:
    return list(Deck().cards)


def card_id(card: Card) -> str:
    # Anything stable works — this is human-readable
    return str(card)


def hand_key(hand: List[Card]) -> str:
    # sort for canonical ordering
    return "-".join(sorted(card_id(c) for c in hand))


def precompute_5card_scores(output_path: str) -> None:
    full_deck = get_full_deck()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("hand,score\n")

        for hand_tuple in combinations(full_deck, 5):
            hand = list(hand_tuple)
            score = score_hand(hand, is_crib=True)
            key = hand_key(hand)
            f.write(f"{key},{score}\n")


if __name__ == "__main__":
    precompute_5card_scores("crib_5card_scores.csv")
