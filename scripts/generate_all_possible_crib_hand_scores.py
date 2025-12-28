
import sqlite3
import itertools

from crib_ai_trainer.old.scoring_old import score_hand
from crib_ai_trainer.players.rule_based_player import get_full_deck

# You said this exists already:
# from your_module import get_full_deck, score_hand
# For example:
# full_deck = get_full_deck()

DB_PATH = "crib_cache.db"


def normalize_hand(cards):
    """
    Sort + stringify so identical hands map to one key.
    Example key: "5C|7D|9H|JS|KH"
    """
    return "|".join(sorted(str(c) for c in cards))


def setup_db(conn):
    cur = conn.cursor()

    # Hand lookup table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hand_scores (
            hand_key TEXT PRIMARY KEY,
            score INTEGER
        )
        """
    )

    # Placeholder for future crib averages
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crib_averages (
            context_key TEXT PRIMARY KEY,
            avg_value REAL
        )
        """
    )

    conn.commit()


def build_hand_table(full_deck):
    conn = sqlite3.connect(DB_PATH)
    setup_db(conn)
    cur = conn.cursor()

    total = 0

    for combo in itertools.combinations(full_deck, 5):
        hand_key = normalize_hand(combo)

        # Skip if already stored
        cur.execute(
            "SELECT 1 FROM hand_scores WHERE hand_key = ?",
            (hand_key,),
        )
        if cur.fetchone():
            continue

        score = score_hand(list(combo))

        cur.execute(
            "INSERT INTO hand_scores (hand_key, score) VALUES (?, ?)",
            (hand_key, score),
        )

        total += 1
        if total % 5000 == 0:
            conn.commit()
            print(f"Saved {total} hands...")

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    full_deck = get_full_deck()
    build_hand_table(full_deck)
    

# old?
# from itertools import combinations
# from typing import List
# # adjust imports to your project
# from playingCards import Deck, Card
# from scoring import score_hand


# def get_full_deck() -> List[Card]:
#     return list(Deck().cards)


# def card_id(card: Card) -> str:
#     # Anything stable works — this is human-readable
#     return str(card)


# def hand_key(hand: List[Card]) -> str:
#     # sort for canonical ordering
#     return "-".join(sorted(card_id(c) for c in hand))


# def precompute_5card_scores(output_path: str) -> None:
#     full_deck = get_full_deck()

#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write("hand,score\n")

#         for hand_tuple in combinations(full_deck, 5):
#             hand = list(hand_tuple)
#             score = score_hand(hand, is_crib=True)
#             key = hand_key(hand)
#             f.write(f"{key},{score}\n")


# if __name__ == "__main__":
#     precompute_5card_scores("crib_5card_scores.csv")
