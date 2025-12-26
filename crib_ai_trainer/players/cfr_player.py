from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from logging import getLogger
from cribbage.playingcards import Card
from cribbage.cribbagegame import score_play as score_pegging_play, RANK_VALUE

logger = getLogger(__name__)

class CFRPlayer:
    def __init__(self, name: str = "cfr", iterations: int = 1000, load_path: str = None):
        self.name = name
        self.iterations = iterations
        self.regrets_pegging: Dict[str, Dict[int, float]] = {}
        self.strategy_sum_pegging: Dict[str, Dict[int, float]] = {}
        self.regrets_discard: Dict[str, Dict[int, float]] = {}
        self.strategy_sum_discard: Dict[str, Dict[int, float]] = {}
        if load_path is not None:
            self.load(load_path)
    def save(self, path: str):
        import json
        data = {
            'regrets_pegging': self.regrets_pegging,
            'strategy_sum_pegging': self.strategy_sum_pegging,
            'regrets_discard': self.regrets_discard,
            'strategy_sum_discard': self.strategy_sum_discard
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        import json
        import os
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.regrets_pegging = {k: {int(a): v for a, v in d.items()} for k, d in data.get('regrets_pegging', {}).items()}
            self.strategy_sum_pegging = {k: {int(a): v for a, v in d.items()} for k, d in data.get('strategy_sum_pegging', {}).items()}
            self.regrets_discard = {k: {int(a): v for a, v in d.items()} for k, d in data.get('regrets_discard', {}).items()}
            self.strategy_sum_discard = {k: {int(a): v for a, v in d.items()} for k, d in data.get('strategy_sum_discard', {}).items()}

    # State abstractions to keep tables manageable
    def _peg_state_key(self, count: int, history_since_reset: List[Card]) -> str:
        ranks = [c.rank for c in history_since_reset[-4:]]  # last 4 ranks
        return f"c:{count}|r:{','.join(map(str, ranks))}"

    def _discard_state_key(self, hand: List[Card]) -> str:
        ranks = sorted([c.rank for c in hand])
        return f"h:{','.join(map(str, ranks))}"

    def _regret_match(self, regrets: Dict[int, float]) -> Dict[int, float]:
        pos = {a: max(0.0, r) for a, r in regrets.items()}
        s = sum(pos.values())
        if s <= 1e-9:
            # uniform
            m = len(regrets)
            return {a: 1.0 / m for a in regrets}
        return {a: r / s for a, r in pos.items()}

    def _update_strategy_sum(self, table: Dict[str, Dict[int, float]], key: str, strat: Dict[int, float]) -> None:
        if key not in table:
            table[key] = {a: 0.0 for a in strat}
        for a, p in strat.items():
            table[key][a] += p

    def _avg_strategy(self, table: Dict[str, Dict[int, float]], key: str) -> Dict[int, float]:
        s = table.get(key, {})
        tot = sum(s.values())
        if tot <= 1e-9:
            # uniform
            m = len(s) if s else 1
            return {a: 1.0 / m for a in s} if s else {0: 1.0}
        return {a: v / tot for a, v in s.items()}

    def train_pegging(self, legal_actions_fn, opponent_policy_fn) -> None:
        # CFR iteration over abstract states sampled from opponent policy
        for _ in range(self.iterations):
            for count in range(0, 32):
                # sample short histories
                for ranks in ([], [5], [10], [11], [7, 8], [3, 3]):
                    hist = [Card('H', r) for r in ranks]
                    key = self._peg_state_key(count, hist)
                    actions = legal_actions_fn(count)
                    if not actions:
                        continue
                    # initialize
                    self.regrets_pegging.setdefault(key, {i: 0.0 for i in range(len(actions))})
                    # evaluate utilities (immediate points)
                    utils = []
                    for i, c in enumerate(actions):
                        u = score_pegging_play(hist, c, count)
                        utils.append(u)
                    # current strategy from regrets
                    strat = self._regret_match(self.regrets_pegging[key])
                    self._update_strategy_sum(self.strategy_sum_pegging, key, strat)
                    # average utility
                    avg_u = sum(strat[i] * utils[i] for i in range(len(actions)))
                    # update regrets
                    for i in range(len(actions)):
                        self.regrets_pegging[key][i] += utils[i] - avg_u

    def train_discard(self, hand_sampler_fn) -> None:
        for _ in range(self.iterations):
            hand = hand_sampler_fn()
            key = self._discard_state_key(hand)
            n = len(hand)
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
            # initialize
            self.regrets_discard.setdefault(key, {k: 0.0 for k in range(len(pairs))})
            # utility: simple heuristic prefers low cards kept
            utils = []
            for idx, (i, j) in enumerate(pairs):
                kept = [hand[k] for k in range(n) if k not in (i, j)]
                u = -sum(RANK_VALUE[c.rank] for c in kept) * 0.01
                utils.append(u)
            strat = self._regret_match(self.regrets_discard[key])
            self._update_strategy_sum(self.strategy_sum_discard, key, strat)
            avg_u = sum(strat[i] * utils[i] for i in range(len(pairs)))
            for i in range(len(pairs)):
                self.regrets_discard[key][i] += utils[i] - avg_u

    def choose_discard(self, hand: List[Card], dealer_is_self: bool) -> Tuple[Card, Card]:
        key = self._discard_state_key(hand)
        n = len(hand)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        if key not in self.strategy_sum_discard:
            # uniform
            i, j = pairs[0]
            return (hand[i], hand[j])
        strat = self._avg_strategy(self.strategy_sum_discard, key)
        # pick argmax
        best_idx = max(strat.keys(), key=lambda k: strat[k])
        i, j = pairs[best_idx]
        return (hand[i], hand[j])

    def play_pegging(self, playable: List[Card], count: int, history_since_reset: List[Card]) -> Optional[Card]:
        if not playable:
            return None
        key = self._peg_state_key(count, history_since_reset)
        if key not in self.strategy_sum_pegging:
            # pick lowest value
            return sorted(playable, key=lambda c: RANK_VALUE[c.rank])[0]
        # map actions to indices by rank ordering
        actions_sorted = sorted(range(len(playable)), key=lambda i: RANK_VALUE[playable[i].rank])
        strat = self._avg_strategy(self.strategy_sum_pegging, key)
        # choose highest-prob action among available
        # Fallback to first action if mismatch
        best_idx = max(strat.keys(), key=lambda k: strat[k])
        best_idx = actions_sorted[min(best_idx, len(actions_sorted) - 1)]
        return playable[best_idx]
