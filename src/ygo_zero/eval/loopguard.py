"""Break exact-repetition loops in policy play (NOTEBOOK 2026-10-06 "loops").

Deterministic play (argmax, or a sampled policy that is very sure) can cycle through interface choices that undo each
other: SELECT x / UNSELECT x in a SELECT_UNSELECT_CARD, SPSUMMON -> FINISH (cancel) -> the same idle command, an
activation cancelled at its first prompt. The state comes back exactly, the policy picks the same option, and the
game hits the decision cap (round 1: 39 cut playtest games of ft-swv, every one a 1-2 option cycle repeated ~1500×).

LoopGuard remembers, per slot (one env), which options were taken in each exact decision state of the current turn;
when a state recurs, those options are masked, so the policy must try something else (an UNSELECT loop then FINISHes,
a cancel loop picks another command). The state key is the observation without the history (board, chain, options,
globals); a slot's memory is cleared when its turn number changes. If every legal option was already taken there,
nothing is masked. Exact repeats within a turn do not happen in normal play, so the guard changes nothing else.
"""

from __future__ import annotations

import numpy as np

STATE_KEYS = ("cards", "card_mask", "global", "chain", "chain_mask", "actions", "action_mask")


def state_key(obs: dict[str, np.ndarray]) -> int:
    return hash(b"".join(np.ascontiguousarray(obs[k]).tobytes() for k in STATE_KEYS if k in obs))


class LoopGuard:
    def __init__(self) -> None:
        self.turn: dict[int, int] = {}
        self.seen: dict[int, dict[int, set[int]]] = {}
        self.masked = 0  # decisions where the guard removed at least one option (for logs / tests)

    def reset(self) -> None:
        self.turn.clear()
        self.seen.clear()

    def mask(self, slot: int, obs: dict[str, np.ndarray], z: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (z with options already taken in this exact state set to -inf, state key). `z` holds -inf on
        illegal options. Call `note(slot, key, action)` with the option finally played."""
        turn = int(obs["global"][2])
        if self.turn.get(slot) != turn:
            self.turn[slot] = turn
            self.seen[slot] = {}
        key = state_key(obs)
        taken = self.seen[slot].get(key)
        if taken:
            z2 = z.copy()
            z2[[a for a in taken if a < len(z2)]] = -np.inf
            if np.isfinite(z2).any():
                self.masked += 1
                return z2, key
        return z, key

    def note(self, slot: int, key: int, action: int) -> None:
        self.seen.setdefault(slot, {}).setdefault(key, set()).add(int(action))
