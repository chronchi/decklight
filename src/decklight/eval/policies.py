"""Fixed baseline policies for the gauntlet. A policy maps a batch of observations (the VecEnv buffers,
restricted to `rows`) to one option index per row."""

from typing import Protocol

import numpy as np

from decklight.engine.actions import Kind


class Policy(Protocol):
    name: str

    def __call__(self, buf: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray: ...


class RandomPolicy:
    """Uniform over legal options."""

    def __init__(self, seed: int = 0) -> None:
        self.name, self.rng = "random", np.random.default_rng(seed)

    def __call__(self, buf, rows):
        counts = buf["action_mask"][rows].sum(1)
        return (self.rng.random(len(rows)) * counts).astype(np.int32)


# Higher = preferred. Proactive play first, passing/ending last; anything unlisted scores 1.
_GREEDY = np.ones(32, np.float32)
for _k, _v in {
    Kind.ACTIVATE: 6, Kind.SPSUMMON: 6, Kind.SUMMON: 5, Kind.ATTACK: 5, Kind.TO_BATTLE: 4, Kind.SSET: 3,
    Kind.MSET: 2, Kind.YES: 3, Kind.NO: 1, Kind.FINISH: 1, Kind.CANCEL: 0, Kind.REPOS: 0.5, Kind.TO_MAIN2: 0.7,
    Kind.TO_END: 0.5, Kind.PASS: 0.5,
}.items():  # fmt: skip
    _GREEDY[_k] = _v


class GreedyPolicy:
    """Always does *something*: activate / summon / attack when it can, random tie-breaks. A slightly
    stronger floor than random (it attacks and uses effects), not a real player."""

    def __init__(self, seed: int = 0) -> None:
        self.name, self.rng = "greedy", np.random.default_rng(seed)

    def __call__(self, buf, rows):
        kinds = buf["actions"][rows, :, 0]
        score = _GREEDY[kinds] + self.rng.random(kinds.shape) * 0.1
        score[~buf["action_mask"][rows]] = -np.inf
        return score.argmax(1).astype(np.int32)
