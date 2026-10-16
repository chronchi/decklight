"""Bot interface for the EDOPro host: one seat's pending decision -> option index."""

import random
from typing import Protocol


class Bot(Protocol):
    def act(self, env) -> int:
        """Option index into `env.opts` for the env's current pending decision."""
        ...


class RandomBot:
    """Uniform random legal player (D2 stand-in; D3 plugs in the real product)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, env) -> int:
        return self.rng.randrange(len(env.opts))

    def rps(self) -> int:
        return self.rng.randint(1, 3)
