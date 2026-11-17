"""Bot interface for the EDOPro host: one seat's pending decision -> option index."""

import random
from typing import Protocol


class Bot(Protocol):
    def next_choice(self, env) -> int:
        """Option index into `env.opts` for the env's current pending decision."""
        ...

    def rps(self) -> int:
        """Rock/paper/scissors throw (1/2/3)."""
        ...


class RandomBot:
    """Uniform random legal player (D2 stand-in; D3 plugs in the real product)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def next_choice(self, env) -> int:
        return self.rng.randrange(len(env.opts))

    def rps(self) -> int:
        return self.rng.randint(1, 3)


class StreamBot:
    """Deterministic test player: option = splitmix64(seed) stream % number of options. The same stream as
    `scripts/app/play_dump.py` and the Rust dumpers, so a Python and a Rust server facing it play the same game."""

    _M = (1 << 64) - 1

    def __init__(self, seed: int = 0) -> None:
        self.state = seed & self._M

    def _next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._M
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._M
        return z ^ (z >> 31)

    def next_choice(self, env) -> int:
        return self._next() % len(env.opts)

    def rps(self) -> int:
        return 1 + self._next() % 3
