"""Who decides: agents behind one interface, chosen by name (docs/plans/2026-10-24-edopro-client.md §3.2).

An agent answers one seat's pending decision from a `SeatView` (decklight.play.view). Transports (EDOPro host,
EDOPro client, web) declare what they provide; `check_needs` refuses an agent whose needs are not met, at startup.

    agent = make_agent("raw", ckpt=..., device="cpu")
    check_needs(agent, provides=frozenset(), where="edopro client")
    index = agent.act(view)          # index into view.opts

Add an agent: write a class with `name`, `needs`, `rps`, `go_first`, `act`, and decorate it with `@register`.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol

from decklight.play.view import SeatView

ENGINE = "engine"  # the transport holds the authoritative local engine (needed by the product search)


class Agent(Protocol):
    name: str
    needs: frozenset[str]

    def rps(self) -> int:
        """Rock/paper/scissors throw (1/2/3)."""
        ...

    def go_first(self) -> bool:
        """Turn-order choice after winning RPS."""
        ...

    def act(self, view: SeatView) -> int:
        """Index into `view.opts` for the view's pending decision."""
        ...


AGENTS: dict[str, Callable[..., Agent]] = {}


def register(cls):
    """Class decorator: make the agent available to `make_agent(cls.name, ...)`."""
    if cls.name in AGENTS:
        raise ValueError(f"agent {cls.name!r} registered twice")
    AGENTS[cls.name] = cls
    return cls


def make_agent(name: str, **cfg) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; known: {sorted(AGENTS)}")
    return AGENTS[name](**cfg)


class MissingCapability(RuntimeError):
    pass


def check_needs(agent: Agent, provides: frozenset[str], where: str) -> None:
    missing = agent.needs - provides
    if missing:
        raise MissingCapability(
            f"agent {agent.name!r} needs {sorted(missing)}, which {where} does not provide"
        )


@register
class RandomAgent:
    """Uniform random legal player (smoke tests, protocol checks)."""

    name = "random"
    needs: frozenset[str] = frozenset()

    def __init__(self, seed: int = 0, **_) -> None:
        self.rng = random.Random(seed)

    def rps(self) -> int:
        return self.rng.randint(1, 3)

    def go_first(self) -> bool:
        return True

    def act(self, view: SeatView) -> int:
        return self.rng.randrange(len(view.opts))
