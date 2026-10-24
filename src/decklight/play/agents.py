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


@register
class RawPolicyAgent:
    """The trained policy, argmax over legal options (client mode: needs no engine)."""

    name = "raw"
    needs: frozenset[str] = frozenset()

    def __init__(self, ckpt=None, emb=None, device: str = "cpu", **_) -> None:
        from decklight.search.product import load_policy, resolve_product_paths

        ckpt_p, _value_p, emb_p = resolve_product_paths(ckpt, None, emb)
        self.policy = load_policy(ckpt_p, emb_p, device=device)

    def rps(self) -> int:
        return random.randint(1, 3)

    def go_first(self) -> bool:
        return True

    def act(self, view: SeatView) -> int:
        import numpy as np

        obs = view.obs()
        batch = {k: np.ascontiguousarray(v)[None] for k, v in obs.items()}
        z = np.asarray(self.policy.logits(batch)[0], dtype=np.float64)
        legal = np.asarray(obs["action_mask"])[: len(view.opts)].astype(bool)
        if not legal.any():
            raise RuntimeError("raw agent: no legal option")
        z = z[: len(view.opts)]
        if not np.isfinite(z[legal]).all():
            raise RuntimeError("raw agent: NaN logits over legal options")
        z[~legal] = -np.inf
        return int(np.argmax(z))
