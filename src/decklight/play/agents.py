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
class ProductAgent:
    """The full product (policy + search) for engine transports (host, web).

    Search runs lockstep with the authoritative engine through the host/web
    driver (`Bundle.make_launcher`), not through one `act` call per decision, so
    `act` fails fast with that pointer. The client (no local engine) is refused
    at startup by `check_needs` (`needs == {"engine"}`).
    """

    name = "product"
    needs: frozenset[str] = frozenset({ENGINE})

    def __init__(self, ckpt=None, value=None, emb=None, device: str = "cpu",
                 fast: bool = False, **_) -> None:
        self.ckpt = ckpt
        self.value = value
        self.emb = emb
        self.device = device
        self.fast = fast

    def rps(self) -> int:
        return random.randint(1, 3)

    def go_first(self) -> bool:
        return True

    def launcher(self):
        from decklight.edopro.product_bot import ProductLauncher

        return ProductLauncher(ckpt=self.ckpt, value_ckpt=self.value, emb=self.emb,
                               device=self.device, fast=self.fast)

    def act(self, view: SeatView) -> int:
        raise RuntimeError("product search runs through the host/web driver "
                           "(Bundle.make_launcher), not act()")


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


@register
class RemoteAgent:
    """The PPO learner over a PolicyServer socket (decklight.rl.remote): samples, logs, writes one episode file.

    Config from the environment (the arbiter starts the client per game): DL_POLICY_SOCK (the server's unix
    socket), DL_EPISODE_DIR (where `finalize` writes the episode). Imports no torch."""

    name = "remote"
    needs: frozenset[str] = frozenset()

    def __init__(self, sock=None, episode_dir=None, **_) -> None:
        import os

        from decklight.rl.remote import EpisodeLog, PolicyClient

        sock = sock or os.environ.get("DL_POLICY_SOCK")
        self.episode_dir = episode_dir or os.environ.get("DL_EPISODE_DIR")
        if not sock or not self.episode_dir:
            raise ValueError("remote agent needs DL_POLICY_SOCK and DL_EPISODE_DIR")
        if not (os.path.isabs(sock) and os.path.isabs(self.episode_dir)):  # the client runs in its own directory
            raise ValueError(f"remote agent needs absolute paths: {sock}, {self.episode_dir}")
        self.client = PolicyClient(sock)
        self.log = EpisodeLog()
        self.seat: int | None = None
        self.path = None

    def rps(self) -> int:
        return random.randint(1, 3)

    def go_first(self) -> bool:
        return True

    def act(self, view: SeatView) -> int:
        import numpy as np

        obs = view.obs()
        legal = np.asarray(obs["action_mask"])[: len(view.opts)].astype(bool)
        if not legal.any():
            raise RuntimeError("remote agent: no legal option")
        a, lp, v, ver = self.client.act(obs)
        if not (0 <= a < len(view.opts) and legal[a]):
            raise RuntimeError(f"remote agent: server picked illegal option {a} of {len(view.opts)}")
        self.log.add(obs, a, lp, v, ver)
        self.seat = getattr(view, "seat", self.seat)
        return a

    def finalize(self, record, deck: str = "") -> None:
        """Write the episode (+1 win, -1 loss, 0 draw/unknown winner); idempotent; no file for 0 decisions."""
        if self.path is not None or not len(self.log):
            return
        winner = getattr(record, "winner", None)
        w = winner[0] if isinstance(winner, (tuple, list)) else winner
        reward = 0.0 if w not in (0, 1) or self.seat is None else (1.0 if w == self.seat else -1.0)
        meta = {"deck": deck, "seat": self.seat, "winner": w, "turns": int(getattr(record, "turns", 0) or 0),
                "n": len(self.log)}
        self.path = self.log.write(self.episode_dir, reward, meta)
        self.client.close()
