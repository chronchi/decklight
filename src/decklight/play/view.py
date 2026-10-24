"""What one seat knows: the interface agents read (docs/plans/2026-10-24-edopro-client.md §3.1).

Two implementations, one observation code path:
- `EngineView`: a seat of a local authoritative `PyEnv` (host, web, search).
- `StreamView` (decklight.play.stream): rebuilt from the filtered EDOPro message stream (network client).

Both build observations with `decklight.engine.obs.build_obs`, so a policy sees the same arrays
whichever transport feeds it (the client-parity audit measures where they still differ).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from decklight.engine import obs as O
from decklight.engine.actions import Decision, Option


class BoardSource(Protocol):
    """What `engine.obs.Board` reads. `obs.DuelSource` adapts a duel; `stream.StreamState` rebuilds it."""

    turn: int
    phase: int
    first_player: int
    turn_player: int
    memory: object  # engine.knowledge.HandMemory-compatible

    def query_location(self, player: int, location: int, flags: int = ...) -> list: ...

    def field_state(self) -> tuple[tuple[int, int], list]:
        """(lps, chain links); a duel's version parses it from `query_field` (see `obs.DuelSource`)."""
        ...


class SeatView(Protocol):
    seat: int

    @property
    def decision(self) -> Decision | None:
        """The pending decision for this seat (None when nothing is asked)."""
        ...

    @property
    def opts(self) -> list[Option]:
        """Legal options of the pending decision (capped at obs.MAX_OPTIONS)."""
        ...

    def obs(self) -> dict[str, np.ndarray]:
        """Observation for the pending decision, the policy's input."""
        ...

    def take(self, index: int) -> bytes | None:
        """Answer with option `index`; the core response once the decision is complete, else None."""
        ...

    @property
    def needs_answer(self) -> bool:
        """A pending decision waits for the agent (False once its response is ready)."""
        ...

    @property
    def done(self) -> bool:
        """The duel ended."""
        ...

    @property
    def winner(self) -> int | None:
        """The winner (None while running)."""
        ...


class EngineView:
    """A `SeatView` over a live `PyEnv` (host, web, search): no filtering, full knowledge."""

    def __init__(self, env) -> None:
        self.env = env

    @property
    def seat(self) -> int:
        return self.env.player

    @property
    def decision(self) -> Decision | None:
        return self.env.pending_decision()

    @property
    def opts(self) -> list[Option]:
        return list(self.env.opts) if self.decision is not None else []

    def obs(self) -> dict[str, np.ndarray]:
        env = self.env
        board = O.Board(O.DuelSource(env.duel), env.player)
        return O.build_obs(board, env.dec, env.opts, env.history, O.DuelSource(env.duel))

    def take(self, index: int) -> bytes | None:
        self.env.step(index)
        return None

    @property
    def needs_answer(self) -> bool:
        return self.decision is not None and not self.env.done

    @property
    def done(self) -> bool:
        return self.env.done

    @property
    def winner(self) -> int | None:
        return self.env.winner
