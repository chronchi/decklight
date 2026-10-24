"""What one seat knows: the interface agents read (docs/plans/2026-10-24-edopro-client.md §3.1).

Two implementations, one observation code path:
- `EngineView`: a seat of a local authoritative `PyEnv` (host, web, search).
- `StreamView` (decklight.play.stream): rebuilt from the filtered EDOPro message stream (network client).

Both build observations with `decklight.engine.obs` through the same function, so a policy sees the same arrays
whichever transport feeds it (the client-parity audit measures where they still differ).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from decklight.engine.actions import Decision, Option


class BoardSource(Protocol):
    """What `engine.obs.Board` reads. `engine.duel.Duel` satisfies it; `stream.StreamState` rebuilds it."""

    turn: int
    phase: int
    first_player: int
    turn_player: int
    memory: object  # engine.knowledge.HandMemory-compatible

    def query_location(self, player: int, location: int, flags: int = ...) -> list: ...

    def query_field(self) -> bytes: ...


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
