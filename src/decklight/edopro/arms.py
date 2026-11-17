"""Forced chain-0 activation arms for the `tsoj-arms` experiment.

`ArmedAgent` wraps the raw policy. For an arm other than `p`, at the K-th window where
P would NOT activate (opponent's turn, empty chain, at least one legal ACTIVATE) it plays
a forced activation instead: `top` takes the ACTIVATE option P scores highest, `rand` a
uniform random legal one. Exactly one decision per game is forced; afterwards P plays on.
Every game writes one JSON line (also when no K-th window came: ``forced: false``).
"""

from __future__ import annotations

import json
import random
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decklight.engine.actions import Kind


@dataclass(frozen=True)
class ArmSpec:
    kind: str  # "p" | "top" | "rand"
    k: int = 0


def parse_arm(spec: str) -> ArmSpec:
    """`p` | `top@K` | `rand@K` (K >= 1)."""
    if spec in ("", "p"):
        return ArmSpec("p", 0)
    kind, sep, raw = spec.partition("@")
    if not sep or kind not in ("top", "rand"):
        raise ValueError(f"bad arm {spec!r} (want p, top@K or rand@K)")
    try:
        k = int(raw)
    except ValueError:
        raise ValueError(f"bad arm {spec!r}: K is not an integer") from None
    if k < 1:
        raise ValueError(f"bad arm {spec!r}: K must be >= 1")
    return ArmSpec(kind, k)


def activate_indices(obs: dict, opts) -> list[int]:
    """Legal ACTIVATE option indices, in policy order (obs mask and opts agree)."""
    mask = np.asarray(obs["action_mask"]).astype(bool)[: len(opts)]
    return [i for i, o in enumerate(opts) if mask[i] and o.kind == Kind.ACTIVATE]


def is_window(obs: dict, opts, p_index: int) -> tuple[bool, list[int]]:
    """``filter_window`` (explore_arms.py) plus "P itself would not activate"."""
    g = np.asarray(obs["global"])
    if int(g[5]) != 0:  # opponent's turn
        return False, []
    if int(g[20]) != 0:  # empty chain
        return False, []
    acts = activate_indices(obs, opts)
    if not acts:
        return False, []
    if opts[p_index].kind == Kind.ACTIVATE:
        return False, []
    return True, acts


_NAMES: dict[int, str] | None = None


def _name(code: int) -> str:
    global _NAMES
    if _NAMES is None:
        try:
            from decklight.web.view import DB

            _NAMES = DB.names()
        except Exception:  # noqa: BLE001 - names are a nicety; never break a duel
            _NAMES = {}
    return _NAMES.get(int(code), str(int(code)))


def _option_name(opt) -> str:
    code = int(opt.code or (opt.card.code if opt.card is not None else 0))
    return _name(code) if code else opt.kind.name


class ArmedAgent:
    """The raw policy plus one forced chain-0 activation (arm != p)."""

    name = "raw"
    needs: frozenset[str] = frozenset()

    def __init__(self, inner, spec: ArmSpec, seed: int = 0, game_index: int = 0,
                 log_path: str | Path | None = None, **_) -> None:
        self.inner = inner
        self.spec = spec
        self.rng = random.Random(int(seed) + int(game_index))
        self.log_path = Path(log_path) if log_path else None
        self.response_counter = None  # set by the session: P responses sent so far
        self.seat: int | None = None
        self.n_decisions = 0
        self.windows = 0
        self.forced = False
        self.forced_turn = 0
        self.forced_card = ""
        self.p_choice = ""
        self.n_activations = 0
        self.forced_decision = -1
        self.opening_hand: list[int] = []
        self._logged = False

    def rps(self) -> int:
        return self.inner.rps()

    def go_first(self) -> bool:
        return self.inner.go_first()

    def _scores(self, obs: dict, n: int) -> np.ndarray | None:
        policy = getattr(self.inner, "policy", None)
        if policy is None:
            return None
        batch = {k: np.ascontiguousarray(v)[None] for k, v in obs.items()}
        z = np.asarray(policy.logits(batch)[0], dtype=np.float64)[:n]
        legal = np.asarray(obs["action_mask"])[:n].astype(bool)
        z[~legal] = -np.inf
        return z

    def act(self, view) -> int:
        if self.spec.kind == "p":
            idx = self.inner.act(view)
        else:
            obs = view.obs()
            opts = view.opts
            z = self._scores(obs, len(opts))
            if z is None or not np.isfinite(z).any():
                idx = self.inner.act(view)
            else:
                idx = int(np.argmax(z))
                window, acts = is_window(obs, opts, idx)
                if window:
                    self.windows += 1
                    if self.windows == self.spec.k and not self.forced:
                        pick = max(acts, key=lambda i: z[i]) if self.spec.kind == "top" \
                            else self.rng.choice(acts)
                        self.forced = True
                        self.forced_turn = int(np.asarray(obs["global"])[2])
                        self.forced_card = _option_name(opts[pick])
                        self.p_choice = _option_name(opts[idx])
                        self.n_activations = len(acts)
                        self.forced_decision = (
                            int(self.response_counter()) if self.response_counter is not None
                            else self.n_decisions
                        )
                        idx = pick
        self.seat = getattr(view, "seat", self.seat)
        if not self.opening_hand:
            state = getattr(view, "state", None)
            try:
                from decklight.engine import constants as C

                hand = state.lists[(self.seat, C.LOCATION_HAND)] if state is not None else []
                self.opening_hand = sorted(int(c.code) for c in hand[:5] if c.code)
            except (AttributeError, KeyError):
                self.opening_hand = []
        self.n_decisions += 1
        return idx

    def finalize(self, record: Any, deck: str = "") -> None:
        """Write the game's one arms.jsonl line (idempotent); also finalizes the inner agent (remote: episode file)."""
        inner_fin = getattr(self.inner, "finalize", None)
        if inner_fin is not None:
            inner_fin(record, deck=deck)
        if self.log_path is None or self._logged:
            return
        self._logged = True
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        winner = getattr(record, "winner", None)
        if isinstance(winner, tuple):
            winner = winner[0]
        row = {
            "arm": self.spec.kind, "k": self.spec.k, "forced": self.forced,
            "turn": self.forced_turn, "card": self.forced_card,
            "p_choice": self.p_choice, "n_activations": self.n_activations,
            "decision": self.forced_decision,
            "seat": self.seat, "deck": deck, "hand": self.opening_hand,
            "n_responses": int(getattr(record, "n_responses", 0)),
            "winner": winner, "turns": int(getattr(record, "turns", 0)),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
