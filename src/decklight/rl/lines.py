"""Combo-line opponents for RL (docs/plans/2026-11-03-opp-turn-interaction.md, step 2).

The opponent's turn comes from an engine-verified line (ADR 0003), not from P: P pilots combos poorly, so P-driven
opponents teach on weak combos (NOTEBOOK 2026-11-03 "60/20/20"). A line start is a DAGS start (`dags.start_record`
shape) plus:
- `our_seat`: the learner's seat (the deck that answers the line);
- `script`: the opponent seat's decisions of the line, in order, as `[fingerprint, choice]`;
- `config` / `auto_single`: the record-style duel config (`record_config`), so evaluation can replay the start in a
  single env (`CppEnv` / `PyEnv`) from `prefix`.

During an episode the opponent plays `choice` while its decision's legal-option fingerprint equals the script's; the
first mismatch (our bot interrupted, so the state changed) ends the script and P (the league's base) continues the turn.
The fingerprint hashes the legal option rows (`buf["actions"]` under `action_mask`), identical between `CppEnv` and the
C++ VecEnv (checked 2026-11-03), so a script built by single-env replay matches rollout rows.

Two builders:
- `second_start`: a `plan_turns` record of the opponent's turn 2 into our turn-1 board (our deck seat 0, P built it);
  the start is the line's first decision, `script` is the rest of the opponent's turn.
- `first_start`: a verified going-first line (opponent seat 0) with our deck put in seat 1 and our opening hand placed;
  our side passes during the replay; valid only if the opponent ends turn 1 with the line's field.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from decklight.demos.dataset import record_config
from decklight.engine import constants as C
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import Engine
from decklight.rl import dags


def fingerprint(actions, mask) -> int:
    """crc32 of the legal option rows (int64 bytes, so dtype differences between buffers do not matter)."""
    m = np.asarray(mask).astype(bool).reshape(-1)
    rows = np.asarray(actions)[: len(m)][m]
    return zlib.crc32(np.ascontiguousarray(rows, dtype=np.int64).tobytes())


def _fp(obs) -> int:
    return fingerprint(obs["actions"], obs["action_mask"])


def second_start(engine: Engine, rec: dict) -> dict:
    """Line start from a `plan_turns` turn record (labels: "prefix" * k, then the line). Our seat = the other deck."""
    k = rec["labels"].index(next(x for x in rec["labels"] if x != "prefix"))
    actors = rec["actors"]
    opp = actors[k]
    env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
    obs = env.reset()
    script = []
    for j, c in enumerate(rec["choices"]):
        if obs is None:
            raise ValueError(f"{rec['line_id']}: duel ended before choice {j}")
        if j >= k and env.player == opp:
            script.append([_fp(obs), int(c)])
        obs = env.step(int(c))
    st = dags.start_record(rec, k, "line2")
    st.update(kind="second", line_id=rec["line_id"], our_seat=1 - opp, script=script, config=rec["config"],
              auto_single=rec.get("auto_single", True))
    return st


def _field_codes(env, player: int) -> list[int]:
    return sorted(int(x.code) for loc in (C.LOCATION_MZONE, C.LOCATION_SZONE)
                  for x in env.duel.query_location(player, loc) if x is not None)  # fmt: skip


def first_start(engine: Engine, rec: dict, ours, our_name: str, seed: int) -> dict | None:
    """Line start from a verified going-first record: seat 1 = `ours` with 5 cards placed in hand (sampled by `seed`),
    our side passes; None when the opponent's line no longer reaches its own turn-1 field (or a choice is illegal).
    Uses PyEnv for the field check and the passive answers, then rebuilds the fingerprints with CppEnv."""
    from decklight.demos.game import Responder
    from decklight.engine.env import PyEnv

    c = rec["config"]
    if int(c.get("hand", 0)) != 0:  # the 5 placed cards would come on top of a dealt hand (review 2026-11-04)
        raise ValueError(f"{rec['line_id']}: hand={c['hand']}, expected 0 with placed hands")
    orig = PyEnv(engine, record_config(rec), rec.get("auto_single", True))
    obs, opp_choices = orig.reset(), []
    for ch in rec["choices"]:
        if obs is None or orig.duel.turn > 1:
            break
        if orig.player == 0:
            opp_choices.append(int(ch))
        obs = orig.step(int(ch))
    want = _field_codes(orig, 0)
    rng = np.random.default_rng(seed)
    main = list(ours.main)
    hand = [int(main[i]) for i in rng.choice(len(main), 5, replace=False)]
    rest = list(main)
    for h in hand:
        rest.remove(h)
    setup = [list(p) for p in c.get("setup", [])] + [[1, h, C.LOCATION_HAND, 0] for h in hand]
    new = {"config": dict(c, deck1_main=rest, deck1_extra=list(ours.extra), setup=setup, seed=int(c["seed"])),
           "auto_single": rec.get("auto_single", True), "choices": [], "decks": [rec.get("deck", ""), our_name]}
    for attempt in ("py", "cpp"):
        env = (PyEnv if attempt == "py" else CppEnv)(engine, record_config(new), new["auto_single"])
        obs, passive, it, script, choices = env.reset(), Responder(seed), iter(opp_choices), [], []
        turn = (lambda: env.duel.turn) if attempt == "py" else (lambda: int(obs["global"][2]))
        while obs is not None and turn() <= 1 and len(choices) < 600:
            if env.player == 0:
                ch = next(it, None)
                if ch is None:
                    break
                if attempt == "cpp":
                    script.append([_fp(obs), ch])
            else:
                ch = passive.passive(obs)
            choices.append(int(ch))
            try:
                obs = env.step(int(ch))
            except Exception:  # noqa: BLE001 - an illegal choice: our hand changed the opponent's options
                return None
        if attempt == "py" and _field_codes(env, 0) != want:
            return None
    new["choices"] = choices
    st = dags.start_record(new, 0, "line1")
    st.update(kind="first", line_id=rec["line_id"], our_seat=1, our_deck=our_name, script=script, config=new["config"],
              auto_single=new["auto_single"])
    return st


def load_line_starts(path: str | Path) -> tuple[list[tuple], list[dict]]:
    """Line-starts file -> (set_starts tuples, per-start {our_seat, script, kind, line_id})."""
    starts, meta = [], []
    for raw in Path(path).read_text().splitlines():
        if raw.strip():
            r = json.loads(raw)
            starts.append(dags.to_set_starts(r))
            meta.append({"our_seat": int(r["our_seat"]), "script": [(int(a), int(b)) for a, b in r["script"]],
                         "kind": r.get("kind", "?"), "line_id": r.get("line_id", "")})
    if not starts:
        raise ValueError(f"no line starts in {path}")
    return starts, meta


class LineTracker:
    """Per-env script position. `begin(i, meta)` at episode start (meta None = not a line episode); `next(i, actions,
    mask)` for an opponent decision returns the scripted choice, or None once the script is over or diverged."""

    def __init__(self, n: int) -> None:
        self.meta: list[dict | None] = [None] * n
        self.pos = np.zeros(n, dtype=np.int64)
        self.on = np.zeros(n, dtype=bool)
        self.followed = 0  # scripted decisions played
        self.broken = 0  # scripts ended by a fingerprint mismatch (our bot changed the line)
        self.completed = 0  # scripts played to the end
        self.broken0 = 0  # mismatches at the script's first decision: the start is the opponent's decision, our bot
        #                   has not acted yet, so these can only be fingerprint/replay bugs (train_rl fails on them)

    def begin(self, i: int, meta: dict | None) -> None:
        self.meta[i] = meta
        self.pos[i] = 0
        self.on[i] = meta is not None and bool(meta["script"])

    def next(self, i: int, actions, mask) -> int | None:
        if not self.on[i]:
            return None
        script = self.meta[i]["script"]
        fp, ch = script[self.pos[i]]
        if fingerprint(actions, mask) != fp:
            self.on[i] = False
            self.broken += 1
            self.broken0 += int(self.pos[i] == 0)
            return None
        self.pos[i] += 1
        self.followed += 1
        if self.pos[i] >= len(script):
            self.on[i] = False
            self.completed += 1
        return ch
