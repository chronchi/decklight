"""Engine-message cues for the web client (server side).

Plays full random games on 3 pool decks (fast: `PyEnv` directly, no sockets)
and asserts over `web.cues.build_cues`: every cue type is known; every
`activate` link is eventually solving/negated/disabled (unless the duel ends
first); `damage.lp_after` matches an independent LP tally that itself matches
the board; no opponent hidden code ever appears in a cue; turns/phases come in
order; move/summon/activate/attack all occur. Plus a golden test on the
Mitsurugi fixture and an incremental (`since`) slicing test.
"""

import json
import random
import struct
from collections import Counter
from pathlib import Path

from decklight.decks import load_ydk
from decklight.engine import constants as C
from decklight.engine.duel import DuelConfig
from decklight.engine.env import PyEnv
from decklight.engine.obs import Board
from decklight.engine.query import lps
from decklight.web.cues import KNOWN_CUE_TYPES, build_cues, cues_for_trace

FIXTURES = Path("tests/fixtures/decks")
GAMES = [(1000, "BlueEyes", "SkyStrikerAce"), (1001, "Branded", "SkyStrikerAce"),
         (1002, "SkyStrikerAce", "BlueEyes")]  # fmt: skip
GOLDEN = Path("tests/fixtures/web/mitsurugi-20261016")

REF_TYPES = {"move", "draw", "summon", "target", "attack", "battle"}


def _hidden_sets(env: PyEnv, viewer: int) -> set[int]:
    """Opponent codes in hidden zones (hand/face-down field/deck/face-down
    banished/face-down extra), mirroring the client-visibility test."""
    opp = 1 - viewer
    visible = Board(env.duel, viewer).visible_codes()
    hidden = set()
    for loc in (C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED, C.LOCATION_EXTRA):
        for c in env.duel.query_location(opp, loc):
            if c is not None and c.code and c.code not in visible:
                hidden.add(c.code)
    return hidden


def _play(engine, seed: int, a: str, b: str):
    """A full random game plus per-step (trace length, hidden sets) checkpoints."""
    decks = {p.stem: load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))}
    cfg = DuelConfig(decks[a], decks[b], seed=seed)
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    rng = random.Random(seed)
    ckpts = [(len(env.duel.trace), _hidden_sets(env, 0), _hidden_sets(env, 1))]
    steps = 0
    while not env.done:
        env.step(rng.randrange(len(env.opts)))
        steps += 1
        assert steps < 20000, "random game did not finish"
        ckpts.append((len(env.duel.trace), _hidden_sets(env, 0), _hidden_sets(env, 1)))
    assert env.winner in (0, 1)
    return env, ckpts


def _refs(cue: dict) -> list[dict]:
    t = cue["t"]
    if t == "move":
        return [cue["card"]]
    if t == "draw":
        return cue["cards"]
    if t == "summon":
        return [cue["card"]]
    if t == "target":
        return ([] if cue["source"] is None else [cue["source"]]) + cue["cards"]
    if t in ("attack", "battle"):
        return [cue["attacker"]] + ([] if cue["target"] is None else [cue["target"]])
    return []


def _lp_tally(trace, lp0: int = 8000) -> list[int]:
    """Independent LP track from the raw messages (damage/recover/pay/lpupdate)."""
    lp = [lp0, lp0]
    for raw in trace:
        t = raw[0]
        if t in (C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_PAY_LPCOST):
            p, amount = raw[1], struct.unpack_from("<I", raw, 2)[0]
            if t == C.MSG_RECOVER:
                lp[p] += amount
            else:
                lp[p] = max(0, lp[p] - amount)
        elif t == C.MSG_LPUPDATE:
            p, val = raw[1], struct.unpack_from("<I", raw, 2)[0]
            lp[p] = val - (1 << 32) if val >= 1 << 31 else val
    return lp


def _check_game(env: PyEnv, ckpts) -> tuple[Counter, Counter]:
    """All per-game cue invariants; returns (cue-type counts, skipped counts)."""
    trace = env.duel.trace
    types, skipped = Counter(), Counter()
    seen = set()
    for viewer in (0, 1):
        cues, new_len = build_cues(env, viewer)
        assert new_len == len(trace)
        assert {c["t"] for c in cues} <= KNOWN_CUE_TYPES
        types.update(c["t"] for c in cues)
        seen.update(c["t"] for c in cues)
        # Chains: every activate link resolves (or the duel ends first).
        outs: dict[int, list] = {}
        for c in cues:
            if c["t"] in ("chain_solving", "chain_negated", "chain_disabled", "chain_solved"):
                outs.setdefault(c["link"], []).append((c["t"], c["n"]))
        wins = [c["n"] for c in cues if c["t"] == "win"]
        for c in cues:
            if c["t"] != "activate":
                continue
            after = [o for o in outs.get(c["link"], []) if o[1] > c["n"]]
            assert after or any(w > c["n"] for w in wins), f"link {c['link']} at {c['n']} never resolves"
        # Turns count 1, 2, ...; phases are the brief's labels.
        turns = [c["turn"] for c in cues if c["t"] == "turn"]
        assert turns == list(range(1, len(turns) + 1)) and turns
        assert {c["phase"] for c in cues if c["t"] == "phase"} <= {
            "Draw",
            "Standby",
            "Main 1",
            "Battle",
            "Main 2",
            "End",
        }
        # Cue order follows the trace.
        assert [c["n"] for c in cues] == sorted(c["n"] for c in cues)
        # No opponent hidden code in any cue. Announced codes (activations,
        # reveals) are public by the rules from their cue on; everything else
        # must be visible in the first post-cue board state. `announced` grows
        # in cue order, so showing a code before it is announced still fails.
        announced = set()
        bad = []
        for c in cues:
            if c["t"] not in REF_TYPES:
                if c["t"] == "activate" and c["card"]["code"]:
                    announced.add(c["card"]["code"])
                for r in c.get("cards", []):
                    if r["code"]:
                        announced.add(r["code"])
                continue
            hidden = ckpts[-1][1 + viewer]
            for tl, h0, h1 in ckpts:
                if tl > c["n"]:
                    hidden = (h0, h1)[viewer]
                    break
            for r in _refs(c):
                if r["code"] and r["code"] in hidden and r["code"] not in announced:
                    bad.append((c["n"], c["t"], r["code"], r["name"]))
        assert not bad, f"hidden info leaked to viewer {viewer}: {bad[:5]}"
    # LP: the independent tally matches the board, and every lp_after matches it.
    assert _lp_tally(trace) == list(lps(env.duel.query_field()))
    lp = [env.config.lp, env.config.lp]
    cues, _ = build_cues(env, 0)
    for c in cues:
        if c["t"] not in ("damage", "recover", "pay"):
            continue
        raw = trace[c["n"]]
        p, amount = raw[1], struct.unpack_from("<I", raw, 2)[0]
        me = p == 0
        if c["t"] == "recover":
            lp[p] += amount
        else:
            lp[p] = max(0, lp[p] - amount)
        assert c["player"] == ("me" if me else "opp") and c["amount"] == amount
        assert c["lp_after"] == lp[p], c
    _, _, sk = cues_for_trace(trace, viewer=0, lp0=env.config.lp, cards=env.duel.engine.cards)
    skipped.update(sk)
    return types, skipped


def test_cue_games(engine):
    types, skipped = Counter(), Counter()
    for seed, a, b in GAMES:
        env, ckpts = _play(engine, seed, a, b)
        t, s = _check_game(env, ckpts)
        types.update(t)
        skipped.update(s)
    for need in ("move", "summon", "activate", "attack"):
        assert types[need] > 0, f"no {need} cue across the test games"
    print(f"\ncue types: {dict(sorted(types.items()))}")
    print(f"skipped: {dict(skipped.most_common(10))}")


def test_cues_incremental(engine):
    """Two slices (0..k, k..end) concatenate to the full cue list (viewer 1)."""
    env, _ = _play(engine, *GAMES[0])
    full, end = build_cues(env, 1)
    k = len(env.duel.trace) // 2
    first, mid = build_cues(env, 1, since=0)
    assert mid == end
    second, end2 = build_cues(env, 1, since=k)
    assert end2 == end
    assert [c for c in full if c["n"] < k] == first[: len([c for c in full if c["n"] < k])]
    assert [c for c in full if c["n"] >= k] == second


def test_golden_fixture(engine):
    """The Mitsurugi record replays to its trace; first cues verified by hand."""
    rec = json.loads((GOLDEN / "game.jsonl").read_text())
    human, bot = load_ydk(GOLDEN / "human.ydk"), load_ydk(GOLDEN / "bot.ydk")
    cfg = (
        DuelConfig(human, bot, seed=rec["seed"])
        if rec["human_seat"] == 0
        else DuelConfig(bot, human, seed=rec["seed"])
    )
    env = PyEnv.replay(engine, cfg, rec["choices"], auto_single=False)
    cues, _ = build_cues(env, rec["human_seat"])
    assert {c["t"] for c in cues} <= KNOWN_CUE_TYPES
    # Opening draws come before the first turn; the opponent's hand is hidden.
    assert cues[0]["t"] == "draw" and cues[0]["player"] == "me"
    assert cues[0]["count"] == 5 and all(r["code"] for r in cues[0]["cards"])
    assert cues[1]["t"] == "draw" and cues[1]["player"] == "opp"
    assert all(r["code"] == 0 and r["name"] == "" for r in cues[1]["cards"])
    assert cues[2] == {"t": "turn", "n": cues[2]["n"], "turn": 1, "player": "me"}
    # Saji is normal summoned from the hand, then activates (link 1) with its
    # search text; Impermanence (link 2) answers and disables it.
    summons = [c for c in cues if c["t"] == "summon"]
    assert summons[0]["card"]["name"] == "Mitsurugi no Mikoto, Saji"
    assert summons[0]["how"] == "normal"
    assert summons[0]["from"]["loc"] == "hand"
    first = next(c for c in cues if c["t"] == "activate")
    assert first["link"] == 1 and "Mitsurugi" in (first["effect"] or "")
    assert any(c["t"] == "chain_disabled" and c["link"] == 1 for c in cues)
    assert any(c["t"] == "target" and c["source"] is not None for c in cues)
