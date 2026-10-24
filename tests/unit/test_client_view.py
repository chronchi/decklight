"""Client-view tests (brief §Tests): the stream seat vs the engine seat.

1. The `obs.py` refactor keeps PyEnv observations byte-identical (vs CppEnv, 2 games).
2. `StreamView` on a real recorded game, both seats: option lists identical at every
   decision, `cards`/`global`/`chain`/`actions` exact, response bytes identical, and
   the stream's final MSG_WIN winner = the record's winner.
3. `history_from_message` on hand-built messages (one per mapped type + skip cases).
4. `make_agent("raw")` registered (model load mocked); `check_needs` refuses engines.
"""

import random
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from decklight.decks import load_ydk
from decklight.edopro import filter as F
from decklight.edopro import protocol as P
from decklight.edopro.product_bot import product_to_host
from decklight.engine import constants as C
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.play import agents as AG
from decklight.play.stream import StreamState, StreamView, deck_card_state, history_from_message
from decklight.play.view import EngineView

ROOT = Path(__file__).resolve().parents[2]
GAMES = ROOT / "runs/ft-curve-20261022-1928/card-val/base/games-0.jsonl"


def _deck(name: str):
    for d in ("gauntlet", "pool", "newcards"):
        for p in (ROOT / "decks" / d).glob("*.ydk"):
            if p.stem.lower() == name.lower():
                return load_ydk(p)
    raise FileNotFoundError(name)


# --- 1. obs refactor: PyEnv vs CppEnv, byte-identical ---


def _replay(env, seed, n=60):
    rng = random.Random(seed)
    obs = env.reset()
    out = []
    while obs is not None and len(out) < n:
        out.append({k: np.ascontiguousarray(v) for k, v in obs.items()})
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        obs = env.step(rng.choice(legal))
    return out, env.winner


def test_obs_refactor_byte_identical():
    CppEnv = pytest.importorskip("decklight.engine.cpp_env").CppEnv
    engine = Engine()
    cfgs = [
        DuelConfig(_deck("blue-eyes"), _deck("blackwing"), seed=7022),
        DuelConfig(_deck("blackwing"), _deck("blue-eyes"), seed=7165),
    ]
    n = 0
    for cfg in cfgs:
        po, pw = _replay(PyEnv(engine, cfg, auto_single=True), 7)
        co, cw = _replay(CppEnv(engine, cfg, True), 7)
        assert pw == cw
        assert len(po) == len(co) and len(po) > 10
        for a, b in zip(po, co):
            keys = set(a) | set(b) - {"priv_code"}  # privileged critic channel, not an obs
            assert set(a) | {"priv_code"} == set(b) | {"priv_code"}
            for k in keys:
                assert k in a and k in b, k
                assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
                assert (a[k] == b[k]).all(), k
        n += len(po)
    assert n > 100


# --- 2. StreamView on a real recorded game ---


def _resolve_cfg(engine, rec):
    ygoagent = "ya_decisions" in rec or "deck_s" in rec
    order = ("gauntlet", "pool", "newcards") if ygoagent else ("pool", "newcards", "gauntlet")
    cands = []
    for name in rec["decks"]:
        paths = [p for d in order for p in (ROOT / "decks" / d).glob("*.ydk")
                 if p.stem.lower() == name.lower()]
        assert paths, f"deck {name} not found"
        cands.append(paths)
    for p0 in cands[0]:
        for p1 in cands[1]:
            cfg = DuelConfig(load_ydk(p0), load_ydk(p1), seed=rec["seed"])
            env = PyEnv(engine, cfg, auto_single=True)
            try:
                obs = env.reset()
                for c in rec["choices"]:
                    if obs is None:
                        break
                    obs = env.step(int(c))
                if (obs is None and env.done and env.winner == rec["winner"]
                        and [e.actor for e in env.history] == rec["actors"]):
                    return cfg
            except (IndexError, ValueError):
                continue
    raise AssertionError(f"no deck combo replays game {rec.get('game_idx')}")


def _stream_game(rec, seat):
    engine = Engine()
    cfg = _resolve_cfg(engine, rec)
    truth = PyEnv(engine, cfg, auto_single=True)
    obs = truth.reset()
    truth_obs = []
    last = None
    if obs is not None:
        if truth.player == seat:
            truth_obs.append(obs)
        last = truth.dec
    for c in rec["choices"]:
        obs = truth.step(int(c))
        if obs is not None and truth.dec is not last:
            if truth.player == seat:
                truth_obs.append(obs)
            last = truth.dec
    assert truth.done and truth.winner == rec["winner"]

    host_choices = product_to_host(engine, cfg, rec["choices"])
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    deck = cfg.deck0 if seat == 0 else cfg.deck1
    st = StreamView({"main": list(deck.main), "extra": list(deck.extra)}, engine.cards, seat=seat)
    n0, e0 = len(cfg.deck0.main), len(cfg.deck0.extra)
    n1, e1 = len(cfg.deck1.main), len(cfg.deck1.extra)
    st.feed(P.build_start(cfg.lp, n0, e0, n1, e1, seat))
    for s in (0, 1):
        p = F.refresh_packet(env.duel, ("loc", s, C.LOCATION_EXTRA), seat)
        if p is not None:
            st.feed(p)
    sent = 0
    ti, last_dec = 0, None
    compared = 0
    i = 0
    while not env.done:
        new = env.duel.trace[sent:]
        sent = len(env.duel.trace)
        for raw in new:
            for ref in F.before_refreshes(raw):
                p = F.refresh_packet(env.duel, ref, seat)
                if p is not None:
                    st.feed(p)
            msg = F.filtered_message(raw, seat)
            if msg is not None:
                st.feed(msg)
            for ref in F.after_refreshes(raw):
                p = F.refresh_packet(env.duel, ref, seat)
                if p is not None:
                    st.feed(p)
        if env.done:
            break
        assert env.player in (0, 1)
        if env.player == seat and len(env.opts) > 1 and env.dec is not last_dec:
            assert st.decision is not None and st.needs_answer
            assert [o.kind for o in st.opts] == [o.kind for o in env.opts]
            ta, sa = truth_obs[ti], st.obs()
            for k in ("cards", "card_mask", "global", "chain", "chain_mask",
                      "actions", "action_mask"):
                assert ta[k].shape == sa[k].shape, k
                assert (ta[k] == sa[k]).all(), f"{k} differs at step {i}"
            acts = ta["actions"]
            assert (acts == sa["actions"]).all() and (ta["action_mask"] == sa["action_mask"]).all()
            ti += 1
            compared += 1
        last_dec = env.dec
        c = int(host_choices[i])
        if env.player == seat and st.needs_answer:
            st.take(c)
        n_resp = len(env.duel.responses)
        was_seat = env.player == seat
        env.step(c)
        i += 1
        if was_seat and len(env.duel.responses) > n_resp:
            assert st._resp == env.duel.responses[-1], f"response differs at step {i - 1}"
    assert ti == len(truth_obs), f"consumed {ti} of {len(truth_obs)}"
    assert env.duel.winner == rec["winner"]
    # The WIN message rides the final pump, after the last step.
    for raw in env.duel.trace[sent:]:
        msg = F.filtered_message(raw, seat)
        if msg is not None:
            st.feed(msg)
    assert st.winner == rec["winner"]
    assert compared > 0
    return compared


def test_stream_parity_recorded_game():
    if not GAMES.is_file():
        pytest.skip("recorded games absent")
    import json

    recs = [json.loads(x) for x in GAMES.read_text().splitlines() if x.strip()]
    rec = next(r for r in recs if not r.get("truncated"))
    total = 0
    for seat in (0, 1):
        total += _stream_game(rec, seat)
    assert total > 10


# --- 3. history_from_message on hand-built messages ---


def _loc(con, loc, seq, pos=1):
    return bytes([con, loc]) + struct.pack("<I", seq) + struct.pack("<I", pos)


def _state():
    cards = {1: None}
    return StreamState([1], [], cards, seat=0)


def test_history_from_message_table():
    st = _state()
    st.turn, st.phase, st.turn_player = 3, 4, 1

    raw = bytes([C.MSG_SUMMONING]) + struct.pack("<I", 100) + _loc(1, C.LOCATION_MZONE, 2, 1)
    rows = history_from_message(st, raw)
    assert [(r.row[0], r.actor, r.public) for r in rows] == [(1, 1, True), (19, 1, False)]
    assert rows[0].row[2] == 100 and rows[1].row[6] == C.MSG_SELECT_PLACE

    raw = bytes([C.MSG_SPSUMMONING]) + struct.pack("<I", 0) + _loc(1, C.LOCATION_MZONE, 0, 1)
    rows = history_from_message(st, raw)
    assert [r.row[0] for r in rows] == [2, 19]  # facedown sp summon: code hidden

    raw = bytes([C.MSG_FLIPSUMMONING]) + struct.pack("<I", 101) + _loc(1, C.LOCATION_MZONE, 0, 1)
    assert [(r.row[0], r.row[2]) for r in history_from_message(st, raw)] == [(3, 101)]

    raw = bytes([C.MSG_SET]) + struct.pack("<I", 0) + _loc(1, C.LOCATION_SZONE, 3, 10)
    rows = history_from_message(st, raw)
    assert [(r.row[0], r.row[2], r.public) for r in rows] == [(5, 0, False), (19, 0, False)]

    desc = (7 << 20) | 9
    raw = (bytes([C.MSG_CHAINING]) + struct.pack("<I", 102) + _loc(1, C.LOCATION_SZONE, 1, 1)
           + bytes([1, C.LOCATION_SZONE]) + struct.pack("<I", 2) + struct.pack("<Q", desc)
           + struct.pack("<I", 1))
    (row,) = history_from_message(st, raw)
    assert (row.row[0], row.row[2], row.row[3], row.row[4], row.public) == (6, 102, 7, 9, True)

    raw = bytes([C.MSG_ATTACK]) + _loc(1, C.LOCATION_MZONE, 0, 1) + _loc(0, 0, 0, 0)
    (row,) = history_from_message(st, raw)
    assert (row.row[0], row.row[5], row.public) == (7, 1, True)  # direct attack

    raw = bytes([C.MSG_POS_CHANGE]) + struct.pack("<I", 103) + bytes([1, C.LOCATION_MZONE, 2, 1, 4])
    (row,) = history_from_message(st, raw)
    assert (row.row[0], row.row[2]) == (3, 103)

    raw = bytes([C.MSG_NEW_PHASE]) + struct.pack("<H", C.PHASE_BATTLE_START)
    (row,) = history_from_message(st, raw)
    assert (row.row[0], row.actor, row.row[6]) == (8, 1, C.MSG_SELECT_IDLECMD)
    raw = bytes([C.MSG_NEW_PHASE]) + struct.pack("<H", C.PHASE_MAIN2)
    (row,) = history_from_message(st, raw)
    assert (row.row[0], row.row[6]) == (9, C.MSG_SELECT_BATTLECMD)

    # skip cases: our own summon, chain-open summon/poschange, our turn phase, draws
    assert history_from_message(st, bytes([C.MSG_SUMMONING]) + struct.pack("<I", 1) + _loc(0, 4, 0)) == []
    st.chain.append(None)
    assert history_from_message(st, bytes([C.MSG_SUMMONING]) + struct.pack("<I", 1) + _loc(1, 4, 0)) == []
    assert history_from_message(st, bytes([C.MSG_POS_CHANGE]) + struct.pack("<I", 1) + bytes([1, 4, 0, 1, 4])) == []
    st.chain.clear()
    st.turn_player = 0
    assert history_from_message(st, bytes([C.MSG_NEW_PHASE]) + struct.pack("<H", C.PHASE_BATTLE_START)) == []
    assert history_from_message(st, bytes([C.MSG_DRAW, 1, 0, 0, 0, 0])) == []


# --- 4. agents ---


def test_raw_agent_registered(monkeypatch):
    assert "raw" in AG.AGENTS and "random" in AG.AGENTS

    class FakePolicy:
        def __init__(self, z):
            self.z = z

        def logits(self, batch):
            n = next(iter(batch.values())).shape[0]
            return np.tile(np.asarray(self.z, np.float32), (n, 1))

    monkeypatch.setattr(
        "decklight.search.product.resolve_product_paths", lambda *a: ("ckpt", None, "emb")
    )
    monkeypatch.setattr(
        "decklight.search.product.load_policy", lambda ckpt, emb, device="cpu", **k: FakePolicy([0.1, 0.5, 0.3])
    )
    agent = AG.make_agent("raw", ckpt="x", emb="y")
    assert agent.needs == frozenset()

    class FakeView:
        seat = 0

        def __init__(self, n):
            self._n = n

        @property
        def opts(self):
            from decklight.engine.actions import Kind, Option

            return [Option(Kind.SELECT) for _ in range(self._n)]

        def obs(self):
            return {
                "cards": np.zeros((160, 21), np.int32),
                "action_mask": np.array([True, False, True] + [False] * 61),
            }

    assert agent.act(FakeView(3)) == 2  # argmax over legal options (index 1 masked out)
    assert agent.rps() in (1, 2, 3) and agent.go_first() is True

    # check_needs refuses an agent whose needs are not provided
    class Needy:
        name = "needy"
        needs = frozenset({"engine"})

    with pytest.raises(AG.MissingCapability):
        AG.check_needs(Needy(), frozenset(), "edopro client")
    AG.check_needs(AG.make_agent("random"), frozenset(), "edopro client")


def test_engine_view_passthrough():
    engine = Engine()
    cfg = DuelConfig(_deck("blue-eyes"), _deck("blackwing"), seed=11)
    env = PyEnv(engine, cfg, auto_single=True)
    env.reset()
    view = EngineView(env)
    assert view.seat == env.player
    assert view.decision is env.pending_decision()
    assert [o.kind for o in view.opts] == [o.kind for o in env.opts]
    ref = env._observe()
    got = view.obs()
    assert set(got) == set(ref)
    for k in ref:
        assert got[k].shape == ref[k].shape and (got[k] == ref[k]).all(), k
    assert view.needs_answer and not view.done
    view.take(0)
    assert view.seat == env.player


def test_deck_card_state_matches_engine():
    from collections import Counter

    from decklight.engine.obs import STATUS_OBS_MASK

    engine = Engine()
    for name in ("blue-eyes", "blackwing"):
        deck = _deck(name)
        cfg = DuelConfig(deck, deck, seed=7)
        env = PyEnv(engine, cfg, auto_single=True)
        env.reset()
        hand = Counter(c.code for c in env.duel.query_location(0, C.LOCATION_HAND) if c)
        rest = list((Counter(deck.main) - hand).elements())
        want = sorted(
            (c.code, c.type, c.level, c.rank, c.attribute, c.race, c.attack, c.defense,
             c.base_attack, c.base_defense, c.status & STATUS_OBS_MASK)
            for c in env.duel.query_location(0, C.LOCATION_DECK) if c
        )
        got = sorted(
            (c.code, c.type, c.level, c.rank, c.attribute, c.race, c.attack, c.defense,
             c.base_attack, c.base_defense, c.status & STATUS_OBS_MASK)
            for c in (deck_card_state(code, engine.cards, 0, i) for i, code in enumerate(rest))
        )
        assert got == want


def test_refresh_packet_matches_host_query():
    engine = Engine()
    cfg = DuelConfig(_deck("blue-eyes"), _deck("blackwing"), seed=7)
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    for ref in (("loc", 0, C.LOCATION_MZONE), ("loc", 1, C.LOCATION_HAND),
                ("loc", 0, C.LOCATION_EXTRA), ("single", 0, C.LOCATION_HAND, 0)):
        for viewer in (0, 1):
            p = F.refresh_packet(env.duel, ref, viewer)
            cards = env.duel.query_location(ref[1], ref[2])
            if ref[0] == "single" and (ref[3] >= len(cards) or cards[ref[3]] is None):
                assert p is None
                continue
            assert p is not None and p[0] in (C.MSG_UPDATE_DATA, C.MSG_UPDATE_CARD)
            public = viewer != ref[1]
            if ref[0] == "loc":
                assert p == F.encode_update_data(ref[1], ref[2], cards, public)
            else:
                assert p == F.encode_update_card(ref[1], ref[2], ref[3], cards[ref[3]], public)
