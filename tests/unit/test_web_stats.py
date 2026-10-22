"""DuelStats through the Mitsurugi fixture (see tests/unit/test_web_replay.py)."""

import json
from pathlib import Path

import pytest

from decklight.decks import load_ydk
from decklight.engine import messages as M
from decklight.engine.actions import Kind
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.web.cues import build_cues
from decklight.web.stats import DuelStats
from decklight.web.view import build_view

FIXTURE = Path("tests/fixtures/web/mitsurugi-20261016")


def _replay(engine):
    rec = json.loads((FIXTURE / "game.jsonl").read_text())
    human, bot = load_ydk(FIXTURE / "human.ydk"), load_ydk(FIXTURE / "bot.ydk")
    hs = rec["human_seat"]
    cfg = DuelConfig(human, bot, seed=rec["seed"]) if hs == 0 else DuelConfig(bot, human, seed=rec["seed"])
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    st = DuelStats(hs, lp0=getattr(cfg, "lp", 8000))
    since = 0
    for choice in rec["choices"]:
        if env.done:
            break
        # resolved human chain windows (mirrors Server.on_choose / auto-pass)
        if env.player == hs and len(env.opts) > 1:
            dec = env.pending_decision()
            if dec is not None and isinstance(dec.msg, M.SelectChain):
                had = any(o.kind == Kind.ACTIVATE for o in env.opts)
                if had:
                    used = env.opts[int(choice)].kind == Kind.ACTIVATE
                    st.note_chain_window(True, used)
        env.step(int(choice))
        cues, since = build_cues(env, hs, since=since)
        st.feed_cues(cues)
    st.note_view(build_view(env, hs))
    cues_all, _ = build_cues(env, hs)
    return rec, env, st, cues_all


def test_web_stats_fixture(engine):
    rec, env, st, cues = _replay(engine)
    msg = st.stats_message()
    print("\n" + json.dumps(msg, indent=1)[:4000])

    # final LP matches the board
    view = build_view(env, rec["human_seat"])
    assert msg["lp_you"] == view["you"]["lp"]
    assert msg["lp_bot"] == view["bot"]["lp"]

    # damage totals equal the sum of the damage cues
    dmg_me = sum(c["amount"] for c in cues if c["t"] == "damage" and c["player"] == "me")
    dmg_opp = sum(c["amount"] for c in cues if c["t"] == "damage" and c["player"] == "opp")
    # dealt_you = damage to the bot, dealt_bot = damage to you
    assert msg["dealt_you"] == dmg_opp
    assert msg["dealt_bot"] == dmg_me

    # summon/activation counts equal the cue counts
    summons = [c for c in cues if c["t"] == "summon"]
    assert msg["ledger"]["you"]["summons"] + msg["ledger"]["bot"]["summons"] == len(summons)
    assert msg["ledger"]["you"]["summons_ss"] + msg["ledger"]["bot"]["summons_ss"] == sum(
        1 for c in summons if c.get("how") not in ("normal", "tribute", "flip")
    )
    acts = [c for c in cues if c["t"] == "activate"]
    assert msg["ledger"]["you"]["activations"] + msg["ledger"]["bot"]["activations"] == len(acts)

    # the fixture's negation: Saji disabled, Impermanence the cause
    assert msg["neg_were_you"] == 1  # your Saji was negated
    assert msg["neg_bot"] == 1  # the bot negated it
    texts = [(m.get("text", "") + " " + m.get("sub", "")) for m in msg["moments"]]
    assert any("negated" in t and "by Infinite Impermanence" in t for t in texts), texts

    # response windows are sane
    assert msg["windows_used"] <= msg["windows_total"]
    assert msg["windows_total"] == 2 and msg["windows_used"] == 0

    # opponent's cards seen: only codes that were ever public
    public = {
        c["card"]["code"] for c in cues if c["t"] in ("summon", "activate", "move") and c["card"]["code"]
    }
    public |= {
        r["code"] for c in cues if c["t"] in ("draw", "confirm") for r in c.get("cards", []) if r["code"]
    }
    for c in cues:
        if c["t"] == "target":
            if c.get("source") and c["source"]["code"]:
                public.add(c["source"]["code"])
            public |= {r["code"] for r in c.get("cards", []) if r["code"]}
        if c["t"] in ("attack", "battle"):
            for k in ("attacker", "target"):
                r = c.get(k)
                if isinstance(r, dict) and r.get("code"):
                    public.add(r["code"])
    for e in msg["opp_seen"]:
        assert e["code"] in public, f"opp_seen {e} was never public"

    # chart + ledger shape
    assert msg["chart_labels"][0] == "T1" and msg["chart_labels"][-1] == "now"
    assert len(msg["moments"]) <= 5
    assert msg["ledger"]["you"]["hand"] == len(view["you"]["hand"])


def _hand_cues():
    """Material Link summon (own costs) + 3 opp cards killed by my resolving link."""
    me_spell = {"code": 11, "name": "My Spell", "owner": "me"}
    me_mon = lambda code, name: {"code": code, "name": name, "owner": "me"}
    opp_mon = lambda code, name: {"code": code, "name": name, "owner": "opp"}
    return [
        {"t": "turn", "n": 0, "turn": 4},
        # my Link summon: two materials leave my field outside any resolve
        {
            "t": "move",
            "n": 1,
            "card": me_mon(21, "My L1"),
            "from": {"side": "me", "loc": "mzone", "seq": 0},
            "to": {"side": "me", "loc": "unknown", "seq": 0},
            "reason": "material",
        },
        {
            "t": "move",
            "n": 2,
            "card": me_mon(22, "My L2"),
            "from": {"side": "me", "loc": "mzone", "seq": 1},
            "to": {"side": "me", "loc": "unknown", "seq": 0},
            "reason": "material",
        },
        {
            "t": "summon",
            "n": 3,
            "card": me_mon(23, "My Link"),
            "how": "link",
            "from": {"side": "me", "loc": "extra", "seq": 0},
        },
        # my spell resolves and destroys 3 bot monsters
        {"t": "activate", "n": 4, "card": me_spell, "link": 1},
        {"t": "chain_solving", "n": 5, "link": 1},
        {
            "t": "move",
            "n": 6,
            "card": opp_mon(31, "Bot A"),
            "from": {"side": "opp", "loc": "mzone", "seq": 0},
            "to": {"side": "opp", "loc": "grave", "seq": 0},
            "reason": "destroyed",
        },
        {
            "t": "move",
            "n": 7,
            "card": opp_mon(32, "Bot B"),
            "from": {"side": "opp", "loc": "mzone", "seq": 1},
            "to": {"side": "opp", "loc": "grave", "seq": 1},
            "reason": "destroyed",
        },
        {
            "t": "move",
            "n": 8,
            "card": opp_mon(33, "Bot C"),
            "from": {"side": "opp", "loc": "mzone", "seq": 2},
            "to": {"side": "opp", "loc": "banished", "seq": 0},
            "reason": "banished",
        },
        {"t": "chain_solved", "n": 9, "link": 1},
        {"t": "chain_end", "n": 10},
        # a cost outside any resolve: my set spell sent from the field
        {
            "t": "move",
            "n": 11,
            "card": me_mon(24, "My Cost"),
            "from": {"side": "me", "loc": "szone", "seq": 0},
            "to": {"side": "me", "loc": "grave", "seq": 0},
            "reason": "sent",
        },
    ]


def test_wipe_counts_only_opponent_resolving_link():
    st = DuelStats(viewer=0)
    st.feed_cues(_hand_cues())
    msg = st.stats_message()
    # own Link materials + the outside cost never count
    assert st.field_lost == {(4, "opp"): 3}, st.field_lost
    wipes = [m for m in msg["moments"] if m.get("kind") == "wipe"]
    assert len(wipes) == 1 and wipes[0]["text"] == "Bot lost 3 cards from the field", msg["moments"]
    assert not any(m.get("text", "").startswith("You lost") for m in msg["moments"])
    # plays, not sightings: the activated spell counts 1, summoned link counts 1
    plays = {e["code"]: e["count"] for e in msg["you_used"]}
    assert plays.get(11) == 1 and plays.get(23) == 1, msg["you_used"]
    # a merely-seen opponent card keeps count 0
    st2 = DuelStats(viewer=0)
    st2.feed_cues(
        [
            {
                "t": "confirm",
                "n": 0,
                "player": "opp",
                "cards": [{"code": 99, "name": "Seen Only", "owner": "opp"}],
            }
        ]
    )
    msg2 = st2.stats_message()
    assert msg2["opp_seen"] == [{"code": 99, "name": "Seen Only", "count": 0}], msg2["opp_seen"]


CUSTOM_GAME = Path("runs/custom-20261016-2221/g11/games-0.jsonl")
CUSTOM_EXTRA = Path("data/cache/custom/zedja")


def test_web_stats_long_custom_game(monkeypatch):
    """Tearlaments vs Wiccanthrope (viewer seat 1): plays + opponent-only wipes."""
    if not CUSTOM_GAME.is_file() or not CUSTOM_EXTRA.is_dir():
        pytest.skip("custom game or extra cards missing (not in git)")
    monkeypatch.setenv("YGO_EXTRA_CARDS", str(CUSTOM_EXTRA))
    engine = Engine()
    rec = json.loads(CUSTOM_GAME.read_text().splitlines()[0])
    human = load_ydk("decks/custom/wiccanthrope.ydk")
    bot = load_ydk("decks/pool/tearlaments.ydk")
    # decks in the record are [tearlaments(seat0), wiccanthrope(seat1)]: viewer seat 1
    cfg = DuelConfig(bot, human, seed=rec["seed"])
    env = PyEnv(engine, cfg, auto_single=True)
    env.reset()
    hs = 1
    st = DuelStats(hs, lp0=getattr(cfg, "lp", 8000))
    since = 0
    for choice in rec["choices"]:
        if env.done:
            break
        env.step(int(choice))
        cues, since = build_cues(env, hs, since=since)
        st.feed_cues(cues)
    assert env.done and env.winner == rec["winner"] == 1
    st.note_view(build_view(env, hs))
    cues_all, _ = build_cues(env, hs)
    msg = st.stats_message()
    print("\nmoments: " + json.dumps(msg["moments"], indent=1))
    print("opp_seen: " + json.dumps(msg["opp_seen"], indent=1)[:3000])
    print("you_used: " + json.dumps(msg["you_used"], indent=1)[:3000])

    # damage totals equal the damage cue sums (me took 7000, opp took 8600)
    dmg_me = sum(c["amount"] for c in cues_all if c["t"] == "damage" and c["player"] == "me")
    dmg_opp = sum(c["amount"] for c in cues_all if c["t"] == "damage" and c["player"] == "opp")
    assert (dmg_me, dmg_opp) == (7000, 8600), (dmg_me, dmg_opp)
    assert msg["dealt_bot"] == dmg_me and msg["dealt_you"] == dmg_opp

    # plays, not sightings: 3 copies max, re-plays bounded (<= 6)
    for e in msg["opp_seen"] + msg["you_used"]:
        assert e["count"] <= 6, e

    # wipe moments count only non-material moves: the turn-4 "You lost 6"
    # (own Link materials + own cost) is gone, "Bot lost 3" remains
    assert not any(m.get("text", "").startswith("You lost") for m in msg["moments"])
    wipes = [m for m in msg["moments"] if m.get("kind") == "wipe"]
    assert wipes and all("Bot lost 3" in m["text"] for m in wipes), msg["moments"]
    material = [
        c
        for c in cues_all
        if c["t"] == "move"
        and c.get("reason") == "material"
        and isinstance(c.get("from"), dict)
        and c["from"].get("loc") in ("mzone", "szone", "field", "emz")
    ]
    assert len(material) >= 4, "expected own Link materials in the cues"
    # every counted loss is a non-material departure
    assert all(
        v <= sum(1 for c in cues_all if c["t"] == "move" and c.get("reason") != "material")
        for v in st.field_lost.values()
    )

    # key moments are newest-first
    turns = [m.get("turn", 0) for m in msg["moments"]]
    assert turns == sorted(turns, reverse=True), msg["moments"]


def test_winchance_moments_and_off_gate():
    """note_winchance history feeds the chart; moments between two samples carry
    "you were 61%, then 38%"; winchance_on=False leaves no trace at all."""
    st = DuelStats(0)
    st.feed_cues([{"t": "turn", "n": 0, "turn": 2}])
    st.note_winchance(2, 0.61)
    st.feed_cues([{"t": "damage", "n": 1, "player": "me", "amount": 1500, "lp_after": 6500}])
    st.note_winchance(2, 0.38)
    msg = st.stats_message()
    assert msg["winchance"] == {"p": 0.38, "turn": 2, "history": [[2, 0.61], [2, 0.38]]}
    subs = [m.get("sub", "") for m in msg["moments"]]
    assert any("you were 61%, then 38%" in s for s in subs), subs
    off = st.stats_message(winchance_on=False)
    assert "winchance" not in off
    assert all("you were" not in m.get("sub", "") for m in off["moments"]), off["moments"]
