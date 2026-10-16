"""Round-trip tests for engine/inverse.py (EDOPro play D1)."""

import random

import numpy as np

from ygo_zero.engine import responses as R
from ygo_zero.engine.duel import DuelConfig
from ygo_zero.engine.env import PyEnv
from ygo_zero.engine.inverse import MSG_NAMES, choices_for_response, decision_groups

GAMES = [  # (deck0, deck1, seed): 5 random-policy games across all 4 fixture decks
    ("BlueEyes", "Branded", 1000),
    ("Branded", "Labrynth", 1001),
    ("Labrynth", "SkyStrikerAce", 1002),
    ("SkyStrikerAce", "BlueEyes", 1003),
    ("BlueEyes", "Branded", 1004),
]

MANDATORY = {
    "MSG_SELECT_IDLECMD", "MSG_SELECT_BATTLECMD", "MSG_SELECT_CARD", "MSG_SELECT_CHAIN",
    "MSG_SELECT_YESNO", "MSG_SELECT_POSITION", "MSG_SELECT_PLACE",
}  # fmt: skip


def _play(engine, decks, d0, d1, seed, rng_seed, cap=2000):
    """A full random-policy game; returns (config, choices trimmed to completed decisions)."""
    cfg = DuelConfig(decks[d0], decks[d1], seed=seed)
    env = PyEnv(engine, cfg)
    env.reset()
    rng = random.Random(rng_seed)
    choices, complete, n_resp = [], 0, 0
    while env.dec is not None and len(choices) < cap:
        env.step(rng.randrange(len(env.opts)))
        choices.append(env.choices[-1])
        if len(env.duel.responses) > n_resp:
            n_resp, complete = len(env.duel.responses), len(choices)
    assert env.dec is None, "random game did not finish"
    return cfg, choices[:complete]


def _send(env, cfg, choices, k, seq):
    """The first new response after sending `seq` on a replay of choices[:k]."""
    probe = PyEnv.replay(env.engine, cfg, choices[:k])
    n_before = len(probe.duel.responses)
    for i in seq:
        probe.step(i)
    assert len(probe.duel.responses) > n_before, "inverse sequence produced no response"
    return probe.duel.responses[n_before]


def test_round_trip_random_games(engine, decks):
    """Every stored response inverts to the record's indices or bytes-identical ones."""
    covered, alternates, n_dec = set(), 0, 0
    for d0, d1, seed in GAMES:
        cfg, choices = _play(engine, decks, d0, d1, seed, rng_seed=seed)
        groups = decision_groups(engine, cfg, choices)
        assert sum(len(sl) for _, sl, _ in groups) == len(choices)
        env = PyEnv(engine, cfg)
        env.reset()
        k = 0
        for t, sl, resp in groups:
            if not sl:
                continue  # decided entirely by auto-single steps
            covered.add(MSG_NAMES.get(t, hex(t)))
            n_dec += 1
            assert env.dec is not None and env.dec.msg.type == t
            seq = choices_for_response(env, resp)
            assert seq is not None, (d0, d1, seed, MSG_NAMES.get(t), resp.hex())
            if list(seq) != sl:
                assert _send(env, cfg, choices, k, seq) == resp, (d0, d1, seed, sl, seq)
                alternates += 1
            for i in sl:
                env.step(i)
            k += len(sl)
        assert [r for _, _, r in groups] == list(env.duel.responses)
    missing = MANDATORY - covered
    assert not missing, f"mandatory message types not reached: {sorted(missing)}"
    print(f"\ninverse round trip: {n_dec} decisions, {alternates} alternates")
    print(f"covered: {sorted(covered)}")
    print(f"SELECT_SUM reached: {'MSG_SELECT_SUM' in covered}, "
          f"ANNOUNCE reached: {bool({'MSG_ANNOUNCE_RACE', 'MSG_ANNOUNCE_ATTRIB', 'MSG_ANNOUNCE_CARD'} & covered)}")


def test_round_trip_covers_multi_select_card(engine, decks):
    """At least one multi-pick SELECT_CARD inverts (multi-step search path)."""
    found = 0
    for d0, d1, seed in GAMES:
        cfg, choices = _play(engine, decks, d0, d1, seed, rng_seed=seed)
        env = PyEnv(engine, cfg)
        env.reset()
        k = 0
        for t, sl, resp in decision_groups(engine, cfg, choices):
            if sl and MSG_NAMES.get(t) == "MSG_SELECT_CARD" and len(sl) > 1:
                seq = choices_for_response(env, resp)
                assert seq is not None
                if list(seq) != sl:
                    assert _send(env, cfg, choices, k, seq) == resp
                found += 1
            for i in sl:
                env.step(i)
            k += len(sl)
    assert found > 0, "no multi-pick SELECT_CARD reached"


def _env_at(engine, decks, want):
    """Replay the first fixture game up to a pending decision of message `want`."""
    cfg, choices = _play(engine, decks, *GAMES[0], rng_seed=GAMES[0][2])
    env = PyEnv(engine, cfg)
    env.reset()
    for c in choices:
        if env.dec is not None and MSG_NAMES.get(env.dec.msg.type) == want:
            return env
        env.step(c)
    raise AssertionError(f"{want} never pending")


def test_invalid_response_gives_none(engine, decks):
    env = _env_at(engine, decks, "MSG_SELECT_CARD")
    assert choices_for_response(env, b"\x01\x02\x03") is None  # malformed
    assert choices_for_response(env, R.cards([63])) is None  # no such card index
    idle = _env_at(engine, decks, "MSG_SELECT_IDLECMD")
    assert choices_for_response(idle, R.CANCEL) is None  # idle has no cancel
    assert choices_for_response(idle, R.cards([0])) is None  # wrong family


def test_env_not_mutated(engine, decks):
    cfg, choices = _play(engine, decks, *GAMES[0], rng_seed=GAMES[0][2])
    groups = decision_groups(engine, cfg, choices)
    env = PyEnv(engine, cfg)
    obs0 = env.reset()
    snap = ({k: v.copy() for k, v in obs0.items()}, list(env.choices),
            len(env.duel.responses), env.dec.step, list(env.dec.selected))
    n = 0
    for t, sl, resp in groups:
        if sl:
            choices_for_response(env, resp)
            obs1 = env._observe()
            for k in snap[0]:
                assert np.array_equal(obs1[k], snap[0][k]), f"obs {k} mutated at {MSG_NAMES.get(t)}"
            assert env.choices == snap[1] and len(env.duel.responses) == snap[2]
            assert env.dec.step == snap[3] and env.dec.selected == snap[4]
            n += 1
            if n >= 10:
                break
        for i in sl:
            env.step(i)
        if env.dec is not None:
            obs0 = env._observe()
            snap = ({k: v.copy() for k, v in obs0.items()}, list(env.choices),
                    len(env.duel.responses), env.dec.step, list(env.dec.selected))
