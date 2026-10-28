"""Rust core shim (`decklight._ygoenv_rs`) + `Env.clone()`: experiment R1.

Skipped when the Rust extension is not built (`scripts/build_rust_core.sh` +
`YGO_CORE=rust scripts/build_ext.sh`). CPU only, no model, choices by a
seeded `random.Random`.

NOTE: `decklight._ygoenv` and `decklight._ygoenv_rs` are two pybind modules
binding the same C++ types, so they cannot be imported in one process (pybind
reports "type already registered"). This module therefore skips when the
default module is already loaded (e.g. combined runs where another test got
there first); run it standalone for the full Rust checks. The default-core
check runs in a subprocess for the same reason.
"""

import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

if "decklight._ygoenv" in sys.modules:
    pytest.skip(
        "default decklight._ygoenv already loaded: the two pybind modules "
        "cannot share a process; run this file standalone",
        allow_module_level=True,
    )

os.environ["YGO_CORE"] = "rust"  # before the cpp_env import below
rs = pytest.importorskip(
    "decklight._ygoenv_rs",
    reason="Rust extension not built: scripts/build_rust_core.sh + YGO_CORE=rust scripts/build_ext.sh",
)

from decklight.decks.ydk import load_ydk
from decklight.engine import cpp_env as cpp_env_mod
from decklight.engine.duel import DuelConfig, Engine

assert cpp_env_mod._ygoenv is rs, "cpp_env did not pick up the Rust module"

ROOT = Path(__file__).resolve().parents[2]
DECK0 = load_ydk(ROOT / "decks/pool/snake-eye.ydk")
DECK1 = load_ydk(ROOT / "decks/pool/branded.ydk")
MAX_STEPS = 100_000


@pytest.fixture(scope="module")
def engine():
    return Engine()


def mk_env(engine, seed, swap=False):
    return cpp_env_mod.CppEnv(engine, DuelConfig(DECK1 if swap else DECK0, DECK0 if swap else DECK1, seed=seed))


def obs_equal(a, b):
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def play(env, rng):
    """Random game to the end; returns the choice list."""
    obs, choices = env.reset(), []
    for _ in range(MAX_STEPS):
        if obs is None:
            break
        n = int(obs["action_mask"].sum())
        assert n > 0
        i = rng.randrange(n)
        choices.append(i)
        obs = env.step(i)
    else:
        raise AssertionError("game did not finish")
    assert not env.errors, env.errors
    assert env.winner in (0, 1, 2)
    return choices


def test_random_games_finish_without_errors(engine):
    resyncs = 0
    for game in range(10):
        env = mk_env(engine, seed=1000 + game, swap=bool(game % 2))
        play(env, random.Random(5000 + game))
        resyncs += env.knowledge_resyncs
    print(f"\nrandom_games_ok=10 resyncs_total={resyncs}")
    # The patched MSG_SHUFFLE_HAND (face-up bit + old indices) is parsed by Duel::track_message.
    assert resyncs == 0


def test_clone_aa_matches_prefix_replay(engine):
    """Clone A/A: original, clone and prefix-replay agree on obs, option counts and winner."""
    checks = 0
    for game in range(3):
        seed, swap = 2000 + game, bool(game % 2)
        full = play(mk_env(engine, seed, swap), random.Random(6000 + game))
        ks = [k for k in (0, 3, 10, 30) if k < len(full)]
        assert ks
        for k in ks:
            prefix = full[:k]
            e1 = mk_env(engine, seed, swap)
            obs = e1.reset()
            for i in prefix:
                obs = e1.step(i)
            assert obs is not None
            c = e1.clone()
            e3 = mk_env(engine, seed, swap)
            assert obs_equal(obs, e3.replay_from_reset(prefix))
            assert e1.player == c.player == e3.player
            rng = random.Random(7000 + game * 100 + k)
            cont = []
            while obs is not None:
                n = int(obs["action_mask"].sum())
                i = rng.randrange(n)
                cont.append(i)
                obs_c = c.step(i)
                obs = e1.step(i)
                assert obs_equal(obs, obs_c), f"game {game} k={k}: clone obs differs"
                assert (obs is None) or int(obs["action_mask"].sum()) == int(
                    obs_c["action_mask"].sum()
                ), f"game {game} k={k}: legal option counts differ"
                checks += 1
            assert e1.winner == c.winner
            e4 = mk_env(engine, seed, swap)
            assert e4.replay_from_reset(prefix + cont) is None
            assert e4.winner == e1.winner
    print(f"\nclone_aa_checks={checks}")
    assert checks > 0


def test_clone_is_independent(engine):
    """Stepping the clone leaves the original's next observation unchanged."""
    for game in range(3):
        e1 = mk_env(engine, seed=3000 + game)
        obs, prefix = e1.reset(), []
        rng = random.Random(8000 + game)
        for _ in range(5):
            i = rng.randrange(int(obs["action_mask"].sum()))
            prefix.append(i)
            obs = e1.step(i)
        assert obs is not None
        c = e1.clone()
        # Step the clone a few times (it diverges); the original must not notice.
        rc = random.Random(9000 + game)
        tmp = c.reset()
        for _ in range(3):
            if tmp is None:
                break
            tmp = c.step(rc.randrange(int(tmp["action_mask"].sum())))
        i = rng.randrange(int(obs["action_mask"].sum()))
        got = e1.step(i)
        ref = mk_env(engine, seed=3000 + game)
        ref.replay_from_reset(prefix)
        assert obs_equal(got, ref.step(i)), f"game {game}: original changed by clone steps"


def test_clone_outlives_original(engine):
    """A clone keeps playing after its original (and sibling clones) are gone: each clone
    logs into its own Duel (OCG_DuelClone's payload3), never into a freed one."""
    for game in range(3):
        seed = 4000 + game
        full = play(mk_env(engine, seed), random.Random(10000 + game))
        k = min(20, len(full) - 1)
        root = mk_env(engine, seed)
        root.replay_from_reset(full[:k])
        clones = [root.clone() for _ in range(4)]
        del root
        keep = clones[1]
        del clones
        obs, rest = keep.step(full[k]), full[k + 1 :]
        for i in rest:
            obs = keep.step(i)
        assert obs is None and not keep.errors
        ref = mk_env(engine, seed)
        assert ref.replay_from_reset(full) is None
        assert keep.winner == ref.winner


def test_card_left_out_of_the_build_is_an_error(engine):
    """A card outside YGO_CARDS_ONLY would play without effects: the shim reports it as an error."""
    from dataclasses import replace

    from decklight.engine.resources import load_cards

    out = subprocess.run([sys.executable, str(ROOT / "scripts/rust_card_codes.py")], capture_output=True, text=True,
                         check=True).stdout
    built = {int(t[1:]) for t in out.split()}
    cards = load_cards()
    code = next(c for c in sorted(cards) if c not in built and not cards[c].alias
                and cards[c].type & 0x21 == 0x21 and not cards[c].type & 0x4000)  # effect monster, not a token
    deck = replace(DECK0, main=(code, *DECK0.main[1:]))
    env = cpp_env_mod.CppEnv(engine, DuelConfig(deck, DECK1, seed=1))
    env.reset()
    assert any(f"card {code} has no compiled script" in e for e in env.errors), env.errors


def test_location_query_returns_cardids(engine):
    """QUERY_CARDID: the Rust duel reports non-zero card ids."""
    env = mk_env(engine, seed=42)
    env.reset()
    cards = env.cards()
    assert len(cards["cardid"]) > 0
    assert bool((cards["cardid"] != 0).any()), "no non-zero card ids from the Rust duel"


def test_cpp_default_clone_raises():
    """The default C++ module still imports, and its `clone()` raises RuntimeError
    (in a subprocess: the two extensions cannot share a process)."""
    root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent("""\
        from decklight.decks.ydk import load_ydk
        from decklight.engine import cpp_env
        from decklight.engine.duel import DuelConfig, Engine
        e = Engine()
        env = cpp_env.CppEnv(e, DuelConfig(load_ydk('decks/pool/snake-eye.ydk'),
            load_ydk('decks/pool/branded.ydk'), seed=7))
        env.reset()
        try:
            env.clone()
        except RuntimeError:
            raise SystemExit(0)
        raise SystemExit('clone() did not raise')
        """)
    env = dict(os.environ)
    env.pop("YGO_CORE", None)
    p = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )
    assert p.returncode == 0, f"default clone() check failed: {p.returncode}\n{p.stderr[-2000:]}"
