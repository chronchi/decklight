"""Observable history (train = deploy, NOTEBOOK 2026-11-10 13:30): `obs_history` holds the viewer's own decisions and
the opponent's public events (`engine.events`), the history a live client records."""

from pathlib import Path

import numpy as np
import pytest
import torch
from test_policy_trim import _emb

from decklight.decks import load_ydk
from decklight.engine.actions import Kind
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.model.policy import PolicyNet, observable_history

ROOT = Path(__file__).resolve().parents[2]


def _games(n_games: int = 2, seed: int = 3):
    """Observations of random-legal games (PyEnv), both seats."""
    decks = [load_ydk(p) for p in sorted((ROOT / "tests/fixtures/decks").glob("*.ydk"))]
    engine, rng = Engine(), np.random.default_rng(seed)
    for g in range(n_games):
        env = PyEnv(engine, DuelConfig(decks[g % len(decks)], decks[(g + 1) % len(decks)], seed=seed + g))
        obs, n = env.reset(), 0
        while obs is not None and n < 300:
            yield env, obs
            obs = env.step(int(rng.integers(int(obs["action_mask"].sum()))))
            n += 1


def test_obs_history_keeps_own_decisions_and_opponent_events_only():
    opp_rows = own_rows = 0
    for env, obs in _games():
        h, m = obs["obs_history"], obs["obs_history_mask"].astype(bool)
        opp = h[m][:, 10] == 1
        assert not np.isin(h[m][opp, 0], [Kind.PASS, Kind.SELECT, Kind.YES, Kind.NO, Kind.OPTION]).any()
        opp_rows += int(opp.sum())
        own_rows += int((~opp).sum())
        me = env.player
        own = [e for e in env.history if e.actor == me]
        assert sum(1 for e in env.obs_history(me) if e.actor == me) == len(own)
    assert opp_rows and own_rows


def test_net_flag_reads_obs_history_and_marks_weights():
    torch.manual_seed(0)
    plain = PolicyNet(_emb(), d=16, layers=2, heads=2, dropout=0.0).eval()
    obsv = PolicyNet(_emb(), d=16, layers=2, heads=2, dropout=0.0, observable=True).eval()
    state = plain.state_dict()
    with pytest.raises(RuntimeError):  # full-history weights do not load silently into an observable net
        obsv.load_state_dict(state)
    obsv.load_state_dict({**state, "hist_observable": torch.ones((), dtype=torch.uint8)})
    with pytest.raises(RuntimeError):  # and an observable checkpoint does not load into a plain net
        plain.load_state_dict(obsv.state_dict())
    with pytest.raises(ValueError):
        PolicyNet(_emb(), d=16, layers=2, heads=2, hist=64, observable=True)
    batch = [obs for _, obs in _games(1)][:64]
    t = {k: torch.from_numpy(np.stack([o[k] for o in batch])) for k in batch[0]}
    with torch.no_grad():
        a, b = obsv(t), plain(observable_history(t))
    fin = torch.isfinite(a)
    assert torch.equal(fin, torch.isfinite(b)) and torch.allclose(a[fin], b[fin], atol=1e-5, rtol=1e-4)
    with pytest.raises(KeyError):
        observable_history({k: v for k, v in t.items() if not k.startswith("obs_history")})


def test_singletons_answered_by_the_host_are_not_in_obs_history():
    """A host session (auto_single=False) answers singletons with step(0); a client never records those, so its
    obs_history equals the auto_single=True env's at every real pick (DeepSeek review 2026-11-10)."""
    decks = [load_ydk(p) for p in sorted((ROOT / "tests/fixtures/decks").glob("*.ydk"))]
    cfg, engine = DuelConfig(decks[0], decks[1], seed=10), Engine()
    auto, host = PyEnv(engine, cfg), PyEnv(engine, cfg, auto_single=False)
    a, h = auto.reset(), host.reset()
    rng, picks = np.random.default_rng(0), 0
    while a is not None and picks < 200:
        while h is not None and int(h["action_mask"].sum()) == 1:
            h = host.step(0)
        for k in ("obs_history", "obs_history_mask"):
            assert np.array_equal(a[k], h[k]), (picks, k)
        i = int(rng.integers(int(a["action_mask"].sum())))
        a, h, picks = auto.step(i), host.step(i), picks + 1
    assert picks > 50


def test_onnx_export_accepts_observable_nets():
    from decklight.model.onnx_export import _check_v1, history_source

    _check_v1(PolicyNet(_emb(), d=16, layers=2, heads=2))
    net = PolicyNet(_emb(), d=16, layers=2, heads=2, observable=True)
    _check_v1(net)
    assert history_source(net) == "obs_history"
    assert history_source(PolicyNet(_emb(), d=16, layers=2, heads=2)) == "history"
