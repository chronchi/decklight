"""Practice "Ask the bot" hints (web-ux-plan step 4, brief hints-whatif).

Unit: top-3 ordering, the fairness spy (human obs only, like the review),
the cost cap with a fake launcher. The practice-only gate is a server rule,
covered here through the websocket (tiny launcher) as well as in
test_web_server.py.
"""

import json
import random
from pathlib import Path

import numpy as np
from starlette.testclient import TestClient

from decklight.analysis.hints import compute_hint
from decklight.analysis.review import load_record
from decklight.engine.env import PyEnv
from decklight.web.server import create_app

FIXTURE = Path("tests/fixtures/web/mitsurugi-20261016")
FIXTURES = Path("tests/fixtures/decks")
TOKEN = "test-token"


class _FakeCalib:
    def p(self, v: float, turn: int) -> float:
        return float(np.clip((float(v) + 1.0) / 2.0, 0.03, 0.97))


class _FakeLauncher:
    """ProductLauncher-shaped stub: controlled logits, spied obs, optional value."""

    def __init__(self, logits_fn=None, value_v: float = 0.0, delay: float = 0.0) -> None:
        self._fn = logits_fn
        self._v = value_v
        self._delay = delay
        self.policy_seen: list[dict] = []
        self.value_seen: list[dict] = []

    def base_logits(self, obs_list):
        import time

        if self._delay:
            time.sleep(self._delay)
        for o in obs_list:
            self.policy_seen.append({k: np.array(v) for k, v in o.items()})
        if self._fn is not None:
            return self._fn(obs_list)
        n = len(obs_list)
        w = int(obs_list[0]["action_mask"].shape[0])
        return np.zeros((n, w), np.float32)

    @property
    def value(self):
        def fn(batch):
            self.value_seen.append({k: np.array(v) for k, v in batch.items()})
            n = next(iter(batch.values())).shape[0]
            return np.full(n, self._v, np.float32)

        return fn


def _human_obs_ok(obs: dict, first_player: int, human: int) -> bool:
    g = obs["global"][0] if obs["global"].ndim > 1 else obs["global"]
    return int(g[4]) == int(first_player == human)


def _env_at_first_human_multi(engine):
    rec, cfg, hs, _d = load_record(FIXTURE / "game.jsonl")
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    for ch in rec["choices"]:
        if env.done:
            raise AssertionError("no human multi-option decision in the fixture")
        if env.player == hs and len(env.opts) > 1:
            return env, cfg, hs, rec
        env.step(int(ch))
    raise AssertionError("no human multi-option decision in the fixture")


def test_hint_top3_ordering(engine):
    """Options come back ordered by policy probability, at most 3, probs sum <= 1."""
    env, cfg, hs, _rec = _env_at_first_human_multi(engine)
    n = len(env.opts)
    assert n > 1
    width = int(env._observe()["action_mask"].shape[0])
    # Logits that rank option (n-1) first, then 0, then 1, ... : descending games.
    table = [(n - 1 - i) % n for i in range(n)]
    logits = np.zeros(width, np.float32)
    for rank, idx in enumerate(table):
        logits[idx] = float(n - rank)
    launcher = _FakeLauncher(logits_fn=lambda ol: np.tile(logits, (len(ol), 1)))
    out = compute_hint(engine, cfg, env, hs, launcher, None)
    got = [o["index"] for o in out["options"]]
    assert got == table[:3], f"{got} != {table[:3]}"
    assert len(out["options"]) <= 3
    assert sum(o["p"] for o in out["options"]) <= 1.0 + 1e-6
    assert out["ms"] < 3000, f"hint took {out['ms']} ms"


def test_hint_fairness_spy_human_obs_only(engine):
    """Every value obs is the human seat's obs (the review's fairness rule)."""
    rec, cfg, hs, _d = load_record(FIXTURE / "game.jsonl")
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    launcher = _FakeLauncher()
    for ch in rec["choices"]:
        if env.done:
            break
        if env.player == hs and len(env.opts) > 1:
            compute_hint(engine, cfg, env, hs, launcher, _FakeCalib())
            break
        env.step(int(ch))
    assert launcher.value_seen, "value never called"
    env0 = PyEnv(engine, cfg, auto_single=False)
    env0.reset()
    first_player = int(env0.duel.first_player)
    for batch in launcher.value_seen:
        n = next(iter(batch.values())).shape[0]
        for i in range(n):
            obs = {k: batch[k][i] for k in batch}
            assert _human_obs_ok(obs, first_player, hs), "hint value saw a non-human obs"
    assert launcher.policy_seen
    assert any(_human_obs_ok(o, first_player, hs) for o in launcher.policy_seen)


def test_hint_cost_cap_fake_launcher(engine):
    """Three hints on the fixture answer in well under 3 s with a fake launcher."""
    import time

    rec, cfg, hs, _d = load_record(FIXTURE / "game.jsonl")
    launcher = _FakeLauncher(delay=0.005)
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    done = 0
    t0 = time.monotonic()
    for ch in rec["choices"]:
        if env.done or done >= 3:
            break
        if env.player == hs and len(env.opts) > 1:
            out = compute_hint(engine, cfg, env, hs, launcher, _FakeCalib())
            assert len(out["options"]) <= 3
            assert sum(o["p"] for o in out["options"]) <= 1.0 + 1e-6
            assert all(o["est"] is not None for o in out["options"])
            assert out["ms"] < 3000
            done += 1
        env.step(int(ch))
    dt = (time.monotonic() - t0) * 1000.0
    assert done == 3, "fixture has fewer than 3 human multi-option decisions"
    assert dt < 3000, f"3 hints took {dt:.0f} ms"


def _tiny_forward():
    import torch

    from decklight.decks import load_ydk
    from decklight.model.bc_data import OBS_KEYS
    from decklight.model.policy import PolicyNet

    decks = [load_ydk(p) for p in sorted(FIXTURES.glob("*.ydk"))]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {
        "codes": np.array(codes),
        "card": rng.normal(size=(n, 8)).astype(np.float32),
        "effects": rng.normal(size=(n, 2, 8)).astype(np.float32),
        "n_effects": np.ones(n, np.int64),
    }
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()

    @torch.no_grad()
    def forward(obs_list):
        t = {k: torch.from_numpy(np.ascontiguousarray(np.stack([o[k] for o in obs_list]))) for k in OBS_KEYS}
        return net(t).numpy()

    return forward


class _TinyLauncher:
    def __init__(self, forward) -> None:
        self.forward = forward
        self.value = None

    def base_logits(self, obs_list):
        return self.forward(obs_list)

    def start_duel(
        self, engine, config, bot_seat, *, game_seed, ws_seed, human_fn=None, prefix=(), prefix_decisions=None
    ):
        from decklight.edopro.product_bot import ProductDriver

        return ProductDriver(
            engine,
            config,
            bot_seat=bot_seat,
            base_logits=self.forward,
            value=None,
            search_kwargs={},
            game_seed=game_seed,
            ws_seed=ws_seed,
            fast=True,
            human_fn=human_fn,
            prefix=prefix,
            prefix_decisions=prefix_decisions,
        ).start()

    def record_info(self, seed: int) -> dict:
        return {"ckpt": "tiny-test", "fast": True, "game_seed": seed, "search": "tiny"}


def _app(tmp_path, engine, forward):
    return create_app(
        token=TOKEN,
        engine=engine,
        make_opponent=lambda mode: _TinyLauncher(forward),
        record_root=tmp_path / "runs",
        pics_dir=tmp_path / "pics",
    )


def _wait_prompt(ws, limit=2000):
    for _ in range(limit):
        m = json.loads(ws.receive_text())
        assert m.get("t") == "state", m.get("t")
        if m["prompt"] is not None:
            return m
    raise AssertionError("no human prompt arrived")


def test_hint_practice_only_gate(tmp_path, engine):
    """Duel mode rejects hints; practice mode answers top-3 and records hints_used."""
    forward = _tiny_forward()
    human_text = (FIXTURES / "BlueEyes.ydk").read_text()
    bot_text = (FIXTURES / "Branded.ydk").read_text()
    # Duel (default): rejected.
    app = _app(tmp_path / "duel", engine, forward)
    with TestClient(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        ws.send_json(
            {
                "t": "new_game",
                "deck": {"ydk": human_text},
                "bot_deck": {"ydk": bot_text},
                "first": "me",
                "mode": "product",
            }
        )
        _wait_prompt(ws)
        ws.send_json({"t": "hint"})
        err = json.loads(ws.receive_text())
        assert err["t"] == "error" and err["text"] == "Hints are only in practice mode"
    # Practice: answered, probs sum <= 1, then the record counts the hint.
    app = _app(tmp_path / "prac", engine, forward)
    with TestClient(app).websocket_connect(f"/ws?t={TOKEN}") as ws:
        ws.receive_text()
        ws.send_json(
            {
                "t": "new_game",
                "deck": {"ydk": human_text},
                "bot_deck": {"ydk": bot_text},
                "first": "me",
                "mode": "product",
                "practice": True,
            }
        )
        first = _wait_prompt(ws)
        assert first["practice"] is True
        ws.send_json({"t": "hint"})
        hm = json.loads(ws.receive_text())
        assert hm["t"] == "hint" and hm["prompt"] == first["prompt"]["id"]
        assert 1 <= len(hm["options"]) <= 3
        assert sum(o["p"] for o in hm["options"]) <= 1.0 + 1e-6
        got_idx = [o["index"] for o in hm["options"]]
        assert len(set(got_idx)) == len(got_idx)
        assert all(0 <= i < len(first["prompt"]["options"]) for i in got_idx)
        ws.send_json({"t": "surrender"})
        for _ in range(50):
            m = json.loads(ws.receive_text())
            if m.get("result") is not None:
                break
        assert m["result"] is not None
    subs = [p for p in (tmp_path / "prac" / "runs").iterdir() if p.is_dir()]
    assert len(subs) == 1
    rec = json.loads((subs[0] / "game.jsonl").read_text())
    assert rec["practice"] is True
    assert rec["hints_used"] == 1
    _ = random.Random(0)  # keep the import used (seed convention of the file)
