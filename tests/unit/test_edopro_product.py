"""EDOPro play D3: the product as the bot (each < 5 min with `taskset -c 0-3`).

1. Equivalence: search_match's routed path (--opp greedy, one-hot unguarded R) in-process,
   then ProductDriver on the same deal/seed with a scripted human replaying R's choices:
   S's choices, the search log (cands/chosen) and the winner are identical (both seats).
2. Host end to end with the product (tiny model) and D2's fake client: the duel finishes,
   the record carries the product settings and replays.
3. --fast plays without search: zero searched decisions in the log.
4. auto_single mapping: host-env (auto_single=False) sequence <-> CppEnv (auto_single=True)
   sequence round trip on a recorded game.
"""

import json
import random
import threading

import numpy as np
import pytest
import torch

from decklight.decks import load_ydk
from decklight.edopro.bot import RandomBot
from decklight.edopro.host import DuelHost
from decklight.edopro.product_bot import (
    ProductDriver,
    host_to_product,
    product_to_host,
)
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig
from decklight.engine.env import PyEnv
from decklight.eval.policies import GreedyPolicy
from decklight.model.bc_data import OBS_KEYS
from decklight.model.policy import PolicyNet
from decklight.search.honest import HonestRule
from decklight.search.play import argmax_select, search_game_gen
from decklight.search.product import RouteWindowSearch, seat_of, tag_game
from decklight.search.turn1 import _batch, drive_many


def _load_sibling(name: str):
    """Import a sibling test module by path (tests/ is not a package)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_host_test = _load_sibling("test_edopro_host")
FakeClient, _free_port = _host_test.FakeClient, _host_test._free_port

FIXTURES = "tests/fixtures/decks"


def _tiny_forward():
    """A tiny d=16 PolicyNet on synthetic embeddings (as test_search_play builds it)."""
    decks = [load_ydk(p) for p in sorted(__import__("pathlib").Path(FIXTURES).glob("*.ydk"))]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()

    @torch.no_grad()
    def forward(obs_list):
        t = {k: torch.from_numpy(np.ascontiguousarray(np.stack([o[k] for o in obs_list])))
             for k in OBS_KEYS}
        return net(t).numpy()

    return forward


def _cfg(engine_seed=8):
    from pathlib import Path

    decks = [load_ydk(p) for p in sorted(Path(FIXTURES).glob("*.ydk"))]
    return DuelConfig(decks[2], decks[3], seed=engine_seed)


def _never(buf):
    raise AssertionError("unused")


KW = {"own_turn1": True, "conf_skip": 1.01, "own_frac": 1.0,
      "honest": True, "det_tries": 8, "det_fallback": "skip"}  # fmt: skip


def _reference_game(engine, cfg, s_seat, forward, seed=0, game_seed=0, ho=None):
    """search_match's routed path in-process: RouteWindowSearch + tag_game, R = greedy one-hot,
    per_seat guard with R unguarded (search_match.py :615-633, :690-705)."""
    route: dict = {}
    ws = RouteWindowSearch(engine, _never, k=2, horizon=1, samples=1, seed=seed, route=route)
    gen = tag_game(0, s_seat, search_game_gen(
        engine, cfg, ws, s_seat, argmax_select, game_seed=game_seed,
        guard_mode="per_seat", unguarded_seat=1 - s_seat, ho=ho, **KW), route)
    opp = GreedyPolicy(seed=3)

    def fwd(obs_list: list[dict]) -> np.ndarray:
        base_at, base_obs, r_at, r_obs, r_me = [], [], [], [], []
        for i, o in enumerate(obs_list):
            tag = route.pop(id(o), ...)
            if tag is ...:
                raise AssertionError("untagged obs in the batch")
            if tag is None:
                base_at.append(i)
                base_obs.append(o)
            else:
                _, ss = tag
                if seat_of(o) == ss:
                    base_at.append(i)
                    base_obs.append(o)
                else:
                    r_at.append(i)
                    r_obs.append(o)
                    r_me.append(seat_of(o))
        a_width = int(obs_list[0]["action_mask"].shape[0])
        lg = np.zeros((len(obs_list), a_width), np.float32)
        if base_obs:
            b = forward(base_obs)
            for k, i in enumerate(base_at):
                lg[i] = b[k]
        if r_obs:
            buf = _batch(r_obs)
            buf["player"] = np.array(r_me, np.int64)
            acts = opp(buf, np.arange(len(r_obs)))
            for i, a in zip(r_at, acts, strict=True):
                lg[i, int(a)] = 1.0
        return lg

    return dict(drive_many([(0, gen)], fwd, 2))[0]


def _s_choices(res, s_seat):
    return [c for c, a in zip(res.choices, res.actors, strict=True) if a == s_seat]


@pytest.mark.parametrize("ho", [False, True])
@pytest.mark.parametrize("s_seat", [0, 1])
def test_routed_equivalence(engine, s_seat, ho):
    """The key test: the driver plays S exactly as search_match's routed path does (with and without the product's
    honest override, search_match --ho 1 = HonestRule())."""
    forward, cfg = _tiny_forward(), _cfg()
    ref = _reference_game(engine, cfg, s_seat, forward, ho=HonestRule() if ho else None)
    assert not ref.truncated and ref.winner in (0, 1)
    r_choices = [c for c, a in zip(ref.choices, ref.actors, strict=True) if a == 1 - s_seat]
    assert r_choices, "R made 0 decisions (routing broke)"

    drv = ProductDriver(engine, cfg, bot_seat=s_seat, base_logits=forward, value=None,
                        search_kwargs={"k": 2, "samples": 1, "ho": ho, **KW},
                        game_seed=0, ws_seed=0)
    drv.start()
    drv.feed_human(r_choices)
    drv.join(timeout=600)
    res = drv.res
    drv.close()
    assert res is not None and not res.truncated
    assert res.winner == ref.winner
    assert _s_choices(res, s_seat) == _s_choices(ref, s_seat)
    assert res.n_searched == ref.n_searched
    assert [(e["cands"], e["chosen"]) for e in res.log] == \
        [(e["cands"], e["chosen"]) for e in ref.log]


# --- 2. host end to end with the product (tiny model) ---

class _TinyLauncher:
    """A ProductLauncher-shaped stub holding the tiny model (no checkpoints)."""

    def __init__(self, forward) -> None:
        self.forward = forward

    def rps(self) -> int:
        return 1

    def start_duel(self, engine, config, bot_seat, *, game_seed, ws_seed, human_fn=None):
        return ProductDriver(
            engine, config, bot_seat=bot_seat, base_logits=self.forward, value=None,
            search_kwargs={"k": 2, "samples": 1, "honest": False, "conf_skip": 1.01,
                            "own_frac": 1.0, "own_turn1": True},
            game_seed=game_seed, ws_seed=ws_seed).start()

    def record_info(self, seed: int) -> dict:
        return {"ckpt": "tiny-test", "fast": False, "game_seed": seed, "ws_seed": seed, "search": "tiny"}


def test_host_end_to_end_product(tmp_path, engine):
    from pathlib import Path

    decks = [load_ydk(p) for p in sorted(Path(FIXTURES).glob("*.ydk"))]
    bot_deck, human_deck = decks[2], decks[3]
    mixed = list(human_deck.main) + list(human_deck.extra)
    host = DuelHost(engine, bot_deck, _TinyLauncher(_tiny_forward()), seed=1234, record_root=tmp_path)
    port = _free_port()
    thread = threading.Thread(target=host.serve, args=("127.0.0.1", port, True), daemon=True)
    thread.start()
    client = FakeClient(port, mixed, set(mixed), seed=0)
    client.run()
    thread.join(timeout=300)

    assert client.start_seen and client.win is not None and client.duel_end
    assert client.retries == 0
    assert not thread.is_alive(), "server did not exit after --once duel"

    subs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subs) == 1 and (subs[0] / "human.ydk").is_file()
    rec = json.loads((subs[0] / "game.jsonl").read_text())
    assert rec["winner"] == client.win[0] and rec["winner"] in (0, 1)
    assert rec["product"]["ckpt"] == "tiny-test"
    env = PyEnv.replay(engine, DuelConfig(bot_deck, human_deck, seed=rec["seed"])
                       if rec["human_seat"] == 1 else
                       DuelConfig(human_deck, bot_deck, seed=rec["seed"]),
                       rec["choices"], auto_single=False)
    assert env.done and env.winner == rec["winner"]


# --- 3. fast mode: no search ---

def test_fast_plays_without_search(engine):
    rng = random.Random(0)
    drv = ProductDriver(engine, _cfg(), bot_seat=0, base_logits=_tiny_forward(), value=None,
                        search_kwargs={}, game_seed=0, ws_seed=0, fast=True,
                        human_fn=lambda obs: rng.randrange(int(obs["action_mask"].sum())))
    drv.start()
    drv.join(timeout=600)
    res = drv.res
    drv.close()
    assert res is not None and not res.truncated and res.winner in (0, 1)
    assert res.n_searched == 0 and res.log == []


# --- 4. auto_single mapping round trip ---

def _random_host_game(engine, seed: int):
    """A full random-vs-random duel on the host env (auto_single=False): all choices recorded."""
    from pathlib import Path

    decks = [load_ydk(p) for p in sorted(Path(FIXTURES).glob("*.ydk"))]
    cfg = DuelConfig(decks[0], decks[1], seed=seed)
    for attempt in range(200):
        env = PyEnv(engine, cfg, auto_single=False)
        env.reset()
        rng = random.Random(seed * 1000 + attempt)
        n = 0
        while not env.done and n < 3000:
            # random legal option (the mapping only needs a legal recorded sequence)
            env.step(rng.randrange(len(env.opts)))
            n += 1
        if env.done:
            return cfg, list(env.choices)
    raise AssertionError("random host game never finished")


def test_auto_single_round_trip(engine):
    cfg, host_choices = _random_host_game(engine, 11)
    assert len(host_choices) > 10
    prod_choices = host_to_product(engine, cfg, host_choices)
    assert len(prod_choices) < len(host_choices)  # singletons were dropped
    assert product_to_host(engine, cfg, prod_choices) == [int(c) for c in host_choices]
    assert host_to_product(engine, cfg, product_to_host(engine, cfg, prod_choices)) == \
        [int(c) for c in prod_choices]
    # the product sequence replays on the CppEnv (auto_single=True) to the same winner
    host_env = PyEnv.replay(engine, cfg, host_choices, auto_single=False)
    prod_env = CppEnv.replay(engine, cfg, prod_choices, auto_single=True)
    assert host_env.done and prod_env.done
    assert int(prod_env.winner) == host_env.winner


def test_bot_interface_still_random(engine):
    """D2's RandomBot kept working behind the renamed interface."""
    env = type("E", (), {"opts": [1, 2, 3]})
    b1, b2 = RandomBot(7), RandomBot(7)
    assert [b1.next_choice(env) for _ in range(3)] == [b2.next_choice(env) for _ in range(3)]
    assert RandomBot(7).rps() in (1, 2, 3)
