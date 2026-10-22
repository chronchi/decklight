"""Deck-aware RL league: YGO episodes are always gauntlet-vs-gauntlet (tiny d=16 nets, real deck lists).

Covers the rollout's episode-draw logic (`Collector._draw_episode`, no model forward)
plus the C++ VecEnv gauntlet pool it conditions on:
- every YGO draw has both decks in the gauntlet slice; pool-deck episodes never draw YGO;
- the gauntlet-episode share is ~= F (+-3 pts) and the YGO share ~= F * G;
- with F = G = 0 (and no fixed/greedy shares) the draws equal the old code's for the same seed;
- the C++ pool flags `buf["ga"]` at about rate F with in-slice decks, and draws
  bit-identical decks with p = 0.
"""

from pathlib import Path

import numpy as np
import torch

from decklight.decks import load_ydk
from decklight.engine.cpp_env import VecEnv
from decklight.model.policy import PolicyNet
from decklight.rl.league import BASE, GREEDY, SELF, YGO, League, pfsp_weight
from decklight.rl.ppo import SlotStore, gather
from decklight.rl.rollout import Collector

ROOT = Path(__file__).resolve().parents[2]
N_ENVS, ROUNDS = 8, 250  # 2000 episode draws
F, G = 0.3, 0.5


def _deck_lists():
    """Real training pool split: (pool decks, gauntlet decks), gauntlet slice at the end."""
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    pool = sorted((ROOT / "decks/pool").glob("*.ydk"))
    pool = [load_ydk(p) for p in pool if p.stem not in hold]
    ga = [load_ydk(p) for p in sorted((ROOT / "decks/gauntlet").glob("*.ydk"))]
    assert pool and len(ga) == 31, (len(pool), len(ga))
    return pool, ga


def _tiny(n_decks: int, codes: list[int], value: bool):
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    return PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=value).eval()


class FakeVec:
    """Mimics the C++ VecEnv episode-draw contract: `buf` carries obs-shaped arrays plus
    the gauntlet flag and deck indices of each env's current episode."""

    SHAPES = {"cards": ((160, 21), np.int32), "card_mask": ((160,), bool),
              "global": ((24,), np.int32), "chain": ((16, 5), np.int32),
              "chain_mask": ((16,), bool), "actions": ((64, 10), np.int32),
              "action_mask": ((64,), bool), "history": ((32, 13), np.int32),
              "history_mask": ((32,), bool)}  # fmt: skip

    def __init__(self, n: int, n_decks: int, ga_p: float, ga_start: int, ga_count: int, seed: int):
        self.num_envs = n
        self.buf = {k: np.zeros((n,) + s, d) for k, (s, d) in self.SHAPES.items()}
        self.buf["ga"] = np.zeros(n, bool)
        self.buf["deck0"] = np.full(n, -1, np.int32)
        self.buf["deck1"] = np.full(n, -1, np.int32)
        self.n_decks, self.ga_p = n_decks, ga_p
        self.ga_start, self.ga_count = ga_start, ga_count
        self._rng = np.random.default_rng(seed)

    def async_reset(self):
        for i in range(self.num_envs):
            self.next_episode(i)

    def next_episode(self, i: int):
        if self._rng.random() < self.ga_p:
            d0 = self.ga_start + int(self._rng.integers(self.ga_count))
            d1 = self.ga_start + int(self._rng.integers(self.ga_count))
            self.buf["ga"][i] = True
        else:
            d0, d1 = int(self._rng.integers(self.n_decks)), int(self._rng.integers(self.n_decks))
            self.buf["ga"][i] = False
        self.buf["deck0"][i], self.buf["deck1"][i] = d0, d1


class StubYgo:
    def reset_keys(self, game_ids):
        pass


def _ga_collector(pool, ga, seed: int, p_greedy: float = 0.0):
    decks = pool + ga
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    league = League(0.4, 0.15, 0, np.random.default_rng(seed + 999), p_greedy=p_greedy,
                    ga_ygo_share=G)
    vec = FakeVec(N_ENVS, len(decks), F, len(pool), len(ga), seed + 7)
    store = SlotStore(512, gather(vec.buf, np.arange(1), 0, False))
    dev = torch.device("cpu")
    col = Collector(vec, _tiny(len(decks), codes, True), _tiny(len(decks), codes, False),
                    store, 0, False, dev, 0.3, -0.5, 2, np.random.default_rng(seed), "fp32",
                    league, StubYgo(), greedy=None, ga_share=F, ga_range=(len(pool), len(ga)),
                    fail=None)
    return col, vec, league


def _old_sample(rng, p_self: float, p_ygo: float) -> str:
    """Literal copy of League.sample before fixed/greedy support (base-only pool)."""
    r = float(rng.random())
    if r < p_self:
        return SELF
    if r < p_self + p_ygo:
        return YGO
    w = pfsp_weight(0.5)
    x = float(rng.random()) * w
    return BASE if x - w <= 0 else BASE


def test_ga_draws_condition_on_decks():
    pool, ga = _deck_lists()
    col, vec, _ = _ga_collector(pool, ga, seed=3)
    lo, cnt = len(pool), len(ga)
    draws = []
    for _ in range(ROUNDS):
        for i in range(N_ENVS):
            vec.next_episode(i)
            col._draw_episode(i)
            draws.append((str(col.opp[i]), bool(vec.buf["ga"][i]),
                          int(vec.buf["deck0"][i]), int(vec.buf["deck1"][i])))
    assert len(draws) == N_ENVS * ROUNDS
    ga_rate = sum(g for _, g, _, _ in draws) / len(draws)
    assert abs(ga_rate - F) < 0.03, ga_rate
    ygo = [d for d in draws if d[0] == YGO]
    assert ygo, "no YGO draws at F=0.3, G=0.5 over 2000 draws"
    for _, g, d0, d1 in ygo:  # every YGO episode is gauntlet-vs-gauntlet
        assert g and lo <= d0 < lo + cnt and lo <= d1 < lo + cnt, (g, d0, d1)
    assert all(o != YGO for o, g, _, _ in draws if not g)  # pool episodes never draw YGO
    ygo_rate = len(ygo) / len(draws)
    assert abs(ygo_rate - F * G) < 0.03, ygo_rate


def test_zero_flags_match_old_draws():
    rng_new = np.random.default_rng(11)
    rng_old = np.random.default_rng(11)
    lg = League(0.4, 0.15, 6, rng_new)
    for _ in range(2000):
        assert lg.sample() == _old_sample(rng_old, 0.4, 0.15)
    assert set(lg.sampling_probs()) == {SELF, YGO, BASE}


def test_probs_cover_fixed_and_greedy():
    lg = League(0.4, 0.1, 6, np.random.default_rng(0), fixed={"ft": object()}, p_fixed=0.2,
                p_greedy=0.05, ga_ygo_share=0.5)
    p = lg.sampling_probs()
    assert abs(sum(p.values()) - 1.0) < 1e-9, p
    assert set(p) == {SELF, YGO, GREEDY, BASE, "ft"}, p
    seen = {lg.sample() for _ in range(2000)}
    assert {"ft", GREEDY} <= seen, seen
    seen_nga = {lg.sample_non_ygo() for _ in range(2000)}
    assert YGO not in seen_nga and {"ft", GREEDY} <= seen_nga, seen_nga


def test_vecenv_ga_pool(engine):
    from decklight.decks import load_ydk as _load

    decks = [_load(p) for p in sorted((Path(__file__).resolve().parents[1] / "fixtures" / "decks").glob("*.ydk"))]
    n = len(decks)
    v = VecEnv(engine, decks, 256, 4, seed=5)
    v.set_ga_share(0.5, n - 2, 2)
    v.reset()
    ga = v.buf["ga"]
    assert abs(float(ga.mean()) - 0.5) < 0.1, float(ga.mean())
    m = np.asarray(ga, dtype=bool)
    assert all(n - 2 <= int(d) < n for d in v.buf["deck0"][m])
    assert all(n - 2 <= int(d) < n for d in v.buf["deck1"][m])
    assert all(0 <= int(d) < n for d in v.buf["deck0"])
    # p = 0 draws bit-identical decks to no call at all
    a = VecEnv(engine, decks, 8, 2, seed=0)
    a.reset()
    b = VecEnv(engine, decks, 8, 2, seed=0)
    b.set_ga_share(0.0, n - 2, 2)
    b.reset()
    assert (a.buf["deck0"] == b.buf["deck0"]).all()
    assert (a.buf["deck1"] == b.buf["deck1"]).all()
    assert not np.asarray(b.buf["ga"]).any()


def test_vecenv_pool_count(engine):
    """set_pool_count(n): ordinary draws stay in [0, n); no call = old draws."""
    from decklight.decks import load_ydk as _load

    decks = [_load(p) for p in sorted((Path(__file__).resolve().parents[1] / "fixtures" / "decks").glob("*.ydk"))]
    n = len(decks)
    assert n >= 3
    v = VecEnv(engine, decks, 8, 2, seed=0)
    v.set_pool_count(n - 2)
    v.reset()
    for _ in range(ROUNDS):
        v.reset()
        assert (np.asarray(v.buf["deck0"]) < n - 2).all()
        assert (np.asarray(v.buf["deck1"]) < n - 2).all()
    # pool_count == all decks draws bit-identical decks to no call (same seed)
    a = VecEnv(engine, decks, 8, 2, seed=0)
    a.reset()
    b = VecEnv(engine, decks, 8, 2, seed=0)
    b.set_pool_count(n)
    b.reset()
    assert (a.buf["deck0"] == b.buf["deck0"]).all()
    assert (a.buf["deck1"] == b.buf["deck1"]).all()


def test_vecenv_ga_mixed(engine):
    """Mixed ga episodes: exactly one seat gauntlet, buf['ga'] 1/2 matches it, ~half each."""
    from decklight.decks import load_ydk as _load

    decks = [_load(p) for p in sorted((Path(__file__).resolve().parents[1] / "fixtures" / "decks").glob("*.ydk"))]
    n = len(decks)
    n_pool, ga_start, ga_count = n - 2, n - 2, 2
    v = VecEnv(engine, decks, 8, 4, seed=5)
    v.set_pool_count(n_pool)
    v.set_ga_share(0.5, ga_start, ga_count, mixed=True)
    v.reset()
    flags: list[int] = []
    for _ in range(ROUNDS):
        v.reset()
        ga = np.asarray(v.buf["ga"], dtype=np.int64)
        d0 = np.asarray(v.buf["deck0"])
        d1 = np.asarray(v.buf["deck1"])
        for f, a, b in zip(ga.tolist(), d0.tolist(), d1.tolist()):
            flags.append(int(f))
            if f == 0:
                assert 0 <= a < n_pool and 0 <= b < n_pool, (a, b)
            elif f == 1:
                assert ga_start <= a < ga_start + ga_count, (a, b)
                assert 0 <= b < n_pool, (a, b)
            elif f == 2:
                assert 0 <= a < n_pool, (a, b)
                assert ga_start <= b < ga_start + ga_count, (a, b)
            else:
                raise AssertionError(f"bad ga flag {f}")
    arr = np.array(flags)
    assert set(np.unique(arr).tolist()) <= {0, 1, 2}
    assert abs((arr != 0).mean() - 0.5) < 0.05, (arr != 0).mean()
    ones = (arr == 1).sum()
    twos = (arr == 2).sum()
    assert ones > 0 and twos > 0 and abs(ones - twos) / (ones + twos) < 0.15, (ones, twos)
    # old mode untouched by the new flag: same seed without mixed gives ga in {0, 1} only
    w = VecEnv(engine, decks, 8, 4, seed=5)
    w.set_ga_share(0.5, ga_start, ga_count)
    w.reset()
    assert set(np.unique(np.asarray(w.buf["ga"])).tolist()) <= {0, 1}
