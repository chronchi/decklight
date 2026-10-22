"""search_match --opp ygoagent: same deals as playtest's --ygoagent match; S+search vs YgoAgentPolicy."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ga_schedule_matches_playtest_ygoagent_deals():
    """ga_schedule = playtest's ygo-agent match deals: ring pairs over the sorted gauntlet
    stems, one seed, both seats (S plays deck di, like A's deck in playtest's per_game)."""
    from decklight.eval.match import ring_pairs

    search_match = _load("search_match")
    names = sorted(p.stem for p in (ROOT / "decks/gauntlet").glob("*.ydk"))
    assert len(names) >= 2
    sched = search_match.ga_schedule(names, 4, 1)
    pairs = ring_pairs(len(names), 4)
    assert len(sched) == len(pairs) * 2  # 1 seed x both seats
    for n, ((di, dj), entry) in enumerate(zip(pairs, sched[::2], strict=True)):
        assert entry == (di, dj, 0, 0, 2 * n)  # S first
        assert sched[2 * n + 1] == (di, dj, 0, 1, 2 * n + 1)  # S second
    # S's deck is always di (pairs with base_r3's raw games on (deck_a, deck_b, seed, a_first))
    for di, dj, s, s_seat, idx in sched:
        assert sched[idx] == (di, dj, s, s_seat, idx)
        assert names[di] is not None


def test_seat_of_matches_env_player():
    """seat_of(obs) recovers the acting seat: global[4] is 1 iff seat 0 (first player) acts."""
    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig, Engine

    search_match = _load("search_match")
    decks = [load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))]
    env = CppEnv(Engine(), DuelConfig(decks[2], decks[3], seed=8))
    obs = env.reset()
    assert int(env.player) == 0
    assert search_match.seat_of(obs) == 0
    # a few steps in, seat_of still tracks env.player
    for _ in range(10):
        if obs is None:
            break
        a = int(np.flatnonzero(obs["action_mask"])[0])
        me = int(env.player)
        assert search_match.seat_of(obs) == me
        obs = env.step(a)


def _tiny_ckpt(tmp_path: Path):
    """A real (random) d=16 BC checkpoint + emb over the fixture decks' codes."""
    from decklight.decks import load_ydk
    from decklight.model.policy import PolicyNet

    decks = [load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    run = tmp_path / "run" / "bc"
    run.mkdir(parents=True)
    np.savez(tmp_path / "emb.npz", **emb)
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, run / "best.pt")
    (run / "metrics.jsonl").write_text(json.dumps({"args": {"d": 16, "layers": 1, "heads": 2}}) + "\n")
    return run / "best.pt", tmp_path / "emb.npz"


def test_ygoagent_two_games(tmp_path):
    """S (tiny d=16 policy + search) vs ygo-agent for 2 games: rows carry the paired-gate
    fields and search stats, and --summarize reports an S win rate with first/second splits."""
    model = ROOT / "data/models/0546_26550M.tflite"
    codes = ROOT / "data/models/code_list.txt"
    if not model.is_file() or not codes.is_file():
        pytest.skip("ygo-agent weights not on this machine")
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        pytest.skip("ai_edge_litert not installed")
    search_match = _load("search_match")
    ckpt, emb = _tiny_ckpt(tmp_path)
    out = tmp_path / "m"
    argv = ["search_match", "--ckpt", str(ckpt), "--emb", str(emb), "--opp", "ygoagent",
            "--gauntlet-decks", str(FIXTURES / "decks"), "--ga-opps", "1", "--ga-seeds", "1",
            "--games-limit", "2", "--k", "2", "--samples", "1", "--horizon", "1",
            "--conf-skip", "1.01", "--own-frac", "1.0", "--own-turn1", "1",
            "--width", "2", "--batch", "16", "--out", str(out)]  # fmt: skip
    old = sys.argv
    sys.argv = argv
    try:
        search_match.main()
    finally:
        sys.argv = old
    rows = [json.loads(x) for x in (out / "games-0.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 2
    for r in rows:
        assert r["winner"] in (0, 1, 2, None)
        assert r["winner"] is not None or r["truncated"]
        # paired-gate fields (playtest per_game format + search stats)
        assert r["deck_s"] in ("BlueEyes", "Branded", "Labrynth", "SkyStrikerAce")
        assert r["deck_r"] in ("BlueEyes", "Branded", "Labrynth", "SkyStrikerAce")
        assert r["deck_s"] != r["deck_r"]
        assert r["s_first"] == (r["s_seat"] == 0)
        assert r["decks"][r["s_seat"]] == r["deck_s"]
        w = r["winner"]
        want = 1.0 if w == r["s_seat"] else (0.0 if w in (0, 1) else 0.5)
        assert r["win"] == want
        assert r["cut"] == r["truncated"]
    assert sum(r["n_searched"] for r in rows) >= 1
    old = sys.argv
    sys.argv = ["search_match", "--summarize", str(out)]
    try:
        summ = search_match.summarize(out)
    finally:
        sys.argv = old
    assert summ["games"] == 2
    assert 0.0 <= summ["s_win_rate"] <= 1.0
    assert summ["first"]["n"] + summ["second"]["n"] == 2
    assert json.loads((out / "summary.json").read_text()) == summ


def test_ygoagent_drives_r_and_only_r(tmp_path):
    """R's outer moves are ygo-agent's own picks (replayed with a fresh YgoAgentPolicy, no loop guard), and
    ygo-agent scored exactly the outer R decisions (no rollout leak)."""
    from decklight.eval import ygoagent as yg

    model = ROOT / "data/models/0546_26550M.tflite"
    codes = ROOT / "data/models/code_list.txt"
    if not model.is_file() or not codes.is_file():
        pytest.skip("ygo-agent weights not on this machine")
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        pytest.skip("ai_edge_litert not installed")
    search_match = _load("search_match")
    ckpt, emb = _tiny_ckpt(tmp_path)
    out = tmp_path / "m"
    scored = []

    orig_call = yg.YgoAgentPolicy.__call__

    def counting(self, buf, rows, keys=None):
        scored.append(len(np.asarray(rows)))
        return orig_call(self, buf, rows, keys)

    yg.YgoAgentPolicy.__call__ = counting
    argv = ["search_match", "--ckpt", str(ckpt), "--emb", str(emb), "--opp", "ygoagent",
            "--gauntlet-decks", str(FIXTURES / "decks"), "--ga-opps", "1", "--ga-seeds", "1",
            "--games-limit", "2", "--k", "2", "--samples", "1", "--horizon", "1",
            "--conf-skip", "1.01", "--own-frac", "1.0", "--own-turn1", "1",
            "--width", "2", "--batch", "16", "--out", str(out)]  # fmt: skip
    old = sys.argv
    sys.argv = argv
    try:
        search_match.main()
    finally:
        sys.argv = old
        yg.YgoAgentPolicy.__call__ = orig_call
    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig, Engine

    engine = Engine()
    pol = yg.YgoAgentPolicy(seed=3)
    n_r = 0
    for raw in (out / "games-0.jsonl").read_text().splitlines():
        r = json.loads(raw)
        assert search_match.seat_of  # the router used below
        cfg = DuelConfig(*(load_ydk(FIXTURES / "decks" / f"{d}.ydk") for d in r["decks"]), seed=r["seed"])
        env = CppEnv(engine, cfg)
        obs = env.reset()
        # Card-faithful (NOTEBOOK 2026-10-09 08:40): ygo-agent's pick is played as given (no loop guard), exactly as in
        # the report card's match; the old shared-guard replica encoded the bug that forbade its own pick.
        for a in r["choices"]:
            me = int(env.player)
            if me != r["s_seat"]:
                n_r += 1
                buf = {k: np.ascontiguousarray(v[None]) for k, v in obs.items()}
                buf["player"] = np.array([me], np.int64)
                want = int(pol(buf, np.array([0]), keys=np.array([r["game_idx"]]))[0])
                assert want == int(a), (r["game_idx"], n_r)
            obs = env.step(int(a))
    assert sum(scored) == n_r > 0  # ygo-agent saw exactly the outer R decisions: none from rollouts
