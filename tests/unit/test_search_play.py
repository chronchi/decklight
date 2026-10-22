"""Live search match (src/decklight/search/play.py, scripts/search_match.py --summarize) over a tiny PolicyNet."""

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


def _tiny_forward():
    """A tiny d=16 PolicyNet on synthetic embeddings (as the search/RL tests do), as a batched logits fn."""
    from decklight.decks import load_ydk
    from decklight.model.bc_data import OBS_KEYS
    from decklight.model.policy import PolicyNet

    decks = [load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))]
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


def _cfg():
    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig

    decks = [load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))]
    return DuelConfig(decks[2], decks[3], seed=8)


def _play(engine, forward, **kw):
    from decklight.search.play import argmax_select, search_game_gen
    from decklight.search.turn1 import drive_many
    from decklight.search.window import WindowSearch

    def never(buf):
        raise AssertionError("unused")

    cfg = _cfg()
    ws = WindowSearch(engine, never, k=kw.pop("k", 2), horizon=1, samples=1, seed=0)
    gen = search_game_gen(engine, cfg, ws, kw.pop("s_seat", 0), argmax_select, kw.pop("game_seed", 0),
                          own_turn1=True, **kw)
    return dict(drive_many([(0, gen)], forward, 2))[0], cfg


def test_full_game_searches_and_replays(engine):
    """One live game under drive_many: it ends, S searches >= 1 decision, and the choice list
    replays to the same winner on a fresh env."""
    from decklight.engine.cpp_env import CppEnv

    res, cfg = _play(engine, _tiny_forward(), conf_skip=1.01, own_frac=1.0)
    assert not res.truncated
    assert res.winner in (0, 1)
    assert res.n_searched >= 1
    assert res.n_override <= res.n_searched
    assert len(res.choices) == len(res.actors)
    env = CppEnv(engine, cfg)
    obs = env.reset()
    for c in res.choices:
        assert obs is not None
        obs = env.step(int(c))
    assert obs is None and int(env.winner) == res.winner


def test_conf_skip_gates_search(engine):
    """conf_skip=0.0 searches nothing; conf_skip=1.01 searches every eligible S decision."""
    from decklight.engine.cpp_env import CppEnv
    from decklight.eval.loopguard import LoopGuard
    from decklight.search.turn1 import _masked
    from decklight.search.window import WindowSearch, is_window

    forward = _tiny_forward()
    res0, _ = _play(engine, forward, conf_skip=0.0, own_frac=1.0)
    assert res0.n_searched == 0 and res0.n_branches == 0
    assert not res0.truncated and res0.winner in (0, 1)

    res1, cfg = _play(engine, forward, conf_skip=1.01, own_frac=1.0)
    assert res1.n_searched >= 1
    # every eligible S decision (window, or own turn with >= 2 options from turn 1) with >= 2
    # candidates was searched: replay the same trajectory with the same guard discipline
    def never(buf):
        raise AssertionError("unused")

    ws = WindowSearch(engine, never, k=2, horizon=1, samples=1, seed=0)
    guard = LoopGuard()
    env = CppEnv(engine, cfg)
    obs = env.reset()
    want = set()
    for k, c in enumerate(res1.choices):
        me = int(env.player)
        if me == 0:
            window = is_window(obs, me)
            own = int(obs["global"][5]) == 1 and int(obs["action_mask"].sum()) >= 2
            if window or own:
                z = _masked(forward([obs])[0], obs)
                z_g, key = guard.mask(0, obs, z)
                if len(ws._candidates(z_g, obs, window)) >= 2:
                    want.add(k)
                guard.note(0, key, int(c))
            else:
                _, key = guard.mask(0, obs, _masked(forward([obs])[0], obs))
                guard.note(0, key, int(c))
        else:
            _, key = guard.mask(0, obs, _masked(forward([obs])[0], obs))
            guard.note(0, key, int(c))
        obs = env.step(int(c))
    assert set(res1.searched_idx) == want


def test_honest_skip_skips_searches_and_ends(engine):
    """honest + det_fallback=skip with det_tries=1: some game skips >= 1 search and still ends."""
    forward = _tiny_forward()
    skipped = 0
    for seed in range(4):
        res, _ = _play(engine, forward, conf_skip=1.01, own_frac=1.0,
                       honest=True, det_tries=1, det_fallback="skip", game_seed=seed)
        assert res.det_proposals > 0
        assert not res.truncated and res.winner in (0, 1)
        skipped += res.det_skipped
        if skipped > 0:
            break
    assert skipped > 0, "det_tries=1 never failed to sample a world"


def test_summarize_win_rate(tmp_path):
    """--summarize on two fake games-*.jsonl files reports the right S win rate."""
    search_match = _load("search_match")
    games = [
        # (s_seat, winner): S wins, S wins, S wins, S loses -> 0.75
        (0, 0, "blue-eyes", "branded"),
        (1, 1, "blue-eyes", "branded"),
        (0, 0, "branded", "blue-eyes"),
        (1, 0, "branded", "blue-eyes"),
    ]
    for proc in (0, 1):
        with (tmp_path / f"games-{proc}.jsonl").open("w") as f:
            for i in games[proc * 2 : proc * 2 + 2]:
                s_seat, winner, d0, d1 = i
                f.write(json.dumps({
                    "decks": [d0, d1], "seed": 0, "s_seat": s_seat, "game_idx": 0,
                    "winner": winner, "truncated": False, "turns": 6,
                    "choices": [0, 1], "actors": [0, 1],
                    "n_searched": 2, "n_override": 1, "search_s": 0.5,
                    "n_branches": 8, "n_capped": 0,
                }) + "\n")  # fmt: skip
    old = sys.argv
    sys.argv = ["search_match", "--summarize", str(tmp_path)]
    try:
        summ = search_match.summarize(tmp_path)
    finally:
        sys.argv = old
    assert summ["games"] == 4
    assert summ["s_win_rate"] == 0.75
    assert summ["ci"][0] <= 0.75 <= summ["ci"][1]
    assert summ["per_deck"] == {"blue-eyes": 0.5, "branded": 1.0}
    assert summ["searched_per_game"] == 2.0
    assert summ["override_rate"] == 0.5
    assert summ["capped_share"] == 0.0
    assert json.loads((tmp_path / "summary.json").read_text()) == summ


def test_log_holds_per_sample_scores(engine):
    """Every searched decision logs its per-sample selection scores; their mean per candidate is the logged score."""
    res, _ = _play(engine, _tiny_forward(), conf_skip=1.01, own_frac=1.0)
    assert res.log
    for e in res.log:
        assert len(e["samples"]) == len(e["cands"])
        for col, sc in zip(e["samples"], e["scores"], strict=True):
            assert len(col) == 1  # samples=1 in _play
            assert abs(sum(col) / len(col) - sc) < 1e-3


def test_guard_modes_validate_and_play(engine):
    """per_seat + an unguarded seat plays to the end (a win, or the decision cap: the random tiny net may loop when
    unguarded); an unknown guard_mode fails loud."""
    res, _ = _play(engine, _tiny_forward(), conf_skip=1.01, own_frac=1.0, guard_mode="per_seat", unguarded_seat=1)
    assert res.winner in (0, 1) or res.truncated
    with pytest.raises(ValueError):
        _play(engine, _tiny_forward(), guard_mode="nope")


def _tiny_critic(tmp_path, priv=True):
    """A tiny d=16 privileged critic checkpoint (known + value head) with train_rl-style metrics args."""
    from decklight.decks import load_ydk
    from decklight.model.policy import PolicyNet

    decks = [load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))]
    codes = sorted({c for d in decks for c in list(d.main) + list(d.extra)} - {0})
    rng = np.random.default_rng(0)
    n = len(codes)
    emb = {"codes": np.array(codes), "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}  # fmt: skip
    np.savez(tmp_path / "emb.npz", **emb)
    torch.manual_seed(0)
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, known=True, value=True)
    torch.nn.init.normal_(net.known_emb.weight)  # nonzero, so revealed cards move the value
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
               tmp_path / "critic.pt")
    args = {"d": 16, "layers": 1, "heads": 2, "hist": 0, "known": False, "rl": {"priv_critic": "1" if priv else "0"}}
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"args": args}) + "\n")
    return tmp_path / "critic.pt", tmp_path / "emb.npz"


def test_priv_value_scores_sampled_worlds(engine, tmp_path):
    """Honest search with the privileged critic as leaf value: leaf obs carry priv_code (sampled world's hidden
    cards), the critic reads it (values move when it is zeroed), and the game ends."""
    from decklight.eval.value import PrivValueNet
    from decklight.search.play import argmax_select, search_game_gen
    from decklight.search.turn1 import drive_many
    from decklight.search.window import WindowSearch

    ckpt, emb = _tiny_critic(tmp_path)
    pv = PrivValueNet(ckpt, emb)
    seen = []

    def value(obs):
        seen.append({k: v.copy() for k, v in obs.items()})
        return pv.values(obs)

    def never(buf):
        raise AssertionError("unused")

    ws = WindowSearch(engine, never, k=2, horizon=1, samples=2, seed=0, value=value)
    gen = search_game_gen(engine, _cfg(), ws, 0, argmax_select, 0, own_turn1=True, conf_skip=1.01, own_frac=1.0,
                          honest=True, det_tries=32, det_fallback="skip")
    res = dict(drive_many([(0, gen)], _tiny_forward(), 2))[0]
    assert res.winner in (0, 1) and not res.truncated
    assert seen and any(e["scores"] for e in res.log)
    hidden = [o for o in seen if (o["priv_code"] > 0).any()]
    assert hidden, "no leaf obs carried a hidden card's code"
    o = hidden[0]
    z = {**o, "priv_code": np.zeros_like(o["priv_code"])}
    assert not np.allclose(pv.values(o), pv.values(z))


def test_priv_value_rejects_non_priv_ckpt(tmp_path):
    from decklight.eval.value import PrivValueNet

    ckpt, emb = _tiny_critic(tmp_path, priv=False)
    with pytest.raises(ValueError):
        PrivValueNet(ckpt, emb)


def test_margin_keeps_policy_move_unless_clear(engine):
    """A margin above any score gap plays the policy's move at every searched decision (overrides 0, log keeps the
    search's best); margin 0 logs chosen == best."""
    res, _ = _play(engine, _tiny_forward(), conf_skip=1.01, own_frac=1.0, margin=1e9)
    assert res.log and res.n_override == 0
    assert all(e["chosen"] == e["played"] for e in res.log if e["played"] in e["cands"])
    res0, _ = _play(engine, _tiny_forward(), conf_skip=1.01, own_frac=1.0)
    assert all(e["chosen"] == e["best"] for e in res0.log)


def test_turn_limits_for_label_generation(engine):
    """max_search_turn bounds the searched turns; stop_turn ends the game after that turn (truncated, no winner)."""
    res, _ = _play(engine, _tiny_forward(), conf_skip=1.01, own_frac=1.0, max_search_turn=2, stop_turn=3)
    assert res.log and all(e["turn"] <= 2 for e in res.log)
    assert max(res.decision_turns) <= 3
    if res.truncated:
        assert res.winner is None


def test_max_searches_caps_then_plays_raw(engine):
    """max_searches=2: at most 2 search attempts, the game still ends, and search_cut marks the raw remainder
    (a looping game slowed a proc for 45+ min; NOTEBOOK 2026-10-11 00:10)."""
    fw = _tiny_forward()
    full, _ = _play(engine, fw, conf_skip=1.01, own_frac=1.0)
    assert full.n_searched > 2 and not full.search_cut
    res, _ = _play(engine, fw, conf_skip=1.01, own_frac=1.0, max_searches=2)
    assert res.n_searched + res.det_skipped == 2
    assert res.search_cut
    assert res.winner in (0, 1) and not res.truncated
