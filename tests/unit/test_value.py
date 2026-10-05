"""Learned value v2: outcome shards, value training, calibration
(scripts/build_value_shards.py, scripts/train_value.py, src/ygo_zero/eval/value.py).

Only a tiny randomly initialised model (d=32, layers=1) is instantiated here (RAM rule).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build_value_shards = _load("build_value_shards")
train_value = _load("train_value")


def _write_games(engine, tmp_path: Path) -> Path:
    """3 finished random-policy games (winner from the env) + 1 cut game (winner 3)."""
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig

    decks = sorted((FIXTURES / "decks").glob("*.ydk"))
    out = tmp_path / "rec" / "games.jsonl"
    out.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)

    def play(seed: int, cap: int):
        d0, d1 = load_ydk(decks[seed % len(decks)]), load_ydk(decks[(seed + 1) % len(decks)])
        cfg = DuelConfig(d0, d1, seed=seed)
        env = CppEnv(engine, cfg)
        obs, ch, act, tu = env.reset(), [], [], []
        while obs is not None and len(ch) < cap:
            legal = np.flatnonzero(obs["action_mask"])
            a = int(rng.choice(legal))
            ch.append(a)
            act.append(int(env.player))
            tu.append(int(obs["global"][2]))
            obs = env.step(a)
        conf = {"deck0_main": list(d0.main), "deck0_extra": list(d0.extra), "deck1_main": list(d1.main),
                "deck1_extra": list(d1.extra), "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw,
                "lp": cfg.lp, "setup": [], "shuffle": cfg.shuffle}  # fmt: skip
        done = obs is None
        return conf, ch, act, tu, (int(env.winner) if done else None)

    with out.open("w") as f:
        finished, seed = 0, 50
        while finished < 3 and seed < 110:  # random games run 300-700 decisions; the 400 cap never finishes
            conf, ch, act, tu, w = play(seed, 1500)
            seed += 1
            if w not in (0, 1):
                continue  # unfinished or draw: not a usable outcome label
            f.write(json.dumps({"game_id": f"v{finished}", "line_id": f"ta{finished}-vs-tb{finished}/v{finished}",
                                "decks": [f"ta{finished}", f"tb{finished}"], "config": conf, "auto_single": True,
                                "choices": ch, "actors": act, "turns": tu, "winner": w}) + "\n")  # fmt: skip
            finished += 1
        assert finished == 3
        conf, ch, act, tu, _ = play(99, 10)  # cut after at most 10 decisions
        f.write(json.dumps({"game_id": "vcut", "line_id": "taX-vs-tbX/vcut", "decks": ["taX", "tbX"],
                            "config": conf, "auto_single": True, "choices": ch, "actors": act,
                            "turns": tu, "winner": 3}) + "\n")  # fmt: skip
    return out


def _build(src: Path, tmp_path: Path) -> Path:
    out = tmp_path / "vshards"
    old = sys.argv
    sys.argv = ["build_value_shards", "--records", str(src), "--out", str(out),
                "--stride", "2", "--jobs", "1"]
    try:
        build_value_shards.main()
    finally:
        sys.argv = old
    return out


def test_value_shards_label_z_and_skip_cut(engine, tmp_path):
    from ygo_zero.model.bc_data import OBS_KEYS

    src = _write_games(engine, tmp_path)
    out = _build(src, tmp_path)
    shards = sorted(out.glob("*.npz"))
    assert len(shards) == 1
    p = dict(np.load(shards[0]))
    for k in OBS_KEYS:
        assert k in p
    for k in ("z", "turn", "game", "deck", "deck0", "deck1", "actor", "src", "hscore", "game_ids", "game_row"):
        assert k in p, k
    game_ids = [str(x) for x in p["game_ids"].tolist()]
    assert "vcut" not in game_ids  # cut games are skipped
    assert len(game_ids) == 3
    gid_of = {int(g): str(s) for g, s in zip(p["game_row"].tolist(), game_ids, strict=True)}
    recs = {r["game_id"]: r for r in map(json.loads, src.read_text().splitlines())}
    assert len(p["z"]) > 0
    seen: dict[str, set[int]] = {}
    for z, a, g in zip(p["z"].tolist(), p["actor"].tolist(), p["game"].tolist(), strict=True):
        gs = gid_of[g]
        seen.setdefault(gs, set()).add(g)
        rec = recs[gs]
        assert z == (1.0 if a == rec["winner"] else -1.0)  # z matches the actor vs the winner
    assert all(v == {next(iter(v))} for v in seen.values())  # one game id per game


def _tiny_emb(path: Path):
    rng = np.random.default_rng(0)
    n, d = 50, 8
    np.savez(path, codes=np.arange(1, n + 1), card=rng.normal(size=(n, d)).astype(np.float32),
             effects=rng.normal(size=(n, 2, d)).astype(np.float32), n_effects=np.ones(n, dtype=np.int64))


def test_value_train_and_load(engine, tmp_path):
    import torch

    from ygo_zero.eval.value import ValueNet
    from ygo_zero.model.bc_data import OBS_KEYS
    from ygo_zero.model.policy import PolicyNet

    src = _write_games(engine, tmp_path)
    shards = _build(src, tmp_path)
    emb = tmp_path / "emb.npz"
    _tiny_emb(emb)
    init = tmp_path / "init"
    init.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    net = PolicyNet(dict(np.load(emb)), d=32, layers=1, heads=2)
    torch.save({k: v for k, v in net.state_dict().items()}, init / "best.pt")
    (init / "metrics.jsonl").write_text(json.dumps({"args": {"d": 32, "layers": 1, "heads": 2}}) + "\n")
    del net

    out = tmp_path / "value"
    old = sys.argv
    sys.argv = ["train_value", "--shards", str(shards), "--emb", str(emb), "--init", str(init / "best.pt"),
                "--out", str(out), "--epochs", "1", "--batch", "8", "--device", "cpu", "--threads", "1",
                "--holdout", "ta0", "--seed", "0"]
    try:
        train_value.main()
    finally:
        sys.argv = old
    recs = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x.strip()]
    assert recs[0]["epoch"] == -1 and recs[0]["value"] is True and recs[0]["args"]["d"] == 32
    last = recs[-1]
    assert last["epoch"] == 0 and np.isfinite(last["train_loss"])
    assert "train" in last and "holdout" in last and last["holdout"]["n"] > 0  # game 0 (deck ta0)
    for split, m in last.items():
        if split in ("epoch", "time", "train_loss", "args"):
            continue
        assert m["n"] > 0
        for k in ("mse", "acc", "acc_turn", "cal", "hacc"):
            assert k in m, (split, k)
        assert np.isfinite(m["mse"]) and 0 <= m["acc"] <= 1 and 0 <= m["hacc"] <= 1
        assert len(m["cal"]["bins"]) == 10 and np.isfinite(m["cal"]["ece"])
    assert (out / "best.pt").is_file() and (out / "last.pt").is_file()

    vn = ValueNet(out / "best.pt", emb)
    p = dict(np.load(min(shards.glob("*.npz"))))
    batch = {k: np.ascontiguousarray(p[k][:8]) for k in OBS_KEYS}
    v = vn.values(batch)
    assert v.shape == (8,) and np.all(np.isfinite(v)) and (v >= -1).all() and (v <= 1).all()
