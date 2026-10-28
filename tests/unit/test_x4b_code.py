"""X4b code: multi-list fine-tune, --opp-decks self-play, per-deck distillation teachers.

CPU only, tiny tensors; no real model training or games beyond stub self-play.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"x4b_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"x4b_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


train_rl = _load("train_rl")
train_bc = _load("train_bc")
gen_selfplay = _load("gen_selfplay")


def _deck(main, extra=()):
    from decklight.decks import Deck

    return Deck(tuple(main), tuple(extra), (), "")


# --- 1. multi-list focus ---


def test_apply_focus_round_robin():
    pool = [_deck([i]) for i in range(10)]
    gaunt = [_deck([100 + i]) for i in range(5)]
    foci = [_deck([901]), _deck([902]), _deck([903])]
    train, idx, new_pool, new_gaunt = train_rl.apply_focus_deck(pool, gaunt, foci, 0.5, 0)
    assert len(idx) == 10  # k = round(0.5*10/0.5), same as the single-deck case
    assert len(new_pool) == 20 and new_gaunt == gaunt
    slots = [train[i] for i in sorted(idx)]
    assert slots == [foci[i % 3] for i in range(10)]  # the focus set splits the share evenly
    assert len(idx) / len(new_pool) == pytest.approx(0.5)


def test_apply_focus_dedups_each_focus_deck():
    pool = [_deck([i]) for i in range(10)]
    gaunt = [_deck([100 + i]) for i in range(5)]
    foci = [_deck([1]), _deck([2]), _deck([999])]
    _train, idx, new_pool, _new_gaunt = train_rl.apply_focus_deck(pool, gaunt, foci, 0.5, 0)
    assert len(new_pool) == 16  # 8 others + k=8 copies
    rest = [d for i, d in enumerate(new_pool) if i not in idx]
    assert all(not any(train_rl.same_deck(d, f) for f in foci) for d in rest)
    assert len(idx) / len(new_pool) == pytest.approx(0.5)


def test_apply_focus_k_lt_nfocus_raises():
    pool = [_deck([i]) for i in range(2)]
    foci = [_deck([901]), _deck([902]), _deck([903])]
    with pytest.raises(ValueError, match="focus"):
        train_rl.apply_focus_deck(pool, [], foci, 0.1, 0)  # k=1 < 3 focus decks


def test_apply_focus_single_list_matches_scalar():
    pool = [_deck([i]) for i in range(10)]
    gaunt = [_deck([100 + i]) for i in range(5)]
    focus = _deck([999])
    a = train_rl.apply_focus_deck(pool, gaunt, focus, 0.5, 0)
    b = train_rl.apply_focus_deck(pool, gaunt, [focus], 0.5, 0)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
    g1 = train_rl.apply_focus_deck(pool, gaunt, focus, 0.5, 1)
    g2 = train_rl.apply_focus_deck(pool, gaunt, [focus], 0.5, 1)
    assert g1[0] == g2[0] and g1[1] == g2[1]


# --- 3. per-deck distillation ---


def _tiny_net(seed=0):
    from decklight.model.policy import PolicyNet

    rng = np.random.default_rng(42)
    codes = np.arange(1, 50)
    n = len(codes)
    emb = {"codes": codes, "card": rng.normal(size=(n, 8)).astype(np.float32),
           "effects": rng.normal(size=(n, 2, 8)).astype(np.float32), "n_effects": np.ones(n, np.int64)}
    torch.manual_seed(seed)
    net = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _tiny_obs(batch=6, seed=1, n_legal=4):
    from decklight.model.bc_data import OBS_KEYS

    torch.manual_seed(seed)
    shapes = {"cards": (batch, 160, 21), "card_mask": (batch, 160), "global": (batch, 24),
              "chain": (batch, 16, 5), "chain_mask": (batch, 16), "actions": (batch, 64, 10),
              "action_mask": (batch, 64), "history": (batch, 32, 13), "history_mask": (batch, 32)}
    obs = {}
    for k in OBS_KEYS:
        s = shapes[k]
        if k.endswith("_mask"):
            m = torch.zeros(s, dtype=torch.bool)
            m[:, :n_legal] = True
            obs[k] = m
        else:
            obs[k] = torch.randint(0, 4, s)
    obs["actions"] = torch.randint(0, 4, (batch, 64, 10))
    return obs


def test_per_deck_kl_uses_own_teacher():
    student = _tiny_net(0)
    teachers = [_tiny_net(1), _tiny_net(2)]
    obs = _tiny_obs()
    bt = np.array([0, 1, 0, 1, 0, 1])
    with torch.no_grad():
        slogits = student(obs)
    kl = train_bc.batch_teacher_kl(slogits, obs, bt, teachers)
    want = []
    for i in range(6):
        with torch.no_grad():
            r = teachers[bt[i]]({k: v[i : i + 1] for k, v in obs.items()})
        want.append(train_bc.kl_to_init(slogits[i : i + 1].float(), r.float(),
                                       obs["action_mask"][i : i + 1]).item())
    np.testing.assert_allclose(kl.detach().numpy(), np.array(want), rtol=1e-4, atol=1e-6)


def test_ce_coef_zero_is_pure_kl():
    torch.manual_seed(0)
    per = torch.rand(6)
    w = torch.ones(6)
    kl = torch.rand(6)
    loss = train_bc.mixed_loss(per, w, kl, kl_coef=2.0, ce_coef=0.0)
    assert loss.item() == (2.0 * kl.mean()).item()  # bit for bit, not just close


def test_default_ce_coef_matches_old_loss_bit_for_bit():
    torch.manual_seed(1)
    per = torch.rand(8)
    w = torch.tensor([1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0])
    kl = torch.rand(8)
    old = (per * w).sum() / w.sum() + 0.7 * kl.mean()
    new = train_bc.mixed_loss(per, w, kl, kl_coef=0.7, ce_coef=1.0)
    assert new.item() == old.item()  # default args: today's loss exactly


def _synthetic_shards(path: Path, decks, n: int = 32, seed: int = 0):
    from decklight.model.bc_data import OBS_KEYS

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_legal = 4
    data: dict = {}
    shapes = {"cards": (n, 160, 21), "card_mask": (n, 160), "global": (n, 24),
              "chain": (n, 16, 5), "chain_mask": (n, 16), "actions": (n, 64, 10),
              "action_mask": (n, 64), "history": (n, 32, 13), "history_mask": (n, 32)}
    for k in OBS_KEYS:
        s = shapes[k]
        data[k] = (rng.integers(0, 2, size=s).astype(bool) if k.endswith("_mask")
                   else rng.integers(0, 4, size=s).astype(np.int64))
        if k.endswith("_mask"):
            data[k][:, :n_legal] = True
    data["actions"] = rng.integers(0, 4, size=(n, 64, 10)).astype(np.int64)
    act = rng.integers(0, n_legal, size=n)
    eq = np.zeros((n, 64), dtype=bool)
    eq[np.arange(n), act] = True
    data.update(action=act.astype(np.int64), equiv=eq, weight=np.ones(n, np.float32),
                line=np.array([f"l{i % 8}" for i in range(n)]),
                deck=np.array([decks[i % len(decks)] for i in range(n)]),
                kind=np.array(["x"] * n), src=np.array(["t"] * n))
    path.mkdir(parents=True)
    np.savez_compressed(path / "syn.npz", **data)


def _tiny_emb(path: Path):
    rng = np.random.default_rng(42)
    codes = np.arange(1, 50)
    n = len(codes)
    np.savez_compressed(path, codes=codes, card=rng.normal(size=(n, 8)).astype(np.float32),
                        effects=rng.normal(size=(n, 2, 8)).astype(np.float32),
                        n_effects=np.ones(n, np.int64))


def _run_bc(argv):
    old = sys.argv
    sys.argv = argv
    try:
        train_bc.main()
    finally:
        sys.argv = old


def test_map_missing_teacher_ckpt_exits_2(tmp_path):
    m = {"default": str(tmp_path / "missing.pt"), "decks": {"d0": str(tmp_path / "missing.pt")}}
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit) as e:
        _run_bc(["train_bc", "--shards", "x", "--emb", "y", "--out", str(tmp_path / "o"),
                 "--kl-ref-map", str(mp)])
    assert e.value.code == 2


def test_map_zero_deck_entries_exits_2(tmp_path):
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"default": "wherever.pt", "decks": {}}))
    with pytest.raises(SystemExit) as e:
        _run_bc(["train_bc", "--shards", "x", "--emb", "y", "--out", str(tmp_path / "o"),
                 "--kl-ref-map", str(mp)])
    assert e.value.code == 2


def test_map_deck_without_rows_exits_2(tmp_path, capsys):
    shards = tmp_path / "shards"
    _synthetic_shards(shards, ["d0", "d1"])
    _tiny_emb(tmp_path / "emb.npz")
    teacher = tmp_path / "t.pt"
    teacher.write_bytes(b"placeholder")  # exists, so the startup check passes; never loaded
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"default": str(teacher), "decks": {"nosuchdeck": str(teacher)}}))
    with pytest.raises(SystemExit) as e:
        _run_bc(["train_bc", "--shards", str(shards), "--emb", str(tmp_path / "emb.npz"),
                 "--out", str(tmp_path / "o"), "--kl-ref-map", str(mp)])
    assert e.value.code == 2
    assert "ALERT train_bc" in capsys.readouterr().out


def test_ce_coef_without_teacher_exits_2():
    with pytest.raises(SystemExit) as e:
        _run_bc(["train_bc", "--shards", "x", "--emb", "y", "--out", "z", "--ce-coef", "0.5"])
    assert e.value.code == 2


# --- 2. --opp-decks self-play ---


def test_opp_decks_split(engine, tmp_path):
    stu, opp = tmp_path / "stu", tmp_path / "opp"
    stu.mkdir()
    opp.mkdir()
    for f in ("BlueEyes.ydk", "Branded.ydk"):
        shutil.copy(FIXTURES / "decks" / f, stu / f)
    for f in ("Labrynth.ydk", "SkyStrikerAce.ydk"):
        shutil.copy(FIXTURES / "decks" / f, opp / f)
    out = tmp_path / "sp"
    old = sys.argv
    sys.argv = ["gen_selfplay", "--out", str(out), "--games", "4",
                "--decks", str(stu), "--opp-decks", str(opp),
                "--ygoagent-decks", str(FIXTURES / "decks"),
                "--policy", "stub", "--league", "self:0.5,greedy:0.5",
                "--seed", "3", "--threads", "2"]
    try:
        gen_selfplay.main()
    finally:
        sys.argv = old
    recs = [json.loads(x) for f in sorted(out.glob("*.jsonl"))
            for x in f.read_text().splitlines() if x.strip()]
    assert len(recs) == 4
    stu_stems = {"BlueEyes", "Branded"}
    opp_stems = {"Labrynth", "SkyStrikerAce"}
    for r in recs:
        pre, _, _gid = r["line_id"].partition("/")
        s_stu, _, s_opp = pre.partition("-vs-")
        assert s_stu in stu_stems, r["line_id"]  # student deck from --decks
        assert s_opp in opp_stems, r["line_id"]  # opponent deck from --opp-decks
        seat = r["student_seat"]
        assert r["decks"][seat] == s_stu and r["decks"][1 - seat] == s_opp
        assert Path(out / f"{s_stu}.jsonl").is_file()  # filed under the student deck, as today
