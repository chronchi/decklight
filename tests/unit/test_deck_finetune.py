"""Per-deck fine-tuning + per-deck card (Part E): focus list, finetune invariants, deal list, summarize pairing.

Fast: no GPU, no training runs, no real games.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from decklight.decks import Deck

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    p = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _deck(main, extra=()) -> Deck:
    return Deck(tuple(main), tuple(extra), (), "")


# --- Part A: focus-list construction ---


def test_focus_copies_share():
    tr = _load("train_rl")
    k = tr.focus_copies(180, 0.5)
    assert k == 180
    assert k / (180 + k) == pytest.approx(0.5)
    k2 = tr.focus_copies(180, 0.25)
    assert k2 == 60
    assert k2 / (180 + k2) == pytest.approx(0.25)


def test_focus_copies_rejects():
    tr = _load("train_rl")
    with pytest.raises(ValueError):
        tr.focus_copies(180, 0.0)
    with pytest.raises(ValueError):
        tr.focus_copies(180, 1.0)
    with pytest.raises(ValueError):
        tr.focus_copies(0, 0.5)


def test_same_deck_order_insensitive():
    tr = _load("train_rl")
    assert tr.same_deck(_deck([1, 2], [3]), _deck([2, 1], [3]))
    assert not tr.same_deck(_deck([1, 2], [3]), _deck([1, 2], [4]))
    assert not tr.same_deck(_deck([1, 2]), _deck([1, 3]))


def test_apply_focus_deck_overfitting_rule():
    tr = _load("train_rl")
    pool = [_deck([i]) for i in range(10)]
    gaunt = [_deck([100 + i]) for i in range(5)]
    focus = _deck([999])
    train, idx, new_pool, new_gaunt = tr.apply_focus_deck(pool, gaunt, focus, 0.5, 0)
    assert len(idx) == 10  # k = round(0.5*10/0.5)
    assert len(new_pool) == 20 and len(new_gaunt) == 5
    assert train == new_pool + new_gaunt
    assert idx == set(range(10, 20))
    # every focus index holds the focus deck; the share over the drawn pool region is 0.5
    assert all(train[i] == focus for i in idx)
    assert len(idx) / len(new_pool) == pytest.approx(0.5)


def test_apply_focus_deck_dedups_focus():
    tr = _load("train_rl")
    pool = [_deck([i]) for i in range(10)]
    gaunt = [_deck([100 + i]) for i in range(5)]
    focus = _deck([3])  # identical to pool[3]
    train, idx, new_pool, new_gaunt = tr.apply_focus_deck(pool, gaunt, focus, 0.5, 0)
    assert len(new_pool) == 18  # 9 others + 9 copies
    assert all(not tr.same_deck(d, focus) for d in new_pool if d != focus)
    assert len(idx) / len(new_pool) == pytest.approx(0.5)
    assert train == new_pool + new_gaunt


def test_apply_focus_deck_learner_gauntlet():
    tr = _load("train_rl")
    pool = [_deck([i]) for i in range(10)]
    gaunt = [_deck([100 + i]) for i in range(5)]
    focus = _deck([999])
    train, idx, _new_pool, _new_gaunt = tr.apply_focus_deck(pool, gaunt, focus, 0.5, 1)
    assert len(idx) == 15  # k = round(0.5*15/0.5), copies appended at the end
    assert train == pool + gaunt + [focus] * 15
    assert idx == set(range(15, 30))
    assert len(idx) / len(train) == pytest.approx(0.5)


def test_focus_log_fields():
    tr = _load("train_rl")
    rec = tr.focus_log_fields(50, 100, 20, 12)
    assert rec == {"focus_seat_share": 0.5, "focus_wr_vs_bc": 0.6, "focus_n_vs_bc": 20}
    rec0 = tr.focus_log_fields(0, 0, 0, 0)
    assert rec0 == {"focus_seat_share": 0.0, "focus_wr_vs_bc": 0.0, "focus_n_vs_bc": 0}


# --- Part B: finetune_deck invariant checks ---


def _ckpt(path: Path, policy_val: float, value_val: float, raw: bool = False):
    state = {"trunk.weight": torch.full((2,), policy_val),
             "value_head.weight": torch.full((2,), value_val)}
    torch.save(state if raw else {"model": state}, path)


def test_check_invariants_identical_policy_fails(tmp_path):
    ft = _load("finetune_deck")
    base = tmp_path / "base.pt"
    val = tmp_path / "value.pt"
    final = tmp_path / "final.pt"
    _ckpt(base, 1.0, 0.0)
    _ckpt(val, 9.0, 2.0)
    _ckpt(final, 1.0, 3.0)  # policy == P exactly
    with pytest.raises(SystemExit):
        ft.check_invariants(final, base, val)


def test_check_invariants_identical_value_fails(tmp_path):
    ft = _load("finetune_deck")
    base = tmp_path / "base.pt"
    val = tmp_path / "value.pt"
    final = tmp_path / "final.pt"
    _ckpt(base, 1.0, 0.0)
    _ckpt(val, 9.0, 2.0)
    _ckpt(final, 1.5, 2.0)  # policy trained, value head == value_p2 exactly
    with pytest.raises(SystemExit):
        ft.check_invariants(final, base, val)


def test_check_invariants_trained_passes(tmp_path):
    ft = _load("finetune_deck")
    base = tmp_path / "base.pt"
    val = tmp_path / "value.pt"
    final = tmp_path / "final.pt"
    _ckpt(base, 1.0, 0.0)
    _ckpt(val, 9.0, 2.0)
    _ckpt(final, 1.5, 2.5)
    ft.check_invariants(final, base, val)  # no raise


# --- Part D: deck_card deals ---


def _gauntlet(n: int = 31) -> list[tuple[str, Deck]]:
    return [(f"d{i:02d}", _deck([i])) for i in range(n)]


def test_build_deals_target_never_opponent():
    dc = _load("deck_card")
    gaunt = _gauntlet()
    target = _deck([5])
    stem, opps = dc.find_target(target, gaunt)
    assert stem == "d05"
    assert len(opps) == 30
    assert stem not in opps
    seeds = 8
    deals = dc.build_deals(stem, opps, seeds, 9000)
    assert len(deals) == 30 * 2 * seeds
    assert all(d["target"] == stem and d["target"] != d["opp"] for d in deals)
    for opp in opps:
        sub = [d for d in deals if d["opp"] == opp]
        assert len(sub) == 2 * seeds
        assert {d["seat"] for d in sub} == {0, 1}  # both seats
        assert sorted({d["seed"] for d in sub}) == list(range(9000 + opps.index(opp) * seeds,
                                                              9000 + (opps.index(opp) + 1) * seeds))


def test_find_target_missing_fails():
    dc = _load("deck_card")
    with pytest.raises(SystemExit):
        dc.find_target(_deck([9999]), _gauntlet())


def test_parse_arm():
    dc = _load("deck_card")
    name, ckpt, value = dc.parse_arm("ft=runs/x.pt:runs/y.pt")
    assert (name, str(value)) == ("ft", "runs/y.pt")
    assert str(ckpt) == "runs/x.pt"
    name, ckpt, value = dc.parse_arm("raw=runs/x.pt:")
    assert value is None
    with pytest.raises(SystemExit):
        dc.parse_arm("noequals")
    with pytest.raises(SystemExit):
        dc.parse_arm("a=b=c")


def test_pairs_schedule_s_always_first():
    sm = _load("search_match")
    names = ["T", "O1", "O2"]
    sched = sm.pairs_schedule(names, [("T", "O1"), ("T", "O2")], 2, 9000, True)
    assert len(sched) == 2 * 2 * 2  # pairs x seeds x seats
    assert all(di == 0 for di, _, _, _, _ in sched)  # S always plays T (index 0)
    assert {(dj, s, seat) for _, dj, s, seat, _ in sched} == {
        (1, 9000, 0), (1, 9000, 1), (1, 9001, 0), (1, 9001, 1),
        (2, 9002, 0), (2, 9002, 1), (2, 9003, 0), (2, 9003, 1)}
    assert [idx for *_, idx in sched] == list(range(8))
    # without seed-per-pair every pair shares the seed range
    sched2 = sm.pairs_schedule(names, [("T", "O1"), ("T", "O2")], 2, 9000, False)
    assert sorted({s for _, _, s, _, _ in sched2}) == [9000, 9001]
    with pytest.raises(SystemExit):
        sm.pairs_schedule(names, [("T", "Nope")], 2, 9000, True)


# --- Part D: summarize pairing ---


def _write_games(path: Path, wins: dict[tuple, float]):
    with open(path, "w") as f:
        f.writelines(json.dumps({"decks": list(decks), "seed": seed, "s_seat": seat,
                                 "winner": seat if w == 1.0 else 1 - seat,
                                 "truncated": False}) + "\n"
                     for (decks, seed, seat), w in wins.items())


def test_summarize_pairing(tmp_path):
    dc = _load("deck_card")
    out = tmp_path / "card"
    keys = [(("T", f"O{i}"), 9000 + i, seat) for i in range(2) for seat in (0, 1)]
    a = {k: 1.0 for k in keys}
    a[keys[0]] = 0.0  # arm a wins 3/4
    b = {k: 0.0 for k in keys}
    b[keys[0]] = 1.0  # arm b wins 1/4 (paired flips)
    for name, wins in (("a", a), ("b", b)):
        (out / name).mkdir(parents=True)
        _write_games(out / name / "games-0.jsonl", wins)
    ckpt_a, ckpt_b = tmp_path / "a.pt", tmp_path / "b.pt"
    ckpt_a.write_bytes(b"a")
    ckpt_b.write_bytes(b"b")
    card = dc.summarize(out, [("a", ckpt_a, None), ("b", ckpt_b, None)])
    assert card["arms"]["a"]["win_rate"] == pytest.approx(0.75)
    assert card["arms"]["a"]["n"] == 4
    assert card["arms"]["b"]["win_rate"] == pytest.approx(0.25)
    assert card["diff_vs_first"]["first"] == "a"
    assert card["diff_vs_first"]["b"]["diff_pp"] == pytest.approx(-50.0)
    assert card["diff_vs_first"]["b"]["n"] == 4
    assert card["seats"]["first"]["a"]["n"] == 2
    assert card["seats"]["second"]["b"]["n"] == 2
    assert (out / "card.json").is_file()


def test_summarize_identical_checkpoints_fail(tmp_path):
    dc = _load("deck_card")
    out = tmp_path / "card"
    keys = [(("T", "O0"), 9000, seat) for seat in (0, 1)]
    for name in ("a", "b"):
        (out / name).mkdir(parents=True)
        _write_games(out / name / "games-0.jsonl", {k: 1.0 for k in keys})
    ckpt = tmp_path / "same.pt"
    ckpt.write_bytes(b"same")
    with pytest.raises(SystemExit):
        dc.summarize(out, [("a", ckpt, None), ("b", ckpt, None)])


def test_summarize_different_deals_fail(tmp_path):
    dc = _load("deck_card")
    out = tmp_path / "card"
    (out / "a").mkdir(parents=True)
    (out / "b").mkdir(parents=True)
    _write_games(out / "a" / "games-0.jsonl", {(("T", "O0"), 9000, 0): 1.0})
    _write_games(out / "b" / "games-0.jsonl", {(("T", "O1"), 9000, 0): 1.0})
    ca, cb = tmp_path / "a.pt", tmp_path / "b.pt"
    ca.write_bytes(b"a")
    cb.write_bytes(b"b")
    with pytest.raises(SystemExit):
        dc.summarize(out, [("a", ca, None), ("b", cb, None)])


def test_summarize_zero_games_fail(tmp_path):
    dc = _load("deck_card")
    out = tmp_path / "card"
    (out / "a").mkdir(parents=True)
    (out / "a" / "games-0.jsonl").write_text("")
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"a")
    with pytest.raises(SystemExit):
        dc.summarize(out, [("a", ckpt, None)])


def test_build_arm_cmd_has_device():
    dc = _load("deck_card")
    cmd = dc.build_arm_cmd(Path("scripts/search_match.py"), Path("a.pt"), Path("v.pt"),
                           1, "T:O", Path("decks/gauntlet"), 8, 9000, 4, 0, Path("out/ft"), "cuda")
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "cuda"
    cmd0 = dc.build_arm_cmd(Path("scripts/search_match.py"), Path("a.pt"), None,
                            0, "T:O", Path("decks/gauntlet"), 8, 9000, 4, 1, Path("out/base"), "cpu")
    assert "--device" in cmd0 and cmd0[cmd0.index("--device") + 1] == "cpu"
    assert "--raw" in cmd0
