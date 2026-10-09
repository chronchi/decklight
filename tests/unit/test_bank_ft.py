"""bank_ft: split by start, fine-tune (tiny d=16 net), play from banked starts, paired report."""

import json

from test_hint_bank import _load, _starts_file, _tiny_ckpt


def _bank(engine, tmp_path):
    """4 bank rows = 2 starts x 2 actor seats; each row is 8 decisions of option 0 from the deal."""
    from ygo_zero.eval.turn1 import config_to_dict, dags_config, replay_prefix

    rows = []
    for sid, st in enumerate(json.loads(x) for x in _starts_file(tmp_path).read_text().splitlines()):
        cfg = dags_config(st)
        for seat in (0, 1):
            env, obs = replay_prefix(engine, cfg, [])
            choices, actors = [], []
            for _ in range(8):
                actors.append(int(env.player))
                choices.append(0)
                obs = env.step(0)
                assert obs is not None
            rows.append({"start_id": sid, "config": config_to_dict(cfg), "prefix": [], "choices": choices,
                         "actors": actors, "actor_seat": seat, "actor_win": True})
    p = tmp_path / "bank.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_split_is_by_start_and_seeded():
    bf = _load("bank_ft")
    rows = [{"start_id": s, "actor_seat": a} for s in range(10) for a in (0, 1) for _ in range(3)]
    tr, ho = bf.split_bank(rows, 0.25, seed=0)
    assert len(ho) == 5 and len(tr) == 15 and not set(tr) & set(ho)
    assert (tr, ho) == bf.split_bank(rows, 0.25, seed=0)


def test_train_play_report_end_to_end(engine, tmp_path, monkeypatch):
    bf = _load("bank_ft")
    ckpt, emb = _tiny_ckpt(tmp_path)
    bank = _bank(engine, tmp_path)
    out = tmp_path / "ft"
    run = lambda *a: (monkeypatch.setattr("sys.argv", ["bank_ft.py", *map(str, a)]), bf.main())
    run("train", "--bank", bank, "--init", ckpt, "--emb", emb, "--out", out, "--epochs", "2", "--lr", "1e-2",
        "--batch", "4")
    recs = [json.loads(x) for x in (out / "ft/metrics.jsonl").read_text().splitlines()]
    assert recs[0]["args"]["d"] == 16 and recs[1]["epoch"] == "init" and recs[-1]["epoch"] == 2
    assert recs[-1]["train"]["ce"] < recs[1]["train"]["ce"]  # it fits the train decisions
    assert (out / "ft/best.pt").is_file() and (out / "ft/final.pt").is_file()
    split = json.loads((out / "split.json").read_text())
    assert len(split["train"]) == 3 and len(split["held"]) == 1
    for proc in (0, 1):
        run("play", "--out", out, "--opp", ckpt, "--emb", emb, "--games", "1", "--max-decisions", "40",
            "--nproc", "2", "--proc", proc)
    plays = [json.loads(x) for f in sorted(out.glob("play-*.jsonl")) for x in f.read_text().splitlines()]
    assert len(plays) == 4 * 3 * 1 * 2  # starts x actors (base, ft, ftfinal) x opponents x (greedy + 1 sampled)
    run("report", "--out", out)
    rep = json.loads((out / "report.json").read_text())
    assert set(rep) == {f"{sp}/opp/{m}/{a}" for sp in ("train", "held") for m in ("greedy", "sampled")
                        for a in ("ft", "ftfinal")}
    assert rep["train/opp/greedy/ft"]["n"] == 3
