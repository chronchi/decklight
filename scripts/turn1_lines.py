"""Turn-1 line generation + scoring (self-imitation step 1: headroom measurement, later the data generator).

    python scripts/turn1_lines.py --ckpt <policy best.pt> --critic <critic.pt> --decks <split> --hands H --k K \
        --seed S --out <dir> [--procs P] [--threads T] [--device cuda] [--max-decks N]

For each deck and hand: 1 greedy + K sampled (temperature 1) turn-1 lines (player 0 acting, opponent
passive), scored by end field and by the privileged critic's value of the turn-2 position. Decks with
verified going-first records (train/holdout splits) also replay the expert line on its own config.
Outputs replays only (config + choices): <out>/lines.jsonl + <out>/summary.json (+ printed table).

Gauntlet decks are EVAL ONLY: their rows carry "train_ok": false, like holdout (train split: true).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import Deck, load_ydk  # noqa: E402
from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import DuelConfig, Engine  # noqa: E402
from ygo_zero.eval.bc_policy import BCPolicy  # noqa: E402
from ygo_zero.eval.turn1 import (  # noqa: E402
    CAP,
    best_of,  # noqa: F401 (re-exported for tests)
    config_from_dict,
    config_to_dict,
    critic_vp0,
    fail_loud,
    hand_config,
    load_critic_local,
    my_field,  # noqa: F401 (re-exported for tests)
    opp_for,
    resolve_decks,
    run_args,
    run_turn1,
    replay_turn1,
    stable_seed,
    summarize,
    verified_records,
)

_W: dict = {}


def _init(cfg: dict) -> None:
    """Per-worker state (spawn): its own engine, greedy + sampler policies and privileged critic."""
    import torch as _torch

    _torch.set_num_threads(cfg["threads"])
    engine = Engine()
    greedy = BCPolicy(cfg["ckpt"], cfg["emb"], temperature=0.0, threads=cfg["threads"], device=cfg["device"])
    sampler = BCPolicy(cfg["ckpt"], cfg["emb"], temperature=1.0, seed=cfg["seed"],
                       threads=cfg["threads"], device=cfg["device"])
    emb = dict(np.load(cfg["emb"]))
    critic = load_critic_local(Path(cfg["critic"]), emb, run_args(Path(cfg["critic"])))
    critic.to(_torch.device(cfg["device"])).eval()
    _W.update(engine=engine, greedy=greedy, sampler=sampler, critic=critic,
              hist=int(run_args(Path(cfg["critic"])).get("hist", 0) or 0),
              device=_torch.device(cfg["device"]), cfg=cfg)


def _score(ends: list[dict]) -> list[float]:
    return critic_vp0(_W["critic"], ends, _W["hist"], _W["device"])


def _row(deck, split, hand, source, kind, sample, cfg_dict, res, v, train_ok, auto_single=True) -> dict:
    return {"deck": deck, "split": split, "hand": int(hand), "source": source, "kind": kind,
            "sample": int(sample), "config": cfg_dict, "auto_single": bool(auto_single),
            "choices": [int(c) for c in res["choices"]], "p0_decisions": int(res["p0_decisions"]),
            "field": int(res["field"]), "v_p0": float(v), "stuck": bool(res["stuck"]),
            "train_ok": bool(train_ok)}


def _deck_task(task: dict) -> list[dict]:
    """All rows for one deck: H random hands (greedy + K samples) + verified records (greedy + K + expert)."""
    from ygo_zero.demos.dataset import record_config

    cfg, engine = _W["cfg"], _W["engine"]
    deck, split = task["deck"], task["split"]
    H, K, S, train_ok = cfg["hands"], cfg["k"], cfg["seed"], task["train_ok"]
    table = {name: Deck(tuple(m), tuple(e)) for name, (m, e) in task["table"].items()}
    names = sorted(table)
    rows: list[dict] = []
    for h in range(H):
        opp = opp_for(names, deck, S, h)
        dc = hand_config(table[deck], table[opp], stable_seed(S, deck, h))
        envs = [CppEnv(engine, dc) for _ in range(1 + K)]
        _W["sampler"].rng = np.random.default_rng(stable_seed(S, deck, h, "sample", "random"))
        res = run_turn1(envs, [_W["greedy"]] + [_W["sampler"]] * K)
        vals = _score(res)
        cd = config_to_dict(dc)
        rows.append(_row(deck, split, h, "random", "greedy", -1, cd, res[0], vals[0], train_ok))
        for j in range(K):
            rows.append(_row(deck, split, h, "random", "sample", j, cd, res[1 + j], vals[1 + j], train_ok))
    if split in ("train", "holdout"):
        for i, rec in enumerate(verified_records(deck, H)):
            dc = record_config(rec)
            auto = rec.get("auto_single", True)
            try:
                envs = [CppEnv(engine, dc, auto) for _ in range(1 + K)]
            except Exception:  # noqa: BLE001 - an unreadable record config skips the record
                continue
            _W["sampler"].rng = np.random.default_rng(stable_seed(S, deck, i, "sample", "verified"))
            res = run_turn1(envs, [_W["greedy"]] + [_W["sampler"]] * K)
            try:
                exp = replay_turn1(engine, dc, [int(c) for c in rec["choices"]], auto)
                exp["choices"] = [int(c) for c in rec["choices"]]
            except Exception:  # noqa: BLE001 - a record that no longer replays is skipped
                continue
            vals = _score(res + [exp])
            cd = rec["config"]
            rows.append(_row(deck, split, i, "verified", "greedy", -1, cd, res[0], vals[0], train_ok, auto))
            for j in range(K):
                rows.append(_row(deck, split, i, "verified", "sample", j, cd, res[1 + j], vals[1 + j], train_ok, auto))
            exp_row = _row(deck, split, i, "verified", "expert", -1, cd, exp, vals[-1], train_ok, auto)
            exp_row["stuck"] = False
            rows.append(exp_row)
    return rows


def _summarize_deck(rows: list[dict], K: int) -> dict:
    klist = sorted({k for k in (1, 4, 16, K) if 1 <= k <= K})

    def hands_of(source: str) -> list[dict]:
        by_hand: dict[int, dict[str, list]] = {}
        for r in rows:
            if r["source"] != source:
                continue
            by_hand.setdefault(r["hand"], {}).setdefault(r["kind"], []).append(r)
        hands = []
        for h in sorted(by_hand):
            g = by_hand[h]
            if "greedy" not in g:
                continue
            gr = g["greedy"][0]
            hands.append({
                "greedy": (gr["field"], gr["v_p0"]),
                "samples": [(s["field"], s["v_p0"]) for s in sorted(g.get("sample", []), key=lambda s: s["sample"])],
                "expert": ((g["expert"][0]["field"], g["expert"][0]["v_p0"])
                           if g.get("expert") else None),
            })
        return hands

    rh, vh = hands_of("random"), hands_of("verified")
    rand_rows = [r for r in rows if r["source"] == "random"]
    return {"random": summarize(rh, klist), "verified": summarize(vh, klist),
            "stuck_rate": float(np.mean([r["stuck"] for r in rand_rows])) if rand_rows else None,
            "runs": len(rows)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="policy checkpoint (its metrics.jsonl gives the arch)")
    ap.add_argument("--critic", type=Path, required=True, help="privileged critic checkpoint")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--decks", required=True, help="train | holdout | gauntlet | comma-separated ydk paths")
    ap.add_argument("--hands", type=int, required=True)
    ap.add_argument("--k", type=int, required=True, help="sampled lines per hand")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-decks", type=int, default=0)
    args = ap.parse_args(argv)
    if args.hands < 1 or args.k < 1:
        print("turn1_lines: --hands and --k must be >= 1", file=sys.stderr)
        return 2
    t0 = time.time()
    split, decks, train_ok = resolve_decks(args.decks, args.seed, args.max_decks)
    if not decks:
        raise fail_loud([f"--decks {args.decks!r} resolved to 0 decks"], args.out)
    table = {}
    for name, path in decks:
        d = load_ydk(path)
        table[name] = (tuple(d.main), tuple(d.extra))
    wcfg = {"ckpt": str(args.ckpt), "critic": str(args.critic), "emb": str(args.emb),
            "seed": args.seed, "hands": args.hands, "k": args.k,
            "threads": args.threads, "device": args.device}
    tasks = [{"deck": name, "split": split, "train_ok": train_ok, "table": table} for name, _ in decks]
    if args.procs <= 1:
        _init(wcfg)
        nested = [_deck_task(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.procs, len(tasks)), initializer=_init, initargs=(wcfg,)) as pool:
            nested = pool.map(_deck_task, tasks, chunksize=1)
    rows = [r for part in nested for r in part]
    rows.sort(key=lambda r: (r["deck"], r["source"], r["hand"],
                             {"greedy": 0, "sample": 1, "expert": 2}[r["kind"]], r["sample"]))
    problems: list[str] = []
    if not rows:
        problems.append("0 rows written")
    else:
        if any(isinstance(r["v_p0"], float) and np.isnan(r["v_p0"]) for r in rows):
            problems.append("NaN critic values")
        vs = [r["v_p0"] for r in rows]
        if len(set(vs)) == 1:
            problems.append(f"critic values all identical ({vs[0]}: untrained/zero output?)")
        by_deck: dict[str, list[dict]] = {}
        for r in rows:
            by_deck.setdefault(r["deck"], []).append(r)
        for d, dr in by_deck.items():
            if all(r["stuck"] for r in dr):
                problems.append(f"deck {d}: all {len(dr)} runs stuck (cap {CAP})")
        if args.k >= 4:
            for d, dr in by_deck.items():
                gh: dict[tuple[str, int], list[dict]] = {}
                for r in dr:
                    if r["kind"] in ("greedy", "sample"):
                        gh.setdefault((r["source"], r["hand"]), []).append(r)
                if gh and all(all(s["choices"] == next(g["choices"] for g in grp if g["kind"] == "greedy")
                                  for s in grp if s["kind"] == "sample") for grp in gh.values()):
                    problems.append(f"deck {d}: greedy and every sample identical on every hand (no sampling?)")
    if problems:
        raise fail_loud(problems, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "lines.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_deck = {}
    for r in rows:
        by_deck.setdefault(r["deck"], []).append(r)
    deck_sum = {d: (_summarize_deck(dr, args.k) | {"split": split}) for d, dr in by_deck.items()}

    def pool_sum(source: str) -> dict:
        hands = []
        # pooled hands across decks of this split
        by_hand: dict[tuple[str, int], dict] = {}
        for r in rows:
            if r["source"] != source:
                continue
            by_hand.setdefault((r["deck"], r["hand"]), {}).setdefault(r["kind"], []).append(r)
        for (d_, h_), g in sorted(by_hand.items()):
            if "greedy" not in g:
                continue
            gr = g["greedy"][0]
            hands.append({"greedy": (gr["field"], gr["v_p0"]),
                          "samples": [(s["field"], s["v_p0"])
                                      for s in sorted(g.get("sample", []), key=lambda s: s["sample"])],
                          "expert": ((g["expert"][0]["field"], g["expert"][0]["v_p0"])
                                     if g.get("expert") else None)})
        klist = sorted({k for k in (1, 4, 16, args.k) if 1 <= k <= args.k})
        return summarize(hands, klist)

    rand_rows = [r for r in rows if r["source"] == "random"]
    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "split": split, "K": args.k, "cap": CAP, "seconds": round(time.time() - t0, 1),
        "decks": deck_sum,
        "splits": {split: {
            "random": pool_sum("random"), "verified": pool_sum("verified"),
            "stuck_rate": float(np.mean([r["stuck"] for r in rand_rows])) if rand_rows else None,
            "runs": len(rows)}},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    w = max([len(d) for d in deck_sum] + [4])
    print(f"{'deck':{w}s} src rnd/ver  greedy_f greedy_v " +
          " ".join(f"best{k}f" for k in sorted({k for k in (1, 4, 16, args.k) if 1 <= k <= args.k})) +
          "  expert_f expert_v gate stuck")
    for d, s in sorted(deck_sum.items()):
        r, v = s["random"], s["verified"]
        bk = [f"{r.get(f'best{k}_by_field', [float('nan')])[0]:.2f}"
              for k in sorted({k for k in (1, 4, 16, args.k) if 1 <= k <= args.k})]
        gate = f"{v.get('critic_ranks_expert_above_greedy', float('nan')):.2f}" if v.get("hands") else "-"
        ef = f"{v.get('expert_field', float('nan')):.2f}" if v.get("hands") else "-"
        evv = f"{v.get('expert_v', float('nan')):+.2f}" if v.get("hands") else "-"
        print(f"{d:{w}s} {split[:3]} {r.get('hands', 0):>3d}/{v.get('hands', 0):<3d} "
              f"{r.get('greedy_field', float('nan')):8.2f} {r.get('greedy_v', 0):+7.2f} " +
              " ".join(f"{b:>7s}" for b in bk) + f"  {ef:>8s} {evv:>7s} {gate:>4s} {s['stuck_rate']:.2f}")
    print(f"wrote {len(rows)} rows, {len(deck_sum)} decks -> {args.out} ({summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
