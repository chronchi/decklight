"""Turn-1 search teacher (ADR 0006 M2): searched turn-1 records for distillation, and the combo-test check.

    uv run python scripts/gen_search.py --ckpt <best.pt> --device cpu --hands 200 --out runs/<name>/search/p0
    uv run python scripts/gen_search.py --ckpt <best.pt> --combo 8 --hands 3 --out runs/<name>/combo-p0.json
    uv run python scripts/gen_search.py --policy greedy ...   # laptop / tests: no model

gen (default): `--hands` going-first openings, each on a random pool TRAIN deck (holdout decks never) vs a random pool
deck, seed `--seed + i`; player 0's turn 1 is searched (src/ygo_zero/search/turn1.py, top-`--k` rollouts) and saved as
a game record in `<out>/<deck>.jsonl` (labels "search" / "opp"; kind `search` once under data/demos/search/), with
the policy's own greedy score, the searched score and the per-decision search log.
--combo D: the combo test's hands instead (D decks with verified lines, half train / half holdout, `--hands` lines
each, same sample as playtest.py for the same --seed): greedy vs searched vs expert end field, JSON to --out.
Exit criterion M2: searched > greedy on the combo test.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from ygo_zero.decks import load_ydk
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.eval.policies import _GREEDY
from ygo_zero.search.turn1 import Turn1Search

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def greedy_logits(buf: dict) -> np.ndarray:
    """Deterministic stand-in policy (tests, laptop): GreedyPolicy's kind preference, ties by option order."""
    kinds = buf["actions"][:, :, 0]
    return _GREEDY[kinds].astype(np.float64) - 1e-3 * np.arange(kinds.shape[1])


def config_dict(cfg: DuelConfig) -> dict:
    return {"deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
            "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
            "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw, "lp": cfg.lp, "setup": [], "shuffle": cfg.shuffle}  # fmt: skip


def pool_decks() -> tuple[list[str], set[str]]:
    pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    return pool, hold


def gen(search: Turn1Search, args) -> None:
    pool, hold = pool_decks()
    train = [d for d in pool if d not in hold]
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    hands = []
    for i in range(args.hands):
        d0, d1 = rng.choice(train), rng.choice(pool)
        hands.append((d0, d1, args.seed + i))
    jobs = ((h, search.both_gen(DuelConfig(load_ydk(ROOT / "decks/pool" / f"{h[0]}.ydk"),
                                           load_ydk(ROOT / "decks/pool" / f"{h[1]}.ydk"), seed=h[2])))
            for h in hands)  # fmt: skip
    t0, gains, ov, srch = time.time(), [], 0, 0
    for (d0, d1, seed), (g, r) in search.run_many(jobs, args.width):
        cfg = DuelConfig(load_ydk(ROOT / "decks/pool" / f"{d0}.ydk"), load_ydk(ROOT / "decks/pool" / f"{d1}.ydk"),
                         seed=seed)  # fmt: skip
        gains.append(r.end_score - g)
        ov, srch = ov + r.overrides, srch + r.searched
        rec = {"game_id": f"s1-{seed}", "line_id": f"{d0}/s1-{seed}", "level": "L4", "decks": [d0, d1],
               "config": config_dict(cfg), "auto_single": True, "choices": r.choices, "labels": r.labels,
               "actors": r.actors, "turns": [1] * len(r.choices), "notes": [], "greedy_score": g,
               "end_score": r.end_score, "search": r.log, "k": args.k,
               "source": f"turn-1 search k={args.k} over {args.policy}"}  # fmt: skip
        with (args.out / f"{d0}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"hand {len(gains)} {d0}: greedy {g:.2f} searched {r.end_score:.2f} ({r.decisions} decisions, "
              f"{r.searched} searched, {r.overrides} overrides) {time.time() - t0:.0f} s", flush=True)  # fmt: skip
    if gains:
        print(f"hands {len(gains)}: mean gain {statistics.mean(gains):.2f}, improved {sum(x > 0 for x in gains)}, "
              f"worse {sum(x < 0 for x in gains)}, overrides {ov}/{srch}, wall {time.time() - t0:.0f} s", flush=True)  # fmt: skip


def combo(search: Turn1Search, engine, args) -> None:
    from playtest import _replay_field, verified_end_boards, verified_records

    from ygo_zero.demos.dataset import record_config

    pool, hold = pool_decks()
    rng = random.Random(args.seed)  # same deck sample as playtest.py --seed
    train_d = [d for d in pool if d not in hold and verified_end_boards(d)]
    hold_d = [d for d in pool if d in hold and verified_end_boards(d)]
    sample = rng.sample(train_d, min(args.combo // 2, len(train_d))) + rng.sample(hold_d, min(args.combo // 2, len(hold_d)))
    hands = []
    for d in sample:
        for r in verified_records(d, limit=args.hands):
            try:
                hands.append((d, r, record_config(r), _replay_field(engine, r)))
            except Exception as ex:  # noqa: BLE001 - a record that no longer replays is skipped
                print(f"skip {r.get('line_id')}: {ex!r}"[:200], flush=True)
    rows, t0 = [], time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = ((i, search.both_gen(h[2])) for i, h in enumerate(hands))
    for i, (g, s) in search.run_many(jobs, args.width):
        d, r, _, e = hands[i]
        rows.append({"deck": d, "holdout": d in hold, "line_id": r["line_id"], "greedy": int(g), "search": int(s.end_score),
                     "expert": e, "overrides": s.overrides, "searched": s.searched})  # fmt: skip
        print(f"combo {d[:30]:30s} {'H' if d in hold else ' '} greedy {int(g)} search {int(s.end_score)} expert {e} "
              f"({s.overrides}/{s.searched} overrides) {time.time() - t0:.0f} s", flush=True)  # fmt: skip
        args.out.write_text(json.dumps({"summary": None, "rows": rows, "k": args.k}, indent=1))  # a timeout keeps rows
    summ = {}
    if rows:
        summ = {k: round(statistics.mean(r[k] for r in rows), 2) for k in ("greedy", "search", "expert")}
        summ |= {"hands": len(rows), "search_gt_greedy": sum(r["search"] > r["greedy"] for r in rows),
                 "search_lt_greedy": sum(r["search"] < r["greedy"] for r in rows),
                 "search_ge_expert": sum(r["search"] >= r["expert"] for r in rows),
                 "greedy_ge_expert": sum(r["greedy"] >= r["expert"] for r in rows)}  # fmt: skip
    print(f"combo summary: {summ}", flush=True)
    args.out.write_text(json.dumps({"summary": summ, "rows": rows, "k": args.k}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="bc", choices=["bc", "greedy"])
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="torch threads")
    ap.add_argument("--k", type=int, default=4, help="candidates (policy top-k) per decision")
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--width", type=int, default=64, help="hands searched concurrently (one batched forward per step)")
    ap.add_argument("--batch", type=int, default=512, help="max obs per forward")
    ap.add_argument("--combo", type=int, default=0, help="combo-test mode on this many decks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    engine = Engine()
    if args.policy == "bc":
        from ygo_zero.eval.bc_policy import BCPolicy

        logits = BCPolicy(args.ckpt, args.emb, threads=args.threads, device=args.device, batch=args.batch).logits
    else:
        logits = greedy_logits
    search = Turn1Search(engine, logits, k=args.k, batch=args.batch)
    if args.combo:
        combo(search, engine, args)
    else:
        gen(search, args)


if __name__ == "__main__":
    main()
