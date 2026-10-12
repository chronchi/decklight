"""Override replication audit (retrain plan v2 stage S; critique-v1 finding 3).

Does a search override hold up when the same decision is searched again with fresh honest worlds and many samples?
For a random sample of overridden decisions in search_match games (`chosen != played`), replay the game to the
decision, draw `--samples` new determinized worlds (seen cards of the searching seat only), roll out the policy's pick
and the search's choice on each (common worlds, so the difference is paired) and score with the value at h1, exactly as
the search did. Reports the share whose sign holds (choice > pick) and the share significant at 2 SE.

The precision gate (plan v2): >= 0.85 of stage-L labels / ho overrides must keep their sign before any search label
enters training. Run it on hppo games as the baseline the honest rule must beat.

Usage (vast GPU box; not on the laptop):
    python scripts/audit/replicate_overrides.py --games runs/<name>/ho --n 300 --samples 32 \
        --ckpt data/cache/base_r3/best.pt --value data/cache/value_ppo3/best.pt --out runs/<name>/repl-ho.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.search.determinize import Determinizer, SeenTracker, key_rank_of_choice
from ygo_zero.search.honest import paired_gap
from ygo_zero.search.turn1 import _batch, drive_many
from ygo_zero.search.window import WindowSearch


def fail(msg: str) -> None:
    print(f"ALERT replicate_overrides: {msg}", flush=True)
    raise SystemExit(2)


def overrides(games_dir: Path) -> list[tuple[dict, dict]]:
    out = []
    for f in sorted(games_dir.glob("games-*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            for e in r.get("search", []):
                if int(e["chosen"]) != int(e["played"]):
                    out.append((r, e))
    return out


def hw_labels(hw_dir: Path) -> list[tuple[dict, dict]]:
    """Stage L decisive labels (gen_window --ho 1 records): (record, entry) with pick = the runner-up."""
    out = []
    for f in sorted(hw_dir.glob("p*/*.jsonl")) + sorted(hw_dir.glob("*.jsonl")):
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            r.setdefault("game_idx", f"{f.parent.name}/{f.name}:{i}")
            for e in r.get("search", []):
                h = e.get("ho") or {}
                if h.get("status") == "label" and h.get("runner_up") is not None:
                    out.append((r, {**e, "played": int(h["runner_up"]), "chosen": int(e["chosen"])}))
    return out


def item_gen(engine, ws: WindowSearch, rec: dict, e: dict, deck_dir: Path, samples: int, tries: int, seed: int):
    k = int(e["k"])
    choices = [int(c) for c in rec["choices"]]
    if "s_seat" in rec:  # search_match game: decks by name, the searching seat
        names = rec["decks"]
        cfg = DuelConfig(load_ydk(deck_dir / f"{names[0]}.ydk"), load_ydk(deck_dir / f"{names[1]}.ydk"),
                         seed=int(rec["seed"]))
        me = int(rec["s_seat"])
    else:  # gen_window record (stage L): stored config, the acting player at k
        from ygo_zero.demos.dataset import record_config

        cfg, me = record_config(rec), int(rec["actors"][k])
    env = CppEnv(engine, cfg, rec.get("auto_single", True))
    obs = env.reset()
    tr = SeenTracker(me)
    tr.update(env)
    keys = []
    for c in choices[:k]:
        if obs is None:
            return {"error": "game ended before k"}
        keys.append(key_rank_of_choice(obs, c))
        obs = env.step(c)
        tr.update(env)
    if obs is None or int(env.player) != me:
        return {"error": "replay mismatch at k"}
    pick, ch = int(e["played"]), int(e["chosen"])
    if "s_seat" in rec and choices[k] != ch:
        return {"error": f"recorded choice {choices[k]} != chosen {ch}"}
    det = Determinizer(engine, cfg, rec.get("auto_single", True), me, choices[:k], tr.seen, obs,
                       random.Random(f"{seed}:{rec['game_idx']}:{k}"),
                       prefix_keys=keys, tries=tries)
    worlds = [w for w in (det.sample() for _ in range(samples)) if w is not None]
    if len(worlds) < samples // 2:
        return {"error": f"only {len(worlds)} worlds"}
    branches = [(choices[:k] + [x], j, x, s) for j, x in enumerate((pick, ch)) for s in range(len(worlds))]
    yield from ws.rollouts_gen(cfg, rec.get("auto_single", True), branches, me, turn_start=int(e["turn"]),
                               game_seed=10_000 + (int(rec["game_idx"]) if str(rec["game_idx"]).isdigit()
                                                   else zlib.crc32(str(rec["game_idx"]).encode())),
                               decision_k=k, worlds=worlds)
    sp, sc = ws.last_samples[0], ws.last_samples[1]
    m, se, n = paired_gap(sc, sp)
    orig = e.get("ho") or {}
    return {"game_idx": rec["game_idx"], "k": k, "turn": e["turn"], "kind": e.get("kind"), "pick": pick, "chosen": ch,
            "gap": m, "se": se if math.isfinite(se) else None, "n": n, "holds": m > 0,
            "sig": bool(math.isfinite(se) and m >= 2 * se), "orig_gap": orig.get("gap"), "orig_se": orig.get("se")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, required=True, help="search_match games dir, or a stage L hw dir with --hw")
    ap.add_argument("--hw", action="store_true", help="stage L decisive labels: best vs runner-up (gen_window --ho 1)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--det-tries", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--value", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--deck-dir", type=Path, default=ROOT / "decks/gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summarize", action="store_true", help="merge <out stem>-*.jsonl into a summary and exit")
    args = ap.parse_args()

    if args.summarize:
        rows = [json.loads(x) for f in sorted(args.out.parent.glob(f"{args.out.stem}-*.jsonl"))
                for x in f.read_text().splitlines()]
        ok = [r for r in rows if "error" not in r]
        if not ok:
            fail(f"no replicated decisions in {args.out.parent}/{args.out.stem}-*.jsonl")
        summ = {"n": len(ok), "errors": len(rows) - len(ok),
                "sign_holds": round(sum(r["holds"] for r in ok) / len(ok), 3),
                "sig_2se": round(sum(r["sig"] for r in ok) / len(ok), 3),
                "mean_gap": round(float(np.mean([r["gap"] for r in ok])), 4)}
        for kind in ("window", "own"):
            sub = [r for r in ok if r.get("kind") == kind]
            if sub:
                summ[kind] = {"n": len(sub), "sign_holds": round(sum(r["holds"] for r in sub) / len(sub), 3)}
        args.out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps(summ))
        return

    items = hw_labels(args.games) if args.hw else overrides(args.games)
    if not items:
        fail(f"no overrides in {args.games}")
    random.Random(args.seed).shuffle(items)
    items = items[: args.n][args.proc :: args.nproc]

    from ygo_zero.eval.bc_policy import BCPolicy
    from ygo_zero.eval.value import ValueNet

    engine = Engine()
    pol = BCPolicy(args.ckpt, args.emb, threads=1, device=args.device, batch=args.batch)
    value = ValueNet(args.value, args.emb, device=args.device, batch=args.batch).values

    def forward(obs_list):
        return np.concatenate([pol.logits(_batch(obs_list[i : i + args.batch]))
                               for i in range(0, len(obs_list), args.batch)])

    def never(buf):
        raise AssertionError("unused")

    jobs = []
    for i, (rec, e) in enumerate(items):
        ws = WindowSearch(engine, never, k=2, horizon=1, samples=args.samples, seed=args.seed + 7, value=value)
        jobs.append((i, item_gen(engine, ws, rec, e, args.deck_dir, args.samples, args.det_tries, args.seed)))
    path = args.out.parent / f"{args.out.stem}-{args.proc}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with path.open("w") as f:
        for _, res in drive_many(jobs, forward, args.width):
            f.write(json.dumps(res) + "\n")
            n_ok += "error" not in res
    print(f"proc {args.proc}: {len(items)} items, {n_ok} replicated", flush=True)
    if items and n_ok == 0:
        fail("0 decisions replicated")


if __name__ == "__main__":
    main()
