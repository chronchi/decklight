"""Stage M play-out anchor (retrain plan v3 §Stage M): keep a muse turn only if it is not worse than the policy's own.

For every turn record of scripts/plan_turns.py (`labels` = "prefix" up to the state, then muse's verified turn), two
arms are played out from the same state by the same policy (both seats):
- muse: replay the whole record (muse's turn), then the policy continues;
- policy: replay only the prefix, then the policy plays the turn itself and continues.
Each arm has `--samples` rollouts (sample 0 argmax, the rest at `--rollout-temp`), scored for the acting seat at the end
of turn `start + horizon - 1` (horizon 2 = through the opponent's next turn) by the learned value (or ±1 when the duel
ended). Keep when mean(muse) >= mean(policy) - `--z` * SE(diff) (plan: not worse by more than 1 SE).

Output: one JSON line per record in `--out` (line_id, bucket, deck, muse, policy, diff, se, keep) and a summary line
per bucket. Exit 2 on 0 records, or when every diff is exactly 0 (the arms did not differ: replay bug).

    uv run python scripts/turn_anchor.py --turns 'data/demos/turns/*/*.jsonl' --ckpt data/cache/base_r3/best.pt \
        --value data/cache/value_ppo3/best.pt --device cuda --out runs/x/anchor.jsonl --proc 0 --nproc 1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decklight.demos.dataset import record_config  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402
from decklight.search.window import WindowSearch  # noqa: E402

MUSE, POLICY = 0, 1


def turn_start(rec: dict) -> int:
    """Index of the first decision of muse's turn (first label that is not "prefix")."""
    return next((k for k, lab in enumerate(rec["labels"]) if lab != "prefix"), len(rec["labels"]))


def anchor_gen(ws: WindowSearch, rec: dict, samples: int, key: int):
    """Generator for drive_many: per-sample scores of both arms -> (muse samples, policy samples)."""
    k = turn_start(rec)
    ch = [int(c) for c in rec["choices"]]
    me, t0 = int(rec["actors"][k]), int(rec["turns"][k])
    branches = [(ch, MUSE, -1, s) for s in range(samples)] + [(ch[:k], POLICY, -2, s) for s in range(samples)]
    yield from ws.rollouts_gen(record_config(rec), bool(rec.get("auto_single", True)), branches, me,
                               game_seed=key, decision_k=k, turn_start=t0)  # fmt: skip
    # last_samples is set inside the generator's final step, so it belongs to this call (drive_many never interleaves
    # another generator between that step and this read)
    return [list(ws.last_samples[MUSE]), list(ws.last_samples[POLICY])]


def mean_var(x: list[float]) -> tuple[float, float]:
    m = sum(x) / len(x)
    return m, (sum((v - m) ** 2 for v in x) / (len(x) - 1) if len(x) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", nargs="+", required=True, help="globs of plan_turns records")
    ap.add_argument("--policy", default="bc", choices=["bc", "greedy"])
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--value", type=Path, default=None)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--rollout-temp", type=float, default=1.0)
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.samples < 2:
        ap.error("--samples must be >= 2 (the keep rule needs a variance)")

    recs = []
    for f in sorted({f for g in args.turns for f in glob.glob(g)}):
        for raw in Path(f).read_text().splitlines():
            if raw.strip():
                r = json.loads(raw)
                if zlib.crc32(r["line_id"].encode()) % args.nproc == args.proc:
                    recs.append(r)
    if not recs:
        print(f"ALERT turn_anchor: 0 turn records for proc {args.proc}/{args.nproc} in {args.turns}", flush=True)
        return 2

    if args.policy == "bc":
        from decklight.eval.bc_policy import BCPolicy

        logits = BCPolicy(args.ckpt, args.emb, threads=1, device=args.device, batch=args.batch).logits
    else:
        from gen_search import greedy_logits as logits
    value = None
    if args.value:
        from decklight.eval.value import ValueNet

        value = ValueNet(args.value, args.emb, device=args.device, batch=args.batch).values
    ws = WindowSearch(Engine(), logits, horizon=args.horizon, batch=args.batch, seed=args.seed, value=value,
                      samples=args.samples, rollout_temp=args.rollout_temp)  # fmt: skip

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = ((i, anchor_gen(ws, r, args.samples, zlib.crc32(r["line_id"].encode()))) for i, r in enumerate(recs))
    per_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    diffs = []
    with args.out.open("a") as fo:
        for n, (i, (sm, sp)) in enumerate(ws.run_many(jobs, args.width), 1):
            r = recs[i]
            if len(sm) != args.samples or len(sp) != args.samples:
                print(f"ALERT turn_anchor: {r['line_id']}: {len(sm)}/{len(sp)} samples, want {args.samples}")
                return 2
            (mm, vm), (mp, vp) = mean_var(sm), mean_var(sp)
            se = math.sqrt(vm / len(sm) + vp / len(sp))
            keep = mm >= mp - args.z * se
            diffs.append(mm - mp)
            b = r.get("bucket", "?")
            per_bucket[b][0] += 1
            per_bucket[b][1] += keep
            fo.write(json.dumps({"line_id": r["line_id"], "bucket": b, "deck": r.get("deck"), "muse": round(mm, 4),
                                 "policy": round(mp, 4), "diff": round(mm - mp, 4), "se": round(se, 4),
                                 "keep": keep, "samples": [sm, sp]}) + "\n")  # fmt: skip
            if n % 20 == 0:
                print(f"{n}/{len(recs)} anchored, {time.time() - t0:.0f} s", flush=True)
    for b, (n, k) in sorted(per_bucket.items()):
        print(f"bucket {b}: kept {k}/{n} ({k / n:.2f})", flush=True)
    print(f"done: {len(diffs)} records, mean diff {sum(diffs) / len(diffs):+.4f}, capped {ws.capped}, "
          f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
    if all(d == 0 for d in diffs):
        print("ALERT turn_anchor: every muse - policy diff is exactly 0 (the arms did not differ)", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
