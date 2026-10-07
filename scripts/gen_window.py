"""Mid-game search teacher (ADR 0006 M3): searched targets for response windows and own-turn plays on recorded games.

    uv run python scripts/gen_window.py --ckpt <best.pt> --device cuda --records 'data/demos/teacher/*.jsonl' 'data/demos/gold/*.jsonl' \
        --games 500 --out runs/<name>/window/p0 [--k 4] [--horizon 1] [--own-frac 0.1] [--proc 0 --nproc 8]
    uv run python scripts/gen_window.py --policy greedy ...   # laptop / tests: no model

Each source game is replayed; selected decisions (src/ygo_zero/search/window.py: chain windows on the opponent's
turn where the actor can activate, plus a sample of own-turn decisions) get a searched target. Output: one record per
game in `<out>/<source file name>`: the source's config/choices/actors/turns, `targets` (searched option or null,
aligned with choices), `labels` ("window" / "own" at searched decisions, "skip" elsewhere), the search log, and the
source game id. Kinds once under data/demos/window/: `window`, `window_own` (dataset.py); the replay follows the
recorded choices, the target is the searched option.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_search import greedy_logits

from ygo_zero.demos.dataset import record_config
from ygo_zero.engine.duel import Engine
from ygo_zero.search.window import WindowSearch

KEEP = ("decks", "config", "auto_single", "choices", "actors", "turns", "winner", "end_turn", "level")


def load_games(patterns: list[str], games: int, seed: int, proc: int, nproc: int) -> list[tuple[str, dict]]:
    """(source file name, record) for this process's share of a seeded random sample of games with actors."""
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    recs = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "actors" in r and r.get("choices"):
                    recs.append((Path(f).name, r))
    random.Random(seed).shuffle(recs)
    recs = recs[:games] if games else recs
    return recs[proc::nproc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="bc", choices=["bc", "greedy"])
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--records", nargs="+", required=True, help="globs of source game records")
    ap.add_argument("--games", type=int, default=0, help="games sampled over all processes (0 = all)")
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--value", type=Path, default=None, help="learned value checkpoint (ADR 0007 v2) to score branch ends")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=1, help="turns played out (1 = to the end of the current turn)")
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="score horizons (default: --horizon alone)")
    ap.add_argument("--samples", type=int, default=1, help="rollouts per candidate (sample 0 is argmax)")
    ap.add_argument("--rollout-temp", type=float, default=1.0)
    ap.add_argument("--select", type=str, default=None, help="score key selecting the target (default: board/value@h<H0>)")
    ap.add_argument("--tie-eps", type=float, default=0.0, help="candidates within eps of best form the target set")
    ap.add_argument("--conf-skip", type=float, default=1.01,
                    help="skip decisions whose policy top probability is >= this (S0 used 0.9; 1.01 = search all)")
    ap.add_argument("--own-frac", type=float, default=0.1, help="share of own-turn decisions (turn >= 2) searched")
    ap.add_argument("--honest", type=int, default=0, choices=(0, 1), help="search determinized worlds (honest search)")
    ap.add_argument("--det-tries", type=int, default=32, help="rejection-sampling proposals per honest world")
    ap.add_argument("--det-fallback", default="skip", choices=("true", "skip"),
                    help="failed world: fall back to the true config, or skip the decision")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.policy == "bc":
        from ygo_zero.eval.bc_policy import BCPolicy

        pol = BCPolicy(args.ckpt, args.emb, threads=1, device=args.device, batch=args.batch)
        logits = pol.logits
        if args.device.startswith("cuda") and args.bf16:
            import torch

            def logits(buf, _f=pol.logits):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return _f(buf)
    else:
        logits = greedy_logits
    value = None
    if args.value:
        from ygo_zero.eval.value import ValueNet

        value = ValueNet(args.value, args.emb, device=args.device, batch=args.batch).values
    ws = WindowSearch(Engine(), logits, k=args.k, horizon=args.horizon, own_frac=args.own_frac, batch=args.batch,
                      seed=args.seed, value=value,
                      horizons=tuple(args.horizons) if args.horizons else None,
                      samples=args.samples, rollout_temp=args.rollout_temp,
                      select=args.select, tie_eps=args.tie_eps, conf_skip=args.conf_skip)  # fmt: skip
    games = load_games(args.records, args.games, args.seed, args.proc, args.nproc)
    args.out.mkdir(parents=True, exist_ok=True)
    eff_hs = tuple(args.horizons) if args.horizons else (args.horizon,)
    jobs = ((i, ws.game_gen(rec, record_config(rec), i, honest=bool(args.honest),
                            det_tries=args.det_tries, det_fallback=args.det_fallback))
            for i, (_, rec) in enumerate(games))
    t0 = time.time()
    tot = {"games": 0, "window": 0, "own": 0, "fired": 0, "agree_window": 0, "agree_own": 0, "bad_replay": 0,
           "det_proposals": 0, "det_accepted": 0, "det_fallbacks": 0, "det_skipped": 0, "det_s": 0.0}
    agree: dict[str, int] = {}
    agree_n = 0
    conf_sum, set_sum, conf_n = 0.0, 0, 0
    for i, res in ws.run_many(jobs, args.width):
        src, rec = games[i]
        tot["games"] += 1
        tot["det_proposals"] += res.det_proposals
        tot["det_accepted"] += res.det_accepted
        tot["det_fallbacks"] += res.det_fallbacks
        tot["det_skipped"] += res.det_skipped
        tot["det_s"] += res.det_s
        if not res.replay_ok:
            tot["bad_replay"] += 1
            continue
        n = len(rec["choices"])
        targets: list[int | None] = [None] * n
        labels = ["skip"] * n
        target_sets: list[list[int] | None] = [None] * n
        target_conf: list[float | None] = [None] * n
        for k, t in res.targets.items():
            targets[k], labels[k] = t, res.kinds[k]
            target_sets[k] = res.sets.get(k)
            target_conf[k] = res.conf.get(k)
        for e in res.log:
            tot[e["kind"]] += 1
            tot["fired"] += e["kind"] == "window" and e["fired"]
            tot[f"agree_{e['kind']}"] += e["chosen"] == e["played"]
            agree_n += 1
            conf_sum += float(e.get("conf", 1.0))
            set_sum += len(e.get("set", [e["chosen"]]))
            conf_n += 1
            cands, chosen = e["cands"], e["chosen"]
            for kk, col in (e.get("scores_all") or {}).items():
                amax = max(range(len(cands)), key=lambda j: (col[j], -j))
                agree[kk] = agree.get(kk, 0) + (cands[amax] == chosen)
        gid = rec.get("game_id") or rec.get("line_id", f"g{i}")
        hs_str = ",".join(str(h) for h in eff_hs)
        out = {k: rec[k] for k in KEEP if k in rec} | {
            "game_id": f"w-{gid}", "line_id": f"{rec.get('line_id', gid)}/w", "source_game": gid,
            "targets": targets, "labels": labels, "notes": [], "search": res.log,
            "target_sets": target_sets, "target_conf": target_conf,
            "source": f"window search k={args.k} h={args.horizon} hs=[{hs_str}] samples={args.samples} "
                      f"temp={args.rollout_temp} select={args.select} tie_eps={args.tie_eps} over {args.policy}"
                      + (f", value {args.value}" if args.value else "")
                      + f" honest={int(args.honest)} det_fallback={args.det_fallback}",
        }  # fmt: skip
        with (args.out / src).open("a") as f:
            f.write(json.dumps(out) + "\n")
        if tot["games"] % 20 == 0:
            print(f"{tot} capped rollouts {ws.capped} {time.time() - t0:.0f} s", flush=True)
    w = max(tot["window"], 1)
    per_key = {kk: f"{v / max(agree_n, 1):.3f}" for kk, v in sorted(agree.items())}
    mean_conf = conf_sum / max(conf_n, 1)
    mean_set = set_sum / max(conf_n, 1)
    det_extra = ""
    skip_rate = 0.0
    if args.honest:
        props, acc, fb, sk = tot["det_proposals"], tot["det_accepted"], tot["det_fallbacks"], tot["det_skipped"]
        worlds = acc + fb
        accept_rate = acc / props if props else 0.0
        fb_rate = fb / worlds if worlds else 0.0
        eligible = conf_n + sk
        skip_rate = sk / eligible if eligible else 0.0
        det_extra = (f"honest worlds: accept {accept_rate:.3f} ({acc}/{props}), "
                     f"fallback {fb_rate:.3f} ({fb}/{worlds}), "
                     f"skipped {skip_rate:.3f} ({sk}/{eligible}); ")
    print(f"done: {tot}; fire rate {tot['fired'] / w:.3f}, agreement with the played move at windows "
          f"{tot['agree_window'] / w:.3f}, own {tot['agree_own'] / max(tot['own'], 1):.3f}; "
          f"per-key agreement with target {per_key}, mean conf {mean_conf:.3f}, mean set {mean_set:.2f}; "
          f"{det_extra}"
          f"capped rollouts {ws.capped} (scored as losses); "
          f"wall {time.time() - t0:.0f} s",
          flush=True)  # fmt: skip
    if args.honest and skip_rate > 0.5:
        print(f"ALERT gen_window: honest skip rate {skip_rate:.3f} > 50% "
              f"({tot['det_skipped']}/{conf_n + tot['det_skipped']})", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
