"""Score turn-1 lines by playing them out (self-imitation round 2 line score).

    python scripts/score_lines.py --ckpt <policy> --lines <lines.jsonl> --out <scored.jsonl> [--cands 4] [--samples 8]
        [--horizon 2] [--temp 1.0] [--max-rollout 600] [--procs P] [--device cuda] [--batch 512] [--width W]
        [--value <value ckpt>] [--only-improvable] [--task-timeout 300]

Turn-1 field count rewards spending everything on turn 1 (round 1: the fine-tune discarded a hand
trap and fired Mirrorjade on its own turn). This scores each candidate line by replaying it, then
playing on through the end of the opponent's turn 2 with the policy itself (both players, sampled),
so held hand traps and quick effects count when they get used.

Method: per (deck, source, hand) group the greedy line plus up to --cands-1 distinct sampled lines
are each rolled out S times (sample 0 argmax, the rest at --temp) with WindowSearch.rollouts_gen
(turn_start=1, horizons=(--horizon,): horizon 2 snapshots at turn > 2, i.e. the end of turn 2).
Score = mean board@hH from player 0's view (window.py position_score; finished duels score +-50).
Common random numbers: rollout sample s of every candidate of a hand shares one seed
(game_seed from the hand, decision_k=0, cand_action=0 constant), so line differences are not
sampling noise. Each candidate runs as one rollouts_gen call with one branch per sample
(candidate index = sample index), which yields per-sample scores; the call's own WindowSearch
counts its capped branches, giving the per-line rollout_capped without touching window.py.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.demos.dataset import record_config  # noqa: E402
from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import Engine  # noqa: E402
from ygo_zero.eval.bc_policy import BCPolicy  # noqa: E402
from ygo_zero.eval.turn1 import stable_seed  # noqa: E402
from ygo_zero.search.turn1 import drive_many  # noqa: E402
from ygo_zero.search.window import WindowSearch  # noqa: E402

_W: dict = {}


def _init(cfg: dict) -> None:
    """Per-worker state (spawn): its own engine + policy (+ value net when asked)."""
    import torch as _torch

    _torch.set_num_threads(cfg["threads"])
    engine = Engine()
    pol = BCPolicy(cfg["ckpt"], cfg["emb"], temperature=0.0,
                   threads=cfg["threads"], batch=cfg["batch"], device=cfg["device"])
    value = None
    if cfg.get("value"):
        from ygo_zero.eval.value import ValueNet

        value = ValueNet(cfg["value"], cfg["emb"], device=cfg["device"],
                         batch=cfg["batch"], threads=cfg["threads"]).values
    _W.update(engine=engine, policy=pol, value=value, cfg=cfg)


def group_key(r: dict) -> tuple:
    """One scored group: (deck, source, start) + actor/turn_start (old rows lack actor/turn_start,
    so they default to actor 0, turn_start 1 and group exactly as before)."""
    return (r["deck"], r["source"], r.get("start", r["hand"]),
            int(r.get("actor", 0)), int(r.get("turn_start", 1)))


def _fv(r: dict) -> tuple:
    return (r.get("field", 0), r.get("v_p0", r.get("v_actor", 0.0)), -r.get("sample", 0))


def pick_candidates(rows: list[dict], cands: int) -> list[dict]:
    """Greedy + up to cands-1 distinct (by choices) non-stuck samples: best by field, best by
    v_p0, then the most frequent other lines (deterministic order). Expert rows are skipped."""
    greedies = sorted((r for r in rows if r.get("kind") == "greedy"), key=lambda r: r.get("sample", -1))
    if not greedies:
        return []
    g = greedies[0]
    out = [g]
    if cands <= 1:
        return out
    seen = {tuple(g["choices"])}
    pool: list[dict] = []
    for r in rows:
        if r.get("kind") != "sample" or r.get("stuck"):
            continue
        k = tuple(r["choices"])
        if k in seen:
            continue
        seen.add(k)
        pool.append(r)
    if not pool:
        return out
    freq = Counter(tuple(r["choices"]) for r in rows
                   if r.get("kind") == "sample" and not r.get("stuck"))
    best_f = max(pool, key=_fv)
    out.append(best_f)
    rest = [r for r in pool if tuple(r["choices"]) != tuple(best_f["choices"])]
    if rest and len(out) < cands:
        best_v = max(rest, key=lambda r: (r.get("v_p0", r.get("v_actor", 0.0)), r.get("field", 0),
                                          -r.get("sample", 0)))
        out.append(best_v)
        rest = [r for r in rest if tuple(r["choices"]) != tuple(best_v["choices"])]
    if rest and len(out) < cands:
        rest.sort(key=lambda r: (-freq[tuple(r["choices"])], r["sample"]))
        out.extend(rest[: cands - len(out)])
    return out


def branch_turn(engine: Engine, row: dict) -> int | None:
    """Turn number at the end of a line's replay (None when the replayed duel ended).

    Turn-1 lines run through the start of turn 2, so this is 2 for a live line; whole-turn
    lines run through the start of turn_start + 1.
    """
    env = CppEnv(engine, record_config({"config": row["config"]}), row.get("auto_single", True))
    obs = env.reset()
    for c in [int(c) for c in row["choices"]]:
        if obs is None:
            return None
        obs = env.step(c)
    return None if obs is None else int(obs["global"][2])


def _score_candidates(cand_rows: list[dict], deck: str, source: str, hand: int | str) -> list[dict]:
    """Roll out each candidate line S times; returns the rows with rollout fields added.

    The score is from the actor's view (me=actor) with turn_start=actor's turn; horizon H=2
    means through the end of the opponent's next turn. Same CRN seeding per start across
    candidates (old rows: stable_seed(deck, source, hand), unchanged).
    """
    cfg, engine = _W["cfg"], _W["engine"]
    S, H = cfg["samples"], cfg["horizon"]
    me = int(cand_rows[0].get("actor", 0))
    t0 = int(cand_rows[0].get("turn_start", 1))
    skey = cand_rows[0].get("start", hand)
    gs = stable_seed(deck, source, skey) % (2**31 - 1)
    bkey = f"board@h{H}"
    dkey = f"done@h{H}"
    vkey = f"value@h{H}"
    jobs = []
    wss = []
    for ci, row in enumerate(cand_rows):
        dc = record_config({"config": row["config"]})
        auto = bool(row.get("auto_single", True))
        prefix = [int(c) for c in row["choices"]]
        branches = [(prefix, s, 0, s) for s in range(S)]  # candidate index = sample: per-sample scores
        ws = WindowSearch(engine, _W["policy"].logits, k=4, horizon=H, max_rollout=cfg["max_rollout"],
                          batch=cfg["batch"], seed=cfg["seed"], value=_W["value"],
                          horizons=(H,), samples=S, rollout_temp=cfg["temp"])  # fmt: skip
        wss.append(ws)
        # cand_action=0 for every branch + decision_k=0 + hand game_seed: sample s shares one
        # seed across candidates (common random numbers); s=0 is argmax, the rest sample.
        jobs.append((ci, ws.rollouts_gen(dc, auto, branches, me,
                                         game_seed=gs, decision_k=0, turn_start=t0)))
    res = dict(drive_many(jobs, wss[0]._forward, cfg["width"]))
    scored = []
    for ci, row in enumerate(cand_rows):
        out = res[ci]
        samples = [float(x) for x in out[bkey]]
        done = [float(x) for x in out[dkey]]
        r = dict(row)
        r["rollout"] = float(sum(samples) / len(samples))
        r["rollout_samples"] = samples
        r["rollout_done"] = float(sum(done) / len(done))
        if _W["value"] is not None:
            r["rollout_value"] = float(sum(float(x) for x in out[vkey]) / len(out[vkey]))
        r["rollout_capped"] = int(wss[ci].capped)
        scored.append(r)
    return scored


def _group_task(task: dict) -> dict:
    key, rows = tuple(task["key"]), task["rows"]
    deck, source, hand = key[0], key[1], key[2]
    cands = pick_candidates(rows, task["cands"])
    if not cands:
        return {"key": list(key), "rows": []}
    return {"key": list(key), "rows": _score_candidates(cands, deck, source, hand)}


def _alert(out: Path, problems: list[str] | str) -> None:
    if isinstance(problems, str):
        problems = [problems]
    ap = Path(str(out) + ".ALERT")
    ap.parent.mkdir(parents=True, exist_ok=True)
    with open(ap, "a") as f:
        for p in problems:
            line = f"ALERT score_lines: {p}"
            print(line, file=sys.stderr, flush=True)
            f.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="policy checkpoint (its metrics.jsonl gives the arch)")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--lines", type=Path, required=True, help="turn1_lines lines.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="scored lines output (jsonl)")
    ap.add_argument("--cands", type=int, default=4, help="candidates scored per hand (greedy + samples)")
    ap.add_argument("--samples", type=int, default=8, help="rollouts per candidate (sample 0 is argmax)")
    ap.add_argument("--horizon", type=int, default=2, help="score horizon (2 = through the end of turn 2)")
    ap.add_argument("--temp", type=float, default=1.0, help="rollout sampling temperature (samples 1..S-1)")
    ap.add_argument("--max-rollout", type=int, default=600)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--width", type=int, default=32, help="concurrent rollout branches per group")
    ap.add_argument("--value", type=Path, default=None, help="learned value checkpoint to score branch ends")
    ap.add_argument("--only-improvable", action="store_true",
                    help="skip groups where no sample reaches greedy's field")
    ap.add_argument("--seed", type=int, default=0, help="base rollout seed (common random numbers per hand)")
    ap.add_argument("--task-timeout", type=int, default=300, help="s without any finished group before giving up")
    args = ap.parse_args(argv)
    if args.cands < 1 or args.samples < 1 or args.horizon < 1:
        print("score_lines: --cands, --samples and --horizon must be >= 1", file=sys.stderr)
        return 2
    t0 = time.time()
    rows = [json.loads(line) for line in args.lines.read_text().splitlines() if line.strip()]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("kind") == "expert":  # human/LLM lines are not scored
            continue
        groups.setdefault(group_key(r), []).append(r)
    if args.only_improvable:
        keep = {}
        for k, g in groups.items():
            gr = [r for r in g if r.get("kind") == "greedy"]
            if not gr:
                continue
            if any(s.get("kind") == "sample" and not s.get("stuck") and s["field"] >= gr[0]["field"]
                   for s in g):
                keep[k] = g
        groups = keep
    tasks = [{"key": list(k), "rows": g, "cands": args.cands} for k, g in sorted(groups.items())]
    if not tasks:
        _alert(args.out, "0 groups to score (no greedy+sample groups in the input?)")
        return 3
    wcfg = {"ckpt": str(args.ckpt), "emb": str(args.emb), "seed": args.seed, "samples": args.samples,
            "horizon": args.horizon, "temp": args.temp, "max_rollout": args.max_rollout,
            "threads": args.threads, "device": args.device, "batch": args.batch, "width": args.width,
            "value": str(args.value) if args.value else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored: list[dict] = []
    done = 0

    def take(d: dict) -> None:
        nonlocal done
        done += 1
        if d["rows"]:
            scored.extend(d["rows"])
            with open(args.out, "a") as f:
                for r in d["rows"]:
                    f.write(json.dumps(r) + "\n")
        key = d["key"]
        n_c = len(d["rows"])
        print(f"PROGRESS group {done}/{len(tasks)}: {key[0]} {key[1]} h{key[2]} "
              f"{n_c} candidates, {len(scored)} rows, {time.time() - t0:.0f} s total", flush=True)

    if args.procs <= 1:
        _init(wcfg)
        for t in tasks:
            take(_group_task(t))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.procs, len(tasks)), initializer=_init, initargs=(wcfg,)) as pool:
            it = pool.imap_unordered(_group_task, tasks, chunksize=1)
            while done < len(tasks):
                try:  # watchdog: a rollout hung in the engine must not stall the job silently
                    d = it.next(timeout=args.task_timeout)
                except StopIteration:
                    break
                except mp.TimeoutError:
                    msg = (f"no group finished in {args.task_timeout} s: {len(tasks) - done} group(s) hung; "
                           f"keeping the {done} finished")
                    print(f"ALERT {msg}", file=sys.stderr, flush=True)
                    _alert(args.out, msg)
                    pool.terminate()
                    break
                take(d)
    problems: list[str] = []
    if not scored:
        problems.append("0 groups scored")
    else:
        vals = [r["rollout"] for r in scored]
        if any(isinstance(v, float) and np.isnan(v) for v in vals):
            problems.append("NaN rollout scores")
        elif len(set(vals)) == 1:
            problems.append(f"all {len(vals)} rollout scores identical ({vals[0]}: rollouts dead?)")
        tot_s = sum(len(r["rollout_samples"]) for r in scored)
        tot_c = sum(r["rollout_capped"] for r in scored)
        if tot_s and tot_c / tot_s > 0.10:
            problems.append(f"{tot_c}/{tot_s} rollout samples capped (>10% hit max_rollout)")
    if problems:
        _alert(args.out, problems)
        return 3
    # Summary over scored groups: greedy vs best-by-field vs best-by-rollout.
    by_group: dict[tuple, list[dict]] = {}
    for r in scored:
        by_group.setdefault(group_key(r), []).append(r)
    g_roll, f_roll, r_roll, agree = [], [], [], 0
    tot_s = sum(len(r["rollout_samples"]) for r in scored)
    tot_c = sum(r["rollout_capped"] for r in scored)
    for g in by_group.values():
        gr = [r for r in g if r.get("kind") == "greedy"]
        sm = [r for r in g if r.get("kind") == "sample"]
        if not gr:
            continue
        g_roll.append(gr[0]["rollout"])
        if not sm:
            continue
        bf = max(sm, key=lambda r: (r.get("field", 0), r.get("v_p0", r.get("v_actor", 0.0))))
        br = max(sm, key=lambda r: (r["rollout"], r.get("field", 0)))
        f_roll.append(bf["rollout"])
        r_roll.append(br["rollout"])
        agree += tuple(bf["choices"]) == tuple(br["choices"])
    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "groups": len(by_group), "candidates": len(scored),
        "mean_rollout_greedy": float(sum(g_roll) / len(g_roll)) if g_roll else None,
        "mean_rollout_best_by_field": float(sum(f_roll) / len(f_roll)) if f_roll else None,
        "mean_rollout_best_by_rollout": float(sum(r_roll) / len(r_roll)) if r_roll else None,
        "share_best_field_is_best_rollout": float(agree / len(f_roll)) if f_roll else None,
        "capped_share": float(tot_c / tot_s) if tot_s else None,
        "seconds": round(time.time() - t0, 1),
    }
    with open(str(args.out) + ".summary.json", "w") as f:
        f.write(json.dumps(summary, indent=1))
    print(f"wrote {len(scored)} scored rows, {len(by_group)} groups -> {args.out} "
          f"(greedy {summary['mean_rollout_greedy']:+.2f} best-field {summary['mean_rollout_best_by_field']:+.2f} "
          f"best-rollout {summary['mean_rollout_best_by_rollout']:+.2f}, "
          f"agree {summary['share_best_field_is_best_rollout']:.2f}, capped {summary['capped_share']:.3f}, "
          f"{summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
