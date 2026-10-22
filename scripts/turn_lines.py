"""Whole-turn line generation (self-imitation beyond turn-1 combos).

    python scripts/turn_lines.py --ckpt <policy best.pt> --starts turn1 --decks <split> \
        --n N --k K --seed S --out <dir> [--opp policy|passive]
    python scripts/turn_lines.py --ckpt <policy best.pt> --starts dags[:<path>] --decks <split> \
        --n N --k K --seed S --out <dir> [--opp policy|passive] [--min-turn T] [--max-turn T]

A start = (config, prefix choices); the acting player `actor` is whoever's turn it is after
the prefix; the other player's decisions in that turn come from the policy greedily
(--opp policy, default: it chains/negates with its real hand) or the passive responder
(--opp passive). From the start state 1 greedy + K sampled (temperature 1) copies run in
lockstep with batched forwards; the run stops when the next turn starts, the duel ends, or
a 400-decision cap (stuck).

`--starts turn1`: fresh hands exactly like turn1_lines random hands (deck split, opp_for,
hand_config, N hands per deck), prefix empty. `--starts dags`: rows of starts.jsonl where
BOTH deck names are in the requested split, deterministically sampled (stratified by turn).

Outputs replays only (config + full choices = prefix + new): <out>/lines.jsonl +
<out>/summary.json (+ printed table). Row fields match turn1_lines rows so the scorer and
selector reuse works, plus `start`, `prefix_len`, `actor`, `turn_start`, `actor_decisions`,
`opp_field`, `lp_diff` (and `v_actor`/`v_p0` only with --critic).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import Deck, load_ydk  # noqa: E402
from decklight.engine.cpp_env import CppEnv  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402
from decklight.eval.bc_policy import BCPolicy  # noqa: E402
from decklight.eval.turn1 import (  # noqa: E402
    CAP,
    actor_field_of,
    config_to_dict,
    critic_vp0,
    dags_config,
    fail_loud,
    hand_config,
    load_critic_local,
    lp_diff_of,
    opp_for,
    replay_prefix,
    resolve_decks,
    run_args,
    run_turn_from_obs,
    stable_seed,
    turn_player_of,
)

DEFAULT_STARTS = ROOT / "data" / "cache" / "dags" / "starts.jsonl"

_W: dict = {}


def _init(cfg: dict) -> None:
    """Per-worker state (spawn): its own engine, greedy + sampler policies (optional critic)."""
    import torch as _torch

    _torch.set_num_threads(cfg["threads"])
    engine = Engine()
    greedy = BCPolicy(cfg["ckpt"], cfg["emb"], temperature=0.0, threads=cfg["threads"], device=cfg["device"])
    sampler = BCPolicy(cfg["ckpt"], cfg["emb"], temperature=1.0, seed=cfg["seed"],
                       threads=cfg["threads"], device=cfg["device"])
    critic = None
    if cfg.get("critic"):
        import numpy as _np

        emb = dict(_np.load(cfg["emb"]))
        critic = load_critic_local(Path(cfg["critic"]), emb, run_args(Path(cfg["critic"])))
        critic.to(_torch.device(cfg["device"])).eval()
        hist = int(run_args(Path(cfg["critic"])).get("hist", 0) or 0)
    else:
        hist = 0
    _W.update(engine=engine, greedy=greedy, sampler=sampler, critic=critic, hist=hist,
              device=_torch.device(cfg["device"]), cfg=cfg)


def _score(ends: list[dict], actor: int) -> list[float] | None:
    if _W["critic"] is None:
        return None
    vp0 = critic_vp0(_W["critic"], ends, _W["hist"], _W["device"])
    return [v if actor == 0 else -v for v in vp0]


def _row(task: dict, kind: str, sample: int, cfg_dict: dict, full: list[int],
         res: dict, v_actor) -> dict:
    r = {"deck": task["actor_deck"], "opp": task["opp_deck"], "split": task["split"],
         "hand": int(task["hand"]), "source": task["source"], "start": task["start"],
         "kind": kind, "sample": int(sample), "config": cfg_dict,
         "auto_single": True, "choices": [int(c) for c in full],
         "prefix_len": int(task["prefix_len"]), "actor": int(task["actor"]),
         "turn_start": int(task["turn_start"]),
         "actor_decisions": int(res["actor_decisions"]),
         "p0_decisions": int(res["p0_decisions"]),
         "opp_decisions": int(res.get("opp_decisions", 0)),
         "opp_options": int(res.get("opp_options", 0)),
         "field": int(res["field"]), "opp_field": int(res.get("opp_field", 0)),
         "lp_diff": int(res.get("lp_diff", 0)),
         "stuck": bool(res["stuck"]), "train_ok": bool(task["train_ok"])}
    if task.get("game_id") is not None:
        r["game_id"] = task["game_id"]
    if v_actor is not None:
        r["v_actor"] = float(v_actor)
        r["v_p0"] = float(v_actor if int(task["actor"]) == 0 else -v_actor)
    return r


def _start_task(task: dict) -> dict:
    """Rows of one unit of work: one start (1 greedy + K sampled whole-turn lines)."""
    t0 = time.time()
    cfg, engine = _W["cfg"], _W["engine"]
    K, S = cfg["k"], cfg["seed"]
    if task["source"] == "turn1":
        table = {name: Deck(tuple(m), tuple(e)) for name, (m, e) in task["table"].items()}
        dc = hand_config(table[task["deck0"]], table[task["opp_deck"]],
                         stable_seed(S, task["actor_deck"], task["hand"]))
        prefix: list[int] = []
    else:
        dc = dags_config({"main0": task["main0"], "extra0": task["extra0"],
                          "main1": task["main1"], "extra1": task["extra1"],
                          "seed": task["seed"], "hand": task.get("hand_n", 5),
                          "draw": task.get("draw", 1), "lp": task.get("lp", 8000),
                          "setup": task.get("setup", [])})
        prefix = [int(c) for c in task["prefix"]]
    K_envs = 1 + K
    envs, obs_list = [], []
    try:
        for _ in range(K_envs):
            env = CppEnv(engine, dc, True)
            ob = env.reset()
            if ob is None:
                raise ValueError("game ended before the first decision")
            for c in prefix:
                if ob is None:
                    raise ValueError("game ended inside the prefix")
                from decklight.demos.game import _opts

                if not 0 <= int(c) < len(_opts(ob)):
                    raise ValueError(f"prefix choice {c} illegal")
                ob = env.step(int(c))
            if ob is None:
                raise ValueError("game ended at the end of the prefix")
            envs.append(env)
            obs_list.append(ob)
    except Exception as e:  # noqa: BLE001 - a prefix that no longer replays drops the start
        return {"rows": [], "dropped": f"{task['start']}: {e!r}"[:200]}
    turn = int(obs_list[0]["global"][2])
    actor = turn_player_of(obs_list[0], envs[0].player)
    task = dict(task, actor=int(actor), turn_start=int(turn),
                actor_deck=task["seat_decks"][int(actor)], opp_deck=task["seat_decks"][1 - int(actor)])
    _W["sampler"].rng = np.random.default_rng(stable_seed(S, task["start"], "sample"))
    opp_pol = _W["greedy"] if cfg["opp"] == "policy" else None
    res = run_turn_from_obs(envs, obs_list, [_W["greedy"]] + [_W["sampler"]] * K,
                            int(actor), int(turn),
                            opp=cfg["opp"], opp_pol=opp_pol, cap=CAP)
    vals = _score(res, int(actor))
    cd = config_to_dict(dc)
    rows = []
    full_g = prefix + res[0]["choices"]
    rows.append(_row(task, "greedy", -1, cd, full_g, res[0], vals[0] if vals else None))
    for j in range(K):
        full = prefix + res[1 + j]["choices"]
        rows.append(_row(task, "sample", j, cd, full, res[1 + j], vals[1 + j] if vals else None))
    secs = round(time.time() - t0, 1)
    for r in rows:
        r["secs"] = secs
    return {"rows": rows, "dropped": None}


def _bucket(turn) -> str:
    if turn is None:
        return "unknown"
    try:
        t = int(turn)
    except (TypeError, ValueError):
        return "unknown"
    if t <= 2:
        return "2"
    if t <= 5:
        return "3-5"
    return "6+"


def sample_starts(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic stratified sample of N starts (buckets 2 / 3-5 / 6+ round-robin)."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {"2": [], "3-5": [], "6+": [], "unknown": []}
    for r in rows:
        buckets[_bucket(r.get("turn"))].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    out: list[dict] = []
    order = ["2", "3-5", "6+", "unknown"]
    while len(out) < n and any(buckets[b] for b in order):
        for b in order:
            if buckets[b] and len(out) < n:
                out.append(buckets[b].pop(0))
    return out


def load_dags_starts(path: Path, deck_names: set[str], n: int, seed: int,
                     min_turn: int | None, max_turn: int | None) -> tuple[list[dict], dict, int]:
    """Filtered + sampled starts rows. Returns (sampled, pool_histogram, n_skipped_empty_decks)."""
    pool_hist: dict[str, int] = {}
    rows: list[dict] = []
    skipped_empty = 0
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        r = json.loads(raw)
        decks = r.get("decks") or []
        if len(decks) != 2 or not all(decks):
            skipped_empty += 1
            continue
        if decks[0] not in deck_names or decks[1] not in deck_names:
            continue
        t = r.get("turn")
        if min_turn is not None and (t is None or int(t) < min_turn):
            continue
        if max_turn is not None and (t is None or int(t) > max_turn):
            continue
        rows.append(r)
        pool_hist[_bucket(t)] = pool_hist.get(_bucket(t), 0) + 1
    sampled = sample_starts(rows, n, seed)
    return sampled, pool_hist, skipped_empty


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="policy checkpoint (its metrics.jsonl gives the arch)")
    ap.add_argument("--critic", type=Path, default=None, help="optional privileged critic checkpoint (stores v_actor)")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--starts", required=True, help="turn1 | dags[:<starts.jsonl path>]")
    ap.add_argument("--decks", required=True, help="train | holdout | gauntlet | comma-separated ydk paths")
    ap.add_argument("--n", type=int, required=True, help="hands per deck (turn1) or starts sampled (dags)")
    ap.add_argument("--k", type=int, required=True, help="sampled lines per start")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opp", default="policy", choices=["policy", "passive"],
                    help="how the other player's windows play (default: policy greedily)")
    ap.add_argument("--min-turn", type=int, default=None)
    ap.add_argument("--max-turn", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--task-timeout", type=int, default=300, help="s without any finished start before giving up")
    args = ap.parse_args(argv)
    if args.n < 1 or args.k < 1:
        print("turn_lines: --n and --k must be >= 1", file=sys.stderr)
        return 2
    t0 = time.time()
    split, decks, train_ok = resolve_decks(args.decks, args.seed)
    if not decks:
        raise fail_loud(["--decks {args.decks!r} resolved to 0 decks"], args.out)
    table = {}
    for name, path in decks:
        d = load_ydk(path)
        table[name] = (tuple(d.main), tuple(d.extra))
    names = {name for name, _ in decks}

    tasks: list[dict] = []
    if args.starts == "turn1":
        for h in range(args.n):  # hand-major: every deck gets its first hands early
            for name, _ in decks:
                opp = opp_for(sorted(table), name, args.seed, h)
                tasks.append({"deck0": name, "actor_deck": name, "opp_deck": opp,
                              "seat_decks": [name, opp], "split": split, "train_ok": train_ok,
                              "table": table, "source": "turn1", "hand": h,
                              "start": f"{name}-h{h}", "prefix": [], "prefix_len": 0,
                              "actor": 0, "turn_start": 1, "game_id": None})
    elif args.starts == "dags" or args.starts.startswith("dags:"):
        spath = Path(args.starts.split(":", 1)[1]) if ":" in args.starts else DEFAULT_STARTS
        sampled, pool_hist, skipped_empty = load_dags_starts(
            spath, names, args.n, args.seed, args.min_turn, args.max_turn)
        hist: dict[str, int] = {}
        for r in sampled:
            hist[_bucket(r.get("turn"))] = hist.get(_bucket(r.get("turn")), 0) + 1
        print(f"turn_lines: dags pool {len(sampled)}/{sum(pool_hist.values())} sampled "
              f"(pool {pool_hist}, sample {hist}, skipped-empty-decks {skipped_empty})", flush=True)
        for i, r in enumerate(sampled):
            pre = [int(c) for c in r.get("prefix", [])]
            tasks.append({"split": split, "train_ok": train_ok, "source": "dags",
                          "hand": i, "start": f"{r.get('game_id')}:{len(pre)}",
                          "game_id": r.get("game_id"), "prefix": pre, "prefix_len": len(pre),
                          "seat_decks": [r["decks"][0], r["decks"][1]],
                          "actor_deck": r["decks"][0], "opp_deck": r["decks"][1],
                          "actor": -1, "turn_start": int(r["turn"]) if r.get("turn") is not None else -1,
                          "main0": r["main0"], "extra0": r["extra0"],
                          "main1": r["main1"], "extra1": r["extra1"],
                          "seed": int(r["seed"]), "hand_n": int(r.get("hand", 5)),
                          "draw": int(r.get("draw", 1)), "lp": int(r.get("lp", 8000)),
                          "setup": r.get("setup", [])})
    else:
        print(f"turn_lines: --starts must be turn1 or dags[:path], got {args.starts!r}", file=sys.stderr)
        return 2
    if not tasks:
        raise fail_loud(["0 starts (no tasks built: deck filter or turn filter removed everything?)"], args.out)

    wcfg = {"ckpt": str(args.ckpt), "critic": str(args.critic) if args.critic else None,
            "emb": str(args.emb), "seed": args.seed, "k": args.k,
            "threads": args.threads, "device": args.device, "opp": args.opp}
    args.out.mkdir(parents=True, exist_ok=True)
    part = open(args.out / "lines.partial.jsonl", "w")
    rows: list[dict] = []
    dropped = 0

    def take(d: dict) -> None:
        nonlocal dropped
        take.n += 1
        if d.get("dropped"):
            dropped += 1
            print(f"PROGRESS {len(rows)} rows, task {take.n}/{len(tasks)}: dropped start "
                  f"({d['dropped'][:120]}), {time.time() - t0:.0f} s total", flush=True)
            return
        res = d["rows"]
        rows.extend(res)
        for r in res:
            part.write(json.dumps(r) + "\n")
        part.flush()
        if res:
            r0 = res[0]
            print(f"PROGRESS {len(rows)} rows, task {take.n}/{len(tasks)}: {r0['deck']} {r0['source']} "
                  f"{r0['start']} actor {r0['actor']} t{r0['turn_start']} {r0['secs']} s, "
                  f"greedy field {r0['field']}, {time.time() - t0:.0f} s total", flush=True)

    take.n = 0
    if args.procs <= 1:
        _init(wcfg)
        for t in tasks:
            take(_start_task(t))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.procs, len(tasks)), initializer=_init, initargs=(wcfg,)) as pool:
            it = pool.imap_unordered(_start_task, tasks, chunksize=1)
            while take.n < len(tasks):
                try:  # watchdog: one start hung in the engine must not stall the job silently
                    d = it.next(timeout=args.task_timeout)
                except StopIteration:
                    break
                except mp.TimeoutError:
                    msg = (f"no start finished in {args.task_timeout} s: {len(tasks) - take.n} task(s) hung; "
                           f"keeping the {take.n} finished")
                    print(f"ALERT {msg}", file=sys.stderr, flush=True)
                    args.out.mkdir(parents=True, exist_ok=True)
                    with open(args.out / "ALERT", "a") as fa:
                        fa.write(msg + "\n")
                    pool.terminate()
                    break
                take(d)
    part.close()
    rows.sort(key=lambda r: (r["deck"], r["source"], r["start"],
                             {"greedy": 0, "sample": 1}[r["kind"]], r["sample"]))
    problems: list[str] = []
    if not rows:
        problems.append("0 rows written")
    else:
        if dropped / max(len(tasks), 1) > 0.05:
            problems.append(f"{dropped}/{len(tasks)} prefix replay failures (>5%)")
        by_deck: dict[str, list[dict]] = {}
        for r in rows:
            by_deck.setdefault(r["deck"], []).append(r)
        for d, dr in by_deck.items():
            if all(r["stuck"] for r in dr):
                problems.append(f"deck {d}: all {len(dr)} runs stuck (cap {CAP})")
        if args.k >= 4:
            for d, dr in by_deck.items():
                gh: dict[str, list[dict]] = {}
                for r in dr:
                    if r["kind"] in ("greedy", "sample"):
                        gh.setdefault(r["start"], []).append(r)
                if gh and all(all(s["choices"] == next(g["choices"] for g in grp if g["kind"] == "greedy")
                                   for s in grp if s["kind"] == "sample") for grp in gh.values()):
                    problems.append(f"deck {d}: greedy and every sample identical on every start (no sampling?)")
        by_actor: dict[int, list[dict]] = {}
        for r in rows:
            by_actor.setdefault(int(r["actor"]), []).append(r)
        for a, ar in by_actor.items():
            starts_of = {r["start"] for r in ar}
            zero = {r["start"] for r in ar if int(r["actor_decisions"]) == 0}
            if starts_of and len(zero) / len(starts_of) > 0.5:
                problems.append(f"actor {a}: {len(zero)}/{len(starts_of)} starts with 0 actor decisions (>50%)")
    if problems:
        raise fail_loud(problems, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "lines.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_deck = {}
    for r in rows:
        by_deck.setdefault(r["deck"], []).append(r)
    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "split": split, "K": args.k, "cap": CAP, "opp": args.opp,
        "seconds": round(time.time() - t0, 1), "starts": len(tasks), "dropped_prefix": dropped,
        "decks": {d: {"runs": len(dr), "starts": len({r["start"] for r in dr}),
                      "stuck_rate": float(sum(r["stuck"] for r in dr) / len(dr)),
                      "actors": sorted({int(r["actor"]) for r in dr}),
                      "turns": sorted({int(r["turn_start"]) for r in dr})} for d, dr in by_deck.items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {len(rows)} rows, {len(by_deck)} decks, {len(tasks) - dropped}/{len(tasks)} starts "
          f"-> {args.out} ({summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
