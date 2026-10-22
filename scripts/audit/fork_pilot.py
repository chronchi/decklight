"""Fork pilot: measure the RL signal at forked decisions (plan 2026-10-20-rl-signal M1).

Pass 1 (engine replay only, laptop OK): replay recorded raw-policy games, tag every decision with
>= 2 legal options into categories (ht / place / t1 / any), print per-category counts and sample
--n-per decisions per category (at most 2 per game). --select-only writes the sample and stops.

Pass 2 (vast GPU box): for each sampled decision, replay to it, draw --samples honest worlds with a
Determinizer, play the played option plus up to --k-extra alternatives on the same worlds to the end
of the turn (value@h1) and the end of the game (done@h99), and write one row per decision.

--summarize merges the per-proc rows into headroom / fork-signal / critic-check numbers.

Usage (vast GPU box):
    python scripts/audit/fork_pilot.py --games runs/<job>/raw --ckpt data/cache/P/best.pt \\
        --value data/cache/value_p2/best.pt --out data/cache/fork_pilot/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.obs import LOC_ID
from decklight.search.determinize import Determinizer, SeenTracker, key_rank_of_choice
from decklight.search.honest import paired_gap
from decklight.search.turn1 import _batch, _masked, drive_many
from decklight.search.window import WindowSearch

CATS = ("ht", "place", "t1", "any")
_HAND = LOC_ID[C.LOCATION_HAND]


def fail(msg: str) -> None:
    print(f"ALERT fork_pilot: {msg}", flush=True)
    raise SystemExit(2)


def tag_decision(obs) -> set[str]:
    """Categories of one decision with >= 2 legal options (empty set otherwise)."""
    acts, mask = obs["actions"], obs["action_mask"].astype(bool)
    n = int(mask.sum())
    if n < 2:
        return set()
    tags = {"any"}
    kinds = acts[mask, 0]
    msg = int(obs["global"][21])
    if msg == C.MSG_SELECT_CHAIN and np.any(kinds == Kind.PASS):
        locs = acts[mask, 7]
        if bool(np.any((kinds == Kind.ACTIVATE) & (locs == _HAND))):
            tags.add("ht")
    if bool(np.all(kinds == Kind.PLACE)):
        tags.add("place")
    if int(obs["global"][2]) == 1 and int(obs["global"][5]) == 1:
        tags.add("t1")
    return tags


def load_games(games_dir: Path) -> list[dict]:
    recs = []
    for f in sorted(games_dir.glob("games-*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def pass1(engine, recs: list[dict], seed: int, n_per: int):
    """Replay every game; return (sampled decisions, stats dict, n_games)."""
    from decklight.analysis.events import deck_path
    from decklight.decks import load_ydk

    per_game: list[dict] = []  # {game_idx, k, tags, actor, turn, deck, n_opts, played}
    cat_n: Counter = Counter()
    cat_games: dict[str, set] = {c: set() for c in CATS}
    hist: Counter = Counter()
    n_decisions = 0
    n_mismatch = 0
    for rec in recs:
        cfg = DuelConfig(
            load_ydk(deck_path(rec["decks"][0])), load_ydk(deck_path(rec["decks"][1])), seed=int(rec["seed"])
        )
        env = CppEnv(engine, cfg, rec.get("auto_single", True))
        obs = env.reset()
        for k, c in enumerate(rec["choices"]):
            if obs is None or int(env.player) != int(rec["actors"][k]):
                n_mismatch += 1
                break
            n_decisions += 1
            n_opts = int(obs["action_mask"].sum())
            hist[n_opts] += 1
            tags = tag_decision(obs)
            for t in tags:
                cat_n[t] += 1
                cat_games[t].add(rec["game_idx"])
            if tags:
                per_game.append(
                    {
                        "game_idx": rec["game_idx"],
                        "k": k,
                        "tags": sorted(tags),
                        "actor": int(env.player),
                        "turn": int(obs["global"][2]),
                        "deck": rec["decks"][int(env.player)],
                        "n_opts": n_opts,
                        "played": int(c),
                    }
                )
            obs = env.step(int(c))
    mismatch_rate = n_mismatch / max(n_decisions, 1)
    n_games = len(recs)
    stats = {
        "games": n_games,
        "decisions": n_decisions,
        "mismatch_rate": mismatch_rate,
        "cats": {
            c: {"n": cat_n[c], "games_with": len(cat_games[c]), "dpg": cat_n[c] / n_games if n_games else 0.0}
            for c in CATS
        },
        "hist": {str(k): v for k, v in sorted(hist.items())},
    }
    # Sample per category: uniform, deterministic, at most 2 per game.
    by_cat: dict[str, list[dict]] = {c: [d for d in per_game if c in d["tags"]] for c in CATS}
    sampled: dict[tuple, dict] = {}
    for c in CATS:
        pool = by_cat[c]
        if not pool:
            if cat_n[c] >= 1:
                fail(f"0 sampled decisions in category {c} with {cat_n[c]} decisions")
            continue
        rng = random.Random(f"{seed}:{c}")
        order = list(pool)
        rng.shuffle(order)
        per_game_count: Counter = Counter()
        take = 0
        for d in order:
            if take >= n_per:
                break
            if per_game_count[d["game_idx"]] >= 2:
                continue
            per_game_count[d["game_idx"]] += 1
            take += 1
            key = (d["game_idx"], d["k"])
            if key in sampled:
                sampled[key]["sampled_in"].append(c)
            else:
                sampled[key] = {**d, "sampled_in": [c]}
        if cat_n[c] >= 1 and not any(c in v["sampled_in"] for v in sampled.values()):
            fail(f"0 sampled decisions in category {c} with {cat_n[c]} decisions")
    return sorted(sampled.values(), key=lambda d: (d["game_idx"], d["k"])), stats


def print_pass1(stats: dict) -> None:
    print(
        f"{stats['games']} games, {stats['decisions']} decisions, "
        f"replay mismatch rate {stats['mismatch_rate']:.3%}",
        flush=True,
    )
    for c in CATS:
        s = stats["cats"][c]
        print(
            f"  {c}: {s['n']} decisions in {s['games_with']} games, {s['dpg']:.2f} decisions/game", flush=True
        )
    print(f"option-count histogram: {stats['hist']}", flush=True)


def pick_candidates(obs, played: int, k_extra: int, logits: np.ndarray) -> tuple[list[int], list[float]]:
    """Played option first, then PASS (when not played), then the policy's top options."""
    acts, mask = obs["actions"], obs["action_mask"].astype(bool)
    z = logits.astype(np.float64)
    z[~mask] = -np.inf
    legal = [int(i) for i in np.flatnonzero(mask)]
    cands = [played]
    passes = [i for i in legal if int(acts[i, 0]) == Kind.PASS and i != played]
    cands += passes[:1]
    order = sorted(legal, key=lambda i: (-z[i], i))
    for i in order:
        if len(cands) >= 1 + k_extra:
            break
        if i not in cands:
            cands.append(i)
    if len(set(cands)) != len(cands):
        fail(f"duplicate candidate indices at {cands}")
    zmax = max(z[i] for i in legal)
    ez = {i: math.exp(z[i] - zmax) for i in legal}
    tot = sum(ez.values())
    return cands, [ez[i] / tot for i in cands]


def fork_gen(
    engine,
    ws: WindowSearch,
    rec: dict,
    sel: dict,
    cfg,
    det_tries: int,
    n_samples: int,
    seed: int,
    worlds_mode="honest",
):
    """One decision's fork: yield obs for policy logits, then roll out candidates on shared worlds.
    `worlds_mode` "true": every sample is the real state (an RL fork clones the self-play env, so the samples
    differ only by the rollout policy's sampling); "honest": Determinizer worlds of the acting seat. Honest worlds
    were rejected for half the states (forkpilot-20261020-1849: deck-search option order, opponent plays of hidden
    cards), and the kept half skews easy."""
    k = int(sel["k"])
    me = int(sel["actor"])
    choices = [int(c) for c in rec["choices"]]
    env = CppEnv(engine, cfg, rec.get("auto_single", True))
    obs = env.reset()
    tr = SeenTracker(me)
    tr.update(env)
    keys = []
    for c in choices[:k]:
        keys.append(key_rank_of_choice(obs, c))
        obs = env.step(c)
        tr.update(env)
    z = _masked((yield [obs])[0], obs)
    cands, probs = pick_candidates(obs, int(sel["played"]), sel["_k_extra"], z)
    kinds = [int(obs["actions"][i, 0]) for i in cands]
    turn = int(obs["global"][2])
    if worlds_mode == "true":
        worlds = [(cfg, choices[:k])] * n_samples
    else:
        det = Determinizer(
            engine,
            cfg,
            rec.get("auto_single", True),
            me,
            choices[:k],
            tr.seen,
            obs,
            random.Random(f"{seed}:{sel['game_idx']}:{k}"),
            prefix_keys=keys,
            tries=det_tries,
        )
        worlds = [w for w in (det.sample() for _ in range(n_samples)) if w is not None]
    if len(worlds) < n_samples // 2:
        return {
            "game_idx": sel["game_idx"],
            "k": k,
            "tags": sel["tags"],
            "error": f"only {len(worlds)} worlds",
        }
    capped_before = ws.capped
    branches = [(choices[:k] + [x], j, x, s) for j, x in enumerate(cands) for s in range(len(worlds))]
    game_seed = 10_000 + (
        int(sel["game_idx"]) if str(sel["game_idx"]).isdigit() else zlib.crc32(str(sel["game_idx"]).encode())
    )
    yield from ws.rollouts_gen(
        cfg,
        rec.get("auto_single", True),
        branches,
        me,
        turn_start=turn,
        game_seed=game_seed,
        decision_k=k,
        worlds=worlds,
    )
    v1 = [list(map(float, col)) for col in ws.last_samples_all["value@h1"]]
    end = [list(map(float, col)) for col in ws.last_samples_all["done@h99"]]
    if any(math.isnan(x) for col in v1 + end for x in col):
        fail(f"NaN scores at game {sel['game_idx']} k {k}")
    return {
        "game_idx": sel["game_idx"],
        "k": k,
        "turn": turn,
        "actor": me,
        "deck": sel["deck"],
        "tags": sel["tags"],
        "n_opts": int(sel["n_opts"]),
        "cands": cands,
        "played_j": 0,
        "probs": probs,
        "kinds": kinds,
        "v1": v1,
        "end": end,
        "capped": [capped_before, ws.capped],
        "worlds": worlds_mode,
    }


def split_headroom(end: list[list[float]], played_j: int = 0) -> float:
    """Unbiased gain of the best alternative over played: pick best on even worlds, measure on odd
    worlds, then the reverse, and average (split-sample, so luck does not inflate it)."""
    tot = 0.0
    for pick, measure in ((slice(0, None, 2), slice(1, None, 2)), (slice(1, None, 2), slice(0, None, 2))):
        means = [float(np.mean(col[pick])) if len(col[pick]) else -math.inf for col in end]
        best = max(range(len(end)), key=lambda j: (means[j], -j))
        tot += float(np.mean(end[best][measure])) - float(np.mean(end[played_j][measure]))
    return tot / 2


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation with averaged tied ranks."""

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def summarize(stem_rows: list[dict], stats: dict) -> dict:
    usable = [
        r for r in stem_rows if "error" not in r and r.get("end") and min(len(c) for c in r["end"]) >= 2
    ]
    if stem_rows and len(usable) / len(stem_rows) < 0.5:
        fail(f"only {len(usable)}/{len(stem_rows)} rows usable")
    if not usable:
        fail(f"no usable rows in {len(stem_rows)} rows")
    out: dict = {"rows": len(stem_rows), "usable": len(usable), "cats": {}}
    n_branch = sum(len(r["cands"]) * len(r["end"][0]) for r in usable)
    n_capped = sum(r["capped"][1] - r["capped"][0] for r in usable)
    out["capped_share"] = n_capped / n_branch if n_branch else 0.0
    for c in CATS:
        rows = [r for r in usable if c in r["tags"]]
        d = {"n": len(rows)}
        if rows:
            gains = [split_headroom(r["end"], r["played_j"]) for r in rows]
            d["headroom_mean"] = float(np.mean(gains))
            d["headroom_se"] = float(st.stdev(gains) / math.sqrt(len(gains))) if len(gains) > 1 else 0.0
            d["points_per_game"] = d["headroom_mean"] / 2 * 100 * stats["cats"][c]["dpg"]
            sig, ratios = 0, []
            for r in rows:
                means = [float(np.mean(col)) for col in r["end"]]
                best = max(range(len(means)), key=lambda j: (means[j], -j))
                m, se, _ = paired_gap(r["end"][best], r["end"][r["played_j"]])
                if m > 0 and math.isfinite(se) and m >= 2 * se:
                    sig += 1
                ratios.append(
                    abs(m / se)
                    if se not in (0.0, math.inf) and math.isfinite(se)
                    else (math.inf if m > 0 else 0.0)
                )
            d["fork_sig_share"] = sig / len(rows)
            med = float(np.median([x for x in ratios if math.isfinite(x)] or [0.0]))
            d["fork_med_abs_gap_se"] = med
            sp = [
                spearman([float(np.mean(col)) for col in r["v1"]], [float(np.mean(col)) for col in r["end"]])
                for r in rows
                if len(r["cands"]) >= 3
            ]
            d["critic_spearman_n"] = len(sp)
            d["critic_spearman_mean"] = float(np.mean(sp)) if sp else None
            d["critic_argmax_agree"] = sum(
                max(range(len(r["cands"])), key=lambda j: (float(np.mean(r["v1"][j])), -j))
                == max(range(len(r["cands"])), key=lambda j: (float(np.mean(r["end"][j])), -j))
                for r in rows
            ) / len(rows)
            ht = [
                r
                for r in rows
                if c == "ht"
                and int(r["kinds"][r["played_j"]]) == Kind.ACTIVATE
                and Kind.PASS in [Kind(x) for x in r["kinds"]]
            ]
            if ht:
                gv = [
                    float(np.mean(r["v1"][r["kinds"].index(int(Kind.PASS))]))
                    - float(np.mean(r["v1"][r["played_j"]]))
                    for r in ht
                ]
                ge = [
                    float(np.mean(r["end"][r["kinds"].index(int(Kind.PASS))]))
                    - float(np.mean(r["end"][r["played_j"]]))
                    for r in ht
                ]
                d["ht_pass_minus_act_v1"] = float(np.mean(gv))
                d["ht_pass_minus_act_end"] = float(np.mean(ge))
                d["ht_pass_minus_act_n"] = len(ht)
        out["cats"][c] = d
    return out


def print_summary(summ: dict) -> None:
    print(f"{summ['usable']}/{summ['rows']} rows usable, capped share {summ['capped_share']:.3%}")
    print(
        f"{'cat':6} {'n':>4} {'headroom':>9} {'se':>7} {'pt/g':>6} {'sig':>5} {'|g|/se':>6} "
        f"{'spear':>6} {'agree':>6}"
    )
    for c in CATS:
        d = summ["cats"][c]
        if not d["n"]:
            print(f"{c:6} {0:>4}  (no rows)")
            continue
        sp = f"{d['critic_spearman_mean']:.2f}" if d["critic_spearman_mean"] is not None else "n/a"
        print(
            f"{c:6} {d['n']:>4} {d['headroom_mean']:>+9.4f} {d['headroom_se']:>7.4f} "
            f"{d['points_per_game']:>+6.2f} {d['fork_sig_share']:>5.2f} {d['fork_med_abs_gap_se']:>6.2f} "
            f"{sp:>6} {d['critic_argmax_agree']:>6.2f}",
            flush=True,
        )
        if "ht_pass_minus_act_v1" in d:
            print(
                f"  ht pass-activate: v1 {d['ht_pass_minus_act_v1']:+.4f}, "
                f"end {d['ht_pass_minus_act_end']:+.4f} (n={d['ht_pass_minus_act_n']})",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, required=True, help="search_match game files dir (games-*.jsonl)")
    ap.add_argument("--n-per", type=int, default=100)
    ap.add_argument("--k-extra", type=int, default=3)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--det-tries", type=int, default=32)
    ap.add_argument(
        "--worlds", choices=("honest", "true"), default="honest", help="true: forks of the real state"
    )
    ap.add_argument("--max-rollout", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--value", type=Path, default=None)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        rows = [
            json.loads(x)
            for f in sorted(args.out.parent.glob(f"{args.out.stem}-*.jsonl"))
            for x in f.read_text().splitlines()
            if x.strip()
        ]
        p1_path = args.out.parent / f"{args.out.stem}.pass1.json"
        if not p1_path.is_file():
            fail(f"no pass1 stats at {p1_path}")
        summ = summarize(rows, json.loads(p1_path.read_text()))
        args.out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
        print_summary(summ)
        if summ["capped_share"] >= 0.02:
            print(f"ALERT fork_pilot: capped share {summ['capped_share']:.3%} >= 2%", flush=True)
        return

    engine = Engine()
    recs = load_games(args.games)
    if not recs:
        fail(f"no games in {args.games}")
    sampled, stats = pass1(engine, recs, args.seed, args.n_per)
    if stats["mismatch_rate"] > 0.05:
        fail(f"replay mismatch rate {stats['mismatch_rate']:.3%} > 5%")
    print_pass1(stats)
    if args.proc == 0:  # the summary needs the per-category counts even when the fork procs are time-capped
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.parent.joinpath(f"{args.out.stem}.pass1.json").write_text(json.dumps(stats, indent=1))

    if args.select_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for d in sampled:
                f.write(json.dumps(d) + "\n")
        args.out.parent.joinpath(f"{args.out.stem}.pass1.json").write_text(json.dumps(stats, indent=1))
        print(f"{len(sampled)} sampled decisions -> {args.out}", flush=True)
        return

    if args.ckpt is None or args.value is None:
        fail("pass 2 needs --ckpt and --value")
    from decklight.analysis.events import deck_path
    from decklight.decks import load_ydk
    from decklight.eval.bc_policy import BCPolicy
    from decklight.eval.value import ValueNet

    mine = sampled[args.proc :: args.nproc]
    pol = BCPolicy(args.ckpt, args.emb, threads=1, device=args.device, batch=args.batch)
    value = ValueNet(args.value, args.emb, device=args.device, batch=args.batch).values

    def forward(obs_list):
        return np.concatenate(
            [pol.logits(_batch(obs_list[i : i + args.batch])) for i in range(0, len(obs_list), args.batch)]
        )

    def never(buf):
        raise AssertionError("unused")

    by_game = {r["game_idx"]: r for r in recs}
    cfgs = {}
    for d in mine:
        rec = by_game[d["game_idx"]]
        if d["game_idx"] not in cfgs:
            cfgs[d["game_idx"]] = (
                rec,
                DuelConfig(
                    load_ydk(deck_path(rec["decks"][0])),
                    load_ydk(deck_path(rec["decks"][1])),
                    seed=int(rec["seed"]),
                ),
            )

    def make_ws() -> WindowSearch:
        # one per decision, as replicate_overrides does: drive_many interleaves the jobs, so a shared instance would
        # mix `capped` (and last_samples_all) between decisions
        return WindowSearch(
            engine,
            never,
            k=64,
            horizons=(1, 99),
            samples=args.samples,
            seed=args.seed,
            value=value,
            max_rollout=args.max_rollout,
            rollout_temp=1.0,
        )

    jobs = []
    for i, d in enumerate(mine):
        rec, cfg = cfgs[d["game_idx"]]
        jobs.append(
            (
                i,
                fork_gen(
                    engine,
                    make_ws(),
                    rec,
                    {**d, "_k_extra": args.k_extra},
                    cfg,
                    args.det_tries,
                    args.samples,
                    args.seed,
                    args.worlds,
                ),
            )
        )
    path = args.out.parent / f"{args.out.stem}-{args.proc}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with path.open("w") as f:
        for _, res in drive_many(jobs, forward, args.width):
            f.write(json.dumps(res) + "\n")
            f.flush()  # a time-capped proc keeps every finished row
            n_ok += "error" not in res
    if args.proc == 0:
        path.parent.joinpath(f"{args.out.stem}.pass1.json").write_text(json.dumps(stats, indent=1))
    print(f"proc {args.proc}: {len(mine)} decisions, {n_ok} forked", flush=True)
    if mine and n_ok == 0:
        fail("0 rows written")


if __name__ == "__main__":
    main()
