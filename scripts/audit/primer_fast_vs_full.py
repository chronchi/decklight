"""Fast vs Full primers (pre-registered in NOTEBOOK 2026-10-19): does a primer built from raw-P games describe the Full
product's play as well as a primer built from Full games?

Inputs: the two scored summary caches of ``runs/primer-fvf-*`` (``score-full`` and ``score-fast``). A game is
identified by (seed, decks); Full games are split in two halves by a hash of that key. Primers compared, each judged
on the Full half it was not built from (A->B and B->A, pooled per deck):
- ``full``: Full games of the other half (the reference; ~30 games/deck);
- ``fast_same``: Fast games on exactly the other half's deals (same games count as ``full``);
- ``fast5``: every Fast game except those on the judged half's deals (~285/deck; about Full's cost).
No Fast primer ever sees a judged deal: the same opening hand would leak the opener into the test (the first smoke
built fast primers from all deals and scored +15 pp on openers in-sample).
Metrics per deck, on the held-out half:
- ``opener_exact``: share of the deck's going-first games whose first-3-play sequence is one of the primer's top-3
  openers; ``opener_first``: the same for the first play only;
- ``key5_share``: share of the deck's positive play credit (value_p2 impact) carried by the primer's top-5 key cards;
- ``jacc_key5`` / ``jacc_open3``: overlap with the ``full`` primer of the same half (agreement; ``full`` vs the other
  Full half is the noise floor).
``--split seed`` (default) holds out one Full deal seed per fold instead of hash halves (see its help).
Decision rule: bootstrap over decks (paired) of fast5 - full and fast_same - full on opener_exact and key5_share.

Usage: uv run python scripts/audit/primer_fast_vs_full.py runs/<name> [--json out.json] [--show 3]
Exit 2 when a cache is missing/unscored, a half is empty, or fewer than 8 decks have both halves.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

MIN_DECKS = 8
BOOT = 10000


def _key(s: dict) -> str:
    return f"{s['seed']}|{'|'.join(s['decks'])}"


def _half(s: dict) -> int:
    return hashlib.sha1(_key(s).encode()).digest()[0] & 1


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def _boot(diffs: list[float], rng: random.Random) -> tuple[float, float, float]:
    m = sum(diffs) / len(diffs)
    bs = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOT))
    return m, bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT)]


def main(argv=None) -> int:
    import ygo_zero.analysis.primer as P
    from ygo_zero.engine.resources import load_cards
    from ygo_zero.web.view import DB

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument(
        "--split",
        choices=("seed", "hash"),
        default="seed",
        help="held-out folds: seed = leave one Full deal seed out (default; a seed largely fixes each deck's opening "
        "hand, so the pre-registered hash halves leaked hands into the test, NOTEBOOK 2026-10-19); hash = halves",
    )
    args = ap.parse_args(argv)

    summ = {}
    for t in ("full", "fast"):
        p = args.run / f"score-{t}" / "summaries.json.gz"
        if not p.exists():
            print(f"ALERT primer_fast_vs_full: missing {p}", flush=True)
            return 2
        summ[t] = json.loads(gzip.decompress(p.read_bytes()))
        for s in summ[t]:
            P._canon_summary(s)
        bad = sum(not P._is_scored(s) for s in summ[t])
        if bad:
            print(f"ALERT primer_fast_vs_full: {bad} unscored {t} summaries", flush=True)
            return 2
    cards, names = load_cards(), DB.names()
    P.MIN_GAMES = 1

    seeds = sorted({s["seed"] for s in summ["full"]})
    fold_of = (lambda s: seeds.index(s["seed"])) if args.split == "seed" else _half
    FOLDS = range(len(seeds) if args.split == "seed" else 2)
    full_half = {h: [s for s in summ["full"] if fold_of(s) == h] for h in FOLDS}
    if not all(full_half.values()):
        print("ALERT primer_fast_vs_full: a Full fold is empty", flush=True)
        return 2
    full_keys = {h: {_key(s) for s in full_half[h]} for h in FOLDS}
    all_full = set().union(*full_keys.values())

    def judged(s, h):
        # a Fast game the fold-h primers must not see: same seed (seed split) or same deal (hash split)
        return s["seed"] == seeds[h] if args.split == "seed" else _key(s) in full_keys[h]

    fast_on = lambda keys: [s for s in summ["fast"] if _key(s) in keys]
    fast1 = fast_on(all_full)
    print(
        f"games: full {len(summ['full'])} (folds {'/'.join(str(len(v)) for v in full_half.values())}, {args.split}), "
        f"fast {len(summ['fast'])} (on full deals {len(fast1)})",
        flush=True,
    )
    if len(fast1) < 0.9 * len(all_full):
        print(
            f"ALERT primer_fast_vs_full: only {len(fast1)} Fast games on {len(all_full)} Full deals",
            flush=True,
        )
        return 2

    def build(ss):
        return P.aggregate(copy.deepcopy(ss), cards, names, {})["decks"]

    prim = {}  # source -> half used for judging -> primers
    for h in FOLDS:  # built from the other folds, judged on h
        prim.setdefault("full", {})[h] = build([s for s in summ["full"] if fold_of(s) != h])
        prim.setdefault("fast_same", {})[h] = build(fast_on(all_full - full_keys[h]))
        prim.setdefault("fast5", {})[h] = build([s for s in summ["fast"] if not judged(s, h)])

    # held-out events per (deck hash): openers going first, positive credit per code
    hash_of_name: dict[str, str] = {}

    def deck_hash(name):
        if name not in hash_of_name:
            from ygo_zero.analysis.events import deck_path
            from ygo_zero.decks import load_ydk

            hash_of_name[name] = P.hash_of(load_ydk(deck_path(name)))
        return hash_of_name[name]

    ev = {h: {} for h in FOLDS}  # h -> deck hash -> {"openers": [tuple], "credit": Counter}
    for h in FOLDS:
        for s in full_half[h]:
            steps = s["steps"]
            credits = P._play_credits(steps, float(s["w_end"]))
            for seat in (0, 1):
                d = ev[h].setdefault(deck_hash(s["decks"][seat]), {"openers": [], "credit": Counter()})
                if s["first_player"] == seat:
                    seq = [
                        st["code"]
                        for st in steps
                        if st["player"] == seat
                        and st["turn"] == 1
                        and st["kind"] in P.PLAY_KINDS
                        and st["code"]
                    ][:3]
                    if seq:
                        d["openers"].append(tuple(seq))
            for i, c in credits.items():
                ev[h][deck_hash(s["decks"][int(steps[i]["player"])])]["credit"][steps[i]["code"]] += c

    decks = sorted(set.intersection(*(set(ev[h]) for h in FOLDS)))
    if len(decks) < MIN_DECKS:
        print(f"ALERT primer_fast_vs_full: {len(decks)} decks in both halves (< {MIN_DECKS})", flush=True)
        return 2

    def top(entry, field, n):
        if not entry:
            return []
        if field == "openers":
            return [tuple(o["plays"]) for o in entry.get("openers", [])[:n]]
        return [c["code"] for c in entry.get(field, [])[:n]]

    sources = ("full", "fast_same", "fast5")
    per_deck: dict[str, dict[str, dict]] = {src: {} for src in sources}
    for k in decks:
        for src in sources:
            oe, of, ks, jk, jo = [], [], [], [], []
            for h in FOLDS:
                e = prim[src][h].get(k)
                ops = top(e, "openers", 3)
                firsts = {o[0] for o in ops}
                for seq in ev[h][k]["openers"]:
                    oe.append(seq in ops)
                    of.append(seq[0] in firsts)
                pos = {c: v for c, v in ev[h][k]["credit"].items() if v > 0}
                tot = sum(pos.values())
                if tot > 0:
                    ks.append(sum(pos.get(c, 0.0) for c in top(e, "key_cards", 5)) / tot)
                ref = prim["full"][h].get(k)
                # agreement with the Full primer of the same fold; for "full" itself, with the next fold's Full
                # primer = the Full-vs-Full noise floor
                ref = prim["full"][(h + 1) % len(FOLDS)].get(k) if src == "full" else ref
                jk.append(_jacc(top(e, "key_cards", 5), top(ref, "key_cards", 5)))
                jo.append(_jacc(top(e, "openers", 3), top(ref, "openers", 3)))
            per_deck[src][k] = {
                "opener_exact": _mean(oe),
                "opener_first": _mean(of),
                "key5_share": _mean(ks),
                "jacc_key5": _mean(jk),
                "jacc_open3": _mean(jo),
                "n_open": len(oe),
            }

    rng = random.Random(0)
    report: dict = {"decks": len(decks), "games_full": len(summ["full"]), "games_fast": len(summ["fast"])}
    for src in sources:
        report[src] = {
            m: round(_mean(per_deck[src][k][m] for k in decks), 3)
            for m in ("opener_exact", "opener_first", "key5_share", "jacc_key5", "jacc_open3")
        }
    for src in ("fast_same", "fast5"):
        for m in ("opener_exact", "opener_first", "key5_share"):
            diffs = [
                per_deck[src][k][m] - per_deck["full"][k][m]
                for k in decks
                if per_deck[src][k][m] is not None and per_deck["full"][k][m] is not None
            ]
            mean, lo, hi = _boot(diffs, rng)
            report[f"{src}-full {m}"] = [round(mean, 3), round(lo, 3), round(hi, 3)]
    worse = [m for m in ("opener_exact", "key5_share") if report[f"fast5-full {m}"][2] < 0]
    report["verdict"] = "fast5 worse on " + ", ".join(worse) if worse else "fast5 not worse (keep Fast)"
    print(json.dumps(report, indent=1), flush=True)
    if all(report[f"fast_same-full {m}"][0] == 0 for m in ("opener_exact", "key5_share")):
        print("ALERT primer_fast_vs_full: fast_same equals full exactly (same games?)", flush=True)
        return 2

    # side by side for the decks with the most going-first games
    names_of = lambda codes: "; ".join(names.get(c, f"#{c}") for c in codes)
    for k in sorted(decks, key=lambda k: -per_deck["full"][k]["n_open"])[: args.show]:
        e = prim["full"][1].get(k) or {}
        print(f"\n== {e.get('archetype', k)} ({', '.join(e.get('files', [])[:1])})")
        for src in ("full", "fast_same", "fast5"):
            pe = prim[src][1].get(k)
            print(f"  {src:9s} key: {names_of(top(pe, 'key_cards', 5))}")
            for o in top(pe, "openers", 3):
                print(f"  {src:9s} opener: {names_of(o)}")
    if args.json:
        args.json.write_text(json.dumps({"report": report, "per_deck": per_deck}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
