"""Where did checkpoint B change the play of checkpoint A? Pair two search_match arms deal by
deal, count outcome flips, and find each flip's first divergent decision.

    uv run python scripts/audit/flip_audit.py --card <card dir> --a base --b final --out <dir>
        [--sample 30] [--seed 0]

`<card dir>` holds one subdir per arm with games-*.jsonl (scripts/deck_card.py layout). Arms
pair by (decks, seed, s_seat). Writes <out>/flips.json (flip counts, and per flip the first
divergent decision: turn, decision type, A's pick vs B's pick, rendered) plus a seeded sample
of flips (both directions) rendered side by side into <out>/pairs/NNN.txt. Exit 2 when fewer
than 90% of A's games pair up (as in paired_sm).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sm_common as SM


def pair_arms(a_recs: list[dict], b_recs: list[dict]) -> tuple[dict, list, list]:
    """Pair records by deal key. Returns (pairs, a_only_keys, b_only_keys).

    pairs: {key: (a_rec, b_rec)}; key = (decks, seed, s_seat).
    """
    da = {SM.deal_key(r): r for r in a_recs}
    db = {SM.deal_key(r): r for r in b_recs}
    pairs = {k: (da[k], db[k]) for k in da.keys() & db.keys()}
    return pairs, sorted(set(da) - set(db)), sorted(set(db) - set(da))


def flip_dir(a_cls: str, b_cls: str) -> str:
    """Outcome-change direction from A's result class to B's (e.g. 'L->W')."""
    return f"{a_cls}->{b_cls}"


def flip_counts(pairs: dict) -> tuple[dict[str, int], dict[str, list]]:
    """({direction: n}, {direction: [keys]}) over paired deals."""
    counts: dict[str, int] = {}
    keys: dict[str, list] = {}
    for k, (a, b) in pairs.items():
        d = flip_dir(SM.result_class(a), SM.result_class(b))
        counts[d] = counts.get(d, 0) + 1
        keys.setdefault(d, []).append(k)
    return counts, keys


def deal_str(key: tuple) -> str:
    decks, seed, seat = key
    return f"{decks[0]}-vs-{decks[1]} seed={seed} seat=P{seat}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", type=Path, required=True, help="card dir holding one subdir per arm")
    ap.add_argument("--a", default="base", help="reference arm")
    ap.add_argument("--b", default="final", help="changed arm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=30, help="max rendered pairs per flip direction")
    ap.add_argument("--seed", type=int, default=0, help="seed for the flip sample")
    ap.add_argument("--gauntlet", type=Path, default=SM.GAUNTLET)
    args = ap.parse_args(argv)

    a_recs = SM.iter_records(args.card / args.a)
    b_recs = SM.iter_records(args.card / args.b)
    if not a_recs or not b_recs:
        print(f"ALERT flip_audit: 0 games in {args.card / args.a} or {args.card / args.b}", flush=True)
        return 1
    pairs, a_only, b_only = pair_arms(a_recs, b_recs)
    if len(pairs) < 0.9 * len(a_recs):
        print(f"ALERT flip_audit: only {len(pairs)} of {len(a_recs)} {args.a} games pair up", flush=True)
        return 2
    counts, keys = flip_counts(pairs)
    n = len(pairs)
    print(f"{args.a} -> {args.b}: {n} paired deals")
    for d in sorted(counts):
        print(f"  {d}: {counts[d]} ({counts[d] / n:.1%})")

    engine, names = SM.Engine(), SM.CardNames()
    flips = []
    bad = 0
    for d in sorted(keys):
        for key in sorted(keys[d]):
            ra, rb = pairs[key]
            try:
                ta = SM.replay_trace(ra, engine, names, args.gauntlet)
                tb = SM.replay_trace(rb, engine, names, args.gauntlet)
            except Exception as e:  # noqa: BLE001 - report, keep going, fail at the end
                print(f"ALERT flip_audit: replay of {deal_str(key)} raised {e!r}", flush=True)
                bad += 1
                continue
            if not ta["ok"] or not tb["ok"]:
                print(f"ALERT flip_audit: replay of {deal_str(key)} does not reproduce "
                      f"a={ra.get('winner')}/b={rb.get('winner')}", flush=True)
                bad += 1
            div = SM.first_divergence(ta, tb)
            flips.append({"deal": deal_str(key), "key": [list(key[0]), key[1], key[2]],
                          "game_idx": ra.get("game_idx"), "dir": d,
                          "a_class": SM.result_class(ra), "b_class": SM.result_class(rb),
                          "divergence": div})
    flips.sort(key=lambda f: (f["dir"], f["deal"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "flips.json").write_text(json.dumps(
        {"a": args.a, "b": args.b, "n_pairs": n, "a_only": len(a_only), "b_only": len(b_only),
         "counts": counts, "flips": flips}, indent=1))
    print(f"wrote {args.out / 'flips.json'} ({len(flips)} flips detailed)")

    rng = random.Random(args.seed)
    pdir = args.out / "pairs"
    pdir.mkdir(parents=True, exist_ok=True)
    n_pair = 0
    for d in ("L->W", "W->L"):
        cands = [f for f in flips if f["dir"] == d]
        rng.shuffle(cands)
        for i, f in enumerate(cands[: args.sample]):
            key = (tuple(f["key"][0]), f["key"][1], f["key"][2])
            ra, rb = pairs[key]
            ta_text, _ = SM.render_record(ra, engine, names, args.gauntlet)
            tb_text, _ = SM.render_record(rb, engine, names, args.gauntlet)
            div = f["divergence"] or {}
            head = (f"dir {d} | {f['deal']} | game {f['game_idx']} | "
                    f"first divergence at decision {div.get('idx')} "
                    f"(T{div.get('turn')} P{div.get('actor')} {div.get('kind_name')}):\n"
                    f"  A: {div.get('a_pick')}\n  B: {div.get('b_pick')}\n")
            (pdir / f"{d.replace('->', '-to-')}-{i:03d}.txt").write_text(
                head + f"\n===== A ({args.a}) =====\n" + ta_text + f"\n===== B ({args.b}) =====\n" + tb_text)
            n_pair += 1
    print(f"wrote {n_pair} side-by-side pairs -> {pdir}/")
    if bad:
        print(f"ALERT flip_audit: {bad} replays did not reproduce the recorded winner", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
