"""Code-only behaviour features over search_match arms (no LLM): what changed, in numbers.

    uv run python scripts/audit/play_features.py --card <card dir> --arms base,it0040,...,final
        [--out features.json] [--workers 4]

Replays every game of every arm and reports, per arm:
- mix of OUR decision types (summon / activate / set / attack / chain-respond / pass / other);
- cards we activated or summoned on duel turn 1 / turn 2 (top-20 by max |share - base|);
- mean actions in our first turn to act;
- distinct first-turn line signatures per games (diversity) and the most common line's share.
JSON + table. Exit non-zero when an arm has 0 games or a replay does not reproduce its winner.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sm_common as SM

CHAIN_MSGS = {"SELECT_CHAIN", "SELECT_EFFECTYN", "SELECT_YESNO"}
PASS_KINDS = {"PASS", "NO", "CANCEL", "FINISH", "TO_END", "TO_BATTLE", "TO_MAIN2"}
SUMMON_KINDS = {"SUMMON", "SPSUMMON"}
SET_KINDS = {"MSET", "SSET"}

_W: dict = {}


def bucket(d: dict) -> str:
    """Exclusive decision-type bucket for one of our decisions."""
    if d["msg"] in CHAIN_MSGS:
        return "chain"
    if d["kind_name"] in SUMMON_KINDS:
        return "summon"
    if d["kind_name"] == "ACTIVATE":
        return "activate"
    if d["kind_name"] in SET_KINDS:
        return "set"
    if d["kind_name"] == "ATTACK":
        return "attack"
    if d["kind_name"] in PASS_KINDS:
        return "pass"
    return "other"


def _init(gauntlet: str) -> None:
    _W["engine"], _W["names"], _W["gauntlet"] = SM.Engine(), SM.CardNames(), Path(gauntlet)


def _feat(rec: dict) -> dict:
    t = SM.replay_trace(rec, _W["engine"], _W["names"], _W["gauntlet"])
    if not t["ok"]:
        return {"ok": False, "game_idx": rec.get("game_idx")}
    s = rec["s_seat"]
    ours = [d for d in t["decisions"] if d["actor"] == s]
    mix = Counter(bucket(d) for d in ours)
    t1 = sorted({d["name"] for d in ours if d["turn"] == 1 and d["kind_name"] in SUMMON_KINDS | {"ACTIVATE"}
                 and d["code"]})
    t2 = sorted({d["name"] for d in ours if d["turn"] == 2 and d["kind_name"] in SUMMON_KINDS | {"ACTIVATE"}
                 and d["code"]})
    first_turn = min((d["turn"] for d in ours), default=None)
    sig = tuple((d["kind_name"], d["name"]) for d in ours if d["turn"] == first_turn) if first_turn else ()
    return {"ok": True, "mix": dict(mix), "n_ours": len(ours), "t1": t1, "t2": t2,
            "first_n": len(sig), "sig": sig, "score": SM.our_score(rec)}


def summarize(feats: list[dict]) -> dict:
    """Aggregate per-game features into arm-level stats (pure; tested)."""
    ok = [f for f in feats if f["ok"]]
    n = len(ok)
    mix = Counter()
    for f in ok:
        mix.update(f["mix"])
    n_ours = sum(f["n_ours"] for f in ok)
    t1 = Counter(c for f in ok for c in f["t1"])
    t2 = Counter(c for f in ok for c in f["t2"])
    sigs = Counter(f["sig"] for f in ok)
    top, top_n = sigs.most_common(1)[0] if sigs else ((), 0)
    return {
        "n": n,
        "win_rate": round(sum(f["score"] for f in ok) / n, 3) if n else None,
        "mix": {k: {"n": mix.get(k, 0), "share": round(mix.get(k, 0) / n_ours, 3)} for k in
                ["summon", "activate", "set", "attack", "chain", "pass", "other"]} if n_ours else {},
        "mean_first_n": round(sum(f["first_n"] for f in ok) / n, 2) if n else None,
        "distinct_lines": len(sigs), "diversity": round(len(sigs) / n, 3) if n else None,
        "top_line": {"share": round(top_n / n, 3) if n else None, "n": top_n,
                     "sig": [[k, nm] for k, nm in top][:12]},
        "cards_turn1": {c: round(t1[c] / n, 3) for c in t1},
        "cards_turn2": {c: round(t2[c] / n, 3) for c in t2},
    }


def top_changes(stats: dict, base: str, turn: str, k: int = 20) -> list[dict]:
    """Cards ranked by max |share - base share| across non-base arms (pure; tested)."""
    cards = set()
    for s in stats.values():
        cards |= set(s[f"cards_{turn}"].keys())
    rows = []
    for c in cards:
        b = stats[base][f"cards_{turn}"].get(c, 0.0)
        shares = {a: stats[a][f"cards_{turn}"].get(c, 0.0) for a in stats}
        delta = max(abs(v - b) for a, v in shares.items() if a != base) if len(stats) > 1 else 0.0
        rows.append({"card": c, "base": b, "shares": shares, "max_delta": round(delta, 3)})
    rows.sort(key=lambda r: -r["max_delta"])
    return rows[:k]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", type=Path, required=True)
    ap.add_argument("--arms", required=True, help="comma-separated arm names, first = base")
    ap.add_argument("--out", type=Path, default=None, help="features.json to write (default: <card>/features.json)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gauntlet", type=Path, default=SM.GAUNTLET)
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    recs = {}
    for a in arms:
        recs[a] = SM.iter_records(args.card / a)
        if not recs[a]:
            print(f"ALERT play_features: 0 games in {args.card / a}", flush=True)
            return 1
    stats = {}
    for a in arms:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(args.gauntlet),)) as ex:
            feats = list(ex.map(_feat, recs[a], chunksize=16))
        bad = sum(1 for f in feats if not f["ok"])
        if bad:
            print(f"ALERT play_features: {bad}/{len(feats)} replays in {a} do not reproduce the winner",
                  flush=True)
            return 2
        stats[a] = summarize(feats)
        s = stats[a]
        print(f"{a}: n={s['n']} win={s['win_rate']} first_n={s['mean_first_n']} "
              f"distinct={s['distinct_lines']} divers={s['diversity']} top={s['top_line']['share']}", flush=True)
    out = {"arms": arms, "stats": stats,
           "top_turn1": top_changes(stats, arms[0], "turn1"),
           "top_turn2": top_changes(stats, arms[0], "turn2")}
    out_path = args.out or args.card / "features.json"
    out_path.write_text(json.dumps(out, indent=1))

    hdr = f"{'arm':10s}{'n':>5s}{'win':>6s}" + "".join(f"{b:>9s}" for b in
          ["summon", "activate", "set", "attack", "chain", "pass", "other"]) + \
        f"{'1st_n':>7s}{'divers':>8s}{'top':>6s}"
    print(hdr)
    for a in arms:
        s = stats[a]
        row = f"{a:10s}{s['n']:>5d}{s['win_rate']:>6.2f}"
        row += "".join(f"{s['mix'].get(b, {}).get('share', 0):>9.2f}" for b in
                       ["summon", "activate", "set", "attack", "chain", "pass", "other"])
        print(row + f"{s['mean_first_n']:>7.1f}{s['diversity']:>8.2f}{s['top_line']['share']:>6.2f}")
    for turn, key in ((1, "top_turn1"), (2, "top_turn2")):
        print(f"--- turn {turn}: top cards by |share - {arms[0]}| ---")
        print(f"{'card':44s}{'base':>6s}" + "".join(f"{a:>8s}" for a in arms[1:]))
        for r in out[key]:
            print(f"{r['card'][:44]:44s}{r['base']:>6.2f}" +
                  "".join(f"{r['shares'][a]:>8.2f}" for a in arms[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
