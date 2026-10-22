"""Relabel multi-score window search records offline (ADR 0007): pick each searched decision's target by another
score from the search log, without re-running the search.

    uv run python scripts/relabel_window.py --src 'runs/<name>/window/p*' --select done@h3 value@h3 --name d3
    uv run python scripts/relabel_window.py --src ... --compare value@h1 board@h1 done@h3,value@h3

--select takes score keys compared lexicographically (done@h3 then value@h3: win within 3 turns first, the learned
value breaks ties), ties kept in candidate order (PASS first at a window, as in the search). Output: one record per
game under data/demos/sw-<name>/ (files p<k>-<source name>; kind sw-<name> in dataset.py); targets, target_sets
(the single best) and target_conf (share of the separating score keys whose top candidates include the target, as in
src/decklight/search/window.py) are replaced at searched decisions. --compare prints, for each pair of labellings,
how often they pick the same option, and each one's fire rate at windows. Pure JSON: safe on the laptop.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pick(entry: dict, keys: list[str]) -> int:
    """Index into entry["cands"] of the best candidate by `keys` (lexicographic, ties → earlier candidate)."""
    cols = [entry["scores_all"][k] for k in keys]
    n = len(entry["cands"])
    return max(range(n), key=lambda j: (*(c[j] for c in cols), -j))


def confidence(entry: dict, best: int) -> float:
    sa = entry["scores_all"]
    keys = [k for k in sa if not k.startswith("cards@") and max(sa[k]) > min(sa[k])]
    if not keys:
        return 1.0
    return sum(1 for k in keys if sa[k][best] == max(sa[k])) / len(keys)


def fires(entry: dict, j: int) -> bool:
    """A window target fires when it is not the first candidate (PASS is ranked first at a window)."""
    return entry["kind"] == "window" and j != 0


def load(src: list[str]) -> list[tuple[str, dict]]:
    out = []
    for d in sorted({p for pat in src for p in glob.glob(pat)}):
        dp = Path(d)
        for f in sorted(dp.glob("*.jsonl")) if dp.is_dir() else [dp]:
            for line in f.read_text().splitlines():
                if line.strip():
                    out.append((f"{dp.name}-{f.name}" if dp.is_dir() else f.name, json.loads(line)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="globs of gen_window output dirs or files")
    ap.add_argument("--select", nargs="+", help="score keys, lexicographic")
    ap.add_argument("--name", help="output dir data/demos/sw-<name>")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/demos")
    ap.add_argument("--compare", nargs="+", help="labellings to compare: comma-joined key lists")
    args = ap.parse_args()
    recs = load(args.src)
    entries = [e for _, r in recs for e in r.get("search", [])]
    if not entries:
        sys.exit(f"ALERT relabel_window: no search entries in {args.src}")
    have = set(entries[0].get("scores_all", {}))
    if args.compare:
        labs = [c.split(",") for c in args.compare]
        for keys in labs:
            missing = [k for k in keys if k not in have]
            if missing:
                sys.exit(f"ALERT relabel_window: score keys {missing} not in the log (have {sorted(have)})")
        picks = [[pick(e, keys) for e in entries] for keys in labs]
        win = [i for i, e in enumerate(entries) if e["kind"] == "window"]
        print(f"{len(recs)} games, {len(entries)} searched decisions ({len(win)} windows)")
        for a, keys in enumerate(labs):
            fr = sum(fires(entries[i], picks[a][i]) for i in win) / max(len(win), 1)
            played = sum(entries[i]["cands"][picks[a][i]] == entries[i]["played"] for i in range(len(entries)))
            print(f"{','.join(keys):24s} fire rate {fr:.3f}  agrees with the played move {played / len(entries):.3f}")
        for a in range(len(labs)):
            for b in range(a + 1, len(labs)):
                same = sum(x == y for x, y in zip(picks[a], picks[b], strict=True)) / len(entries)
                print(f"  {','.join(labs[a])} vs {','.join(labs[b])}: same target {same:.3f}")
    if args.select:
        if not args.name:
            sys.exit("--select needs --name")
        missing = [k for k in args.select if k not in have]
        if missing:
            sys.exit(f"ALERT relabel_window: score keys {missing} not in the log (have {sorted(have)})")
        out = args.out_root / f"sw-{args.name}"
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.jsonl"):
            f.unlink()
        n_t = fired = n_w = 0
        for fname, r in recs:
            r = dict(r)
            targets, sets, conf = list(r["targets"]), list(r["target_sets"]), list(r["target_conf"])
            for e in r.get("search", []):
                j = pick(e, args.select)
                k = e["k"]
                targets[k], sets[k], conf[k] = e["cands"][j], [e["cands"][j]], round(confidence(e, j), 4)
                n_t += 1
                n_w += e["kind"] == "window"
                fired += fires(e, j)
            r["targets"], r["target_sets"], r["target_conf"] = targets, sets, conf
            r["source"] = f"{r.get('source', '')}; relabelled select={','.join(args.select)}"
            with (out / fname).open("a") as fh:
                fh.write(json.dumps(r) + "\n")
        if n_t == 0:
            sys.exit(f"ALERT relabel_window: 0 targets written to {out}")
        print(f"sw-{args.name}: {len(recs)} games, {n_t} targets, fire rate {fired / max(n_w, 1):.3f} -> {out}")


if __name__ == "__main__":
    main()
