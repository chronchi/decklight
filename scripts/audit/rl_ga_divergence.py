"""For flipped deals (base won / tag lost and the reverse), print the first differing decision in each game: what
the base chose vs what the tag chose (render_games decision lines), plus a tally by decision kind.

    uv run python scripts/audit/rl_ga_divergence.py runs/<rl5ga run> --base it0 --tag it200 [--show 40]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/audit"))

from render_games import decisions_lines
from rl_ga_compare import divergence, fields, load

from ygo_zero.demos.names import CardNames
from ygo_zero.engine.duel import Engine


def line_at(lines: list[str], idx: int) -> str:
    for ln in lines:
        if ln.startswith(f"[{idx}] "):
            return ln
    return f"[{idx}] (forced)"


def kind_of(ln: str) -> str:
    m = re.match(r"\[\d+\] T\d+ P\d+ (\S+?)(?::\S+)? (\S+)", ln)
    return f"{m.group(1)} {m.group(2)}" if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--base", default="it0")
    ap.add_argument("--tag", default="it200")
    ap.add_argument("--show", type=int, default=30)
    args = ap.parse_args()
    names, engine = CardNames(), Engine()
    base, other = load(args.run, args.base), load(args.run, args.tag)
    for code in ("WL", "LW"):
        ks = [k for k in sorted(set(base) & set(other))
              if (base[k]["a_win"], other[k]["a_win"]) == ((1, 0) if code == "WL" else (0, 1))]
        tally = Counter()
        print(f"\n===== {code} ({len(ks)}): {args.base} {'won' if code == 'WL' else 'lost'}, {args.tag} "
              f"{'lost' if code == 'WL' else 'won'}")
        for n, k in enumerate(ks):
            sa, sb = (dict(g["state"]) if g["path"] is None else json.loads(g["path"].read_text())
                      for g in (base[k], other[k]))
            for st in (sa, sb):
                if "actors" not in st:
                    fields(engine, st)  # fills actors/turns (train_rl eval format)
            d = divergence(sa, sb)
            if d is None:
                continue
            i = d[0]
            la, lb = line_at(decisions_lines(sa, names, engine), i), line_at(decisions_lines(sb, names, engine), i)
            ka, kb = kind_of(la), kind_of(lb)
            tally[(ka.split()[0], ka.split()[-1] if ka != "?" else "?", kb.split()[-1] if kb != "?" else "?")] += 1
            if n < args.show:
                print(f"- {k[0].split('.')[0]} vs {k[1].split('.')[0]} s{k[2]} {'1st' if k[3] else '2nd'}:")
                print(f"    {args.base}: {la[:170]}")
                print(f"    {args.tag}: {lb[:170]}")
        print(f"  tally (prompt, {args.base} kind, {args.tag} kind):")
        for t, c in tally.most_common(15):
            print(f"    {c:3d} {t}")


if __name__ == "__main__":
    main()
