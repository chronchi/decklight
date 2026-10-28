"""X2: paired V (fresh list) minus T (training list) for P raw, from scripts/launch/launch-x2.sh's two arms.

The arms play different S decks, so paired_sm's (decks, seed, s_seat) key never matches; with --seed-per-pair each
(seed, s_seat) is one pair index of both pair files (same opponent, same seed), so that is the key here. Prints the
overall and per-seat difference (paired_sm.report), then per archetype (V's -vN suffix dropped): n, T and V win rates,
V - T. Exit 2 when fewer than 90% of T's games pair up or a pair's opponent differs between arms. Engine-free.

    python scripts/audit/x2_paired.py runs/x2-<stamp>/t runs/x2-<stamp>/v
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_sm import report


def games(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("games-*.jsonl")):
        for raw in f.read_text().splitlines():
            if raw.strip():
                g = json.loads(raw)
                s = g["s_seat"]
                score = 0.5 if g.get("truncated") else float(g["winner"] == s)
                out[(g["seed"], s)] = (g["decks"][s], g["decks"][1 - s], score)
    return out


def main() -> int:
    t, v = games(Path(sys.argv[1])), games(Path(sys.argv[2]))
    keys = sorted(set(t) & set(v))
    if len(keys) < 0.9 * len(t):
        print(f"ALERT x2_paired: only {len(keys)} of {len(t)} games pair up")
        return 2
    bad = [k for k in keys if t[k][1] != v[k][1]]
    if bad:
        print(f"ALERT x2_paired: {len(bad)} pairs have different opponents, e.g. {bad[0]}: {t[bad[0]]} vs {v[bad[0]]}")
        return 2
    print(f"A=T {sys.argv[1]} ({sum(x[2] for x in t.values()) / len(t):.3f}, n={len(t)})  "
          f"B=V {sys.argv[2]} ({sum(x[2] for x in v.values()) / len(v):.3f}, n={len(v)})")
    print(report("all", [v[k][2] - t[k][2] for k in keys]))
    for seat, label in ((0, "first"), (1, "second")):
        print(report(label, [v[k][2] - t[k][2] for k in keys if k[1] == seat]))
    by: dict[str, list] = {}
    for k in keys:
        by.setdefault(re.sub(r"-v\d+$", "", v[k][0]), []).append(k)
    print("archetype | n | T | V | V-T pp")
    for arch, ks in sorted(by.items()):
        tw, vw = (sum(x[k][2] for k in ks) / len(ks) for x in (t, v))
        print(f"{arch} | {len(ks)} | {tw:.3f} | {vw:.3f} | {(vw - tw) * 100:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
