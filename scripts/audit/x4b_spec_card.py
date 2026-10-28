"""X4b step 2: the specialist card from scripts/vast_x4b_spec.sh (runs/<name>/card/<variant>/games-*.jsonl).

Per archetype a: S-a minus P-a on its own lists (paired by decks, seed, seat; all, per seat, train lists vs the
held-out list from decks/x4b/split.json) and S-a-unrel minus P-unrel on the unrelated set. Then the pre-registered
stop rule (NOTEBOOK 2026-10-28 "X4b"): stop when fewer than 3 of 5 specialists gain >= +3 pp on their archetype.
Writes <card>/../spec_card.json; prints a table and a STOP_RULE line. Engine-free.
Exit 2: an arm missing, < 90% of games pair up, or a specialist's moves equal P's in > 90% of its games (its
weights did not change what it plays).

    python scripts/audit/x4b_spec_card.py runs/x4b-spec-<stamp>/card <arch> [<arch> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_sm import report

ROOT = Path(__file__).resolve().parents[2]
GAIN_PP, NEED = 3.0, 3


def games(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("games-*.jsonl")):
        for raw in f.read_text().splitlines():
            if raw.strip():
                g = json.loads(raw)
                s = g["s_seat"]
                score = 0.5 if g.get("truncated") else float(g["winner"] == s)
                out[(tuple(g["decks"]), g["seed"], s)] = (score, g["decks"][s], g.get("choices"))
    return out


def paired(a: dict, b: dict, label: str) -> tuple[list, list]:
    if not a or not b:
        raise SystemExit(f"ALERT x4b_spec_card: {label}: an arm has no games")
    keys = sorted(set(a) & set(b))
    if len(keys) < 0.9 * len(a):
        raise SystemExit(f"ALERT x4b_spec_card: {label}: only {len(keys)} of {len(a)} games pair up")
    same = sum(1 for k in keys if a[k][2] is not None and a[k][2] == b[k][2])
    if same > 0.9 * len(keys):
        raise SystemExit(f"ALERT x4b_spec_card: {label}: {same}/{len(keys)} games move-identical to P")
    return keys, [b[k][0] - a[k][0] for k in keys]


def stats(diffs: list[float]) -> dict:
    n = len(diffs)
    if n == 0:  # e.g. a smoke --games-limit that stops before the held-out list
        return {"n": 0, "diff_pp": None, "ci_pp": None}
    m = sum(diffs) / n
    se = (sum((x - m) ** 2 for x in diffs) / (n - 1) / n) ** 0.5 if n > 1 else 0.0
    return {"n": n, "diff_pp": round(m * 100, 2), "ci_pp": [round((m - 1.96 * se) * 100, 2),
                                                           round((m + 1.96 * se) * 100, 2)]}


def main() -> int:
    card, archs = Path(sys.argv[1]), sys.argv[2:]
    split = json.loads((ROOT / "decks/x4b/split.json").read_text())
    p_unrel = games(card / "P-unrel")
    res: dict = {"archetypes": {}}
    for a in archs:
        p, s = games(card / f"P-{a}"), games(card / f"S-{a}")
        keys, d = paired(p, s, a)
        held = split["heldout"][a]
        r = {"P_wr": round(sum(p[k][0] for k in keys) / len(keys), 4),
             "S_wr": round(sum(s[k][0] for k in keys) / len(keys), 4),
             "arch": stats(d),
             "first": stats([x for k, x in zip(keys, d, strict=True) if k[2] == 0]),
             "second": stats([x for k, x in zip(keys, d, strict=True) if k[2] == 1]),
             "train_lists": stats([x for k, x in zip(keys, d, strict=True) if p[k][1] != held]),
             "heldout": stats([x for k, x in zip(keys, d, strict=True) if p[k][1] == held])}
        print(f"{a}: P {r['P_wr']:.3f} S {r['S_wr']:.3f}")
        print("  " + report("arch", d))
        su = games(card / f"S-{a}-unrel")
        _, du = paired(p_unrel, su, f"{a}-unrel")
        r["unrelated"] = stats(du)
        print("  " + report("unrelated", du))
        print("  " + ", ".join(f"{lab} {v['diff_pp']:+.1f} pp (n={v['n']})" if v["n"] else f"{lab} n=0"
                               for lab, v in (("train lists", r["train_lists"]), (f"held-out {held}", r["heldout"]),
                                              ("first", r["first"]), ("second", r["second"]))))
        res["archetypes"][a] = r
    gainers = [a for a, r in res["archetypes"].items() if r["arch"]["diff_pp"] >= GAIN_PP]
    res["gainers"] = gainers
    res["stop"] = len(gainers) < NEED
    (card.parent / "spec_card.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"STOP_RULE {'STOP' if res['stop'] else 'GO'}: {len(gainers)} of {len(archs)} specialists gain "
          f">= +{GAIN_PP:.0f} pp on their archetype ({', '.join(gainers) or 'none'}; need {NEED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
