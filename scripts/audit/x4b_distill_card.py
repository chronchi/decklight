"""X4b step 6: the student card from scripts/vast_x4b_distill.sh (runs/<name>/card/<variant>/games-*.jsonl).

D minus P, paired on the same deals (decks, seed, seat): unrelated, related, each archetype, the ygo-agent gauntlet;
all and per seat. Pre-registered gates (NOTEBOOK 2026-10-28 "X4b"):
  H1 transfer:      unrelated D - P >= +2 pp and its 95% CI lower bound > 0, and D - C > 0 (C-unrel arm);
  non-inferiority:  unrelated CI lower bound > -1.5 pp;
  H2 retention:     D's archetype gain >= 50% of the specialist's (--spec runs/<spec>/spec_card.json) on >= 3 of 5.
Writes <card>/../distill_card.json; prints a table and GATE lines. Exit 2 when an arm is missing, < 90% of games
pair up, or D plays P's moves in > 90% of games (nothing distilled). Engine-free.

    python scripts/audit/x4b_distill_card.py runs/x4b-distill-<stamp>/card <arch> ... [--spec spec_card.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_sm import report
from x4b_spec_card import games, paired, stats


def main() -> int:
    argv = sys.argv[1:]
    spec_path = None
    if "--spec" in argv:
        i = argv.index("--spec")
        spec_path = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    card, archs = Path(argv[0]), argv[1:]
    spec = json.loads(spec_path.read_text()) if spec_path else None
    res: dict = {"sets": {}}
    for s in ["unrel", "rel", *archs, "ga"]:
        p, d = games(card / f"P-{s}"), games(card / f"D-{s}")
        keys, diffs = paired(p, d, s)
        r = {"P_wr": round(sum(p[k][0] for k in keys) / len(keys), 4),
             "D_wr": round(sum(d[k][0] for k in keys) / len(keys), 4), "all": stats(diffs),
             "first": stats([x for k, x in zip(keys, diffs, strict=True) if k[2] == 0]),
             "second": stats([x for k, x in zip(keys, diffs, strict=True) if k[2] == 1])}
        res["sets"][s] = r
        print(f"{s}: P {r['P_wr']:.3f} D {r['D_wr']:.3f}  " + report("D-P", diffs)
              + f"; first {r['first']['diff_pp']}, second {r['second']['diff_pp']}")
    for s in ("unrel", "rel"):
        if (card / f"C-{s}").is_dir():
            c, d = games(card / f"C-{s}"), games(card / f"D-{s}")
            keys, diffs = paired(c, d, f"{s} D vs C")
            res["sets"][s]["D_minus_C"] = stats(diffs)
            print(f"{s}: C {sum(c[k][0] for k in keys) / len(keys):.3f}  " + report("D-C", diffs))
    u = res["sets"]["unrel"]["all"]
    dc = res["sets"]["unrel"].get("D_minus_C")
    h1 = u["diff_pp"] >= 2.0 and u["ci_pp"][0] > 0 and dc is not None and dc["diff_pp"] > 0
    ni = u["ci_pp"][0] > -1.5
    res["H1"], res["noninferior"] = h1, ni
    print(f"GATE H1 {'PASS' if h1 else 'FAIL'}: unrelated D-P {u['diff_pp']:+.2f} pp {u['ci_pp']} (need >= +2, lower > 0), "
          f"D-C {dc['diff_pp'] if dc else 'missing'} (need > 0)")
    print(f"GATE non-inferiority {'PASS' if ni else 'FAIL'}: unrelated lower bound {u['ci_pp'][0]:+.2f} (need > -1.5)")
    if spec:
        kept = {}
        for a in archs:
            g = spec["archetypes"][a]["arch"]["diff_pp"]
            dg = res["sets"][a]["all"]["diff_pp"]
            kept[a] = None if g is None or g <= 0 else round(dg / g, 3)
            print(f"  {a}: specialist {g:+.2f} pp, D {dg:+.2f} pp, kept {kept[a]}")
        n_ok = sum(1 for v in kept.values() if v is not None and v >= 0.5)
        res["H2"], res["kept"] = n_ok >= 3, kept
        print(f"GATE H2 {'PASS' if n_ok >= 3 else 'FAIL'}: {n_ok} of {len(archs)} archetypes keep >= 50% of the "
              "specialist's gain (need 3)")
    (card.parent / "distill_card.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
