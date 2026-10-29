"""X4b vs ygo-agent: our raw policies pilot the X4b lists, ygo-agent pilots its gauntlet lists (scripts/launch/
launch-x4b-ga.sh; runs/<name>/<variant>/games-*.jsonl). User 2026-10-29: "how well the new D with decklists seen or
related in the training and unrelated archetypes play against ygo-agent ... raw policies".

Per set: P and D win rates, D - P paired (all, per seat); per archetype also the specialist S - P and D - P, and
D's share of S's gain. Also each list's share of main+extra cards that ygo-agent does not know (its vintage
vocabulary maps them to one "unknown" row), since unknown cards can flatter our absolute win rate (not the paired
diffs). Writes <run>/ga_card.json. Exit 2: an arm missing or < 90% of games pair up. Engine-free.

    python scripts/audit/x4b_ga_card.py runs/x4b-ga-<stamp> <arch> [<arch> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_sm import report
from x4b_spec_card import games, paired, stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def unknown_share() -> dict[str, float]:
    from decklight.decks import load_ydk
    from decklight.eval.ygoagent import vintage_code_ids

    known = set(vintage_code_ids())
    out = {}
    for p in sorted((ROOT / "decks/x4b").rglob("*.ydk")):
        d = load_ydk(p)
        cards = list(d.main) + list(d.extra)
        out[p.stem] = round(sum(c not in known for c in cards) / max(len(cards), 1), 3)
    return out


def arm(run: Path, a: str, b: str, label: str) -> tuple[dict, list]:
    pa, pb = games(run / a), games(run / b)
    keys, d = paired(pa, pb, label)
    r = {"A": a, "B": b, "A_wr": round(sum(pa[k][0] for k in keys) / len(keys), 4),
         "B_wr": round(sum(pb[k][0] for k in keys) / len(keys), 4), "diff": stats(d),
         "first": stats([x for k, x in zip(keys, d, strict=True) if k[2] == 0]),
         "second": stats([x for k, x in zip(keys, d, strict=True) if k[2] == 1])}
    print(f"{label}: {a} {r['A_wr']:.3f} {b} {r['B_wr']:.3f}  " + report(f"{b}-{a}", d)
          + f"; first {r['first']['diff_pp']}, second {r['second']['diff_pp']}")
    return r, keys


def main() -> int:
    run, archs = Path(sys.argv[1]), sys.argv[2:]
    split = json.loads((ROOT / "decks/x4b/split.json").read_text())
    unk = unknown_share()
    res: dict = {"sets": {}, "archetypes": {}, "unknown_share": unk}
    for s, lists in (("related", split["related"]),
                     ("unrelated", split["unrelated_fresh"] + split["unrelated_holdout"])):
        res["sets"][s], _ = arm(run, f"P-ga-{s}", f"D-ga-{s}", s)
        res["sets"][s]["unknown_share_mean"] = round(sum(unk[x] for x in lists) / len(lists), 3)
    pooled_p, pooled_d = {}, {}
    for a in archs:
        dp, _ = arm(run, f"P-ga-{a}", f"D-ga-{a}", f"{a} D")
        sp, _ = arm(run, f"P-ga-{a}", f"S-ga-{a}", f"{a} S")
        g, dg = sp["diff"]["diff_pp"], dp["diff"]["diff_pp"]
        kept = round(dg / g, 3) if g and g > 0 else None
        lists = [*split["train"][a], split["heldout"][a]]
        res["archetypes"][a] = {"D": dp, "S": sp, "kept": kept,
                                "unknown_share_mean": round(sum(unk[x] for x in lists) / len(lists), 3)}
        print(f"  {a}: S-P {g:+.2f}, D-P {dg:+.2f}, kept {kept}, unknown cards {res['archetypes'][a]['unknown_share_mean']}")
        pooled_p.update(games(run / f"P-ga-{a}"))
        pooled_d.update(games(run / f"D-ga-{a}"))
    keys, d = paired(pooled_p, pooled_d, "archetypes pooled")
    res["sets"]["archetypes"] = {"A_wr": round(sum(pooled_p[k][0] for k in keys) / len(keys), 4),
                                 "B_wr": round(sum(pooled_d[k][0] for k in keys) / len(keys), 4), "diff": stats(d)}
    print(f"archetypes pooled: P {res['sets']['archetypes']['A_wr']:.3f} D {res['sets']['archetypes']['B_wr']:.3f}  "
          + report("D-P", d))
    (run / "ga_card.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
