"""Verify extracted combo lines in the engine (ADR 0002 stage 3).

Usage:
    uv run python scripts/verify_lines.py --lines data/demos/lines/blue-eyes.jsonl \\
        --ydk tests/fixtures/decks/BlueEyes.ydk [--seeds 0 1 2] --out data/demos/verified/blue-eyes.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ygo_zero.decks import Deck, load_ydk
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict, make_config, verify_line
from ygo_zero.engine import constants as C
from ygo_zero.engine.duel import Engine

_EXTRA = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK


def augmented(engine: Engine, deck: Deck, line: ComboLine) -> Deck:
    """Fixture deck + the line's hand/required codes (extra-deck types go to the extra deck)."""
    main, extra = list(deck.main), list(deck.extra)
    for c in line.hand_codes + line.requires_codes:
        if c is None:
            continue
        cd = engine.cards.get(c)
        if cd is not None and cd.type & _EXTRA:
            extra.append(c)
        else:
            main.append(c)
    return Deck(tuple(main), tuple(extra), deck.side, deck.name + "+line")


def stats(v: Verdict) -> str:
    mine = [x for x in v.labels if x != "auto"]
    inf = sum(x == "inferred" for x in mine)
    pct = 100.0 * inf / len(mine) if mine else 0.0
    return f"{pct:.0f}% inferred ({inf}/{len(mine)})"


def verify_all(
    engine: Engine, deck: Deck, lines: list[ComboLine], seeds: list[int], budget: int, tag: str
) -> tuple[list[dict], dict[str, Verdict]]:
    """Verify every line under every seed; print per-line results; return ok records + reps."""
    records, reps = [], {}
    for line in lines:
        verdicts = [(s, verify_line(engine, line, deck, seed=s, budget=budget)) for s in seeds]
        ok = [(s, v) for s, v in verdicts if v.ok]
        rep = ok[0][1] if ok else verdicts[0][1]
        reps[line.id] = rep
        print(
            f"{line.id}{tag}: {rep.reason} steps={len(line.steps)} nodes={rep.nodes} "
            f"{stats(rep)} seeds_ok={[s for s, _ in ok]}"
        )
        print(f"    {rep.detail}")
        if ok:
            s0, v0 = ok[0]
            cfg = make_config(line, deck, s0)
            records.append(
                {
                    "line_id": line.id,
                    "source_url": line.source_url,
                    "deck": line.deck + tag,
                    "config": {
                        "deck0_main": list(cfg.deck0.main),
                        "deck0_extra": list(cfg.deck0.extra),
                        "deck1_main": list(cfg.deck1.main),
                        "deck1_extra": list(cfg.deck1.extra),
                        "seed": s0,
                        "hand": cfg.hand,
                        "draw": cfg.draw,
                        "setup": [[p.team, p.code, p.location, p.sequence] for p in cfg.setup],
                    },
                    "auto_single": True,
                    "choices": v0.choices,
                    "labels": v0.labels,
                    "end_board": line.end_board,
                    "end_board_codes": line.end_board_codes,
                    "seeds_ok": [s for s, _ in ok],
                }
            )
    return records, reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--ydk", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=2000)
    args = ap.parse_args()

    engine = Engine()
    deck = load_ydk(args.ydk)
    lines = [
        ComboLine.from_json(json.loads(s)) for s in Path(args.lines).read_text().splitlines() if s.strip()
    ]

    print(f"== {args.lines} vs {args.ydk} (seeds={args.seeds}) ==")
    records, reps = verify_all(engine, deck, lines, args.seeds, args.budget, "")

    verified = {r["line_id"] for r in records}
    aug_lines = [line for line in lines if line.id not in verified and reps[line.id].reason == "missing_card"]
    aug_records: list[dict] = []
    if aug_lines:
        print(f"== augmented deck ({len(aug_lines)} lines with missing_card) ==")
        adecks = {line.id: augmented(engine, deck, line) for line in aug_lines}
        for line in aug_lines:
            recs, areps = verify_all(engine, adecks[line.id], [line], args.seeds, args.budget, "+augmented")
            aug_records += recs
            reps[line.id + "+augmented"] = areps[line.id]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("".join(json.dumps(r) + "\n" for r in records + aug_records))
    print(f"wrote {len(records) + len(aug_records)} verified records to {outp}")

    totals = Counter()
    ok_aug = {r["line_id"] for r in aug_records}
    for line in lines:
        if line.id in verified:
            totals["ok"] += 1
        elif line.id in ok_aug:
            totals["ok[augmented]"] += 1
        else:
            totals[reps[line.id].reason] += 1
    print("totals:", dict(totals))


if __name__ == "__main__":
    main()
