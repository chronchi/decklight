"""Hand-trap misuse on turn 1 (going first): share of lines in which player 0 sets or activates a hand trap during its
own turn 1, for the greedy line, the best-by-field sample and the best-by-rollout sample of each hand.

    python scripts/audit/handtrap_use.py <scored.jsonl> [--n 400] [--seed 0] [--write-clean OUT]
Hand traps are recognised by name (HT below; proactive cards such as Bystials are not in it). On turn 1 against a
passive opponent a pure hand trap has no proper use, yet setting one turns a hand card (0.5) into a field card (1.0)
in the play-out score. `--write-clean OUT` writes every row of the input except SAMPLE rows that set/use a hand trap
(greedy rows are kept so selection still compares against greedy): the training-data veto (NOTEBOOK 2026-10-07 23:50).
Laptop-safe: engine replay only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_lines as SL  # noqa: E402

HT = ("Ash Blossom", "Effect Veiler", "Ghost Ogre", "Ghost Belle", "Ghost Mourner", "Maxx \"C\"", "Mulcharmy",
      "Nibiru", "Droll & Lock Bird", "D.D. Crow", "PSY-Framegear Gamma", "Dimension Shifter", "Retaliating \"C\"",
      "Fuwalos")
PAT = re.compile(r"(IDLECMD (MSET|SSET|ACTIVATE)|CHAIN ACTIVATE) (" + "|".join(re.escape(h) for h in HT) + ")")


def misuses(r: dict, engine, nn) -> bool:
    if "actor" not in r:  # old turn-1 rows, passive opponent: any use is a misuse
        return any(PAT.search(line) for line in SL.render(r, engine, nn))
    # whole-turn rows (show_turns.render, actor's own turn, opponent active): setting / activating a hand trap from
    # the idle command is always a waste; chaining one is a misuse only when it answers our own action (previous
    # decision was ours), not the opponent's.
    import show_turns as ST
    lines = [x for x in ST.render(r, engine, nn) if x.startswith("    ME ") or x.startswith("    OPP ")]
    for i, x in enumerate(lines):
        if x.startswith("    ME ") and PAT.search(x):
            if "IDLECMD" in x or (i > 0 and lines[i - 1].startswith("    ME ")):
                return True
    return False


def write_clean(src: Path, out: Path) -> int:
    nn, engine = SL.CardNames().names, SL.Engine()
    kept = dropped = 0
    with open(out, "w") as f:
        for x in src.read_text().splitlines():
            r = json.loads(x)
            if r["kind"] != "greedy" and misuses(r, engine, nn):
                dropped += 1
                continue
            kept += 1
            f.write(x + "\n")
    print(f"handtrap veto: kept {kept}, dropped {dropped} sample rows -> {out}")
    if kept == 0:
        print("0 rows kept", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", type=Path)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-clean", type=Path, default=None)
    a = ap.parse_args()
    if a.write_clean:
        return write_clean(a.scored, a.write_clean)
    g = defaultdict(list)
    for x in a.scored.read_text().splitlines():
        r = json.loads(x)
        g[(r["deck"], r["source"], r["hand"])].append(r)
    keys = sorted(g)
    random.Random(a.seed).shuffle(keys)
    nn, engine = SL.CardNames().names, SL.Engine()
    use, n = defaultdict(int), 0
    for k in keys[: a.n]:
        rs = g[k]
        gr = next((r for r in rs if r["kind"] == "greedy"), None)
        sm = [r for r in rs if r["kind"] != "greedy"]
        if not (gr and sm):
            continue
        n += 1
        picks = {"greedy": gr, "best_field": max(sm, key=lambda r: (r["field"], r["v_p0"])),
                 "best_rollout": max(sm, key=lambda r: r.get("rollout", r["field"]))}
        for name, r in picks.items():
            if misuses(r, engine, nn):
                use[name] += 1
    if n == 0:
        print("no hands", file=sys.stderr)
        return 2
    for name in ("greedy", "best_field", "best_rollout"):
        print(f"{name:<13} hand trap set/used on own turn 1: {use[name] / n:.3f} ({use[name]}/{n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
