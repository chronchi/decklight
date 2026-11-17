#!/usr/bin/env python3
"""Does our side interrupt the opponent's turn-2 line when it went first? Read from the tsoj client logs.

    python3 scripts/tsoj/turn2_interrupts.py LABEL=LOGDIR[@FROM,TO] ... [--out turn2.json]

LOGDIR holds the WindBot client logs (`LIVE mm:ss ...` lines: TURN n (me|opp), `opp chains X`, `me chains X`,
`pick <kind> [i/n]`). FROM,TO (local mtime, `YYYY-mm-ddTHH:MM:SS`) select one stage's logs when several stages
share the directory. Only games where we took turn 1 and turn 2 started are counted. Per arm:
- window: on turn 2 we were offered at least one chain choice with an activation in it (`pick activate ...`, or
  `pick pass [i/n]` with n >= 2);
- interrupt: we picked `activate` at least once on turn 2; `opp_actions_before`: the opponent's chains + summons
  logged on turn 2 before our first activation (smaller = earlier in their line);
- died_t2: the game ended on turn 2 with the opponent winning (their line was not stopped);
- window_taken: activations / chain windows on turn 2; first_window_taken: we activated at our first turn-2 window;
- win: by whether we interrupted. Unpaired (log names collide, so a few games per stage are missing).
Exit 3 when an arm has no first-player games.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PICK = re.compile(r"pick (\w+)\s*(.*?)\s*\[(\d+)/(\d+)\]\s*$")


def parse(path: Path) -> dict | None:
    turn, me_first, t2, t2_end, winner = 0, None, [], None, None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(" ", 2)[-1].strip() if raw.startswith("LIVE") else raw.strip()
        m = re.match(r"TURN (\d+) \((me|opp)\)", line)
        if m:
            turn = int(m.group(1))
            if turn == 1:
                me_first = m.group(2) == "me"
            continue
        if line.startswith("WIN "):
            winner = winner or line.split()[1]
            if turn == 2:
                t2_end = True
            continue
        if turn == 2:
            t2.append(line)
    if not me_first or not t2:
        return None
    window, acts, before, opp_n, picks = False, [], None, 0, []  # picks: 1 activate / 0 pass, per chain window
    for line in t2:
        if line.startswith(("opp chains", "opp summons")):
            opp_n += 1
        m = PICK.search(line)
        if not m:
            continue
        kind, card, n = m.group(1), m.group(2), int(m.group(4))
        if kind == "activate":
            window = True
            picks.append(1)
            acts.append(card)
            if before is None:
                before = opp_n
        elif kind == "pass" and n >= 2:
            window = True
            picks.append(0)
    return {"window": window, "acts": acts, "before": before, "died_t2": bool(t2_end and winner == "opp"),
            "win": winner == "me", "done": winner is not None, "picks": picks}


def rate(k: int, n: int) -> dict:
    if not n:
        return {"k": k, "n": 0, "p": None, "se": None}  # strict JSON: no NaN
    p = k / n
    return {"k": k, "n": n, "p": round(p, 3), "se": round(math.sqrt(p * (1 - p) / n), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    out, rc = {}, 0
    for spec in a.arms:
        label, _, rest = spec.partition("=")
        d, _, win = rest.partition("@")
        lo = hi = None
        if win:
            lo, hi = (datetime.fromisoformat(x).timestamp() for x in win.split(","))
        files = [p for p in sorted(Path(d).glob("*.log")) if lo is None or lo < p.stat().st_mtime <= hi]
        g = [x for x in map(parse, files) if x and x["done"]]
        if not g:
            print(f"ALERT turn2_interrupts: {label}: 0 first-player games in {len(files)} logs", flush=True)
            rc = 3
            continue
        w = [x for x in g if x["window"]]
        it = [x for x in w if x["acts"]]
        no = [x for x in w if not x["acts"]]
        bef = sorted(x["before"] for x in it)
        cards = Counter(c for x in it for c in x["acts"])
        nw = sum(len(x["picks"]) for x in w)
        pass_first = [x["picks"].index(1) if 1 in x["picks"] else len(x["picks"]) for x in w]
        out[label] = {
            "logs": len(files), "first_games": len(g),
            "window": rate(len(w), len(g)),
            "interrupt_given_window": rate(len(it), len(w)),
            "interrupt_all": rate(len(it), len(g)),
            "windows_per_game": round(nw / max(len(w), 1), 2),
            "window_taken": rate(sum(sum(x["picks"]) for x in w), nw),
            "first_window_taken": rate(sum(x["picks"][0] for x in w), len(w)),
            "passes_before_first_act_mean": round(sum(pass_first) / max(len(w), 1), 2),
            "acts_per_interrupting_game": round(sum(len(x["acts"]) for x in it) / max(len(it), 1), 2),
            "opp_actions_before_median": bef[len(bef) // 2] if bef else None,
            "died_t2": rate(sum(x["died_t2"] for x in g), len(g)),
            "died_t2_given_window_no_interrupt": rate(sum(x["died_t2"] for x in no), len(no)),
            "died_t2_given_interrupt": rate(sum(x["died_t2"] for x in it), len(it)),
            "win_first": rate(sum(x["win"] for x in g), len(g)),
            "win_given_interrupt": rate(sum(x["win"] for x in it), len(it)),
            "win_given_window_no_interrupt": rate(sum(x["win"] for x in no), len(no)),
            "top_cards": cards.most_common(10),
        }
        print(json.dumps({label: out[label]}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
