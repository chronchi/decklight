"""End-board composition of selected turn-1 lines vs greedy: monsters / face-up spells-traps / set cards.

    python scripts/audit/board_mix.py <lines.jsonl> <selected dir (select_turn1_lines --out)>

Checks that a bigger "field" is real board, not padding (e.g. setting every card in hand). Laptop-safe (engine only).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.demos.dataset import record_config  # noqa: E402
from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import Engine  # noqa: E402


def mix(engine: Engine, row: dict) -> Counter:
    env = CppEnv(engine, record_config(row), row.get("auto_single", True))
    obs, last = env.reset(), None
    for c in row["choices"]:
        if obs is None or int(obs["global"][2]) > 1:
            break
        last = (obs, env.player)
        obs = env.step(int(c))
    o, who = (obs, env.player) if obs is not None else last
    ctrl = 0 if who == 0 else 1
    k = Counter()
    for r, m in zip(o["cards"], o["card_mask"], strict=True):
        if m and int(r[3]) == ctrl and int(r[1]) in (3, 4):
            k["monster" if int(r[1]) == 3 else ("set" if int(r[5]) == 6 else "faceup_st")] += 1
    return k


def main() -> int:
    lines, sel = Path(sys.argv[1]), Path(sys.argv[2])
    rows = {}
    for x in lines.read_text().splitlines():
        r = json.loads(x)
        rows[(r["deck"], r["source"], r["hand"], r["kind"], r["sample"])] = r
    engine = Engine()
    g_tot, b_tot, n = Counter(), Counter(), 0
    for f in sorted(sel.glob("p0-*.jsonl")):
        for x in f.read_text().splitlines():
            rec = json.loads(x)
            deck, src, h, s = rec["game_id"].removeprefix("si-").rsplit("-", 3)
            h, s = int(h[1:]), int(s[1:])
            g_tot += mix(engine, rows[(deck, src, h, "greedy", -1)])
            b_tot += mix(engine, rows[(deck, src, h, "sample", s)])
            n += 1
    print(f"{n} selected lines; mean per line (greedy -> selected):")
    for k in ("monster", "faceup_st", "set"):
        print(f"  {k:<10} {g_tot[k] / n:5.2f} -> {b_tot[k] / n:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
