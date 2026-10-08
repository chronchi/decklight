"""Read play-out-scored whole-turn lines (turn_lines.py -> score_lines.py rows with actor / turn_start / prefix_len):
per start, the board at the start of the turn, then the greedy line and the best-by-rollout sample (actor's decisions
only, after the prefix), each with its end boards, LP and rollout score.

    python scripts/audit/show_turns.py <scored.jsonl> [--bucket t2|t3-5|t6+|t1] [--min-gain 0.5] [--max N] [--seed S]
Laptop-safe: engine replay only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_lines as SL  # noqa: E402
from show_lines import H, _opts, hidden_card, option_desc, record_config, CppEnv  # noqa: E402


def bucket(t: int) -> str:
    return "t1" if t == 1 else "t2" if t == 2 else "t3-5" if t <= 5 else "t6+"


def side(obs, nn, who, me) -> str:
    """Hand count, field and LP of absolute seat `who` from obs seen by seat `me`."""
    rel = 0 if who == me else 1
    c, m = obs["cards"], obs["card_mask"]
    hand = sum(1 for i in range(len(m)) if m[i] and int(c[i, 1]) == 2 and int(c[i, 3]) == rel)
    return f"hand {hand} | {SL.board(dict(obs, _player=me), nn, who)}"


def render(row, engine, nn, show_moves=True) -> list[str]:
    env = CppEnv(engine, record_config(row), row.get("auto_single", True))
    obs = env.reset()
    a, pl = int(row["actor"]), int(row["prefix_len"])
    out = []
    last = None
    for i, c in enumerate(row["choices"]):
        if obs is None:
            break
        if i == pl:
            out.append(f"    START actor  {side(obs, nn, a, env.player)}")
            out.append(f"    START opp    {side(obs, nn, 1 - a, env.player)}")
        o = _opts(obs)
        if show_moves and i >= pl and len(o) > 1:
            k, code = o[int(c)]
            nm = nn.get(code, "") if code else hidden_card(obs, int(c))
            msg = H._MSGS.get(int(obs["global"][21])) or ""
            who = "ME " if env.player == a else "OPP"
            out.append(f"    {who} {msg[11:]} {H._kind(k)} {nm} {option_desc(obs, int(c))[:70]!r} /{len(o)}")
        last = (obs, env.player)
        obs = env.step(int(c))
    end, me = (obs, env.player) if obs is not None else last
    out.append(f"    END actor    {side(end, nn, a, me)}")
    out.append(f"    END opp      {side(end, nn, 1 - a, me)}  lp_diff {row['lp_diff']:+d}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scored", type=Path)
    ap.add_argument("--bucket")
    ap.add_argument("--min-gain", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    g = defaultdict(list)
    for x in a.scored.read_text().splitlines():
        r = json.loads(x)
        g[(r["deck"], r.get("start", r["hand"]))].append(r)
    keys = sorted(g)
    random.Random(a.seed).shuffle(keys)
    nn, engine = SL.CardNames().names, SL.Engine()
    shown = 0
    for k in keys:
        rs = g[k]
        gr = next((r for r in rs if r["kind"] == "greedy"), None)
        sm = [r for r in rs if r["kind"] != "greedy"]
        if not (gr and sm) or (a.bucket and bucket(gr["turn_start"]) != a.bucket):
            continue
        br = max(sm, key=lambda r: r["rollout"])
        if br["rollout"] - gr["rollout"] < a.min_gain:
            continue
        shown += 1
        if shown > a.max:
            break
        print(f"== {gr['deck']} vs {gr['opp']} start {gr.get('start')} turn {gr['turn_start']} actor {gr['actor']}")
        for name, r in (("greedy", gr), ("best_rollout", br)):
            ss = " ".join(f"{s:+.1f}" for s in r["rollout_samples"])
            print(f"  -- {name}: rollout {r['rollout']:+.2f} [{ss}]")
            print("\n".join(render(r, engine, nn)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
