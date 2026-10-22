"""Read turn-1 lines (scripts/turn1_lines.py lines.jsonl) as text: opening hand, player 0's decisions, end board.

    python scripts/audit/show_lines.py <lines.jsonl> [--deck D] [--hand H] [--kinds greedy,best_v,best_f,expert]
        [--max-hands N] [--stats]

Per (deck, source, hand) group prints the greedy line, the best sample by critic value (best_v), the best by end field
(best_f) and the expert line (verified hands), each with field and v_p0, so a reader can judge whether the line the
critic prefers is actually a better combo. `--stats` prints whole-data distributions instead: per deck greedy vs best
field / v_p0, how often best_v and best_f agree, lines that end with an empty board, and degenerate lines (many
decisions, field 0; v high but field below greedy: the critic liking something odd).
Laptop-safe: engine replay only, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hand_play as H  # noqa: E402
from decklight.demos.game import _opts, hidden_card, option_desc  # noqa: E402
from decklight.demos.names import CardNames  # noqa: E402
from decklight.engine.cpp_env import CppEnv  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402
from decklight.demos.dataset import record_config  # noqa: E402

LOC = {2: "hand", 3: "M", 4: "S", 5: "GY", 6: "banished", 7: "extra"}


def board(obs, nn: dict, me: int) -> str:
    """Player `me`'s field (monsters, spells/traps) from an obs taken in either player's view."""
    c, m = obs["cards"], obs["card_mask"]
    ctrl = 0 if int(obs.get("_player", me)) == me else 1
    out = []
    for i in range(len(m)):
        if m[i] and int(c[i, 1]) in (3, 4) and int(c[i, 3]) == ctrl:
            out.append(f"{LOC[int(c[i, 1])]}:{nn.get(int(c[i, 0]), '?') if c[i, 0] else 'set'}")
    return ", ".join(out) or "(empty)"


def render(row: dict, engine: Engine, nn: dict) -> list[str]:
    env = CppEnv(engine, record_config(row), row.get("auto_single", True))
    obs = env.reset()
    hand = None  # read at player 0's first decision (the first obs can be player 1's view, our hand hidden)
    out = ["    hand: ?"]
    last = None
    for c in row["choices"]:
        if obs is None:
            break
        o = _opts(obs)
        if env.player == 0 and hand is None:
            hand = [nn.get(int(r[0]), "?") for r, mk in zip(obs["cards"], obs["card_mask"], strict=True)
                    if mk and int(r[1]) == 2 and int(r[3]) == 0]  # fmt: skip
            out[0] = f"    hand now: {', '.join(hand)}"
        if env.player == 0 and len(o) > 1:
            k, code = o[int(c)]
            nm = nn.get(code, "") if code else hidden_card(obs, int(c))
            msg = H._MSGS.get(int(obs["global"][21])) or ""
            out.append(f"    {msg[11:]} {H._kind(k)} {nm} {option_desc(obs, int(c))!r} /{len(o)}")
        last = (obs, env.player)
        obs = env.step(int(c))
    end, who = (obs, env.player) if obs is not None else last
    end = dict(end, _player=who)
    out.append(f"    END field={row['field']} v_p0={row['v_p0']:+.2f} stuck={row['stuck']}: {board(end, nn, 0)}")
    return out


def groups_of(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[(r["deck"], r.get("source", "random"), r["hand"])].append(r)
    return g


def pick(rs: list[dict]) -> dict:
    gr = next((r for r in rs if r["kind"] == "greedy"), None)
    ex = next((r for r in rs if r["kind"] == "expert"), None)
    ss = [r for r in rs if r["kind"] == "sample" and not r["stuck"]]
    bv = max(ss, key=lambda r: (r["v_p0"], r["field"]), default=None)
    bf = max(ss, key=lambda r: (r["field"], r["v_p0"]), default=None)
    return {"greedy": gr, "best_v": bv, "best_f": bf, "expert": ex}


def stats(rows: list[dict]) -> None:
    per = defaultdict(lambda: defaultdict(list))
    odd = []
    for (deck, src, h), rs in sorted(groups_of(rows).items()):
        p = pick(rs)
        g, bv, bf = p["greedy"], p["best_v"], p["best_f"]
        if not (g and bv):
            continue
        d = per[deck]
        d["g_f"].append(g["field"]); d["g_v"].append(g["v_p0"])  # noqa: E702
        d["bv_f"].append(bv["field"]); d["bv_v"].append(bv["v_p0"])  # noqa: E702
        d["bf_f"].append(bf["field"]); d["agree"].append(bv is bf)  # noqa: E702
        d["stuck"].append(sum(r["stuck"] for r in rs) / len(rs))
        if p["expert"]:
            d["ex_f"].append(p["expert"]["field"]); d["ex_v"].append(p["expert"]["v_p0"])  # noqa: E702
        if bv["field"] < g["field"] and bv["v_p0"] > g["v_p0"] + 0.1:
            odd.append(f"  {deck} {src} h{h}: best_v field {bv['field']} < greedy {g['field']} but v {bv['v_p0']:+.2f} > "
                       f"{g['v_p0']:+.2f}")
        for r in rs:
            if r["field"] == 0 and r.get("p0_decisions", 0) >= 15:
                odd.append(f"  {deck} {src} h{h} {r['kind']}#{r.get('sample', '')}: {r['p0_decisions']} decisions, field 0")

    def m(v: list) -> str:
        return f"{sum(v) / len(v):5.2f}" if v else "    -"

    print(f"{'deck':<44} n  greedy f/v      best_v f/v      best_f f  agree stuck  expert f/v")
    for deck, d in sorted(per.items()):
        print(f"{deck:<44} {len(d['g_f']):2d} {m(d['g_f'])}/{m(d['g_v'])}  {m(d['bv_f'])}/{m(d['bv_v'])}  {m(d['bf_f'])}"
              f"  {m(d['agree'])} {m(d['stuck'])}  {m(d['ex_f'])}/{m(d['ex_v'])}")
    print(f"\nodd lines ({len(odd)}):")
    print("\n".join(odd[:80]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lines", type=Path)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--hand", type=int, default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--kinds", default="greedy,best_v,best_f,expert")
    ap.add_argument("--max-hands", type=int, default=3)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(x) for x in a.lines.read_text().splitlines() if x.strip()]
    if a.stats:
        stats(rows)
        return 0
    nn, engine = CardNames().names, Engine()
    shown = 0
    for (deck, src, h), rs in sorted(groups_of(rows).items()):
        if (a.deck and deck != a.deck) or (a.hand is not None and h != a.hand) or (a.source and src != a.source):
            continue
        if shown >= a.max_hands:
            break
        shown += 1
        print(f"== {deck} {src} hand {h}")
        p = pick(rs)
        for kind in a.kinds.split(","):
            r = p.get(kind)
            if r is None:
                continue
            same = kind == "best_f" and r is p["best_v"] and "best_v" in a.kinds.split(",")
            print(f"  -- {kind}{' (= best_v)' if same else ''}" + (f" sample {r.get('sample')}" if kind.startswith("best") else ""))
            if not same:
                print("\n".join(render(r, engine, nn)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
