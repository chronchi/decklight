#!/usr/bin/env python3
"""Where do two policies first play differently? (window PPO follow-up, NOTEBOOK 2026-11-08 19:45: X beats A by
+7.7 pp held-out without interrupting more; user: "agreed with your two points" — the second being the divergence
analysis of the A vs X games.)

Input: `line_eval.py --ckpt A --vs-ckpt B --temp 0 --repeats 1 --keep-choices` shard outputs on the same `--starts`.
Both sides play argmax and the opponent (script, then P argmax) is deterministic, so the two games of a pair are
identical up to the first decision where A and B choose differently. For every pair, that decision is replayed and
described: turns after the line's turn (0 = the line's turn), whose decision, the decision message, A's and B's
option (kind, card, card location), and both results. Summary: identical pairs, the divergence turn / message /
option-kind distribution with the paired win difference in each group, and the first `--examples` decisions.

    uv run python scripts/audit/divergence.py --starts data/cache/line_starts/starts-v2-heavy.second.jsonl \
        --games runs/<job>/div-*.json --out runs/<job>/divergence.json

Exit 3: no pair, a start not found for a game, or a divergence on the opponent's side (the opponent must be
deterministic, so that means the replay is not).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import line_eval as LE

from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.duel import Engine
from decklight.engine.env import PyEnv
from decklight.engine.obs import LOC_ID
from decklight.web.view import DB

G_MSG = 21  # obs["global"]: decision message type (engine/obs.py)
MSG_NAME = {v: k[4:] for k, v in vars(C).items() if k.startswith("MSG_SELECT") or k == "MSG_ANNOUNCE_RACE"}
LOC_NAME = {v: k[9:].lower() for k, v in ((k, LOC_ID[getattr(C, k)]) for k in vars(C) if k.startswith("LOCATION_")
                                          and getattr(C, k) in LOC_ID)}  # fmt: skip


def _option(obs, a: int, names: dict) -> dict:
    row = obs["actions"][a]
    code = int(row[2])
    return {"kind": Kind(int(row[0])).name, "card": names.get(code, code) if code else "",
            "loc": LOC_NAME.get(int(row[7]), int(row[7]))}  # fmt: skip


def find_start(starts: list[dict], g: dict) -> dict | None:
    for st in starts:
        if st["line_id"] == g["line_id"] and g["choices"][: len(st["prefix"])] == list(st["prefix"]):
            return st
    return None


def diverge(engine, st: dict, ca: list[int], cb: list[int], names: dict) -> dict | None:
    k = next((i for i, (x, y) in enumerate(zip(ca, cb, strict=False)) if x != y), None)
    if k is None:
        return None
    env = PyEnv(engine, LE.record_config(st), st.get("auto_single", True))
    obs = env.reset()
    t0 = None
    for i, c in enumerate(ca[:k]):
        if i == len(st["prefix"]):
            t0 = int(env.duel.turn)
        obs = env.step(int(c))
    if t0 is None:
        t0 = int(env.duel.turn)
    legal = np.flatnonzero(obs["action_mask"])
    return {"index": k - len(st["prefix"]), "turn": int(env.duel.turn) - t0, "ours": int(env.player) == int(st["our_seat"]),
            "msg": MSG_NAME.get(int(obs["global"][G_MSG]), int(obs["global"][G_MSG])), "n_legal": len(legal),
            "A": _option(obs, ca[k], names), "B": _option(obs, cb[k], names)}  # fmt: skip


def group(rows: list[dict], key) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r["B_won"] - r["A_won"])
    return [{"key": k, "n": len(v), "B_minus_A": round(sum(v) / len(v), 3)} for k, v in sorted(g.items(), key=lambda x: -len(x[1]))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--games", nargs="+", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    starts = [json.loads(x) for x in a.starts.read_text().splitlines() if x.strip()]
    games = [g for f in a.games for g in json.loads(f.read_text())["games"]]
    if not games or "choices" not in games[0]["A"]:
        print("ALERT divergence: no games, or games without choices (run line_eval with --keep-choices)", flush=True)
        return 3
    engine, names = Engine(), DB.names()
    rows, same = [], Counter()
    for i, g in enumerate(games):
        st = find_start(starts, g["A"])
        if st is None:
            print(f"ALERT divergence: no start for {g['A']['line_id']}", flush=True)
            return 3
        wa, wb = g["A"]["won"], g["B"]["won"]
        if wa is None or wb is None:
            same["unfinished"] += 1
            continue
        d = diverge(engine, st, g["A"]["choices"], g["B"]["choices"], names)
        if d is None:
            same["identical"] += 1
            continue
        if not d["ours"]:
            print(f"ALERT divergence: {st['line_id']} diverged on the opponent's decision {d}", flush=True)
            return 3
        rows.append({"line_id": st["line_id"], **d, "A_won": wa, "B_won": wb})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(games)} pairs", flush=True)
    if not rows:
        print("ALERT divergence: no diverging pair", flush=True)
        return 3
    out = {
        "pairs": len(games), **same, "diverged": len(rows),
        "B_minus_A_diverged": round(sum(r["B_won"] - r["A_won"] for r in rows) / len(rows), 3),
        "by_turn": group(rows, lambda r: min(r["turn"], 3)),
        "by_msg": group(rows, lambda r: r["msg"]),
        "by_kinds": group(rows, lambda r: f"{r['A']['kind']} -> {r['B']['kind']}")[:15],
        "by_turn_kinds": group(rows, lambda r: f"t{min(r['turn'], 3)} {r['A']['kind']} -> {r['B']['kind']}")[:20],
        "examples": rows[: a.examples], "rows": rows,
    }  # fmt: skip
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "examples")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
