"""Dump the going-first turn 1 (and seat 0's plays on turn 2) of every raw search_match game vs ygoagent, for both
pilots: ours (s_seat 0 = seat 0) and ygo-agent (s_seat 1, it sits in seat 0). One jsonl row per game: deck, pilot,
seed, win (seat 0 won), opening hand, seat 0's turn-1 moves, its end-of-turn-1 field, its turn-2 moves, snaps[turn] = {lp (seat 0, seat 1), f0, f1 (field cards), h0, h1 (hand sizes)}.

    python scripts/audit/turn1_lines_vs_ya.py runs/<final-card>/raw out.jsonl
Laptop-safe: engine replay only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_divergence as S  # noqa: E402

from ygo_zero.demos.game import _opts, hidden_card, option_desc  # noqa: E402

FIELD = (3, 4)  # obs loc column: mzone, szone (as eval/turn1.FIELD_LOCS)


def cards(obs, me: int, seat: int, locs) -> list[str]:
    rel = 0 if seat == me else 1
    c, m = obs["cards"], obs["card_mask"]
    out = []
    for i in range(len(m)):
        if m[i] and int(c[i, 1]) in locs and int(c[i, 3]) == rel:
            code = int(c[i, 0])
            out.append(NN.get(code, f"#{code}") if code else "(hidden)")
    return sorted(out)


def dump(g: dict, engine) -> dict:
    from ygo_zero.engine.cpp_env import CppEnv

    env = CppEnv(engine, S._config(g), True)
    obs = env.reset()
    hand, t1, t2, end, snaps = None, [], [], None, {}
    for c in g["choices"]:
        if obs is None:
            break
        turn, p = int(obs["global"][2]), int(env.player)
        if turn not in snaps and turn <= 5:
            lp = (int(obs["global"][0]), int(obs["global"][1]))
            snaps[turn] = {"lp": lp if p == 0 else lp[::-1], "f0": len(cards(obs, p, 0, FIELD)),
                           "f1": len(cards(obs, p, 1, FIELD)), "h0": len(cards(obs, p, 0, (2,))),
                           "h1": len(cards(obs, p, 1, (2,)))}
        if turn > 2:
            obs = env.step(int(c))
            continue
        if hand is None and p == 0:
            hand = cards(obs, p, 0, (2,))
        if turn == 2 and end is None:
            end = cards(obs, p, 0, FIELD)
        o = _opts(obs)
        if p == 0 and len(o) > 1:
            k, code = o[int(c)]
            nm = NN.get(code, "") if code else hidden_card(obs, int(c))
            mv = f"{S.H._kind(k)} {nm} {option_desc(obs, int(c))[:50]}".strip()
            (t1 if turn == 1 else t2).append(mv)
        obs = env.step(int(c))
    return {"deck": g["decks"][0], "opp": g["decks"][1], "pilot": "ours" if g["s_seat"] == 0 else "ya",
            "seed": g["seed"], "win": int(g["winner"] == 0), "turns": g["turns"], "hand": hand, "t1": t1,
            "end": end or [], "t2": t2, "snaps": snaps}


if __name__ == "__main__":
    from ygo_zero.demos.names import CardNames
    from ygo_zero.engine.duel import Engine

    NN = CardNames().names
    eng = Engine()
    G = S.load(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        for g in G.values():
            f.write(json.dumps(dump(g, eng)) + "\n")
