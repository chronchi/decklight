#!/usr/bin/env python3
"""Cycle-1 G5 (docs/lab/prereg/2026-11-10-master-cycle1.md): going first, does a policy end turn 1 at once with a live
hand? For seeds 0, 1, ... deal the deck in seat 0 (it goes first), auto-answer forced steps, and stop at the first
turn-1 Main-1 idle decision of seat 0. The hand is live when it holds a monster or a card offered as an activation
from the hand at that decision (searchers, field spells: the prereg's "≥ 1 monster or a searcher"). On the first
`--hands` live hands, each arm's softmax probability of TO_END (end the turn) and whether it is the argmax.
Every arm sees the same observation. Rule (reported, not blocking): the candidate's mean p(to_end) ≤ half the base's.

    uv run python scripts/audit/t1_to_end.py --deck decks/meta-2026-11/invoked.ydk --hands 200 \
        --arm X=runs/window-ppo-20261108-0940/train/export-X/policy/best.pt \
        --arm X2=runs/c1-round-20261110-1927/X2/best.pt --out runs/c1-g5/t1_to_end.json

Exit 3: no live hand, an arm without TO_END probabilities, or two arms with identical probabilities everywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

from decklight.decks.ydk import load_ydk
from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.eval.bc_policy import BCPolicy
from decklight.web.view import DB

EMB = "data/cards/emb/qwen3-0.6b-d512.npz"


def probs(pol: BCPolicy, obs: dict, n: int) -> np.ndarray:
    z = np.asarray(pol.logits({k: np.ascontiguousarray(v)[None] for k, v in obs.items()})[0], dtype=np.float64)[:n]
    z[~np.asarray(obs["action_mask"])[:n].astype(bool)] = -np.inf
    e = np.exp(z - z.max())
    return e / e.sum()


def first_main1(env: PyEnv, obs: dict) -> dict | None:
    """Step (first legal option) until seat 0's first turn-1 idle decision; None if the duel never reaches it."""
    for _ in range(200):
        if obs is None or env.done:
            return None
        if env.player == 0 and obs["global"][21] == C.MSG_SELECT_IDLECMD:  # seat 0 goes first: its first idle is T1
            return obs
        obs = env.step(int(np.flatnonzero(np.asarray(obs["action_mask"])[: len(env.opts)])[0]))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opp", default=None, help="opponent deck (default: the same list)")
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--max-seeds", type=int, default=2000)
    ap.add_argument("--arm", action="append", required=True, help="LABEL=CKPT; the first is the base")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = dict(s.split("=", 1) for s in a.arm)
    pols = {k: BCPolicy(v, EMB, threads=2) for k, v in arms.items()}
    deck, opp = load_ydk(a.deck), load_ydk(a.opp or a.deck)
    stats, names, engine = DB.stats(), DB.names(), Engine()
    rows, dead, seed = [], 0, 0
    print(f"t1_to_end: {a.deck}, arms {list(arms)}, target {a.hands} live hands", flush=True)
    while len(rows) < a.hands and seed < a.max_seeds:
        env = PyEnv(engine, DuelConfig(deck0=deck, deck1=opp, seed=seed))
        obs = first_main1(env, env.reset())
        seed += 1
        if obs is None:
            continue
        hand = [c.code for c in env.duel.query_location(0, C.LOCATION_HAND) if c is not None]
        kinds = [o.kind for o in env.opts]
        mons = [c for c in hand if stats.get(c, (0,) * 6)[5] & C.TYPE_MONSTER]
        acts = [o for o in env.opts if o.kind == Kind.ACTIVATE and o.card is not None
                and getattr(o.card, "location", 0) == C.LOCATION_HAND]  # fmt: skip
        if Kind.TO_END not in kinds:
            continue
        if not mons and not acts:
            dead += 1
            continue
        j = kinds.index(Kind.TO_END)
        row = {"seed": seed - 1, "hand": [names.get(c, str(c)) for c in hand], "options": len(env.opts)}
        for k, pol in pols.items():
            p = probs(pol, obs, len(env.opts))
            row[k] = {"p_to_end": round(float(p[j]), 4), "argmax_to_end": int(p.argmax()) == j,
                      "argmax": kinds[int(p.argmax())].name}  # fmt: skip
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}/{a.hands} live hands (seeds {seed}, dead {dead}): "
                  + " ".join(f"{k} {mean(r[k]['p_to_end'] for r in rows):.3f}" for k in arms), flush=True)  # fmt: skip
        env.duel.close()
    if not rows:
        print("ALERT t1_to_end: no live hand", flush=True)
        return 3
    keys = list(arms)
    for x, y in zip(keys, keys[1:]):
        if all(r[x]["p_to_end"] == r[y]["p_to_end"] for r in rows):
            print(f"ALERT t1_to_end: {x} and {y} give identical probabilities", flush=True)
            return 3
    summ = {k: {"mean_p_to_end": round(mean(r[k]["p_to_end"] for r in rows), 4),
                "argmax_to_end": round(mean(r[k]["argmax_to_end"] for r in rows), 4),
                "p_gt_0.5": round(mean(r[k]["p_to_end"] > 0.5 for r in rows), 4)} for k in keys}  # fmt: skip
    base = summ[keys[0]]["mean_p_to_end"]
    for k in keys[1:]:
        summ[k]["halved_vs_base"] = summ[k]["mean_p_to_end"] <= base / 2
    out = {"deck": a.deck, "arms": arms, "hands": len(rows), "seeds": seed, "dead_hands": dead, "summary": summ,
           "rows": rows}  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"hands": len(rows), "dead": dead, **summ}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
