"""Audit DAGS start positions by curriculum class (hopeless / trainable / solved), from the learner's side.

    uv run python scripts/audit/dags_hopeless.py --curric runs/<rl run>/rl/curriculum.json \
        --starts data/cache/dags/starts-train-1k.jsonl [--show 12] [--out <dir>]

A pass-rate slot is (start, seat): slot = 2*start + seat, the learner plays `seat` from the start position against
the league. Classes (curriculum.py): solved = n >= min_n and every game won, hopeless = n >= min_n and none won,
trainable = n >= min_n otherwise, few = 0 < n < min_n, unseen = n = 0.
Per class: learner-minus-opponent LP, field, hand and graveyard sizes at the start; turn; whether it is the
learner's turn; whether the cut is mid-turn (pending chain or a non-idle prompt); how many starts per game and per
deck. With --show K: K hopeless slots rendered (LP, both fields by card name, the learner's hand) for reading.
Engine replay only, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.demos.names import CardNames  # noqa: E402
from ygo_zero.engine import constants as C  # noqa: E402
from ygo_zero.engine.duel import Engine  # noqa: E402
from ygo_zero.eval.turn1 import dags_config, replay_prefix  # noqa: E402

LOC = {2: "hand", 3: "mzone", 4: "szone", 5: "grave", 6: "removed", 7: "extra"}
IDLE = {C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD}


def seat_of(obs) -> int:
    """Absolute seat of the acting player (global[4] = I went first)."""
    return 0 if int(obs["global"][4]) == 1 else 1


def snapshot(obs, learner: int) -> dict:
    """Counts and card lists from the learner's side (the obs is from the acting player's view)."""
    acting = seat_of(obs)
    flip = acting != learner  # controller column is relative to the viewer
    g = obs["global"]
    lp_me, lp_opp = (int(g[1]), int(g[0])) if flip else (int(g[0]), int(g[1]))
    cnt = {"me": Counter(), "opp": Counter()}
    cards = {"me": defaultdict(list), "opp": defaultdict(list)}
    c, m = obs["cards"], obs["card_mask"]
    for j in np.flatnonzero(m):
        loc, ctrl, code = int(c[j, 1]), int(c[j, 3]), int(c[j, 0])
        side = "me" if (ctrl == 0) != flip else "opp"
        if loc in LOC:
            cnt[side][LOC[loc]] += 1
            cards[side][LOC[loc]].append(code)
    return {"lp_me": lp_me, "lp_opp": lp_opp, "cnt": cnt, "cards": cards, "turn": int(g[2]),
            "learner_turn": (int(g[5]) == 1) != flip, "learner_acts": not flip, "chain": int(g[20]),
            "msg": int(g[21]), "phase": int(g[3])}


def classify(n: int, w: float, min_n: int) -> str:
    if n == 0:
        return "unseen"
    if n < min_n:
        return "few"
    if w >= n:
        return "solved"
    if w <= 0:
        return "hopeless"
    return "trainable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curric", type=Path, required=True)
    ap.add_argument("--starts", type=Path, default=ROOT / "data/cache/dags/starts-train-1k.jsonl")
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None, help="write per-slot rows (slots.jsonl) here")
    args = ap.parse_args()
    cur = json.loads(args.curric.read_text())
    rows = [json.loads(line) for line in args.starts.read_text().splitlines() if line.strip()]
    if cur["n_starts"] != 2 * len(rows):
        raise SystemExit(f"curriculum has {cur['n_starts']} slots, starts file {len(rows)} starts")
    engine, names = Engine(), CardNames()
    snaps: dict[int, dict] = {}
    bad = 0
    for i, r in enumerate(rows):
        try:
            _, obs = replay_prefix(engine, dags_config(r), [int(x) for x in r.get("prefix", [])])
        except ValueError:
            bad += 1
            continue
        snaps[i] = obs
    if bad:
        print(f"WARNING {bad} starts do not replay")
    per = defaultdict(list)
    out_rows = []
    for slot in range(cur["n_starts"]):
        s, seat = divmod(slot, 2)
        if s not in snaps:
            continue
        n, w = int(cur["n"][slot]), float(cur["wins"][slot])
        k = classify(n, w, int(cur["min_n"]))
        sn = snapshot(snaps[s], seat)
        sn.update(slot=slot, start=s, seat=seat, n=n, w=w, cls=k, game=rows[s].get("game_id"),
                  decks=rows[s].get("decks"), prefix_len=len(rows[s].get("prefix", [])))
        per[k].append(sn)
        out_rows.append({kk: v for kk, v in sn.items() if kk != "cards"} | {"cnt": {a: dict(b) for a, b in sn["cnt"].items()}})
    print(f"{len(rows)} starts, {cur['n_starts']} slots; per class: " + ", ".join(f"{k} {len(v)}" for k, v in per.items()))
    hdr = ("class", "slots", "LP me-opp", "LP me", "LP opp", "field me-opp", "hand me-opp", "grave me-opp", "turn",
           "my turn", "I act", "mid-turn", "seat0", "games")
    print("  ".join(f"{h:>12}" for h in hdr))
    for k in ("hopeless", "trainable", "solved", "few", "unseen"):
        v = per.get(k, [])
        if not v:
            continue

        def m(f):
            return np.mean([f(x) for x in v])

        def fld(x, side):
            return x["cnt"][side]["mzone"] + x["cnt"][side]["szone"]

        vals = (k, len(v), m(lambda x: x["lp_me"] - x["lp_opp"]), m(lambda x: x["lp_me"]), m(lambda x: x["lp_opp"]),
                m(lambda x: fld(x, "me") - fld(x, "opp")), m(lambda x: x["cnt"]["me"]["hand"] - x["cnt"]["opp"]["hand"]),
                m(lambda x: x["cnt"]["me"]["grave"] - x["cnt"]["opp"]["grave"]), m(lambda x: x["turn"]),
                m(lambda x: x["learner_turn"]), m(lambda x: x["learner_acts"]),
                m(lambda x: x["chain"] > 0 or x["msg"] not in IDLE), m(lambda x: x["seat"] == 0),
                len({x["game"] for x in v}))
        print("  ".join(f"{x:>12.2f}" if isinstance(x, float) else f"{x!s:>12}" for x in vals))
    for k in ("hopeless", "solved"):
        v = per.get(k, [])
        lp = Counter("me<=1000" if x["lp_me"] <= 1000 else "opp<=1000" if x["lp_opp"] <= 1000 else "both>1000" for x in v)
        print(f"{k}: LP buckets {dict(lp)}; turns {dict(sorted(Counter(min(x['turn'], 8) for x in v).items()))}; "
              f"decks (learner side) {Counter(x['decks'][x['seat']] for x in v).most_common(6)}")
    hop = per.get("hopeless", [])
    if args.show and hop:
        rng = np.random.default_rng(0)
        print(f"\n=== {min(args.show, len(hop))} random hopeless slots (learner's view)")
        for x in [hop[i] for i in rng.choice(len(hop), min(args.show, len(hop)), replace=False)]:
            def nm(codes):
                return ", ".join(names.names.get(c, f"#{c}") if c else "(set/hidden)" for c in codes) or "-"
            print(f"- slot {x['slot']} ({x['decks'][x['seat']]} vs {x['decks'][1 - x['seat']]}), seat {x['seat']}, "
                  f"turn {x['turn']} ({'learner' if x['learner_turn'] else 'opponent'}'s turn, "
                  f"{'learner' if x['learner_acts'] else 'opponent'} to act, chain {x['chain']}), games 0/{x['n']}, "
                  f"LP {x['lp_me']} vs {x['lp_opp']}")
            for side, who in (("me", "learner"), ("opp", "opponent")):
                cs = x["cards"][side]
                print(f"    {who}: monsters [{nm(cs['mzone'])}] | s/t [{nm(cs['szone'])}] | hand {len(cs['hand'])}"
                      + (f" [{nm(cs['hand'])}]" if side == "me" and x["learner_acts"] else "")
                      + f" | grave {len(cs['grave'])}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "slots.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))


if __name__ == "__main__":
    main()
