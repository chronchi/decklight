"""Step 1a of the opponent-turn plan (docs/plans/2026-11-03-opp-turn-interaction.md): which decks really combo?

Per candidate deck, with the opponent passive (no interaction), so this measures combo power, not matchup play:
- going first (decks with verified line records): the expert line's end field vs P's (`playtest.combo_test`, same
  opening hand) and P's best of K sampled turn-1 runs;
- going second, into our board: our deck (default SkyDeck's list) plays turn 1 with P, then the candidate deck plays
  turn 2 with P (argmax) and with K sampled runs of P; our side passes everything on turn 2. Measured: our LP lost, kill
  rate, the candidate's field cards at the end of turn 2. Our turn 1 is always P's argmax, so all runs of a deal face
  the same board; sampled runs are seeded per (deck, deal, sample), so results do not depend on deck order or sharding.
  `lp_lost` = damage dealt to us on turn 2 (from the duel's messages), `lp_gained` = our recovery that turn; `kill` =
  our LP 0 while theirs > 0 (draws are `draw_or_other_end`). `second_best_of_k` is over the K samples only.

A deck whose P runs fall far below its expert line, or whose going-second runs neither kill nor build, is a weak
opponent under P: P-driven games with it would teach on weak combos (NOTEBOOK 2026-11-03 "60/20/20").

    uv run python scripts/audit/combo_pool.py --decks fire-king,raise-moon-ygopd --seeds 2 --samples 2 --hands 2 \
        --out /tmp/combo-pool-smoke.json
    # full (vast, one process per core): --shard i/n splits the deck list
    uv run python scripts/audit/combo_pool.py --all --seeds 8 --samples 4 --hands 6 --shard 0/32 --out runs/<job>/s0.json

Exits 3 when a deck list is empty or a whole metric is zero for every deck (nothing ran); 4 when results were written
but some deck raised an ALERT (error, missing hands, cut run, no turn 2, LP check failed).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import playtest as PT  # noqa: E402

from decklight.decks import load_ydk  # noqa: E402
from decklight.demos.game import Responder  # noqa: E402
from decklight.engine import constants as C  # noqa: E402
from decklight.engine import query as Q  # noqa: E402
from decklight.engine.duel import DuelConfig, Engine  # noqa: E402
from decklight.engine.env import PyEnv  # noqa: E402
from decklight.eval.bc_policy import BCPolicy  # noqa: E402
from decklight.search.product import resolve_product_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "decks/pool"
# Decks measured whether or not they have verified lines yet.
PINNED_DECKS = ("fire-king", "galaxy-eyes", "kashtira", "sacred-beasts-meta1", "raise-moon-ygopd",
               "cyberdark-ygopd", "firewall-cyberse-ygopd", "rocket-warrior-ygopd")  # fmt: skip
MAX_DECISIONS = 600


def _act(pol: BCPolicy, obs: dict) -> int:
    return int(pol({k: np.asarray(v)[None] for k, v in obs.items()}, np.arange(1), slots=np.zeros(1, np.int64))[0])


def turn_lp(trace: list[bytes], player: int) -> dict[int, dict[str, int]]:
    """Per turn, the damage, recovery and LP costs of `player`, read from the duel's raw messages (MSG_NEW_TURN counts
    turns; MSG_DAMAGE / MSG_RECOVER / MSG_PAY_LPCOST = type, player u8, amount u32)."""
    out: dict[int, dict[str, int]] = {}
    turn = 0
    for raw in trace:
        t = raw[0]
        if t == C.MSG_NEW_TURN:
            turn += 1
        elif t in (C.MSG_DAMAGE, C.MSG_RECOVER, C.MSG_PAY_LPCOST) and raw[1] == player:
            k = {C.MSG_DAMAGE: "damage", C.MSG_RECOVER: "recover", C.MSG_PAY_LPCOST: "cost"}[t]
            d = out.setdefault(turn, {"damage": 0, "recover": 0, "cost": 0})
            d[k] += int.from_bytes(raw[2:6], "little")
    return out


def going_second(engine, ours_pol: BCPolicy, their_pol: BCPolicy, ours, theirs, seed: int) -> dict:
    """Our deck goes first with `ours_pol` (always the argmax policy, so every run of a deal faces the same board of
    ours), their deck passive; then their deck plays turn 2 with `their_pol` while we pass."""
    for p in {id(ours_pol): ours_pol, id(their_pol): their_pol}.values():
        if getattr(p, "guard", None) is not None:
            p.guard.reset()
    env = PyEnv(engine, DuelConfig(deck0=ours, deck1=theirs, seed=seed), auto_single=True)
    obs, n, passive = env.reset(), 0, Responder(seed)
    reached2 = False
    while obs is not None and n < MAX_DECISIONS:
        turn = int(env.duel.turn)
        if turn > 2:
            break
        reached2 |= turn == 2
        me_acting = env.player == 0
        if turn == 1:
            a = _act(ours_pol, obs) if me_acting else passive.passive(obs)
        else:
            a = passive.passive(obs) if me_acting else _act(their_pol, obs)
        obs = env.step(a)
        n += 1
    lp = [v - (1 << 32) if v >= 1 << 31 else v for v in Q.lps(env.duel.query_field())]  # LP below 0 wraps as uint32
    per = turn_lp(env.duel.trace, 0)
    t2 = per.get(2, {"damage": 0, "recover": 0, "cost": 0})
    # invariant: our LP = 8000 - all damage - costs + recovery (checks the message parsing; LP floors at 0 in the core)
    expect = 8000 - sum(d["damage"] + d["cost"] - d["recover"] for d in per.values())
    field = [sum(c is not None for loc in (C.LOCATION_MZONE, C.LOCATION_SZONE)
                 for c in env.duel.query_location(p, loc)) for p in (0, 1)]  # fmt: skip
    return {"lp_lost": t2["damage"], "lp_gained": t2["recover"], "kill": bool(lp[0] <= 0 < lp[1]),
            "draw_or_other_end": bool(env.duel.ended and not (lp[0] <= 0 < lp[1])),
            "field": int(field[1]), "our_field": int(field[0]), "decisions": n,
            "cut": n >= MAX_DECISIONS, "reached_turn2": reached2,
            "lp_check": max(lp[0], 0) == max(expect, 0)}  # fmt: skip


def measure_deck(engine, pol, sampler, ours, name: str, seeds: int, samples: int, hands: int) -> dict:
    deck = load_ydk(POOL / f"{name}.ydk")
    if not deck.main:
        raise ValueError(f"empty main deck {name}")
    out: dict = {"deck": name, "alerts": []}
    t = time.time()
    tag = zlib.crc32(name.encode())
    if PT.verified_records(name, limit=1):
        if sampler is not None:  # reproducible per deck, whatever ran before in this process
            sampler.rng = np.random.default_rng([tag, 1])
        out["first"] = PT.combo_test(engine, pol, name, hands, sampler if samples else None, samples)
        got = (out["first"] or {}).get("hands", 0)
        if got < min(hands, len(PT.verified_records(name, limit=hands))):
            out["alerts"].append(f"first: {got} hands replayed of {hands} requested")
    runs_p, runs_k = [], []
    for s in range(seeds):
        seed = 77000 + s
        runs_p.append(going_second(engine, pol, pol, ours, deck, seed))
        rs = []
        for j in range(samples):
            sampler.rng = np.random.default_rng([tag, seed, j])
            rs.append(going_second(engine, pol, sampler, ours, deck, seed))
        runs_k.append(rs)
    for r in runs_p + [x for rs in runs_k for x in rs]:
        if r["cut"] or not r["reached_turn2"] or not r["lp_check"]:
            out["alerts"].append(f"second run cut={r['cut']} reached_turn2={r['reached_turn2']} lp_check={r['lp_check']}")
            break

    def agg(runs):
        return {"lp_lost": round(statistics.mean(r["lp_lost"] for r in runs)),
                "lp_gained": round(statistics.mean(r["lp_gained"] for r in runs)),
                "kill": round(sum(r["kill"] for r in runs) / len(runs), 3),
                "field": round(statistics.mean(r["field"] for r in runs), 2),
                "our_field_left": round(statistics.mean(r["our_field"] for r in runs), 2),
                "cut": sum(r["cut"] for r in runs)}  # fmt: skip

    out["second_runs_p"] = runs_p
    out["second_p"] = agg(runs_p)
    if samples:
        # over the K sampled runs only (argmax excluded): per deal, the run with a kill, else the most damage
        out["second_best_of_k"] = agg([max(rs, key=lambda r: (r["kill"], r["lp_lost"])) for rs in runs_k])
        out["second_best_field_of_k"] = round(statistics.mean(max(r["field"] for r in rs) for rs in runs_k), 2)
        out["second_sample_kill"] = round(sum(r["kill"] for rs in runs_k for r in rs) / (seeds * samples), 3)
    out["seconds"] = round(time.time() - t, 1)
    return out


def candidates(args) -> list[str]:
    if args.decks:
        names = args.decks.split(",")
    else:
        names = sorted({p.stem for p in POOL.glob("*.ydk") if PT.verified_records(p.stem, limit=1)} | set(PINNED_DECKS))
    missing = [n for n in names if not (POOL / f"{n}.ydk").exists()]
    if missing:
        raise SystemExit(f"ALERT combo_pool: no such pool deck {missing}")
    i, n = (int(x) for x in args.shard.split("/"))
    return names[i::n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default="", help="comma-separated pool deck stems (default with --all: every deck "
                    "with verified lines + the 8 pinned decks)")  # fmt: skip
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--our-deck", type=Path, default=ROOT / "decks/custom/skydeck-tcg2026-10.ydk")
    ap.add_argument("--seeds", type=int, default=8, help="going-second deals per deck")
    ap.add_argument("--samples", type=int, default=4, help="sampled P runs per deal / hand (best-of-K headroom)")
    ap.add_argument("--hands", type=int, default=6, help="going-first verified hands per deck")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--emb", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not a.decks and not a.all:
        ap.error("--decks or --all")
    names = candidates(a)
    if not names:
        print("ALERT combo_pool: empty deck list", flush=True)
        return 3
    ckpt, _value, emb = resolve_product_paths(a.ckpt, None, a.emb)
    engine = Engine()
    pol = BCPolicy(ckpt, emb, device=a.device)
    sampler = BCPolicy(ckpt, emb, temperature=1.0, seed=0, device=a.device) if a.samples else None
    ours = load_ydk(a.our_deck)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        try:
            r = measure_deck(engine, pol, sampler, ours, name, a.seeds, a.samples, a.hands)
        except Exception as e:  # noqa: BLE001 - one broken deck must not end the shard; it is reported
            r = {"deck": name, "error": f"{type(e).__name__}: {e}"}
            print(f"ALERT combo_pool {name}: {r['error']}", flush=True)
        for al in r.get("alerts", []):
            print(f"ALERT combo_pool {name}: {al}", flush=True)
        rows.append(r)
        f, s2, sb = r.get("first") or {}, r.get("second_p") or {}, r.get("second_best_of_k") or {}
        print(f"{name:40s} first P {f.get('policy_end_board')} expert {f.get('expert_end_board')} "
              f"best {f.get('best_of_k_end_board')} | second P lp {s2.get('lp_lost')} kill {s2.get('kill')} "
              f"field {s2.get('field')} | best-of-K lp {sb.get('lp_lost')} kill {sb.get('kill')} ({r.get('seconds')} s)",
              flush=True)  # fmt: skip
        a.out.write_text(json.dumps({"our_deck": str(a.our_deck), "ckpt": str(ckpt), "rows": rows}, indent=1))
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("ALERT combo_pool: no deck measured", flush=True)
        return 3
    if all(r["second_p"]["lp_lost"] == 0 and r["second_p"]["field"] == 0 for r in ok):
        print("ALERT combo_pool: going second measured zero for every deck (their turn 2 did nothing?)", flush=True)
        return 3
    firsts = [r["first"] for r in ok if r.get("first")]
    if firsts and all(not f.get("policy_end_board") for f in firsts):
        print("ALERT combo_pool: going first measured zero for every deck", flush=True)
        return 3
    if len(ok) < len(rows) or any(r["alerts"] for r in ok):
        return 4  # results written, but some decks need a look (ALERT lines above)
    return 0


if __name__ == "__main__":
    sys.exit(main())
