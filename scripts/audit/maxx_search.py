"""Why does the product's search fire Maxx "C" on its own turn? (user 2026-10-21: "why would search fire and overturn
the policy")

Replays search_match games (our seat) and joins each searched own-turn decision (log entry k = choice index) where a
Maxx "C" activation from hand is a candidate. Per decision it reads the candidates' per-world leaf values
(`samples`, value_p2 at the end of the turn, world s shared across candidates) and reports:
  - how often Maxx is a candidate, the search's best, and played over the policy's own move (an override);
  - the gap mean(Maxx) - mean(policy's move), its paired SE and z, for overrides and for all Maxx candidates;
  - the same gap for the other overrides (non-Maxx), as a reference for how strong a typical override is;
  - selection by noise: with 4 candidates and 4 samples each, how often the argmax would leave a candidate whose
    true value equals the others' (permutation of samples across candidates within a decision).

    taskset -c 0-3 uv run python scripts/audit/maxx_search.py --games runs/value-p2-20261015-2044/hppo-p2 \
        --out data/cache/audit/maxx_search-p2.json
Exit 2 on 0 searched Maxx decisions or a join mismatch (a logged k whose replayed actor is not our seat).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from principles import LOC_HAND  # noqa: E402

from decklight.decks import load_ydk  # noqa: E402
from decklight.engine.actions import Kind  # noqa: E402
from decklight.engine.cpp_env import CppEnv  # noqa: E402
from decklight.engine.duel import DuelConfig, Engine  # noqa: E402

MAXX = 23434538


def ydk(n: str) -> Path:
    return next(ROOT / d / f"{n}.ydk" for d in ("decks/gauntlet", "decks/pool", "decks/newcards")
                if (ROOT / d / f"{n}.ydk").exists())


def gap(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """(mean a - mean b, paired SE, z) over the shared worlds (sample s = world s for every candidate)."""
    n = min(len(a), len(b))
    d = np.array(a[:n]) - np.array(b[:n])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return m, se, (m / se if se > 0 else (float("inf") if m > 0 else 0.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", type=Path)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    engine = Engine()
    rows = [x for d in a.games for f in sorted(d.glob("games-*.jsonl")) for x in f.read_text().splitlines() if x]
    if a.max_games:
        rows = rows[: a.max_games]
    rng = random.Random(0)
    maxx_dec, other_ovr, examples = [], [], []
    n_own_searched = 0
    for x in rows:
        r = json.loads(x)
        me = int(r["s_seat"])
        log = {e["k"]: e for e in r["search"] if e["kind"] == "own"}
        if not log:
            continue
        cfg = DuelConfig(load_ydk(ydk(r["decks"][0])), load_ydk(ydk(r["decks"][1])), seed=int(r["seed"]))
        env = CppEnv(engine, cfg, True)
        obs = env.reset()
        for k, c in enumerate(r["choices"]):
            if obs is None:
                break
            e = log.get(k)
            if e is not None:
                if int(env.player) != me:
                    print(f"ALERT maxx_search: game {r['game_idx']} k {k}: replayed actor {env.player} != s_seat {me}")
                    return 2
                n_own_searched += 1
                acts, g = obs["actions"], obs["global"]
                cands, s = e["cands"], e["samples"]
                mx = [j for j, o in enumerate(cands) if int(acts[o, 0]) == Kind.ACTIVATE
                      and int(acts[o, 7]) == LOC_HAND and int(acts[o, 2]) == MAXX]
                jp = cands.index(e["played"]) if e["played"] in cands else None
                means = [float(np.mean(v)) for v in s]
                # noise-only argmax: shuffle all samples of this decision across candidates (equal true values)
                pool = [v for col in s for v in col]
                rng.shuffle(pool)
                sh = [float(np.mean(pool[i * len(s[0]):(i + 1) * len(s[0])])) for i in range(len(s))]
                noise_leave = int(np.argmax(sh)) != (jp if jp is not None else 0)
                if mx and jp is not None:
                    j = mx[0]
                    gm, gse, gz = gap(s[j], s[jp])
                    pol_kind = int(acts[e["played"], 0])
                    rec = {"turn": e["turn"], "phase": int(g[3]), "chosen_maxx": e["chosen"] == cands[j],
                           "played_maxx": e["played"] == cands[j], "rank": j, "gap": gm, "se": gse, "z": gz,
                           "spread": float(np.mean([np.std(v) for v in s])), "noise_leave": noise_leave,
                           "pol_kind": Kind(pol_kind).name, "n_cands": len(cands)}
                    maxx_dec.append(rec)
                    if rec["chosen_maxx"] and not rec["played_maxx"] and len(examples) < a.examples:
                        examples.append({"game": f"{r['game_idx']}", "k": k, **rec,
                                         "means": [round(m, 3) for m in means], "samples": s})
                elif jp is not None and e["chosen"] != e["played"]:
                    jc = cands.index(e["chosen"])
                    other_ovr.append(gap(s[jc], s[jp]))
            obs = env.step(int(c))
    if not maxx_dec:
        print("ALERT maxx_search: 0 searched own-turn decisions with a Maxx candidate")
        return 2

    def summ(recs: list[dict]) -> dict:
        if not recs:
            return {"n": 0}
        z = np.array([q["z"] for q in recs])
        return {"n": len(recs), "gap_mean": round(float(np.mean([q["gap"] for q in recs])), 4),
                "gap_median": round(float(np.median([q["gap"] for q in recs])), 4),
                "se_median": round(float(np.median([q["se"] for q in recs])), 4),
                "z_median": round(float(np.median(z[np.isfinite(z)])), 2) if np.isfinite(z).any() else None,
                "share_z_ge_2": round(float(np.mean(z >= 2)), 3), "share_z_ge_3": round(float(np.mean(z >= 3)), 3),
                "share_gap_ge_0.05": round(float(np.mean([q["gap"] >= 0.05 for q in recs])), 3)}

    ovr = [q for q in maxx_dec if q["chosen_maxx"] and not q["played_maxx"]]
    out = {
        "games": len(rows), "own_searched": n_own_searched,
        "maxx_candidate": len(maxx_dec),
        "maxx_is_policy_move": sum(q["played_maxx"] for q in maxx_dec),
        "maxx_chosen": sum(q["chosen_maxx"] for q in maxx_dec),
        "maxx_override": len(ovr),
        "maxx_override_drawstandby": sum(q["phase"] in (0x1, 0x2) for q in ovr),
        "maxx_kept_by_search": sum(q["played_maxx"] and q["chosen_maxx"] for q in maxx_dec),
        "maxx_cand_rank_hist": {str(r_): sum(q["rank"] == r_ for q in maxx_dec) for r_ in range(4)},
        "override_policy_move_kind": {kd: sum(q["pol_kind"] == kd for q in ovr) for kd in {q["pol_kind"] for q in ovr}},
        "maxx_overrides": summ(ovr),
        "maxx_all_candidates_vs_policy": summ([q for q in maxx_dec if not q["played_maxx"]]),
        "other_overrides": summ([{"gap": g_, "se": s_, "z": z_} for g_, s_, z_ in other_ovr]),
        "noise_only_leave_rate_maxx_decisions": round(float(np.mean([q["noise_leave"] for q in maxx_dec])), 3),
        "examples": examples,
    }
    txt = json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1)
    print(txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
