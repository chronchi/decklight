"""Leaf choice, offline (retrain plan-v2 §Leaf choice): which value net predicts game outcomes better on held-out games?

Replays search_match game rows (games-*.jsonl: deck names, seed, choices, winner) and keeps both players' decisions:
every turn-start state (the first decision of a turn: the kind of state an h1 leaf is) plus every --stride-th other
decision. Target z = +1 when the acting player won, -1 when it lost; drawn/cut games are skipped. For each value net:
MSE, ECE (10 bins of p = (v + 1) / 2), split all / turn-start, and own turn vs opponent's turn. ΔMSE between the first
two nets has a game-level bootstrap SE (states of one game are not independent).

Pre-registered rule (plan-v2): on turn-start states, the lower MSE wins unless its ECE is worse by > 0.02;
|ΔMSE| < 1 SE = inconclusive. The verdict is the last line: `leaf: <name>` or `leaf: inconclusive`.
Exit 2 on 0 states, a non-finite value or two nets with identical values.

    uv run python scripts/audit/leaf_compare.py --games runs/<job>/p --value new=data/cache/value_p/best.pt \\
        --value ppo3=data/cache/value_ppo3/best.pt --device cuda --out runs/<job>/leaf.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.model.bc_data import OBS_KEYS  # noqa: E402


def game_rows(dirs: list[Path], max_games: int, seed: int):
    rows = [json.loads(x) for d in dirs for f in sorted(d.glob("games-*.jsonl")) for x in f.read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    return rows[:max_games] if max_games else rows


def deck_paths(names: list[str]) -> list[Path]:
    ddir = ROOT / ("decks/gauntlet" if (ROOT / "decks/gauntlet" / f"{names[0]}.ydk").exists() else "decks/pool")
    return [ddir / f"{n}.ydk" for n in names]


def states(rows, stride: int, seed: int):
    """Yields (obs, z, turn_start, own_turn, game index) for both players' decisions."""
    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig, Engine

    engine = Engine()
    rng = random.Random(seed)
    for gi, r in enumerate(rows):
        w = r.get("winner")
        if w not in (0, 1) or r.get("truncated"):
            continue
        d0, d1 = deck_paths(r["decks"])  # seat order, as scripts/audit/principles.py iter_games reads them
        env = CppEnv(engine, DuelConfig(load_ydk(d0), load_ydk(d1), seed=int(r["seed"])), True)
        obs = env.reset()
        off, last_turn = rng.randrange(stride), None
        for k, c in enumerate(r["choices"]):
            if obs is None:
                break
            g = obs["global"]
            turn = int(g[2])
            ts = turn != last_turn
            last_turn = turn
            if ts or k % stride == off:
                z = 1.0 if int(env.player) == w else -1.0
                yield {kk: np.array(obs[kk]) for kk in OBS_KEYS}, z, ts, int(g[5]) == 1, gi
            obs = env.step(int(c))


def metrics(v: np.ndarray, z: np.ndarray) -> dict:
    p, win = np.clip((v + 1) / 2, 0, 1), z > 0
    b = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[b == i].mean() - win[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
    return {"n": int(len(v)), "mse": round(float(((v - z) ** 2).mean()), 5), "ece": round(float(ece), 4),
            "acc": round(float(((v >= 0) == win).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", type=Path, required=True)
    ap.add_argument("--value", action="append", required=True, help="name=checkpoint (the first two are compared)")
    ap.add_argument("--emb", default=str(ROOT / "data/cards/emb/qwen3-0.6b-d512.npz"))
    ap.add_argument("--max-games", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from decklight.eval.value import ValueNet

    nets = {}
    for spec in a.value:
        name, path = spec.split("=", 1)
        nets[name] = ValueNet(path, a.emb, device=a.device, batch=1024, threads=4)
    t0 = time.time()
    obs_buf, zs, tss, owns, gis = [], [], [], [], []
    vals = {n: [] for n in nets}

    def flush() -> None:
        if not obs_buf:
            return
        batch = {k: np.stack([o[k] for o in obs_buf]) for k in OBS_KEYS}
        for n, net in nets.items():
            vals[n].append(net.values(batch))
        obs_buf.clear()

    for o, z, ts, own, gi in states(game_rows(a.games, a.max_games, a.seed), a.stride, a.seed):
        obs_buf.append(o)
        zs.append(z)
        tss.append(ts)
        owns.append(own)
        gis.append(gi)
        if len(obs_buf) >= 4096:
            flush()
    flush()
    if not zs:
        print("ALERT leaf_compare: 0 states")
        return 2
    z, ts, own, gi = np.array(zs), np.array(tss), np.array(owns), np.array(gis)
    v = {n: np.concatenate(x) for n, x in vals.items()}
    if any(not np.isfinite(x).all() for x in v.values()):
        print("ALERT leaf_compare: non-finite value")
        return 2
    out = {"games": int(len(set(gi.tolist()))), "states": int(len(z)), "seconds": round(time.time() - t0)}
    for n, x in v.items():
        out[n] = {"all": metrics(x, z), "turn_start": metrics(x[ts], z[ts]),
                  "turn_start_own": metrics(x[ts & own], z[ts & own]),
                  "turn_start_opp": metrics(x[ts & ~own], z[ts & ~own])}
    na, nb = list(v)[:2]
    if np.array_equal(v[na], v[nb]):
        print(f"ALERT leaf_compare: {na} and {nb} give identical values (same weights?)")
        return 2
    d = (v[na] - z) ** 2 - (v[nb] - z) ** 2  # < 0: na better
    m = ts
    games = np.unique(gi[m])
    by_game = {g: (d[m][gi[m] == g].sum(), (gi[m] == g).sum()) for g in games}
    rng = np.random.default_rng(a.seed)
    boots = []
    for _ in range(300):
        s = rng.choice(games, len(games))
        num = sum(by_game[g][0] for g in s)
        den = sum(by_game[g][1] for g in s)
        boots.append(num / den)
    dmse, se = float(d[m].mean()), float(np.std(boots))
    ea, eb = out[na]["turn_start"]["ece"], out[nb]["turn_start"]["ece"]
    if abs(dmse) <= se:
        verdict = "inconclusive"
    else:
        better, worse = (na, nb) if dmse < 0 else (nb, na)
        e_b, e_w = (ea, eb) if dmse < 0 else (eb, ea)
        verdict = better if e_b - e_w <= 0.02 else worse
    out["turn_start_dmse"] = {"pair": f"{na} - {nb}", "dmse": round(dmse, 5), "se": round(se, 5), "verdict": verdict}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for n in v:
        print(f"{n}: all {out[n]['all']}; turn-start {out[n]['turn_start']}; own {out[n]['turn_start_own']['mse']} "
              f"opp {out[n]['turn_start_opp']['mse']}")
    print(f"turn-start dMSE {na} - {nb} = {dmse:+.5f} (SE {se:.5f}), {out['states']} states, {out['games']} games")
    print(f"leaf: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
