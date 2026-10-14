"""Where does the Maxx "C" own-turn habit come from? (user 2026-10-14: "run the two cheap tests please"; NOTEBOOK 18:00)

A window = an own-turn decision with no opponent chain link where activating Maxx "C" (23434538) from hand is legal.
Per player-game (one player in one game with >= 1 window) it reports windows, fires, the share with >= 1 fire, split by
T1 / later turns and Draw-Standby / other phases. Per-game counts avoid the per-window denominator trap (a policy that
passes is offered the next window again; NOTEBOOK 2026-10-13 19:00).

With --ckpt it also scores every window with that policy: p = softmax mass on the Maxx option(s) over legal options.
  - greedy rate = share of windows where Maxx is the argmax; sampled rate = mean p (same states);
  - per game along the observed trajectory, up to the first fire: greedy fires or not vs 1 - prod(1 - p) for sampling;
  - agree = share of windows where the policy's argmax equals the move played (≈ 1 on the policy's own argmax games:
    a sanity check that the replay and the scoring line up).

    taskset -c 0-3 uv run python scripts/audit/maxx_habit.py --games runs/<job>/raw --seat s --ckpt data/cache/P/best.pt
    taskset -c 0-3 uv run python scripts/audit/maxx_habit.py --records 'data/demos/teacher/*.jsonl' --max-games 1500
Exit 2 on 0 windows or (with --ckpt, --seat s) agree < 0.9 on the policy's own games.
"""

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from principles import LOC_HAND  # noqa: E402

from ygo_zero.engine.actions import Kind  # noqa: E402
from ygo_zero.eval.loopguard import state_key  # noqa: E402
from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import DuelConfig, Engine  # noqa: E402
from ygo_zero.model.bc_data import OBS_KEYS  # noqa: E402

MAXX = 23434538


def game_iter(dirs: list[Path], engine: Engine, seat: str, max_games: int):
    """(game id, player, obs, choice) for one seat of search_match games (s = our pilot, ya = the other seat)."""
    from ygo_zero.decks import load_ydk

    rows = [(f.name, x) for d in dirs for f in sorted(d.glob("games-*.jsonl")) for x in f.read_text().splitlines()]
    random.Random(0).shuffle(rows)
    for fname, x in rows[: max_games or None]:
        r = json.loads(x)
        me = int(r["s_seat"]) if seat == "s" else 1 - int(r["s_seat"])
        ddir = ROOT / ("decks/gauntlet" if (ROOT / "decks/gauntlet" / f"{r['decks'][0]}.ydk").exists() else "decks/pool")
        cfg = DuelConfig(load_ydk(ddir / f"{r['decks'][0]}.ydk"), load_ydk(ddir / f"{r['decks'][1]}.ydk"),
                         seed=int(r["seed"]))
        env = CppEnv(engine, cfg, True)
        obs = env.reset()
        for c in r["choices"]:
            if obs is None:
                break
            if int(env.player) == me:
                yield f"{fname}:{r['game_idx']}", me, obs, int(c)
            obs = env.step(int(c))


def record_iter(pattern: str, engine: Engine, max_games: int):
    """Teacher records (ygo-agent vs ygo-agent): both players' decisions; game id tagged sampled / argmax."""
    from ygo_zero.demos.dataset import record_config

    files = sorted(glob.glob(pattern, recursive=True))
    random.Random(0).shuffle(files)
    n = 0
    for f in files:
        for raw in open(f):
            r = json.loads(raw)
            if "config" not in r:
                continue
            env = CppEnv(engine, record_config(r, r["config"]["seed"]), bool(r.get("auto_single", True)))
            obs = env.reset()
            tag = "sampled" if "sampled" in r.get("source", "") else "argmax"
            for c in r["choices"]:
                if obs is None:
                    break
                yield f"{tag}|{r['line_id']}", int(env.player), obs, int(c)
                obs = env.step(int(c))
            n += 1
            if max_games and n >= max_games:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="*", type=Path, default=[])
    ap.add_argument("--seat", choices=("s", "ya"), default="s")
    ap.add_argument("--records", default="")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--emb", default=str(ROOT / "data/cards/emb/qwen3-0.6b-d512.npz"))
    ap.add_argument("--name", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    engine = Engine()
    pol = None
    if a.ckpt:
        from ygo_zero.eval.bc_policy import BCPolicy

        pol = BCPolicy(a.ckpt, a.emb, threads=1, loop_guard=False)
    src = record_iter(a.records, engine, a.max_games) if a.records else game_iter(a.games, engine, a.seat, a.max_games)
    # per (game, player): list of windows (t1, drawstandby, fired, obs-or-None, legal maxx idx, choice)
    pg: dict = defaultdict(list)
    buf, refs = [], []
    probs: dict = {}

    def flush() -> None:
        if not buf:
            return
        lg = pol.logits({k: np.stack([o[k] for o in buf]) for k in OBS_KEYS})
        for (key, j, mx, legal, ch, taken), z in zip(refs, lg):
            z = z[: len(legal)].astype(np.float64)
            z[~legal] = -np.inf
            p = np.exp(z - z.max())
            p /= p.sum()
            zg = z.copy()
            zg[[t for t in taken if t < len(zg)]] = -np.inf
            if not np.isfinite(zg).any():
                zg = z
            ag = int(np.argmax(zg))
            # (sampled p, greedy fires, agree unguarded, guarded greedy fires, agree guarded, guard forced Maxx)
            probs[(key, j)] = (float(p[mx].sum()), int(np.argmax(z)) in mx, int(np.argmax(z)) == ch, ag in mx,
                               ag == ch, ag in mx and int(np.argmax(z)) not in mx)
        buf.clear()
        refs.clear()

    seen: dict = {}  # (game, player) -> (turn, {state key: options taken}): the LoopGuard the player's play went through
    for gid, player, obs, ch in src:
        g, acts, mask = obs["global"], obs["actions"], obs["action_mask"].astype(bool)
        gk = (gid, player)
        if gk not in seen or seen[gk][0] != int(g[2]):
            seen[gk] = (int(g[2]), {})
        sk = state_key(obs)
        taken = set(seen[gk][1].get(sk, set()))
        seen[gk][1].setdefault(sk, set()).add(ch)
        if int(g[5]) != 1:
            continue
        cm = obs["chain_mask"].astype(bool)
        if cm.any() and bool((obs["chain"][cm][:, 4] == 1).any()):
            continue
        mx = [i for i in range(len(mask)) if mask[i] and int(acts[i, 0]) == Kind.ACTIVATE
              and int(acts[i, 7]) == LOC_HAND and int(acts[i, 2]) == MAXX]
        if not mx:
            continue
        key = (gid, player)
        pg[key].append((int(g[2]) == 1, int(g[3]) in (0x1, 0x2), ch in mx, bool(taken)))
        if pol is not None:
            buf.append({k: np.array(obs[k]) for k in OBS_KEYS})
            refs.append((key, len(pg[key]) - 1, mx, mask.copy(), ch, taken))
            if len(buf) >= 256:
                flush()
    if pol is not None:
        flush()
    if not pg:
        print("ALERT maxx_habit: 0 windows")
        return 2

    out = {"name": a.name}
    groups = {"all": list(pg)}
    if a.records:
        groups = {t: [k for k in pg if k[0].startswith(t)] for t in ("argmax", "sampled")}
    for gname, keys in groups.items():
        if not keys:
            continue
        w = [pg[k] for k in keys]
        res = {"player_games": len(keys),
               "windows_per_game": round(float(np.mean([len(x) for x in w])), 2),
               "fires_per_game": round(float(np.mean([sum(f for _, _, f, _ in x) for x in w])), 3),
               "share_games_fired": round(float(np.mean([any(f for _, _, f, _ in x) for x in w])), 3),
               "share_games_fired_t1": round(float(np.mean([any(f and t1 for t1, _, f, _ in x) for x in w])), 3),
               "share_games_fired_drawstandby": round(float(np.mean([any(f and ds for _, ds, f, _ in x) for x in w])), 3),
               "per_window_rate": round(sum(f for x in w for _, _, f, _ in x) / sum(len(x) for x in w), 4),
               "windows_repeated_state": round(sum(r for x in w for *_, r in x) / sum(len(x) for x in w), 4),
               "fires_at_repeated_state": round(sum(f and r for x in w for _, _, f, r in x)
                                                / max(1, sum(f for x in w for _, _, f, _ in x)), 4)}
        if pol is not None:
            pr = [[probs[(k, j)] for j in range(len(pg[k]))] for k in keys]
            flat = [q for x in pr for q in x]
            res["model_sampled_rate"] = round(float(np.mean([q[0] for q in flat])), 4)
            res["model_greedy_rate"] = round(float(np.mean([q[1] for q in flat])), 4)
            res["agree"] = round(float(np.mean([q[2] for q in flat])), 4)
            res["model_guarded_greedy_rate"] = round(float(np.mean([q[3] for q in flat])), 4)
            res["agree_guarded"] = round(float(np.mean([q[4] for q in flat])), 4)
            res["guard_forced_maxx"] = int(sum(q[5] for q in flat))
            res["observed_fires"] = int(sum(f for x in w for _, _, f, _ in x))
            # per game along the observed trajectory up to (and including) the first observed fire
            g_greedy, g_sampled = [], []
            for k, x in zip(keys, pr):
                fires = [f for _, _, f, _ in pg[k]]
                n = fires.index(True) + 1 if True in fires else len(fires)
                g_greedy.append(any(q[1] for q in x[:n]))
                g_sampled.append(1 - float(np.prod([1 - q[0] for q in x[:n]])))
            res["model_games_fire_greedy"] = round(float(np.mean(g_greedy)), 3)
            res["model_games_fire_sampled"] = round(float(np.mean(g_sampled)), 3)
            for t1 in (True, False):
                sel = [q for k in keys for (tt, *_), q in zip(pg[k], probs_list(probs, k, len(pg[k]))) if tt == t1]
                if sel:
                    res[f"model_{'t1' if t1 else 't2p'}_sampled_rate"] = round(float(np.mean([q[0] for q in sel])), 4)
                    res[f"model_{'t1' if t1 else 't2p'}_greedy_rate"] = round(float(np.mean([q[1] for q in sel])), 4)
            # where the mass sits: histogram of p at the windows
            ps = np.array([q[0] for q in flat])
            res["p_hist"] = {f"{lo:.1f}-{lo + 0.1:.1f}": int(((ps >= lo) & (ps < lo + 0.1 + (lo >= 0.9))).sum())
                             for lo in np.arange(0, 1, 0.1)}
        out[gname] = res
        print(gname, json.dumps(res))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    if pol is not None and a.games and a.seat == "s" and out.get("all", {}).get("agree_guarded", 1) < 0.9:
        print(f"ALERT maxx_habit: agree_guarded {out['all']['agree_guarded']} < 0.9 on the policy's own games (replay/scoring mismatch?)")
        return 2
    return 0


def probs_list(probs: dict, key, n: int) -> list:
    return [probs[(key, j)] for j in range(n)]


if __name__ == "__main__":
    sys.exit(main())
