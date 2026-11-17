#!/usr/bin/env python3
"""P vs tsoj's bot as a yardstick: win rate per records file, with the bot's node count, and the node choice.

    uv run python scripts/tsoj/gauntlet_summary.py LABEL=RECORDS.jsonl[:WALL_S] ... [--out summary.json] [--choose]

LABEL is the bot's node count of that file (e.g. 500=runs/.../n500.jsonl:1234). WALL_S is the run's wall time, for
games per hour. Per file: P's win rate with a 95% bootstrap CI over openings (both games of an opening and its
mirror line are one cluster: same seed, same deck pair), split by P's deck and by whether P went first (seat 0
goes first: NOTEBOOK 2026-11-07 gauntlet), median seconds per game of each side (`result.time`), and games per hour.
`--choose` prints the pre-registered node count N* (NOTEBOOK 2026-11-07 gauntlet pre-registration): the smallest
label with P's win rate <= 0.40, else the largest label. `--paired BASE` (tsoj PPO eval): for every other label,
win(label) - win(BASE) over the games both files played (same seed, decks and seat order), with a cluster bootstrap
95% CI (opening + its mirror = one cluster) and better/worse counts per game. Exit 3 on a file with 0 good games.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def bad(rec: dict) -> bool:
    reason = (rec.get("result") or {}).get("reason")
    return isinstance(reason, str) and reason.startswith(("bot_error", "illegal"))


def cluster_ci(clusters: dict, n_boot: int = 4000, seed: int = 0) -> list[float] | None:
    keys = list(clusters)
    if not keys:
        return None
    s = np.array([sum(clusters[k]) for k in keys], float)
    n = np.array([len(clusters[k]) for k in keys], float)
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def games_by_key(path: Path) -> dict:
    """P's result per game, keyed by (seed, deck names, seat order); bad games left out."""
    out = {}
    for x in path.read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        if bad(r):
            continue
        ps = r["agents"].index("dl")
        w = (r.get("result") or {}).get("winner")
        out[(int(r["seed"]), tuple(d.get("name") for d in r["decks"]), tuple(r["agents"]))] = \
            1.0 if w == ps else 0.5 if w not in (0, 1) else 0.0
    return out


def summarize(path: Path, wall: float | None) -> dict:
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    good = [r for r in recs if not bad(r)]
    wins, clusters, by_deck, by_first = [], defaultdict(list), defaultdict(list), defaultdict(list)
    t_bot, t_p = [], []
    for r in good:
        ps = r["agents"].index("dl")
        w = (r.get("result") or {}).get("winner")
        x = 1.0 if w == ps else 0.5 if w not in (0, 1) else 0.0
        wins.append(x)
        names = [d.get("name") for d in r["decks"]]
        clusters[(int(r["seed"]), tuple(sorted(names)))].append(x)
        by_deck[names[ps]].append(x)
        by_first["first" if ps == 0 else "second"].append(x)
        t = (r.get("result") or {}).get("time") or []
        if len(t) == 2:
            t_bot.append(t[1 - ps])
            t_p.append(t[ps])
    mean = lambda v: round(sum(v) / len(v), 4) if v else None
    return {
        "file": str(path), "games": len(recs), "bad": len(recs) - len(good), "p_win": mean(wins),
        "ci95": cluster_ci(clusters), "openings": len(clusters),
        "by_first": {k: {"n": len(v), "p_win": mean(v)} for k, v in sorted(by_first.items())},
        "by_deck": {k: {"n": len(v), "p_win": mean(v)} for k, v in sorted(by_deck.items())},
        "bot_s_per_game": round(statistics.median(t_bot), 1) if t_bot else None,
        "p_s_per_game": round(statistics.median(t_p), 1) if t_p else None,
        "games_per_hour": round(len(recs) * 3600 / wall, 1) if wall else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--paired", default=None, help="label to pair every other label against")
    a = ap.parse_args()
    out, rc, paths = {}, 0, {}
    for spec in a.files:
        label, _, rest = spec.partition("=")
        path, _, wall = rest.rpartition(":") if rest.count(":") and rest.rsplit(":", 1)[1].isdigit() else (rest, "", "")
        paths[label] = path
        s = summarize(Path(path), float(wall) if wall else None)
        out[label] = s
        if not s["games"] - s["bad"]:
            print(f"ALERT gauntlet_summary: {path}: 0 good games", flush=True)
            rc = 3
        print(json.dumps({label: {k: s[k] for k in ("games", "bad", "p_win", "ci95", "by_first", "bot_s_per_game",
                                                     "p_s_per_game", "games_per_hour")}}), flush=True)
    if a.choose:
        ok = sorted((int(k) for k, s in out.items() if s["p_win"] is not None and s["p_win"] <= 0.40))
        out["chosen"] = ok[0] if ok else max(int(k) for k in out)
        print(f"N*={out['chosen']}", flush=True)
    if a.paired:
        out["paired"] = {}
        base = games_by_key(Path(paths[a.paired]))
        for label, path in paths.items():
            if label == a.paired:
                continue
            other = games_by_key(Path(path))
            keys = sorted(set(base) & set(other))
            clusters: dict = defaultdict(list)
            better = worse = 0
            for k in keys:
                d = other[k] - base[k]
                clusters[(k[0], tuple(sorted(k[1])))].append(d)
                better += d > 0
                worse += d < 0
            diffs = [x for v in clusters.values() for x in v]
            res = {"pairs": len(keys), "diff": round(sum(diffs) / len(diffs), 4) if diffs else None,
                   "ci95": cluster_ci(clusters), "better": better, "worse": worse}
            out["paired"][label] = res
            print(json.dumps({"paired": {f"{label}-{a.paired}": res}}), flush=True)
            if not keys:
                print(f"ALERT gauntlet_summary: 0 common games {label} vs {a.paired}", flush=True)
                rc = 3
            elif len(keys) >= 100 and better == worse == 0:  # same result in every game: same weights?
                print(f"ALERT gauntlet_summary: {label} = {a.paired} in all {len(keys)} games (same checkpoint?)",
                      flush=True)
                rc = 3
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
