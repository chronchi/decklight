"""Belief shards: "which of the opponent's cards are in its hidden zones" (PLAN §0a "Hidden-information bound" (b)).

Replays records (config + choices) in the C++ env and keeps every stride-th decision (seeded offset per game) where the
actor faces >= 1 hidden opponent card. Per kept state, from the actor's view:
- the policy obs (OBS_KEYS), card memory (card_known, known_code) and the long salient history cut to --hist entries
  (hist_long / hist_long_mask), so one shard serves the no-history, v1-window and long-history models;
- cand [K]: distinct codes of the opponent's main deck (0-padded; K = MAX_CAND), cand_n [K]: its copies in the list;
- y [K] uint8: 1 if a hidden opponent card (hand / face-down field, not remembered via card_known == 1) has that code
  (labels from priv_code, the true identities CppEnv exposes);
- base_p [K] float32: the uniform-belief baseline the head must beat, P(c in hidden) under a hypergeometric draw of the
  h hidden cards from the unseen part of the list (list minus every opponent-owned card the actor can see or
  remembers), i.e. what today's determinizer samples from;
- n_hidden, turn, game (global int id), deck_me / deck_opp (stems, for held-out-deck splits), src.

    taskset -c 0-3 uv run python scripts/build_belief_shards.py --records 'data/demos/teacher/*.jsonl' \\
        'data/demos/sw-t1/*.jsonl' --out data/cache/belief_shards --stride 4 --hist 128 --jobs 4

Engine only (no torch): fine on the laptop. Exits 1 if no state was kept or any label is set on a non-candidate.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ygo_zero.model.bc_data import KNOWN_KEYS, OBS_KEYS

MAX_CAND = 64
LOC_HAND, LOC_MZONE, LOC_SZONE, LOC_EXTRA = 2, 3, 4, 7
_ENGINE = None


def _init() -> None:
    global _ENGINE
    from ygo_zero.engine.duel import Engine

    _ENGINE = Engine()


def _offset(game_id: str, stride: int) -> int:
    seed = hashlib.md5(game_id.encode()).hexdigest()
    return random.Random(seed).randrange(max(stride, 1))


def hidden_rows(obs: dict) -> np.ndarray:
    """Opponent rows whose identity the actor does not know: hand / monster / spell-trap zones, no visible code, not
    remembered (card_known != 1)."""
    c = obs["cards"]
    known = obs.get("card_known")
    hid = obs["card_mask"] & (c[:, 3] == 1) & (c[:, 0] == 0) & np.isin(c[:, 1], (LOC_HAND, LOC_MZONE, LOC_SZONE))
    if known is not None:
        hid &= known != 1
    return hid


def seen_opponent_codes(obs: dict) -> Counter:
    """Opponent-owned cards the actor can identify now: visible rows owned by the opponent (outside the extra deck)
    plus remembered hidden cards (known_code where card_known == 1)."""
    c = obs["cards"]
    vis = obs["card_mask"] & (c[:, 0] > 0) & (c[:, 4] == 1) & (c[:, 1] != LOC_EXTRA)
    out = Counter(int(x) for x in c[vis, 0])
    known = obs.get("card_known")
    if known is not None:
        out.update(int(x) for x in obs["known_code"][known == 1] if x > 0)
    return out


def hypergeom_in(N: int, n_c: int, h: int) -> float:
    """P(at least one of n_c copies among h draws without replacement from N)."""
    if n_c <= 0 or h <= 0:
        return 0.0
    if h > N - n_c:
        return 1.0
    # 1 - C(N - n_c, h) / C(N, h), in logs
    lg = math.lgamma
    log_none = (lg(N - n_c + 1) - lg(N - n_c - h + 1)) - (lg(N + 1) - lg(N - h + 1))
    return 1.0 - math.exp(log_none)


def belief_target(obs: dict, opp_main: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """(cand, cand_n, y, base_p, n_hidden, n_unlabelled) for one state; cand padded to MAX_CAND."""
    lst = Counter(int(x) for x in opp_main if x)
    codes = sorted(lst)[:MAX_CAND]
    hid = hidden_rows(obs)
    h = int(hid.sum())
    truth = Counter(int(x) for x in obs["priv_code"][hid] if x > 0)
    seen = seen_opponent_codes(obs)
    pool = {c: max(0, lst[c] - seen.get(c, 0)) for c in codes}
    N = sum(pool.values())
    cand = np.zeros(MAX_CAND, np.int32)
    cand_n = np.zeros(MAX_CAND, np.int8)
    y = np.zeros(MAX_CAND, np.uint8)
    base = np.zeros(MAX_CAND, np.float32)
    for i, c in enumerate(codes):
        cand[i], cand_n[i] = c, min(lst[c], 127)
        y[i] = 1 if truth.get(c, 0) > 0 else 0
        base[i] = hypergeom_in(N, pool[c], h)
    unlabelled = sum(v for c, v in truth.items() if c not in pool)  # extra-deck / token / non-list hidden cards
    return cand, cand_n, y, base, h, unlabelled


def _shard_one(job: tuple) -> tuple[str, int, int, int]:
    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.cpp_env import CppEnv

    src_file, out_file, src_tag, rec_ids, game_ids, game_strs, stride, hist = job
    recs = [json.loads(line) for line in Path(src_file).read_text().splitlines() if line.strip()]
    cols: dict[str, list] = {k: [] for k in (*OBS_KEYS, *KNOWN_KEYS, "hist_long", "hist_long_mask")}
    extra: dict[str, list] = {k: [] for k in ("cand", "cand_n", "y", "base_p", "n_hidden", "turn", "game",
                                                 "deck_me", "deck_opp", "actor")}
    n_games = unlab = 0
    for rec, gid, gstr in zip((recs[i] for i in rec_ids), game_ids, game_strs, strict=True):
        try:
            cfg = record_config(rec)
            env = CppEnv(_ENGINE, cfg, rec.get("auto_single", True))
            obs = env.reset()
            off = _offset(str(gstr), stride)
            decks = rec.get("decks") or ["", ""]
            mains = (list(cfg.deck0.main), list(cfg.deck1.main))
            kept = 0
            for k, c in enumerate(rec["choices"]):
                if obs is None:
                    break
                me = int(env.player)
                if (k + off) % stride == 0 and int(obs["action_mask"].sum()) >= 2 and hidden_rows(obs).any():
                    cand, cand_n, y, base, h, ul = belief_target(obs, mains[1 - me])
                    unlab += ul
                    for key in (*OBS_KEYS, *KNOWN_KEYS):
                        cols[key].append(obs[key].copy())
                    cols["hist_long"].append(obs["hist_long"][:hist].copy())
                    cols["hist_long_mask"].append(obs["hist_long_mask"][:hist].copy())
                    for key, v in (("cand", cand), ("cand_n", cand_n), ("y", y), ("base_p", base), ("n_hidden", h),
                                   ("turn", int(obs["global"][2])), ("game", gid),
                                   ("deck_me", str(decks[me]) if me < len(decks) else ""),
                                   ("deck_opp", str(decks[1 - me]) if 1 - me < len(decks) else ""), ("actor", me)):
                        extra[key].append(v)
                    kept += 1
                obs = env.step(int(c))
            n_games += kept > 0
        except Exception as e:  # noqa: BLE001 - one bad record must not kill a shard build
            print(f"belief_shards: skipping {gstr} in {src_file}: {e!r}"[:300], flush=True)
    n = len(extra["y"])
    if not n:
        return out_file, n_games, 0, unlab
    np.savez_compressed(
        out_file,
        **{k: np.stack(v) for k, v in cols.items()},
        cand=np.stack(extra["cand"]), cand_n=np.stack(extra["cand_n"]), y=np.stack(extra["y"]),
        base_p=np.stack(extra["base_p"]), n_hidden=np.array(extra["n_hidden"], np.int16),
        turn=np.array(extra["turn"], np.int16), game=np.array(extra["game"], np.int32),
        deck_me=np.array(extra["deck_me"]), deck_opp=np.array(extra["deck_opp"]),
        actor=np.array(extra["actor"], np.int8), src=np.array([src_tag] * n),
    )  # fmt: skip
    return out_file, n_games, n, unlab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="record file globs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--hist", type=int, default=128, help="hist_long entries kept per state")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    files = [f for f in sorted({f for g in args.records for f in glob.glob(g)}) if Path(f).stat().st_size]
    args.out.mkdir(parents=True, exist_ok=True)
    work, gid = [], 0
    for f in files:
        recs = [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
        # full duels only: setup / opening records place cards by hand, so their hidden zones are not a draw
        sel = [i for i, r in enumerate(recs) if "config" in r and not r["config"].get("setup") and not r.get("opening")]
        if args.max_games is not None:
            sel = sel[: max(0, args.max_games - gid)]
        if not sel:
            continue
        tag = Path(f).parent.name
        ids = list(range(gid, gid + len(sel)))
        strs = [f"{tag}/{recs[i].get('game_id', recs[i].get('line_id', i))}" for i in sel]
        gid += len(sel)
        work.append((f, str(args.out / f"{tag}__{Path(f).stem}.npz"), tag, sel, ids, strs, args.stride, args.hist))
        if args.max_games is not None and gid >= args.max_games:
            break
    print(f"{len(work)} files, {gid} games", flush=True)
    tg = ts = tu = 0
    with mp.get_context("fork").Pool(max(1, args.jobs), initializer=_init) as pool:
        for out, g, s, u in pool.imap_unordered(_shard_one, work):
            tg, ts, tu = tg + g, ts + s, tu + u
    print(f"games {tg} states {ts} hidden cards outside the candidate list {tu}", flush=True)
    if ts == 0:
        print("ALERT build_belief_shards: 0 states kept", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
