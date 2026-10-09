"""Play and SAVE full games on the gauntlet decks (why do we lose to ygo-agent? 2026-10-05).

    uv run python scripts/gauntlet_games.py --ckpt data/cache/rl_init/best.pt --out runs/<name> --device cuda

Matchups (all ordered deck pairs incl. mirrors, both seats per seed): bc-vs-ga (BC as A), ga-vs-ga, bc-vs-bc.
With `--ckpts tag=path ...` (RL checkpoint comparison, 2026-10-08): only <tag>-ga games per checkpoint on the
in-run eval's schedule (`--deck-dir decks/gauntlet --ring 4`: each deck vs its next 4, both seats, seeds 0..--seeds-1),
so games pair across checkpoints by (deck_a, deck_b, seed, a_first); seed 0 reproduces the train_rl eval exactly.
Each finished game -> <out>/games/<matchup>-<k>/state.json in hand_play's state format (decks as repo paths, so
`hand_play.py show/log` and `data/cache/dump_game.py` read it after copying into data/demos/gold_work/), with extra
fields `matchup`, `a_seat`, `winner`. Summary per game: <out>/games.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.bc_policy import BCPolicy
from ygo_zero.eval.ygoagent import YgoAgentPolicy

ROOT = Path(__file__).resolve().parents[1]


def play(engine, decks, names, a, b, seeds, tag, out: Path, threads: int, log, pairs=None) -> float:
    n = len(decks)
    pairs = pairs if pairs is not None else [(i, j) for i in range(n) for j in range(n)]
    sched, meta = [], []
    for i, j in pairs:
        for s in seeds:
            sched.append((i, j, s))  # A first
            meta.append((i, j, 0))
            sched.append((j, i, s))  # A second
            meta.append((i, j, 1))
    for p in (a, b):
        if hasattr(p, "_states"):
            p._states.clear()
    vec = VecEnv(engine, decks, min(64, len(sched)), threads, seed=0)
    vec.set_schedule(sched)
    vec.set_max_decisions(1500 if pairs is not None else 3000)  # winner 3 = cut; 1500 = the eval's cap
    buf = vec.reset()
    a_seat = np.array([m[2] for m in meta] + [-2], np.int32)
    rec: dict[int, dict] = {}
    k, wins = 0, 0.0
    while vec.remaining > 0:
        player, game = buf["player"].copy(), buf["game"].copy()
        live = np.flatnonzero(player >= 0)
        for i in live:  # the decision about to be taken: actor and turn
            r = rec.setdefault(int(game[i]), {"choices": [], "actors": [], "turns": []})
            r["actors"].append(int(player[i]))
            r["turns"].append(int(buf["global"][i, 2]))
        actions = np.zeros(vec.num_envs, np.int32)
        mine = live[player[live] == a_seat[game[live]]]
        theirs = live[player[live] != a_seat[game[live]]]
        if len(mine):
            actions[mine] = a(buf, mine)
        if len(theirs):
            actions[theirs] = b(buf, theirs)
        for i in live:
            rec[int(game[i])]["choices"].append(int(actions[i]))
        vec.step(actions)
        for ep in vec.pop_finished():
            r = rec.pop(int(ep.index))
            assert r["choices"] == list(ep.choices), "recorded choices differ from the env's"
            d0, d1, seed = sched[ep.index]
            da, db, seat = meta[ep.index]
            st = {"decks": [names[d0], names[d1]], "seed": int(seed), **r, "labels": ["gold"] * len(r["choices"]),
                  "notes": [], "matchup": tag, "a_seat": seat, "winner": int(ep.winner)}  # fmt: skip
            d = out / "games" / f"{tag}-{k:03d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "state.json").write_text(json.dumps(st))
            win = 0.5 if ep.winner not in (0, 1) else float(ep.winner == seat)
            log.write(json.dumps({"game": d.name, "matchup": tag, "deck_a": names[da].split("/")[-1],
                                  "deck_b": names[db].split("/")[-1], "seed": int(seed), "a_first": seat == 0,
                                  "a_win": win, "turns": int(ep.turns), "decisions": len(r["choices"])}) + "\n")  # fmt: skip
            log.flush()
            k += 1
            wins += win
    print(f"{tag}: {k} games, A wr {wins / max(k, 1):.3f}", flush=True)
    return wins / max(k, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3, help="bc-vs-ga seeds per ordered pair")
    ap.add_argument("--ref-seeds", type=int, default=1, help="ga-vs-ga / bc-vs-bc seeds per ordered pair")
    ap.add_argument("--ckpts", nargs="*", default=None, metavar="TAG=PATH",
                    help="compare checkpoints: <tag>-ga games each on the eval schedule (replaces --ckpt's matchups)")
    ap.add_argument("--deck-dir", type=Path, default=None, help="decks (default tests/fixtures/decks)")
    ap.add_argument("--ring", type=int, default=0, help="each deck vs its next k decks (0 = all ordered pairs)")
    ap.add_argument("--ga-workers", type=int, default=8)
    ap.add_argument("--expect", nargs="*", default=[], metavar="TAG=WR",
                    help="fail (exit 2) if a checkpoint's seed-0 win rate differs from its in-run eval")
    args = ap.parse_args()
    if args.ckpts:
        compare(args)
        return
    files = sorted((ROOT / "tests/fixtures/decks").glob("*.ydk"))
    decks = [load_ydk(f) for f in files]
    names = [str(f.relative_to(ROOT)) for f in files]
    engine = Engine()
    bc = BCPolicy(args.ckpt, args.emb, device=args.device)
    ga, ga2 = YgoAgentPolicy(seed=3), YgoAgentPolicy(seed=4)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "games.jsonl").open("a") as log:
        play(engine, decks, names, bc, ga, range(args.seeds), "bc-ga", args.out, args.threads, log)
        play(engine, decks, names, ga, ga2, range(100, 100 + args.ref_seeds), "ga-ga", args.out, args.threads, log)
        bc2 = BCPolicy(args.ckpt, args.emb, device=args.device)
        play(engine, decks, names, bc, bc2, range(200, 200 + args.ref_seeds), "bc-bc", args.out, args.threads, log)


def compare(args) -> None:
    from ygo_zero.eval.match import ring_pairs

    files = sorted((args.deck_dir or ROOT / "tests/fixtures/decks").glob("*.ydk"))
    decks = [load_ydk(f) for f in files]
    names = [str(f.relative_to(ROOT)) for f in files]
    pairs = ring_pairs(len(decks), args.ring) if args.ring > 0 else None
    engine = Engine()
    expect = {t: float(w) for t, w in (e.split("=", 1) for e in args.expect)}
    args.out.mkdir(parents=True, exist_ok=True)
    bad = []
    with (args.out / "games.jsonl").open("a") as log:
        for spec in args.ckpts:
            tag, path = spec.split("=", 1)
            pol = BCPolicy(Path(path), args.emb, device=args.device)
            if tag in expect:  # seed 0 alone = the in-run eval (same schedule, greedy policy, ygo-agent seed 3)
                ga = YgoAgentPolicy(seed=3, workers=args.ga_workers)
                wr0 = play(engine, decks, names, pol, ga, range(1), f"{tag}-s0", args.out, args.threads, log, pairs)
                if abs(wr0 - expect[tag]) > 0.0021:
                    bad.append(f"{tag}: seed-0 wr {wr0:.3f} != in-run eval {expect[tag]:.3f}")
                    print(f"ALERT {bad[-1]}", flush=True)
                seeds = range(1, args.seeds)
            else:
                seeds = range(args.seeds)
            ga = YgoAgentPolicy(seed=3, workers=args.ga_workers)
            play(engine, decks, names, pol, ga, seeds, f"{tag}-ga", args.out, args.threads, log, pairs)
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
