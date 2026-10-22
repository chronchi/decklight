"""Self-play games as BC records (kind `teacher`): ygo-agent 0546 on its own deck lists.

Each finished game becomes one JSON line in <out>/<deck0>-vs-<deck1>.jsonl, in the gold
record format (scripts/hand_play.py `save`), so the BC dataset/shard pipeline reads it
unchanged. Labels mirror gold: a PASS (kind 11) at a chain window is `pass`, every other
choice is `gold`.

    uv run python scripts/gen_teacher.py --out data/demos/teacher --games N [--decks decks/gauntlet]
        [--sample-frac 0.5] [--seed S] [--threads T] [--policy ygoagent|random]

Schedule: N games over all ordered deck pairs (incl. mirrors, cycled) with duel seeds
S, S+1, .... A deterministic fraction `--sample-frac` of the games is played by sampling
policies (diversity), the rest argmax; the two groups run as two separate VecEnv
schedules. Both seats are played by the policy: two separate policy objects, one per
seat (separate LSTM states). Games cut at the decision cap (winner == TRUNCATED) are
dropped. `--policy random` (tests only) uses RandomPolicy on both seats: the laptop
must never instantiate YgoAgentPolicy (RAM rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.cpp_env import VecEnv
from decklight.engine.duel import Engine

MAX_DECISIONS = 3000


def make_policies(policy: str, sample: bool, seed: int):
    """Two separate policy objects, one per seat (separate LSTM states)."""
    if policy == "random":
        from decklight.eval.policies import RandomPolicy

        return RandomPolicy(seed), RandomPolicy(seed + 1)
    from decklight.eval.ygoagent import YgoAgentPolicy

    return YgoAgentPolicy(seed=seed, sample=sample), YgoAgentPolicy(seed=seed + 1, sample=sample)


def run_schedule(engine, decks, stems, sched, sample, policy, seed, threads, out: Path,
                 tag: str, counters: dict) -> None:
    """Play one VecEnv schedule (all games with the same sample flag); append records."""
    if not sched:
        return
    pol0, pol1 = make_policies(policy, sample, seed)
    vec = VecEnv(engine, decks, min(64, len(sched)), threads, seed=seed)
    vec.set_schedule([(i, j, s) for i, j, s, _ in sched])
    vec.set_max_decisions(MAX_DECISIONS)
    buf = vec.reset()
    rec: dict[int, dict] = {}
    while vec.remaining > 0:
        player, game = buf["player"].copy(), buf["game"].copy()
        live = np.flatnonzero(player >= 0)
        for i in live:  # the decision about to be taken: actor, turn and prompt message
            r = rec.setdefault(int(game[i]), {"choices": [], "actors": [], "turns": [], "msgs": [],
                                              "kinds": []})
            r["actors"].append(int(player[i]))
            r["turns"].append(int(buf["global"][i, 2]))
            r["msgs"].append(int(buf["global"][i, 21]))
        actions = np.zeros(vec.num_envs, np.int32)
        seat0 = live[player[live] == 0]
        seat1 = live[player[live] == 1]
        if len(seat0):
            actions[seat0] = pol0(buf, seat0)
        if len(seat1):
            actions[seat1] = pol1(buf, seat1)
        for i in live:  # chosen option's kind, read before step overwrites buf
            rec[int(game[i])]["kinds"].append(int(buf["actions"][i, int(actions[i]), 0]))
            rec[int(game[i])]["choices"].append(int(actions[i]))
        vec.step(actions)
        for ep in vec.pop_finished():
            r = rec.pop(int(ep.index))
            assert r["choices"] == list(ep.choices), "recorded choices differ from the env's"
            counters["decisions"] += len(r["choices"])
            if ep.winner == VecEnv.TRUNCATED:
                counters["cut"] += 1
                continue
            i, j, s, k = sched[ep.index]
            labels = ["pass" if (m == C.MSG_SELECT_CHAIN and kd == int(Kind.PASS)) else "gold"
                      for m, kd in zip(r["msgs"], r["kinds"], strict=True)]
            cfg = vec.episode_config(ep)
            s0, s1 = stems[i], stems[j]
            rec_line = {
                "game_id": f"ga-{s}-{k}",
                "line_id": f"{s0}-vs-{s1}/ga-{s}-{k}",
                "level": "L4",
                "decks": [s0, s1],
                "config": {
                    "deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
                    "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
                    "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw, "lp": cfg.lp,
                    "setup": [], "shuffle": cfg.shuffle,
                },
                "auto_single": True,
                "choices": r["choices"], "labels": labels, "actors": r["actors"],
                "turns": r["turns"], "notes": [],
                "winner": int(ep.winner),
                # the env's final turn, not turns[-1]: a duel can end with no decision on
                # its last turn (e.g. deck-out at the draw), and replay reproduces ep.turns
                "end_turn": int(ep.turns),
                "source": f"{'ygo-agent 0546' if policy == 'ygoagent' else 'random'} "
                          f"self-play ({tag})",
            }  # fmt: skip
            with (out / f"{s0}-vs-{s1}.jsonl").open("a") as f:
                f.write(json.dumps(rec_line) + "\n")
            counters["written"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--decks", type=Path, default=ROOT / "decks" / "gauntlet")
    ap.add_argument("--sample-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--policy", choices=["ygoagent", "random"], default="ygoagent")
    args = ap.parse_args()
    t0 = time.time()
    files = sorted(p for p in args.decks.glob("*.ydk") if p.is_file())
    if not files:
        raise SystemExit(f"no .ydk decks in {args.decks}")
    decks = [load_ydk(f) for f in files]
    stems = [f.stem for f in files]
    n = len(decks)
    # N games over all ordered pairs incl. mirrors, cycled, with duel seeds S, S+1, ...
    pairs = [(i, j) for i in range(n) for j in range(n)]
    sampled = np.random.default_rng(args.seed).random(args.games) < args.sample_frac
    argmax_sched, sampled_sched = [], []
    for k in range(args.games):
        i, j = pairs[k % len(pairs)]
        (sampled_sched if sampled[k] else argmax_sched).append((i, j, args.seed + k, k))
    args.out.mkdir(parents=True, exist_ok=True)
    counters = {"written": 0, "cut": 0, "decisions": 0}
    engine = Engine()
    run_schedule(engine, decks, stems, argmax_sched, False, args.policy, args.seed,
                 args.threads, args.out, "argmax", counters)
    run_schedule(engine, decks, stems, sampled_sched, True, args.policy, args.seed,
                 args.threads, args.out, "sampled", counters)
    dt = time.time() - t0
    print(f"games written {counters['written']}, cut at the cap {counters['cut']}, "
          f"decisions {counters['decisions']}, {dt:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
