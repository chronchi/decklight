"""DAgger records (kind `dagger`): OUR policy plays, ygo-agent labels every student decision.

The student (a BC checkpoint) plays one seat; at every student decision the labeler
(ygo-agent 0546, argmax) says what it would do there and that label is the training
target. A deterministic fraction `--beta` of the games follows the labeler's action
(beta games); the rest follow the student's own action. The opponent seat always
plays ygo-agent 0546 (argmax). Three separate policy objects (separate LSTM states).

Each finished game becomes one JSON line in <out>/<student deck>-vs-<opp deck>.jsonl,
in the gold record format plus `student_seat`, `beta` and `targets` (the labeler's
option index at student decisions, null elsewhere). Cut games (decision cap) are
still written, with winner 3. Labels mirror gold on student decisions (a PASS
(kind 11) at a chain window is `pass`, every other choice is `gold`); opponent
decisions are `opp` (never trained on).

    uv run python scripts/gen_dagger.py --out data/demos/dagger --games N --ckpt runs/<run>/bc/best.pt
        [--decks decks/gauntlet] [--beta 0.3] [--seed S] [--threads T] [--device cuda]

Schedule: N games over all ordered deck pairs (incl. mirrors, cycled) with duel seeds
S, S+1, ...; the student's seat alternates by game (even k: seat 0, odd k: seat 1).
`--policy stub` (tests and the laptop) uses RandomPolicy (student, opponent) and
GreedyPolicy (labeler): the laptop must never instantiate YgoAgentPolicy or BCPolicy.
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


def make_policies(policy: str, ckpt: Path | None, emb: Path, device: str, seed: int):
    """Student / labeler / opponent as three separate policy objects (separate LSTM states)."""
    if policy == "stub":
        from decklight.eval.policies import GreedyPolicy, RandomPolicy

        return RandomPolicy(seed), GreedyPolicy(seed + 7), RandomPolicy(seed + 11)
    if ckpt is None:
        raise SystemExit("--ckpt is required unless --policy stub")
    from decklight.eval.bc_policy import BCPolicy
    from decklight.eval.ygoagent import YgoAgentPolicy

    student = BCPolicy(ckpt, emb, device=device)
    return student, YgoAgentPolicy(seed=seed + 7, sample=False), YgoAgentPolicy(seed=seed + 11, sample=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--decks", type=Path, default=ROOT / "decks" / "gauntlet")
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--policy", choices=["ygoagent", "stub"], default="ygoagent")
    args = ap.parse_args()
    t0 = time.time()
    files = sorted(p for p in args.decks.glob("*.ydk") if p.is_file())
    if not files:
        raise SystemExit(f"no .ydk decks in {args.decks}")
    decks = [load_ydk(f) for f in files]
    stems = [f.stem for f in files]
    n = len(decks)
    # N games over all ordered pairs incl. mirrors, cycled, with duel seeds S, S+1, ...;
    # the student's seat alternates by game; beta games (follow the labeler) are a
    # deterministic fraction from the seed.
    pairs = [(i, j) for i in range(n) for j in range(n)]
    beta_flags = list(np.random.default_rng(args.seed).random(args.games) < args.beta)
    sched = []
    for k in range(args.games):
        i, j = pairs[k % len(pairs)]
        sched.append((i, j, args.seed + k, k, k % 2, bool(beta_flags[k])))
    args.out.mkdir(parents=True, exist_ok=True)
    student, labeler, opponent = make_policies(args.policy, args.ckpt, args.emb, args.device, args.seed)
    if args.policy == "stub":
        student_tag, labeler_tag = "stub", "greedy"
    else:
        assert args.ckpt is not None
        student_tag, labeler_tag = args.ckpt.parent.name, "ygo-agent 0546"
    vec = VecEnv(Engine(), decks, min(64, len(sched)), args.threads, seed=args.seed)
    vec.set_schedule([(i, j, s) for i, j, s, _, _, _ in sched])
    vec.set_max_decisions(MAX_DECISIONS)
    seat_of = {k: seat for _, _, _, k, seat, _ in sched}
    beta_of = {k: b for _, _, _, k, _, b in sched}
    buf = vec.reset()
    rec: dict[int, dict] = {}
    counters = {"written": 0, "cut": 0, "student": 0, "agree": 0, "chain": 0, "chain_agree": 0}
    while vec.remaining > 0:
        player, game = buf["player"].copy(), buf["game"].copy()
        live = np.flatnonzero(player >= 0)
        for i in live:  # the decision about to be taken: actor and turn
            r = rec.setdefault(int(game[i]), {"choices": [], "targets": [], "labels": [],
                                              "actors": [], "turns": []})
            r["actors"].append(int(player[i]))
            r["turns"].append(int(buf["global"][i, 2]))
        actions = np.zeros(vec.num_envs, np.int32)
        stu_rows = np.array([i for i in live if int(player[i]) == seat_of[int(game[i])]])
        opp_rows = np.array([i for i in live if int(player[i]) != seat_of[int(game[i])]])
        if len(stu_rows):
            lab = labeler(buf, stu_rows)
            stu = student(buf, stu_rows)
            for i, a_lab, a_stu in zip(stu_rows, lab, stu, strict=True):
                g = int(game[i])
                msg = int(buf["global"][i, 21])
                kd = int(buf["actions"][i, int(a_lab), 0])
                r = rec[g]
                r["targets"].append(int(a_lab))
                r["labels"].append("pass" if (msg == C.MSG_SELECT_CHAIN and kd == int(Kind.PASS)) else "gold")
                counters["student"] += 1
                counters["agree"] += int(a_lab == a_stu)
                if msg == C.MSG_SELECT_CHAIN:
                    counters["chain"] += 1
                    counters["chain_agree"] += int(a_lab == a_stu)
                actions[i] = a_lab if beta_of[g] else a_stu
        if len(opp_rows):
            actions[opp_rows] = opponent(buf, opp_rows)
            for i in opp_rows:
                r = rec[int(game[i])]
                r["targets"].append(None)
                r["labels"].append("opp")
        for i in live:  # the taken option
            rec[int(game[i])]["choices"].append(int(actions[i]))
        vec.step(actions)
        for ep in vec.pop_finished():
            r = rec.pop(int(ep.index))
            assert r["choices"] == list(ep.choices), "recorded choices differ from the env's"
            i, j, s, k, seat, beta = sched[ep.index]
            if ep.winner == VecEnv.TRUNCATED:
                counters["cut"] += 1
            cfg = vec.episode_config(ep)
            s0, s1 = stems[i], stems[j]
            s_stu, s_opp = (s0, s1) if seat == 0 else (s1, s0)
            rec_line = {
                "game_id": f"dg-{s}-{k}",
                "line_id": f"{s_stu}-vs-{s_opp}/dg-{s}-{k}",
                "level": "L4",
                "decks": [s0, s1],
                "config": {
                    "deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
                    "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
                    "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw, "lp": cfg.lp,
                    "setup": [], "shuffle": cfg.shuffle,
                },
                "auto_single": True,
                "choices": r["choices"], "labels": r["labels"], "actors": r["actors"],
                "turns": r["turns"], "notes": [],
                "winner": int(ep.winner),
                "end_turn": int(ep.turns),
                "student_seat": seat,
                "beta": beta,
                "targets": r["targets"],
                "source": f"dagger: student {student_tag}, labeler {labeler_tag} "
                          f"({'beta' if beta else 'student'})",
            }  # fmt: skip
            with (args.out / f"{s_stu}-vs-{s_opp}.jsonl").open("a") as f:
                f.write(json.dumps(rec_line) + "\n")
            counters["written"] += 1
    dt = time.time() - t0
    sd = counters["student"]
    print(f"games written {counters['written']}, cut at the cap {counters['cut']}, "
          f"student decisions {sd}, agreement {counters['agree'] / max(1, sd):.1%} "
          f"(chain {counters['chain_agree'] / max(1, counters['chain']):.1%} "
          f"on {counters['chain']}), {dt:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
