"""Playtest a trained BC policy: does it actually play well, not just imitate?

    uv run python scripts/playtest.py --ckpt runs/<run>/bc/best.pt [--decks 12] [--seeds 4] [--narrate 2]

1. combo test: per deck, the policy pilots turn 1 alone (going first, opponent passive) on sampled hands; its end board
   (cards it controls on the field) vs the deck's verified expert lines (mean end-board size of the verified records).
2. matches: policy vs random and greedy (both seats, all ordered deck pairs of the sample) -> win rate (Wilson CI);
   `--ygoagent` adds the ygo-agent gauntlet opponent.
3. narrate: N games policy vs greedy written turn by turn to runs/<run>/playtest/game-<k>.txt for a human audit.
Results: runs/<run>/playtest/report.json (+ printed). Laptop: cores 0-3, torch threads 2.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from ygo_zero.decks import load_ydk
from ygo_zero.demos.game import Responder
from ygo_zero.demos.names import CardNames
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.eval.bc_policy import BCPolicy
from ygo_zero.eval.match import play_match, wilson
from ygo_zero.eval.policies import GreedyPolicy, RandomPolicy

ROOT = Path(__file__).resolve().parents[1]
FIELD_LOCS = (3, 4)  # obs card column 1: 3 mzone, 4 szone (docs/obs_spec.md; not the OCG bit flags)


def my_field(obs) -> int:
    """Cards the acting player's side controls on the field (mzone + szone), from the obs card table."""
    c, m = obs["cards"], obs["card_mask"]
    return int(sum(1 for i in np.flatnonzero(m) if int(c[i, 1]) in FIELD_LOCS and int(c[i, 3]) == 0))


def verified_end_boards(deck: str) -> list[int]:
    sizes = []
    for f in list(ROOT.glob(f"data/demos/*/verified/{deck}.jsonl")) + list(ROOT.glob(f"data/demos/planned*/improved/verified/{deck}.jsonl")):
        for raw in f.read_text().splitlines():
            if raw.strip():
                sizes.append(len(json.loads(raw).get("end_board_codes") or []))
    return sizes


def combo_test(engine, pol: BCPolicy, deck_name: str, hands: int, seed0: int = 500) -> dict:
    deck = load_ydk(ROOT / "decks" / "pool" / f"{deck_name}.ydk")
    passive = Responder(0)
    ends, steps = [], []
    for s in range(hands):
        env = CppEnv(engine, DuelConfig(deck, deck, seed=seed0 + s))
        obs, n, last = env.reset(), 0, 0
        while obs is not None and int(obs["global"][2]) <= 1 and n < 400:
            if env.player == 0:
                buf = {k: v[None] for k, v in obs.items()}
                a = int(pol(buf, np.array([0]))[0])
                last = my_field(obs)
                n += 1
            else:
                a = passive.passive(obs)
            obs = env.step(a)
        if obs is not None:  # turn 2 started: the obs is the opponent's view; count p0's field from it
            c, m = obs["cards"], obs["card_mask"]
            last = int(sum(1 for i in np.flatnonzero(m) if int(c[i, 1]) in FIELD_LOCS and int(c[i, 3]) == 1))
        ends.append(last)
        steps.append(n)
    ref = verified_end_boards(deck_name)
    return {"deck": deck_name, "policy_end_board": round(statistics.mean(ends), 2), "policy_decisions": round(statistics.mean(steps), 1),
            "expert_end_board": round(statistics.mean(ref), 2) if ref else None, "expert_lines": len(ref)}  # fmt: skip


def narrate(engine, pol, deck_a: str, deck_b: str, seed: int, names: dict, out: Path, max_turns: int = 10) -> None:
    da, db = (load_ydk(ROOT / "decks" / "pool" / f"{d}.ydk") for d in (deck_a, deck_b))
    env = CppEnv(engine, DuelConfig(da, db, seed=seed))
    greedy = GreedyPolicy(seed)
    obs, lines, last_lp = env.reset(), [f"{deck_a} (policy, p0) vs {deck_b} (greedy, p1), seed {seed}"], None
    keep = {Kind.SUMMON, Kind.SPSUMMON, Kind.MSET, Kind.SSET, Kind.ACTIVATE, Kind.ATTACK, Kind.TO_BATTLE, Kind.TO_END}
    while obs is not None and int(obs["global"][2]) <= max_turns:
        p = env.player
        buf = {k: v[None] for k, v in obs.items()}
        a = int((pol if p == 0 else greedy)(buf, np.array([0]))[0])
        g = obs["global"]
        lp = (int(g[0]), int(g[1])) if p == 0 else (int(g[1]), int(g[0]))
        if lp != last_lp:
            lines.append(f"   LP p0={lp[0]} p1={lp[1]}")
            last_lp = lp
        row = obs["actions"][a]
        if int(row[0]) in keep:
            who = "POLICY" if p == 0 else "greedy"
            code = int(row[2])
            lines.append(f"t{int(g[2])} {who:6s} {Kind(int(row[0])).name:9s} {names.get(code, '') if code else ''}")
        obs = env.step(a)
    lines.append(f"winner: {env.winner}")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--decks", type=int, default=12, help="sampled pool decks (half train, half holdout)")
    ap.add_argument("--hands", type=int, default=6, help="combo-test hands per deck")
    ap.add_argument("--seeds", type=int, default=3, help="match seeds per ordered deck pair")
    ap.add_argument("--narrate", type=int, default=2)
    ap.add_argument("--ygoagent", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(args.seed)
    pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    train_d = [d for d in pool if d not in hold and verified_end_boards(d)]
    hold_d = [d for d in pool if d in hold and verified_end_boards(d)]
    sample = rng.sample(train_d, min(args.decks // 2, len(train_d))) + rng.sample(hold_d, min(args.decks // 2, len(hold_d)))
    engine, names = Engine(), CardNames().names
    pol = BCPolicy(args.ckpt, args.emb)
    out_dir = args.ckpt.parent.parent / "playtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"ckpt": str(args.ckpt), "decks": sample, "holdout": sorted(set(sample) & hold)}

    combos = [combo_test(engine, pol, d, args.hands) | {"holdout": d in hold} for d in sample]
    report["combo"] = combos
    for c in combos:
        print(f"combo {c['deck'][:34]:34s} {'H' if c['holdout'] else ' '} policy end board {c['policy_end_board']:.1f} "
              f"({c['policy_decisions']:.0f} decisions) | expert {c['expert_end_board']} ({c['expert_lines']} lines)")

    decks = [load_ydk(ROOT / "decks" / "pool" / f"{d}.ydk") for d in sample]
    pairs = [(i, j) for i in range(len(decks)) for j in range(len(decks)) if i != j]
    opps = [("random", RandomPolicy(1)), ("greedy", GreedyPolicy(2))]
    if args.ygoagent:
        from ygo_zero.eval.ygoagent import YgoAgentPolicy

        opps.append(("ygoagent", YgoAgentPolicy(seed=3)))
    report["matches"] = {}
    for name, opp in opps:
        games = play_match(engine, decks, pol, opp, pairs, range(args.seeds), num_envs=32, num_threads=2)
        w = sum(g.win for g in games)
        lo, hi = wilson(w, len(games))
        report["matches"][name] = {"games": len(games), "win_rate": round(w / len(games), 3), "ci": [round(lo, 3), round(hi, 3)]}
        print(f"vs {name:8s}: {w:.0f}/{len(games)} = {w / len(games):.1%}  [{lo:.1%}, {hi:.1%}]")

    for k in range(args.narrate):
        a, b = rng.sample(sample, 2)
        narrate(engine, pol, a, b, 900 + k, names, out_dir / f"game-{k}.txt")
    report["seconds"] = round(time.time() - t0)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"report: {out_dir / 'report.json'} ({report['seconds']} s)")


if __name__ == "__main__":
    main()
