"""Playtest a trained BC policy: does it actually play well, not just imitate?

    uv run python scripts/playtest.py --ckpt runs/<run>/bc/best.pt [--decks 12] [--seeds 4] [--narrate 2]

1. combo test: per deck, the policy pilots turn 1 alone (going first, opponent passive) on the expert lines' own hands;
   its end board (cards it controls on the field) vs the expert line's. `--samples K` adds best-of-K sampled runs
   (temperature 1): the gap best-of-K vs greedy is the headroom search/RL can harvest (pass@k vs pass@1, Kolibri §3.2.5).
2. matches: policy vs random and greedy (both seats, all ordered deck pairs of the sample) -> win rate (Wilson CI);
   `--ygoagent` adds the ygo-agent gauntlet opponent.
3. narrate: N games policy vs greedy written turn by turn to runs/<run>/playtest/game-<k>.txt for a human audit.
4. `--vs-ckpt`: head-to-head vs another BC checkpoint (e.g. the base model) on the same deck pairs, both seats.
5. `--ygoagent`: matches vs ygo-agent 0546 on `--gauntlet-decks` (default decks/gauntlet: its own 31 lists, inside its
   864-card vocabulary), each deck vs its next `--ga-opps` decks, both seats, `--ga-seeds` seeds per ordered pair
   (248 games at the default 1; was 4 decks/72 games until 2026-10-05).
6. `--selfplay K`: K full games policy vs itself (both seats), saved in hand_play's state format under
   runs/<run>/playtest/selfplay/<name>/state.json, for the same muse review as the gold games (dump_game.py).
The combo test runs the greedy run and the K samples of a hand in lockstep (one batched forward per step) and
reports `stuck`: the share of runs that hit the 400-decision cap (the policy never ended its turn).
Every section prints its wall time.
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
import torch

from decklight.decks import load_ydk
from decklight.demos.game import Responder
from decklight.demos.names import CardNames
from decklight.engine.actions import Kind
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine
from decklight.eval.bc_policy import BCPolicy
from decklight.eval.match import game_state, play_match, ring_pairs, wilson
from decklight.eval.policies import GreedyPolicy, RandomPolicy

ROOT = Path(__file__).resolve().parents[1]
FIELD_LOCS = (3, 4)  # obs card column 1: 3 mzone, 4 szone (docs/obs_spec.md; not the OCG bit flags)


def my_field(obs) -> int:
    """Cards the acting player's side controls on the field (mzone + szone), from the obs card table."""
    c, m = obs["cards"], obs["card_mask"]
    return int(sum(1 for i in np.flatnonzero(m) if int(c[i, 1]) in FIELD_LOCS and int(c[i, 3]) == 0))


def verified_records(deck: str, limit: int = 12) -> list[dict]:
    """The deck's going-first verified line records (turn-1 lines), biggest declared end board first."""
    recs = []
    for f in list(ROOT.glob(f"data/demos/*/verified/{deck}.jsonl")) + list(
        ROOT.glob(f"data/demos/planned*/improved/verified/{deck}.jsonl")
    ):
        for raw in f.read_text().splitlines():
            if raw.strip():
                r = json.loads(raw)
                if r.get("me", 0) == 0:
                    recs.append(r)
    recs.sort(key=lambda r: -len(r.get("end_board_codes") or []))
    return recs[:limit]


def verified_end_boards(deck: str) -> list[int]:
    """Expert reference with the SAME measure as the policy: replay each record to the end of turn 1 and count the
    cards player 0 controls on the field."""
    from decklight.demos.dataset import record_config

    engine, sizes = Engine(), []
    for r in verified_records(deck):
        try:
            env = CppEnv(engine, record_config(r), r.get("auto_single", True))
            obs = env.reset()
            last = 0
            for c in r["choices"]:
                if env.player == 0:
                    last = my_field(obs)
                obs = env.step(c)
                if obs is None:
                    break
            if obs is not None and env.player == 0:
                last = my_field(obs)
            sizes.append(last)
        except Exception:  # noqa: BLE001, S112 - a record that no longer replays is skipped
            continue
    return sizes


def _turn1_fields(envs: list, pols: list) -> list[tuple[int, int]]:
    """Player 0's turn 1 (opponent passive) for several envs in lockstep: pols[i] acts for envs[i]; envs sharing a policy object get one
    batched forward per step. Returns [(end field, p0 decisions)] per env."""
    passive = Responder(0)
    for p in {id(p): p for p in pols}.values():  # each call is a fresh set of games for the loop guard
        if getattr(p, "guard", None) is not None:
            p.guard.reset()
    obs = [e.reset() for e in envs]
    last, n = [0] * len(envs), [0] * len(envs)
    live = [True] * len(envs)
    while any(live):
        groups: dict[int, list[int]] = {}
        for i, e in enumerate(envs):
            if not live[i]:
                continue
            if obs[i] is None or int(obs[i]["global"][2]) > 1 or n[i] >= 400:
                live[i] = False
                continue
            if e.player == 0:
                last[i] = my_field(obs[i])
                groups.setdefault(id(pols[i]), []).append(i)
            else:
                obs[i] = e.step(passive.passive(obs[i]))
        for idx in groups.values():
            buf = {k: np.stack([obs[i][k] for i in idx]) for k in obs[idx[0]]}
            acts = pols[idx[0]](buf, np.arange(len(idx)), slots=np.array(idx))
            for i, a in zip(idx, acts, strict=True):
                n[i] += 1
                obs[i] = envs[i].step(int(a))
    out = []
    for i in range(len(envs)):
        if obs[i] is not None and int(obs[i]["global"][2]) > 1:  # p0's cards seen from player 1's view
            c, m = obs[i]["cards"], obs[i]["card_mask"]
            last[i] = int(sum(1 for j in np.flatnonzero(m) if int(c[j, 1]) in FIELD_LOCS and int(c[j, 3]) == 1))
        out.append((last[i], n[i]))
    return out


def _replay_field(engine, rec: dict) -> int:
    """The expert line's end field: replay ALL of the record's choices (both players, as recorded) through turn 1."""
    from decklight.demos.dataset import record_config

    env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
    obs, last = env.reset(), 0
    for c in rec["choices"]:
        if obs is None or int(obs["global"][2]) > 1:
            break
        if env.player == 0:
            last = my_field(obs)
        obs = env.step(c)
    if obs is not None:
        if int(obs["global"][2]) > 1 or env.player == 1:
            c, m = obs["cards"], obs["card_mask"]
            ctrl = 0 if env.player == 0 else 1
            last = int(sum(1 for i in np.flatnonzero(m) if int(c[i, 1]) in FIELD_LOCS and int(c[i, 3]) == ctrl))
        else:
            last = my_field(obs)
    return last


def same_weights(a: Path, b: Path) -> bool:
    """True when two checkpoints hold identical tensors (fp16 best.pt or {"model": ...} files)."""
    sa, sb = (torch.load(x, map_location="cpu") for x in (a, b))
    sa, sb = sa.get("model", sa), sb.get("model", sb)
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k].float(), sb[k].float()) for k in sa)


def combo_test(engine, pol: BCPolicy, deck_name: str, hands: int, sampler: BCPolicy | None = None, k: int = 0) -> dict:
    """Paired test on the expert's own hands: for each verified going-first line record (same config and seed, so the
    same opening hand), the policy plays turn 1 from scratch; compare its end field with the expert line's."""
    from decklight.demos.dataset import record_config

    pairs = []
    for r in verified_records(deck_name, limit=hands):
        try:
            kk = k if sampler is not None else 0
            envs = [CppEnv(engine, record_config(r), r.get("auto_single", True)) for _ in range(1 + kk)]
            res = _turn1_fields(envs, [pol] + [sampler] * kk)
            (mine, n), expert = res[0], _replay_field(engine, r)
            best = max(f for f, _ in res)
            stuck = sum(nn >= 400 for _, nn in res) / len(res)
            pairs.append((mine, expert, n, best, stuck))
        except Exception:  # noqa: BLE001, S112 - a record that no longer replays is skipped
            continue
    if not pairs:
        return {"deck": deck_name, "hands": 0, "policy_end_board": None, "expert_end_board": None,
                "policy_decisions": 0, "ge_expert": None, "expert_lines": 0}  # fmt: skip
    return {
        "deck": deck_name, "hands": len(pairs),
        "policy_end_board": round(statistics.mean(p[0] for p in pairs), 2),
        "expert_end_board": round(statistics.mean(p[1] for p in pairs), 2),
        "policy_decisions": round(statistics.mean(p[2] for p in pairs), 1),
        "ge_expert": round(sum(p[0] >= p[1] for p in pairs) / len(pairs), 2),
        "best_of_k_end_board": round(statistics.mean(p[3] for p in pairs), 2) if k else None,
        "best_of_k_ge_expert": round(sum(p[3] >= p[1] for p in pairs) / len(pairs), 2) if k else None,
        "expert_lines": len(pairs),
        "stuck": round(statistics.mean(p[4] for p in pairs), 3),
    }  # fmt: skip


def narrate(engine, pol, deck_a: str, deck_b: str, seed: int, names: dict, out: Path, max_turns: int = 10) -> None:
    da, db = (load_ydk(ROOT / "decks" / "pool" / f"{d}.ydk") for d in (deck_a, deck_b))
    env = CppEnv(engine, DuelConfig(da, db, seed=seed))
    greedy = GreedyPolicy(seed)
    if getattr(pol, "guard", None) is not None:
        pol.guard.reset()
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


def selfplay_game(engine, pol, deck_a: str, deck_b: str, seed: int, out_dir: Path, max_steps: int = 3000) -> dict:
    """One full game, the policy on both seats, saved as hand_play state (decks, seed, choices, labels, actors,
    turns, notes) so `hand_play.py show/log` and `data/cache/dump_game.py` read it like a gold game. Every decision
    is labelled "gold" (the review rubric skips single-option prompts itself). Stops at duel end or max_steps."""
    da, db = (load_ydk(ROOT / "decks" / "pool" / f"{d}.ydk") for d in (deck_a, deck_b))
    env = CppEnv(engine, DuelConfig(da, db, seed=seed))
    if getattr(pol, "guard", None) is not None:
        pol.guard.reset()
    obs = env.reset()
    st = {"decks": [deck_a, deck_b], "seed": seed, "choices": [], "labels": [], "actors": [], "turns": [], "notes": []}
    while obs is not None and len(st["choices"]) < max_steps:
        a = int(pol({k: v[None] for k, v in obs.items()}, np.array([0]))[0])
        st["choices"].append(a)
        st["labels"].append("gold")
        st["actors"].append(int(env.player))
        st["turns"].append(int(obs["global"][2]))
        obs = env.step(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "state.json").write_text(json.dumps(st))
    over = obs is None
    return {"decks": [deck_a, deck_b], "seed": seed, "choices": len(st["choices"]), "over": over,
            "winner": env.winner if over else None, "turns": st["turns"][-1] if st["turns"] else 0}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--decks", type=int, default=12, help="sampled pool decks (half train, half holdout)")
    ap.add_argument("--hands", type=int, default=6, help="combo-test hands per deck")
    ap.add_argument("--seeds", type=int, default=3, help="match seeds per ordered deck pair")
    ap.add_argument("--narrate", type=int, default=2)
    ap.add_argument("--samples", type=int, default=0, help="best-of-K sampled turn-1 runs per hand (headroom)")
    ap.add_argument("--ygoagent", action="store_true", help="matches vs ygo-agent 0546 on --gauntlet-decks")
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks/gauntlet", help="ygo-agent's own lists")
    ap.add_argument("--ga-opps", type=int, default=4, help="vs ygo-agent: each gauntlet deck vs its next k decks")
    ap.add_argument("--ga-seeds", type=int, default=1,
                    help="vs ygo-agent: seeds per ordered gauntlet pair (1 = today's 248 games; 3 ~= 744)")
    ap.add_argument("--vs-ckpt", type=Path, default=None, help="head-to-head vs this BC checkpoint (e.g. the base)")
    ap.add_argument("--newcards", type=Path, default=None,
                    help="new-card test (ADR 0007): matches on these unseen-archetype decks (decks/newcards) vs greedy and --vs-ckpt")
    ap.add_argument("--selfplay", type=int, default=0, help="full policy-vs-policy games saved for review")
    ap.add_argument("--save-games", type=int, default=0,
                    help="per match, save up to K games as replays under <out_dir>/games/<match>/<k>/state.json")
    ap.add_argument("--selfplay-decks", nargs="*", default=None, help="deck pool for --selfplay (default: the sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2, help="VecEnv threads for matches")
    args = ap.parse_args()
    if args.vs_ckpt and same_weights(args.ckpt, args.vs_ckpt):
        # fail loud (mix-20261005-1738): playing a checkpoint against an identical copy is a wasted report card
        print(f"ALERT {args.ckpt} has the same weights as --vs-ckpt {args.vs_ckpt}: wrong checkpoint", flush=True)
        raise SystemExit(2)
    t0 = time.time()
    rng = random.Random(args.seed)
    pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    train_d = [d for d in pool if d not in hold and verified_end_boards(d)]
    hold_d = [d for d in pool if d in hold and verified_end_boards(d)]
    sample = rng.sample(train_d, min(args.decks // 2, len(train_d))) + rng.sample(hold_d, min(args.decks // 2, len(hold_d)))
    engine, names = Engine(), CardNames().names
    pol = BCPolicy(args.ckpt, args.emb, device=args.device)
    sampler = BCPolicy(args.ckpt, args.emb, temperature=1.0, seed=args.seed, device=args.device) if args.samples else None
    out_dir = args.ckpt.parent.parent / "playtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"ckpt": str(args.ckpt), "decks": sample, "holdout": sorted(set(sample) & hold)}
    report["setup_seconds"] = round(time.time() - t0)

    def flush(section: str, t: float) -> None:  # report.json after every section: a timeout keeps what finished
        report.setdefault("section_seconds", {})[section] = round(time.time() - t)
        print(f"[{section}: {time.time() - t:.0f} s]", flush=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1))

    def match(name: str, opp, decks: list, deck_files: list[str], n_seeds: int, k: int = 0) -> None:
        pairs = ring_pairs(len(decks), k or len(decks))
        games = play_match(engine, decks, pol, opp, pairs, range(n_seeds), num_envs=64, num_threads=args.threads)
        w = sum(g.winner_is_a for g in games)
        lo, hi = wilson(w, len(games))
        first = [g for g in games if g.a_first]
        report["matches"][name] = {
            "games": len(games), "win_rate": round(w / len(games), 3), "ci": [round(lo, 3), round(hi, 3)],
            "win_rate_first": round(sum(g.winner_is_a for g in first) / max(len(first), 1), 3),
            "mean_turns": round(statistics.mean(g.turns for g in games), 1),
            "cut": sum(g.truncated for g in games),
            # per game (deck A, deck B, seed, A first, result, turns, cut): where the losses come from
            "per_game": [[decks[g.deck_a].name, decks[g.deck_b].name, g.seed, g.a_first, g.winner_is_a, g.turns,
                          g.truncated] for g in games],
        }  # fmt: skip
        print(f"vs {name:8s}: {w:.0f}/{len(games)} = {w / len(games):.1%}  [{lo:.1%}, {hi:.1%}]  "
              f"mean turns {report['matches'][name]['mean_turns']}, cut {report['matches'][name]['cut']}", flush=True)
        if args.save_games > 0 and games:
            opp_name = getattr(opp, "name", type(opp).__name__)
            losses = [g for g in games if g.winner_is_a == 0.0]
            wins = [g for g in games if g.winner_is_a == 1.0]
            draws = [g for g in games if g.winner_is_a == 0.5]
            ordered, i, j = [], 0, 0
            while len(ordered) < args.save_games and (i < len(losses) or j < len(wins)):
                if i < len(losses) and len(ordered) < args.save_games:
                    ordered.append(losses[i])
                    i += 1
                if j < len(wins) and len(ordered) < args.save_games:
                    ordered.append(wins[j])
                    j += 1
            for g in losses[i:] + wins[j:] + draws:  # one-sided matches still save K games
                if len(ordered) >= args.save_games:
                    break
                ordered.append(g)
            for kk, g in enumerate(ordered[: args.save_games]):
                st = game_state(deck_files, g, engine, match=name, opponent=opp_name)
                d = out_dir / "games" / name / str(kk)
                d.mkdir(parents=True, exist_ok=True)
                (d / "state.json").write_text(json.dumps(st))
                env2 = CppEnv(
                    engine,
                    DuelConfig(load_ydk(ROOT / st["decks"][0]), load_ydk(ROOT / st["decks"][1]),
                               seed=st["seed"]),
                )
                obs2 = env2.reset()
                for c in st["choices"]:
                    obs2 = env2.step(int(c))
                    if obs2 is None:
                        break
                if g.truncated:
                    ok = obs2 is not None
                elif g.winner_is_a == 1.0:
                    ok = env2.winner == (0 if g.a_first else 1)
                elif g.winner_is_a == 0.0:
                    ok = env2.winner == (1 if g.a_first else 0)
                else:
                    ok = env2.winner == 2
                if not ok:
                    print(f"ALERT playtest: replay mismatch {name} {kk}", flush=True)
                    raise SystemExit(4)

    # matches first (cheap, batched over VecEnv), then self-play games, then the slower combo test
    t = time.time()
    report["matches"] = {}
    decks = [load_ydk(ROOT / "decks" / "pool" / f"{d}.ydk") for d in sample]
    deck_files = [f"decks/pool/{d}.ydk" for d in sample]
    if args.vs_ckpt:
        match("base", BCPolicy(args.vs_ckpt, args.emb, device=args.device), decks, deck_files, args.seeds)
    match("greedy", GreedyPolicy(2), decks, deck_files, args.seeds)
    match("random", RandomPolicy(1), decks, deck_files, args.seeds)
    if args.ygoagent:
        from decklight.eval.ygoagent import YgoAgentPolicy

        gfiles = sorted(args.gauntlet_decks.glob("*.ydk"))
        gdecks = [load_ydk(f) for f in gfiles]
        match("ygoagent", YgoAgentPolicy(seed=3), gdecks,
              [f.resolve().relative_to(ROOT.resolve()).as_posix() for f in gfiles],
              args.ga_seeds, args.ga_opps)  # 31 lists x 4 opponents x seeds x 2 seats
    if args.newcards:  # every deck plays every other new-card deck, both seats: generalization to unseen cards
        nfiles = sorted(args.newcards.glob("*.ydk"))
        ndecks = [load_ydk(f) for f in nfiles]
        nrel = [f.resolve().relative_to(ROOT.resolve()).as_posix() for f in nfiles]
        if args.vs_ckpt:
            match("new_base", BCPolicy(args.vs_ckpt, args.emb, device=args.device), ndecks, nrel, args.seeds)
        match("new_greedy", GreedyPolicy(2), ndecks, nrel, args.seeds)
    flush("matches", t)

    if args.selfplay:
        t = time.time()
        sp_pool = args.selfplay_decks or sample
        sp = []
        for k in range(args.selfplay):
            a, b = rng.sample(sp_pool, 2)
            sp.append(selfplay_game(engine, pol, a, b, 700 + k, out_dir / "selfplay" / f"sp{k}"))
            print(f"selfplay sp{k} {a} vs {b}: {sp[-1]['choices']} choices, turn {sp[-1]['turns']}, "
                  f"{'winner ' + str(sp[-1]['winner']) if sp[-1]['over'] else 'UNFINISHED'}", flush=True)
        report["selfplay"] = sp
        flush("selfplay", t)

    t = time.time()
    combos = []
    for d in sample:
        combos.append(combo_test(engine, pol, d, args.hands, sampler, args.samples) | {"holdout": d in hold})
        c = combos[-1]
        print(f"combo {c['deck'][:34]:34s} {'H' if c['holdout'] else ' '} same {c['hands']} hands: policy field "
              f"{c['policy_end_board']} vs expert {c['expert_end_board']}, policy >= expert on {c['ge_expert']} "
              f"({c['policy_decisions']} decisions); best-of-{args.samples} {c.get('best_of_k_end_board')} "
              f">= expert on {c.get('best_of_k_ge_expert')}; stuck {c.get('stuck')}", flush=True)
        report["combo"] = combos
    ok = [c for c in combos if c["hands"]]
    if ok:
        report["combo_mean"] = {
            "policy": round(statistics.mean(c["policy_end_board"] for c in ok), 2),
            "expert": round(statistics.mean(c["expert_end_board"] for c in ok), 2),
            "best_of_k": round(statistics.mean(c["best_of_k_end_board"] for c in ok), 2) if args.samples else None,
            "stuck": round(statistics.mean(c["stuck"] for c in ok), 3),
        }  # fmt: skip
        print(f"combo mean: {report['combo_mean']}", flush=True)
    flush("combo", t)

    t = time.time()
    for k in range(args.narrate):
        a, b = rng.sample(sample, 2)
        narrate(engine, pol, a, b, 900 + k, names, out_dir / f"game-{k}.txt")
    flush("narrate", t)
    report["seconds"] = round(time.time() - t0)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"report: {out_dir / 'report.json'} ({report['seconds']} s)")


if __name__ == "__main__":
    main()
