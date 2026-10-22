"""League self-play records (kind `selfplay`, ADR 0007): states + outcomes for search labels and the value head.

The student (a BC checkpoint, sampled with a temperature for diverse states) plays one
seat against a LEAGUE of opponents (itself, frozen past checkpoints, ygo-agent, greedy,
random) on pool decks. Every finished game is a free value target (winner); the window
search labels the states. To avoid narrowing, the student samples and the opponents vary.

Each finished game becomes one JSON line in <out>/<student deck>.jsonl, in the gold
record format plus `student_seat`, `opponent` (league entry name) and `temp`. Labels are
"self" at student decisions and "opp" elsewhere (never trained by BC directly). Cut
games (decision cap) are still written, with winner 3.

    uv run python scripts/gen_selfplay.py --ckpt <student best.pt> --out runs/<name>/selfplay/p0 --games N \
        [--temp 1.0] [--league 'self:0.4,base=runs/x/best.pt:0.2,ygoagent:0.2,greedy:0.1,random:0.1'] \
        [--decks decks/pool] [--ygoagent-decks decks/gauntlet] [--seed S] [--threads T] [--device cuda]

Schedule: N games; per game (seeded) an opponent entry is drawn by league weight, the
student deck (always from --decks) and the opponent deck are drawn, the duel seed is
S + k and the student's seat alternates (even k: seat 0, odd k: seat 1). The opponent
deck comes from --ygoagent-decks when the opponent is ygoagent (it only knows those
cards), else from --decks. Stems listed in --holdout-file (default decks/pool/HOLDOUT)
and `kewl-tune` are never drawn from --decks. `--policy stub` (tests and the laptop)
uses RandomPolicy (student) and Greedy/Random policies (league entries): the laptop must
never instantiate YgoAgentPolicy or BCPolicy.
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
from decklight.engine.cpp_env import VecEnv
from decklight.engine.duel import Engine

MAX_DECISIONS = 3000
ALWAYS_EXCLUDED = {"kewl-tune"}
KNOWN_ENTRIES = ("self", "ygoagent", "greedy", "random")


def parse_league(spec: str) -> list[tuple[str, Path | None, float]]:
    """Parse 'self:0.4,base=runs/x/best.pt:0.2,ygoagent:0.2' into (name, ckpt, weight) entries."""
    entries: list[tuple[str, Path | None, float]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        weight = 1.0
        head = tok
        if ":" in tok:
            maybe_head, _, maybe_w = tok.rpartition(":")
            try:
                weight = float(maybe_w)
            except ValueError:
                pass
            else:
                head = maybe_head
        if not head:
            raise SystemExit(f"bad league entry {tok!r} in {spec!r}")
        ckpt: Path | None = None
        if "=" in head:
            name, _, ckpt_s = head.partition("=")
            name, ckpt_s = name.strip(), ckpt_s.strip()
            if not name or not ckpt_s:
                raise SystemExit(f"bad league entry {tok!r} in {spec!r}")
            ckpt = Path(ckpt_s)
        else:
            name = head.strip()
        if weight < 0:
            raise SystemExit(f"negative weight in league entry {tok!r}")
        entries.append((name, ckpt, weight))
    if not entries:
        raise SystemExit(f"empty --league {spec!r}")
    if sum(w for _, _, w in entries) <= 0:
        raise SystemExit(f"--league weights sum to zero: {spec!r}")
    return entries


def load_holdout(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def make_policies(policy: str, ckpt: Path | None, entries: list[tuple[str, Path | None, float]],
                  emb: Path, device: str, seed: int, temp: float):
    """Student policy plus one policy object per league entry (separate objects, reused across games)."""
    if policy == "stub":
        from decklight.eval.policies import GreedyPolicy, RandomPolicy

        student = RandomPolicy(seed)
        league_pols = {}
        for n, (name, _, _) in enumerate(entries):
            league_pols[name] = RandomPolicy(seed + 100 + n) if name == "random" else GreedyPolicy(seed + 100 + n)
        return student, league_pols, "stub"
    if ckpt is None:
        raise SystemExit("--ckpt is required unless --policy stub")
    from decklight.eval.bc_policy import BCPolicy
    from decklight.eval.policies import GreedyPolicy, RandomPolicy
    from decklight.eval.ygoagent import YgoAgentPolicy

    student = BCPolicy(ckpt, emb, temperature=temp, seed=seed, device=device)
    league_pols = {}
    for n, (name, eckpt, _) in enumerate(entries):
        if eckpt is not None:
            league_pols[name] = BCPolicy(eckpt, emb, temperature=0.0, seed=seed + 100 + n, device=device)
        elif name == "self":
            league_pols[name] = BCPolicy(ckpt, emb, temperature=0.0, seed=seed + 100 + n, device=device)
        elif name == "ygoagent":
            league_pols[name] = YgoAgentPolicy(seed=seed + 100 + n, sample=False)
        elif name == "greedy":
            league_pols[name] = GreedyPolicy(seed + 100 + n)
        elif name == "random":
            league_pols[name] = RandomPolicy(seed + 100 + n)
        else:
            raise SystemExit(f"unknown league entry {name!r} (want self|ygoagent|greedy|random or name=<ckpt>)")
    return student, league_pols, ckpt.parent.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--league", default="self:0.5,ygoagent:0.2,greedy:0.15,random:0.15")
    ap.add_argument("--decks", type=Path, default=ROOT / "decks" / "pool")
    ap.add_argument("--ygoagent-decks", type=Path, default=ROOT / "decks" / "gauntlet")
    ap.add_argument("--holdout-file", type=Path, default=ROOT / "decks" / "pool" / "HOLDOUT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--num-envs", type=int, default=64, help="VecEnv size (big batches win on a small GPU: NOTEBOOK 2026-10-16 17:45)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--policy", choices=["bc", "stub"], default="bc")
    args = ap.parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be > 0")
    t0 = time.time()
    entries = parse_league(args.league)
    names = [n for n, _, _ in entries]
    weights = np.array([w for _, _, w in entries], float)
    weights /= weights.sum()
    excluded = load_holdout(args.holdout_file) | ALWAYS_EXCLUDED
    pool_files = sorted(p for p in args.decks.glob("*.ydk")
                        if p.is_file() and p.stem not in excluded)
    if not pool_files:
        raise SystemExit(f"no .ydk decks in {args.decks} after exclusions")
    pool_decks = [load_ydk(f) for f in pool_files]
    pool_stems = [f.stem for f in pool_files]
    np_pool = len(pool_decks)
    yg_files: list[Path] = []
    if "ygoagent" in names:
        yg_files = sorted(p for p in args.ygoagent_decks.glob("*.ydk") if p.is_file())
        if not yg_files:
            raise SystemExit(f"league needs ygoagent but no .ydk decks in {args.ygoagent_decks}")
    yg_decks = [load_ydk(f) for f in yg_files]
    yg_stems = [f.stem for f in yg_files]
    all_decks = pool_decks + yg_decks
    all_stems = pool_stems + yg_stems
    rng = np.random.default_rng(args.seed)
    draws = rng.random(args.games)
    cum = np.cumsum(weights)
    sched = []
    for k in range(args.games):
        opp = names[int(np.searchsorted(cum, draws[k], side="left").item())]
        sp = int(rng.integers(np_pool))
        s_stem = pool_stems[sp]
        if opp == "ygoagent":
            yg = int(rng.integers(len(yg_decks)))
            o_stem, o_idx = yg_stems[yg], np_pool + yg
        else:
            op = int(rng.integers(np_pool))
            o_stem, o_idx = pool_stems[op], op
        seat = k % 2
        i, j = (sp, o_idx) if seat == 0 else (o_idx, sp)
        sched.append((i, j, args.seed + k, k, seat, opp, s_stem, o_stem))
    args.out.mkdir(parents=True, exist_ok=True)
    student, league_pols, student_tag = make_policies(
        args.policy, args.ckpt, entries, args.emb, args.device, args.seed, args.temp)
    vec = VecEnv(Engine(), all_decks, min(args.num_envs, len(sched)), args.threads, seed=args.seed)
    vec.set_schedule([(i, j, s) for i, j, s, _, _, _, _, _ in sched])
    vec.set_max_decisions(MAX_DECISIONS)
    seat_of = {k: seat for _, _, _, k, seat, _, _, _ in sched}
    opp_of = {k: opp for _, _, _, k, _, opp, _, _ in sched}
    buf = vec.reset()
    rec: dict[int, dict] = {}
    entry_games = {n: 0 for n in names}
    entry_wins = {n: 0 for n in names}
    counters = {"written": 0, "cut": 0, "wins": 0, "decisions": 0}
    for _, _, _, k, _, opp, _, _ in sched:
        entry_games[opp] += 1
    while vec.remaining > 0:
        player, game = buf["player"].copy(), buf["game"].copy()
        live = np.flatnonzero(player >= 0)
        counters["decisions"] += len(live)
        for i in live:  # the decision about to be taken: actor and turn
            r = rec.setdefault(int(game[i]), {"choices": [], "labels": [], "actors": [], "turns": []})
            seat = seat_of[int(game[i])]
            r["actors"].append(int(player[i]))
            r["turns"].append(int(buf["global"][i, 2]))
            r["labels"].append("self" if int(player[i]) == seat else "opp")
        actions = np.zeros(vec.num_envs, np.int32)
        stu_rows = np.array([i for i in live if int(player[i]) == seat_of[int(game[i])]])
        if len(stu_rows):
            actions[stu_rows] = student(buf, stu_rows)
        opp_rows = np.array([i for i in live if int(player[i]) != seat_of[int(game[i])]])
        if len(opp_rows):
            by_entry: dict[str, list[int]] = {}
            for i in opp_rows:
                by_entry.setdefault(opp_of[int(game[int(i)])], []).append(int(i))
            for entry, rows in by_entry.items():
                actions[np.array(rows)] = league_pols[entry](buf, np.array(rows))
        for i in live:  # the taken option
            rec[int(game[i])]["choices"].append(int(actions[i]))
        vec.step(actions)
        for ep in vec.pop_finished():
            r = rec.pop(int(ep.index))
            assert r["choices"] == list(ep.choices), "recorded choices differ from the env's"
            i, j, _s, k, seat, opp, s_stu, s_opp = sched[ep.index]
            if ep.winner == VecEnv.TRUNCATED:
                counters["cut"] += 1
            elif int(ep.winner) == seat:
                counters["wins"] += 1
                entry_wins[opp] += 1
            cfg = vec.episode_config(ep)
            s0, s1 = all_stems[i], all_stems[j]
            rec_line = {
                "game_id": f"sp-{args.seed}-{k}",
                "line_id": f"{s_stu}-vs-{s_opp}/sp-{args.seed}-{k}",
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
                "opponent": opp,
                "temp": args.temp,
                "source": f"selfplay: student {student_tag} vs {opp} temp={args.temp}",
            }  # fmt: skip
            with (args.out / f"{s_stu}.jsonl").open("a") as f:
                f.write(json.dumps(rec_line) + "\n")
            counters["written"] += 1
    dt = time.time() - t0
    n = counters["written"]
    per = ", ".join(f"{e}: {entry_wins[e] / max(1, entry_games[e]):.1%} ({entry_games[e]})" for e in names)
    dps = counters["decisions"] / max(dt, 1e-9)
    print(f"selfplay games {n}, cut at the cap {counters['cut']}, "
          f"student win rate {counters['wins'] / max(1, n):.1%} [{per}], "
          f"{counters['decisions']} decisions at {dps:.0f}/s, {dt:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
