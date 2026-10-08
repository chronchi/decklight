"""Hinted self-distillation bank (PLAN section 0a step 3, OPSD section 3.2.1.4).

    uv run python scripts/hint_bank.py --ckpt <policy best.pt> --out runs/<name>/hint-bank \\
        --n 200 --games 4 --seed 0 [--opp-ckpt <base best.pt>] [--value <critic.pt>]
    uv run python scripts/hint_bank.py --summarize runs/<name>/hint-bank

For N DAGS starts (train split only: both decks in pool - HOLDOUT - kewl-tune, the same
filter `turn_lines.load_dags_starts` applies), for the start's acting player ("actor"):
K raw games (actor = policy argmax, opponent = base policy) and K hinted games with the
same seeds (actor = policy + honest search at its own decisions via
`search.play.search_game_gen` + `WindowSearch`, `--honest 1 --det-fallback skip`).

"Hard" start = raw wins 0/K. The bank holds the actor's decisions from hinted games
WON on hard starts, stored as replays (config + full response list + per-decision
actor flags), ~1 KB/game. `train_rl --hint-bank` replays them for its auxiliary loss.

We have no hint-conditioned policy, so "hinted -> unhinted" honestly means the
unhinted policy is trained on successes found with the search's help.

Sharding: `--nproc/--proc` like search_match (each proc writes `bank-<proc>.jsonl` +
`games-<proc>.jsonl`); `--summarize <out>` merges them into `bank.jsonl` + `summary.json`
and fails loud on 0 hard starts or 0 banked decisions. With `--nproc 1` (default) the
summarize runs inline, so the out dir directly holds `bank.jsonl` + `summary.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import NoReturn

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import DuelConfig, Engine  # noqa: E402
from ygo_zero.eval.turn1 import (  # noqa: E402
    config_to_dict,
    replay_prefix,
    resolve_decks,
)
from ygo_zero.search.play import argmax_select, search_game_gen  # noqa: E402
from ygo_zero.search.turn1 import _batch, drive_many  # noqa: E402

from search_match import RouteWindowSearch, tag_game  # noqa: E402  # sharded-driver routing

DEFAULT_STARTS = ROOT / "data" / "cache" / "dags" / "starts.jsonl"


def fail(msg: str) -> NoReturn:
    print(f"ALERT hint_bank: {msg}", flush=True)
    raise SystemExit(1)


def sample_train_starts(path: Path, n: int, seed: int) -> tuple[list[tuple[int, dict]], int]:
    """N train-split starts with their file indices (both decks in the train split).

    Same deck filter as `turn_lines.load_dags_starts` (rows whose two deck names are
    both in pool - HOLDOUT - kewl-tune); the file index is kept so bank rows can point
    back at the train_rl starts list (`--hint-start-bonus`). Returns ((idx, row), skipped).
    """
    _, decks, _ = resolve_decks("train", seed)
    names = {name for name, _ in decks}
    cands: list[tuple[int, dict]] = []
    skipped = 0
    for idx, raw in enumerate(path.read_text().splitlines()):
        if not raw.strip():
            continue
        r = json.loads(raw)
        ds = r.get("decks") or []
        if len(ds) != 2 or not all(ds):
            skipped += 1
            continue
        if ds[0] not in names or ds[1] not in names:
            continue
        cands.append((idx, r))
    if not cands:
        fail(f"0 train-split starts in {path}")
    return random.Random(seed).sample(cands, min(n, len(cands))), skipped


def start_state(engine: Engine, row: dict) -> tuple[DuelConfig, CppEnv, dict, int, int]:
    """Replay a starts row's prefix; returns (cfg, env, obs, actor, turn_start)."""
    from ygo_zero.eval.turn1 import dags_config

    cfg = dags_config(row)
    env, obs = replay_prefix(engine, cfg, [int(c) for c in row.get("prefix", [])])
    actor = int(env.player)
    return cfg, env, obs, actor, int(obs["global"][2])


def raw_game(engine: Engine, cfg: DuelConfig, prefix: list[int], actor: int,
             pol, opp_pol, game_idx: int, max_dec: int) -> dict:
    """One full game from a DAGS prefix, both seats argmax (actor: pol, other: opp_pol)."""
    if getattr(pol, "guard", None) is not None:
        pol.guard.reset()
    if getattr(opp_pol, "guard", None) is not None and opp_pol is not pol:
        opp_pol.guard.reset()
    env = CppEnv(engine, cfg)
    obs = env.reset()
    choices: list[int] = []
    actors: list[int] = []
    for c in prefix:
        if obs is None:
            raise ValueError("game ended inside the prefix")
        n_opts = int(obs["action_mask"].sum())
        if not 0 <= int(c) < n_opts:
            raise ValueError(f"prefix choice {c} illegal with {n_opts} options")
        choices.append(int(c))
        actors.append(int(env.player))
        obs = env.step(int(c))
    n = 0
    while obs is not None and n < max_dec:
        me = int(env.player)
        p = pol if me == actor else opp_pol
        buf = _batch([obs])
        a = int(p(buf, np.array([0]), slots=np.array([game_idx]))[0])
        choices.append(a)
        actors.append(me)
        obs = env.step(a)
        n += 1
    return {"choices": choices, "actors": actors,
            "winner": None if obs is not None else int(env.winner),
            "truncated": obs is not None}


def seat_of(obs: dict) -> int:
    """Absolute seat of the acting player: global[4] ("I went first") is 1 iff seat 0 acts."""
    return 0 if int(obs["global"][4]) == 1 else 1


def summarize(out: Path) -> dict:
    """Merge <out>/games-*.jsonl + bank-*.jsonl into bank.jsonl + summary.json (fail loud)."""
    gfiles = sorted(out.glob("games-*.jsonl"))
    recs = [json.loads(x) for f in gfiles for x in f.read_text().splitlines() if x.strip()]
    if not recs:
        fail(f"no games in {out}/games-*.jsonl")
    by_start: dict[int, list[dict]] = {}
    for r in recs:
        by_start.setdefault(int(r["start_id"]), []).append(r)
    hard = sorted(s for s, rs in by_start.items()
                  if any(r["kind"] == "raw" for r in rs)
                  and sum(1 for r in rs if r["kind"] == "raw" and r["actor_win"]) == 0)
    if not hard:
        fail(f"0 hard starts ({len(by_start)} starts, all won a raw game)")
    banked = [json.loads(x) for f in sorted(out.glob("bank-*.jsonl")) if f.is_file()
              for x in f.read_text().splitlines() if x.strip()]
    banked = [r for r in banked if int(r["start_id"]) in set(hard) and r.get("actor_win")]
    if not banked:
        fail(f"0 banked decisions ({len(hard)} hard starts, no hinted win on any)")
    decisions = sum(sum(1 for a in r["actors"] if a == r["actor_seat"]) for r in banked)
    if decisions == 0:
        fail("0 banked decisions (banked rows hold no actor decisions)")
    (out / "bank.jsonl").write_text("\n".join(json.dumps(r) for r in banked) + "\n")
    raw = [r for r in recs if r["kind"] == "raw"]
    hinted = [r for r in recs if r["kind"] == "hinted" and int(r["start_id"]) in set(hard)]
    summ = {
        "starts": len(by_start),
        "hard": len(hard),
        "hard_share": round(len(hard) / len(by_start), 3),
        "raw_games": len(raw),
        "raw_win_rate": round(sum(r["actor_win"] for r in raw) / len(raw), 3) if raw else 0.0,
        "hinted_games_on_hard": len(hinted),
        "hinted_success_share": round(sum(r["actor_win"] for r in hinted) / len(hinted), 3)
        if hinted else 0.0,
        "games_banked": len(banked),
        "decisions_banked": decisions,
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"hint_bank: {summ['starts']} starts, {summ['hard']} hard ({summ['hard_share']}), "
          f"raw wr {summ['raw_win_rate']} ({summ['raw_games']}), hinted success "
          f"{summ['hinted_success_share']} ({summ['hinted_games_on_hard']}), banked "
          f"{summ['games_banked']} games / {summ['decisions_banked']} decisions", flush=True)
    return summ


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, default=None, help="policy checkpoint (actor + search prior)")
    ap.add_argument("--opp-ckpt", type=Path, default=None, help="opponent checkpoint (default: --ckpt)")
    ap.add_argument("--value", type=Path, default=None, help="learned value checkpoint scoring branch ends")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--starts", type=Path, default=DEFAULT_STARTS, help="DAGS starts.jsonl")
    ap.add_argument("--n", type=int, default=200, help="starts sampled (train split)")
    ap.add_argument("--games", type=int, default=4, help="K: raw + hinted games per start (same seeds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4, help="search: policy candidates per searched decision")
    ap.add_argument("--samples", type=int, default=4, help="search: rollouts per candidate")
    ap.add_argument("--horizon", type=int, default=1, help="search: turns played out")
    ap.add_argument("--rollout-temp", type=float, default=1.0)
    ap.add_argument("--conf-skip", type=float, default=0.9, help="skip search when softmax(z).max() >= this")
    ap.add_argument("--det-tries", type=int, default=8, help="rejection-sampling proposals per honest world")
    ap.add_argument("--max-decisions", type=int, default=1500, help="per game, prefix excluded")
    ap.add_argument("--width", type=int, default=32, help="hinted games searched concurrently")
    ap.add_argument("--batch", type=int, default=256, help="max obs per policy forward")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--summarize", type=Path, default=None, help="merge <dir> to bank.jsonl + summary.json")
    ap.add_argument("--games-limit", type=int, default=0, help="debug: cap the hinted games")
    args = ap.parse_args(argv)
    if args.summarize is not None:
        summarize(args.summarize)
        return
    if args.ckpt is None or args.out is None:
        ap.error("--ckpt and --out are required (or --summarize <dir>)")
    if args.n < 1 or args.games < 1:
        ap.error("--n and --games must be >= 1")

    from ygo_zero.eval.bc_policy import BCPolicy

    t0 = time.time()
    engine = Engine()
    sampled, skipped = sample_train_starts(args.starts, args.n, args.seed)
    mine = sampled[args.proc :: args.nproc]
    print(f"hint_bank: {len(sampled)}/{args.n} train starts from {args.starts} "
          f"(skipped-empty-decks {skipped}), proc {args.proc}/{args.nproc} runs {len(mine)}",
          flush=True)
    pol = BCPolicy(args.ckpt, args.emb, temperature=0.0, threads=1, device=args.device,
                   batch=args.batch)
    opp_pol = pol if args.opp_ckpt in (None, args.ckpt) else BCPolicy(
        args.opp_ckpt, args.emb, temperature=0.0, threads=1, device=args.device, batch=args.batch)
    value = None
    if args.value:
        from ygo_zero.eval.value import ValueNet

        value = ValueNet(args.value, args.emb, device=args.device, batch=args.batch).values

    def base_logits(obs_list: list[dict], who) -> np.ndarray:
        p = pol if who == "actor" else opp_pol
        out = [p.logits(_batch(obs_list[i : i + args.batch])) for i in range(0, len(obs_list), args.batch)]
        lg = np.concatenate(out)
        if bool(np.isnan(lg).any()):
            raise RuntimeError("ALERT hint_bank: NaN logits from the policy forward")
        return lg

    route: dict[int, tuple[int, int] | None] = {}

    def forward(obs_list: list[dict]) -> np.ndarray:
        # Actor outer decisions + all search rollouts go to the actor policy; the
        # opponent's outer decisions go to the base policy (search_match routing).
        actor_at, actor_obs, opp_at, opp_obs = [], [], [], []
        for i, o in enumerate(obs_list):
            tag = route.pop(id(o), ...)
            if tag is ...:
                fail("untagged obs in the batch: every request passes through tag_game")
            if tag is None:
                actor_at.append(i)
                actor_obs.append(o)
            else:
                idx, actor_seat = tag
                if seat_of(o) == actor_seat:
                    actor_at.append(i)
                    actor_obs.append(o)
                else:
                    opp_at.append(i)
                    opp_obs.append(o)
        if len(route) > 10_000:
            fail(f"{len(route)} stale route tags (obs tagged but never forwarded)")
        a_width = int(obs_list[0]["action_mask"].shape[0])
        lg = np.zeros((len(obs_list), a_width), np.float32)
        for at, obs, who in ((actor_at, actor_obs, "actor"), (opp_at, opp_obs, "opp")):
            if not obs:
                continue
            b = base_logits(obs, who)
            if b.shape[1] != a_width:
                fail(f"policy width {b.shape[1]} != action options {a_width}")
            for k, i in enumerate(at):
                lg[i] = b[k]
        return lg

    args.out.mkdir(parents=True, exist_ok=True)
    gpath = args.out / f"games-{args.proc}.jsonl"
    bpath = args.out / f"bank-{args.proc}.jsonl"
    hard_ids: list[int] = []
    hinted_meta: list[tuple[int, int]] = []  # (file_idx, game_seed) per drive_many job
    hinted_gens: list = []
    n_raw_wins = 0
    n_raw = 0
    try:
        with gpath.open("w") as gf, bpath.open("w") as bf:
            for file_idx, row in mine:
                prefix = [int(c) for c in row.get("prefix", [])]
                try:
                    cfg, _, _, actor, _ = start_state(engine, row)
                except ValueError as e:
                    print(f"ALERT hint_bank: start {file_idx} prefix does not replay: {e}",
                          flush=True)
                    continue
                raw_wins = 0
                for g in range(args.games):
                    rec = raw_game(engine, cfg, prefix, actor, pol, opp_pol,
                                   game_idx=g, max_dec=args.max_decisions)
                    won = rec["winner"] == actor
                    raw_wins += won
                    n_raw_wins += won
                    n_raw += 1
                    gf.write(json.dumps({"start_id": file_idx, "kind": "raw", "game_seed": g,
                                         "actor": actor, "winner": rec["winner"],
                                         "actor_win": bool(won),
                                         "truncated": rec["truncated"]}) + "\n")
                if raw_wins > 0:
                    continue  # not hard: no hinted games, nothing banked
                hard_ids.append(file_idx)
                for g in range(args.games):
                    ws = RouteWindowSearch(
                        engine, _never, k=args.k, horizon=args.horizon, samples=args.samples,
                        rollout_temp=args.rollout_temp, batch=args.batch, seed=args.seed,
                        value=value, own_frac=1.0, route=route,
                    )  # fmt: skip
                    gen = tag_game(len(hinted_gens), actor, search_game_gen(
                        engine, cfg, ws, actor, argmax_select, game_seed=g,
                        own_turn1=True, conf_skip=args.conf_skip, own_frac=1.0,
                        honest=True, det_tries=args.det_tries, det_fallback="skip",
                        max_decisions=args.max_decisions + len(prefix), prefix=prefix,
                    ), route)  # fmt: skip
                    hinted_meta.append((file_idx, g))
                    hinted_gens.append((gen, row, cfg, prefix, actor))
            # Phase 2: hinted games on hard starts (batched through drive_many).
            jobs = ((idx, gen) for idx, (gen, *_) in enumerate(hinted_gens))
            if args.games_limit:
                jobs = ((idx, gen) for idx, (gen, *_) in
                        enumerate(hinted_gens[: args.games_limit]))
                hinted_meta = hinted_meta[: args.games_limit]
            n_hinted = n_banked = 0
            for idx, res in drive_many(jobs, forward, args.width):
                file_idx, g = hinted_meta[idx]
                _, row, cfg, prefix, actor = hinted_gens[idx]
                if res.winner is None and not res.truncated:
                    fail(f"hinted game {idx} ended with neither winner nor truncation")
                won = res.winner == actor
                gf.write(json.dumps({"start_id": file_idx, "kind": "hinted", "game_seed": g,
                                     "actor": actor, "winner": res.winner,
                                     "actor_win": bool(won), "truncated": res.truncated,
                                     "n_searched": res.n_searched,
                                     "n_override": res.n_override,
                                     "det_skipped": res.det_skipped}) + "\n")
                n_hinted += 1
                if won:
                    bf.write(json.dumps({
                        "start_id": file_idx, "config": config_to_dict(cfg),
                        "prefix": prefix, "choices": res.choices, "actors": res.actors,
                        "actor_seat": actor, "winner": res.winner,
                        "actor_win": True, "turn_start": int(row.get("turn") or 0),
                    }) + "\n")
                    n_banked += 1
                if (n_hinted % 10) == 0:
                    print(f"{n_hinted}/{len(hinted_meta)} hinted games, banked {n_banked}, "
                          f"{time.time() - t0:.0f} s", flush=True)
    except RuntimeError as e:
        if str(e).startswith("ALERT hint_bank:"):
            print(str(e), flush=True)
            raise SystemExit(1) from None
        raise
    print(f"proc {args.proc}: {len(mine)} starts, {n_raw} raw games "
          f"(actor wr {n_raw_wins / max(n_raw, 1):.3f}), {len(hard_ids)} hard, "
          f"{n_hinted} hinted, {n_banked} banked, wall {time.time() - t0:.0f} s", flush=True)
    if args.nproc == 1:
        summarize(args.out)


def _never(buf) -> np.ndarray:
    raise AssertionError("WindowSearch.logits is unused: play.py drives rollouts via drive_many")


if __name__ == "__main__":
    main()
