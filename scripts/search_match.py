"""Live match: base BC policy + WindowSearch (S) vs the same policy raw (R).

    uv run python scripts/search_match.py --ckpt runs/<run>/bc/best.pt --out runs/<run>/search-match \\
        [--value runs/<run>/value/best.pt] [--decks 8] [--seeds 3] [--k 4] [--samples 4] [--nproc 8 --proc 0]
    uv run python scripts/search_match.py --summarize runs/<run>/search-match

Decks are the report card's base-match sample (scripts/playtest.py: half train,
half holdout pool decks with verified lines, seeded the same way); every
ordered deck pair plays `--seeds` seeds twice with S in each seat (seat and
deck advantages cancel, as in playtest matches). S searches its window and
own-turn decisions (src/ygo_zero/search/play.py); R plays the raw policy.
`<out>/games-<proc>.jsonl` holds one line per game; `--summarize` merges all
procs into `summary.json` (S win rate with a Wilson 95% CI, per-deck win
rates, searched decisions per game, override rate, seconds per searched
decision, capped-branch share).

`--opp ygoagent` swaps R (the raw policy) for ygo-agent 0546 on the SAME
deals as playtest's `--ygoagent` match (decks/gauntlet, ring pairs, seeds,
both seats). The search's internal rollouts still model the opponent with
the base policy; only R's outer-game decisions come from ygo-agent. Rows
gain the paired-gate fields deck_s/deck_r/s_first/win/cut.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import NoReturn

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from playtest import verified_end_boards  # noqa: E402  # same deck sample as the report card

from ygo_zero.decks import load_ydk  # noqa: E402
from ygo_zero.engine.duel import DuelConfig, Engine  # noqa: E402
from ygo_zero.eval.match import ring_pairs, wilson  # noqa: E402
from ygo_zero.search.play import argmax_select, search_game_gen  # noqa: E402
from ygo_zero.search.turn1 import _batch, drive_many  # noqa: E402
from ygo_zero.search.window import WindowSearch  # noqa: E402


def ga_schedule(names: list[str], ga_opps: int, ga_seeds: int) -> list[tuple[int, int, int, int, int]]:
    """The playtest --ygoagent deal schedule: each deck vs its next k decks (ring_pairs),
    ga_seeds seeds, both seats. Entry: (di, dj, seed, s_seat, idx); S plays deck di."""
    sched = []
    for di, dj in ring_pairs(len(names), ga_opps):
        for s in range(ga_seeds):
            for s_seat in (0, 1):
                sched.append((di, dj, s, s_seat, len(sched)))
    return sched


class RouteWindowSearch(WindowSearch):
    """WindowSearch that tags every rollout obs id into `route` as None (inside the search).

    search_game_gen yields outer decisions ([obs]) and rollout batches through one
    channel; the shared forward tells them apart by these tags (see tag_game): a
    rollout obs goes to the base policy, an outer obs to S's or R's policy by seat.
    """

    def __init__(self, *a, route: dict, **k) -> None:
        super().__init__(*a, **k)
        self._route = route

    def rollouts_gen(self, *a, **k):
        gen = super().rollouts_gen(*a, **k)
        try:
            req = next(gen)
            while True:
                for o in req:
                    self._route[id(o)] = None
                fb = yield req
                req = gen.send(fb)
        except StopIteration as stop:
            return stop.value


def tag_game(idx: int, s_seat: int, gen, route: dict):
    """Tag every obs id a game yields with (idx, s_seat); rollout obs are pre-tagged
    None by RouteWindowSearch (setdefault keeps that tag: marking runs first, as the
    inner generator drives rollouts_gen before this wrapper sees the request)."""
    try:
        req = next(gen)
        while True:
            for o in req:
                route.setdefault(id(o), (idx, s_seat))
            fb = yield req
            req = gen.send(fb)
    except StopIteration as stop:
        return stop.value


def seat_of(obs: dict) -> int:
    """Absolute seat of the acting player: global[4] ("I went first") is 1 iff seat 0 acts."""
    return 0 if int(obs["global"][4]) == 1 else 1


def sample_decks(n: int, seed: int) -> list[str]:
    """The report card's base-match deck sample for --decks/--seed (see playtest.py)."""
    pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    train_d = [d for d in pool if d not in hold and verified_end_boards(d)]
    hold_d = [d for d in pool if d in hold and verified_end_boards(d)]
    rng = random.Random(seed)
    return rng.sample(train_d, min(n // 2, len(train_d))) + rng.sample(
        hold_d, min(n // 2, len(hold_d))
    )


def s_score(rec: dict) -> float:
    """S's score of a game record: 1 win, 0 loss, 0.5 draw/cut (as in playtest matches)."""
    w, seat = rec["winner"], rec["s_seat"]
    if w == seat:
        return 1.0
    return 0.0 if w in (0, 1) else 0.5


def fail(msg: str) -> NoReturn:
    print(f"ALERT search_match: {msg}", flush=True)
    raise SystemExit(1)


def summarize(out: Path) -> dict:
    """Merge <out>/games-*.jsonl into summary.json; fail loud on empty or degenerate results."""
    files = sorted(out.glob("games-*.jsonl"))
    recs = [json.loads(x) for f in files for x in f.read_text().splitlines() if x.strip()]
    if not recs:
        fail(f"no games in {out}/games-*.jsonl")
    bad = [r for r in recs if r["winner"] is None and not r.get("truncated")]
    if bad:
        fail(f"{len(bad)} games ended with neither winner nor truncation")
    no_ya = [r for r in recs if r.get("ya_decisions") == 0]
    if no_ya:
        fail(f"{len(no_ya)} games where ygo-agent made 0 decisions (routing broke)")
    n = len(recs)
    wins = sum(s_score(r) for r in recs)
    searched = sum(r["n_searched"] for r in recs)
    raw = all(r.get("raw") for r in recs)  # --raw: the paired no-search baseline
    if raw and searched:
        fail(f"--raw games searched {searched} decisions")
    if searched == 0 and not raw:
        fail("0 searched decisions overall")
    branches = sum(r.get("n_branches", 0) for r in recs)
    capped = sum(r.get("n_capped", 0) for r in recs)
    capped_share = capped / branches if branches else 0.0
    if capped_share > 0.2:
        fail(f"capped-branch share {capped_share:.3f} > 20% ({capped}/{branches})")
    props = sum(r.get("det_proposals", 0) for r in recs)
    worlds_acc = sum(r.get("det_accepted", 0) for r in recs)
    worlds_fb = sum(r.get("det_fallbacks", 0) for r in recs)
    worlds = worlds_acc + worlds_fb
    skipped = sum(r.get("det_skipped", 0) for r in recs)
    skip_mode = any(r.get("det_fallback", "true") == "skip" for r in recs) or skipped > 0
    lo, hi = wilson(wins, n)
    by_deck: dict[str, list[float]] = {}
    for r in recs:
        by_deck.setdefault(r["decks"][r["s_seat"]], []).append(s_score(r))
    summ = {
        "games": n,
        "s_win_rate": round(wins / n, 3),
        "ci": [round(lo, 3), round(hi, 3)],
        "per_deck": {d: round(sum(v) / len(v), 3) for d, v in sorted(by_deck.items())},
        "per_deck_n": {d: len(v) for d, v in sorted(by_deck.items())},
        "searched_per_game": round(searched / n, 2),
        "override_rate": round(sum(r["n_override"] for r in recs) / max(searched, 1), 3),
        "sec_per_search": round(sum(r["search_s"] for r in recs) / max(searched, 1), 3),
        "capped_share": round(capped_share, 4),
        "mean_turns": round(statistics.mean(r["turns"] for r in recs), 1),
        "cut": sum(1 for r in recs if r.get("truncated")),
    }
    for label, seat in (("first", 0), ("second", 1)):
        sub = [s_score(r) for r in recs if r["s_seat"] == seat]
        if sub:
            summ[label] = {"win_rate": round(sum(sub) / len(sub), 3), "n": len(sub)}
    if worlds:
        summ["det_accept_rate"] = round(worlds_acc / props, 3) if props else 0.0
        summ["det_fallback_rate"] = round(worlds_fb / worlds, 3)
    if skipped or skip_mode:
        summ["det_skipped"] = skipped
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(
        f"S win rate {summ['s_win_rate']} [{summ['ci'][0]}, {summ['ci'][1]}] n={n}; "
        f"searched {summ['searched_per_game']}/game, override {summ['override_rate']}, "
        f"{summ['sec_per_search']} s/search, capped {summ['capped_share']}, "
        f"mean turns {summ['mean_turns']}, cut {summ['cut']}",
        flush=True,
    )
    if worlds:
        print(f"honest worlds: accept {summ['det_accept_rate']} ({worlds_acc}/{props}), "
              f"fallback {summ['det_fallback_rate']} ({worlds_fb}/{worlds})", flush=True)
    if skipped or skip_mode:
        attempted = searched + skipped
        skip_rate = skipped / attempted if attempted else 0.0
        print(f"honest skipped searches {skipped}/{attempted} ({skip_rate:.3f})", flush=True)
    for d in sorted(summ["per_deck"]):
        print(f"  {d}: {summ['per_deck'][d]} (n={summ['per_deck_n'][d]})", flush=True)
    if "first" in summ:
        print(f"S first {summ['first']['win_rate']} (n={summ['first']['n']}); "
              f"S second {summ['second']['win_rate']} (n={summ['second']['n']})", flush=True)
    # after writing the summary, so a high fallback rate is reported with its numbers (the match is part-oracle)
    if skip_mode:
        attempted = searched + skipped
        if attempted and skipped / attempted > 0.5:
            fail(f"honest skipped searches {skipped / attempted:.3f} > 50% ({skipped}/{attempted})")
    elif worlds and worlds_fb / worlds > 0.2:
        fail(f"honest-world fallback rate {worlds_fb / worlds:.3f} > 20% ({worlds_fb}/{worlds})")
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="base BC checkpoint (both players' policy)")
    ap.add_argument("--value", type=Path, default=None, help="learned value checkpoint scoring branch ends")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decks", type=int, default=8, help="sampled pool decks (half train, half holdout)")
    ap.add_argument("--seeds", type=int, default=3, help="match seeds per ordered deck pair")
    ap.add_argument("--seed", type=int, default=0, help="deck sample + search RNG seed")
    ap.add_argument("--k", type=int, default=4, help="policy candidates per searched decision")
    ap.add_argument("--samples", type=int, default=4, help="rollouts per candidate (sample 0 is argmax)")
    ap.add_argument("--horizon", type=int, default=1, help="turns played out (1 = end of the current turn)")
    ap.add_argument("--rollout-temp", type=float, default=1.0)
    ap.add_argument("--raw", type=int, default=0,
                    help="1 = never search (conf-skip 0): S plays the raw policy, the paired no-search baseline")
    ap.add_argument("--conf-skip", type=float, default=0.9, help="skip search when softmax(z).max() >= this")
    ap.add_argument("--own-frac", type=float, default=1.0, help="share of S own-turn decisions searched")
    ap.add_argument("--own-turn1", type=int, default=0, choices=(0, 1), help="also search S turn-1 plays")
    ap.add_argument("--det-tries", type=int, default=8, help="rejection-sampling proposals per honest world")
    ap.add_argument("--det-fallback", default="true", choices=("true", "skip"),
                    help="failed world: fall back to the true config, or skip the decision")
    ap.add_argument("--honest", type=int, default=0, choices=(0, 1), help="search determinized worlds (honest search)")
    ap.add_argument("--opp", default="self", choices=("self", "ygoagent"),
                    help="R: the raw base policy (self) or ygo-agent 0546 (ygoagent, gauntlet deals)")
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks/gauntlet",
                    help="--opp ygoagent: ygo-agent's own lists (as in playtest --ygoagent)")
    ap.add_argument("--ga-opps", type=int, default=4, help="--opp ygoagent: each deck vs its next k decks")
    ap.add_argument("--ga-seeds", type=int, default=1, help="--opp ygoagent: seeds per pair (playtest uses 1)")
    ap.add_argument("--width", type=int, default=32, help="games searched concurrently")
    ap.add_argument("--batch", type=int, default=512, help="max obs per policy forward")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--games-limit", type=int, default=0, help="debug: cap the total games")
    ap.add_argument("--summarize", type=Path, default=None, help="merge <dir>/games-*.jsonl to summary.json")
    args = ap.parse_args()
    if args.summarize is not None:
        summarize(args.summarize)
        return
    if args.raw:
        args.conf_skip = 0.0  # softmax max is always >= 0: no decision is searched
    if args.ckpt is None or args.out is None:
        ap.error("--ckpt and --out are required (or --summarize <dir>)")

    from ygo_zero.eval.bc_policy import BCPolicy

    t0 = time.time()
    engine = Engine()
    pol = BCPolicy(args.ckpt, args.emb, threads=1, device=args.device, batch=args.batch)
    value = None
    if args.value:
        from ygo_zero.eval.value import ValueNet

        value = ValueNet(args.value, args.emb, device=args.device, batch=args.batch).values

    def base_logits(obs_list: list[dict]) -> np.ndarray:
        out = [pol.logits(_batch(obs_list[i : i + args.batch])) for i in range(0, len(obs_list), args.batch)]
        lg = np.concatenate(out)
        if bool(np.isnan(lg).any()):
            raise RuntimeError("ALERT search_match: NaN logits from the policy forward")
        return lg

    def never(buf) -> np.ndarray:
        raise AssertionError("WindowSearch.logits is unused: play.py drives rollouts via drive_many")

    route: dict[int, tuple[int, int] | None] = {}  # id(obs) -> None (rollout) | (game idx, s_seat)
    ya_n: dict[int, int] = {}  # game idx -> ygo-agent decisions (0 in a finished game = routing broke)
    opp = None
    if args.opp == "ygoagent":
        from ygo_zero.eval.ygoagent import YgoAgentPolicy

        opp = YgoAgentPolicy(seed=3)  # as in playtest's --ygoagent match
        gfiles = sorted(args.gauntlet_decks.glob("*.ydk"))
        if len(gfiles) < 2:
            fail(f"only {len(gfiles)} gauntlet decks in {args.gauntlet_decks}")
        names = [p.stem for p in gfiles]
        decks = [load_ydk(f) for f in gfiles]
        sched = ga_schedule(names, args.ga_opps, args.ga_seeds)
    else:
        names = sample_decks(args.decks, args.seed)
        if len(names) < 2:
            fail(f"only {len(names)} decks sampled")
        decks = None
        pair_names = [(a, b) for a in range(len(names)) for b in range(len(names)) if a != b]
        sched = []
        for di, dj in pair_names:
            for s in range(args.seeds):
                for s_seat in (0, 1):
                    sched.append((di, dj, s, s_seat, len(sched)))
    if args.games_limit:
        sched = sched[: args.games_limit]
    mine = sched[args.proc :: args.nproc]
    print(f"opp {args.opp}, decks {names}: {len(sched)} games, proc {args.proc}/{args.nproc} runs {len(mine)}",
          flush=True)

    def forward(obs_list: list[dict]) -> np.ndarray:
        if opp is None:
            return base_logits(obs_list)
        # R's outer decisions go to ygo-agent; everything else (S outer + all rollouts,
        # which model the opponent with the base policy) goes to the base forward.
        base_at, base_obs, ya_at, ya_obs, ya_keys, ya_me = [], [], [], [], [], []
        for i, o in enumerate(obs_list):
            tag = route.pop(id(o), ...)
            if tag is ...:
                fail("untagged obs in the batch (opp=ygoagent): every request passes through tag_game")
            if tag is None:
                base_at.append(i)
                base_obs.append(o)
            else:
                idx, s_seat = tag
                if seat_of(o) == s_seat:
                    base_at.append(i)
                    base_obs.append(o)
                else:
                    ya_at.append(i)
                    ya_obs.append(o)
                    ya_keys.append(idx)
                    ya_n[idx] = ya_n.get(idx, 0) + 1
                    ya_me.append(seat_of(o))
        if len(route) > 10_000:  # obs tagged but never forwarded: a recycled id() could misroute R to base
            fail(f"{len(route)} stale route tags (obs tagged but never forwarded)")
        a_width = int(obs_list[0]["action_mask"].shape[0])
        lg = np.zeros((len(obs_list), a_width), np.float32)
        if base_obs:
            b = base_logits(base_obs)
            if b.shape[1] != a_width:
                fail(f"policy width {b.shape[1]} != action options {a_width}")
            for k, i in enumerate(base_at):
                lg[i] = b[k]
        if ya_obs:
            buf = _batch(ya_obs)
            buf["player"] = np.array(ya_me, np.int64)  # CppEnv obs carry no seat; global[4] does
            acts = opp(buf, np.arange(len(ya_obs)), keys=np.array(ya_keys, np.int64))
            for i, a in zip(ya_at, acts, strict=True):  # one-hot argmax: the inner argmax plays R's pick
                lg[i, int(a)] = 1.0
        return lg

    def jobs():
        for di, dj, s, s_seat, idx in mine:
            if opp is None:
                cfg = DuelConfig(
                    load_ydk(ROOT / "decks" / "pool" / f"{names[di]}.ydk"),
                    load_ydk(ROOT / "decks" / "pool" / f"{names[dj]}.ydk"),
                    seed=s,
                )
                ws = WindowSearch(
                    engine, never, k=args.k, horizon=args.horizon, samples=args.samples,
                    rollout_temp=args.rollout_temp, batch=args.batch, seed=args.seed, value=value,
                )  # fmt: skip
                gen = search_game_gen(
                    engine, cfg, ws, s_seat, argmax_select, game_seed=idx,
                    own_turn1=bool(args.own_turn1), conf_skip=args.conf_skip, own_frac=args.own_frac,
                    honest=bool(args.honest), det_tries=args.det_tries, det_fallback=args.det_fallback,
                )  # fmt: skip
            else:
                seat0, seat1 = (di, dj) if s_seat == 0 else (dj, di)
                cfg = DuelConfig(decks[seat0], decks[seat1], seed=s)
                ws = RouteWindowSearch(
                    engine, never, k=args.k, horizon=args.horizon, samples=args.samples,
                    rollout_temp=args.rollout_temp, batch=args.batch, seed=args.seed, value=value,
                    route=route,
                )  # fmt: skip
                gen = tag_game(idx, s_seat, search_game_gen(
                    engine, cfg, ws, s_seat, argmax_select, game_seed=idx,
                    own_turn1=bool(args.own_turn1), conf_skip=args.conf_skip, own_frac=args.own_frac,
                    honest=bool(args.honest), det_tries=args.det_tries, det_fallback=args.det_fallback,
                ), route)  # fmt: skip
            yield idx, gen

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"games-{args.proc}.jsonl"
    got, tot_s = 0, 0
    try:
        with path.open("w") as f:
            for idx, res in drive_many(jobs(), forward, args.width):
                di, dj, s, s_seat, _ = sched[idx]
                if res.winner is None and not res.truncated:
                    fail(f"game {idx} ended with neither winner nor truncation")
                if opp is not None:
                    opp.reset_keys([idx])  # drop ygo-agent's LSTM state for the finished game
                rec = {
                    "decks": [names[di], names[dj]] if opp is None else
                             [names[di if s_seat == 0 else dj], names[dj if s_seat == 0 else di]],
                    "seed": s, "s_seat": s_seat, "game_idx": idx,
                    "winner": res.winner, "truncated": res.truncated, "turns": res.turns,
                    "choices": res.choices, "actors": res.actors,
                    "raw": bool(args.raw), "n_searched": res.n_searched, "n_override": res.n_override,
                    "search_s": round(res.search_s, 3),
                    "n_branches": res.n_branches, "n_capped": res.n_capped,
                    "det_proposals": res.det_proposals, "det_accepted": res.det_accepted,
                    "det_fallbacks": res.det_fallbacks, "det_skipped": res.det_skipped,
                    "det_fallback": args.det_fallback, "det_s": round(res.det_s, 3),
                }
                if opp is not None:  # paired-gate fields (playtest per_game format + search stats)
                    w = res.winner
                    rec.update(deck_s=names[di], deck_r=names[dj], s_first=(s_seat == 0), ya_decisions=ya_n.pop(idx, 0),
                               win=1.0 if w == s_seat else (0.0 if w in (0, 1) else 0.5),
                               cut=res.truncated)
                f.write(json.dumps(rec) + "\n")
                got += 1
                tot_s += res.n_searched
                if got % 10 == 0:
                    print(f"{got}/{len(mine)} games, searched {tot_s}, {time.time() - t0:.0f} s", flush=True)
    except RuntimeError as e:
        if str(e).startswith("ALERT search_match:"):
            print(str(e), flush=True)
            raise SystemExit(1) from None
        raise
    branches = 0  # capped share needs branch totals: re-read this proc's file (cheap, local)
    capped = 0
    for raw in path.read_text().splitlines():
        r = json.loads(raw)
        branches += r.get("n_branches", 0)
        capped += r.get("n_capped", 0)
    share = capped / branches if branches else 0.0
    print(f"proc {args.proc}: {got} games, {tot_s} searched, capped share {share:.4f} "
          f"({capped}/{branches}), wall {time.time() - t0:.0f} s", flush=True)
    if tot_s == 0 and args.nproc == 1 and not args.raw:
        fail("0 searched decisions overall")
    if share > 0.2:
        fail(f"capped-branch share {share:.3f} > 20% ({capped}/{branches})")


if __name__ == "__main__":
    main()
