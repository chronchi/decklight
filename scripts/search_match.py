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
from ygo_zero.eval.match import wilson  # noqa: E402
from ygo_zero.search.play import argmax_select, search_game_gen  # noqa: E402
from ygo_zero.search.turn1 import _batch, drive_many  # noqa: E402
from ygo_zero.search.window import WindowSearch  # noqa: E402


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
    n = len(recs)
    wins = sum(s_score(r) for r in recs)
    searched = sum(r["n_searched"] for r in recs)
    if searched == 0:
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
    lo, hi = wilson(wins, n)
    by_deck: dict[str, list[float]] = {}
    for r in recs:
        by_deck.setdefault(r["decks"][r["s_seat"]], []).append(s_score(r))
    if worlds and worlds_fb / worlds > 0.2:
        fail(f"honest-world fallback rate {worlds_fb / worlds:.3f} > 20% ({worlds_fb}/{worlds})")
    summ = {
        "games": n,
        "s_win_rate": round(wins / n, 3),
        "ci": [round(lo, 3), round(hi, 3)],
        "per_deck": {d: round(sum(v) / len(v), 3) for d, v in sorted(by_deck.items())},
        "per_deck_n": {d: len(v) for d, v in sorted(by_deck.items())},
        "searched_per_game": round(searched / n, 2),
        "override_rate": round(sum(r["n_override"] for r in recs) / searched, 3),
        "sec_per_search": round(sum(r["search_s"] for r in recs) / searched, 3),
        "capped_share": round(capped_share, 4),
        "mean_turns": round(statistics.mean(r["turns"] for r in recs), 1),
        "cut": sum(1 for r in recs if r.get("truncated")),
    }
    if worlds:
        summ["det_accept_rate"] = round(worlds_acc / props, 3) if props else 0.0
        summ["det_fallback_rate"] = round(worlds_fb / worlds, 3)
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
    for d in sorted(summ["per_deck"]):
        print(f"  {d}: {summ['per_deck'][d]} (n={summ['per_deck_n'][d]})", flush=True)
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
    ap.add_argument("--conf-skip", type=float, default=0.9, help="skip search when softmax(z).max() >= this")
    ap.add_argument("--own-frac", type=float, default=1.0, help="share of S own-turn decisions searched")
    ap.add_argument("--own-turn1", type=int, default=0, choices=(0, 1), help="also search S turn-1 plays")
    ap.add_argument("--honest", type=int, default=0, choices=(0, 1), help="search determinized worlds (honest search)")
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

    def forward(obs_list: list[dict]) -> np.ndarray:
        out = [pol.logits(_batch(obs_list[i : i + args.batch])) for i in range(0, len(obs_list), args.batch)]
        lg = np.concatenate(out)
        if bool(np.isnan(lg).any()):
            raise RuntimeError("ALERT search_match: NaN logits from the policy forward")
        return lg

    def never(buf) -> np.ndarray:
        raise AssertionError("WindowSearch.logits is unused: play.py drives rollouts via drive_many")

    names = sample_decks(args.decks, args.seed)
    if len(names) < 2:
        fail(f"only {len(names)} decks sampled")
    pair_names = [(a, b) for a in range(len(names)) for b in range(len(names)) if a != b]
    sched = []
    for di, dj in pair_names:
        for s in range(args.seeds):
            for s_seat in (0, 1):
                sched.append((di, dj, s, s_seat, len(sched)))
    if args.games_limit:
        sched = sched[: args.games_limit]
    mine = sched[args.proc :: args.nproc]
    print(f"decks {names}: {len(sched)} games, proc {args.proc}/{args.nproc} runs {len(mine)}", flush=True)

    def jobs():
        for di, dj, s, s_seat, idx in mine:
            cfg = DuelConfig(
                load_ydk(ROOT / "decks" / "pool" / f"{names[di]}.ydk"),
                load_ydk(ROOT / "decks" / "pool" / f"{names[dj]}.ydk"),
                seed=s,
            )
            ws = WindowSearch(
                engine, never, k=args.k, horizon=args.horizon, samples=args.samples,
                rollout_temp=args.rollout_temp, batch=args.batch, seed=args.seed, value=value,
            )  # fmt: skip
            yield idx, search_game_gen(
                engine, cfg, ws, s_seat, argmax_select, game_seed=idx,
                own_turn1=bool(args.own_turn1), conf_skip=args.conf_skip, own_frac=args.own_frac,
                honest=bool(args.honest),
            )  # fmt: skip

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"games-{args.proc}.jsonl"
    got, tot_s = 0, 0
    try:
        with path.open("w") as f:
            for idx, res in drive_many(jobs(), forward, args.width):
                di, dj, s, s_seat, _ = sched[idx]
                if res.winner is None and not res.truncated:
                    fail(f"game {idx} ended with neither winner nor truncation")
                f.write(json.dumps({
                    "decks": [names[di], names[dj]], "seed": s, "s_seat": s_seat, "game_idx": idx,
                    "winner": res.winner, "truncated": res.truncated, "turns": res.turns,
                    "choices": res.choices, "actors": res.actors,
                    "n_searched": res.n_searched, "n_override": res.n_override,
                    "search_s": round(res.search_s, 3),
                    "n_branches": res.n_branches, "n_capped": res.n_capped,
                    "det_proposals": res.det_proposals, "det_accepted": res.det_accepted,
                    "det_fallbacks": res.det_fallbacks, "det_s": round(res.det_s, 3),
                }) + "\n")  # fmt: skip
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
    if tot_s == 0 and args.nproc == 1:
        fail("0 searched decisions overall")
    if share > 0.2:
        fail(f"capped-branch share {share:.3f} > 20% ({capped}/{branches})")


if __name__ == "__main__":
    main()
