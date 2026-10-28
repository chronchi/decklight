"""Live match: base BC policy + WindowSearch (S) vs the same policy raw (R).

    uv run python scripts/search_match.py --ckpt runs/<run>/bc/best.pt --out runs/<run>/search-match \\
        [--value runs/<run>/value/best.pt] [--decks 8] [--seeds 3] [--k 4] [--samples 4] [--nproc 8 --proc 0]
    uv run python scripts/search_match.py --summarize runs/<run>/search-match

Decks are the report card's base-match sample (scripts/playtest.py: half train,
half holdout pool decks with verified lines, seeded the same way); every
ordered deck pair plays `--seeds` seeds twice with S in each seat (seat and
deck advantages cancel, as in playtest matches). S searches its window and
own-turn decisions (src/decklight/search/play.py); R plays the raw policy.
`<out>/games-<proc>.jsonl` holds one line per game; `--summarize` merges all
procs into `summary.json` (S win rate with a Wilson 95% CI, per-deck win
rates, searched decisions per game, override rate, seconds per searched
decision, capped-branch share).

`--opp ygoagent` swaps R (the raw policy) for ygo-agent 0546 on the SAME
deals as playtest's `--ygoagent` match (decks/gauntlet, ring pairs, seeds,
both seats). The search's internal rollouts still model the opponent with
the base policy; only R's outer-game decisions come from ygo-agent. Rows
gain the paired-gate fields deck_s/deck_r/s_first/win/cut.

Teacher generation (PLAN §0a step 2): `--save-records DIR` writes every
finished game as a gen_window-format record to `DIR/<S deck>.jsonl` (soft
targets for train_bc --soft-targets); `--opp ckpt` (R = --opp-ckpt, raw
argmax) and `--opp greedy` (R = GreedyPolicy) add cheaper opponent mixes;
`--deal-mode train` draws fresh deals from pool − HOLDOUT − kewl-tune (or
decks/gauntlet for ygoagent) with duel seeds from `--seed-base`, `--games`
games total, both seats. Without these flags everything is bit-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NoReturn

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from playtest import verified_end_boards  # same deck sample as the report card

from decklight.decks import load_ydk
from decklight.engine.duel import DuelConfig, Engine
from decklight.eval.match import ring_pairs, wilson
from decklight.search.honest import HonestRule
from decklight.search.play import argmax_select, search_game_gen
from decklight.search.turn1 import _batch, drive_many
from decklight.search.window import WindowSearch


def ga_schedule(names: list[str], ga_opps: int, ga_seeds: int, seed0: int = 0) -> list[tuple[int, int, int, int, int]]:
    """The playtest --ygoagent deal schedule: each deck vs its next k decks (ring_pairs),
    ga_seeds seeds, both seats. Entry: (di, dj, seed, s_seat, idx); S plays deck di."""
    sched = []
    for di, dj in ring_pairs(len(names), ga_opps):
        for s in range(seed0, seed0 + ga_seeds):
            for s_seat in (0, 1):
                sched.append((di, dj, s, s_seat, len(sched)))
    return sched


# D3: RouteWindowSearch/tag_game/seat_of moved verbatim to src/decklight/search/product.py
# (the EDOPro host driver reuses them); search_match behaviour is bit-identical.
from decklight.search.product import RouteWindowSearch, seat_of, tag_game


def pairs_schedule(names: list[str], want: list[tuple[str, str]], seeds: int, seed0: int,
                   seed_per_pair: bool) -> list[tuple[int, int, int, int, int]]:
    """Explicit deck pairs (S plays the first deck of each pair, both seats): entry
    (di, dj, seed, s_seat, idx), same shape as the eval schedules. --seed-per-pair gives
    pair k its own seed range so a deck meets a new opening hand in every game."""
    missing = [d for pr in want for d in pr if d not in names]
    if missing or any(len(pr) != 2 for pr in want):
        fail(f"--pairs: bad pairs or decks not in the deck set: {missing or want}")
    sched = []
    for k, (a, b) in enumerate(want):
        di, dj = names.index(a), names.index(b)
        s0 = seed0 + (k * seeds if seed_per_pair else 0)
        for s in range(s0, s0 + seeds):
            for s_seat in (0, 1):
                sched.append((di, dj, s, s_seat, len(sched)))
    return sched


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


def train_schedule(names: list[str], games: int, seed_base: int, hold: set[str]) -> list[tuple[int, int, int, int, int]]:
    """Data-generation deals: random distinct deck pairs, duel seeds from seed_base, seats alternate.
    Entry: (di, dj, seed, s_seat, idx), same shape as the eval schedules. Holdout decks are never drawn."""
    rng = random.Random(seed_base)
    sched = []
    for i in range(games):
        di = rng.randrange(len(names))
        dj = rng.randrange(len(names) - 1)
        if dj >= di:
            dj += 1
        if names[di] in hold or names[dj] in hold:
            fail(f"train schedule drew a holdout deck: {names[di]} vs {names[dj]}")
        sched.append((di, dj, seed_base + i, i % 2, i))
    return sched


def config_dict(cfg) -> dict:
    """A DuelConfig as a JSON-able dict (same keys as verified records' configs)."""
    return {
        "deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
        "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
        "seed": int(cfg.seed), "lp": int(cfg.lp), "hand": int(cfg.hand), "draw": int(cfg.draw),
        "setup": [list(p) for p in cfg.setup], "shuffle": bool(cfg.shuffle),
    }  # fmt: skip


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
    no_opp = [r for r in recs if r.get("opp_decisions") == 0]
    if no_opp:
        fail(f"{len(no_opp)} games where R made 0 decisions (routing broke)")
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
        "branches_per_search": round(branches / max(searched, 1), 2),
        "mean_turns": round(statistics.mean(r["turns"] for r in recs), 1),
        "cut": sum(1 for r in recs if r.get("truncated")),
    }
    for label, seat in (("first", 0), ("second", 1)):
        sub = [s_score(r) for r in recs if r["s_seat"] == seat]
        if sub:
            summ[label] = {"win_rate": round(sum(sub) / len(sub), 3), "n": len(sub)}
    ho_dec = [e["ho"] for r in recs for e in r.get("search", []) if "ho" in e]
    if ho_dec:
        st = [d["status"] for d in ho_dec]
        summ["ho"] = {kk: st.count(kk) for kk in ("keep", "override", "undecided")}
        summ["ho"]["escalated"] = sum(1 for d in ho_dec if d.get("rounds", 0) > 0)
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
    seats = [f"S {lb} {summ[lb]['win_rate']} (n={summ[lb]['n']})" for lb in ("first", "second") if lb in summ]
    if seats:  # one seat only under --s-seat-only
        print("; ".join(seats), flush=True)
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
    ap.add_argument("--value-priv", type=int, default=0,
                    help="1: --value is a privileged critic (train_rl --priv-critic critic.pt) reading each sampled "
                         "world's hidden cards; needs --honest 1 (else it reads the real game's hidden cards)")
    ap.add_argument("--emb", type=Path, default=None,
                    help="card embeddings (default: $YGO_EMB or data/cards/emb/qwen3-0.6b-d512.npz)")
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
    ap.add_argument("--guard-legacy", type=int, default=0, choices=(0, 1),
                    help="routed opponents: 1 = one loop guard shared by both seats (E2 and before); 0 = card-faithful")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="override the policy's move only if the search's best beats it by >= this (selection score)")
    ap.add_argument("--max-search-turn", type=int, default=0, help="search only decisions on turns <= this (0 = all)")
    ap.add_argument("--stop-turn", type=int, default=0,
                    help="label generation: end each game after this turn (truncated, no winner; 0 = play out)")
    ap.add_argument("--s-seat-only", type=int, default=-1, choices=(-1, 0, 1),
                    help="play only games with S in this seat (0 = S goes first); -1 all")
    ap.add_argument("--own-turn1", type=int, default=0, choices=(0, 1), help="also search S turn-1 plays")
    ap.add_argument("--det-tries", type=int, default=8, help="rejection-sampling proposals per honest world")
    ap.add_argument("--replay-threads", type=int, default=1,
                    help="thread-parallel world checks and branch replays (speed plan E6, bit-identical); "
                         "1 = today's sequential path (no pool)")
    ap.add_argument("--det-fallback", default="true", choices=("true", "skip"),
                    help="failed world: fall back to the true config, or skip the decision")
    ap.add_argument("--honest", type=int, default=0, choices=(0, 1), help="search determinized worlds (honest search)")
    ap.add_argument("--opp", default="self", choices=("self", "ygoagent", "ckpt", "greedy"),
                    help="R: the raw base policy (self), ygo-agent 0546 (ygoagent, gauntlet deals), "
                         "another BC checkpoint raw argmax (ckpt, --opp-ckpt), or GreedyPolicy (greedy)")
    ap.add_argument("--opp-ckpt", type=Path, default=None, help="--opp ckpt: R's BC checkpoint")
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks/gauntlet",
                    help="--opp ygoagent: ygo-agent's own lists (as in playtest --ygoagent)")
    ap.add_argument("--ga-opps", type=int, default=4, help="--opp ygoagent: each deck vs its next k decks")
    ap.add_argument("--ga-seeds", type=int, default=1, help="--opp ygoagent: seeds per pair (playtest uses 1)")
    ap.add_argument("--ho", type=int, default=0, choices=(0, 1),
                    help="honest override (search/honest.py): merged candidates, paired-significant overrides only")
    ap.add_argument("--ho-z", type=float, default=2.0, help="--ho: SE multiple an override gap must reach")
    ap.add_argument("--ho-min-gap", type=float, default=0.05, help="--ho: absolute gap floor (value units)")
    ap.add_argument("--ho-s-max", type=int, default=8, help="--ho: max samples per candidate (adaptive)")
    ap.add_argument("--ho-step", type=int, default=4, help="--ho: samples added per escalation round")
    ap.add_argument("--max-searches", type=int, default=0,
                    help="per game: after this many search attempts play raw (0 = no cap; a looping game slowed a proc "
                         "for 45+ min, NOTEBOOK 2026-10-11 00:10; normal games search ~32 times)")
    ap.add_argument("--deal-seed0", type=int, default=0,
                    help="eval deals: first duel seed (0 = the card's deals; e.g. 1000 = fresh deals for a final card)")
    ap.add_argument("--deck-set", default="sample", choices=("sample", "holdout", "newcards"),
                    help="eval deals without ygoagent: sample = --decks pool sample (the card's); holdout = every "
                         "HOLDOUT pool deck + kewl-tune; newcards = decks/newcards (unseen archetypes, ADR 0007). "
                         "Every ordered pair, --seeds seeds, both seats")
    ap.add_argument("--deal-mode", default="eval", choices=("eval", "train"),
                    help="eval: the report-card / gauntlet schedules; train: fresh --games deals from "
                         "pool - HOLDOUT - kewl-tune (gauntlet lists for ygoagent), seeds from --seed-base")
    ap.add_argument("--seed-base", type=int, default=100000, help="--deal-mode train: first duel seed")
    ap.add_argument("--games", type=int, default=0, help="--deal-mode train: total games over all procs")
    ap.add_argument("--save-records", type=Path, default=None,
                    help="write every game as a gen_window-format teacher record to DIR/<S deck>.jsonl")
    ap.add_argument("--width", type=int, default=32, help="games searched concurrently")
    ap.add_argument("--claim", type=int, choices=(0, 1), default=None,
                    help="1: procs claim games one by one from <out>/claims (dynamic work queue, no idle tail; "
                         "at most --claim-width in flight); 0: static i %% nproc shards. Default: 1 with "
                         "--save-records (teacher generation), else 0 (eval schedules stay per-proc deterministic)")
    ap.add_argument("--claim-width", type=int, default=8, help="--claim 1: max games in flight per proc")
    ap.add_argument("--batch", type=int, default=512, help="max obs per policy forward")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--games-limit", type=int, default=0, help="debug: cap the total games")
    ap.add_argument("--seed-per-pair", type=int, choices=(0, 1), default=0,
                    help="1: each pair gets its own --seeds deal seeds (from --deal-seed0 + pair index x seeds)")
    ap.add_argument("--pairs", default="",
                    help="explicit deck pairs 'a:b,c:d' or @file (one or more per line; seat 0 deck first) from decks/pool, or decks/newcards with "
                         "--deck-set newcards; showcase games")
    ap.add_argument("--deck-dir", type=Path, default=None,
                    help="--pairs: look deck names up in this dir first, then in decks/pool")
    ap.add_argument("--both-search", type=int, choices=(0, 1), default=0,
                    help="--opp self: both seats search (the product vs itself); log entries carry the seat")
    ap.add_argument("--prof", type=int, choices=(0, 1), default=0,
                    help="1: per-search timings in the log (play is bit-identical), per-proc forward counters "
                         "to prof-<proc>.json, SIGTERM dumps in-flight choice prefixes to inflight-<proc>.jsonl")
    ap.add_argument("--latency", type=int, choices=(0, 1), default=0,
                    help="1: latency mode (implies width 1, claim 0): per-raw-move, per-searched-move and per-own-turn "
                         "wall seconds in the game record (lat_raw, lat_search, lat_turn)")
    ap.add_argument("--summarize", type=Path, default=None, help="merge <dir>/games-*.jsonl to summary.json")
    args = ap.parse_args()
    if args.summarize is not None:
        summarize(args.summarize)
        return
    if args.emb is None:
        args.emb = Path(os.environ.get("YGO_EMB") or ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    if args.raw:
        args.conf_skip = 0.0  # softmax max is always >= 0: no decision is searched
    if args.ckpt is None or args.out is None:
        ap.error("--ckpt and --out are required (or --summarize <dir>)")
    if os.environ.get("YGO_BLOCKING_SYNC") == "1":
        # M0 phase C: blocking CUDA sync instead of spin-wait (must run before torch creates its CUDA context).
        # cudaSetDeviceFlags is a runtime-API symbol: it lives in torch's libcudart, not the driver's libcuda
        # (speed-profile-20261016-1048 phase C). Importing torch loads libcudart.so.12, so the CDLL below reuses it.
        import ctypes

        import torch

        try:
            rc = ctypes.CDLL("libcudart.so.12").cudaSetDeviceFlags(4)  # cudaDeviceScheduleBlockingSync
            if rc != 0:
                fail(f"YGO_BLOCKING_SYNC=1: cudaSetDeviceFlags returned {rc}")
            print("YGO_BLOCKING_SYNC=1: blocking CUDA sync set", flush=True)
        except (OSError, AttributeError) as e:
            fail(f"YGO_BLOCKING_SYNC=1 but libcudart.so.12 / cudaSetDeviceFlags is missing: {e}")
    PROF = bool(args.prof)
    LAT = bool(args.latency)

    from decklight.eval.bc_policy import BCPolicy

    t0 = time.time()
    engine = Engine()
    pol = BCPolicy(args.ckpt, args.emb, threads=1, device=args.device, batch=args.batch)
    value = None
    if args.value_priv and not (args.value and args.honest):
        raise SystemExit("--value-priv 1 needs --value <critic.pt> and --honest 1 (it would see the real hidden cards)")
    if args.value:
        from decklight.eval.value import PrivValueNet, ValueNet

        vcls = PrivValueNet if args.value_priv else ValueNet
        value = vcls(args.value, args.emb, device=args.device, batch=args.batch).values

    def base_logits(obs_list: list[dict]) -> np.ndarray:
        out = [_timed_logits(pol.logits, _timed_batch(obs_list[i : i + args.batch]))
               for i in range(0, len(obs_list), args.batch)]
        lg = np.concatenate(out)
        if bool(np.isnan(lg).any()):
            raise RuntimeError("ALERT search_match: NaN logits from the policy forward")
        return lg

    def never(buf) -> np.ndarray:
        raise AssertionError("WindowSearch.logits is unused: play.py drives rollouts via drive_many")

    # --prof 1 per-process forward counters (written to prof-<proc>.json at exit and on SIGTERM).
    prof_fwd: dict = {"calls": 0, "hist": {}, "t_batch": 0.0, "t_fwd": 0.0, "t_gpu": 0.0,
                      "t_sync": 0.0, "t_h2d": 0.0, "t_d2h": 0.0, "t_ygo": 0.0}
    _use_cuda = False
    if PROF:
        import torch

        _use_cuda = torch.cuda.is_available()

    def _timed_batch(obs_list: list[dict]) -> dict:
        if not PROF:
            return _batch(obs_list)
        t = time.perf_counter()
        buf = _batch(obs_list)
        prof_fwd["t_batch"] += time.perf_counter() - t
        return buf

    def _timed_logits(logits_fn, buf: dict) -> np.ndarray:
        """One policy forward with --prof timing: wall into t_fwd, GPU-timeline time (CUDA events, else wall)
        into t_gpu, host wait beyond GPU time into t_sync. H2D/D2H stay 0: BCPolicy.logits fuses the copies with
        the forward, so they sit inside t_gpu."""
        if not PROF:
            return logits_fn(buf)
        if _use_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            start.record()
            out = logits_fn(buf)
            end.record()
            end.synchronize()
            wall = time.perf_counter() - t
            gpu = start.elapsed_time(end) / 1000.0
            prof_fwd["t_fwd"] += wall
            prof_fwd["t_gpu"] += gpu
            prof_fwd["t_sync"] += max(0.0, wall - gpu)
            return out
        t = time.perf_counter()
        out = logits_fn(buf)
        wall = time.perf_counter() - t
        prof_fwd["t_fwd"] += wall
        prof_fwd["t_gpu"] += wall
        return out

    route: dict[int, tuple[int, int] | None] = {}  # id(obs) -> None (rollout) | (game idx, s_seat)
    r_n: dict[int, int] = {}  # game idx -> R outer decisions (0 in a finished game = routing broke)
    routed = args.opp != "self"  # R leaves the base policy: every obs is tagged, R's go to R's policy
    hold: set[str] = set()
    deck_dir = ROOT / "decks" / "pool"  # --deck-set newcards switches it
    if args.deck_set != "sample" and (args.opp == "ygoagent" or args.deal_mode == "train"):
        ap.error("--deck-set is for eval deals without --opp ygoagent")
    if args.deal_mode == "train" or routed:
        hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    opp = None
    if args.opp == "ygoagent":
        from decklight.eval.ygoagent import YgoAgentPolicy

        opp = YgoAgentPolicy(seed=3)  # as in playtest's --ygoagent match
        gfiles = sorted(args.gauntlet_decks.glob("*.ydk"))
        if len(gfiles) < 2:
            fail(f"only {len(gfiles)} gauntlet decks in {args.gauntlet_decks}")
        names = [p.stem for p in gfiles]
        decks = [load_ydk(f) for f in gfiles]
        if args.deal_mode == "train":
            # User, 2026-10-08: gauntlet decks stay eval-only for OUR side ("Our side on pool decks"): S pilots a
            # pool train deck, only ygo-agent plays its gauntlet lists (the 744-deal gate has S pilot those).
            if args.games <= 0:
                fail("--deal-mode train needs --games N")
            pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk") if p.stem not in hold)
            g_main = {tuple(sorted(d.main)) for d in decks}
            s_names = [n for n in pool if tuple(sorted(load_ydk(ROOT / "decks/pool" / f"{n}.ydk").main)) not in g_main]
            if len(s_names) < 2:
                fail(f"only {len(s_names)} pool train decks for S")
            n_s = len(s_names)
            names = s_names + names
            decks = [load_ydk(ROOT / "decks/pool" / f"{n}.ydk") for n in s_names] + decks
            rng = random.Random(args.seed_base)
            sched = [(rng.randrange(n_s), n_s + rng.randrange(len(gfiles)), args.seed_base + i, i % 2, i)
                     for i in range(args.games)]
        elif args.pairs:
            # "target on our side in both seats" (scripts/deck_card.py): --pairs "T:O1,T:O2,..."
            # names S's deck first; every pair plays --seeds seeds in both seats, S always on di.
            spec = Path(args.pairs[1:]).read_text() if args.pairs.startswith("@") else args.pairs  # @file: long lists
            want = [tuple(x.strip().split(":")) for x in spec.replace("\n", ",").split(",") if x.strip()]
            sched = pairs_schedule(names, want, args.seeds, args.deal_seed0, bool(args.seed_per_pair))
        else:
            sched = ga_schedule(names, args.ga_opps, args.ga_seeds, args.deal_seed0)
    elif args.deal_mode == "train":
        if args.games <= 0:
            fail("--deal-mode train needs --games N")
        pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
        names = [d for d in pool if d not in hold]
        if len(names) < 2:
            fail(f"only {len(names)} train decks after exclusions")
        decks = None
        sched = train_schedule(names, args.games, args.seed_base, hold)
    else:
        if args.pairs:
            spec = Path(args.pairs[1:]).read_text() if args.pairs.startswith("@") else args.pairs  # @file: long lists
            want = [tuple(x.strip().split(":")) for x in spec.replace("\n", ",").split(",") if x.strip()]
            names = sorted({d for pr in want for d in pr})
            if args.deck_set == "newcards":  # --pairs from decks/newcards instead of the pool
                deck_dir = ROOT / "decks" / "newcards"

            def _pair_path(name: str) -> Path:
                if args.deck_dir is not None and (args.deck_dir / f"{name}.ydk").is_file():
                    return args.deck_dir / f"{name}.ydk"
                if (deck_dir / f"{name}.ydk").is_file():
                    return deck_dir / f"{name}.ydk"
                pool = ROOT / "decks" / "pool" / f"{name}.ydk"
                if pool.is_file():
                    return pool
                return deck_dir / f"{name}.ydk"

            missing = [d for d in names if not _pair_path(d).is_file()]
            if missing or any(len(pr) != 2 for pr in want):
                fail(f"--pairs: bad pairs or decks not in {deck_dir.name}: {missing or want}")
        elif args.deck_set == "sample":
            names = sample_decks(args.decks, args.seed)
        elif args.deck_set == "holdout":
            hset = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
            names = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk") if p.stem in hset)
        else:
            deck_dir = ROOT / "decks" / "newcards"
            names = sorted(p.stem for p in deck_dir.glob("*.ydk"))
        if len(names) < 2:
            fail(f"only {len(names)} decks in deck set {args.deck_set}")
        decks = None
        pair_names = [(a, b) for a in range(len(names)) for b in range(len(names)) if a != b]
        if args.pairs:
            pair_names = [(names.index(a), names.index(b)) for a, b in want]
        sched = []
        for k, (di, dj) in enumerate(pair_names):
            # --seed-per-pair: pair k gets its own seed range, so a deck meets a new opening hand in every game (one
            # shared range gives a deck the same hand against every opponent; NOTEBOOK 2026-10-19)
            s0 = args.deal_seed0 + (k * args.seeds if args.seed_per_pair else 0)
            for s in range(s0, s0 + args.seeds):
                for s_seat in (0, 1):
                    sched.append((di, dj, s, s_seat, len(sched)))
    if args.both_search and (args.opp != "self" or args.raw):
        ap.error("--both-search needs --opp self and search (not --raw 1)")
    if args.opp == "ckpt":
        if args.opp_ckpt is None:
            ap.error("--opp ckpt needs --opp-ckpt PATH")
        opp = BCPolicy(args.opp_ckpt, args.emb, threads=1, device=args.device, batch=args.batch)

        def opp_logits(obs_list: list[dict]) -> np.ndarray:
            out = [_timed_logits(opp.logits, _timed_batch(obs_list[i : i + args.batch]))
                   for i in range(0, len(obs_list), args.batch)]
            lg = np.concatenate(out)
            if bool(np.isnan(lg).any()):
                raise RuntimeError("ALERT search_match: NaN logits from the --opp-ckpt forward")
            return lg

    elif args.opp == "greedy":
        from decklight.eval.policies import GreedyPolicy

        opp = GreedyPolicy(seed=3)
    if args.s_seat_only in (0, 1):  # idx stays the full schedule's, so games pair by (decks, seed) with full arms
        sched = [e for e in sched if e[3] == args.s_seat_only]
        if not sched:
            fail(f"--s-seat-only {args.s_seat_only}: empty schedule")
    if args.games_limit:
        sched = sched[: args.games_limit]
    by_idx = {e[4]: e for e in sched}  # idx is the full schedule's (kept by --s-seat-only), not a position in sched
    claim = bool(args.save_records is not None if args.claim is None else args.claim)
    if LAT:
        args.width = 1  # latency mode: one game at a time, so t_wall is the true decision latency
        claim = False
    if claim and args.out is None:
        ap.error("--claim 1 needs --out (claims live in <out>/claims)")
    if claim:
        # dynamic queue (teacher-r1 2026-10-08: static shards left 30 of 32 cores idle for hours behind the procs
        # dealt the slowest deck pairings): every proc walks the whole schedule from its own offset and plays a game
        # only if it creates <out>/claims/<idx> first (O_EXCL is atomic); few games in flight so claims stay late.
        # A proc that dies loses its in-flight games (never replayed): fine for teacher data, not for eval cards.
        args.width = min(args.width, args.claim_width)
        claim_dir = args.out / "claims"
        claim_dir.mkdir(parents=True, exist_ok=True)
        off = (args.proc * len(sched)) // max(args.nproc, 1)
        mine = sched[off:] + sched[:off]
    else:
        mine = sched[args.proc :: args.nproc]

    def claimed(idx: int) -> bool:
        if not claim:
            return True
        try:
            os.close(os.open(claim_dir / str(idx), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
            return True
        except FileExistsError:
            return False
    print(f"opp {args.opp}, decks {names}: {len(sched)} games, proc {args.proc}/{args.nproc} runs {len(mine)}",
          flush=True)

    def forward(obs_list: list[dict]) -> np.ndarray:
        if PROF:
            prof_fwd["calls"] += 1
            prof_fwd["hist"][str(len(obs_list))] = prof_fwd["hist"].get(str(len(obs_list)), 0) + 1
        if not routed:
            return base_logits(obs_list)
        # R's outer decisions go to R's policy; everything else (S outer + all rollouts,
        # which model the opponent with the base policy) goes to the base forward.
        base_at, base_obs, r_at, r_obs, r_keys, r_me = [], [], [], [], [], []
        for i, o in enumerate(obs_list):
            tag = route.pop(id(o), ...)
            if tag is ...:
                fail(f"untagged obs in the batch (opp={args.opp}): every request passes through tag_game")
            if tag is None:
                base_at.append(i)
                base_obs.append(o)
            else:
                idx, s_seat = tag
                if seat_of(o) == s_seat:
                    base_at.append(i)
                    base_obs.append(o)
                else:
                    r_at.append(i)
                    r_obs.append(o)
                    r_keys.append(idx)
                    r_n[idx] = r_n.get(idx, 0) + 1
                    r_me.append(seat_of(o))
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
        if r_obs:
            if args.opp == "ygoagent":
                buf = _timed_batch(r_obs)
                buf["player"] = np.array(r_me, np.int64)  # CppEnv obs carry no seat; global[4] does
                t_ya = time.perf_counter() if PROF else 0.0
                acts = opp(buf, np.arange(len(r_obs)), keys=np.array(r_keys, np.int64))
                if PROF:
                    prof_fwd["t_ygo"] += time.perf_counter() - t_ya
                for i, a in zip(r_at, acts, strict=True):  # one-hot argmax: the inner argmax plays R's pick
                    lg[i, int(a)] = 1.0
            elif args.opp == "greedy":
                buf = _timed_batch(r_obs)
                buf["player"] = np.array(r_me, np.int64)
                t_ya = time.perf_counter() if PROF else 0.0
                acts = opp(buf, np.arange(len(r_obs)))
                if PROF:
                    prof_fwd["t_ygo"] += time.perf_counter() - t_ya
                for i, a in zip(r_at, acts, strict=True):  # one-hot: the inner argmax plays R's pick
                    lg[i, int(a)] = 1.0
            else:  # ckpt: R plays another BC checkpoint raw (argmax through the shared guard discipline)
                b = opp_logits(r_obs)
                if b.shape[1] != a_width:
                    fail(f"opp policy width {b.shape[1]} != action options {a_width}")
                for k, i in enumerate(r_at):
                    lg[i] = b[k]
        return lg

    cfgs: dict[int, DuelConfig] = {}  # game idx -> DuelConfig (teacher records + replay check)
    inflight_choices: dict[int, list] = {}  # --prof 1: idx -> live res.choices (same list; popped when done)
    inflight_meta: dict[int, dict] = {}  # idx -> {decks, seed, s_seat} for the SIGTERM dump

    ho_rule = (HonestRule(z=args.ho_z, min_gap=args.ho_min_gap, s_max=args.ho_s_max, step=args.ho_step)
               if args.ho else None)
    if ho_rule is not None and ho_rule.s_max < args.samples:
        fail(f"--ho-s-max {ho_rule.s_max} < --samples {args.samples}")
    if args.replay_threads < 1:
        fail(f"--replay-threads must be >= 1, got {args.replay_threads}")
    # Speed plan E6: one thread pool per process for world checks + branch replays (bit-identical
    # to replay_threads=1, which keeps today's sequential path with pool None). Named replay_pool:
    # `pool` is already the train deck-name list in the ygoagent branch below.
    replay_pool = None if args.replay_threads == 1 else ThreadPoolExecutor(max_workers=args.replay_threads)
    def jobs():
        for di, dj, s, s_seat, idx in mine:
            if not claimed(idx):
                continue
            if args.opp == "ygoagent":
                seat0, seat1 = (di, dj) if s_seat == 0 else (dj, di)
                cfg = DuelConfig(decks[seat0], decks[seat1], seed=s)
            else:
                if args.deal_mode == "train" and (names[di] in hold or names[dj] in hold):
                    fail(f"train game drew a holdout deck: {names[di]} vs {names[dj]}")
                # S (deck di) sits in seat s_seat, as in the ygoagent branch and the record's "decks" (seat order).
                # Before 2026-10-27 di always sat in seat 0, so with s_seat 1 S piloted dj (NOTEBOOK 2026-10-27 X2).
                seat0, seat1 = (di, dj) if s_seat == 0 else (dj, di)
                path = _pair_path if args.pairs else (lambda n: deck_dir / f"{n}.ydk")
                cfg = DuelConfig(load_ydk(path(names[seat0])), load_ydk(path(names[seat1])), seed=s)
            cfgs[idx] = cfg
            if PROF:
                inflight_meta[idx] = {"decks": [names[di], names[dj]], "seed": s, "s_seat": s_seat}
            if not routed:
                ws = WindowSearch(
                    engine, never, k=args.k, horizon=args.horizon, samples=args.samples,
                    rollout_temp=args.rollout_temp, batch=args.batch, seed=args.seed, value=value,
                    prof=PROF, pool=replay_pool,
                )  # fmt: skip
                gen = search_game_gen(
                    engine, cfg, ws, s_seat, argmax_select, game_seed=idx,
                    own_turn1=bool(args.own_turn1), conf_skip=args.conf_skip, own_frac=args.own_frac,
                    honest=bool(args.honest), det_tries=args.det_tries, det_fallback=args.det_fallback,
                    margin=args.margin, max_search_turn=args.max_search_turn, stop_turn=args.stop_turn,
                    max_searches=args.max_searches, ho=ho_rule, both_search=bool(args.both_search),
                    prof=PROF, latency=LAT, inflight=inflight_choices if PROF else None,
                )  # fmt: skip
            else:
                ws = RouteWindowSearch(
                    engine, never, k=args.k, horizon=args.horizon, samples=args.samples,
                    rollout_temp=args.rollout_temp, batch=args.batch, seed=args.seed, value=value,
                    route=route, prof=PROF, pool=replay_pool,
                )  # fmt: skip
                # Card-faithful loop guard (NOTEBOOK 2026-10-09 08:40): each policy guards its own decisions; ygo-agent
                # and greedy have no guard. `--guard-legacy 1` restores the shared guard of E2 (forbade ygo-agent's
                # own pick at repeated chain windows: 596/744 games differed from the card).
                legacy = bool(args.guard_legacy)
                gen = tag_game(idx, s_seat, search_game_gen(
                    engine, cfg, ws, s_seat, argmax_select, game_seed=idx,
                    own_turn1=bool(args.own_turn1), conf_skip=args.conf_skip, own_frac=args.own_frac,
                    honest=bool(args.honest), det_tries=args.det_tries, det_fallback=args.det_fallback,
                    guard_mode="shared" if legacy else "per_seat",
                    unguarded_seat=None if legacy or args.opp == "ckpt" else 1 - s_seat, margin=args.margin,
                    max_search_turn=args.max_search_turn, stop_turn=args.stop_turn, max_searches=args.max_searches,
                    ho=ho_rule, prof=PROF, latency=LAT, inflight=inflight_choices if PROF else None,
                ), route)  # fmt: skip
            yield idx, gen

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_records is not None:
        args.save_records.mkdir(parents=True, exist_ok=True)
    sel_key = f"value@h{args.horizon}" if args.value else f"board@h{args.horizon}"
    path = args.out / f"games-{args.proc}.jsonl"

    def write_teacher_record(idx: int, s_seat: int, seat_decks: list[str], res) -> int:
        """Append one gen_window-format teacher record for S's deck; replay-check it first.
        Returns the label count (non-skip labels == searched S decisions)."""
        from decklight.demos.dataset import record_config
        from decklight.engine.cpp_env import CppEnv

        n = len(res.choices)
        if len(res.actors) != n or len(res.decision_turns) != n:
            fail(f"game {idx}: choices/actors/turns length mismatch")
        targets: list[int | None] = [None] * n
        labels = ["skip"] * n
        slog = []
        for e in res.log:
            k = int(e["k"])
            cands = [int(x) for x in e.get("cands", [])]
            scores = [float(x) for x in e.get("scores", [])]
            if not cands or len(cands) != len(scores):
                fail(f"game {idx}: searched decision {k} without cands/scores")
            targets[k] = int(e["chosen"])
            labels[k] = "window" if e.get("kind") == "window" else "own"
            sall = e.get("scores_all") or {sel_key: scores}
            slog.append({"k": k, "kind": e.get("kind"), "cands": cands, "scores": scores,
                         "scores_all": {kk: [float(x) for x in col] for kk, col in sall.items()},
                         "chosen": int(e["chosen"]), "played": int(e["played"]), "samples": e.get("samples", [])})
        deck_s = seat_decks[s_seat]
        trec = {
            "decks": list(seat_decks), "config": config_dict(cfgs[idx]), "auto_single": True,
            "choices": [int(c) for c in res.choices], "actors": [int(a) for a in res.actors],
            "turns": [int(t) for t in res.decision_turns], "winner": res.winner,
            "truncated": bool(res.truncated),
            "game_id": f"st-{idx}", "line_id": f"{deck_s}/st-{idx}",
            "targets": targets, "labels": labels, "notes": [], "search": slog,
            "source": f"search-teacher k={args.k} samples={args.samples} horizon={args.horizon} "
                      f"select={sel_key} conf_skip={args.conf_skip} own_frac={args.own_frac} "
                      f"value={args.value} honest={int(args.honest)} "
                      f"det_fallback={args.det_fallback} opp={args.opp} deal={args.deal_mode}",
        }
        env = CppEnv(engine, record_config(trec), True)  # the record must replay from its own config
        obs = env.reset()
        for c in trec["choices"]:
            if obs is None:
                fail(f"game {idx}: teacher record replay ended early")
            obs = env.step(int(c))
        if (obs is None) == bool(res.truncated):
            fail(f"game {idx}: teacher record replay diverges (truncated={res.truncated})")
        if not res.truncated and int(env.winner) != res.winner:
            fail(f"game {idx}: teacher record replay diverges (winner {env.winner} != {res.winner})")
        line = (json.dumps(trec) + "\n").encode()  # one write() under O_APPEND: procs share deck files
        fd = os.open(args.save_records / f"{deck_s}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        return sum(1 for lb in labels if lb != "skip")

    got, tot_s, tot_labels = 0, 0, 0

    def _thread_cpu() -> dict:
        """Per-thread CPU seconds (utime+stime) from /proc/self/task/*/stat."""
        out: dict[str, float] = {}
        try:
            hz = os.sysconf("SC_CLK_TCK")
            task = Path("/proc/self/task")
            tids = [d.name for d in task.iterdir()]
        except OSError:
            return out
        for tid in tids:
            try:
                st = (task / tid / "stat").read_text()
            except OSError:
                continue
            tail = st.rsplit(")", 1)[-1].split()  # after "pid (comm)": field 3 (state) is index 0
            if len(tail) > 13:
                try:
                    out[tid] = round((int(tail[11]) + int(tail[12])) / hz, 2)  # utime (14) + stime (15)
                except ValueError:
                    continue
        return out

    def write_prof() -> None:
        """Write prof-<proc>.json: forward counters plus process/thread/GPU counters."""
        if not PROF:
            return
        import threading

        rec = dict(prof_fwd)
        rec["hist"] = {str(k): v for k, v in prof_fwd["hist"].items()}
        rec["proc"] = args.proc
        rec["process_time"] = round(time.process_time(), 3)
        rec["wall"] = round(time.time() - t0, 3)
        rec["threads"] = threading.active_count()
        try:
            import torch as _t

            rec["torch_threads"] = _t.get_num_threads()
            rec["cuda_reserved"] = _t.cuda.memory_reserved() if _t.cuda.is_available() else None
        except Exception:  # noqa: BLE001 -- profiling must never fail the match
            rec["torch_threads"] = None
            rec["cuda_reserved"] = None
        rec["thread_cpu"] = _thread_cpu()
        rec["games"] = got
        rec["searched"] = tot_s
        (args.out / f"prof-{args.proc}.json").write_text(json.dumps(rec, indent=1))

    if PROF:
        def _on_term(signum, frame) -> None:
            # End-of-phase SIGTERM (M0 throughput arms): dump every in-flight game's choice prefix for the
            # determinism check against cached games, plus the partial forward counters, then die as usual.
            try:
                with (args.out / f"inflight-{args.proc}.jsonl").open("w") as f:
                    for idx, ch in inflight_choices.items():
                        m = inflight_meta.get(idx, {})
                        f.write(json.dumps({"decks": m.get("decks"), "seed": m.get("seed"),
                                            "s_seat": m.get("s_seat"),
                                            "choices": [int(c) for c in ch]}) + "\n")
                try:
                    write_prof()
                except Exception:  # noqa: BLE001, S110 -- the inflight dump matters more
                    pass
            finally:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _on_term)

    try:
        with path.open("w") as f:
            for idx, res in drive_many(jobs(), forward, args.width):
                di, dj, s, s_seat, _ = by_idx[idx]
                inflight_choices.pop(idx, None)  # finished: no longer in flight
                if res.winner is None and not res.truncated:
                    fail(f"game {idx} ended with neither winner nor truncation")
                if args.opp == "ygoagent":
                    opp.reset_keys([idx])  # drop ygo-agent's LSTM state for the finished game
                rec = {
                    "decks": [names[di], names[dj]] if args.opp != "ygoagent" else
                             [names[di if s_seat == 0 else dj], names[dj if s_seat == 0 else di]],
                    "seed": s, "s_seat": s_seat, "game_idx": idx,
                    "winner": res.winner, "truncated": res.truncated, "turns": res.turns,
                    "choices": res.choices, "actors": res.actors,
                    "raw": bool(args.raw), "n_searched": res.n_searched, "n_override": res.n_override,
                    "search_s": round(res.search_s, 3),
                    "n_branches": res.n_branches, "n_capped": res.n_capped, "search_cut": res.search_cut,
                    "det_proposals": res.det_proposals, "det_accepted": res.det_accepted,
                    "det_fallbacks": res.det_fallbacks, "det_skipped": res.det_skipped,
                    "det_fallback": args.det_fallback, "det_s": round(res.det_s, 3),
                    "replay_threads": args.replay_threads,
                    # latency mode: per-move and per-own-turn walls (scripts/audit/latency_report.py)
                    **({"lat_raw": res.lat_raw, "lat_search": res.lat_search, "lat_turn": res.lat_turn}
                       if LAT else {}),
                    # per searched decision: candidates, per-sample rollout scores, pick (offline budget curve and
                    # noise SE; NOTEBOOK 2026-10-09 08:40)
                    "search": [{"k": int(e["k"]), "seat": int(e.get("seat", s_seat)), "kind": e.get("kind"),
                                "turn": int(e.get("turn", 0)),
                                "cands": [int(x) for x in e["cands"]], "samples": e.get("samples", []),
                                "chosen": int(e["chosen"]), "played": int(e["played"]),
                                **({"scores": [float(x) for x in e["scores"]]} if "scores" in e else {}),
                                **({"scores_all": {kk: [float(x) for x in col]
                                                   for kk, col in e["scores_all"].items()}}
                                   if "scores_all" in e else {}),
                                **({"prof": e["prof"]} if "prof" in e else {}),
                                **({"ho": e["ho"]} if "ho" in e else {})} for e in res.log],
                }
                if args.opp == "ygoagent":  # paired-gate fields (playtest per_game format + search stats)
                    w = res.winner
                    rec.update(deck_s=names[di], deck_r=names[dj], s_first=(s_seat == 0), ya_decisions=r_n.pop(idx, 0),
                               win=1.0 if w == s_seat else (0.0 if w in (0, 1) else 0.5),
                               cut=res.truncated)
                elif routed:
                    rec.update(opp_decisions=r_n.pop(idx, 0))
                f.write(json.dumps(rec) + "\n")
                got += 1
                tot_s += res.n_searched
                if args.save_records is not None:
                    tot_labels += write_teacher_record(idx, s_seat, rec["decks"], res)
                if got % 10 == 0:
                    print(f"{got}/{len(mine) if not claim else '?'} games, searched {tot_s}, {time.time() - t0:.0f} s", flush=True)
    except RuntimeError as e:
        if str(e).startswith("ALERT search_match:"):
            print(str(e), flush=True)
            raise SystemExit(1) from None
        raise
    write_prof()  # --prof 1: per-proc forward counters next to the games file (also written on SIGTERM)
    if replay_pool is not None:
        replay_pool.shutdown()  # E6 pool: all games are written by now
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
    if args.save_records is not None:
        # claim mode: a proc may legitimately get 0 games when the others claimed them all
        all_taken = claim and sum(1 for _ in claim_dir.iterdir()) >= len(sched)
        if got == 0 and not all_taken:
            fail(f"0 records written to {args.save_records}")
        if tot_labels == 0 and not args.raw and got > 0:
            fail(f"0 search labels written to {args.save_records}")
        print(f"proc {args.proc}: {tot_labels} search labels -> {args.save_records}", flush=True)
    if share > 0.2:
        fail(f"capped-branch share {share:.3f} > 20% ({capped}/{branches})")


if __name__ == "__main__":
    main()
