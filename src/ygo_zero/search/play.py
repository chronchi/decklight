"""Live game: base policy + WindowSearch (S) vs the same policy raw (R).

`search_game_gen` plays ONE full game through the `drive_many` protocol (see
search/turn1.py): it yields `[obs, ...]` lists and receives the batched policy
logits back, so many games share one forward. Both seats' non-searched moves use
the report card's model selection (scripts/playtest.py builds its BCPolicy with
the default temperature 0.0, i.e. argmax) behind a LoopGuard, exactly the way
BCPolicy.__call__ applies it. Only S's decisions are ever searched:

- (a) a response window (`is_window`: a chain window on the opponent's turn
  where S can ACTIVATE), or
- (b) S's own turn (`global[5] == 1`) with >= 2 legal options from turn 2 on
  (turn >= 1 with `own_turn1`), sampled with probability `own_frac`;

and in both cases only when the policy is not confident: the move is played
raw when `softmax(z).max() >= conf_skip`.

Searching is `WindowSearch._candidates` plus `rollouts_gen` on the branches
`(choices_so_far + [cand], j, cand, sample)`, playing the best candidate by the
search's selected score with the same tie rule as `game_gen`
(`max(..., key=(score, -j))`: ties keep the policy's order, so firing must earn
its card). The policy's own pick is the guard-masked argmax; an override is
counted when the search plays something else. Every searched decision is also
appended to `GameResult.log` (candidates + selection scores + kind, for teacher
records); `decision_turns` records every decision's turn.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ygo_zero.engine.cpp_env import CppEnv, FlightRecorder
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.eval.loopguard import LoopGuard
from ygo_zero.search.determinize import Determinizer, SeenTracker, key_rank_of_choice
from ygo_zero.search.honest import HonestRule, decide, decide_pair, semantic_key
from ygo_zero.search.turn1 import _masked
from ygo_zero.search.window import WindowSearch, is_window

# Takes the loop-guard-masked logits of one decision, returns the option index.
SelectFn = Callable[[np.ndarray], int]


def argmax_select(z: np.ndarray) -> int:
    """The report card's model selection: BCPolicy at temperature 0.0 plays argmax."""
    return int(np.argmax(z))


def softmax_conf(z: np.ndarray) -> float:
    """The policy's confidence: the max softmax probability over the legal options."""
    finite = np.isfinite(z)
    if int(finite.sum()) <= 1:
        return 1.0
    v = np.asarray(z[finite], dtype=np.float64)
    v -= v.max()
    e = np.exp(v)
    return float((e / e.sum()).max())


def _accum_win_prof(wprof: dict, search: WindowSearch) -> None:
    """Fold one rollouts_gen call's `search.last_prof` into the per-decision wprof accumulator."""
    lp = getattr(search, "last_prof", {})
    for kk in ("t_setup", "t_env", "t_py", "t_wait"):
        wprof[kk] += float(lp.get(kk, 0.0))
    for kk in ("rounds", "obs_fwd"):
        wprof[kk] += int(lp.get(kk, 0))
    wprof["rollout_len"].extend(lp.get("rollout_len", []))
    wprof["t_value"] += float(lp.get("t_value", 0.0))


@dataclass
class GameResult:
    winner: int | None  # 0/1 seat, 2 = draw; None when truncated
    truncated: bool  # still running after max_decisions
    turns: int  # turn number of the last decision
    choices: list[int] = field(default_factory=list)
    actors: list[int] = field(default_factory=list)
    n_searched: int = 0  # S decisions where the search picked the move
    n_override: int = 0  # searched decisions where it left the policy's own pick
    search_s: float = 0.0  # wall seconds spent inside search (incl. rollout forwards)
    n_branches: int = 0  # rollout branches launched over the game
    n_capped: int = 0  # of those that hit max_rollout (scored as losses)
    searched_idx: list[int] = field(default_factory=list)  # decision indices searched
    det_proposals: int = 0  # honest worlds proposed (rejection-sampling tries)
    det_accepted: int = 0  # ... accepted (root obs identical to the truth)
    det_fallbacks: int = 0  # ... rejected after `tries` (fell back to the true config)
    det_skipped: int = 0  # ... skipped searches (det_fallback="skip", no world accepted)
    det_s: float = 0.0  # wall seconds spent drawing honest worlds
    choice_keys: list = field(default_factory=list)  # (key, rank) of every choice (semantic replay)
    log: list[dict] = field(default_factory=list)  # one entry per searched S decision (teacher records)
    decision_turns: list[int] = field(default_factory=list)  # turn number of every decision
    search_cut: bool = False  # max_searches reached: the rest of the game was played raw
    lat_raw: list[float] = field(default_factory=list)  # latency mode: wall s per raw move
    lat_search: list[float] = field(default_factory=list)  # latency mode: wall s per searched move
    lat_turn: list[float] = field(default_factory=list)  # latency mode: wall s per own turn


def search_game_gen(
    engine: Engine,
    config: DuelConfig,
    search: WindowSearch,
    s_seat: int,
    select_fn: SelectFn,
    game_seed: int,
    own_turn1: bool = False,
    conf_skip: float = 0.9,
    own_frac: float = 1.0,
    max_decisions: int = 3000,
    auto_single: bool = True,
    honest: bool = False,
    det_tries: int = 8,
    det_fallback: str = "true",
    prefix: Sequence[int] = (),
    guard_mode: str = "shared",
    unguarded_seat: int | None = None,
    margin: float = 0.0,
    max_search_turn: int = 0,
    stop_turn: int = 0,
    max_searches: int = 0,
    ho: HonestRule | None = None,
    both_search: bool = False,
    prof: bool = False,
    latency: bool = False,
    inflight: dict | None = None,
):
    """Play one live game; S (absolute seat `s_seat`) searches, R plays raw. Returns a GameResult.

    With `honest`, S's searched decisions roll out from determinized worlds (everything S has
    not seen re-sampled consistently with S's observation) instead of the true hidden state.
    With `det_fallback="true"` (default) a world that can't be sampled after `det_tries`
    proposals falls back to the true config; with `"skip"` only accepted worlds are used, and a
    decision with no accepted world is not searched (the policy's move is played, `det_skipped`).

    With `prefix`, those option indices are replayed verbatim before the first live
    decision (e.g. a DAGS prefix positioning a mid-game start); they are recorded in
    `choices`/`actors`/`choice_keys` exactly like played decisions, and the loop guard
    starts fresh at the first live decision. Empty (default) plays from the deal.

    Loop guard: `guard_mode="shared"` (default, the historical behaviour) runs both seats through one guard;
    `"per_seat"` gives each seat its own, as the report card does when two different policy objects play (each
    BCPolicy guards only its own decisions). `unguarded_seat` plays that seat's pick as given: ygo-agent and greedy
    have no loop guard in the card. Without this, the shared guard forbade ygo-agent's own pick (an option already
    taken in a same-key state this turn) and its one-hot logits fell back to option 0: all 596 E2-vs-card divergences
    on the 744 deals (NOTEBOOK 2026-10-09 08:40).

    `margin` > 0: a searched decision plays the search's best only if its selection score beats the policy's own
    move's by >= margin (when the policy's move is a candidate); the log keeps both ("best", "chosen").

    Label generation (ExIt pilot): `max_search_turn` > 0 searches only decisions on turns <= it; `stop_turn` > 0 ends
    the game at the first decision past that turn (GameResult.truncated, winner None).

    `max_searches` > 0: after that many search attempts (searched + skipped) the game is played raw
    (`search_cut`). Honest worlds replay the whole prefix up to tries x samples times per search, so a long looping game
    gets slower with every search; one such game held 8 games in flight for 45+ min (NOTEBOOK 2026-10-11 00:10).

    `both_search`: both seats search (product vs product showcase games); each seat determinizes from its own
    SeenTracker. Log entries carry the searching "seat".

    `ho` (honest override, search/honest.py): candidates are merged by meaning, and the search leaves the policy's
    pick only when the best challenger's paired gap is significant; undecided gaps get `ho.step` more samples for the
    two candidates, up to `ho.s_max`. `margin` is ignored with `ho`. The log entry gets an "ho" dict.

    `prof`: per searched decision the log entry gains a "prof" dict of perf_counter timings (determinizer build and
    sampling, the WindowSearch phase breakdown, wall); play is bit-identical with it on or off. It also enables the
    search's own window timings. `latency`: every decision's wall seconds go to `lat_raw` (raw moves) or
    `lat_search` (searched moves), and each own turn's first-to-last S-decision span goes to `lat_turn`.
    `inflight`: when given, `inflight[game_seed]` mirrors `res.choices` live (same list), so a SIGTERM handler can
    dump the choice prefix of games still in flight; the driver pops finished games.
    """
    if guard_mode not in ("shared", "per_seat"):
        raise ValueError(f"guard_mode must be 'shared' or 'per_seat', got {guard_mode!r}")
    if det_fallback not in ("true", "skip"):
        raise ValueError(f"det_fallback must be 'true' or 'skip', got {det_fallback!r}")
    rng = random.Random(search.seed * 1_000_003 + game_seed)
    env = CppEnv(engine, config, auto_single)
    flight = FlightRecorder(config)
    obs = env.reset()
    searchers = (0, 1) if both_search else (s_seat,)
    trackers = {s: SeenTracker(s) for s in searchers} if honest else {}
    tracker = trackers.get(s_seat)  # s_seat's tracker (choice keys and the prefix loop below)
    for t in trackers.values():
        t.update(env)
    guard = LoopGuard()  # slot 0 (shared) or slot = seat (per_seat); see the docstring
    res = GameResult(winner=None, truncated=False, turns=1)
    if inflight is not None:
        inflight[game_seed] = res.choices
    if prof:
        search.prof = True  # window phase timings feed the per-decision prof dict
    lat_open_turn: int | None = None  # latency mode: turn whose own-turn span is open
    lat_open_start = 0.0
    lat_last_end = 0.0
    capped0 = search.capped
    for c in prefix:
        if obs is None:
            raise ValueError("search_game_gen: game ended inside the prefix")
        n_legal = int(obs["action_mask"].sum())
        if not 0 <= int(c) < n_legal:
            raise ValueError(f"search_game_gen: prefix choice {c} illegal with {n_legal} options")
        me = int(env.player)
        res.choices.append(int(c))
        res.actors.append(me)
        res.decision_turns.append(int(obs["global"][2]))
        if tracker is not None:
            res.choice_keys.append(key_rank_of_choice(obs, int(c)))
        obs = env.step(int(c))
        for t in trackers.values():
            t.update(env)
    while obs is not None and len(res.choices) < max_decisions:
        me = int(env.player)
        turn = int(obs["global"][2])
        if stop_turn and turn > stop_turn:
            break
        res.turns = turn
        t_dec = time.perf_counter() if latency else 0.0
        n_log0 = len(res.log)
        is_own = latency and me in searchers and int(obs["global"][5]) == 1
        n_legal = int(obs["action_mask"].sum())
        z = _masked((yield [obs])[0], obs)
        if bool(np.isnan(np.asarray(z, dtype=np.float64)).any()):
            raise RuntimeError(
                f"ALERT search_match: NaN logits at decision {len(res.choices)} (game_seed={game_seed})"
            )
        gslot = me if guard_mode == "per_seat" else 0
        guarded = me != unguarded_seat
        z_g, key = guard.mask(gslot, obs, z) if guarded else (z, None)
        a_pol = select_fn(z_g)
        a = a_pol
        if me in searchers:
            window = is_window(obs, me)
            own = (
                int(obs["global"][5]) == 1
                and n_legal >= 2
                and turn >= (1 if own_turn1 else 2)
                and rng.random() < own_frac
            )
            if max_searches and res.n_searched + res.det_skipped >= max_searches:
                res.search_cut = True
            elif (window or own) and softmax_conf(z_g) < conf_skip and not (max_search_turn and turn > max_search_turn):
                cands = search._candidates(z_g, obs, window, merge=ho is not None and ho.merge)
                if len(cands) >= 2:
                    t0 = time.perf_counter()
                    k = len(res.choices)
                    branches = [
                        (res.choices + [x], j, x, s)
                        for j, x in enumerate(cands)
                        for s in range(search.samples)
                    ]
                    worlds = None
                    det = None
                    skip_search = False
                    t_det_init = 0.0
                    t_det = 0.0
                    if trackers:
                        if prof:
                            t_di = time.perf_counter()
                        det = Determinizer(
                            engine, config, auto_single, me, res.choices, trackers[me].seen,
                            obs, random.Random(f"{search.seed}:{game_seed}:{k}:det"),
                            prefix_keys=res.choice_keys, tries=det_tries,
                        )
                        if prof:
                            t_det_init = time.perf_counter() - t_di
                            t_ds = time.perf_counter()
                        worlds = []
                        for _ in range(search.samples):
                            got = det.sample()
                            if got is None:
                                if det_fallback == "true":
                                    worlds.append((config, list(res.choices)))
                            else:
                                worlds.append(got)
                        if prof:
                            t_det += time.perf_counter() - t_ds
                        res.det_proposals += det.proposals
                        res.det_accepted += det.accepted
                        res.det_fallbacks += det.fallbacks
                        res.det_s += det.seconds
                        if det_fallback == "skip" and not worlds:
                            res.det_skipped += 1
                            skip_search = True
                    if not skip_search:
                        wprof: dict = {"t_setup": 0.0, "t_env": 0.0, "t_py": 0.0, "t_wait": 0.0,
                                       "rounds": 0, "obs_fwd": 0, "rollout_len": [],
                                       "t_value": 0.0}  # fmt: skip
                        scores_all = yield from search.rollouts_gen(
                            config, auto_single, branches, me, turn_start=turn,
                            game_seed=game_seed, decision_k=k, worlds=worlds,
                        )  # fmt: skip
                        if prof:
                            _accum_win_prof(wprof, search)
                        res.search_s += time.perf_counter() - t0
                        if isinstance(scores_all, dict):
                            scores_all = {kk: [float(x) for x in col] for kk, col in scores_all.items()}
                            sel_scores = scores_all[search._select_key()]
                        else:
                            sel_scores = scores_all
                        best = max(range(len(cands)), key=lambda j: (sel_scores[j], -j))
                        a = cands[best]
                        ho_log = None
                        if ho is not None:
                            samples = [list(x) for x in getattr(search, "last_samples", [])]
                            kp = semantic_key(obs, a_pol)
                            pick_j = next((j for j, x in enumerate(cands) if semantic_key(obs, x) == kp), None)
                            if pick_j is None or len(samples) != len(cands):
                                raise RuntimeError(f"ALERT search_match: ho pick not among candidates (game_seed="
                                                   f"{game_seed}, k={k})")
                            t_ho = time.perf_counter()
                            dec = decide(samples, pick_j, ho)
                            rounds = 0
                            while dec.status == "undecided" and dec.n < ho.s_max:
                                n0, extra = dec.n, min(ho.step, ho.s_max - dec.n)
                                pair = (pick_j, dec.challenger_j)
                                if worlds is not None and det is not None:
                                    p0, a0, f0, t0d = det.proposals, det.accepted, det.fallbacks, det.seconds
                                    if prof:
                                        t_ds = time.perf_counter()
                                    for _ in range(extra):
                                        got = det.sample()
                                        if got is not None:
                                            worlds.append(got)
                                        elif det_fallback == "true":
                                            worlds.append((config, list(res.choices)))
                                    if prof:
                                        t_det += time.perf_counter() - t_ds
                                    res.det_proposals += det.proposals - p0
                                    res.det_accepted += det.accepted - a0
                                    res.det_fallbacks += det.fallbacks - f0
                                    res.det_s += det.seconds - t0d
                                xb = [(res.choices + [cands[j]], j, cands[j], s)
                                      for j in pair for s in range(n0, n0 + extra)]
                                yield from search.rollouts_gen(
                                    config, auto_single, xb, me, turn_start=turn,
                                    game_seed=game_seed, decision_k=k, worlds=worlds,
                                )  # fmt: skip
                                if prof:
                                    _accum_win_prof(wprof, search)
                                got_s = getattr(search, "last_samples", [])
                                for j in pair:
                                    samples[j].extend(got_s[j])
                                res.n_branches += len(xb)
                                rounds += 1
                                dec = decide_pair(samples, pick_j, pair[1], ho)
                            a = cands[dec.chosen_j] if dec.chosen_j != pick_j else a_pol
                            ho_log = {"status": dec.status, "gap": round(dec.gap, 4),
                                      "se": round(dec.se, 4) if dec.se != float("inf") else None, "n": dec.n,
                                      "challenger": None if dec.challenger_j is None else int(cands[dec.challenger_j]),
                                      "rounds": rounds}
                            search.last_samples = samples
                            res.search_s += time.perf_counter() - t_ho
                        # margin: keep the policy's own move unless the search's best beats it by >= margin
                        # (NOTEBOOK 2026-10-09: most overrides win by tiny, noisy score gaps)
                        elif (margin > 0 and a != a_pol and a_pol in cands
                                and sel_scores[best] - sel_scores[cands.index(a_pol)] < margin):
                            a = a_pol
                        res.n_searched += 1
                        res.n_branches += len(branches)
                        res.searched_idx.append(k)
                        if prof:
                            wprof = {
                                "t_det_init": t_det_init, "t_det": t_det,
                                "n_prop": det.proposals if det is not None else 0,
                                "n_acc": det.accepted if det is not None else 0,
                                **wprof, "t_wall": time.perf_counter() - t0,
                            }
                        res.log.append({
                            "k": k, "seat": me, "kind": "window" if window else "own", "turn": turn,
                            "cands": [int(x) for x in cands],
                            "scores": [float(x) for x in sel_scores],
                            "scores_all": ({kk: [float(x) for x in col] for kk, col in scores_all.items()}
                                           if isinstance(scores_all, dict)
                                           else {search._select_key(): [float(x) for x in sel_scores]}),
                            "chosen": int(a), "played": int(a_pol), "best": int(cands[best]),
                            "samples": [[round(x, 4) for x in col] for col in getattr(search, "last_samples", [])],
                            **({"ho": ho_log} if ho_log is not None else {}),
                            **({"prof": wprof} if prof else {}),
                        })
                        if a != a_pol:
                            res.n_override += 1
        if guarded:
            guard.note(gslot, key, a)
        if tracker is not None:
            res.choice_keys.append(key_rank_of_choice(obs, a))
        res.choices.append(a)
        res.actors.append(me)
        res.decision_turns.append(turn)
        flight.choice(a)
        obs = env.step(a)
        for t in trackers.values():
            t.update(env)
        if latency:
            w_end = time.perf_counter()
            (res.lat_search if len(res.log) > n_log0 else res.lat_raw).append(round(w_end - t_dec, 4))
            if is_own:
                if lat_open_turn != turn:
                    if lat_open_turn is not None:
                        res.lat_turn.append(round(lat_last_end - lat_open_start, 4))
                    lat_open_turn, lat_open_start = turn, t_dec
                lat_last_end = w_end
    if latency and lat_open_turn is not None:
        res.lat_turn.append(round(lat_last_end - lat_open_start, 4))
    flight.close()
    res.truncated = obs is not None
    res.winner = None if res.truncated else int(env.winner)
    res.n_capped = search.capped - capped0
    return res
