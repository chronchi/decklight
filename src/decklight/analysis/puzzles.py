""" "Win this turn" puzzles mined from game records (web-ux-plan step 5, item 8).

A puzzle is a position at the turn player's first decision of main phase 1 from
a game the turn player went on to win **in that same turn**, with at least 4 of
the solver's own multi-option decisions on the way. The web runs it as a
practice ``DuelSession`` in ``fast`` mode (``ProductLauncher(fast=True)``);
:func:`verify_puzzle` replays exactly that setup — the solver's recorded
choices, the bot answering live — and accepts the candidate only if the solver
still wins in the same turn. When the live bot answers differently from the
record, the recorded choice is illegal or the position diverges, and the
candidate is rejected with a reason.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from decklight.analysis.events import deck_path
from decklight.decks import load_ydk
from decklight.engine import constants as C
from decklight.engine import query as Q
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv

MIN_MOVES = 4  # solver multi-option decisions from the start to the win
TARGET_PUZZLES = 60
PER_GROUP_CAP = 2  # at most 2 puzzles per deck pair + turn
PLAY_KINDS = ("SUMMON", "SPSUMMON", "ACTIVATE")  # option kinds whose code is recorded
# Meaningful plays for difficulty: the solver's game actions (SET = MSET + SSET).
PLAY_CHOICE_KINDS = frozenset({"ACTIVATE", "SUMMON", "SPSUMMON", "MSET", "SSET", "ATTACK", "TO_BATTLE"})
NAIVE_ROLLOUTS = 8  # solver sampled at temperature 1.0, bot live (see naive_win_rate)
NAIVE_KEEP_MAX = 0.5  # keep puzzles the bot's own policy fails at least half the time


def difficulty(n_plays: int) -> str:
    if n_plays <= 3:
        return "easy"
    if n_plays <= 6:
        return "medium"
    return "hard"


def count_plays(solution_steps: Sequence[dict], seat: int) -> int:
    """Meaningful solver plays: ACTIVATE / SUMMON / SPSUMMON / SET / ATTACK / TO_BATTLE."""
    n = 0
    for st in solution_steps:
        try:
            if int(st.get("player", -1)) != int(seat):
                continue
        except (TypeError, ValueError):
            continue
        if str(st.get("kind")) in PLAY_CHOICE_KINDS:
            n += 1
    return int(n)


def goal_for(opp_lp: int) -> str:
    return f"Win this turn \u00b7 opponent has {int(opp_lp):,} LP"


def _signed_lps(env) -> tuple[int, int]:
    raw = Q.lps(env.duel.query_field())
    return tuple(v - (1 << 32) if v >= 1 << 31 else v for v in raw)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(name))[:24] or "deck"


def _replay_steps(engine: Engine, cfg: DuelConfig, choices: Sequence[int]) -> tuple[PyEnv, list[dict]]:
    """Replay host choices (auto_single=True, the record space); one step dict per choice.

    Raises ValueError on illegal choices, overruns, or unfinished replays.
    """
    env = PyEnv(engine, cfg, auto_single=True)
    env.reset()
    steps: list[dict] = []
    for ch in choices:
        if env.done:
            raise ValueError("choices overrun the duel")
        opts = list(env.opts)
        if not 0 <= int(ch) < len(opts):
            raise ValueError(f"illegal choice {ch} of {len(opts)}")
        opt = opts[int(ch)]
        try:
            kind = opt.kind.name
        except Exception:  # noqa: BLE001 -- defensive
            kind = str(opt.kind)
        code = int(env.board.option_code(opt))
        steps.append(
            {
                "player": int(env.player),
                "n_opts": len(opts),
                "kind": kind,
                "code": code,
                "turn": int(env.duel.turn),
                "phase": int(env.duel.phase),
                "turn_player": int(env.duel.turn_player),
                "lp": _signed_lps(env),
                "choice": int(ch),
            }
        )
        env.step(int(ch))
    return env, steps


def replay_false(engine: Engine, cfg: DuelConfig, true_choices: Sequence[int]) -> tuple[PyEnv, list[dict]]:
    """Replay record (auto_single=True-space) choices on a host (auto_single=False) env.

    Returns the env plus one step dict per *host* step (singletons included).
    Each step carries ``ti``: the record index it consumes (None for singletons).
    Raises ValueError when the record underruns, overruns, or misaligns.
    """
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    steps: list[dict] = []
    ti = 0
    while not env.done:
        opts = list(env.opts)
        n = len(opts)
        if n == 1:
            step_ti = None
            c = 0
        else:
            if ti >= len(true_choices):
                raise ValueError("record ends before the duel does")
            step_ti = ti
            c = int(true_choices[ti])
            ti += 1
            if not 0 <= c < n:
                raise ValueError(f"illegal choice {c} of {n}")
        opt = opts[c]
        try:
            kind = opt.kind.name
        except Exception:  # noqa: BLE001 -- defensive
            kind = str(opt.kind)
        code = int(env.board.option_code(opt))
        steps.append(
            {
                "player": int(env.player),
                "n_opts": n,
                "kind": kind,
                "code": code,
                "turn": int(env.duel.turn),
                "phase": int(env.duel.phase),
                "turn_player": int(env.duel.turn_player),
                "lp": _signed_lps(env),
                "choice": c,
                "ti": step_ti,
            }
        )
        env.step(c)
    if ti != len(true_choices):
        raise ValueError(f"record overruns the duel ({len(true_choices) - ti} choices left)")
    return env, steps


def mine_candidates(engine: Engine, records: list[dict]) -> tuple[list[dict], dict]:
    """Mine puzzle candidates from game records (no search; verification is separate).

    The stored ``prefix``/``solution``/``actors`` are host-space choices
    (``PyEnv(auto_single=False)``, the same space the web's DuelSession replays),
    so verification feeds them directly and derives the product prefix with
    ``host_to_product``. Returns (candidates, stats). Records that fail to
    replay raise (fail loud).
    """
    candidates: list[dict] = []
    stats = Counter(scanned=0, truncated=0, no_mp1=0, no_win=0, too_few_moves=0, kept=0)
    for rec in records:
        stats["scanned"] += 1
        if rec.get("truncated") or rec.get("winner") is None:
            stats["truncated"] += 1
            continue
        if any(not isinstance(d, str) for d in rec["decks"]):
            raise ValueError(f"game {rec.get('game_idx')}: puzzles need named decks")
        decks = [load_ydk(deck_path(d)) for d in rec["decks"]]
        cfg = DuelConfig(decks[0], decks[1], seed=rec["seed"])
        choices = [int(c) for c in rec["choices"]]
        env, steps = replay_false(engine, cfg, choices)
        if not env.done or env.winner != rec["winner"]:
            raise ValueError(f"game {rec.get('game_idx')}: replay diverged from the record")
        final_turn = int(env.duel.turn)
        # First multi-option decision of main phase 1 per (turn, turn player).
        starts: dict[tuple[int, int], int] = {}
        for k, st in enumerate(steps):
            if st["phase"] == C.PHASE_MAIN1 and st["player"] == st["turn_player"] and st["n_opts"] > 1:
                starts.setdefault((st["turn"], st["player"]), k)
        if not starts:
            stats["no_mp1"] += 1
            continue
        for (t, s), k in sorted(starts.items()):
            if rec["winner"] != s or final_turn != t:
                continue
            n_moves = sum(1 for st in steps[k:] if st["player"] == s and st["n_opts"] > 1)
            if n_moves < MIN_MOVES:
                stats["too_few_moves"] += 1
                continue
            opp_lp = steps[k]["lp"][1 - s]
            sol = steps[k:]
            sol_steps = [{"player": st["player"], "kind": st["kind"], "code": st["code"]} for st in sol]
            n_plays = count_plays(sol_steps, int(s))
            candidates.append(
                {
                    "id": f"{_slug(rec['decks'][0])}-vs-{_slug(rec['decks'][1])}-t{t}p{s}-{rec['seed']}",
                    "decks": list(rec["decks"]),
                    "seed": int(rec["seed"]),
                    "prefix": [st["choice"] for st in steps[:k]],
                    "seat": int(s),
                    "turn": int(t),
                    "opp_lp": int(opp_lp),
                    "goal": goal_for(int(opp_lp)),
                    "solution": [st["choice"] for st in sol],
                    "actors": [st["player"] for st in sol],
                    "solution_steps": sol_steps,
                    "n_moves": int(n_moves),
                    "n_plays": int(n_plays),
                    "difficulty": difficulty(int(n_plays)),
                    "verified": False,
                }
            )
            stats["kept"] += 1
    stats["no_win"] = (
        stats["scanned"] - stats["truncated"] - stats["kept"] - stats["no_mp1"] - stats["too_few_moves"]
    )
    stats["no_win"] = max(stats["no_win"], 0)
    return candidates, dict(stats)


def select(candidates: list[dict], target: int = TARGET_PUZZLES) -> list[dict]:
    """Variety-first ordering: round-robin over (deck pair, turn) groups, cap 2 each."""
    groups: dict[tuple, list[dict]] = {}
    for c in candidates:
        groups.setdefault((tuple(sorted(c["decks"])), c["turn"]), []).append(c)
    for ls in groups.values():
        ls.sort(key=lambda c: (c["n_moves"], c["id"]))
        del ls[PER_GROUP_CAP:]
    ordered: list[dict] = []
    keys = sorted(groups)
    while len(ordered) < target and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(ordered) < target:
                ordered.append(groups[k].pop(0))
    return ordered


def _launcher_logits(launcher, obs: dict):
    import numpy as np

    if hasattr(launcher, "base_logits"):
        return np.asarray(launcher.base_logits([obs])).reshape(-1)
    if hasattr(launcher, "forward"):
        return np.asarray(launcher.forward([obs])).reshape(-1)
    raise AttributeError("launcher has neither base_logits nor forward")


def _sample_temp1(logits, mask, rng):
    import numpy as np

    z = np.array(logits[: len(mask)], dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    z[~m] = -np.inf
    z -= np.nanmax(z)
    e = np.exp(z)
    e[~m] = 0.0
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        legal = np.flatnonzero(m)
        if len(legal) == 0:
            raise ValueError("no legal options to sample")
        return int(rng.choice(legal))
    return int(rng.choice(len(e), p=e / s))


def _puzzle_session(engine, puzzle, launcher, game_seed=None):
    """Shared DuelSession setup for verify + naive rollouts (prefix not yet replayed).

    Returns (session, prefix, prefix_decisions, seat, start_turn, cfg).
    Raises ValueError/DriverError on mapping failures (callers turn these into rejections).
    """
    from decklight.edopro.product_bot import host_to_product
    from decklight.play.session import DuelSession

    decks = [load_ydk(deck_path(d)) for d in puzzle["decks"]]
    cfg = DuelConfig(decks[0], decks[1], seed=puzzle["seed"])
    seat = int(puzzle["seat"])
    start_turn = int(puzzle["turn"])
    prefix = [int(c) for c in puzzle["prefix"]]
    product_prefix = [int(c) for c in host_to_product(engine, cfg, prefix)]
    env0 = PyEnv(engine, cfg, auto_single=False)
    env0.reset()
    prefix_decisions: list[dict] = []
    for c in prefix:
        if env0.done:
            raise ValueError("prefix overruns the duel")
        if len(env0.opts) > 1:
            prefix_decisions.append({"player": int(env0.player), "n_options": len(env0.opts)})
        env0.step(int(c))
    gs = puzzle["seed"] if game_seed is None else game_seed
    session = DuelSession(
        engine,
        cfg,
        human_seat=seat,
        launcher=launcher,
        seed=int(gs),
        driver_prefix=product_prefix,
        driver_prefix_decisions=prefix_decisions,
    )
    return session, prefix, prefix_decisions, seat, start_turn, cfg


def _replay_prefix(session, prefix) -> tuple[bool, str]:
    for c in prefix:
        if session.env.done:
            return False, "prefix overruns the duel"
        try:
            session.env.step(int(c))
        except (IndexError, ValueError) as e:
            return False, f"prefix diverged: {e}"
    return True, "ok"


def naive_win_rate(
    engine, puzzle: dict, launcher, n_rollouts: int = NAIVE_ROLLOUTS, seed: int | None = None
) -> float:
    """Share of rollouts where the fast policy, sampled at temperature 1.0, still wins this turn.

    From the puzzle start the solver's multi-option choices are sampled from the
    launcher's policy (``base_logits``/``forward``) at temperature 1.0 while the
    bot answers live exactly as in :func:`verify_puzzle` (``session.step_bot``).
    Seeded and reproducible: ``seed`` defaults to the puzzle seed, spawned per rollout.
    """
    import numpy as np

    base = int(puzzle["seed"]) if seed is None else int(seed)
    try:
        children = np.random.SeedSequence(int(base)).spawn(int(n_rollouts))
    except Exception:  # noqa: BLE001 -- defensive: fall back to a simple stream
        children = [
            np.random.SeedSequence((int(base) + 7919 * (r + 1)) & 0xFFFFFFFF) for r in range(int(n_rollouts))
        ]
    wins = 0
    for child in children:
        rng = np.random.default_rng(child)
        try:
            won = _naive_rollout_once(engine, puzzle, launcher, rng)
        except Exception:  # noqa: BLE001 -- a crashed rollout never wins
            won = False
        wins += 1 if won else 0
    return float(wins) / float(max(1, int(n_rollouts)))


def _naive_rollout_once(engine, puzzle: dict, launcher, rng) -> bool:
    """One sampled rollout: True iff the sampled solver still wins in the puzzle turn."""
    from decklight.edopro.product_bot import DriverError
    from decklight.play.session import DriverDiverged

    try:
        session, prefix, prefix_decisions, seat, start_turn, _cfg = _puzzle_session(
            engine, puzzle, launcher, game_seed=None
        )
    except (ValueError, DriverError, RuntimeError):
        return False
    try:
        try:
            session.reset()
        except (DriverDiverged, DriverError, ValueError, RuntimeError):
            return False
        ok, _ = _replay_prefix(session, prefix)
        if not ok:
            return False
        # Mirror verify's prod_len (host multi-option prefix length).
        try:
            session.prod_len = len(prefix_decisions)
        except Exception:  # noqa: BLE001, S110 -- best effort
            pass
        while not session.env.done:
            if int(session.env.duel.turn) != int(start_turn):
                return False
            n = len(session.env.opts)
            player = int(session.env.player)
            try:
                if n > 1 and player == seat:
                    obs = session.env._observe()
                    logits = _launcher_logits(launcher, obs)
                    idx = _sample_temp1(logits, obs["action_mask"], rng)
                    session.step_human(int(idx))
                elif n > 1:
                    session.step_bot()
                else:
                    session.env.step(0)
            except (DriverDiverged, DriverError, IndexError, ValueError):
                return False
        return bool(session.env.winner == seat and int(session.env.duel.turn) == int(start_turn))
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001, S110 -- best effort
            pass


def verify_puzzle(
    engine: Engine, puzzle: dict, launcher, game_seed: int | None = None, ws_seed: int | None = None
) -> tuple[bool, str]:
    """Check a puzzle the way the web runs it: practice DuelSession, solver human, bot live.

    Feeds the solver's recorded choices (prefix, then solution) into the
    session; the bot's multi-option decisions come from ``launcher`` live. The
    candidate is accepted only if the solver still wins in the puzzle turn.
    Returns (ok, reason) and never raises (driver failures are rejections).
    """
    from decklight.edopro.product_bot import DriverError
    from decklight.play.session import DriverDiverged

    try:
        seat = int(puzzle["seat"])
        start_turn = int(puzzle["turn"])
        prefix = [int(c) for c in puzzle["prefix"]]
        try:
            session, _p, prefix_decisions, _s, _t, _c = _puzzle_session(
                engine, puzzle, launcher, game_seed=game_seed
            )
        except ValueError as e:
            return False, f"prefix mapping failed: {e}"
        _ = ws_seed  # the session derives ws_seed from its seed, like the web's what-if
        try:
            session.reset()
        except (DriverDiverged, DriverError, ValueError, RuntimeError) as e:
            return False, f"session start failed: {e}"
        try:
            # The prefix is replayed on the host env directly, exactly like the
            # web's what-if path: the driver replays it verbatim internally
            # (never re-searched), so it is never asked for prefix choices.
            ok, reason = _replay_prefix(session, prefix)
            if not ok:
                return False, reason
            try:
                session.prod_len = len(prefix_decisions)
            except Exception:  # noqa: BLE001, S110 -- best effort
                pass
            sol = [int(c) for c in puzzle["solution"]]
            actors = [int(a) for a in puzzle["actors"]]
            if len(sol) != len(actors):
                return False, "solution/actors length mismatch"
            ok, reason = _feed_paired(session, sol, actors, seat, start_turn)
            if not ok:
                return False, reason
            if not session.env.done:
                return False, "solution ends before the duel does"
            if session.env.winner != seat:
                return False, f"no win (winner seat {session.env.winner})"
            if int(session.env.duel.turn) != start_turn:
                return False, f"win came on turn {session.env.duel.turn}, not {start_turn}"
            return True, f"won on turn {start_turn}"
        finally:
            try:
                session.close()
            except Exception:  # noqa: BLE001, S110 -- best effort; the verdict stands
                pass
    except (DriverDiverged, DriverError, IndexError, ValueError) as e:
        return False, f"rejected: {e}"
    except Exception as e:  # noqa: BLE001 -- never raise out of verification
        return False, f"verify crashed: {type(e).__name__}: {e}"


def _feed_paired(
    session, solution: list[int], actors: list[int], seat: int, start_turn: int
) -> tuple[bool, str]:
    from decklight.play.session import DriverDiverged

    for c, a in zip(solution, actors):
        if session.env.done:
            break
        if int(session.env.duel.turn) != start_turn:
            return False, f"turn advanced to {session.env.duel.turn} without a win"
        if int(session.env.player) != int(a):
            return False, (f"actor mismatch (host player {session.env.player}, record {a})")
        player, n = int(session.env.player), len(session.env.opts)
        try:
            if n > 1 and player == seat:
                session.step_human(int(c))
            elif n > 1:
                session.step_bot()
                if session.env.choices[-1] != int(c):
                    return False, (
                        f"bot answered differently (played {session.env.choices[-1]}, "
                        f"record {c}); the recorded reply is stale"
                    )
            else:
                session.env.step(int(c))
        except (DriverDiverged, IndexError, ValueError) as e:
            return False, f"solution diverged: {e}"
    return True, "ok"


def render_text(engine: Engine, puzzle: dict, names: dict[int, str] | None = None) -> str:
    """The puzzle as text: the start position (both fields, hands, LP) + named moves."""
    if names is None:
        from decklight.web.view import DB

        names = DB.names()
    decks = [load_ydk(deck_path(d)) for d in puzzle["decks"]]
    cfg = DuelConfig(decks[0], decks[1], seed=puzzle["seed"])
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    for c in puzzle["prefix"]:
        if env.done:
            break
        env.step(int(c))
    lp = _signed_lps(env)
    lines = [
        (
            f"puzzle {puzzle['id']} (turn {puzzle['turn']}, solver seat {puzzle['seat']}, "
            f"difficulty {puzzle['difficulty']}, opp LP {puzzle['opp_lp']})"
        ),
        f"{puzzle['decks'][0]} (seat 0, LP {lp[0]}) vs {puzzle['decks'][1]} (seat 1, LP {lp[1]})",
    ]
    for p in (0, 1):
        hand = [c.code for c in env.duel.query_location(p, C.LOCATION_HAND) if c is not None]
        field = [(c.sequence, c.code) for c in env.duel.query_location(p, C.LOCATION_MZONE) if c is not None]
        backrow = [
            (c.sequence, c.code) for c in env.duel.query_location(p, C.LOCATION_SZONE) if c is not None
        ]
        grave = [c.code for c in env.duel.query_location(p, C.LOCATION_GRAVE) if c is not None]

        def nm(code: int) -> str:
            return names.get(code, f"#{code}") if code else "???"

        lines.append(f"seat {p} hand: {', '.join(nm(c) for c in hand) or '(empty)'}")
        lines.append(f"seat {p} monsters: " + (", ".join(f"[{s}] {nm(c)}" for s, c in field) or "(none)"))
        lines.append(f"seat {p} backrow: " + (", ".join(f"[{s}] {nm(c)}" for s, c in backrow) or "(none)"))
        lines.append(f"seat {p} grave: {', '.join(nm(c) for c in grave[:8]) or '(empty)'}")
    lines.append("solution:")
    for i, st in enumerate(puzzle["solution_steps"]):
        code = st["code"]
        lines.append(f"  {i:>3} P{st['player']} {st['kind']} {names.get(code, f'#{code}') if code else '-'}")
    return "\n".join(lines)
