"""Chess-engine-style annotation of stored `search_match` game records (engine plan N0).

Engine replay only: no torch, no model loading. The evals come from the search log
entries already stored in each record (`samples`: per-candidate per-world leaf values
in [-1, 1] from the searching seat's view; worlds are paired across candidates).

Two layers:
- `replay()` walks the record through the C++ env (needs an Engine + deck loader +
  a describe callback) and returns plain-data snapshots.
- `annotate_game()` / `summarize()` / markdown renderers are pure functions over
  plain data (unit-testable without the engine).
"""

from __future__ import annotations

import math
import statistics as st
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime

LOCATIONS = ("deck", "hand", "mzone", "szone", "grave", "removed", "extra")
N_LOCS = 7

# Label thresholds on regret, in v units (win% = v / 2).
LABEL_LEVELS = (("blunder", 0.3), ("mistake", 0.2), ("inaccuracy", 0.1))

CAL_BIN_WIDTH = 0.1
CAL_N_BINS = 10
BRIER_BUCKETS = (((1, 2), "1-2"), ((3, 4), "3-4"), ((5, 8), "5-8"), ((9, 10**9), "9+"))
SIGN_BUCKETS = ("<=2", "3", "4", "5+")


class ReplayMismatch(Exception):
    """The engine replay did not reproduce the record (fail loud, CLI exits 2)."""


def sign_bucket(turn: int) -> str:
    if turn <= 2:
        return "<=2"
    if turn == 3:
        return "3"
    if turn == 4:
        return "4"
    return "5+"


def brier_bucket(turn: int) -> str:
    for (lo, hi), name in BRIER_BUCKETS:
        if lo <= turn <= hi:
            return name
    return "9+"


def _check_finite(values: list[float], where: str) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"NaN/non-finite value in {where}")


def eval_entry(e: dict) -> dict:
    """Annotate one search-log entry (pure; needs no replay state).

    Returns per-candidate means plus best/regret/se/label from the searching
    seat's view, and the same best eval from player 0's view (`v0`).
    """
    cands = [int(c) for c in e["cands"]]
    samples = [[float(x) for x in w] for w in e["samples"]]
    if not cands or not samples or len(samples) != len(cands):
        raise ValueError(f"malformed search entry k={e.get('k')}: cands/samples mismatch")
    n = len(samples[0])
    if n == 0 or any(len(w) != n for w in samples):
        raise ValueError(f"malformed search entry k={e.get('k')}: ragged worlds")
    for w in samples:
        _check_finite(w, f"search entry k={e.get('k')}")
    seat = int(e["seat"])
    means = [st.mean(w) for w in samples]
    if e.get("best") is not None:
        best_opt = int(e["best"])
        if best_opt not in cands:
            raise ValueError(f"entry k={e.get('k')}: best={best_opt} not in cands")
        best_pos = cands.index(best_opt)
    else:
        best_pos = max(range(len(cands)), key=lambda i: means[i])
    v_best = means[best_pos]
    risk = st.pstdev(samples[best_pos]) if n > 1 else 0.0
    played = int(e["played"])
    chosen = int(e["chosen"])
    regret: float | None = None
    se: float | None = None
    gap_se_ratio = 0.0
    significant = False
    if played in cands:
        pos_p = cands.index(played)
        diffs = [a - b for a, b in zip(samples[best_pos], samples[pos_p], strict=True)]
        regret = means[best_pos] - means[pos_p]
        if n > 1:
            sd = st.stdev(diffs)
            se = sd / math.sqrt(n)
        else:
            se = 0.0
        assert se is not None
        if se > 1e-9:
            gap_se_ratio = regret / se
            significant = regret >= 2 * se
        else:
            # No spread across worlds: any positive gap counts as significant.
            # Capped like scripts/audit/search_override_se.py (99.0), not inf,
            # so the JSON output stays finite.
            gap_se_ratio = 99.0 if regret > 0 else 0.0
            significant = regret > 0
    label = "none"
    if regret is not None and significant:
        for name, thr in LABEL_LEVELS:
            if regret >= thr:
                label = name
                break
    v0 = v_best if seat == 0 else -v_best
    return {
        "k": int(e["k"]),
        "turn": int(e["turn"]),
        "seat": seat,
        "kind": str(e.get("kind", "?")),
        "cands": [{"opt": c, "mean": m} for c, m in zip(cands, means, strict=True)],
        "best": cands[best_pos],
        "played": played,
        "chosen": chosen,
        "overridden": chosen != played,
        "v_best": v_best,
        "phat_searcher": (v_best + 1) / 2,
        "v0": v0,
        "p0": (v0 + 1) / 2,
        "risk": risk,
        "regret": regret,
        "regret_pp": (regret / 2 * 100) if regret is not None else None,
        "se": se,
        "gap_se": gap_se_ratio,
        "significant": significant,
        "label": label,
        "msg": e.get("msg"),
        "n_worlds": n,
    }


def replay(
    rec: dict,
    engine,
    deck_loader: Callable[[str], object],
    describe: Callable[[object, dict, int, dict], str],
    names: dict,
) -> tuple[list[dict], dict, dict[int, list[tuple[int, str]]]]:
    """Step `rec["choices"]` through the env; return (snaps, end_snap, descs).

    snaps[k]: pre-step snapshot at decision k (absolute players 0/1).
    end_snap: last observed pre-step snapshot (game-end state for deltas).
    descs[k]: [(cand option idx, description)] at searched decisions.
    Raises ReplayMismatch when a non-truncated replay diverges from the record.
    """
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig

    decks = rec["decks"]
    cfg = DuelConfig(deck_loader(decks[0]), deck_loader(decks[1]), seed=int(rec["seed"]))
    env = CppEnv(engine, cfg)
    obs = env.reset()
    search = {int(e["k"]): e for e in rec.get("search", [])}
    choices = [int(c) for c in rec["choices"]]
    snaps: list[dict] = []
    descs: dict[int, list[tuple[int, str]]] = {}
    for k, ch in enumerate(choices):
        if obs is None:
            if rec.get("truncated"):
                break
            raise ReplayMismatch(f"game {rec.get('game_idx')}: replay ended before choice {k}")
        g = obs["global"]
        actor = int(env.player)
        if actor == 0:
            lp = [int(g[0]), int(g[1])]
            counts = {0: [int(x) for x in g[6:13]], 1: [int(x) for x in g[13:20]]}
        else:
            lp = [int(g[1]), int(g[0])]
            counts = {0: [int(x) for x in g[13:20]], 1: [int(x) for x in g[6:13]]}
        turn = int(g[2])
        tp = actor if int(g[5]) == 1 else 1 - actor
        snaps.append(
            {
                "k": k,
                "turn": turn,
                "actor": actor,
                "turn_player": tp,
                "lp": lp,
                "counts": counts,
                "msg": int(g[21]),
            }
        )
        if k in search:
            e = search[k]
            descs[k] = [(int(c), describe(env, obs, int(c), names)) for c in e["cands"]]
        obs = env.step(ch)
    end_snap = snaps[-1] if snaps else {}
    if rec.get("truncated"):
        return snaps, end_snap, descs
    if obs is not None or int(env.winner) != int(rec["winner"]):
        raise ReplayMismatch(
            f"game {rec.get('game_idx')}: replay winner {env.winner} != record {rec['winner']}"
        )
    return snaps, end_snap, descs


def _card_adv(counts: dict, p0: int = 0, p1: int = 1) -> int:
    def hand_field(p: int) -> int:
        return counts[p][1] + counts[p][2] + counts[p][3]

    return hand_field(p0) - hand_field(p1)


def annotate_game(
    rec: dict,
    snaps: list[dict],
    end_snap: dict,
    descs: dict[int, list[tuple[int, str]]],
    values: dict | None = None,
) -> dict:
    """Build the per-game annotation from a record + replay snapshots (pure).

    `values` is one `score_decisions.py` line for this game (k/actor/turn/v
    lists, actor's-view win estimates in [-1, 1]) or None. Without it the
    output is the search-log annotation; with it the game additionally gains
    `curve_full` (per-decision v0), per-turn v_p0/v_p1/v_comb/disagree, a
    full-curve `tempo` (the search tempo moves to `tempo_search`, likewise
    `swing_turn` -> `swing_turn_search`).
    """
    if not snaps:
        raise ValueError(f"game {rec.get('game_idx')}: no replay snapshots")
    search_by_k = {int(e["k"]): e for e in rec.get("search", [])}
    for k in search_by_k:
        if k < 0 or k >= len(snaps):
            raise ValueError(f"game {rec.get('game_idx')}: search k={k} outside replay")

    # Per-turn table: first-decision state + deltas to next turn start (or game end).
    first_k_of_turn: dict[int, int] = {}
    for s in snaps:
        first_k_of_turn.setdefault(int(s["turn"]), int(s["k"]))
    turn_nos = sorted(first_k_of_turn)
    turns: dict[int, dict] = {}
    for i, t in enumerate(turn_nos):
        s = snaps[first_k_of_turn[t]]
        nxt = snaps[first_k_of_turn[turn_nos[i + 1]]] if i + 1 < len(turn_nos) else end_snap
        turns[t] = {
            "turn_player": int(s["turn_player"]),
            "lp": [int(s["lp"][0]), int(s["lp"][1])],
            "counts": {p: [int(x) for x in s["counts"][p]] for p in (0, 1)},
            "card_adv": _card_adv({p: s["counts"][p] for p in (0, 1)}),
            "d_lp": [int(nxt["lp"][0]) - int(s["lp"][0]), int(nxt["lp"][1]) - int(s["lp"][1])],
            "d_counts": {
                p: [int(b) - int(a) for a, b in zip(s["counts"][p], nxt["counts"][p], strict=True)]
                for p in (0, 1)
            },
            "d_card_adv": _card_adv({p: nxt["counts"][p] for p in (0, 1)})
            - _card_adv({p: s["counts"][p] for p in (0, 1)}),
            "msg": int(s["msg"]),
        }

    # Evals (in decision order), with candidate descriptions and msg types.
    evals: list[dict] = []
    for k in sorted(search_by_k):
        ev = eval_entry(search_by_k[k])
        s = snaps[k]
        ev["msg"] = int(s["msg"])
        dmap = {c: d for c, d in descs.get(k, [])}
        for i, c in enumerate(ev["cands"]):
            w = [float(x) for x in search_by_k[k]["samples"][i]]
            c["desc"] = dmap.get(c["opt"], f"option {c['opt']}")
            c["win_pct"] = (c["mean"] + 1) / 2 * 100
            # SE of the candidate's mean eval across its worlds.
            c["se_mean"] = (st.stdev(w) / math.sqrt(len(w))) if len(w) > 1 else 0.0
            _check_finite([c["mean"], c["win_pct"], c["se_mean"]], f"game eval k={k}")
        order = {c["opt"]: i for i, c in enumerate(ev["cands"])}
        ev["best_desc"] = ev["cands"][order[ev["best"]]]["desc"]
        played_descs = [c["desc"] for c in ev["cands"] if c["opt"] == ev["played"]]
        ev["played_desc"] = played_descs[0] if played_descs else f"option {ev['played']}"
        evals.append(ev)

    # Eval curve: v0 at each turn's first searched decision (None when none).
    t_max = max([int(rec.get("turns", 0))] + turn_nos + [int(e["turn"]) for e in evals] + [1])
    first_eval_of_turn: dict[int, dict] = {}
    for ev in evals:
        first_eval_of_turn.setdefault(int(ev["turn"]), ev)
    curve = {
        t: (first_eval_of_turn[t]["v0"] if t in first_eval_of_turn else None) for t in range(1, t_max + 1)
    }

    # Tempo: win% delta for the turn player between consecutive curve points.
    tempo: dict[int, dict] = {}
    curve_turns = [t for t in range(1, t_max + 1) if curve[t] is not None]
    for i, t in enumerate(curve_turns[:-1] if len(curve_turns) > 1 else []):
        t2 = curve_turns[i + 1]
        dp0 = (curve[t2] + 1) / 2 - (curve[t] + 1) / 2  # player-0 win% delta
        tp = turns[t]["turn_player"] if t in turns else None
        if tp is None:
            continue
        dp_tp = dp0 if tp == 0 else -dp0
        tempo[t] = {
            "to_turn": t2,
            "dp_tp": dp_tp,
            "dp_tp_pp": dp_tp * 100,
            "turn_player": tp,
            "tempo_cards": turns[t]["d_card_adv"] if tp == 0 else -turns[t]["d_card_adv"],
            "tempo_lp": turns[t]["d_lp"][tp],
        }
    swing_turn = max(tempo, key=lambda t: abs(tempo[t]["dp_tp"])) if tempo else None

    labels = [ev["label"] for ev in evals]
    regrets = [ev["regret"] for ev in evals if ev["regret"] is not None]
    ann = {
        "game_idx": rec.get("game_idx"),
        "decks": list(rec.get("decks", [])),
        "seed": rec.get("seed"),
        "seat": rec.get("s_seat"),
        "s_first": bool(rec.get("s_first")),
        "deck_s": rec.get("deck_s"),
        "winner": rec.get("winner"),
        "truncated": bool(rec.get("truncated")),
        "n_choices": len(snaps),
        "turns": turns,
        "evals": evals,
        "curve": curve,
        "tempo": tempo,
        "swing_turn": swing_turn,
        "summary": {
            "n_searched": len(evals),
            "n_overridden": sum(1 for ev in evals if ev["overridden"]),
            "labels": {
                name: sum(1 for lb in labels if lb == name) for name in ("inaccuracy", "mistake", "blunder")
            },
            "mean_regret_pp": (sum(r / 2 * 100 for r in regrets) / len(regrets) if regrets else 0.0),
        },
    }
    if values is not None:
        _attach_values(ann, turns, turn_nos, snaps, values, rec.get("game_idx"))
    _assert_no_nan(ann)
    return ann


def _attach_values(
    ann: dict, turns: dict[int, dict], turn_nos: list[int], snaps: list[dict], values: dict, game_idx
) -> None:
    """Add the value-net full curve, per-turn v_comb/disagreement and full-curve tempo (pure)."""
    vk = [int(k) for k in values.get("k", [])]
    va = [int(x) for x in values.get("actor", [])]
    vt = [int(x) for x in values.get("turn", [])]
    vv = [float(x) for x in values.get("v", [])]
    n = len(snaps)
    if not (len(vk) == len(va) == len(vt) == len(vv) == n):
        raise ValueError(
            f"game {game_idx}: values length {[len(vk), len(va), len(vt), len(vv)]} != {n} replay decisions"
        )
    if vk != list(range(n)):
        raise ValueError(f"game {game_idx}: values k is not 0..{n - 1}")
    for v in vv:
        _check_finite([v], f"game {game_idx} values")
    v_by_k = dict(zip(vk, vv, strict=True))
    curve_full = []
    for k, s in enumerate(snaps):
        v = v_by_k[k]
        a = int(s["actor"])
        if a != va[k] or int(s["turn"]) != vt[k]:
            raise ValueError(f"game {game_idx}: values actor/turn mismatch at k={k}")
        v0 = v if a == 0 else -v
        curve_full.append({"k": k, "actor": a, "turn": int(s["turn"]), "v": v, "v0": v0})
    ann["curve_full"] = curve_full

    # Each player's own value at their first decision in each turn.
    first_v: dict[tuple[int, int], float] = {}
    for k, s in enumerate(snaps):
        first_v.setdefault((int(s["turn"]), int(s["actor"])), v_by_k[k])
    for t in turn_nos:
        v_p0 = first_v.get((t, 0))
        v_p1 = first_v.get((t, 1))
        v_comb: float | None
        disagree: float | None
        if v_p0 is not None and v_p1 is not None:
            v_comb = (v_p0 - v_p1) / 2
            disagree = v_p0 + v_p1
        elif v_p0 is not None:
            v_comb = v_p0  # player 0's view already
            disagree = None
        elif v_p1 is not None:
            v_comb = -v_p1  # player 1's view flipped to player 0's view
            disagree = None
        else:
            v_comb = None
            disagree = None
        turns[t]["v_p0"] = v_p0
        turns[t]["v_p1"] = v_p1
        turns[t]["v_comb"] = v_comb
        turns[t]["disagree"] = disagree

    # Tempo from the full curve: v_comb change between consecutive turn starts,
    # from the turn player's view (win pp = dv * 50). The search tempo is kept.
    ann["tempo_search"] = ann["tempo"]
    ann["swing_turn_search"] = ann["swing_turn"]
    full: dict[int, dict] = {}
    have = [t for t in turn_nos if turns[t]["v_comb"] is not None]
    for i, t in enumerate(have[:-1]):
        t2 = have[i + 1]
        tp = turns[t]["turn_player"]
        dpp = (turns[t2]["v_comb"] - turns[t]["v_comb"]) * 50.0
        if tp == 1:
            dpp = -dpp
        full[t] = {
            "to_turn": t2,
            "dp_tp": dpp / 100.0,
            "dp_tp_pp": dpp,
            "turn_player": tp,
            "tempo_cards": turns[t]["d_card_adv"] if tp == 0 else -turns[t]["d_card_adv"],
            "tempo_lp": turns[t]["d_lp"][tp],
        }
    ann["tempo"] = full
    ann["swing_turn"] = max(full, key=lambda t: abs(full[t]["dp_tp"])) if full else None
    if isinstance(values.get("aux"), dict):
        _attach_aux(ann, turns, turn_nos, snaps, values, game_idx)


def _aux_starts(snaps: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """Per seat: [(duel turn, first decision pos)] of that seat's own-turn starts.

    Own-turn start = the seat's first decision in a duel turn where it is the
    turn player (mirrors eval/aux._own_turn_starts, on replay snaps).
    """
    starts: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    seen: dict[int, set[int]] = {0: set(), 1: set()}
    for j, s in enumerate(snaps):
        a = int(s["actor"])
        t = int(s["turn"])
        if int(s["turn_player"]) == a and t not in seen[a]:
            seen[a].add(t)
            starts[a].append((t, j))
    return starts


def _attach_aux(
    ann: dict, turns: dict[int, dict], turn_nos: list[int], snaps: list[dict], values: dict, game_idx
) -> None:
    """Add per-turn-per-seat aux predictions + realized outcomes (pure).

    Predictions are read at the seat's first decision of the turn; realized
    outcomes are computed from the replay snaps by that seat's next own-turn
    start (or the game end). Stored as turns[t]["aux"][seat] with string seat
    keys for JSON stability (counts use int keys internally; both read back).
    """
    aux = values.get("aux", {})
    n = len(snaps)
    try:
        ad_lp = [[float(x) for x in row] for row in aux.get("d_lp", [])]
        ad_cards = [[float(x) for x in row] for row in aux.get("d_cards", [])]
        a_ends = [float(x) for x in aux.get("ends", [])]
        a_turns = [float(x) for x in aux.get("turns_left", [])]
    except (TypeError, ValueError):
        raise ValueError(f"game {game_idx}: malformed aux arrays")
    if not (len(ad_lp) == len(ad_cards) == len(a_ends) == len(a_turns) == n):
        raise ValueError(
            f"game {game_idx}: aux length {[len(ad_lp), len(ad_cards), len(a_ends), len(a_turns)]}"
            f" != {n} replay decisions"
        )
    for row in ad_lp:
        if len(row) != 2:
            raise ValueError(f"game {game_idx}: aux d_lp row len {len(row)} != 2")
        _check_finite(row, f"game {game_idx} aux d_lp")
    for row in ad_cards:
        if len(row) != 14:
            raise ValueError(f"game {game_idx}: aux d_cards row len {len(row)} != 14")
        _check_finite(row, f"game {game_idx} aux d_cards")
    _check_finite(a_ends, f"game {game_idx} aux ends")
    _check_finite(a_turns, f"game {game_idx} aux turns_left")
    for p in a_ends:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"game {game_idx}: aux ends {p} outside [0, 1]")

    first_pos: dict[tuple[int, int], int] = {}
    for k, s in enumerate(snaps):
        first_pos.setdefault((int(s["turn"]), int(s["actor"])), k)
    starts = _aux_starts(snaps)
    start_turns = {s: [t for t, _ in starts[s]] for s in (0, 1)}
    start_pos = {s: [j for _, j in starts[s]] for s in (0, 1)}
    end_turn = max(int(s["turn"]) for s in snaps)

    import bisect

    for t in turn_nos:
        per_seat: dict = {}
        for s in (0, 1):
            if (t, s) not in first_pos:
                continue
            pos = first_pos[(t, s)]
            pred_lp = [round(ad_lp[pos][0]), round(ad_lp[pos][1])]
            pred_cards = [round(float(x), 2) for x in ad_cards[pos]]
            pred_ends = round(float(a_ends[pos]), 3)
            pred_turns = round(float(a_turns[pos]), 2)
            # next own-turn start strictly after duel turn t
            idx = bisect.bisect_right(start_turns[s], int(t))
            if idx < len(start_pos[s]):
                j = int(start_pos[s][idx])
                now_lp, fut_lp = snaps[pos]["lp"], snaps[j]["lp"]
                real_lp = [int(fut_lp[s]) - int(now_lp[s]), int(fut_lp[1 - s]) - int(now_lp[1 - s])]
                now_c, fut_c = snaps[pos]["counts"], snaps[j]["counts"]
                real_me = [int(b) - int(a) for a, b in zip(now_c[s], fut_c[s], strict=True)]
                real_op = [int(b) - int(a) for a, b in zip(now_c[1 - s], fut_c[1 - s], strict=True)]
                real_cards: list | None = [round(float(x), 2) for x in (real_me + real_op)]
                real_ends = 0
            else:
                real_lp = None
                real_cards = None
                real_ends = 1
            real_turns = int(end_turn) - int(t)
            per_seat[str(s)] = {
                "k": int(pos),
                "pred_lp": pred_lp,
                "pred_cards": pred_cards,
                "pred_ends": pred_ends,
                "pred_turns": pred_turns,
                "real_lp": real_lp,
                "real_cards": real_cards,
                "real_ends": int(real_ends),
                "real_turns": int(real_turns),
            }
        if per_seat:
            turns[t]["aux"] = per_seat


def _assert_no_nan(obj) -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NaN/non-finite float in annotation output")
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nan(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_nan(v)


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(anns: list[dict], input_dirs: list[str], commit: str) -> dict:
    """Aggregate per-game annotations into the run-level summary (pure)."""
    cals: dict[int, dict] = {b: {"n": 0, "won": 0, "p_sum": 0.0} for b in range(CAL_N_BINS)}
    brier: dict[str, list[float]] = defaultdict(list)
    ece_num = 0.0
    sign_acc: dict[str, dict] = {b: {"n": 0, "ok": 0} for b in SIGN_BUCKETS}
    adv_acc: dict[str, dict] = {b: {"n": 0, "ok": 0} for b in SIGN_BUCKETS}
    lp_acc: dict[str, dict] = {b: {"n": 0, "ok": 0} for b in SIGN_BUCKETS}
    # Mining: (kind, msg, bucket, seat_first, deck_s) -> [opportunities, hits].
    mine: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    tempo_vals: dict[tuple[int, int], list[float]] = defaultdict(list)
    swing_counts = {"1": 0, "2": 0, "3": 0, "4+": 0}
    swing_wins = 0
    swing_n = 0
    n_trunc = 0
    has_values = any("curve_full" in a for a in anns)
    # Value-net full curve: (bucket, bin) -> [n, won, p_sum]; v_comb sign; |disagree|.
    vcal: dict[tuple[str, int], list] = defaultdict(lambda: [0, 0, 0.0])
    vcomb_sign: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    disagree_abs: dict[str, list[float]] = defaultdict(list)
    # Aux heads: predicted vs realized at each seat's first decision per turn.
    aux_ends_pred: list[float] = []
    aux_ends_real: list[float] = []
    aux_lp_se = aux_lp_se0 = 0.0
    aux_lp_ae = 0.0
    aux_lp_n = 0
    aux_cards_se = aux_cards_se0 = 0.0
    aux_cards_ae = 0.0
    aux_cards_n = 0
    aux_turns_pred: list[float] = []
    aux_turns_real: list[float] = []
    aux_points = 0
    for ann in anns:
        if ann["truncated"]:
            n_trunc += 1
            continue
        winner = ann["winner"]
        for ev in ann["evals"]:
            p = ev["phat_searcher"]
            b = min(int(p * CAL_N_BINS), CAL_N_BINS - 1)
            cals[b]["n"] += 1
            cals[b]["won"] += 1 if winner == ev["seat"] else 0
            cals[b]["p_sum"] += p
            brier[brier_bucket(int(ev["turn"]))].append((p - (1 if winner == ev["seat"] else 0)) ** 2)
            if ev["v0"] > 0:
                sb = sign_bucket(int(ev["turn"]))
                sign_acc[sb]["n"] += 1
                sign_acc[sb]["ok"] += 1 if winner == 0 else 0
            elif ev["v0"] < 0:
                sb = sign_bucket(int(ev["turn"]))
                sign_acc[sb]["n"] += 1
                sign_acc[sb]["ok"] += 1 if winner == 1 else 0
            key = (
                ev["kind"],
                int(ev["msg"]),
                brier_bucket(int(ev["turn"])),
                bool(ann["s_first"]),
                str(ann["deck_s"]),
            )
            mine[key][0] += 1
            if ev["label"] in ("mistake", "blunder"):
                mine[key][1] += 1
        for t, ts in ann["turns"].items():
            sb = sign_bucket(int(t))
            if ts["card_adv"] != 0:
                adv_acc[sb]["n"] += 1
                adv_acc[sb]["ok"] += 1 if (ts["card_adv"] > 0) == (winner == 0) else 0
            lead = ts["lp"][0] - ts["lp"][1]
            if lead != 0:
                lp_acc[sb]["n"] += 1
                lp_acc[sb]["ok"] += 1 if (lead > 0) == (winner == 0) else 0
            if "curve_full" in ann:
                # Value calibration at turn starts, both seats: each seat's own
                # first-decision value predicts that seat's win.
                for p, key in ((0, "v_p0"), (1, "v_p1")):
                    v = ts.get(key)
                    if v is not None:
                        ph = (v + 1) / 2
                        b = min(int(ph * CAL_N_BINS), CAL_N_BINS - 1)
                        cell = vcal[(sb, b)]
                        cell[0] += 1
                        cell[1] += 1 if winner == p else 0
                        cell[2] += ph
                vc = ts.get("v_comb")
                if vc is not None and vc != 0:
                    vcomb_sign[sb][0] += 1
                    vcomb_sign[sb][1] += 1 if (vc > 0) == (winner == 0) else 0
                d = ts.get("disagree")
                if d is not None:
                    disagree_abs[sb].append(abs(d))
            aux_d = ts.get("aux")
            if isinstance(aux_d, dict):
                for seat_rec in aux_d.values():
                    if not isinstance(seat_rec, dict):
                        continue
                    try:
                        pe = float(seat_rec["pred_ends"])
                        re_ = float(seat_rec["real_ends"])
                        pt = float(seat_rec["pred_turns"])
                        rt = float(seat_rec["real_turns"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    aux_points += 1
                    aux_ends_pred.append(pe)
                    aux_ends_real.append(re_)
                    aux_turns_pred.append(pt)
                    aux_turns_real.append(rt)
                    rl = seat_rec.get("real_lp")
                    pl = seat_rec.get("pred_lp")
                    if isinstance(rl, (list, tuple)) and isinstance(pl, (list, tuple)) and len(rl) == 2 and len(pl) == 2:
                        for p_, r_ in zip(pl, rl, strict=True):
                            e = float(p_) - float(r_)
                            aux_lp_se += e * e
                            aux_lp_se0 += float(r_) * float(r_)
                            aux_lp_ae += abs(e)
                            aux_lp_n += 1
                    rc = seat_rec.get("real_cards")
                    pc = seat_rec.get("pred_cards")
                    if (
                        isinstance(rc, (list, tuple))
                        and isinstance(pc, (list, tuple))
                        and len(rc) == 14
                        and len(pc) == 14
                    ):
                        for p_, r_ in zip(pc, rc, strict=True):
                            e = float(p_) - float(r_)
                            aux_cards_se += e * e
                            aux_cards_se0 += float(r_) * float(r_)
                            aux_cards_ae += abs(e)
                            aux_cards_n += 1
        for t, tp in ann["tempo"].items():
            tempo_vals[(int(t), int(tp["turn_player"]))].append(tp["dp_tp"])
        if ann["swing_turn"] is not None:
            st8 = ann["swing_turn"]
            swing_counts[str(st8) if st8 in (1, 2, 3) else "4+"] += 1
            tp = ann["tempo"][st8]
            swing_winner = tp["turn_player"] if tp["dp_tp"] > 0 else 1 - tp["turn_player"]
            swing_n += 1
            swing_wins += 1 if swing_winner == winner else 0

    n_cal = sum(c["n"] for c in cals.values())
    cal_bins = []
    for b in range(CAL_N_BINS):
        c = cals[b]
        wr = c["won"] / c["n"] if c["n"] else None
        mean_p = c["p_sum"] / c["n"] if c["n"] else None
        if c["n"]:
            ece_num += c["n"] * abs(wr - mean_p)  # type: ignore[operator]
        cal_bins.append(
            {
                "lo": round(b * CAL_BIN_WIDTH, 1),
                "hi": round((b + 1) * CAL_BIN_WIDTH, 1),
                "n": c["n"],
                "won": wr,
                "mean_p": mean_p,
            }
        )
    summary = {
        "meta": {
            "input_dirs": list(input_dirs),
            "n_games": len(anns),
            "n_truncated": n_trunc,
            "n_searched_decisions": sum(a["summary"]["n_searched"] for a in anns),
            "commit": commit,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "calibration": {
            "bins": cal_bins,
            "brier_by_bucket": {k: (sum(v) / len(v) if v else None) for k, v in sorted(brier.items())},
            "ece": (ece_num / n_cal) if n_cal else None,
        },
        "sign_accuracy": {
            "search_eval": {
                k: {"n": v["n"], "acc": (v["ok"] / v["n"] if v["n"] else None)} for k, v in sign_acc.items()
            },
            "card_adv": {
                k: {"n": v["n"], "acc": (v["ok"] / v["n"] if v["n"] else None)} for k, v in adv_acc.items()
            },
            "lp_lead": {
                k: {"n": v["n"], "acc": (v["ok"] / v["n"] if v["n"] else None)} for k, v in lp_acc.items()
            },
        },
        "mining": [
            {
                "kind": k[0],
                "msg": k[1],
                "bucket": k[2],
                "seat_first": k[3],
                "deck_s": k[4],
                "opportunities": v[0],
                "hits": v[1],
                "rate": v[1] / v[0] if v[0] else 0.0,
            }
            for k, v in sorted(
                mine.items(), key=lambda kv: (-kv[1][1], -(kv[1][1] / kv[1][0] if kv[1][0] else 0))
            )
        ][:20],
        "tempo": {
            "by_turn_seat": [
                {
                    "turn": t,
                    "seat": s,
                    "n": len(v),
                    "mean_pp": st.mean(v) * 100,
                    "p10_pp": _quantile(v, 0.10) * 100,
                    "p90_pp": _quantile(v, 0.90) * 100,
                }
                for (t, s), v in sorted(tempo_vals.items())
            ],
            "swing_share": {k: (v / swing_n if swing_n else None) for k, v in swing_counts.items()},
            "swing_winner_winrate": (swing_wins / swing_n) if swing_n else None,
            "n_swing_games": swing_n,
        },
    }
    if has_values:
        vcal_sec = {}
        for sb in SIGN_BUCKETS:
            bins = []
            num = 0.0
            den = 0
            for b in range(CAL_N_BINS):
                n_, won_, ps_ = vcal[(sb, b)]
                wr = won_ / n_ if n_ else None
                mp = ps_ / n_ if n_ else None
                if n_:
                    num += n_ * abs(wr - mp)  # type: ignore[operator]
                    den += n_
                bins.append(
                    {
                        "lo": round(b * CAL_BIN_WIDTH, 1),
                        "hi": round((b + 1) * CAL_BIN_WIDTH, 1),
                        "n": n_,
                        "won": wr,
                        "mean_p": mp,
                    }
                )
            vcal_sec[sb] = {"bins": bins, "ece": (num / den) if den else None}
        summary["value_calibration"] = vcal_sec
        summary["v_comb_sign"] = {
            sb: {"n": vcomb_sign[sb][0],
                 "acc": (vcomb_sign[sb][1] / vcomb_sign[sb][0] if vcomb_sign[sb][0] else None)}
            for sb in SIGN_BUCKETS
        }
        summary["disagree"] = {
            sb: {"n": len(disagree_abs[sb]),
                 "mean_abs": (sum(disagree_abs[sb]) / len(disagree_abs[sb]) if disagree_abs[sb] else None)}
            for sb in SIGN_BUCKETS
        }
    if aux_points:
        summary["aux"] = _summarize_aux(
            aux_ends_pred, aux_ends_real, aux_lp_se, aux_lp_se0, aux_lp_ae, aux_lp_n,
            aux_cards_se, aux_cards_se0, aux_cards_ae, aux_cards_n,
            aux_turns_pred, aux_turns_real, aux_points,
        )
    _assert_no_nan(summary)
    return summary


def _r2(mse: float | None, mse_base: float | None) -> float | None:
    if mse is None or mse_base is None or not math.isfinite(mse) or not math.isfinite(mse_base):
        return None
    if mse_base <= 0:
        return None
    return 1.0 - mse / mse_base


def _summarize_aux(ends_pred, ends_real, lp_se, lp_se0, lp_ae, lp_n,
                   cards_se, cards_se0, cards_ae, cards_n,
                   turns_pred, turns_real, n_points) -> dict:
    """Per-head predicted-vs-realized accuracy over turn-seat points (pure)."""
    n = len(ends_pred)
    bins = [{"lo": round(b * CAL_BIN_WIDTH, 1), "hi": round((b + 1) * CAL_BIN_WIDTH, 1),
             "n": 0, "rate": None, "mean_p": None} for b in range(CAL_N_BINS)]
    brier = brier_base = ece = None
    base_rate = None
    if n:
        base_rate = sum(ends_real) / n
        se = sum((p - y) ** 2 for p, y in zip(ends_pred, ends_real, strict=True))
        se0 = sum((base_rate - y) ** 2 for y in ends_real)
        brier = se / n
        brier_base = se0 / n
        cells: dict[int, dict] = {b: {"n": 0, "y": 0.0, "p": 0.0} for b in range(CAL_N_BINS)}
        for p, y in zip(ends_pred, ends_real, strict=True):
            b = min(int(p * CAL_N_BINS), CAL_N_BINS - 1)
            cells[b]["n"] += 1
            cells[b]["y"] += y
            cells[b]["p"] += p
        num = 0.0
        for b in range(CAL_N_BINS):
            c = cells[b]
            if c["n"]:
                rate = c["y"] / c["n"]
                mp = c["p"] / c["n"]
                num += c["n"] * abs(rate - mp)
                bins[b] = {"lo": bins[b]["lo"], "hi": bins[b]["hi"],
                           "n": c["n"], "rate": rate, "mean_p": mp}
        ece = num / n
    lp_mse = (lp_se / lp_n) if lp_n else None
    lp_mse0 = (lp_se0 / lp_n) if lp_n else None
    cards_mse = (cards_se / cards_n) if cards_n else None
    cards_mse0 = (cards_se0 / cards_n) if cards_n else None
    tn = len(turns_pred)
    t_se = sum((p - y) ** 2 for p, y in zip(turns_pred, turns_real, strict=True)) if tn else None
    t_ae = sum(abs(p - y) for p, y in zip(turns_pred, turns_real, strict=True)) if tn else None
    t_mean = (sum(turns_real) / tn) if tn else None
    t_se0 = sum((t_mean - y) ** 2 for y in turns_real) if tn and t_mean is not None else None
    t_mse = (t_se / tn) if tn and t_se is not None else None
    t_mse0 = (t_se0 / tn) if tn and t_se0 is not None else None
    return {
        "n_points": n_points,
        "ends": {"n": n, "brier": brier, "brier_base": brier_base, "ece": ece,
                 "base_rate": base_rate, "bins": bins},
        "d_lp": {"n": lp_n, "mae": (lp_ae / lp_n) if lp_n else None, "mse": lp_mse,
                 "mse_base": lp_mse0, "r2": _r2(lp_mse, lp_mse0)},
        "d_cards": {"n": cards_n, "mae": (cards_ae / cards_n) if cards_n else None, "mse": cards_mse,
                    "mse_base": cards_mse0, "r2": _r2(cards_mse, cards_mse0)},
        "turns_left": {"n": tn, "mae": (t_ae / tn) if tn and t_ae is not None else None,
                       "mse": t_mse, "mse_base": t_mse0, "r2": _r2(t_mse, t_mse0),
                       "mean": t_mean},
    }


# ---------------------------------------------------------------- markdown ---


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def render_game_md(ann: dict) -> str:
    d0, d1 = (list(ann["decks"]) + ["?", "?"])[:2]
    winner = ann["winner"]
    s = ann["summary"]
    result = (
        f"- Result: **player {winner} wins** ({d0 if winner == 0 else d1})"
        if not ann["truncated"]
        else "- Result: truncated (no winner)"
    )
    lbl = s["labels"]
    stats = (
        f"- {ann['n_choices']} decisions, {s['n_searched']} searched, "
        f"{s['n_overridden']} overridden; labels: {lbl['inaccuracy']} inacc, "
        f"{lbl['mistake']} mistakes, {lbl['blunder']} blunders; "
        f"mean regret {s['mean_regret_pp']:.2f} win-pp (ACPL analogue)"
    )
    lines = [
        f"# Game {ann['game_idx']}: {d0} vs {d1}",
        "",
        f"- Searching seat: {ann['seat']} ({ann['deck_s']}); went first: {ann['s_first']}",
        result,
        stats,
        "- Evals are P(win | both sides continue like our policy P): relative to policy, not absolute.",
    ]
    if "curve_full" in ann:
        lines.append("- Full-curve evals (v_comb) from the value net at every decision of both seats.")
    has_aux = any(isinstance(ts.get("aux"), dict) for ts in ann["turns"].values())
    if has_aux:
        lines.append("- Aux: predicted vs actual by the seat's next own turn (ΔLP me/opp in LP, P(end), turns left).")
    lines += [
        "",
        "## Turns",
        "",
    ]
    if has_aux:
        lines += [
            "| turn | player | LP (own-opp) | hand | field | GY | banished | extra used | card_adv | eval p0 | tempo | aux ΔLP pred/real | P(end) pred/real | T left pred/real |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| turn | player | LP (own-opp) | hand | field | GY | banished | extra used | card_adv | eval p0 | tempo |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    first_turn = min(ann["turns"])
    extra_start = {p: ann["turns"][first_turn]["counts"][p][6] for p in (0, 1)}
    for t in sorted(ann["turns"]):
        ts = ann["turns"][t]
        curve_v0 = ann["curve"].get(t)
        tp_info = ann["tempo"].get(t)
        for p in (0, 1):
            c = ts["counts"][p]
            eu = extra_start[p] - c[6]
            adv = ts["card_adv"] if p == 0 else -ts["card_adv"]
            if p == 0 and ts.get("v_comb") is not None:
                eval_p0 = _pct((ts["v_comb"] + 1) / 2)
            else:
                eval_p0 = _pct((curve_v0 + 1) / 2) if curve_v0 is not None and p == 0 else ""
            tempo_pp = ""
            if tp_info and p == ts["turn_player"]:
                tempo_pp = f"{tp_info['dp_tp_pp']:+.1f}pp"
            base = (
                f"| {t} | P{p}{' (to move)' if p == ts['turn_player'] else ''} "
                f"| {ts['lp'][p]}-{ts['lp'][1 - p]} "
                f"| {c[1]} | {c[2] + c[3]} | {c[4]} | {c[5]} | {eu} | {adv:+d} | "
                f"{eval_p0} | {tempo_pp} |"
            )
            if has_aux:
                base += f" {_aux_lp_cell(ts, p)} | {_aux_ends_cell(ts, p)} | {_aux_turns_cell(ts, p)} |"
            lines.append(base)
    labelled = [ev for ev in ann["evals"] if ev["label"] != "none"]
    lines += ["", "## Labelled moves", ""]
    if not labelled:
        lines.append("None.")
    for ev in labelled:
        lines.append(
            f"### Turn {ev['turn']} decision {ev['k']} ({ev['label']}, seat {ev['seat']}, {ev['kind']})"
        )
        lines.append("")
        lines.append(
            f"Played **{ev['played_desc']}** over search-best **{ev['best_desc']}**: "
            f"regret {ev['regret_pp']:.1f} win-pp (gap/SE {ev['gap_se']:.1f})."
        )
        lines.append("")
        for c in ev["cands"]:
            mark = " <-- played" if c["opt"] == ev["played"] else ""
            mark += " <-- search-best" if c["opt"] == ev["best"] else ""
            lines.append(f"- {c['desc']} — {c['win_pct']:.1f}% ± {c['se_mean'] * 50:.1f}pp{mark}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_summary_md(summary: dict) -> str:
    m = summary["meta"]
    cal = summary["calibration"]
    sa = summary["sign_accuracy"]
    inputs = (
        f"- Inputs: {', '.join(m['input_dirs'])}; games: {m['n_games']}, "
        f"{m['n_truncated']} truncated, {m['n_searched_decisions']} searched"
    )
    lines = [
        "# Annotation summary",
        "",
        inputs,
        f"- Commit {m['commit']}; {m['timestamp']}",
        "- Evals are P(win | both sides continue like our policy P).",
        "",
        "## Calibration (searcher view, non-truncated games)",
        "",
        "| p-hat bin | n | searcher win rate | mean p-hat |",
        "|---|---|---|---|",
    ]
    for b in cal["bins"]:
        wr = "-" if b["won"] is None else f"{b['won']:.3f}"
        mp = "-" if b["mean_p"] is None else f"{b['mean_p']:.3f}"
        lines.append(f"| {b['lo']:.1f}-{b['hi']:.1f} | {b['n']} | {wr} | {mp} |")
    lines.append("")
    brier = ", ".join(
        f"{k}: {v:.3f}" if v is not None else f"{k}: -" for k, v in cal["brier_by_bucket"].items()
    )
    lines.append(f"Brier by turn bucket: {brier}.")
    lines.append(f"ECE: {cal['ece']:.4f}." if cal["ece"] is not None else "ECE: -.")
    lines += [
        "",
        "## Winner sign accuracy",
        "",
        "| bucket | search eval | card_adv | LP lead |",
        "|---|---|---|---|",
    ]
    for b in SIGN_BUCKETS:
        lines.append(
            f"| {b} | {_acc_cell(sa['search_eval'], b)} | {_acc_cell(sa['card_adv'], b)} "
            f"| {_acc_cell(sa['lp_lead'], b)} |"
        )
    lines += [
        "",
        "## Mining: significant >= mistake per searched opportunity (top 20)",
        "",
        "| kind | msg | turns | seat first | deck_s | hits/opps | rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary["mining"]:
        lines.append(
            f"| {r['kind']} | {_msg_name(r['msg'])} | {r['bucket']} | {r['seat_first']} | "
            f"{r['deck_s']} | {r['hits']}/{r['opportunities']} | {r['rate']:.4f} |"
        )
    lines += [
        "",
        "## Tempo (win-pp for the turn player)",
        "",
        "| turn | seat | n | mean | p10 | p90 |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["tempo"]["by_turn_seat"]:
        if r["turn"] > 12:  # long tails are a handful of games each; summary.json keeps them all
            continue
        lines.append(
            f"| {r['turn']} | {r['seat']} | {r['n']} | {r['mean_pp']:.1f} | "
            f"{r['p10_pp']:.1f} | {r['p90_pp']:.1f} |"
        )
    sw = summary["tempo"]
    share = ", ".join(f"T{k}: {v:.2f}" if v is not None else f"T{k}: -" for k, v in sw["swing_share"].items())
    lines += ["", f"Swing-turn share: {share}."]
    if sw["swing_winner_winrate"] is not None:
        lines.append(
            f"Swing winner goes on to win: {sw['swing_winner_winrate']:.3f} (n={sw['n_swing_games']})."
        )
    else:
        lines.append("Swing winner goes on to win: -.")
    if "value_calibration" in summary:
        lines += [
            "",
            "## Value calibration at turn starts (both seats, non-truncated games)",
            "",
            "| turn bucket | n | ECE |",
            "|---|---|---|",
        ]
        for b in SIGN_BUCKETS:
            n_b = sum(x["n"] for x in summary["value_calibration"][b]["bins"])
            ece = summary["value_calibration"][b]["ece"]
            lines.append(f"| {b} | {n_b} | {ece:.4f} |" if ece is not None else f"| {b} | {n_b} | - |")
        lines += [
            "",
            "## Winner sign accuracy of v_comb",
            "",
            "| bucket | v_comb |",
            "|---|---|",
        ]
        for b in SIGN_BUCKETS:
            lines.append(f"| {b} | {_acc_cell(summary['v_comb_sign'], b)} |")
        lines += [
            "",
            "## Disagreement |v_p0 + v_p1|",
            "",
            "| bucket | n | mean |",
            "|---|---|---|",
        ]
        for b in SIGN_BUCKETS:
            d = summary["disagree"][b]
            lines.append(
                f"| {b} | {d['n']} | {d['mean_abs']:.3f} |" if d["mean_abs"] is not None else f"| {b} | {d['n']} | - |"
            )
    if "aux" in summary:
        ax = summary["aux"]
        lines += [
            "",
            "## Aux heads: predicted vs realized (turn-seat points, non-truncated games)",
            "",
            f"- Points: {ax['n_points']}.",
            (
                f"- P(ends): Brier {ax['ends']['brier']:.4f} vs base-rate {_fmt1(ax['ends']['brier_base'])} "
                f"(base rate {_fmt1(ax['ends']['base_rate'])}, ECE {_fmt1(ax['ends']['ece'])})."
            ),
            f"- d_lp (LP units): MAE {_fmt1(ax['d_lp']['mae'])} R² vs 0 {_fmt1(ax['d_lp']['r2'])} (n={ax['d_lp']['n']}).",
            f"- d_cards: MAE {_fmt1(ax['d_cards']['mae'])} R² vs 0 {_fmt1(ax['d_cards']['r2'])} (n={ax['d_cards']['n']}).",
            (
                f"- turns_left: MAE {_fmt1(ax['turns_left']['mae'])} R² vs mean {_fmt1(ax['turns_left']['r2'])} "
                f"(mean {_fmt1(ax['turns_left']['mean'])}, n={ax['turns_left']['n']})."
            ),
            "",
            "| p(ends) bin | n | end rate | mean p |",
            "|---|---|---|---|",
        ]
        for b in ax["ends"]["bins"]:
            if not b["n"]:
                continue
            lines.append(f"| {b['lo']:.1f}-{b['hi']:.1f} | {b['n']} | {b['rate']:.3f} | {b['mean_p']:.3f} |")
    return "\n".join(lines) + "\n"


def _fmt1(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def _aux_seat(ts: dict, p: int) -> dict | None:
    aux_d = ts.get("aux")
    if not isinstance(aux_d, dict):
        return None
    for k in (str(p), p):
        if k in aux_d and isinstance(aux_d[k], dict):
            return aux_d[k]
    return None


def _aux_lp_cell(ts: dict, p: int) -> str:
    rec = _aux_seat(ts, p)
    if rec is None:
        return ""
    pl = rec.get("pred_lp") or [0, 0]
    rl = rec.get("real_lp")
    if rl is None:
        return f"{pl[0]:+d}/{pl[1]:+d} (end)"
    return f"{pl[0]:+d}/{pl[1]:+d} ({rl[0]:+d}/{rl[1]:+d})"


def _aux_ends_cell(ts: dict, p: int) -> str:
    rec = _aux_seat(ts, p)
    if rec is None:
        return ""
    return f"{float(rec['pred_ends']) * 100:.1f}%/{int(rec['real_ends'])}"


def _aux_turns_cell(ts: dict, p: int) -> str:
    rec = _aux_seat(ts, p)
    if rec is None:
        return ""
    return f"{float(rec['pred_turns']):.1f}/{int(rec['real_turns'])}"


def _msg_name(t: int) -> str:
    from ygo_zero.engine import constants as C

    names = {v: k.removeprefix("MSG_SELECT_") for k, v in vars(C).items() if k.startswith("MSG_SELECT_")}
    return names.get(int(t), str(t))


def _acc_cell(d: dict, b: str) -> str:
    v = d[b]
    return "-" if v["acc"] is None else f"{v['acc']:.2f} (n={v['n']})"
