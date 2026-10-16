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
    rec: dict, snaps: list[dict], end_snap: dict, descs: dict[int, list[tuple[int, str]]]
) -> dict:
    """Build the per-game annotation from a record + replay snapshots (pure)."""
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
    _assert_no_nan(ann)
    return ann


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
    _assert_no_nan(summary)
    return summary


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
        "",
        "## Turns",
        "",
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
            eval_p0 = _pct((curve_v0 + 1) / 2) if curve_v0 is not None and p == 0 else ""
            tempo_pp = ""
            if tp_info and p == ts["turn_player"]:
                tempo_pp = f"{tp_info['dp_tp_pp']:+.1f}pp"
            lines.append(
                f"| {t} | P{p}{' (to move)' if p == ts['turn_player'] else ''} "
                f"| {ts['lp'][p]}-{ts['lp'][1 - p]} "
                f"| {c[1]} | {c[2] + c[3]} | {c[4]} | {c[5]} | {eu} | {adv:+d} | "
                f"{eval_p0} | {tempo_pp} |"
            )
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
    return "\n".join(lines) + "\n"


def _msg_name(t: int) -> str:
    from ygo_zero.engine import constants as C

    names = {v: k.removeprefix("MSG_SELECT_") for k, v in vars(C).items() if k.startswith("MSG_SELECT_")}
    return names.get(int(t), str(t))


def _acc_cell(d: dict, b: str) -> str:
    v = d[b]
    return "-" if v["acc"] is None else f"{v['acc']:.2f} (n={v['n']})"
