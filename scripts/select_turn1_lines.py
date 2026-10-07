"""Turn-1 self-imitation records (self-imitation step 2): best sampled line per hand as BC data.

    python scripts/select_turn1_lines.py --lines <dir or lines.jsonl>... --out data/demos/selfimit/<tag> \
        [--by v|field] [--min-dv 0.1] [--min-dfield 0] [--top 1]
    python scripts/select_turn1_lines.py --lines <scored.jsonl>... --out data/demos/selfimit/<tag> \
        --by rollout [--min-dr 0.0]

Reads turn-1 line rows (scripts/turn1_lines.py: one greedy + K sampled lines per (deck, hand) with
field and privileged-critic v_p0). Per (deck, source, hand) the best sampled line that beats the greedy
line becomes a BC training record in the turn-1 search teacher format (scripts/gen_search.py), so the
existing pipeline (build_bc_shards -> bc_data -> demos.dataset.iter_samples) trains on it unchanged.
With --by rollout the rows come from scripts/score_lines.py and the winner is the best rollout
score (played out through the opponent's turn 2, so held hand traps count); rows without a
rollout score are ignored.

Only `train_ok` rows are used (holdout/gauntlet never become training data). Actors/labels/turns are
recovered by replaying the winning choice list in a CppEnv; a line whose replay fails or reaches a
different end field is dropped (fail loud when too many drop).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.demos.game import _opts  # noqa: E402
from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import Engine  # noqa: E402
from ygo_zero.demos.dataset import record_config  # noqa: E402
from ygo_zero.eval.turn1 import FIELD_LOCS, config_from_dict, my_field  # noqa: E402


def _turn(obs) -> int:
    return int(obs["global"][2])


def _p0_field_of(obs, player: int) -> int:
    """Player 0's field cards from an obs viewed by `player` (run_turn1 end-field measure)."""
    c, m = obs["cards"], obs["card_mask"]
    ctrl = 0 if player == 0 else 1
    return int(sum(1 for j in np.flatnonzero(m) if int(c[j, 1]) in FIELD_LOCS and int(c[j, 3]) == ctrl))


def replay_line(engine: Engine, cfg_dict: dict, choices: list[int],
                auto_single: bool) -> tuple[list[int], list[int], int]:
    """Replay a choice list through turn 1 (stops at turn 2 / game over, like replay_turn1).

    Returns (actors, turns, field) with the end field measured exactly the way turn1_lines does.
    Raises on an illegal choice: the line must replay exactly.
    """
    env = CppEnv(engine, record_config({"config": cfg_dict}), auto_single)  # the BC replay (iter_samples) uses record_config
    obs = env.reset()
    actors, turns, last = [], [], 0
    for c in choices:
        if obs is None or _turn(obs) > 1:
            break
        actors.append(int(env.player))
        turns.append(_turn(obs))
        if env.player == 0:
            last = my_field(obs)
        if not 0 <= int(c) < len(_opts(obs)):
            raise ValueError(f"choice {c} illegal with {len(_opts(obs))} options (player {env.player})")
        obs = env.step(int(c))
    fld = last
    if obs is not None:
        if _turn(obs) > 1:
            fld = _p0_field_of(obs, env.player)
        elif env.player == 1:
            c, m = obs["cards"], obs["card_mask"]
            fld = int(sum(1 for j in np.flatnonzero(m) if int(c[j, 1]) in FIELD_LOCS and int(c[j, 3]) == 1))
        else:
            fld = my_field(obs)
    return actors, turns, int(fld)


def read_rows(specs: list[str]) -> list[dict]:
    rows: list[dict] = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            p = p / "lines.jsonl"
        for raw in p.read_text().splitlines():
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def rank_key(by: str, row: dict) -> tuple:
    if by == "rollout":
        return (-row["rollout"], row["sample"])
    return (-row["v_p0"], -row["field"]) if by == "v" else (-row["field"], -row["v_p0"])


def _has_rollout(row: dict) -> bool:
    v = row.get("rollout")
    return isinstance(v, (int, float)) and v == v  # not None, not NaN


def qualifies(by: str, best: dict, greedy: dict, min_dv: float, min_dfield: float,
              min_dr: float = 0.0) -> bool:
    if by == "rollout":
        return bool(best["rollout"] > greedy["rollout"] + min_dr)
    dv = best["v_p0"] - greedy["v_p0"]
    df = best["field"] - greedy["field"]
    if by == "field":
        return bool(best["field"] > greedy["field"] and best["v_p0"] >= greedy["v_p0"])
    return bool(dv >= min_dv and df >= min_dfield)


def select_group(rows: list[dict], by: str, min_dv: float, min_dfield: float, top: int,
                 min_dr: float = 0.0) -> list[dict]:
    """Best qualifying samples of one (deck, source, hand) group, rank order, distinct choices."""
    if by == "rollout":
        greedy = [r for r in rows if r["kind"] == "greedy" and not r["stuck"] and _has_rollout(r)]
        cands = [r for r in rows if r["kind"] == "sample" and not r["stuck"] and _has_rollout(r)]
    else:
        greedy = [r for r in rows if r["kind"] == "greedy" and not r["stuck"]]
        cands = [r for r in rows if r["kind"] == "sample" and not r["stuck"]]
    samples = sorted(cands, key=lambda r: (rank_key(by, r), r["sample"]) if by != "rollout"
                     else rank_key(by, r))
    if not greedy or not samples:
        return []
    g = greedy[0]
    kept, seen = [], set()
    for s in samples:
        key = tuple(s["choices"])
        if key in seen or not qualifies(by, s, g, min_dv, min_dfield, min_dr):
            continue
        seen.add(key)
        kept.append((s, g))
        if len(kept) >= top:
            break
    return kept


def make_record(deck: str, best: dict, greedy: dict, by: str,
                actors: list[int], turns: list[int]) -> dict:
    game_id = f"si-{deck}-{best['source']}-h{best['hand']}-s{best['sample']}"
    rec = {
        "game_id": game_id,
        "line_id": f"{deck}/{game_id}",
        # lines.jsonl rows carry card lists, not the opponent deck's name; the opp label is
        # never trained (SELFIMIT_LABEL_WEIGHT), so only decks[0] matters.
        "decks": [deck, best.get("opp", deck)],
        "config": best["config"],
        "auto_single": bool(best.get("auto_single", True)),
        "choices": [int(c) for c in best["choices"]],
        "labels": ["self" if a == 0 else "opp" for a in actors],
        "actors": [int(a) for a in actors],
        "turns": [int(t) for t in turns],
        "greedy_field": int(greedy["field"]),
        "greedy_v": float(greedy["v_p0"]),
        "end_field": int(best["field"]),
        "end_v": float(best["v_p0"]),
        "source": f"turn1 self-imitation {by}",
    }
    if by == "rollout":  # the selection metric, for analysis (never trained)
        rec["rollout"] = float(best["rollout"])
        rec["greedy_rollout"] = float(greedy["rollout"])
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lines", nargs="+", required=True, help="lines.jsonl files or dirs containing one")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--by", default="v", choices=["v", "field", "rollout"])
    ap.add_argument("--min-dv", type=float, default=0.1)
    ap.add_argument("--min-dfield", type=float, default=0)
    ap.add_argument("--min-dr", type=float, default=0.0,
                    help="--by rollout: keep the best sample only if rollout(best) > rollout(greedy) + this")
    ap.add_argument("--top", type=int, default=1)
    args = ap.parse_args(argv)

    rows = read_rows(args.lines)
    n_read = len(rows)
    ok = [r for r in rows if r.get("train_ok")]
    n_notrain = n_read - len(ok)
    print(f"select_turn1_lines: {n_read} rows read, {len(ok)} train_ok, {n_notrain} skipped (train_ok false)",
          flush=True)
    if not ok:
        print("select_turn1_lines: 0 train_ok rows: nothing selectable", file=sys.stderr, flush=True)
        return 2

    groups: dict[tuple, list[dict]] = {}
    for r in ok:
        if r.get("kind") == "expert":  # human/LLM lines already in the BC data
            continue
        groups.setdefault((r["deck"], r["source"], r["hand"]), []).append(r)

    engine = Engine()
    per_deck: dict[str, list[dict]] = {}
    dvs, dfs = [], []
    dropped, attempted = 0, 0
    for key in sorted(groups):
        deck = key[0]
        for best, greedy in select_group(groups[key], args.by, args.min_dv, args.min_dfield,
                                         args.top, args.min_dr):
            attempted += 1
            try:
                actors, turns, fld = replay_line(
                    engine, best["config"], [int(c) for c in best["choices"]],
                    bool(best.get("auto_single", True)))
            except Exception as e:  # noqa: BLE001 - one bad line must not kill selection
                dropped += 1
                print(f"select_turn1_lines: drop {deck}/{best['source']}/h{best['hand']}/s{best['sample']}: "
                      f"replay failed: {e!r}"[:200], flush=True)
                continue
            if fld != int(best["field"]):
                dropped += 1
                print(f"select_turn1_lines: drop {deck}/{best['source']}/h{best['hand']}/s{best['sample']}: "
                      f"field mismatch replay={fld} stored={best['field']}", flush=True)
                continue
            per_deck.setdefault(deck, []).append(make_record(deck, best, greedy, args.by, actors, turns))
            dvs.append(float(best["v_p0"]) - float(greedy["v_p0"]))
            dfs.append(int(best["field"]) - int(greedy["field"]))

    n_kept = sum(len(v) for v in per_deck.values())
    if attempted and dropped / attempted > 0.05:
        print(f"select_turn1_lines: {dropped}/{attempted} selected lines failed replay (>5%)",
              file=sys.stderr, flush=True)
        return 3
    if n_kept == 0:
        print("select_turn1_lines: 0 lines kept (no sample beats greedy)", file=sys.stderr, flush=True)
        return 3

    args.out.mkdir(parents=True, exist_ok=True)
    for deck in sorted(per_deck):
        with open(args.out / f"p0-{deck}.jsonl", "w") as f:
            for rec in per_deck[deck]:
                f.write(json.dumps(rec) + "\n")
    summary = {
        "by": args.by, "min_dv": args.min_dv, "min_dfield": args.min_dfield, "min_dr": args.min_dr,
        "top": args.top,
        "rows_read": n_read, "train_ok_rows": len(ok), "skipped_train_ok_false": n_notrain,
        "groups": len(groups), "attempted": attempted, "dropped_replay": dropped,
        "kept_total": n_kept,
        "kept_per_deck": {d: len(v) for d, v in sorted(per_deck.items())},
        "mean_dv": float(sum(dvs) / len(dvs)) if dvs else None,
        "mean_dfield": float(sum(dfs) / len(dfs)) if dfs else None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"select_turn1_lines: kept {n_kept} lines "
          f"({', '.join(f'{d}={n}' for d, n in sorted(summary['kept_per_deck'].items()))}), "
          f"mean dv {summary['mean_dv']:+.3f} dfield {summary['mean_dfield']:+.2f}, "
          f"dropped {dropped}/{attempted} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
