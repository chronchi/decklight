"""Build a static annotation viewer dir with history across runs.

Reads annotation output dirs written by `scripts/annotate_games.py`
(`summary.json` + `games/*.json`; games may carry `curve_full` and per-turn
`v_p0`/`v_p1`/`v_comb`/`disagree`/`tempo`/`tempo_search` when annotated with
`--values`). Optional per-run engine events from `scripts/game_events.py`
(`events.jsonl`, one line per game keyed by `game_idx`, `seed`, `decks`).

    taskset -c 0-3 uv run python scripts/build_viewer.py \\
        --run hppo-p2-search=data/cache/ann-full,events=data/cache/events-smoke.jsonl \\
        --out data/cache/viewer --max-games-per-run 300

Running again later with a new `--run` keeps the old ones: `DIR/data/runs.json`
is appended/updated per run name, so the Runs page tracks metrics over time.

Serve the result with `python -m http.server -d DIR` (file:// works for the
page shell, but the JSON loads need http).

Invariants (exit non-zero): a run dir without `summary.json`; 0 games found.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER_HTML = ROOT / "scripts" / "viewer" / "index.html"

LOC_LABELS = ("deck", "hand", "mzone", "szone", "grave", "removed", "extra")


def r3(x):
    """Round floats to 3 decimals for compact viewer JSON (pass through the rest)."""
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    return x


def _parse_run_spec(spec: str) -> tuple[str, Path, Path | None]:
    """Parse NAME=ANN_DIR[,events=FILE]."""
    parts = spec.split(",")
    if "=" not in parts[0]:
        raise ValueError(f"bad --run spec {spec!r}: expected NAME=ANN_DIR[,events=FILE]")
    name, ann = parts[0].split("=", 1)
    name = name.strip()
    if not name or not ann:
        raise ValueError(f"bad --run spec {spec!r}: empty name or dir")
    events = None
    for p in parts[1:]:
        if p.startswith("events="):
            events = Path(p[len("events="):])
        else:
            raise ValueError(f"bad --run spec {spec!r}: unknown option {p!r}")
    return name, Path(ann), events


def _load_events(path: Path | None) -> dict:
    """Map game_idx -> event record (KeyError-free; empty when no file)."""
    out: dict = {}
    if path is None:
        return out
    if not path.is_file():
        print(f"ALERT build_viewer: no such events file: {path}", file=sys.stderr)
        raise SystemExit(2)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "game_idx" in rec:
            out[rec["game_idx"]] = rec
    return out


def _counts_of(counts: dict, p: int) -> list[int]:
    c = counts[str(p)] if str(p) in counts else counts[p]
    return [int(v) for v in c]


def compact_game(ann: dict, ev_rec: dict | None) -> dict:
    """Reduce one full annotation to the fields the viewer shows."""
    turns = ann.get("turns", {})
    turn_nos = sorted(int(t) for t in turns)
    first_turn = turn_nos[0] if turn_nos else None
    extra_start = {}
    if first_turn is not None:
        for p in (0, 1):
            extra_start[p] = _counts_of(turns[str(first_turn)]["counts"], p)[6]

    turn_rows = []
    ev_turns = {}
    if ev_rec is not None:
        for t in ev_rec.get("turns", []):
            ev_turns[int(t["turn"])] = t
    for t in turn_nos:
        ts = turns[str(t)]
        lp = [int(v) for v in ts["lp"]]
        row: dict = {
            "turn": t,
            "turn_player": int(ts["turn_player"]),
            "lp": lp,
            "card_adv": int(ts["card_adv"]),
        }
        for p in (0, 1):
            c = _counts_of(ts["counts"], p)
            row[f"p{p}"] = {
                "hand": c[1],
                "field": c[2] + c[3],
                "gy": c[4],
                "ban": c[5],
                "extra_used": extra_start.get(p, c[6]) - c[6],
            }
        for k in ("v_p0", "v_p1", "v_comb", "disagree"):
            if ts.get(k) is not None:
                row[k] = ts[k]
        tp = (ann.get("tempo") or {}).get(str(t))
        if tp is not None:
            row["tempo_pp"] = tp.get("dp_tp_pp")
        tps = (ann.get("tempo_search") or {}).get(str(t))
        if tps is not None:
            row["tempo_search_pp"] = tps.get("dp_tp_pp")
        et = ev_turns.get(t)
        if et is not None:
            row["ev"] = [
                {k: int(v) for k, v in et["players"][p].items()} for p in (0, 1)
            ]
        turn_rows.append(row)

    evals = []
    for ev in ann.get("evals", []):
        evals.append(
            {
                "k": int(ev["k"]),
                "turn": int(ev["turn"]),
                "seat": int(ev["seat"]),
                "kind": str(ev.get("kind", "?")),
                "label": str(ev.get("label", "none")),
                "regret_pp": ev.get("regret_pp"),
                "gap_se": ev.get("gap_se"),
                "played": int(ev["played"]),
                "best": int(ev["best"]),
                "overridden": bool(ev.get("overridden")),
                "p0_win_pct": ev.get("p0"),
                "played_desc": ev.get("played_desc"),
                "best_desc": ev.get("best_desc"),
                "cands": [
                    {
                        "opt": int(c["opt"]),
                        "desc": c.get("desc", f"option {c['opt']}"),
                        "win_pct": c.get("win_pct"),
                        "se_pp": (c["se_mean"] * 50) if c.get("se_mean") is not None else None,
                    }
                    for c in ev.get("cands", [])
                ],
            }
        )

    curve_full = None
    if ann.get("curve_full") is not None:
        curve_full = [
            {"k": int(p["k"]), "turn": int(p["turn"]), "v0": float(p["v0"])}
            for p in ann["curve_full"]
        ]
    curve = {str(t): v for t, v in (ann.get("curve") or {}).items()}

    turn_starts: dict[str, int] = {}
    if curve_full is not None:
        for p in curve_full:
            turn_starts.setdefault(str(p["turn"]), p["k"])
    else:
        for ev in evals:
            turn_starts.setdefault(str(ev["turn"]), ev["k"])

    last_counts = None
    if turn_nos:
        last_counts = turns[str(turn_nos[-1])]["counts"]
    final_adv = None
    if last_counts is not None:
        c0 = _counts_of(last_counts, 0)
        c1 = _counts_of(last_counts, 1)
        final_adv = (c0[1] + c0[2] + c0[3]) - (c1[1] + c1[2] + c1[3])

    return r3(
        {
            "game_idx": ann.get("game_idx"),
            "decks": list(ann.get("decks", [])),
            "seed": ann.get("seed"),
            "seat": ann.get("seat"),
            "s_first": bool(ann.get("s_first")),
            "deck_s": ann.get("deck_s"),
            "winner": ann.get("winner"),
            "truncated": bool(ann.get("truncated")),
            "n_choices": ann.get("n_choices"),
            "n_turns": len(turn_nos),
            "turns": turn_rows,
            "evals": evals,
            "curve": curve,
            "curve_full": curve_full,
            "turn_starts": turn_starts,
            "tempo_pp": {str(t): v.get("dp_tp_pp") for t, v in (ann.get("tempo") or {}).items()},
            "tempo_search_pp": {
                str(t): v.get("dp_tp_pp") for t, v in (ann.get("tempo_search") or {}).items()
            },
            "swing_turn": ann.get("swing_turn"),
            "swing_turn_search": ann.get("swing_turn_search"),
            "labels": (ann.get("summary") or {}).get("labels", {}),
            "mean_regret_pp": (ann.get("summary") or {}).get("mean_regret_pp"),
            "n_searched": (ann.get("summary") or {}).get("n_searched"),
            "n_overridden": (ann.get("summary") or {}).get("n_overridden"),
            "final_card_adv": final_adv,
        }
    )


def run_headline(summary: dict, compacts: list[dict], events: dict) -> dict:
    """Headline metrics for runs.json from summary.json + per-game compacts + events."""
    n = len(compacts)
    labels = {"inaccuracy": 0, "mistake": 0, "blunder": 0}
    reg_sum = 0.0
    for g in compacts:
        for k in labels:
            labels[k] += int((g.get("labels") or {}).get(k, 0) or 0)
        reg_sum += float(g.get("mean_regret_pp") or 0.0)
    n_int = n_neg = 0
    n_ev_games = 0
    for g in compacts:
        rec = events.get(g["game_idx"])
        if rec is None:
            continue
        n_ev_games += 1
        for t in rec.get("turns", []):
            for p in t["players"]:
                n_int += int(p.get("activations_opp_turn", 0))
                n_neg += int(p.get("negated_by_p_suffered", 0))
    cal = summary.get("calibration", {})
    tempo = summary.get("tempo", {})
    headline = {
        "name": "",  # filled by caller
        "date": summary.get("meta", {}).get("timestamp"),
        "commit": summary.get("meta", {}).get("commit"),
        "n_games": n,
        "ece": cal.get("ece"),
        "brier": cal.get("brier_by_bucket"),
        "sign_acc": {
            sec: {b: v.get("acc") for b, v in summary.get("sign_accuracy", {}).get(sec, {}).items()}
            for sec in ("search_eval", "card_adv", "lp_lead")
        },
        "swing_share": tempo.get("swing_share"),
        "swing_winner_winrate": tempo.get("swing_winner_winrate"),
        "mean_regret_pp": (reg_sum / n) if n else None,
        "labels_per_game": {k: v / n for k, v in labels.items()} if n else dict(labels),
        "interruptions_per_game": (n_int / n_ev_games) if n_ev_games else None,
        "negations_per_game": (n_neg / n_ev_games) if n_ev_games else None,
        "n_event_games": n_ev_games,
    }
    return r3(headline)


def build_run(name: str, ann_dir: Path, events_path: Path | None, out: Path, max_games: int) -> dict:
    summary_path = ann_dir / "summary.json"
    if not summary_path.is_file():
        print(f"ALERT build_viewer: run {name}: missing {summary_path}", file=sys.stderr)
        raise SystemExit(2)
    summary = json.loads(summary_path.read_text())
    games_dir = ann_dir / "games"
    files = sorted(games_dir.glob("game-*.json")) if games_dir.is_dir() else []
    if max_games and len(files) > max_games:
        files = files[:max_games]
    if not files:
        print(f"ALERT build_viewer: run {name}: 0 games in {ann_dir}", file=sys.stderr)
        raise SystemExit(2)
    events = _load_events(events_path)

    run_dir = out / "data" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    compacts = []
    index_rows = []
    for f in files:
        ann = json.loads(f.read_text())
        rec = events.get(ann.get("game_idx"))
        if rec is not None:
            # Keyed by game_idx; seed/decks guard against a stale events file.
            if "seed" in rec and rec["seed"] != ann.get("seed"):
                print(
                    f"ALERT build_viewer: run {name}: events seed mismatch "
                    f"game {ann.get('game_idx')}",
                    file=sys.stderr,
                )
                rec = None
            elif "decks" in rec and list(rec["decks"]) != list(ann.get("decks", [])):
                print(
                    f"ALERT build_viewer: run {name}: events decks mismatch "
                    f"game {ann.get('game_idx')}",
                    file=sys.stderr,
                )
                rec = None
        g = compact_game(ann, rec)
        compacts.append(g)
        (run_dir / f"{g['game_idx']}.json").write_text(json.dumps(g))
        n_bad = sum(int((g.get("labels") or {}).get(k, 0) or 0) for k in ("mistake", "blunder"))
        index_rows.append(
            r3(
                {
                    "idx": g["game_idx"],
                    "decks": g["decks"],
                    "seats": g["seat"],
                    "winner": g["winner"],
                    "turns": g["n_turns"],
                    "n_labels": n_bad,
                    "mean_regret": g["mean_regret_pp"],
                    "swing_turn": g["swing_turn"],
                    "final_card_adv": g["final_card_adv"],
                }
            )
        )
    index_rows.sort(key=lambda r: (r["idx"] is None, r["idx"]))
    (run_dir / "index.json").write_text(json.dumps(index_rows))
    (run_dir / "summary.json").write_text(json.dumps(summary))

    head = run_headline(summary, compacts, events)
    head["name"] = name
    return head


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=ANN_DIR[,events=FILE]")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-games-per-run", type=int, default=0)
    a = ap.parse_args(argv)

    if not VIEWER_HTML.is_file():
        print(f"ALERT build_viewer: missing viewer page {VIEWER_HTML}", file=sys.stderr)
        return 2
    specs = [_parse_run_spec(s) for s in a.run]
    out: Path = a.out
    (out / "data").mkdir(parents=True, exist_ok=True)
    runs_path = out / "data" / "runs.json"
    runs: list[dict] = []
    if runs_path.is_file():
        try:
            runs = json.loads(runs_path.read_text())
        except json.JSONDecodeError:
            print(f"ALERT build_viewer: corrupt {runs_path}, starting fresh", file=sys.stderr)
            runs = []
    by_name = {r["name"]: r for r in runs if "name" in r}
    for name, ann_dir, events_path in specs:
        head = build_run(name, ann_dir, events_path, out, a.max_games_per_run)
        by_name[name] = head
    runs = [by_name[k] for k in sorted(by_name)]
    runs_path.write_text(json.dumps(runs, indent=1))
    shutil.copyfile(VIEWER_HTML, out / "index.html")
    total_games = sum(r.get("n_games", 0) for r in runs)
    print(f"runs={len(runs)} games={total_games} wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
