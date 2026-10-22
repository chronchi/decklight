"""Annotate stored `search_match` games chess-engine-style (engine plan N0, milestone 1).

Reads `games-*.jsonl` records, replays each through the C++ env (no torch, no model),
and writes per-game JSON annotations (+ optional Markdown) and a run summary.

    taskset -c 0-3 uv run python scripts/annotate_games.py --games runs/<job>/hppo-p2 \\
        --out data/cache/ann --md 3 --workers 4

Invariants (exit non-zero): 0 games read; 0 search entries in the whole input;
a replay that diverges from its record (exit 2); NaN anywhere.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decklight.analysis import annotate as A

_ENGINE = None
_NAMES = None


def _worker_init() -> None:
    global _ENGINE, _NAMES
    from decklight.demos.names import CardNames
    from decklight.engine.duel import Engine

    _ENGINE = Engine()
    _NAMES = CardNames().names


def _annotate_one(payload: tuple[dict, dict | None]) -> dict:
    import render_showcase as R

    from decklight.decks import load_ydk

    rec, values = payload

    def deck(name: str):
        return load_ydk(R.deck_path(name))

    snaps, end_snap, descs = A.replay(rec, _ENGINE, deck, R.describe, _NAMES)
    if values is None:
        ann = A.annotate_game(rec, snaps, end_snap, descs)
    else:
        ann = A.annotate_game(rec, snaps, end_snap, descs, values)
    return {"ann": ann, "md": A.render_game_md(ann)}


def _iter_records(dirs: list[Path], max_games: int):
    n = 0
    for d in dirs:
        files = sorted(d.glob("games-*.jsonl"))
        if not files:
            print(f"ALERT annotate: no games-*.jsonl in {d}", file=sys.stderr)
        for f in files:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                yield json.loads(line)
                n += 1
                if max_games and n >= max_games:
                    return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--md", type=int, default=0, help="write Markdown for the first N games")
    ap.add_argument("--md-games", default="", help="comma-separated game_idx for Markdown")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--values", type=Path, default=None,
                    help="optional score_decisions.py values.jsonl for full-curve evals")
    a = ap.parse_args()

    for d in a.games:
        if not d.is_dir():
            print(f"no such dir: {d}", file=sys.stderr)
            return 2
    recs = list(_iter_records(a.games, a.max_games))
    if not recs:
        print("ALERT annotate: 0 games read", file=sys.stderr)
        return 2
    if sum(len(r.get("search", [])) for r in recs) == 0:
        print("ALERT annotate: 0 search entries in the whole input", file=sys.stderr)
        return 2
    by_idx: dict = {}
    if a.values is not None:
        if not a.values.is_file():
            print(f"ALERT annotate: no such values file: {a.values}", file=sys.stderr)
            return 2
        for line in a.values.read_text().splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            by_idx[v["game_idx"]] = v
        missing = [r.get("game_idx") for r in recs if r.get("game_idx") not in by_idx]
        if missing:
            print(f"ALERT annotate: {len(missing)} games without values (e.g. {missing[:3]})", file=sys.stderr)
            return 2

    workers = max(1, min(int(a.workers), 4))
    anns: list[dict] = []
    mds: list[str] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
            payloads = [(r, by_idx.get(r.get("game_idx")) if by_idx else None) for r in recs]
            for out in ex.map(_annotate_one, payloads):
                anns.append(out["ann"])
                mds.append(out["md"])
    except A.ReplayMismatch as e:
        print(f"ALERT annotate: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - fail loud on any worker error
        print(f"ALERT annotate: worker failed: {e!r}", file=sys.stderr)
        return 1

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
        ).stdout.strip()
    except OSError:
        commit = "?"
    summary = A.summarize(anns, [str(d) for d in a.games], commit or "?")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "games").mkdir(exist_ok=True)
    for ann in anns:
        (a.out / "games" / f"game-{ann['game_idx']}.json").write_text(json.dumps(ann, indent=1))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (a.out / "summary.md").write_text(A.render_summary_md(summary))

    md_idx = {int(x) for x in a.md_games.split(",") if x.strip()} if a.md_games else None
    (a.out / "md").mkdir(exist_ok=True)
    n_md = 0
    for ann, md in zip(anns, mds, strict=True):
        if md_idx is not None:
            want = ann["game_idx"] in md_idx
        else:
            want = n_md < a.md
        if want:
            (a.out / "md" / f"game-{ann['game_idx']}.md").write_text(md)
            n_md += 1

    cal = summary["calibration"]["bins"]
    hi, mid = cal[9], cal[4]
    print(
        f"games={len(anns)} searched={summary['meta']['n_searched_decisions']} "
        f"bin0.9-1.0 n={hi['n']} won={hi['won']:.3f} bin0.4-0.5 n={mid['n']} won={mid['won']:.3f}"
    )
    print(f"ECE={summary['calibration']['ece']:.4f} Brier={summary['calibration']['brier_by_bucket']}")
    tp = summary["tempo"]
    print(f"swing_share={tp['swing_share']} swing_winner_winrate={tp['swing_winner_winrate']:.3f}")
    if "value_calibration" in summary:
        eces = {b: summary["value_calibration"][b]["ece"] for b in ("<=2", "3", "4", "5+")}
        accs = {b: summary["v_comb_sign"][b]["acc"] for b in ("<=2", "3", "4", "5+")}
        dis = {b: summary["disagree"][b]["mean_abs"] for b in ("<=2", "3", "4", "5+")}
        print(f"value_ECE={eces} v_comb_sign_acc={accs} mean_|disagree|={dis}")
    if "aux" in summary:
        ax = summary["aux"]
        print(
            f"aux_points={ax['n_points']} ends_Brier={ax['ends']['brier']:.4f}/{ax['ends']['brier_base']:.4f}"
            f" d_lp_MAE={ax['d_lp']['mae']:.1f}_R2={ax['d_lp']['r2']}"
            f" d_cards_MAE={ax['d_cards']['mae']:.3f}_R2={ax['d_cards']['r2']}"
            f" turns_MAE={ax['turns_left']['mae']:.2f}_R2={ax['turns_left']['r2']}"
        )
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
