"""Offline post-game review of a finished web record (thin wrapper over analysis.review).

Writes the review JSON to --out when given, otherwise to stdout. It never
writes next to the record: only the web server caches review.json, in
runs/web/<ts>/.

taskset -c 0-3 uv run python scripts/review_game.py <record-dir-or-game.jsonl> [--max-steps N] [--out review.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None, help="write the review JSON here (default: stdout)")
    a = ap.parse_args()

    from ygo_zero.analysis.review import load_record, review_game
    from ygo_zero.edopro.product_bot import ProductLauncher
    from ygo_zero.engine.duel import Engine
    from ygo_zero.eval.winchance import load_calib

    rec, cfg, _hs, _d = load_record(a.record)
    if rec.get("mode") not in (None, "product", "fast"):
        print(f"record mode is {rec.get('mode')!r}: review needs the full bot", flush=True)
    engine = Engine()
    launcher = ProductLauncher(device="cpu")
    calib = load_calib()

    t0 = time.monotonic()

    def progress(done: int, total: int) -> None:
        print(f"Reviewing… {done}/{total}", flush=True)

    out = review_game(engine, cfg, rec, launcher, calib, max_steps=a.max_steps, progress=progress)
    dt = time.monotonic() - t0
    print(
        f"timing: {out['timing']['seconds']}s for {out['timing']['decisions']} decisions "
        f"(fwd {out['timing']['fwd_ms']}ms value {out['timing']['value_ms']}ms, wall {dt:.1f}s)",
        flush=True,
    )
    print(
        f"counts: {json.dumps(out['counts'])} accuracy {out['accuracy']} missed {out['missed']}", flush=True
    )
    worst = sorted(out["decisions"], key=lambda e: e["delta_pts"])[:5]
    for e in worst:
        print(
            f"k={e['k']} T{e['turn']} {e['phase']} {e['kind']} {e['mark']} {e['delta_pts']:+.1f} "
            f"You: {e['your']['label']} · Agent: {e['agent']['label']}",
            flush=True,
        )
    if a.out is not None:
        a.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}", flush=True)
    else:
        print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
