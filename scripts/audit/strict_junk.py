"""Which stored line records would strict verification change? (retrain stage D, plan v3 §E efficiency)

Strict mode (`YGO_STRICT_VERIFY=1`, demos/verify.py) only removes one thing from the verifier's search: at an optional
chain window, the activation of a card the line never names (tried last, labelled `inferred`). A stored record without
such an activation is found unchanged by the strict search (the depth-first order of every other option is the same;
smoke 2026-10-11: blue-eyes and snake-eye files were byte-identical), so only lines whose record or expansions contain one
need re-verifying. This replays every record and lists those lines: an `inferred` ACTIVATE at a MSG_SELECT_CHAIN
decision of a card the line's text never names (the verifier checks the names of the remaining steps; "anywhere in the
line" is looser, so a few records that strict would change can be missed: a card named only in an earlier step).

Output: `--out` json {source: {"records": n, "flagged_records": n, "lines": [base line ids]}} and per-source counts.

    taskset -c 0-3 uv run python scripts/audit/strict_junk.py --out data/cache/audit/strict_junk.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.cards.embed import card_texts  # noqa: E402
from ygo_zero.demos.dataset import iter_samples  # noqa: E402
from ygo_zero.engine import constants as C  # noqa: E402
from ygo_zero.engine.actions import Kind  # noqa: E402
from ygo_zero.engine.duel import Engine  # noqa: E402

SOURCES = {
    "guides": ["data/demos/verified/*.jsonl", "data/demos/expanded/*.jsonl"],
    "claude": ["data/demos/claude/verified/*.jsonl", "data/demos/claude/expanded*/*.jsonl"],
    "planned": ["data/demos/planned/verified/*.jsonl", "data/demos/planned/expanded*/*.jsonl"],
    **{f"planned-s{i}": [f"data/demos/planned-s{i}/verified/*.jsonl", f"data/demos/planned-s{i}/expanded*/*.jsonl",
                         f"data/demos/planned-s{i}/improved/verified/*.jsonl",
                         f"data/demos/planned-s{i}/improved/expanded*/*.jsonl"] for i in range(1, 5)},
}  # fmt: skip


def lines_dir(src: str) -> Path:
    return ROOT / ("data/demos/lines" if src == "guides" else f"data/demos/{src}/lines")


def base_id(line_id: str) -> str:
    """'deck/plan-h0-a2+imp#h3' -> 'deck/plan-h0-a2' (hand variants '#hN', tags '+x')."""
    return line_id.split("#")[0].split("+")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(SOURCES))
    ap.add_argument("--max-files", type=int, default=0, help="per source, for a smoke")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    engine = Engine()
    texts = card_texts()
    out = {}
    for src, globs in SOURCES.items():
        if args.only and src not in args.only:
            continue
        files = sorted({Path(f) for g in globs for f in glob.glob(str(ROOT / g)) if Path(f).stat().st_size})
        if args.max_files:
            files = files[: args.max_files]
        t0, recs, flagged, bad = time.time(), set(), set(), 0
        text = {}  # base line id -> the line's json text (names are matched in it)
        for lf in lines_dir(src).glob("*.jsonl"):
            for raw in lf.read_text().splitlines():
                if raw.strip():
                    text[json.loads(raw)["id"]] = raw
        if files and not text:
            print(f"ALERT strict_junk: {src}: no lines in {lines_dir(src)}")
            return 2
        lines: Counter = Counter()
        for f in files:
            try:
                for obs, a, _w, meta in iter_samples(engine, f):
                    rid = meta.get("line_id")
                    recs.add((f.name, rid))
                    if meta.get("label") != "inferred" or int(obs["global"][21]) != C.MSG_SELECT_CHAIN:
                        continue
                    if int(obs["actions"][int(a), 0]) != Kind.ACTIVATE or (f.name, rid) in flagged:
                        continue
                    name = texts.get(int(obs["actions"][int(a), 2]), ("", ""))[0]
                    if not name or name not in text.get(base_id(rid), ""):
                        flagged.add((f.name, rid))
                        lines[base_id(rid)] += 1
            except Exception as e:  # a file that no longer replays is reported, not fatal
                bad += 1
                print(f"skip {f}: {e}", file=sys.stderr)
        out[src] = {"files": len(files), "bad_files": bad, "records": len(recs), "flagged_records": len(flagged),
                    "lines": sorted(lines)}  # fmt: skip
        print(f"{src}: {len(files)} files, {len(recs)} records, {len(flagged)} flagged "
              f"({len(flagged) / max(1, len(recs)):.1%}), {len(lines)} lines to re-verify, {time.time() - t0:.0f} s",
              flush=True)  # fmt: skip
        if files and not recs:
            print(f"ALERT strict_junk: {src}: 0 records replayed from {len(files)} files")
            return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
