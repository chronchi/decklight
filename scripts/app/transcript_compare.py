"""Compare two `ws_transcript.py` dumps as parsed JSON (A2c parity gate).

    uv run python scripts/app/transcript_compare.py PY.jsonl.gz RS.jsonl.gz

Allowed differences, and only these: the lobby's `models` / `model_default`, a state's `model_label`, any
`record_path` field, a hint's `ms` (its wall time), a review's `timing` (wall times) and a what-if's
`what_if.source` (the record's name: a directory in Python, a match id in Rust). Everything else (frame order, keys, integers vs floats, exact float values) must match.
Prints a JSON summary and exits 2 on any mismatch with an ALERT line.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def load(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def path_str(path: list) -> str:
    out = ""
    for p in path:
        out += f"[{p}]" if isinstance(p, int) else (("." if out else "") + str(p))
    return out


def allowed(path: str, frame: dict) -> bool:
    if "record_path" in path:
        return True
    if frame.get("t") == "hint" and path == "ms":  # wall time of the hint (A4)
        return True
    if frame.get("t") == "review" and (path == "timing" or path.startswith("timing.")):  # wall times (A4)
        return True
    if frame.get("t") == "state" and path == "what_if.source":
        return True
    if frame.get("t") == "lobby" and (
        path in ("models", "model_default") or path.startswith(("models.", "models["))
    ):
        return True
    return frame.get("t") == "state" and path == "model_label"


def diff(a, b, path: list, frame: dict) -> list | None:
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [path, a, b]
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return [path + [k], a.get(k, "<missing>"), b.get(k, "<missing>")]
            d = diff(a[k], b[k], path + [k], frame)
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + ["<len>"], len(a), len(b)]
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, path + [i], frame)
            if d:
                return d
        return None
    if isinstance(a, float) or isinstance(b, float):
        if float(a) != float(b):
            return [path, a, b]
        return None
    if a != b:
        return [path, a, b]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("py", type=Path)
    ap.add_argument("rs", type=Path)
    a = ap.parse_args(argv)
    n = mism = frames = games = 0
    first = None
    it_a, it_b = load(a.py), load(a.rs)
    while True:
        try:
            x = next(it_a)
        except StopIteration:
            x = None
        try:
            y = next(it_b)
        except StopIteration:
            y = None
        if x is None and y is None:
            break
        n += 1
        frames += 1
        if x is None or y is None:
            mism += 1
            first = first or {"frame": n, "path": "<missing frame>", "py": x, "rs": y}
            break
        gx, gy = x.get("g"), y.get("g")
        if gx is not None:
            games = max(games, int(gx) + 1)
        if gx != gy:
            mism += 1
            first = first or {"frame": n, "path": "g", "py": gx, "rs": gy}
            continue
        fx, fy = x.get("frame", {}), y.get("frame", {})
        frame = fx if isinstance(fx, dict) else {}
        d = diff(fx, fy, [], frame)
        if d and not allowed(path_str(d[0]), frame):
            mism += 1
            if first is None:
                first = {"frame": n, "path": path_str(d[0]), "py": d[1], "rs": d[2]}
    summary = {"frames": frames, "games": games, "mismatches": mism, "first": first}
    print(json.dumps(summary))
    if mism:
        print(f"ALERT transcript_compare: {mism} mismatches (first {first})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
