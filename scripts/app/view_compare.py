"""Compare two `view_dump` files line by line as parsed JSON values (A2a parity gate).

    uv run python scripts/app/view_compare.py A.py.jsonl B.rs.jsonl

Dict key order does not matter; everything else does (an int is not a float, `1.0 != 1`, floats exactly).
Prints `{"lines", "frames", "mismatches", "first"}` and exits 2 on any mismatch with an ALERT on stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def cmp(a: Any, b: Any, path: str) -> str | None:
    ta, tb = type(a), type(b)
    if ta is not tb:
        return path
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            missing = sorted(set(a.keys()) ^ set(b.keys()))
            return f"{path}.{missing[0]}" if missing else path
        for k in a:
            bad = cmp(a[k], b[k], f"{path}.{k}")
            if bad is not None:
                return bad
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}[{min(len(a), len(b))}]"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            bad = cmp(x, y, f"{path}[{i}]")
            if bad is not None:
                return bad
        return None
    return None if a == b else path


def add_stats(v: Any, prompts: dict, events: dict, cues: dict) -> None:
    if not isinstance(v, dict):
        return
    p = v.get("prompt")
    if isinstance(p, dict):
        k = str(p.get("kind"))
        prompts[k] = prompts.get(k, 0) + 1
    for e in v.get("events") or []:
        if isinstance(e, dict):
            k = str(e.get("kind"))
            events[k] = events.get(k, 0) + 1
    for c in v.get("cues") or []:
        if isinstance(c, dict):
            k = str(c.get("t"))
            cues[k] = cues.get(k, 0) + 1
    for c in v.get("cues_all") or []:
        if isinstance(c, dict):
            k = str(c.get("t"))
            cues[k] = cues.get(k, 0) + 1


def lines_of(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: view_compare.py A.jsonl B.jsonl", file=sys.stderr)
        return 2
    a_path, b_path = (Path(x) for x in argv)
    prompts: dict[str, int] = {}
    events: dict[str, int] = {}
    cues: dict[str, int] = {}
    mismatches = 0
    first: dict | None = None
    lines = 0
    frames = 0
    a_it, b_it = lines_of(a_path), lines_of(b_path)
    i = 0
    while True:
        la = next(a_it, None)
        lb = next(b_it, None)
        if la is None and lb is None:
            break
        i += 1
        if la is None or lb is None:
            mismatches += 1
            if first is None:
                first = {
                    "line": i,
                    "path": "$",
                    "py": json.loads(la) if la is not None else "<missing>",
                    "rs": json.loads(lb) if lb is not None else "<missing>",
                }
            continue
        a = json.loads(la)
        b = json.loads(lb)
        lines += 1
        if isinstance(a, dict) and "view" in a:
            frames += 1
        add_stats(a, prompts, events, cues)
        path = cmp(a, b, "$")
        if path is not None:
            mismatches += 1
            if first is None:
                first = {"line": i, "path": path, "py": a, "rs": b}
    print(
        json.dumps(
            {"lines": lines, "frames": frames, "mismatches": mismatches, "first": first},
            separators=(",", ":"),
        )
    )
    print(
        json.dumps(
            {"prompts_by_kind": prompts, "events_by_kind": events, "cues_by_t": cues},
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    if mismatches:
        loc = "?" if first is None else f"line {first['line']} path {first['path']}"
        print(f"ALERT view_compare: {mismatches} mismatching lines; first at {loc}", file=sys.stderr)
        return 2
    print(f"OK: {lines} lines identical", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
