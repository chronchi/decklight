"""Merge stage M turn lines verified on vast (split mode, scripts/vast_stage_m.sh) into data/demos/turns.

Each `<src>/<bucket>/<file>.jsonl` is appended to `data/demos/turns/<bucket>/<file>.jsonl`, skipping lines whose
line_id is already anywhere under data/demos/turns (a round's end copy repeats earlier rounds' p-files). Prints the
per-bucket counts. Exit 2 when the source holds no lines at all. Engine-free.

    python scripts/merge_turns.py runs/stage-m-20261012-2338/turns
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DST = Path("data/demos/turns")


def main() -> int:
    src = Path(sys.argv[1])
    seen = {json.loads(x)["line_id"] for f in DST.glob("*/*.jsonl") for x in f.read_text().splitlines() if x.strip()}
    added, dup, total = Counter(), Counter(), 0
    for f in sorted(src.glob("*/*.jsonl")):
        bucket = f.parent.name
        keep = []
        for x in f.read_text().splitlines():
            if not x.strip():
                continue
            total += 1
            lid = json.loads(x)["line_id"]
            if lid in seen:
                dup[bucket] += 1
                continue
            seen.add(lid)
            keep.append(x)
        if keep:
            (DST / bucket).mkdir(parents=True, exist_ok=True)
            with open(DST / bucket / f.name, "a") as out:
                out.write("\n".join(keep) + "\n")
            added[bucket] += len(keep)
    if total == 0:
        print(f"ALERT merge_turns: no lines under {src}")
        return 2
    for b in sorted(set(added) | set(dup)):
        print(f"{b}: added {added[b]}, already present {dup[b]}")
    print(f"merge_turns: {sum(added.values())} added, {sum(dup.values())} duplicates; "
          f"data/demos/turns now {len(seen)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
