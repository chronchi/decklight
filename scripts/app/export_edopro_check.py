"""Check an exported .yrpX (Games > Export) with a real EDOPro install's core, scripts and cards (laptop only).

    DECKLIGHT_EXPORT_DUMP=/tmp/x.yrpX cargo test -p decklight-server exports_edopro   # writes a whole-game export
    uv run python scripts/app/export_edopro_check.py /tmp/x.yrpX ~/Documents/edopro/ProjectIgnis [--install]

Rebuilds the duel from the file alone, the way EDOPro's Replay screen does (every card at sequence 0, team 0
first), and checks the install's core gives the recorded messages. Exit 2 when it does not. --install copies the
file to the install's replay/ folder to watch it there.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.demos.yrp import edopro_engine
from decklight.play.yrp import parse_yrpx, to_record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("edopro", type=Path)
    ap.add_argument("--install", action="store_true")
    a = ap.parse_args()
    x = parse_yrpx(a.file.read_bytes())
    if x.yrp is None:
        print("FAIL: no embedded replay")
        return 2
    rec, problems = to_record(x, edopro_engine(a.edopro), viewer=None, app="export-check")
    if rec is None:
        print("FAIL: EDOPro's core does not reproduce the file:\n  " + "\n  ".join(problems))
        return 2
    if rec.get("deck_add") != "edopro" or "add_order" in rec:
        print(f"FAIL: reproduces only with another card order (deck_add={rec.get('deck_add')}, add_order={rec.get('add_order')})")
        return 2
    print(f"OK: {len(rec['responses'])} responses, {rec['result']['turns']} turns, winner {rec['result']['winner']}, "
          f"names {x.yrp.names}")
    if a.install:
        dst = a.edopro / "replay" / a.file.name
        shutil.copyfile(a.file, dst)
        print(f"copied to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
