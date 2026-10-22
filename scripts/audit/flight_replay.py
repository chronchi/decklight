"""Replay the in-flight games a segfaulted process left in $YGO_FLIGHT_DIR (engine/cpp_env.py FlightRecorder).

Each file is replayed in a child process, so a crash is caught as the child's exit code instead of killing this
script. Prints one line per file: `<file> <n choices> ok|CRASH <code>`; with --bisect, the shortest prefix that
crashes. Exit 2 when no file reproduces a crash (the crash needs state outside the recorded game)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CHILD = """
import json, sys
from types import SimpleNamespace
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine, Placement
lines = open(sys.argv[1]).read().splitlines()
h = json.loads(lines[0]); ch = [int(x) for x in lines[1:]][: int(sys.argv[2])]
cfg = DuelConfig(SimpleNamespace(main=h['deck0_main'], extra=h['deck0_extra']),
                 SimpleNamespace(main=h['deck1_main'], extra=h['deck1_extra']), h['seed'], flags=h['flags'],
                 lp=h['lp'], hand=h['hand'], draw=h['draw'], setup=tuple(Placement(*p) for p in h['setup']),
                 shuffle=False)
CppEnv.replay(Engine(), cfg, ch)
"""


def run(path: Path, n: int) -> int:
    return subprocess.run([sys.executable, "-c", CHILD, str(path), str(n)], capture_output=True).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--bisect", action="store_true")
    a = ap.parse_args()
    crashed = 0
    for f in a.files:
        n = len(f.read_text().splitlines()) - 1
        rc = run(f, n)
        if rc == 0:
            print(f"{f} {n} ok")
            continue
        crashed += 1
        msg = f"{f} {n} CRASH {rc}"
        if a.bisect:
            lo, hi = 0, n  # run(lo) ok, run(hi) crashes
            if run(f, 0) != 0:
                hi = 0
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if run(f, mid) != 0 else (mid, hi)
            msg += f"; first crashing prefix {hi} (choice {json.dumps(f.read_text().splitlines()[hi]) if hi else '-'})"
        print(msg, flush=True)
    sys.exit(0 if crashed else 2)


if __name__ == "__main__":
    main()
