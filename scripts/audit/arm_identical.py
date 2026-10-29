"""Are two search_match arms the same games? (Forge L3a: may a release's arm be cached and reused?)

Keys games by (decks, seed, seat) like x4b_spec_card and compares the response list and the result. Prints the
counts and `IDENTICAL yes|no` (yes = every shared game has the same choices and result, and >= 99% of A pairs up).
Exit 0 either way (a measurement); exit 2 when an arm has no games.

    python scripts/audit/arm_identical.py runs/<a>/card/P-invoked runs/<b>/card/P-rep-invoked
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x4b_spec_card import games


def main() -> int:
    a, b = games(Path(sys.argv[1])), games(Path(sys.argv[2]))
    if not a or not b:
        print("ALERT arm_identical: an arm has no games")
        return 2
    keys = sorted(set(a) & set(b))
    same_moves = sum(1 for k in keys if a[k][2] is not None and a[k][2] == b[k][2])
    same_result = sum(1 for k in keys if a[k][0] == b[k][0])
    ok = len(keys) >= 0.99 * len(a) and same_moves == len(keys) and same_result == len(keys)
    print(f"A {len(a)} games, B {len(b)}, paired {len(keys)}; identical moves {same_moves}, identical result "
          f"{same_result}; A win rate {sum(a[k][0] for k in keys) / len(keys):.4f}, "
          f"B {sum(b[k][0] for k in keys) / len(keys):.4f}")
    print(f"IDENTICAL {'yes' if ok else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
