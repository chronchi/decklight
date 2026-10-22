"""Coverage of the Lua per-effect extractor over all official card scripts.

uv run python scripts/lua_effects_stats.py
"""

import time
from collections import Counter

from decklight.cards.lua_effects import OFFICIAL_DIR, load_constants, parse_script

FAILURE_SAMPLE = 5
TOP_UNRESOLVED = 15


def main() -> None:
    t0 = time.time()
    constants = load_constants()
    files = sorted(OFFICIAL_DIR.glob("c*.lua"))
    failures: list[str] = []
    n_records = 0
    n_with_record = 0
    n_unresolved_records = 0
    unresolved_counts: Counter[str] = Counter()
    for path in files:
        try:
            code = int(path.stem[1:])
            recs = parse_script(path.read_text(encoding="utf-8"), code, constants)
        except Exception as e:  # noqa: BLE001 - per-script isolation; count and continue
            failures.append(f"{path.name} ({e})")
            continue
        n_records += len(recs)
        n_with_record += bool(recs)
        for rec in recs:
            if rec.unresolved:
                n_unresolved_records += 1
                unresolved_counts.update(rec.unresolved)
    n = len(files)
    print(f"scripts: {n}")
    print(f"parse failures: {len(failures)}")
    for f in failures[:FAILURE_SAMPLE]:
        print(f"  {f}")
    print(f"records: {n_records}")
    print(f"scripts with >=1 record: {100.0 * n_with_record / n:.1f}%")
    if n_records:
        print(f"records with any unresolved field: {100.0 * n_unresolved_records / n_records:.1f}%")
    print(f"top-{TOP_UNRESOLVED} unresolved setter args:")
    for arg, c in unresolved_counts.most_common(TOP_UNRESOLVED):
        print(f"  {c:6d}  {arg}")
    print(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
