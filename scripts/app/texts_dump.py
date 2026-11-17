"""Dump the card text store (`_Db`) for the Rust port's parity check (A1b).

    uv run python scripts/app/texts_dump.py --out data/cache/app/texts.jsonl

One line per code in `names()` union the cdb ids:
`{"code", "name", "text": effect_text(code), "strs": [desc_text(code, i) for i in range(16)], "stats": [...] or null}`,
then one line per system string `{"sys": n, "text": desc_text(0, n)}`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def system_numbers(strings_conf: Path) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    if not strings_conf.exists():
        return out
    for line in strings_conf.read_text(encoding="utf-8").splitlines():
        if line.startswith("!system "):
            _, num, *rest = line.split(" ", 2)
            if num.isdigit() and rest:
                n = int(num)
                if n not in seen:
                    seen.add(n)
                    out.append(n)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "app" / "texts.jsonl")
    a = ap.parse_args(argv)

    from decklight.web import view

    db = view.DB
    names = db.names()
    stats = db.stats()
    codes = sorted(set(names) | set(stats))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_texts = 0
    with a.out.open("w", encoding="utf-8") as fh:
        for code in codes:
            row = {
                "code": int(code),
                "name": names.get(code, ""),
                "text": db.effect_text(code),
                "strs": [db.desc_text(code, i) for i in range(16)],
                "stats": list(stats[code]) if code in stats else None,
            }
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            n_texts += 1
        sys_nums = system_numbers(view.STRINGS_CONF)
        for n in sys_nums:
            fh.write(json.dumps({"sys": n, "text": db.desc_text(0, n)}, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {"codes": len(codes), "sys_strings": len(sys_nums), "out": str(a.out)}, separators=(",", ":")
        ),
        file=sys.stderr,
    )
    return 0 if n_texts else 3


if __name__ == "__main__":
    sys.exit(main())
