""".ydk deck files: `#main`, `#extra`, `!side` sections of passcodes, one per line."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Deck:
    main: tuple[int, ...]
    extra: tuple[int, ...] = ()
    side: tuple[int, ...] = ()
    name: str = field(default="", compare=False)


def parse_ydk(text: str, name: str = "") -> Deck:
    sections: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    cur = sections["main"]
    for line in text.splitlines():
        line = line.strip()
        if line in ("#main", "#extra", "!side"):
            cur = sections[line[1:]]
        elif line.isdigit() and int(line) > 0:
            cur.append(int(line))
    return Deck(tuple(sections["main"]), tuple(sections["extra"]), tuple(sections["side"]), name)


def load_ydk(path: str | Path) -> Deck:
    path = Path(path)
    return parse_ydk(path.read_text(encoding="utf-8"), path.stem)
