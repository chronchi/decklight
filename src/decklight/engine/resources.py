"""What the core needs from the host: card stats (from BabelCDB) and Lua scripts (from CardScripts).

Both are loaded once into memory, mirroring how EDOPro reads them (`gframe/data_manager.cpp`).

Custom cards: `YGO_EXTRA_CARDS` is an `os.pathsep`-separated list of extra card dirs, each laid
out like the friend's repo (`*.cdb` at the top, `script/c<code>.lua`, `pics/`). `default_cdbs()`
appends every `<dir>/*.cdb` after the official files and `index_scripts()` adds every
`<dir>/script/*.lua` without overriding official names. Unset (or no existing dirs) means
behaviour is byte-identical to before: every caller (`engine/duel.py`, `cards/embed.py`,
`cards/effect_strings.py`, `web/view.py`, `web/server.py`, `demos/`) picks the extras up through
these two functions, with no second code path.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from decklight.engine import constants as C

ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = ROOT / "build/core/libocgcore.so"
SCRIPTS_DIR = ROOT / "third_party/CardScripts"
CDB_DIR = ROOT / "third_party/BabelCDB"
EXTRA_CARDS_ENV = "YGO_EXTRA_CARDS"


@dataclass(frozen=True, slots=True)
class CardData:
    """Card stats exactly as handed to the core (OCG_CardData semantics)."""

    code: int
    alias: int
    setcodes: tuple[int, ...]
    type: int
    level: int
    attribute: int
    race: int
    attack: int
    defense: int
    lscale: int
    rscale: int
    link_marker: int


def extra_card_dirs(explicit: list[Path] | tuple[Path, ...] | None = None) -> list[Path]:
    """Existing extra card dirs.

    `explicit` given (even []) wins: no env read. `None` keeps today's `YGO_EXTRA_CARDS`
    fallback (existing dirs only, [] when unset).
    """
    if explicit is not None:
        return [d for d in (Path(p) for p in explicit) if d.is_dir()]
    raw = os.environ.get(EXTRA_CARDS_ENV, "")
    out: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part and (Path(part).is_dir()):
            out.append(Path(part))
    return out


def extra_script_dirs(explicit: list[Path] | tuple[Path, ...] | None = None) -> list[Path]:
    """`<dir>/script` of every existing extra card dir (only those that exist)."""
    if explicit is not None:
        return [d / "script" for d in extra_card_dirs(explicit) if (d / "script").is_dir()]
    return [d / "script" for d in extra_card_dirs() if (d / "script").is_dir()]


def default_cdbs(cdb_dir: Path = CDB_DIR,
                 extra_dirs: list[Path] | tuple[Path, ...] | None = None) -> list[Path]:
    """TCG/OCG cards plus (pre)release sets; later files override earlier ones. No rush/skills/goat.

    `extra_dirs=None` keeps the `YGO_EXTRA_CARDS` fallback; `[]` means no extras.
    """
    extra = sorted(cdb_dir.glob("release-*.cdb")) + sorted(
        p for p in cdb_dir.glob("prerelease-*.cdb") if "rush" not in p.name
    )
    paths = [cdb_dir / "cards.cdb", *extra]
    if Path(cdb_dir) == CDB_DIR:
        for d in extra_card_dirs(explicit=extra_dirs):
            paths += sorted(d.glob("*.cdb"))
    return paths


def load_cards(paths: list[Path] | None = None) -> dict[int, CardData]:
    cards: dict[int, CardData] = {}
    for path in paths or default_cdbs():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT id, alias, setcode, type, atk, def, level, race, attribute FROM datas"
            ).fetchall()
        finally:
            con.close()
        for code, alias, setcode, type_, atk, def_, level, race, attribute in rows:
            setcodes = tuple(s for i in range(4) if (s := (setcode >> (16 * i)) & 0xFFFF))
            link_marker = 0
            if type_ & C.TYPE_LINK:
                link_marker, def_ = def_, 0
            lv = -(level & 0xFF) if level < 0 else level & 0xFF
            cards[code] = CardData(
                code=code,
                alias=alias,
                setcodes=setcodes,
                type=type_,
                level=lv & 0xFFFFFFFF,
                attribute=attribute,
                race=race & 0xFFFFFFFFFFFFFFFF,
                attack=atk,
                defense=def_,
                lscale=(level >> 24) & 0xFF,
                rscale=(level >> 16) & 0xFF,
                link_marker=link_marker,
            )
    return cards


def index_scripts(scripts_dir: Path = SCRIPTS_DIR,
                  extra_dirs: list[Path] | tuple[Path, ...] | None = None) -> dict[str, Path]:
    """Map script file name -> path. Root (system/proc scripts) first, then official, then pre-release,
    then every extra card dir (extras never override official names).

    `extra_dirs=None` keeps the `YGO_EXTRA_CARDS` fallback; `[]` means no extras.
    """
    index: dict[str, Path] = {}
    for sub in ("", "official", "pre-release"):
        for p in sorted((scripts_dir / sub).glob("*.lua")):
            index.setdefault(p.name, p)
    if Path(scripts_dir) == SCRIPTS_DIR:
        for d in extra_script_dirs(explicit=extra_dirs):
            for p in sorted(d.glob("*.lua")):
                index.setdefault(p.name, p)
    return index


class CardSet:
    """An explicit card set: the official cards plus extra custom-card dirs.

    `extra_dirs=()` (the default) is the official set. `None` means "today's env
    fallback" and is resolved once, at construction. The key is the sorted extra
    cdb/script paths, so two bundles with clashing codes get different engines.
    """

    def __init__(self, extra_dirs: tuple[Path, ...] | list[Path] | None = None) -> None:
        if extra_dirs is None:
            resolved = extra_card_dirs()
        else:
            resolved = [p for p in (Path(q) for q in extra_dirs) if p.is_dir()]
        self.extra_dirs = tuple(resolved)

    def key(self) -> tuple[str, ...]:
        """Cache key: the sorted extra cdb/script paths (official cards always included)."""
        names: list[str] = []
        for d in self.extra_dirs:
            names += [str(p) for p in sorted(d.glob("*.cdb"))]
            script_dir = d / "script"
            if script_dir.is_dir():
                names += [str(p) for p in sorted(script_dir.glob("*.lua"))]
        return tuple(sorted(names))

    def engine(self):
        """An `Engine` over this card set (loads the core library)."""
        from decklight.engine.duel import Engine

        return Engine(cards=load_cards(default_cdbs(extra_dirs=list(self.extra_dirs))),
                      scripts=index_scripts(extra_dirs=list(self.extra_dirs)))


_ENGINES: dict[tuple[str, ...], object] = {}


def engine_for(cardset: "CardSet"):
    """One `Engine` per card-set key, created on first use and cached (web server: many
    bundles share one process, each duel uses its bundle's engine)."""
    key = cardset.key()
    engine = _ENGINES.get(key)
    if engine is None:
        engine = cardset.engine()
        _ENGINES[key] = engine
    return engine
