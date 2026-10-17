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

from ygo_zero.engine import constants as C

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


def extra_card_dirs() -> list[Path]:
    """Existing dirs from `YGO_EXTRA_CARDS` (`os.pathsep`-separated); [] when unset."""
    raw = os.environ.get(EXTRA_CARDS_ENV, "")
    out: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part and (Path(part).is_dir()):
            out.append(Path(part))
    return out


def extra_script_dirs() -> list[Path]:
    """`<dir>/script` of every existing extra card dir (only those that exist)."""
    return [d / "script" for d in extra_card_dirs() if (d / "script").is_dir()]


def default_cdbs(cdb_dir: Path = CDB_DIR) -> list[Path]:
    """TCG/OCG cards plus (pre)release sets; later files override earlier ones. No rush/skills/goat."""
    extra = sorted(cdb_dir.glob("release-*.cdb")) + sorted(
        p for p in cdb_dir.glob("prerelease-*.cdb") if "rush" not in p.name
    )
    paths = [cdb_dir / "cards.cdb", *extra]
    if Path(cdb_dir) == CDB_DIR:
        for d in extra_card_dirs():
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


def index_scripts(scripts_dir: Path = SCRIPTS_DIR) -> dict[str, Path]:
    """Map script file name -> path. Root (system/proc scripts) first, then official, then pre-release,
    then every `YGO_EXTRA_CARDS` dir (extras never override official names)."""
    index: dict[str, Path] = {}
    for sub in ("", "official", "pre-release"):
        for p in sorted((scripts_dir / sub).glob("*.lua")):
            index.setdefault(p.name, p)
    if Path(scripts_dir) == SCRIPTS_DIR:
        for d in extra_script_dirs():
            for p in sorted(d.glob("*.lua")):
                index.setdefault(p.name, p)
    return index
