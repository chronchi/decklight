"""Bot catalog: `config/bots/*.toml` entries plus WindBot argument parsing.

Adding a bot = adding a file:

    name = "random-snake-eye"
    deck = "decks/pool/snake-eye.ydk"   # repo-relative or absolute
    agent = "random"                    # make_agent name ("raw" needs StreamView)
    description = "..."
    # optional: seed, ckpt, emb, device

The WindBot `Key=Value` launch arguments (`Program.cs`) and the `--serve`
query parameters (`/?name=&deck=&...`) are parsed here so both the launcher
and the unit tests share one implementation.
"""

from __future__ import annotations

import tomllib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_DIR = REPO_ROOT / "config" / "bots"


@dataclass
class BotSpec:
    name: str
    deck: str  # .ydk path (repo-relative or absolute)
    agent: str = "random"
    seed: int = 0
    ckpt: str | None = None
    emb: str | None = None
    device: str = "cpu"
    description: str = ""

    def deck_path(self, root: Path = REPO_ROOT) -> Path:
        p = Path(self.deck)
        return p if p.is_absolute() else root / p


def load_catalog(directory: str | Path = DEFAULT_CATALOG_DIR) -> dict[str, BotSpec]:
    """All `<name>.toml` entries keyed by their `name` (falls back to the stem)."""
    out: dict[str, BotSpec] = {}
    directory = Path(directory)
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.toml")):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        name = str(data.get("name", path.stem))
        data.pop("name", None)
        try:
            out[name] = BotSpec(name=name, **data)
        except TypeError as e:
            raise ValueError(f"bad bot entry {path}: {e}") from None
    return out


def resolve_deck(ref: str, catalog: dict[str, BotSpec],
                 root: Path = REPO_ROOT) -> tuple[Path | None, BotSpec | None]:
    """`Deck=` value -> (ydk path, catalog entry or None).

    A catalog name wins; otherwise `ref` is a `.ydk` path (repo-relative or
    absolute, `.ydk` suffix optional). Returns (None, None) when neither matches.
    """
    if ref in catalog:
        spec = catalog[ref]
        return spec.deck_path(root), spec
    p = Path(ref)
    if p.suffix != ".ydk":
        p = p.with_suffix(".ydk")
    full = p if p.is_absolute() else root / p
    if full.is_file():
        return full, None
    return None, None


# --- WindBot launch arguments (Program.cs RunFromArgs / RunAsServer) ---

# Key=Value names (case-insensitive) -> canonical field. HostInfo= is the room
# password; password= is the same slot in the --serve URL.
KV_ALIASES = {
    "name": "name", "deck": "deck", "deckfile": "deckfile", "dialog": "dialog",
    "host": "host", "port": "port", "hostinfo": "password", "password": "password",
    "version": "version", "hand": "hand", "debug": "debug", "chat": "chat",
    "roomid": "roomid",
}


def _coerce(key: str, value: str):
    if key in ("port", "version", "hand", "roomid"):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"{key} needs an integer, got {value!r}") from None
    if key in ("debug", "chat"):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} needs a bool, got {value!r}")
    return value


def parse_kv_args(tokens: list[str]) -> tuple[dict, list[str]]:
    """Split `Key=Value` tokens (no leading dash) from argv.

    Returns (kv dict with canonical keys, remaining argv tokens).
    Unknown keys raise ValueError; bad ints/bools raise ValueError.
    """
    kv: dict = {}
    rest: list[str] = []
    for tok in tokens:
        if "=" in tok and not tok.startswith("-"):
            key, _, value = tok.partition("=")
            canon = KV_ALIASES.get(key.strip().lower())
            if canon is None:
                raise ValueError(f"unknown WindBot argument {key!r}")
            kv[canon] = _coerce(canon, value.strip())
        else:
            rest.append(tok)
    return kv, rest


def parse_serve_query(query_string: str) -> dict:
    """`/?name=&deck=&...` query -> canonical dict (same keys as parse_kv_args).

    Accepts exactly the parameters `Program.cs` reads (plus `deckfile`,
    `dialog`, `roomid`, which it also reads but the brief's short list omits).
    """
    params = urllib.parse.parse_qs(query_string, keep_blank_values=True)

    def first(*names):
        for n in names:
            if params.get(n):
                return params[n][0]
        return None

    out: dict = {}
    for key, pname in (("name", "name"), ("deck", "deck"), ("deckfile", "deckfile"),
                       ("dialog", "dialog"), ("host", "host"), ("port", "port"),
                       ("password", "password"), ("version", "version"),
                       ("hand", "hand"), ("debug", "debug"), ("chat", "chat"),
                       ("roomid", "roomid")):
        v = first(pname)
        if v is not None:
            out[key] = _coerce(key, v)
    return out


def list_bots_text(catalog: dict[str, BotSpec]) -> str:
    lines = []
    for name in sorted(catalog):
        spec = catalog[name]
        extra = f" [{spec.agent}] {spec.deck}"
        if spec.description:
            extra += f" — {spec.description.splitlines()[0][:100]}"
        lines.append(f"{name}{extra}")
    return "\n".join(lines) + ("\n" if lines else "")
