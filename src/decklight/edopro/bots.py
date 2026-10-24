"""Bot catalog: thin re-export of `decklight.play.bundle` plus WindBot argument parsing.

Adding a bot = adding a file: `bots/<name>/bot.toml` (user bundles) or
`config/bots/<name>.toml` (shipped examples); see `docs/bots.md` for the schema.

The WindBot `Key=Value` launch arguments (`Program.cs`) and the `--serve`
query parameters (`/?name=&deck=&...`) are parsed here so both the launcher
and the unit tests share one implementation.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from decklight.play.bundle import Bundle, BundleError, bundles_text, discover_bundles, resolve_bundle

list_bots_text = bundles_text
resolve_deck = resolve_bundle

__all__ = [
    "Bundle",
    "BundleError",
    "discover_bundles",
    "list_bots_text",
    "load_catalog",
    "parse_kv_args",
    "parse_serve_query",
    "resolve_deck",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_DIR = REPO_ROOT / "config" / "bots"


def load_catalog(directory: str | Path = DEFAULT_CATALOG_DIR) -> dict[str, Bundle]:
    """All bundles under `directory`, keyed by name (`discover_bundles`)."""
    return discover_bundles(directory)


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
