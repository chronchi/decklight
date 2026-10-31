"""Sidecar provenance for the web client's precomputed content.

`puzzles.json` and `primers.json.gz` are built from bot games; this module
records which model built them (`content_sources.json`) and warns when the
live product has moved on.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from decklight.search.product import sha256_file

SOURCES_PATH = Path(__file__).resolve().parent / "content_sources.json"

KINDS = ("puzzles", "primers")


def load_sources(path=SOURCES_PATH) -> dict:
    """The provenance file, or {} when missing/unreadable (never raises)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- missing or corrupt: no provenance
        return {}


def stamp(
    kind: str,
    *,
    model: str,
    ckpt: Path,
    value: Path | None,
    games: list[str],
    note: str = "",
    path=SOURCES_PATH,
) -> dict:
    """Record the model that built `kind`; rewrite the sidecar file. Returns the entry."""
    if kind not in ("puzzles", "primers"):
        raise ValueError(f"unknown content kind {kind!r} (want puzzles|primers)")
    ckpt_sha = sha256_file(ckpt)
    if value is not None:
        value_sha = sha256_file(value)
        value_label: str | None = Path(value).parent.name or Path(value).stem
    else:
        value_sha = None
        value_label = None
    entry = {
        "model": model,
        "ckpt_sha256": ckpt_sha,
        "value": value_label,
        "value_sha256": value_sha,
        "games": list(games),
        "built": datetime.now(UTC).date().isoformat(),
        "note": note,
    }
    sources = load_sources(path)
    if not isinstance(sources, dict):
        sources = {}
    sources[kind] = entry
    out = json.dumps(sources, indent=1, sort_keys=True) + "\n"
    Path(path).write_text(out, encoding="utf-8")
    return entry


def check_live(ckpt: Path, value: Path | None, path=SOURCES_PATH) -> list[str]:
    """One line per stale content kind; [] when everything matches. Never raises."""
    try:
        sources = load_sources(path)
        if not isinstance(sources, dict):
            sources = {}
        try:
            live_ckpt = sha256_file(ckpt)
        except Exception as e:  # noqa: BLE001 -- unreadable live ckpt: one line per kind
            return [f"{k}: live ckpt unreadable: {e}" for k in KINDS]
        live_value: str | None = None
        if value is not None:
            try:
                live_value = sha256_file(value)
            except Exception as e:  # noqa: BLE001 -- unreadable live value
                return [f"{k}: live value unreadable: {e}" for k in KINDS]
        out: list[str] = []
        for kind in KINDS:
            try:
                entry = sources.get(kind) if isinstance(sources, dict) else None
                if not isinstance(entry, dict):
                    out.append(f"{kind}: no provenance recorded")
                    continue
                rec_model = entry.get("model", "?")
                rec_ckpt = entry.get("ckpt_sha256")
                rec_val = entry.get("value_sha256")
                ckpt_diff = (not isinstance(rec_ckpt, str)) or (live_ckpt != rec_ckpt)
                val_diff = (
                    isinstance(rec_val, str)
                    and live_value is not None
                    and live_value != rec_val
                )
                if ckpt_diff or val_diff:
                    rec_short = rec_ckpt[:8] if isinstance(rec_ckpt, str) else "????????"
                    out.append(
                        f"{kind} were built from model {rec_model} "
                        f"({rec_short}), live product policy is {live_ckpt[:8]}"
                    )
            except Exception as e:  # noqa: BLE001 -- one kind never breaks the other
                out.append(f"{kind}: provenance check failed: {e}")
        return out
    except Exception as e:  # noqa: BLE001 -- never raises
        return [f"provenance check failed: {e}"]
