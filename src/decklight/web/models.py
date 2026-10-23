"""Model discovery for the web lobby: base + per-deck fine-tunes + CLI override.

Pure function over the filesystem (no model loading): the server calls it once
at startup and again for every lobby message, so a fine-tune finished while the
server runs appears after a page reload.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from decklight.search.product import PRODUCT_CKPT, PRODUCT_VALUE

ROOT = Path(__file__).resolve().parents[3]


def _deck_content_key(deck) -> tuple:
    """Multiset of main+extra codes (order-insensitive content comparison)."""
    return (tuple(sorted(deck.main)), tuple(sorted(deck.extra)))


def _load_lobby_decks(decks: list[dict]) -> dict[str, tuple]:
    """Map lobby deck id -> content key, best effort (unknown ids skipped)."""
    from decklight.decks import load_ydk

    out: dict[str, tuple] = {}
    for d in decks or []:
        did = d.get("id") if isinstance(d, dict) else None
        if not did:
            continue
        for cand in (
            ROOT / "decks" / "pool" / f"{did}.ydk",
            ROOT / "decks" / "newcards" / f"{did}.ydk",
        ):
            if cand.is_file():
                try:
                    out[did] = _deck_content_key(load_ydk(cand))
                except (OSError, ValueError):
                    pass
                break
    return out


def _load_deck_key(deck_ref: str | None):
    """Content key of the FINETUNE deck ref (repo-relative or absolute), or None."""
    if not deck_ref:
        return None
    from decklight.decks import load_ydk

    p = Path(deck_ref)
    if not p.is_absolute():
        p = ROOT / deck_ref
    if not p.is_file():
        return None
    try:
        return _deck_content_key(load_ydk(p))
    except (OSError, ValueError):
        return None


def discover_models(
    dirs: list[Path], decks: list[dict], cli_ckpt, cli_value
) -> list[dict]:
    """All selectable models: base first, then fine-tunes newest first.

    Each entry carries `ckpt`/`value` Paths (the server strips them before
    sending the lobby to the client). Prints one ALERT line per skipped model.
    """
    models: list[dict] = [
        {
            "id": "base",
            "short": "base",
            "label": "Base model",
            "kind": "base",
            "deck_name": None,
            "deck_id": None,
            "ckpt": PRODUCT_CKPT,
            "value": PRODUCT_VALUE,
        }
    ]
    lobby_keys = _load_lobby_decks(decks)

    found: list[dict] = []
    for d in dirs or []:
        root = Path(d)
        if not root.is_dir():
            continue
        cands: list[Path] = []
        cands.extend(sorted(root.glob("*/FINETUNE.json")))
        cands.extend(sorted(root.glob("*/*/FINETUNE.json")))
        for f in cands:
            policy = f.parent / "policy" / "best.pt"
            value = f.parent / "value" / "best.pt"
            if not policy.is_file() or not value.is_file():
                why = []
                if not policy.is_file():
                    why.append("policy/best.pt missing")
                if not value.is_file():
                    why.append("value/best.pt missing")
                print(f"ALERT web: model {f.parent} skipped: {', '.join(why)}", flush=True)
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                print(f"ALERT web: model {f.parent} skipped: FINETUNE.json unreadable ({e})", flush=True)
                continue
            deck_ref = data.get("deck", "")
            deck_name = Path(deck_ref).stem if deck_ref else f.parent.name
            try:
                hours = round(float(data.get("hours", 0.0)), 1)
            except (TypeError, ValueError):
                hours = 0.0
            key = _load_deck_key(deck_ref)
            deck_path = None
            if deck_ref:
                dp = Path(deck_ref) if Path(deck_ref).is_absolute() else ROOT / deck_ref
                deck_path = dp if dp.is_file() else None
            deck_id = None
            if key is not None:
                for did, lk in lobby_keys.items():
                    if lk == key:
                        deck_id = did
                        break
            tm = time.localtime(f.stat().st_mtime)
            fid = hashlib.sha256(str(f.resolve()).encode()).hexdigest()[:10]
            found.append(
                {
                    "id": fid,
                    "short": fid[:4],
                    "label": f"Tuned for {deck_name}",
                    "kind": "finetune",
                    "deck_name": deck_name,
                    "deck_id": deck_id,
                    "date": f"{tm.tm_mday} {time.strftime('%b', tm)}",
                    "hours": hours,
                    "ckpt": policy,
                    "value": value,
                    "deck_path": deck_path,
                    "_mtime": f.stat().st_mtime,
                }
            )
    found.sort(key=lambda e: e.pop("_mtime"), reverse=True)
    models.extend(found)

    ckpt_p = Path(cli_ckpt) if cli_ckpt is not None else None
    value_p = Path(cli_value) if cli_value is not None else None
    if ckpt_p is not None or value_p is not None:
        matched = False
        for e in models:
            try:
                e_ckpt = Path(e["ckpt"]).resolve() if e.get("ckpt") is not None else None
                e_val = Path(e["value"]).resolve() if e.get("value") is not None else None
            except OSError:
                continue
            ck_ok = ckpt_p is None or (e_ckpt is not None and _same(ckpt_p, e_ckpt))
            v_ok = value_p is None or (e_val is not None and _same(value_p, e_val))
            if ck_ok and v_ok:
                matched = True
                break
        if not matched:
            ckpt_path = ckpt_p if ckpt_p is not None else Path(PRODUCT_CKPT)
            value_path = value_p if value_p is not None else Path(PRODUCT_VALUE)
            # FINETUNE.json two levels up from the ckpt (or value): <out>/FINETUNE.json.
            ft = None
            for p in (ckpt_p, value_p):
                if p is None:
                    continue
                cand = p.parent.parent / "FINETUNE.json"
                if cand.is_file():
                    ft = cand
                    break
            deck_name = None
            deck_id = None
            extra: dict = {}
            if ft is not None:
                try:
                    data = json.loads(ft.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    data = {}
                deck_ref = data.get("deck", "")
                if deck_ref:
                    deck_name = Path(deck_ref).stem
                    key = _load_deck_key(deck_ref)
                    if key is not None:
                        for did, lk in lobby_keys.items():
                            if lk == key:
                                deck_id = did
                                break
                try:
                    extra["hours"] = round(float(data.get("hours", 0.0)), 1)
                except (TypeError, ValueError):
                    pass
                try:
                    tm2 = time.localtime(ft.stat().st_mtime)
                    extra["date"] = f"{tm2.tm_mday} {time.strftime('%b', tm2)}"
                except OSError:
                    pass
            label = f"Tuned for {deck_name}" if deck_name else "Custom model"
            entry: dict = {
                "id": "cli",
                "short": "cli",
                "label": label,
                "kind": "cli",
                "deck_name": deck_name,
                "deck_id": deck_id,
                "ckpt": ckpt_path,
                "value": value_path,
            }
            entry.update(extra)
            models.append(entry)
    return models


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b
    except OSError:
        return False
