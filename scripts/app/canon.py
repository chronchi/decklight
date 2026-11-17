"""Canonical JSON for the Python oracle objects, byte-identical to the Rust `to_canon_json()` output.

    from scripts.app.canon import canon_msg, canon_card, canon_chain, dumps

`dumps(obj)` is `json.dumps(obj, separators=(",", ":"))`: no spaces, ints as integers (u64 values
above 2^53 included), tuples as lists, bools as true/false.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from decklight.engine import messages as M
from decklight.engine import query as Q


def _value(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, M.CardRef):
        return [v.code, v.controller, v.location, v.sequence, v.position]
    if isinstance(v, M.ChainOption):
        return [_value(v.card), v.description, v.client_mode]
    if isinstance(v, (tuple, list)):
        return [_value(x) for x in v]
    return v


def canon_msg(msg: M.Message) -> dict[str, Any]:
    """`{"cls": <dataclass name>, "type": int, <every field except raw, in declaration order>}`."""
    out: dict[str, Any] = {"cls": type(msg).__name__, "type": int(msg.type)}
    for f in dataclasses.fields(msg):
        if f.name in ("raw", "type"):
            continue
        out[f.name] = _value(getattr(msg, f.name))
    return out


def canon_card(card: Q.CardState) -> dict[str, Any]:
    return {f.name: _value(getattr(card, f.name)) for f in dataclasses.fields(card)}


def canon_chain(link: Q.ChainLink) -> dict[str, Any]:
    return {f.name: _value(getattr(link, f.name)) for f in dataclasses.fields(link)}


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
