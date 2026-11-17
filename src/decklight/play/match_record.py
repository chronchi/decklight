"""Match record v1 (ADR 0010, schema `docs/design/schemas/match-record-v1.json`): one self-describing JSON per duel.

The replay key is the core's own input: the 4-word seed state, each main deck in the order it was added to the core,
and every raw response. That is what EDOPro/Multirole `.yrp` replays store too, so records from our engine and from
other servers replay the same way, independent of our env's option ordering. This module is the reference the Rust
writer is checked against.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from decklight.decks.ydk import Deck
from decklight.engine.duel import Duel, DuelConfig, Engine, seed_words

FORMAT = "decklight.match/1"
ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "match-record-v1.json"
DERIVED = ("events", "annotations", "transcript")


def uuid7(ms: int | None = None) -> str:
    """RFC 9562 UUIDv7: 48-bit Unix ms, version 7, 74 random bits."""
    ms = int(time.time() * 1000) if ms is None else ms
    b = bytearray(ms.to_bytes(6, "big") + secrets.token_bytes(10))
    b[6] = 0x70 | (b[6] & 0x0F)
    b[8] = 0x80 | (b[8] & 0x3F)
    h = b.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _git_head(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dev_pack() -> dict:
    """The card set of a source checkout, until card packs exist (A5): CardScripts + BabelCDB commits."""
    tp = ROOT / "third_party"
    scripts, cdb = _git_head(tp / "CardScripts"), _git_head(tp / "BabelCDB")
    return {
        "pack": f"dev-{scripts[:8]}-{cdb[:8]}",
        "sha256": hashlib.sha256(f"{scripts}\n{cdb}\n".encode()).hexdigest(),
    }


def engine_info() -> dict:
    tp = ROOT / "third_party"
    patches = sorted((tp / "patches").glob("ygopro-core-*.patch"))
    h = hashlib.sha256()
    for p in patches:
        h.update(p.read_bytes())
    return {"core": _git_head(tp / "ygopro-core"), "patches": h.hexdigest()}


def deck_hash(main, extra) -> str:
    return hashlib.sha256(",".join(map(str, sorted(main) + [0] + sorted(extra))).encode()).hexdigest()


def from_duel(
    duel: Duel,
    *,
    seats: list[dict],
    mode: str,
    app: str,
    surrendered: bool = False,
    disconnected: bool = False,
    opts: list[list[int] | None] | None = None,
    ms: list[int] | None = None,
    actors: list[int] | None = None,
    names: tuple[str, str] = ("", ""),
    sides: tuple[tuple[int, ...], tuple[int, ...]] = ((), ()),
    created: str | None = None,
    record_id: str | None = None,
    cards: dict | None = None,
    meta: dict | None = None,
) -> dict:
    """A `full` record of a duel our engine ran. `actors[i]` is the seat of response i (required: the core's
    responses do not say who sent them); `opts[i]` the env option indices that produced it, if known."""
    cfg = duel.config
    n = len(duel.responses)
    if actors is None or len(actors) != n:
        raise ValueError(f"actors must give the seat of each of the {n} responses")
    if cfg.setup:
        raise ValueError("puzzle setups are not in match record v1 yet")
    state = cfg.seed_state if cfg.seed_state is not None else tuple(seed_words(cfg.seed))
    decks = []
    for team, deck in ((0, cfg.deck0), (1, cfg.deck1)):
        main, extra = cfg.main_of(team), list(deck.extra)
        d = {"main": main, "extra": extra, "side": list(sides[team]), "hash": deck_hash(main, extra)}
        nm = names[team] or getattr(deck, "name", "")
        if nm:
            d["name"] = nm
        decks.append(d)
    responses = []
    for i, r in enumerate(duel.responses):
        e = {"seat": int(actors[i]), "r": r.hex(), "opt": None if opts is None else opts[i]}
        if ms is not None:
            e["ms"] = int(ms[i])
        responses.append(e)
    winner = duel.winner if duel.winner in (0, 1) else None
    rec = {
        "format": FORMAT,
        "id": record_id or uuid7(),
        "created": created or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "app": app,
        "engine": engine_info(),
        "cards": cards or dev_pack(),
        "rules": {
            "duel_flags": str(cfg.flags),
            "start_lp": cfg.lp,
            "start_hand": cfg.hand,
            "draw_count": cfg.draw,
        },
        "kind": "full",
        "engine_seed": [str(w) for w in state],
        "decks": decks,
        "seats": seats,
        "mode": mode,
        "responses": responses,
        "result": {
            "winner": winner,
            "reason": int(duel.win_reason),
            "turns": int(duel.turn),
            "surrendered": surrendered,
            "disconnected": disconnected,
        },
        "source": "engine",
    }
    if cfg.seed_state is None:
        rec["seed"] = str(cfg.seed)
    if tuple(cfg.team_order) != (0, 1):
        rec["add_order"] = list(cfg.team_order)
    if cfg.edopro_add:
        rec["deck_add"] = "edopro"
    if duel.first_player is not None:
        rec["first"] = int(duel.first_player)
    for k in ("hints_used", "undos", "what_if", "puzzle", "puzzle_result", "practice"):
        if meta and k in meta:
            rec[k] = meta[k]
    return rec


def config_of(rec: dict) -> DuelConfig:
    """The DuelConfig that replays `rec` exactly: the stored seed state and deck order, no reshuffle."""
    if rec.get("kind") != "full":
        raise ValueError(f"only full records replay (kind={rec.get('kind')!r})")
    d0, d1 = (Deck(tuple(d["main"]), tuple(d["extra"]), tuple(d.get("side", ()))) for d in rec["decks"])
    r = rec["rules"]
    if rec["rules"].get("extra_cards"):
        raise ValueError("extra-rule cards (EDOPro extra rules) are not supported by our engine yet")
    order = tuple(rec.get("add_order", (0, 1)))
    if rec.get("deck_add") not in (None, "edopro"):
        raise ValueError(f"unknown deck_add {rec['deck_add']!r}")
    return DuelConfig(
        d0,
        d1,
        seed=0,
        flags=int(r["duel_flags"]),
        lp=r["start_lp"],
        hand=r["start_hand"],
        draw=r["draw_count"],
        shuffle=False,
        seed_state=tuple(int(w) for w in rec["engine_seed"]),
        team_order=order,
        edopro_add=rec.get("deck_add") == "edopro",
    )


def replay(rec: dict, engine: Engine) -> Duel:
    """Rebuild the duel and play every response through the core (raises InvalidResponse on a bad record)."""
    return Duel.replay(engine, config_of(rec), [bytes.fromhex(e["r"]) for e in rec["responses"]])


def check(rec: dict, engine: Engine) -> list[str]:
    """Problems found by replaying `rec` (empty = it reproduces). The intake job rejects any record with problems."""
    problems = []
    try:
        duel = replay(rec, engine)
    except Exception as e:  # noqa: BLE001 - every failure is a rejection reason, reported, never raised
        return [f"replay failed: {type(e).__name__}: {e}"]
    res = rec["result"]
    finished = not (res.get("surrendered") or res.get("disconnected"))
    want = res["winner"]
    got = duel.winner if duel.winner in (0, 1) else None
    if finished and got != want:
        problems.append(f"winner {got} != recorded {want}")
    if finished and duel.winner is None:
        problems.append("replay did not reach the end of the duel")
    if int(duel.turn) != int(res["turns"]):
        problems.append(f"turns {duel.turn} != recorded {res['turns']}")
    return problems


def validate(rec: dict) -> list[str]:
    """JSON Schema errors (needs the `jsonschema` dev dependency)."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    v = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(rec)]


def shareable(rec: dict) -> dict:
    """The record as shared to the open dataset: derived sections and deck names dropped (ADR 0010 §4)."""
    out = {k: v for k, v in rec.items() if k not in DERIVED}
    out["decks"] = [{k: v for k, v in d.items() if k != "name"} for d in rec["decks"]]
    return out


def save(rec: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{rec['id']}.match.json.gz"
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} record (format={rec.get('format')!r})")
    return rec
