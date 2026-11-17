"""Python oracle for the A2b-3a stream board tracker.

    uv run python scripts/app/stream_dump.py --packets data/cache/app/stream_packets.jsonl.gz \\
        --out data/cache/app/stream_py.jsonl.gz

Reads the per-seat packet streams produced by the Rust `stream_frames` binary (START, filtered
messages, before/after refresh packets) and feeds Python `StreamState`. After every packet it writes
the same JSON record the Rust binary logs: every `query_location`, `field_state`, turn/phase/players,
hand-memory state and the history entries that packet produced. `stream_compare.py` compares the two
files exactly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCATIONS = (
    0x1,  # DECK
    0x2,  # HAND
    0x4,  # MZONE
    0x8,  # SZONE
    0x10,  # GRAVE
    0x20,  # REMOVED
    0x40,  # EXTRA
)


def card_json(c) -> dict:
    return {
        "controller": c.controller,
        "location": c.location,
        "sequence": c.sequence,
        "code": c.code,
        "position": c.position,
        "type": c.type,
        "level": c.level,
        "rank": c.rank,
        "attribute": c.attribute,
        "race": c.race,
        "attack": c.attack,
        "defense": c.defense,
        "base_attack": c.base_attack,
        "base_defense": c.base_defense,
        "owner": c.owner,
        "status": c.status,
        "public": c.public,
        "lscale": c.lscale,
        "rscale": c.rscale,
        "link": c.link,
        "link_marker": c.link_marker,
        "overlay": list(c.overlay),
        "overlay_index": c.overlay_index,
    }


def chain_json(c) -> dict:
    return {
        "code": c.code,
        "controller": c.controller,
        "location": c.location,
        "sequence": c.sequence,
        "position": c.position,
        "triggering_controller": c.triggering_controller,
        "triggering_location": c.triggering_location,
        "triggering_sequence": c.triggering_sequence,
        "description": c.description,
    }


def hist_json(h) -> dict:
    return {"row": list(h.row), "actor": h.actor, "turn": h.turn, "phase": h.phase, "public": bool(h.public)}


def state_json(g: int, seat: int, i: int, st, hist) -> dict:
    loc = []
    for p in (0, 1):
        for location in LOCATIONS:
            cards = [None if c is None else card_json(c) for c in st.query_location(p, location)]
            loc.append([p, location, cards])
    (lp0, lp1), chain = st.field_state()
    return {
        "k": "state",
        "g": g,
        "s": seat,
        "i": i,
        "turn": st.turn,
        "phase": st.phase,
        "first_player": st.first_player,
        "turn_player": st.turn_player,
        "field": {"lp": [lp0, lp1], "chain": [chain_json(c) for c in chain]},
        "loc": loc,
        "memory": {
            "known": [list(st.memory.known[0]), list(st.memory.known[1])],
            "codes": [list(st.memory.codes[0]), list(st.memory.codes[1])],
        },
        "hist": [hist_json(h) for h in hist],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    from decklight.engine.resources import load_cards
    from decklight.play.stream import StreamState, history_from_message

    cards = load_cards()
    states: dict[tuple[int, int], object] = {}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_packets = 0
    n_streams = 0
    with gzip.open(a.packets, "rt", encoding="utf-8") as fh, gzip.open(a.out, "wt", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            k = rec.get("k")
            if k == "stream":
                key = (rec["g"], rec["s"])
                states[key] = StreamState(list(rec["main"]), list(rec["extra"]), cards, int(rec["s"]))
                n_streams += 1
            elif k == "pkt":
                key = (rec["g"], rec["s"])
                st = states.get(key)
                if st is None:
                    print(f"ALERT stream_dump: packet before stream header {key}", file=sys.stderr)
                    return 2
                raw = bytes.fromhex(rec["msg"])
                st.feed(raw)
                hist = history_from_message(st, raw)
                out.write(
                    json.dumps(state_json(rec["g"], rec["s"], rec["i"], st, hist), separators=(",", ":"))
                    + "\n"
                )
                n_packets += 1
            else:
                print(f"ALERT stream_dump: unknown record {k}", file=sys.stderr)
                return 2
    print(json.dumps({"streams": n_streams, "packets": n_packets, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
