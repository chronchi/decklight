"""Parity corpus for the Rust ports of the message/query parsers (A1b).

    uv run python scripts/app/parse_corpus.py --games N --seed S --choice-seed C \
        --pairs A:B C:D ... --out data/cache/app/corpus.jsonl

Plays `PyEnv(..., auto_single=False)` games with the splitmix64 choice stream of
`scripts/app/duel_hash.py` (deck pairs from `decks/pool/` round-robin when `--pairs` is omitted)
and writes, with no spaces:
- one `{"k":"msg","raw":hex,"canon":canon_msg(parse_one(raw))}` per raw trace message;
- at every decision one `{"k":"field","raw":hex(query_field),"chain":[canon_chain...],"lps":[lp0,lp1]}`;
- for players 0/1 x QUERY_LOCATIONS one `{"k":"loc","con",...,"cards":[canon_card or null...]}`;
- one `{"k":"split","buf":hex,"out":[hex...]}` per non-empty `OCG_DuelGetMessage` buffer.

The split-buffer capture monkey-patches `decklight.engine.messages.split_buffer` (the duel calls
the module attribute); `src/` is not modified.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.app.canon import canon_card, canon_chain, canon_msg, dumps
from scripts.app.duel_hash import SplitMix
from scripts.app.play_dump import resolve_deck


def raw_query_location(duel, controller: int, location: int, flags: int) -> bytes:
    from decklight.engine import ocgapi as api

    info = api.OCG_QueryInfo(flags, controller, location, 0, 0)
    length = ctypes.c_uint32()
    ptr = duel.engine.lib.OCG_DuelQueryLocation(duel._handle, ctypes.byref(length), ctypes.byref(info))
    if not ptr or not length.value:
        return b""
    return ctypes.string_at(ptr, length.value)


def round_robin_pairs() -> list[tuple[Path, Path]]:
    decks = sorted((ROOT / "decks" / "pool").glob("*.ydk"))
    if len(decks) < 2:
        raise SystemExit("need at least two decks in decks/pool/")
    return [(decks[i], decks[(i + 1) % len(decks)]) for i in range(len(decks))]


def run(args) -> dict[str, int]:
    from decklight.decks import load_ydk
    from decklight.engine import messages as M
    from decklight.engine import query as Q
    from decklight.engine.duel import DuelConfig, Engine
    from decklight.engine.env import PyEnv

    if args.pairs:
        pairs = [(resolve_deck(a), resolve_deck(b)) for a, b in (p.split(":", 1) for p in args.pairs)]
    else:
        pairs = round_robin_pairs()
    if args.games > 0 and not pairs:
        raise SystemExit("no deck pairs")
    decks = [(load_ydk(a), load_ydk(b)) for a, b in pairs]
    print(f"pairs ({len(pairs)}): " + ", ".join(f"{a.name}:{b.name}" for a, b in pairs), file=sys.stderr)

    split_capture: list[bytes] = []
    original_split = M.split_buffer

    def capturing_split(buf: bytes):
        if buf:
            split_capture.append(bytes(buf))
        return original_split(buf)

    M.split_buffer = capturing_split

    engine = Engine()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"games": 0, "decisions": 0, "msgs": 0, "fields": 0, "locs": 0, "splits": 0}
    emitted = 0

    def w(row: dict) -> None:
        fh.write(dumps(row) + "\n")

    with out.open("w", encoding="utf-8") as fh:
        for g in range(args.games):
            d0, d1 = decks[g % len(decks)]
            seed = args.seed + g
            env = PyEnv(engine, DuelConfig(d0, d1, seed=seed), auto_single=False)
            split_capture.clear()
            env.reset()
            rng = SplitMix(args.choice_seed ^ g)
            emitted = 0
            steps = 0
            while env.dec is not None and steps < args.max_decisions:
                duel = env.duel
                while emitted < len(duel.trace):
                    raw = duel.trace[emitted]
                    emitted += 1
                    try:
                        canon = canon_msg(M.parse_one(raw))
                    except Exception as e:  # fail loud: a corpus line we cannot parse is a bug
                        raise RuntimeError(
                            f"game {g} msg {emitted - 1} t={raw[0] if raw else None}: {e}"
                        ) from e
                    w({"k": "msg", "raw": raw.hex(), "canon": canon})
                    counts["msgs"] += 1
                field = duel.query_field()
                w(
                    {
                        "k": "field",
                        "raw": field.hex(),
                        "chain": [canon_chain(c) for c in Q.parse_chain(field)],
                        "lps": list(Q.lps(field)),
                    }
                )
                counts["fields"] += 1
                for p in (0, 1):
                    for loc in Q.QUERY_LOCATIONS:
                        raw = raw_query_location(duel, p, loc, Q.CARD_FLAGS)
                        cards = Q.parse_location(raw, p, loc)
                        w(
                            {
                                "k": "loc",
                                "con": p,
                                "loc": loc,
                                "raw": raw.hex(),
                                "cards": [None if c is None else canon_card(c) for c in cards],
                            }
                        )
                        counts["locs"] += 1
                n = len(env.opts)
                if n <= 0:
                    raise RuntimeError(f"game {g} decision {steps}: no legal option")
                idx = int(rng.next() % n)
                env.step(idx)
                steps += 1
                counts["decisions"] += 1
            while emitted < len(env.duel.trace):
                raw = env.duel.trace[emitted]
                emitted += 1
                w({"k": "msg", "raw": raw.hex(), "canon": canon_msg(M.parse_one(raw))})
                counts["msgs"] += 1
            for buf in split_capture:
                w({"k": "split", "buf": buf.hex(), "out": [m.hex() for m in original_split(buf)]})
                counts["splits"] += 1
            split_capture.clear()
            env.duel.close()
            counts["games"] += 1
            print(
                f"game {g + 1}/{args.games}: decisions {steps} msgs {counts['msgs']} locs {counts['locs']}",
                file=sys.stderr,
            )
    M.split_buffer = original_split
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice-seed", type=int, default=0)
    ap.add_argument("--pairs", nargs="*", default=None, help="deck0:deck1 entries (names from decks/pool)")
    ap.add_argument("--max-decisions", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "app" / "corpus.jsonl")
    a = ap.parse_args(argv)
    counts = run(a)
    print(json.dumps(counts, separators=(",", ":")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
