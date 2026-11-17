"""Decision-level reference dump for the A1a play layer (PyEnv, auto_single=False).

    uv run python scripts/app/play_dump.py --deck0 A.ydk --deck1 B.ydk --games N --seed S --choice-seed C
    uv run python scripts/app/play_dump.py --pairs A:B C:D --games N --seed S --choice-seed C
    uv run python scripts/app/play_dump.py ... --compare OTHER.jsonl   # exit 2 on any mismatch

One JSON line per decision, then one end line per game (see the brief). The choice stream is the
same splitmix64 as rust/duel-cli (seeded by C^g, index = next() % num_options). Decks for --pairs
are looked up as paths first, then as decks/pool/<name>.ydk.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import struct
import sys
from pathlib import Path

M64 = (1 << 64) - 1


class SplitMix:
    def __init__(self, s: int) -> None:
        self.s = s & M64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def sha_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve_deck(s: str) -> Path:
    p = Path(s)
    if p.exists():
        return p
    pool = Path(__file__).resolve().parents[2] / "decks" / "pool" / f"{s}.ydk"
    if pool.exists():
        return pool
    return p


def parse_pairs(items: list[str]) -> list[tuple[Path, Path]]:
    out = []
    for it in items:
        a, sep, b = it.partition(":")
        if not sep:
            raise ValueError(f"--pairs entry {it!r} needs deck0:deck1")
        out.append((resolve_deck(a), resolve_deck(b)))
    return out


def raw_query_location(duel, controller: int, location: int, flags: int) -> bytes:
    from decklight.engine import ocgapi as api

    info = api.OCG_QueryInfo(flags, controller, location, 0, 0)
    length = ctypes.c_uint32()
    ptr = duel.engine.lib.OCG_DuelQueryLocation(duel._handle, ctypes.byref(length), ctypes.byref(info))
    if not ptr or not length.value:
        return b""
    return ctypes.string_at(ptr, length.value)


def select_hint_of(trace: list[bytes]) -> int:
    from decklight.engine import constants as C
    from decklight.engine import messages as M

    hint, sel = 0, 0
    for raw in trace:
        if not raw:
            continue
        t = raw[0]
        if t == C.MSG_HINT and len(raw) >= 11 and raw[1] == C.HINT_SELECTMSG:
            (hint,) = struct.unpack_from("<Q", raw, 3)
        elif t in M.SELECT_TYPES:
            sel, hint = hint, 0
    return int(sel)


def run(args) -> list[dict]:
    from decklight.decks import load_ydk
    from decklight.engine import constants as C
    from decklight.engine import query as Q
    from decklight.engine.duel import DuelConfig, Engine, canonical_trace
    from decklight.engine.env import PyEnv

    engine = Engine()
    if args.deck0 and args.deck1:
        pairs = [(Path(args.deck0), Path(args.deck1))]
    elif args.pairs:
        pairs = parse_pairs(args.pairs)
    else:
        raise SystemExit("--deck0/--deck1 or --pairs is required")

    decks = [(load_ydk(a), load_ydk(b)) for a, b in pairs]
    loc_order = [
        C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
        C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA,
    ]
    rows: list[dict] = []
    offset = int(getattr(args, "game_offset", 0) or 0)
    for i in range(args.games):
        g = offset + i
        d0, d1 = decks[g % len(decks)]
        seed = args.seed + g
        cfg = DuelConfig(d0, d1, seed=seed)
        env = PyEnv(engine, cfg, auto_single=False)
        env.reset()
        rng = SplitMix(args.choice_seed ^ g)
        k = 0
        while env.dec is not None:
            dec = env.dec
            duel = env.duel
            msg_type, player, step = dec.msg.type, dec.msg.player, dec.step
            selected = [[c.code, c.controller, c.location, c.sequence, c.position] for c in dec.selected]
            opts = []
            for o in env.opts:
                if o.card is None:
                    opts.append([int(o.kind), 0, 0, 0, 0, 0, 0, int(o.code), int(o.desc), int(o.arg)])
                else:
                    c = o.card
                    opts.append([
                        int(o.kind), 1, int(c.code), int(c.controller), int(c.location),
                        int(c.sequence), int(c.position), int(o.code), int(o.desc), int(o.arg),
                    ])
            pending = sha_hex(bytes(dec.msg.raw))
            turn, turn_player = duel.turn, duel.turn_player
            first_player = duel.first_player if duel.first_player is not None else -1
            phase, hint = duel.phase, select_hint_of(duel.trace)
            trace_len = len(duel.trace)
            canon = canonical_trace(duel.trace)
            trace_hex = sha_hex(b"".join(struct.pack("<I", len(m)) + m for m in canon))
            field_hex = sha_hex(duel.query_field())
            loc_parts = []
            for p in (0, 1):
                for loc in loc_order:
                    raw = raw_query_location(duel, p, loc, Q.CARD_FLAGS)
                    loc_parts.append(struct.pack("<I", len(raw)) + raw)
            locs_hex = sha_hex(b"".join(loc_parts))
            hist = env.history_log()
            hist_hex = sha_hex(hist.tobytes())
            n = len(env.opts)
            assert n > 0, f"game {g} decision {k}: no legal option"
            idx = int(rng.next() % n)
            rows.append({
                "g": g, "k": k, "player": int(player), "msg_type": int(msg_type), "step": int(step),
                "selected": selected, "options": opts, "pending": pending, "turn": int(turn),
                "turn_player": int(turn_player), "first_player": int(first_player), "phase": int(phase),
                "select_hint": int(hint), "trace_len": int(trace_len), "trace": trace_hex,
                "field": field_hex, "locs": locs_hex, "hist": hist_hex, "choice": int(idx),
            })
            env.step(idx)
            k += 1
        winner = duel.winner if duel.winner is not None else -1
        canon = canonical_trace(duel.trace)
        trace_hex = sha_hex(b"".join(struct.pack("<I", len(m)) + m for m in canon))
        rows.append({
            "g": g, "end": True, "winner": int(winner), "win_reason": int(duel.win_reason),
            "choices": [int(c) for c in env.choices], "decisions": int(env.decisions),
            "trace_len": len(duel.trace), "trace": trace_hex,
        })
        duel.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck0", type=Path, default=None)
    ap.add_argument("--deck1", type=Path, default=None)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice-seed", type=int, default=0)
    ap.add_argument("--game-offset", type=int, default=0, help="overall game index of the first game (for split runs)")
    ap.add_argument("--compare", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = run(a)
    for r in rows:
        print(json.dumps(r, separators=(",", ":")))
    if a.compare:
        other_lines = [ln for ln in a.compare.read_text().splitlines() if ln.startswith("{")]
        mine = [json.dumps(r, separators=(",", ":")) for r in rows]
        bad_line, bad_field = -1, ""
        n = max(len(mine), len(other_lines))
        for i in range(n):
            if i >= len(mine) or i >= len(other_lines):
                bad_line = i
                break
            if mine[i] != other_lines[i]:
                bad_line = i
                try:
                    x, y = json.loads(mine[i]), json.loads(other_lines[i])
                    for k in list(x.keys()) + [kk for kk in y if kk not in x]:
                        if x.get(k) != y.get(k):
                            bad_field = str(k)
                            break
                    else:
                        bad_field = "<encoding>"
                except (json.JSONDecodeError, AttributeError):
                    bad_field = "<unparseable>"
                break
        if bad_line >= 0 or len(mine) != len(other_lines):
            print(f"ALERT play_dump: first differing line {bad_line} field {bad_field} "
                  f"(mine {len(mine)} rows, other {len(other_lines)} rows)", file=sys.stderr)
            return 2
        print(f"OK: {len(rows)} lines identical", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
