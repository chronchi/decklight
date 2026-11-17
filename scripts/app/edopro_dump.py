"""A2b-1 parity dump: protocol/filter/inverse/decisions/.yrpX over played games.

    uv run python scripts/app/edopro_dump.py --pairs A:B C:D --games 60 --seed 51000 \
        --choice-seed 13 --out data/cache/app/edopro-py.jsonl.gz

Gzip-streamed JSON lines (one per trace message, decision and game), matching
`rust/decklight-edopro/src/bin/edopro_dump.rs` line for line. `scripts/app/edopro_compare.py A B`
compares them (exit 2 + ALERT on any difference). The choice stream is the splitmix64 of
`scripts/app/play_dump.py` (seeded by C^g, index = next() % num_options, one pick at a time).

Trace messages are emitted at the next decision (the env exposes the trace only between decisions);
the refresh packets are therefore queried at that point, identically on both sides. Hint runs are
canonicalised (`Duel.canonical_trace`) so heap-order differences cannot show up as parity failures.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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


def hx(b: bytes | None) -> str | None:
    return None if b is None else b.hex()


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


def refresh_json(ref: tuple) -> list:
    return list(ref)


def _choose(msg, cards, pool, seq) -> bytes | None:
    """Response of a fresh standalone decision after choosing `seq`."""
    from decklight.engine import actions as A

    dec, out = A.make_decision(msg, cards, lambda: pool), None
    for i in seq:
        r = dec.choose(i)
        if r is not None:
            out = bytes(r)
    return out


def emit_trace(rows: list[dict], duel, env, start: int, pools: list[set[int]], cards) -> tuple[int, int]:
    """Emit message lines for trace[start:]; returns (new_start, refresh_packets)."""
    from decklight.edopro import filter as F
    from decklight.engine.duel import canonical_trace

    trace = canonical_trace(duel.trace)
    n_pkts = 0
    for i in range(start, len(trace)):
        raw = trace[i]
        # DecisionOnlyView pool: our deck + every card code on the wire for our seat.
        if raw and raw[0] not in (0x05, 0x01):  # MSG_WIN, MSG_RETRY
            for viewer in (0, 1):
                seen = F.filtered_message(raw, viewer)
                if seen:
                    for j in range(len(seen) - 3):
                        code = int.from_bytes(seen[j : j + 4], "little")
                        if code in cards:
                            pools[viewer].add(code)
        before = F.before_refreshes(raw)
        after = F.after_refreshes(raw)
        before_pkts, after_pkts = [], []
        for ref in before:
            before_pkts.append([hx(F.refresh_packet(duel, ref, v)) for v in (0, 1)])
        for ref in after:
            after_pkts.append([hx(F.refresh_packet(duel, ref, v)) for v in (0, 1)])
        n_pkts += len(before_pkts) + len(after_pkts)
        rows.append(
            {
                "k": "msg",
                "g": -1,
                "i": i,
                "raw": raw.hex(),
                "f0": hx(F.filtered_message(raw, 0)),
                "f1": hx(F.filtered_message(raw, 1)),
                "wait": F.waiting_player(raw),
                "before": [refresh_json(r) for r in before],
                "after": [refresh_json(r) for r in after],
                "before_pkts": before_pkts,
                "after_pkts": after_pkts,
            }
        )
    return len(trace), n_pkts


def game_rows(engine, deck0, deck1, g: int, seed: int, cseed: int) -> tuple[list[dict], dict]:
    from decklight.engine import actions as A
    from decklight.engine import constants as C
    from decklight.engine import inverse as INV
    from decklight.engine.duel import DuelConfig, canonical_trace, seed_words
    from decklight.engine.env import PyEnv
    from decklight.play import match_record as MR
    from decklight.play import yrp as Y

    order = (0, 1) if g % 2 == 0 else (1, 0)
    s = seed + g
    cfg = DuelConfig(deck0, deck1, seed=s, team_order=order)
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    rng = SplitMix(cseed ^ g)
    rows: list[dict] = []
    pools = [set(deck0.main) | set(deck0.extra), set(deck1.main) | set(deck1.extra)]
    last, n_pkts = emit_trace(rows, env.duel, env, 0, pools, engine.cards)
    d = 0

    class _Shim:
        auto_single = False

        def __init__(self, dec):
            self.dec = dec

        def pending_decision(self):
            return self.dec

    while env.dec is not None:
        dec = env.dec
        dec_start = INV._clone(dec)
        raw = bytes(dec.msg.raw)
        player = int(dec.msg.player)
        pool = sorted(pools[player])
        nres0 = len(env.duel.responses)
        seq: list[int] = []
        resp: bytes | None = None
        while resp is None:
            n = len(env.opts)
            if n == 0:
                raise SystemExit(f"ALERT edopro_dump: game {g} decision {d}: no legal option")
            idx = int(rng.next() % n)
            seq.append(idx)
            env.step(idx)
            if len(env.duel.responses) > nres0:
                resp = env.duel.responses[nres0]
            if env.dec is None and resp is None:
                raise SystemExit(f"ALERT edopro_dump: game {g} decision {d}: ended without a response")
        inv = INV.choices_for_response(_Shim(dec_start), resp)
        if inv is None:
            raise SystemExit(f"ALERT edopro_dump: game {g} decision {d}: response not expressible")
        rows.append(
            {
                "k": "inv",
                "g": g,
                "d": d,
                "msg": raw.hex(),
                "resp": resp.hex(),
                "seq": None if inv is None else [int(x) for x in inv],
            }
        )
        stand = A.make_decision(dec.msg, engine.cards, lambda p=pool: p)
        opts = []
        for o in stand.options():
            if o.card is None:
                opts.append([int(o.kind), 0, 0, 0, 0, 0, 0, int(o.code), int(o.desc), int(o.arg)])
            else:
                c = o.card
                opts.append(
                    [
                        int(o.kind),
                        1,
                        int(c.code),
                        int(c.controller),
                        int(c.location),
                        int(c.sequence),
                        int(c.position),
                        int(o.code),
                        int(o.desc),
                        int(o.arg),
                    ]
                )
        # ANNOUNCE_CARD: the client's pool (deck + codes seen on the wire) differs from the env's known codes by
        # design, so option indices differ; the client must still offer the declared code.
        stand_seq, stand_inv = seq, inv
        if raw[0] == C.MSG_ANNOUNCE_CARD:
            j = next((j for j in range(len(opts)) if _choose(dec.msg, engine.cards, pool, [j]) == resp), None)
            if j is None:
                raise SystemExit(f"ALERT edopro_dump: game {g} decision {d}: declared code not in the client pool")
            stand_seq = stand_inv = [j]
        stand_resp = None
        for i in stand_seq:
            r = stand.choose(i)
            if r is not None:
                stand_resp = bytes(r)
        if stand_resp != resp:
            raise SystemExit(f"ALERT edopro_dump: game {g} decision {d}: standalone response differs")
        check = A.make_decision(dec.msg, engine.cards, lambda p=pool: p)
        check_resp = None
        for i in stand_inv:
            r = check.choose(i)
            if r is not None:
                check_resp = bytes(r)
        if check_resp != resp:
            raise SystemExit(f"ALERT edopro_dump: game {g} decision {d}: inverse sequence response differs")
        rows.append(
            {
                "k": "dec",
                "g": g,
                "d": d,
                "msg": raw.hex(),
                "player": player,
                "opts": opts,
                "seq": seq,
                "stand_seq": stand_seq,
                "resp": hx(stand_resp),
            }
        )
        d += 1
        last, k = emit_trace(rows, env.duel, env, last, pools, engine.cards)
        n_pkts += k

    last, k = emit_trace(rows, env.duel, env, last, pools, engine.cards)
    n_pkts += k
    for r in rows:
        r["g"] = g

    # .yrpX built from the game (uncompressed: the LZMA encoder is not the oracle here).
    duel = env.duel
    y = Y.Yrp(
        seed=tuple(seed_words(s)),
        names=[["alice"], ["bob"]],
        lp=cfg.lp,
        hand=cfg.hand,
        draw=cfg.draw,
        flags=cfg.flags,
        decks=Y.yrp_decks(cfg),
        extra_cards=[],
        responses=list(duel.responses),
    )
    # Hint runs follow heap order (NOTEBOOK 2026-10-01): the replay carries the canonical trace, and to_record
    # compares canonical traces on both sides.
    data = Y.write_yrpx(y, canonical_trace(duel.trace), ts=1_790_000_000, compress=False)
    rid = f"00000000-0000-7000-8000-{g:012x}"
    rec, problems = Y.to_record(
        Y.parse_yrpx(data), engine, viewer=0, app="decklight-edopro-dump/0", record_id=rid
    )
    check = [] if rec is None else MR.check(rec, engine)
    game = {
        "k": "game",
        "g": g,
        "order": list(order),
        "yrpx": Y.digest(data),
        "problems": problems,
        "add_order": None if rec is None else list(rec.get("add_order", (0, 1))),
        "winner": None if rec is None else rec["result"]["winner"],
        "turns": None if rec is None else rec["result"]["turns"],
        "seats": None if rec is None else rec["seats"],
        "rseats": None if rec is None else [e["seat"] for e in rec["responses"]],
        "rec": rec,
        "check": check,
        "decisions": d,
    }
    rows.append(game)
    duel.close()
    return rows, {"msgs": last, "refreshes": n_pkts, "decisions": d, "games": 1}


def run(args) -> tuple[list[dict], dict]:
    from decklight.decks import load_ydk
    from decklight.engine.duel import Engine

    if args.deck0 and args.deck1:
        pairs = [(Path(args.deck0), Path(args.deck1))]
    elif args.pairs:
        pairs = parse_pairs(args.pairs)
    else:
        raise SystemExit("--deck0/--deck1 or --pairs is required")
    decks = [(load_ydk(a), load_ydk(b)) for a, b in pairs]
    engine = Engine()
    if not engine.cards:
        raise SystemExit("ALERT edopro_dump: no cards loaded")
    rows: list[dict] = []
    totals = {"msgs": 0, "refreshes": 0, "decisions": 0, "games": 0}
    offset = int(getattr(args, "game_offset", 0) or 0)
    for i in range(args.games):
        g = offset + i
        d0, d1 = decks[g % len(decks)]
        rs, t = game_rows(engine, d0, d1, g, args.seed, args.choice_seed)
        rows.extend(rs)
        for k in totals:
            totals[k] += t[k]
    totals["deck_pairs"] = list(args.pairs) if args.pairs else []
    return rows, totals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck0", type=Path, default=None)
    ap.add_argument("--deck1", type=Path, default=None)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice-seed", type=int, default=0)
    ap.add_argument("--game-offset", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, totals = run(a)
    with gzip.open(a.out, "wt", compresslevel=6) as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(json.dumps({"totals": totals, "out": str(a.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
