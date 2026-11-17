"""Generate `rust/decklight-edopro/tests/fixtures/protocol.json` from the Python oracle.

    uv run python scripts/app/gen_edopro_fixtures.py

Every fixture is the Python `protocol.py` output (hex) plus the inputs, so the Rust unit test can
assert both directions. Run this after changing `protocol.py`; do not edit the JSON by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from decklight.edopro import protocol as P

OUT = (
    Path(__file__).resolve().parents[2] / "rust" / "decklight-edopro" / "tests" / "fixtures" / "protocol.json"
)
YRPX_OUT = OUT.with_name("yrpx.json")


def hx(b: bytes) -> str:
    return b.hex()


def hostinfo_fields(h: P.HostInfo) -> dict:
    return {
        "lflist": h.lflist,
        "rule": h.rule,
        "mode": h.mode,
        "duel_rule": h.duel_rule,
        "no_check_deck_content": h.no_check_deck_content,
        "no_shuffle_deck": h.no_shuffle_deck,
        "start_lp": h.start_lp,
        "start_hand": h.start_hand,
        "draw_count": h.draw_count,
        "time_limit": h.time_limit,
        "duel_flag_high": h.duel_flag_high,
        "handshake": h.handshake,
        "version": list(h.version),
        "team1": h.team1,
        "team2": h.team2,
        "best_of": h.best_of,
        "duel_flag_low": h.duel_flag_low,
        "forbiddentypes": h.forbiddentypes,
        "extra_rules": h.extra_rules,
        "main_min": h.main_min,
        "main_max": h.main_max,
        "extra_min": h.extra_min,
        "extra_max": h.extra_max,
        "side_min": h.side_min,
        "side_max": h.side_max,
    }


def main() -> int:
    infos = [
        P.HostInfo(),
        P.HostInfo(
            rule=4,
            start_lp=4000,
            main_min=40,
            main_max=60,
            side_max=15,
            time_limit=300,
            duel_flag_high=0x1234,
            team1=2,
            team2=1,
            best_of=3,
        ),
        P.HostInfo(
            lflist=0xDEADBEEF,
            mode=1,
            duel_rule=3,
            no_check_deck_content=1,
            no_shuffle_deck=1,
            start_hand=4,
            draw_count=2,
            forbiddentypes=0x7F,
            extra_rules=5,
            extra_min=1,
            extra_max=10,
            side_min=2,
        ),
    ]
    fixtures = {
        "hostinfo": [{"hex": hx(h.pack()), "fields": hostinfo_fields(h)} for h in infos],
        "join_game": [
            {
                "args": [0x1354, 0x11223344, [41, 0, 11, 0], ""],
                "hex": hx(P.pack_join_game(0x1354, 0x11223344, bytes([41, 0, 11, 0]))),
            },
            {
                "args": [0x1234, 7, [1, 2, 3, 4], "pw"],
                "hex": hx(P.pack_join_game(0x1234, 7, bytes([1, 2, 3, 4]), "pw")),
            },
        ],
        "create_game": [
            {"args": [0, "", "", ""], "hex": hx(P.pack_create_game(infos[0]))},
            {
                "args": [1, "room", "pw", "notes"],
                "hex": hx(P.pack_create_game(infos[1], "room", "pw", "notes")),
            },
        ],
        "hs_player_enter": [
            {"args": ["", 0], "hex": hx(P.pack_hs_player_enter("", 0))},
            {"args": ["Alice", 1], "hex": hx(P.pack_hs_player_enter("Alice", 1))},
            {"args": ["üñïçødé", 2], "hex": hx(P.pack_hs_player_enter("üñïçødé", 2))},
        ],
        "player_info": [{"args": [n], "hex": hx(P.pack_player_info(n))} for n in ["", "Bob", "a" * 30]],
        "update_deck": [
            {"args": [[1, 2, 3], [4, 5], [6]], "hex": hx(P.pack_update_deck([1, 2, 3], [4, 5], [6]))},
            {"args": [[], [], []], "hex": hx(P.pack_update_deck([], [], []))},
        ],
        "hand_choice": [{"args": [n], "hex": hx(P.pack_hand_choice(n))} for n in (1, 2, 3)],
        "tp_choice": [{"args": [b], "hex": hx(P.pack_tp_choice(b))} for b in (True, False)],
        "hand_result": [{"args": [1, 3], "hex": hx(P.pack_hand_result(1, 3))}],
        "deck_error": [
            {"args": [4, 39, 40, 60, 12345678], "hex": hx(P.pack_deck_error(4, 39, 40, 60, 12345678))},
            {"args": [0, 0, 0, 0, 0], "hex": hx(P.pack_deck_error(0))},
        ],
        "join_error": [{"args": [2], "hex": hx(P.pack_join_error(2))}],
        "start": [
            {"args": [8000, 40, 15, 40, 15, 0], "hex": hx(P.build_start(8000, 40, 15, 40, 15, 0))},
            {"args": [4000, 1, 2, 3, 4, 1], "hex": hx(P.build_start(4000, 1, 2, 3, 4, 1))},
        ],
        "frame": [
            {"args": [1, ""], "hex": hx(P.pack_frame(1))},
            {"args": [0x30, "aabb"], "hex": hx(P.pack_frame(0x30, bytes.fromhex("aabb")))},
        ],
        "parse_type_change": [
            {"hex": "05", "want": list(P.parse_type_change(bytes([0x05])))},
            {"hex": "a3", "want": list(P.parse_type_change(bytes([0xA3])))},
        ],
        "parse_hs_player_change": [
            {"hex": "09", "want": list(P.parse_hs_player_change(bytes([0x09])))},
            {"hex": "3a", "want": list(P.parse_hs_player_change(bytes([0x3A])))},
            {"hex": "7b", "want": list(P.parse_hs_player_change(bytes([0x7B])))},
        ],
        "parse_time_limit": [
            {"hex": "01002c01", "want": list(P.parse_time_limit(bytes([1, 0, 0x2C, 0x01])))},
            {"hex": "00000000", "want": list(P.parse_time_limit(bytes([0, 0, 0, 0])))},
        ],
        "parse_hand_result": [{"hex": "0103", "want": list(P.parse_hand_result(bytes([1, 3])))}],
        "parse_tp_result": [
            {"hex": "01", "want": P.parse_tp_result(b"\x01")},
            {"hex": "00", "want": P.parse_tp_result(b"\x00")},
        ],
        "parse_create_game": [
            {
                "hex": hx(P.pack_frame(0x11, __import__("struct").pack("<I", 12345))),
                "want": P.parse_create_game(__import__("struct").pack("<I", 12345)),
            }
        ],
        "error_msg": [
            {"hex": hx(P.pack_join_error(1)), "text": P.parse_error_msg(P.pack_join_error(1))},
            {
                "hex": hx(P.pack_deck_error(4, 39, 40, 60, 12345678)),
                "text": P.parse_error_msg(P.pack_deck_error(4, 39, 40, 60, 12345678)),
            },
            {"hex": hx(P.pack_deck_error(2)), "text": P.parse_error_msg(P.pack_deck_error(2))},
            {"hex": "", "text": P.parse_error_msg(b"")},
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

    # A real LZMA-compressed .yrpX from the Python writer, for the Rust parser (Multirole layout).
    from decklight.play import yrp as Y

    y = Y.Yrp(
        seed=(1, 2, 3, 4),
        names=[["alice"], ["bob"]],
        lp=8000,
        hand=5,
        draw=1,
        flags=0x2E800,
        decks=[[([10, 11, 12], [20])], [([13, 14], [21])]],
        extra_cards=[],
        responses=[bytes([1, 0, 0, 0]), bytes([2, 0, 0, 0, 1, 1])],
    )
    data = Y.write_yrpx(y, [bytes([0x5A, 1, 2, 3])], ts=1234, compress=True)
    parsed = Y.parse_yrpx(data)
    yrpx = {
        "hex": hx(data),
        "timestamp": parsed.timestamp,
        "names": parsed.names,
        "flags": str(parsed.flags),
        "seed": [str(w) for w in parsed.yrp.seed],
        "lp": parsed.yrp.lp,
        "hand": parsed.yrp.hand,
        "draw": parsed.yrp.draw,
        "decks": [[[list(m), list(e)] for m, e in team] for team in parsed.yrp.decks],
        "responses": [hx(r) for r in parsed.yrp.responses],
        "core_messages": [hx(m) for m in Y.core_messages(parsed)],
    }
    YRPX_OUT.write_text(json.dumps(yrpx, indent=2, sort_keys=True) + "\n")
    print(f"wrote {YRPX_OUT} ({YRPX_OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
