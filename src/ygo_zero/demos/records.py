"""Verified-record dicts (moved out of scripts/verify_lines.py).

`make_record` builds exactly the record dict `verify_all` used to write, so
`verify_lines.py` output stays byte-identical.
"""

from __future__ import annotations

from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict, going_second, make_config


def make_record(
    engine,
    line: ComboLine,
    deck,
    verdict_ok: Verdict,
    seed: int,
    seeds_ok: list[int],
    tag: str,
) -> dict:
    """Build the BC record for a verified line (first-ok seed `seed`)."""
    cfg = verdict_ok.config or make_config(line, deck, seed, engine)
    return {
        "line_id": line.id,
        "me": 1 if going_second(line) else 0,
        "source_url": line.source_url,
        "deck": line.deck + tag,
        "deck_source": verdict_ok.deck_source,
        "config": {
            "deck0_main": list(cfg.deck0.main),
            "deck0_extra": list(cfg.deck0.extra),
            "deck1_main": list(cfg.deck1.main),
            "deck1_extra": list(cfg.deck1.extra),
            "seed": seed,
            "hand": cfg.hand,
            "draw": cfg.draw,
            "setup": [[p.team, p.code, p.location, p.sequence] for p in cfg.setup],
        },
        "auto_single": True,
        "choices": verdict_ok.choices,
        "labels": verdict_ok.labels,
        "skipped_steps": list(verdict_ok.skipped_steps or []),
        "executed_steps": verdict_ok.executed_steps,
        "inferred_frac": verdict_ok.inferred_frac,
        "hand_inferred": verdict_ok.hand_inferred,
        "end_board": line.end_board,
        "end_board_codes": line.end_board_codes,
        "seeds_ok": list(seeds_ok),
    }
