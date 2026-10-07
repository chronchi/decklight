"""DAGS (Data-Augmented Game Starts, RL v2 step 3): mid-game starts for RL episodes.

A start is one line of the starts file (written by scripts/build_dags_starts.py):
final (shuffled) deck orders + duel seed/hand/draw/lp/setup plus the prefix of option
indices replayed inside VecEnv.start_episode. `load_starts` returns them as the tuples
`VecEnv.set_starts` takes; `cut_points` / `build_config` share the builder's logic so tests
and the builder agree.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ygo_zero.engine.duel import DuelConfig


def deck_key(main, extra) -> tuple[tuple, tuple]:
    """Multiset identity of one deck (order-free): eval-leakage comparisons use this."""
    return (tuple(sorted(main)), tuple(sorted(extra)))


def holdout_keys(pool_dir: Path, holdout_names: set[str]) -> set[tuple[tuple, tuple]]:
    """Multiset keys of the holdout decks (train_rl excludes these from the pool)."""
    from ygo_zero.decks import load_ydk

    keys = set()
    for name in holdout_names:
        p = pool_dir / f"{name}.ydk"
        if not p.is_file():
            continue
        d = load_ydk(p)
        keys.add(deck_key(d.main, d.extra))
    return keys


def record_decks(rec: dict) -> tuple[tuple[tuple, tuple], tuple[tuple, tuple]]:
    c = rec["config"]
    return (deck_key(c["deck0_main"], c["deck0_extra"]),
            deck_key(c["deck1_main"], c["deck1_extra"]))


def is_holdout_game(rec: dict, holdout: set[tuple[tuple, tuple]]) -> bool:
    k0, k1 = record_decks(rec)
    return k0 in holdout or k1 in holdout


def build_config(rec: dict) -> DuelConfig:
    """DuelConfig that replays `rec` (deck lists + seed + hand/draw/lp/setup/shuffle overrides)."""
    c = rec["config"]
    setup = tuple(
        (p[0], p[1], p[2], p[3], p[4] if len(p) > 4 else 0) for p in c.get("setup", [])
    )
    return DuelConfig(
        SimpleNamespace(main=list(c["deck0_main"]), extra=list(c["deck0_extra"])),
        SimpleNamespace(main=list(c["deck1_main"]), extra=list(c["deck1_extra"])),
        seed=int(c["seed"]),
        hand=int(c.get("hand", 5)),
        draw=int(c.get("draw", 1)),
        lp=int(c.get("lp", 8000)),
        setup=setup,
        shuffle=bool(c.get("shuffle", False)),
    )


def final_mains(rec: dict) -> tuple[list[int], list[int]]:
    """Both mains in final (shuffled) order, exactly as DuelConfig produces them."""
    cfg = build_config(rec)
    return cfg.main_of(0), cfg.main_of(1)


def cut_points(n: int, turns: list[int] | None, rng) -> list[int]:
    """Cut indices k (prefix = choices[:k], episode starts at decision k): every k >= 1 at a
    turn boundary (turns[k] != turns[k-1]) plus 2 random k within the game. Sorted, deduped."""
    if n < 2:
        return []
    cuts: set[int] = set()
    if turns is not None and len(turns) == n:
        for k in range(1, n):
            if turns[k] != turns[k - 1]:
                cuts.add(k)
    for _ in range(2):
        cuts.add(int(rng.integers(1, n)))
    return sorted(cuts)


def start_record(rec: dict, k: int, src: str) -> dict[str, Any]:
    """One starts-file line for cut k of `rec` (prefix choices[:k]; caller must verify it replays)."""
    m0, m1 = final_mains(rec)
    c = rec["config"]
    turns = rec.get("turns")
    return {
        "main0": m0,
        "extra0": list(c["deck0_extra"]),
        "main1": m1,
        "extra1": list(c["deck1_extra"]),
        "seed": int(c["seed"]),
        "hand": int(c.get("hand", 5)),
        "draw": int(c.get("draw", 1)),
        "lp": int(c.get("lp", 8000)),
        "setup": [list(p) for p in c.get("setup", [])],
        "prefix": [int(a) for a in rec["choices"][:k]],
        "src": src,
        "turn": turns[k] if turns is not None and len(turns) == len(rec["choices"]) else None,
        "game_id": rec.get("game_id"),
        "decks": list(rec.get("decks") or []),
    }


def to_set_starts(line: dict) -> tuple:
    """One starts-file line -> the tuple VecEnv.set_starts takes."""
    setup = [tuple(p) for p in line.get("setup", [])]
    return (list(line["main0"]), list(line["extra0"]), list(line["main1"]), list(line["extra1"]),
            int(line["seed"]), [int(a) for a in line["prefix"]], int(line.get("hand", 5)),
            int(line.get("draw", 1)), int(line.get("lp", 8000)), setup)


def load_starts(path: str | Path) -> list[tuple]:
    """Starts file -> set_starts tuples (raises if the file holds no starts)."""
    out = [to_set_starts(json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]
    if not out:
        raise ValueError(f"no DAGS starts in {path}")
    return out


def deck_counter(deck) -> Counter:
    return Counter(list(deck.main) + ["E"] + list(deck.extra))
