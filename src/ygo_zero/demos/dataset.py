"""Behaviour-cloning samples from engine-verified combo lines (ADR 0002).

Only replay records are stored (`data/demos/verified/*.jsonl`: config + choices + labels); observations are
regenerated here by replaying each record in the C++ env (ADR 0001 storage rule).

    for obs, action, weight, meta in iter_samples(engine, Path("data/demos/verified")):
        ...  # obs: dict of arrays for the player to act; action: option index

Weights: choices stated by the guide count fully; choices the verifier found by search are plausible but
arbitrary among equivalent ones (zones, which copy), so they count less; the opponent's automatic moves are
not samples at all.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine, Placement

LABEL_WEIGHT = {"stated": 1.0, "inferred": 0.5}


def record_config(rec: dict, seed: int | None = None) -> DuelConfig:
    """The DuelConfig stored in a verified record (optionally with another seed from `seeds_ok`)."""
    c = rec["config"]
    deck0 = Deck(tuple(c["deck0_main"]), tuple(c["deck0_extra"]))
    deck1 = Deck(tuple(c["deck1_main"]), tuple(c["deck1_extra"]))
    setup = tuple(Placement(*p) for p in c.get("setup", []))
    return DuelConfig(
        deck0, deck1, seed=c["seed"] if seed is None else seed, lp=c.get("lp", 8000), hand=c.get("hand", 0),
        draw=c.get("draw", 1), setup=setup,
    )  # fmt: skip


def load_records(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def iter_samples(engine: Engine, path: Path) -> Iterator[tuple[dict, int, float, dict]]:
    """Yield (obs, action, weight, meta) for every agent decision of every verified record.

    Only the record's own seed is replayed: `choices` come from that seed's search, and other seeds in
    `seeds_ok` were verified with their own (unstored) choices."""
    for rec in load_records(path):
        for seed in [rec["config"]["seed"]]:
            env = CppEnv(engine, record_config(rec, seed), rec.get("auto_single", True))
            obs = env.reset()
            for k, (choice, label) in enumerate(zip(rec["choices"], rec["labels"], strict=True)):
                if obs is None:
                    raise ValueError(f"{rec['line_id']} seed {seed}: duel ended at choice {k}")
                if label in LABEL_WEIGHT:
                    # game records (ADR 0004) carry both decks and who acted; lines are named <deck>/<id>
                    deck = rec["decks"][rec["actors"][k]] if "actors" in rec else rec["line_id"].split("/")[0]
                    meta = {"line_id": rec["line_id"], "seed": seed, "k": k, "label": label, "deck": deck}
                    yield {key: v.copy() for key, v in obs.items()}, choice, LABEL_WEIGHT[label], meta
                obs = env.step(choice)
