"""Check Rust match records v1 (A2c): schema, replay through the core, and raw responses vs the Python games.

    uv run python scripts/app/record_check.py --records DIR [--python-records DIR]

`--python-records` is the Python server's `runs/web` root written by the transcript run (`ws_transcript.py
--server python`). Every Rust record is matched to the Python game with the same duel seed and its
`responses[].r` list must equal `env.duel.responses` of that game replayed in PyEnv from game.jsonl's choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_rust(records: Path) -> list[dict]:
    out = []
    for p in sorted(records.glob("*.match.json.gz")):
        import gzip

        with gzip.open(p, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
        rec["_path"] = str(p)
        out.append(rec)
    return out


def python_games(py_root: Path) -> dict[int, dict]:
    out = {}
    for g in sorted(py_root.glob("*/game.jsonl")):
        rec = json.loads(g.read_text().splitlines()[0])
        rec["_dir"] = g.parent
        out[int(rec["seed"])] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--python-records", type=Path, default=None)
    a = ap.parse_args(argv)

    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig, Engine
    from decklight.engine.env import PyEnv
    from decklight.play import match_record as MR

    engine = Engine()
    recs = load_rust(a.records)
    if not recs:
        print("ALERT record_check: no records", file=sys.stderr)
        return 2
    bad = 0
    schema_bad = replay_bad = resp_bad = 0
    for rec in recs:
        problems = MR.validate(rec)
        if problems:
            schema_bad += 1
            print(f"ALERT record_check: {rec['_path']}: schema: {problems[:3]}", file=sys.stderr)
        problems = MR.check(rec, engine)
        if problems:
            replay_bad += 1
            print(f"ALERT record_check: {rec['_path']}: replay: {problems}", file=sys.stderr)

    if a.python_records is not None:
        pys = python_games(a.python_records)
        for rec in recs:
            seed = int(rec.get("seed", -1))
            py = pys.get(seed)
            if py is None:
                resp_bad += 1
                print(f"ALERT record_check: no Python game for seed {seed} ({rec['_path']})", file=sys.stderr)
                continue
            d = py["_dir"]
            human = load_ydk(d / "human.ydk")
            bot = load_ydk(d / "bot.ydk")
            if int(py["human_seat"]) == 0:
                d0, d1 = human, bot
            else:
                d0, d1 = bot, human
            env = PyEnv(engine, DuelConfig(d0, d1, seed=seed), auto_single=False)
            env.reset()
            try:
                for idx in py["choices"]:
                    env.step(int(idx))
            except Exception as e:  # noqa: BLE001
                resp_bad += 1
                print(f"ALERT record_check: seed {seed}: Python replay failed: {e}", file=sys.stderr)
                continue
            want = [r.hex() for r in env.duel.responses]
            got = [e["r"] for e in rec["responses"]]
            if want != got:
                resp_bad += 1
                i = next((i for i, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
                print(
                    f"ALERT record_check: seed {seed}: responses differ at {i} "
                    f"(python {len(want)} vs rust {len(got)})",
                    file=sys.stderr,
                )
            env.duel.close()
        games = len(pys)
    else:
        games = None

    bad = schema_bad + replay_bad + resp_bad
    print(
        json.dumps(
            {
                "records": len(recs),
                "python_games": games,
                "schema_fail": schema_bad,
                "replay_fail": replay_bad,
                "responses_fail": resp_bad,
                "ok": bad == 0,
            }
        )
    )
    if bad:
        print(f"ALERT record_check: {bad} failures", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
