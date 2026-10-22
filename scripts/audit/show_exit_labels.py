"""Read filtered ExIt labels (scripts/exit_filter.py output) as text: the state at each kept searched decision, every
candidate with its mean rollout score and SE, the search's pick (*) and the move played in the game (>).

    python scripts/audit/show_exit_labels.py 'data/demos/exit-p/ya/*.jsonl' [--n 8] [--overrides] [--seed 0] [--stats]

--stats prints whole-data distributions instead: kept labels by turn, the z and candidate-count distributions, and the
option kind of the search pick vs the played move on overrides.
Laptop-safe: engine replay only, no model.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

import hand_play as H
from exit_filter import clean_z

from decklight.demos.dataset import record_config
from decklight.demos.game import _opts, describe_state, hidden_card, option_desc


def _mean_se(xs: list[float]) -> tuple[float, float]:
    return sum(xs) / len(xs), (st.pvariance(xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0


def _label(obs, i: int, nn: dict) -> str:
    k, code = _opts(obs)[i]
    nm = nn.get(code, "") if code else hidden_card(obs, i)
    return f"{H._kind(k)} {nm} {option_desc(obs, i)!r}".strip()


def entries(files: list[str]) -> list[tuple[dict, dict]]:
    out = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out.extend((rec, e) for e in rec.get("search") or [])
    return out


def show(rec: dict, e: dict, engine, nn: dict) -> list[str]:
    from decklight.engine.cpp_env import CppEnv

    env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
    obs = env.reset()
    for c in rec["choices"][: e["k"]]:
        obs = env.step(int(c))
    lines = [f"== {rec.get('line_id', rec.get('game_id'))} decision {e['k']} turn {rec['turns'][e['k']]} "
             f"z {clean_z(e['samples']):.1f}" + ("  OVERRIDE" if e["chosen"] != e["played"] else ""),
             *("   " + s for s in describe_state(obs, nn).splitlines())]  # fmt: skip
    for cand, sm in sorted(zip(e["cands"], e["samples"], strict=True), key=lambda t: -_mean_se(t[1])[0]):
        m, se = _mean_se(sm)
        mark = ("*" if cand == e["chosen"] else " ") + (">" if cand == e["played"] else " ")
        lines.append(f"  {mark} {m:+.3f} ±{se:.3f}  {_label(obs, int(cand), nn)}")
    return lines


def stats(ents: list[tuple[dict, dict]], engine, nn: dict, cap: int) -> None:
    from decklight.engine.cpp_env import CppEnv

    turn = Counter(rec["turns"][e["k"]] for rec, e in ents)
    print(f"labels {len(ents)}; by turn {dict(sorted(turn.items()))}; "
          f"overrides {sum(e['chosen'] != e['played'] for _, e in ents)}")  # fmt: skip
    zs = sorted(min(clean_z(e["samples"]), 99) for _, e in ents)
    nc = Counter(len(e["cands"]) for _, e in ents)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    print(f"z quartiles {q(zs, .25):.1f} {q(zs, .5):.1f} {q(zs, .75):.1f}; inf/99 share {sum(z >= 99 for z in zs) / len(zs):.2f}")
    print(f"candidates per label {dict(sorted(nc.items()))}; samples per candidate "
          f"{Counter(len(e['samples'][0]) for _, e in ents).most_common(3)}")  # fmt: skip
    pair = Counter()
    for rec, e in [x for x in ents if x[1]["chosen"] != x[1]["played"]][:cap]:
        env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
        obs = env.reset()
        for c in rec["choices"][: e["k"]]:
            obs = env.step(int(c))
        o = _opts(obs)
        pair[(H._kind(o[int(e["chosen"])][0]), H._kind(o[int(e["played"])][0]))] += 1
    n = sum(pair.values())
    print(f"override kinds (search pick <- played), first {n} overrides:")
    for (a, b), v in pair.most_common(15):
        print(f"  {v:5d} {v / n:5.1%}  {a} <- {b}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="glob of filtered record jsonl files")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--overrides", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--cap", type=int, default=1500, help="overrides replayed for --stats kind counts")
    a = ap.parse_args()
    from decklight.demos.names import CardNames
    from decklight.engine.duel import Engine

    ents = entries(sorted(glob.glob(a.records)))
    if not ents:
        print("no labels")
        return 1
    nn, engine = CardNames().names, Engine()
    if a.stats:
        stats(ents, engine, nn, a.cap)
        return 0
    if a.overrides:
        ents = [x for x in ents if x[1]["chosen"] != x[1]["played"]]
    random.Random(a.seed).shuffle(ents)
    for rec, e in ents[: a.n]:
        print("\n".join(show(rec, e, engine, nn)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
