"""Explain a failed line's deepest verifier path (moved out of scripts/explain_line.py).

`explain_failure` returns the text the script prints today, so the script is a
thin wrapper with identical output.
"""

from __future__ import annotations

from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict, make_config
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine

LOC = {1: "deck", 2: "hand", 3: "mzone", 4: "szone", 5: "grave", 6: "banished", 7: "extra", 8: "overlay"}


def explain_failure(
    engine: Engine,
    line: ComboLine,
    deck,  # Deck; untyped to avoid a hard import cycle in docs
    verdict: Verdict,
    names: dict[int, str],
) -> str:
    """Replay the verdict's deepest path and describe the board/options there."""
    out: list[str] = [f"{verdict.reason}: {verdict.detail}"]
    for st in line.steps:
        out.append(f"  step {st.i}: {st.action} {st.card} | {st.text[:100]}")
    config = verdict.config or make_config(line, deck, verdict.seed, engine)
    env = CppEnv(engine, config)
    obs = env.reset()
    for i in verdict.choices:
        obs = env.step(i)
    if obs is None:
        out.append("deepest path ends the duel")
        return "\n".join(out)
    cards = obs["cards"][obs["card_mask"]]
    out.append(
        f"\nafter {len(verdict.choices)} choices (deepest point), "
        f"turn {obs['global'][2]}, msg {obs['global'][21]}:"
    )
    for loc in (2, 3, 4, 5, 6):
        mine = [names.get(int(c[0]), int(c[0])) for c in cards if c[3] == 0 and c[1] == loc]
        out.append(f"  my {LOC[loc]}: {mine}")
    n = int(obs["action_mask"].sum())
    out.append("  options:")
    for row in obs["actions"][:n]:
        out.append(f"    {Kind(int(row[0])).name} {names.get(int(row[2]), int(row[2]))}")
    return "\n".join(out)
