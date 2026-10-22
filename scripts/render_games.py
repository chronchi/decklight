"""Render saved playtest games as readable logs (engine only, no model).

    uv run python scripts/render_games.py <dir> [--decisions]

For every `state.json` under `<dir>` (saved by `playtest.py --save-games`), write `game.txt`
next to it: a header from the meta (who the policy is, which seat, opponent, result, turns)
followed by `hand_play._log` output. With `--decisions`, also list every non-forced decision
with its option count, the way `data/cache/dump_game.py` prints them. Also writes
`<dir>/INDEX.txt` with one line per game.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hand_play as H
from decklight.demos.game import _opts, hidden_card, option_desc
from decklight.demos.names import CardNames
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import Engine


def result_of(meta: dict) -> str:
    if meta.get("truncated"):
        return "CUT"
    if meta.get("winner") is None:
        return "DRAW" if "winner" in meta else ("WIN" if meta.get("policy_won") else "LOSS")
    return "WIN" if meta.get("policy_won") else "LOSS"


def header_lines(st: dict) -> list[str]:
    meta = st.get("meta", {})
    decks = st.get("decks", ["?", "?"])
    seat = meta.get("policy_seat", "?")
    opp = meta.get("opponent", "?")
    match = meta.get("match", "?")
    winner = meta.get("winner")
    res = result_of(meta)
    lines = [
        f"match {match}: policy seat P{seat} ({decks[seat] if isinstance(seat, int) and 0 <= seat < len(decks) else '?'})"
        f" vs {opp} — {res}"
        + (f" (winner P{winner})" if isinstance(winner, int) else ""),
        f"decks (seat order): {decks[0] if len(decks) > 0 else '?'} vs {decks[1] if len(decks) > 1 else '?'}"
        f", seed {st.get('seed')}, turns {meta.get('turns', '?')}"
        + (" [truncated]" if meta.get("truncated") else ""),
    ]
    return lines


def decisions_lines(st: dict, names: CardNames, engine: Engine) -> list[str]:
    """Every non-forced decision with its option count (dump_game.py format)."""
    env = CppEnv(engine, H._config(st))
    obs = env.reset()
    nn = names.names
    try:
        notes = dict((a, b) for a, b in st.get("notes", []))
    except (TypeError, ValueError):
        notes = {}
    out = []
    for n, (c, lab, act, t) in enumerate(
        zip(st["choices"], st["labels"], st["actors"], st["turns"], strict=True)
    ):
        msg = H._MSGS.get(int(obs["global"][21]))
        o = _opts(obs)
        k, code = o[int(c)]
        nm = nn.get(code, "") if code else hidden_card(obs, int(c))
        hint = H._hint(env, obs)
        alts = len(o)
        if lab == "gold" and alts > 1:
            line = (
                f"[{n}] T{t} P{act} {msg[11:] if msg else msg}"
                f"{(':' + hint) if hint else ''} {H._kind(k)} {nm} {option_desc(obs, int(c))!r} /{alts}"
                + (f"  WHY: {notes[n]}" if n in notes else "")
            )
            out.append(line)
        obs = env.step(int(c))
        if obs is None:
            break
    return out


def render_one(state_path: Path, decisions: bool, names: CardNames, engine: Engine) -> dict:
    st = json.loads(state_path.read_text())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        H._log(st, names, engine)
    text = "\n".join(header_lines(st)) + "\n\n" + buf.getvalue()
    if decisions:
        dl = decisions_lines(st, names, engine)
        text += "\n--- decisions (non-forced) ---\n"
        text += ("\n".join(dl) + "\n") if dl else "(all forced)\n"
    (state_path.parent / "game.txt").write_text(text)
    return st.get("meta", {})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render saved playtest games as readable logs.")
    ap.add_argument("dir", type=Path, help="directory holding saved games (state.json files below it)")
    ap.add_argument("--decisions", action="store_true", help="also list every non-forced decision")
    args = ap.parse_args(argv)
    root = args.dir
    names, engine = CardNames(), Engine()
    states = sorted(root.rglob("state.json"))
    rows = []
    for sp in states:
        meta = render_one(sp, args.decisions, names, engine)
        rel = sp.relative_to(root).as_posix()
        parts = sp.relative_to(root).parts
        match = meta.get("match", parts[-3] if len(parts) >= 3 else "?")
        k = parts[-2] if len(parts) >= 2 else "?"
        decks = json.loads(sp.read_text()).get("decks", ["?", "?"])
        seat = meta.get("policy_seat", "?")
        res = result_of(meta)
        turns = meta.get("turns", "?")
        rows.append(f"{match} k={k} {decks[0]} vs {decks[1]} policy=P{seat} {res} turns={turns} {rel}")
    (root / "INDEX.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"rendered {len(states)} games -> {root / 'INDEX.txt'}")


if __name__ == "__main__":
    main()
