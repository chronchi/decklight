"""Export hand-played gold games to one JSONL row per gold decision.

Reads data/demos/gold/*.jsonl, replays each record (seed + decks + response list), and writes
data/demos/gold_decisions.jsonl: one row per decision whose label is `gold` (real choices; `auto`
zone answers and `pass` window passes are skipped).

Row: {game_id, index, actor, deck, turn, msg, state, chain, options, choice, why, leak_suspect, junk_why}; `why`
is the note whose start index is the latest one <= index within the same turn and seat (notes are
[start_index, text] pairs in the record). `junk_why` is true when the why is junk ("test", "placeholder",
"x", ..., or under 3 chars): the why is then blanked (the decision is kept).

Selection churn is dropped: within one multi-select prompt sequence (MSG_SELECT_CARD /
SELECT_UNSELECT_CARD / SELECT_TRIBUTE / SELECT_SUM), a SELECT on card code X later followed by an
UNSELECT of X drops both rows, and CANCEL rows are dropped; only the final material set is kept.

`leak_suspect` lists card names mentioned in the `why` (case-insensitive substring) that are, at that
decision, hidden from the acting seat: in the opponent's hand or face-down on the opponent's field.
Hidden cards are approximated from the last observation seen while the *opponent* was acting (its own
hand and face-down cards are visible to it); a name already visible to the acting seat at the decision
(e.g. another copy is public) is not flagged.

    uv run python scripts/export_gold.py [--out data/demos/gold_decisions.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hand_play

from ygo_zero.demos.dataset import record_config
from ygo_zero.demos.game import _opts
from ygo_zero.demos.names import CardNames
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine

GOLD = ROOT / "data" / "demos" / "gold"
_MULTI_MSGS = frozenset(
    {C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_SUM}
)


def _why(notes: list, turns: list[int], actors: list[int], k: int) -> str:
    """The note for decision `k`: latest start <= k within the same turn and seat, else ""."""
    best, text = -1, ""
    for s, t in notes:
        if int(s) <= k and int(s) > best and turns[int(s)] == turns[k] and actors[int(s)] == actors[k]:
            best, text = int(s), str(t)
    return text


def _hidden_snapshot(obs) -> tuple[set[int], set[int]]:
    """The acting player's own hand codes + face-down field codes (visible to them, hidden from the foe)."""
    hand, facedown = set(), set()
    for r in obs["cards"][obs["card_mask"]]:
        code = int(r[0])
        if not code or int(r[3]) != 0:
            continue
        loc, pos = int(r[1]), int(r[5])
        if loc == 2:
            hand.add(code)
        elif (loc == 3 and pos in (2, 4)) or (loc == 4 and pos == 6):
            facedown.add(code)
    return hand, facedown


def _visible_codes(obs) -> set[int]:
    return {int(r[0]) for r in obs["cards"][obs["card_mask"]] if int(r[0])}


def _leak_suspect(why: str, hidden: set[int], visible: set[int], names: CardNames) -> list[str]:
    if not why or not hidden:
        return []
    wl = why.lower()
    out = []
    for code in hidden:
        nm = names.names.get(code, "")
        if nm and nm.lower() in wl and code not in visible and nm not in out:
            out.append(nm)
    return sorted(out)


_JUNK_WHYS = frozenset({"test", "placeholder", "x", "tbd", "todo", "...", "-", "?", "temp", "asdf"})


def _junk_why(why: str) -> bool:
    """Whether a why carries no signal: a known placeholder or under 3 chars (u15 [11] "test")."""
    w = why.strip().lower()
    return bool(w) and (len(w) < 3 or w in _JUNK_WHYS)


def export_game(engine: Engine, names: CardNames, rec: dict, out_f) -> tuple[int, int, int, int]:
    """Returns (rows written, dropped churn/cancel rows, rows with leak_suspect, junk whys blanked)."""
    env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
    obs = env.reset()
    seen: set[int] = set()
    choices, labels = rec["choices"], rec["labels"]
    turns, actors, decks, notes = rec["turns"], rec["actors"], rec["decks"], rec.get("notes", [])
    pending: list[dict] = []  # one entry per gold decision, in replay order
    groups: list[list[int]] = []  # choice indices of each multi-select prompt sequence (any label)
    cur: list[int] = []
    step_info: list[tuple] = []  # per choice: (obs, msg, kind, code, actor, opp_hidden_before)
    snaps: dict[int, set[int]] = {}  # player -> hand + face-down codes from its last acting obs
    for k, (choice, label) in enumerate(zip(choices, labels, strict=True)):
        if obs is None:
            print(f"  !! {rec['game_id']}: duel ended at choice {k}/{len(choices)}")
            break
        actor = env.player
        opp_hidden = set(snaps.get(1 - actor, set()))
        msg = int(obs["global"][21])
        oo = _opts(obs)
        kind = oo[int(choice)][0] if 0 <= int(choice) < len(oo) else None
        code = oo[int(choice)][1] if 0 <= int(choice) < len(oo) else 0
        step_info.append((obs, msg, kind, code, actor, opp_hidden))
        h, f = _hidden_snapshot(obs)
        snaps[actor] = h | f
        # One "prompt sequence" = a maximal run of contiguous multi-select prompts of the same message
        # for the same actor. (The obs step column cannot delimit these: SELECT_UNSELECT_CARD answers
        # every pick with its own engine message, so step stays 0 through a whole material selection.)
        if msg in _MULTI_MSGS and cur and step_info[cur[-1]][1] == msg and step_info[cur[-1]][4] == actor:
            cur.append(k)
        elif msg in _MULTI_MSGS:
            cur = [k]
            groups.append(cur)
        else:
            cur = []
        if label == "gold":
            d = hand_play.decision_json(env, obs, k, decks[actors[k]], names, seen)
            row = {
                "game_id": rec["game_id"],
                "index": d["index"],
                "actor": d["actor"],
                "deck": d["deck"],
                "turn": d["turn"],
                "msg": d["msg"],
                "hint": d["hint"],
                "state": d["state"],
                "chain": d["chain"],
                "options": d["options"],
                "choice": choice,
                "why": _why(notes, turns, actors, k),
            }
            pending.append({"k": k, "row": row, "kind": kind, "code": code})
        obs = env.step(int(choice))
    drop: set[int] = set()
    for g in groups:
        by_kind = {k: (step_info[k][2], step_info[k][3]) for k in g}
        drop.update(k for k in g if by_kind[k][0] == Kind.CANCEL)
        unmatched: dict[int, list[int]] = {}  # code -> SELECT choice indices awaiting an UNSELECT
        for k in g:
            kk, cc = by_kind[k]
            if kk == Kind.SELECT:
                unmatched.setdefault(cc, []).append(k)
            elif kk == Kind.UNSELECT and unmatched.get(cc):
                drop.add(unmatched[cc].pop(0))
                drop.add(k)
    n = n_leak = n_junk = 0
    for p in pending:
        if p["k"] in drop:
            continue
        k = p["k"]
        obs_k, _, _, _, _, opp_hidden = step_info[k]
        junk = _junk_why(p["row"]["why"])
        if junk:
            p["row"]["why"] = ""
            n_junk += 1
        p["row"]["junk_why"] = junk
        leak = _leak_suspect(p["row"]["why"], opp_hidden, _visible_codes(obs_k), names)
        p["row"]["leak_suspect"] = leak
        out_f.write(json.dumps(p["row"]) + "\n")
        n += 1
        n_leak += bool(leak)
    return n, len(drop), n_leak, n_junk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "demos" / "gold_decisions.jsonl"))
    a = ap.parse_args()
    engine, names = Engine(), CardNames()
    total = dropped = leaked = junked = 0
    with open(a.out, "w") as f:
        for path in sorted(GOLD.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n, d, le, ju = export_game(engine, names, rec, f)
                print(
                    f"{rec['game_id']} ({path.name}): {n} gold decisions "
                    f"({d} churn/cancel dropped, {le} with leak_suspect, {ju} junk whys blanked)"
                )
                total += n
                dropped += d
                leaked += le
                junked += ju
    print(f"total: {total} rows ({dropped} churn/cancel dropped, {leaked} with leak_suspect, "
          f"{junked} junk whys blanked) -> {a.out}")


if __name__ == "__main__":
    main()
