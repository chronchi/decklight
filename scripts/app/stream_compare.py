"""Compare stream dumps (A2b-3a/3b gates).

Gate 1 (board tracker = Python), exact JSON line compare:
    uv run python scripts/app/stream_compare.py --gate1 py.jsonl.gz rs.jsonl.gz

Gate 3 (stream frames vs engine frames for the same seat):
    uv run python scripts/app/stream_compare.py --gate3 frames.jsonl.gz [...]

Frames are paired by the unique `f` counter the `stream_frames` binary writes on both
lines of a pair (fallback: the next stream line after each env line). Every pair is
compared. `cues[].n` and `events[].n`/`events[].card.uid` are normalized: the client
uses them only to order and dedupe (web/app.js `freshCues` / `addEvents`), never shows
them, and the stream's trace is the packets this seat received, so its indices cannot
equal the engine trace's.

Gate 3 exits 2 on any prompt-option difference and 3 on any difference whose path is
not in `ALLOWED` (each entry carries its reason). Event rows are compared per game:
every row that exists on both sides must have the same kind/actor/card (a code the
stream never learned, 0, is a wildcard), and the residual classes are allowlisted.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Paths whose difference is a known, evidenced consequence of the packet set (or of an
# engine-view lookup), not a tracker bug. Each value is shown in the report.
ALLOWED = {
    "cues[].n": "ordering/dedupe id only (web/app.js freshCues); the stream trace is the packets this seat received",
    "cues[].target.code": "a face-down card's identity is not in this seat's messages (MSG_SET zeroes the code; MSG_BECOME_TARGET carries no code)",
    "cues[].target.name": "same: the target's identity was never named to this seat",
    "cues[].target.uid": "same: the target's identity was never named to this seat",
    "cues[].attacker.code": "attacker identity not named in this seat's messages (a face-down card; MSG_ATTACK carries only a location)",
    "cues[].attacker.name": "same: the attacker's identity was never named to this seat",
    "cues[].attacker.uid": "same: the attacker's identity was never named to this seat",
    "cues[].cards[].code": "a cue about cards whose identities were never named to this seat (face-down confirm/draw/target)",
    "cues[].cards[].name": "same: the card's identity was never named to this seat",
    "cues[].cards[].uid": "same: the card's identity was never named to this seat",
    "prompt.card.uid": "the prompt card is the chain's last card; its zone comes from build_view's ambiguous chain lookup (engine last match, A2a leak), while the code/name agree",
    "view.chain[].card.uid": "engine build_view chain lookup takes the LAST match by (controller,sequence,code) and can pick a card in another zone with the same code/sequence (A2a leak; the stream shows the chain link's own zone)",
    "view.chain[].card.faceup": "same chain-card lookup: the engine can match a face-down copy in another zone instead of the chain link's card",
    "view.chain[].card.pos": "Trap Monster resolving in the S/T zone: the core flips it to monster stats with no message/refresh",
    "view.chain[].card.atk": "Trap Monster resolving in the S/T zone: live stats arrive with no message/refresh",
    "view.chain[].card.def": "same: live stats arrive with no message/refresh",
    "view.chain[].card.level": "same: live stats arrive with no message/refresh",
    "view.you.szone[].atk": "Trap Monster in the S/T zone: the core flips it to monster stats with no message/refresh",
    "view.you.szone[].def": "same: live stats arrive with no message/refresh",
    "view.you.szone[].level": "same: live stats arrive with no message/refresh",
    "view.bot.szone[].atk": "same: Trap Monster stats are live in the engine and not refreshed to this seat",
    "view.bot.szone[].def": "same: live stats arrive with no message/refresh",
    "view.bot.szone[].level": "same: live stats arrive with no message/refresh",
    "view.you.mzone[].atk": "a live ATK change resolved with no refresh; the stream keeps the last refreshed row",
    "view.you.mzone[].def": "a live DEF change resolved with no refresh; the stream keeps the last refreshed row",
    "view.you.mzone[].level": "a live level change resolved with no refresh; the stream keeps the last refreshed row",
    "view.bot.mzone[].atk": "a live ATK change resolved with no refresh; the stream keeps the last refreshed row",
    "view.bot.mzone[].def": "a live DEF change resolved with no refresh; the stream keeps the last refreshed row",
    "view.bot.mzone[].level": "a live level change resolved with no refresh; the stream keeps the last refreshed row",
    "view.you.deck": "a card of ours held by the opponent whose code this seat never saw is known out of the deck but not to which code it belongs; the engine still counts it out of the deck",
    "view.you.hand[]": "the host computes after-refresh queries at the end of a replay step, so a later MOVE in the same step lands on a hand refresh that already includes it; corrected by the next HAND refresh",
}

# Engine history rows for the opponent that no message in this seat's stream carries.
HIDDEN_KINDS = {
    "pass",
    "yes",
    "no",
    "option",
    "select",
    "unselect",
    "finish",
    "cancel",
    "position",
    "place",
    "number",
    "attribute",
    "race",
    "declare",
    "sort",
    "counter",
    "rps",
}
PHASE_KINDS = {"to_battle", "to_main2", "to_end"}
SET_KINDS = {"mset", "sset"}
# Public actions the stream learns from a message; the engine logs a decision row only
# when the action was a decision (effect-driven sets/activations/position changes have
# none), so a stream-only row of these kinds is not a contradiction.
MESSAGE_ACTIONS = {"activate", "summon", "spsummon", "flip", "mset", "sset", "repos", "attack"}

EVENT_ALLOWED = {
    "engine-only-hidden": (
        "opponent history rows for decisions that send no message to this seat "
        "(pass / yes / no / option / select / position / place / ...)"
    ),
    "stream-only-message": (
        "public action carried by a message that the engine did not record as a decision "
        "(effect-driven set/activation/position change), or a hidden set card's zone kind"
    ),
    "engine-only-bot-action": (
        "opponent decision row whose action produced no message this seat receives "
        "(attack/summon/set/activation cancelled, negated or aborted before its message)"
    ),
    "engine-only-phase": (
        "the engine logs the turn player's phase choice; the stream learns phases from MSG_NEW_PHASE, "
        "which is absent when the core changes phase without a message to this seat"
    ),
    "stream-only-phase": (
        "MSG_NEW_PHASE also announces phase entries the core makes without a phase-choice "
        "decision (e.g. Main 2 after the Battle Phase ends), which have no engine history row"
    ),
}


def cmp(a: Any, b: Any, path: str) -> str | None:
    if type(a) is not type(b):
        return path
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            missing = sorted(set(a.keys()) ^ set(b.keys()))
            return f"{path}.{missing[0]}" if missing else path
        for k in a:
            bad = cmp(a[k], b[k], f"{path}.{k}")
            if bad is not None:
                return bad
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}[{min(len(a), len(b))}]"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            bad = cmp(x, y, f"{path}[{i}]")
            if bad is not None:
                return bad
        return None
    return None if a == b else path


def lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def gate1(a_path: Path, b_path: Path) -> int:
    differences = 0
    first = None
    n = 0
    it_a, it_b = lines(a_path), lines(b_path)
    while True:
        la = next(it_a, None)
        lb = next(it_b, None)
        if la is None and lb is None:
            break
        n += 1
        if la is None or lb is None:
            differences += 1
            if first is None:
                first = {
                    "line": n,
                    "path": "$",
                    "py": "<missing>" if la is None else json.loads(la),
                    "rs": "<missing>" if lb is None else json.loads(lb),
                }
            continue
        a, b = json.loads(la), json.loads(lb)
        bad = cmp(a, b, "$")
        if bad is not None:
            differences += 1
            if first is None:
                first = {"line": n, "path": bad, "py": a, "rs": b}
    print(json.dumps({"lines": n, "differences": differences, "first": first}, separators=(",", ":")))
    if differences:
        print(f"ALERT stream_compare: {differences} board-tracker differences", file=sys.stderr)
        return 2
    return 0


def norm(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return path + "[]"
    return f"{path}.{key}" if path else key


def grouped_diff(a: Any, b: Any, path: str, out: dict, example: dict, ctx: dict) -> None:
    if type(a) is not type(b):
        out[path] += 1
        example.setdefault(path, {"at": ctx, "env": a, "stream": b})
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                p = norm(path, k)
                out[p] += 1
                example.setdefault(p, {"at": ctx, "missing": k})
            else:
                grouped_diff(a[k], b[k], norm(path, k), out, example, ctx)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            p = path + "[]"
            out[p] += 1
            example.setdefault(p, {"at": ctx, "len_env": len(a), "len_stream": len(b)})
        for x, y in zip(a, b):
            grouped_diff(x, y, path + "[]", out, example, ctx)
        return
    if a != b:
        out[path] += 1
        example.setdefault(path, {"at": ctx, "env": a, "stream": b})


def pair_frames(paths: list[Path]):
    """Yield (ctx, env, stream) for every frame pair, in file/line order."""
    for fi, path in enumerate(paths):
        by_f: dict[int, dict[str, dict]] = defaultdict(dict)
        order: list[int] = []
        prev: dict | None = None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for ln, line in enumerate(fh):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("k") != "frame":
                    continue
                src = rec.get("src")
                if "f" in rec:
                    key = int(rec["f"])
                    if key not in by_f:
                        order.append(key)
                    by_f[key][src] = rec
                else:  # old dumps: env is followed by its stream line
                    if src == "env":
                        prev = rec
                    elif prev is not None:
                        key = len(order)
                        order.append(key)
                        by_f[key] = {"env": prev, "stream": rec}
                        prev = None
        for key in order:
            pair = by_f[key]
            env, stream = pair.get("env"), pair.get("stream")
            if env is None or stream is None:
                continue
            ctx = {
                "file": path.name,
                "line": key,
                "g": env.get("g"),
                "s": env.get("s"),
                "d": env.get("d"),
                "turn": (env.get("view") or {}).get("turn"),
                "phase": (env.get("view") or {}).get("phase"),
                "prompt": (env.get("prompt") or {}).get("kind"),
                "pi": stream.get("pi"),
                "fi": fi,
            }
            yield ctx, env, stream


def strip_event_ids(events: list[dict]) -> list[dict]:
    out = []
    for e in events or []:
        d = dict(e)
        d.pop("n", None)
        card = d.get("card")
        if isinstance(card, dict) and str(card.get("uid", "")).startswith("e"):
            card = dict(card)
            card.pop("uid", None)
            d["card"] = card
        out.append(d)
    return out


def strip_cue_ids(cues: list[dict]) -> list[dict]:
    out = []
    for c in cues or []:
        d = dict(c)
        d.pop("n", None)
        out.append(d)
    return out


def event_key(e: dict) -> tuple[str, str, int]:
    card = e.get("card") or {}
    return (e.get("kind") or "", e.get("actor") or "", int(card.get("code") or 0))


def lcs(a: list, b: list, match) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if match(a[i], b[j]):
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    out = []
    while i < n and j < m:
        if match(a[i], b[j]):
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def event_equal(a: tuple, b: tuple) -> bool:
    return a == b


def event_wild(a: tuple, b: tuple) -> bool:
    """Same kind/actor and compatible card code (0 = this seat never learned it)."""
    if a[0] != b[0] or a[1] != b[1]:
        return False
    if a[0] in SET_KINDS and b[0] in SET_KINDS:
        return a[2] == 0 or b[2] == 0 or a[2] == b[2]
    return a[2] == 0 or b[2] == 0 or a[2] == b[2]


def gate3(paths: list[Path], packets: list[Path]) -> int:
    counts: Counter[str] = Counter()
    frames_bad: Counter[str] = Counter()
    examples: dict[str, dict] = {}
    pairs = 0
    equal = 0
    prompt_bad = 0
    # Events are incremental slices: compare each game's whole accumulated history, so a
    # row emitted in a different frame (protocol batching) is not a false difference.
    game_events: dict[tuple, dict[str, list]] = defaultdict(lambda: {"env": [], "stream": []})
    event_ctx: dict[tuple, dict] = {}

    for ctx, env, stream in pair_frames(paths):
        pairs += 1
        local: Counter[str] = Counter()
        # View / prompt: exact (uid/n normalization does not apply here).
        for field in ("view", "prompt"):
            grouped_diff(env.get(field), stream.get(field), field, local, examples, ctx)
        # Cues: n is an ordering/dedupe id. A confirm cue for the opponent's Deck/Extra
        # never reaches this seat (filter drops that message), which shifts the tail.
        ecues = strip_cue_ids(env.get("cues"))
        scues = strip_cue_ids(stream.get("cues"))
        if ecues != scues:
            mp = lcs(ecues, scues, lambda a, b: a == b)
            me = {i for i, _ in mp}
            ms = {j for _, j in mp}
            e_only = [i for i in range(len(ecues)) if i not in me]
            s_only = [j for j in range(len(scues)) if j not in ms]
            if not s_only and e_only and all(ecues[i].get("t") == "confirm" for i in e_only):
                drop = set(e_only)
                counts["cues[] (engine-only confirm)"] += len(e_only)
                examples.setdefault(
                    "cues[] (engine-only confirm)",
                    {
                        "at": ctx,
                        "count": len(e_only),
                        "reason": "MSG_CONFIRM_CARDS for the opponent's Deck/Extra is not sent to this seat",
                    },
                )
                ecues = [c for i, c in enumerate(ecues) if i not in drop]
        grouped_diff(ecues, scues, "cues", local, examples, ctx)
        for p, n in local.items():
            reason = ALLOWED.get(p, "")
            if not reason and p.startswith("prompt.options"):
                prompt_bad += 1
            counts[p] += n
            frames_bad[p] += 1
            if not reason:
                examples.setdefault(p, {"at": ctx})
        if not local:
            equal += 1
        # Events accumulate by game + seat.
        g = (env.get("g"), env.get("s"))
        for src, e in (("env", env), ("stream", stream)):
            for ev in strip_event_ids(e.get("events") or []):
                game_events[g][src].append((event_key(ev), ev))
                event_ctx.setdefault((g, ev.get("kind", ""), ev.get("actor")), ctx)
        if env.get("g") != stream.get("g") or env.get("s") != stream.get("s"):
            print(json.dumps({"ALERT": "frame pair game/seat mismatch", "at": ctx}), file=sys.stderr)
            return 3

    # --- events, per game ---
    ev_counts: Counter[tuple[str, str, str]] = Counter()
    ev_examples: dict[tuple[str, str, str], dict] = {}
    for (g, s), d in sorted(game_events.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        ek = [k for k, _ in d["env"]]
        sk = [k for k, _ in d["stream"]]
        phase_e = Counter(k for k in ek if k[0] in PHASE_KINDS)
        phase_s = Counter(k for k in sk if k[0] in PHASE_KINDS)
        for k in phase_e - phase_s:
            cls = ("engine-only-phase", k[0], k[1], k[2])
            ev_counts[cls] += (phase_e - phase_s)[k]
            ev_examples.setdefault(cls, {"g": g, "s": s, "kind": k[0], "actor": k[1], "code": k[2]})
        for k in phase_s - phase_e:
            cls = ("stream-only-phase", k[0], k[1], k[2])
            ev_counts[cls] += (phase_s - phase_e)[k]
            ev_examples.setdefault(cls, {"g": g, "s": s, "kind": k[0], "actor": k[1], "code": k[2]})

        a = [k for k in ek if k[0] not in PHASE_KINDS]
        b = [k for k in sk if k[0] not in PHASE_KINDS]
        m1 = lcs(a, b, event_equal)
        ia = {i for i, _ in m1}
        ib = {j for _, j in m1}
        ra = [(i, a[i]) for i in range(len(a)) if i not in ia]
        rb = [(j, b[j]) for j in range(len(b)) if j not in ib]
        m2 = lcs([k for _, k in ra], [k for _, k in rb], event_wild)
        ia2 = {ra[i][0] for i, _ in m2}
        ib2 = {rb[j][0] for _, j in m2}
        for i in range(len(a)):
            if i in ia or i in ia2:
                continue
            k = a[i]
            if k[0] in HIDDEN_KINDS:
                world = "engine-only-hidden"
            elif k[1] == "bot" and k[0] in MESSAGE_ACTIONS:
                world = "engine-only-bot-action"
            else:
                world = "engine-only"
            cls = (world, k[0], k[1], k[2])
            ev_counts[cls] += 1
            ev_examples.setdefault(cls, {"g": g, "s": s, "kind": k[0], "actor": k[1], "code": k[2]})
        for j in range(len(b)):
            if j in ib or j in ib2:
                continue
            k = b[j]
            cls = ("stream-only-message" if k[0] in MESSAGE_ACTIONS else "stream-only", k[0], k[1], k[2])
            ev_counts[cls] += 1
            ev_examples.setdefault(cls, {"g": g, "s": s, "kind": k[0], "actor": k[1], "code": k[2]})

    event_table = []
    unexpected_events = 0
    for cls, n in sorted(ev_counts.items(), key=lambda kv: str(kv[0])):
        world, kind, actor, _code = cls
        reason = EVENT_ALLOWED.get(world, "")
        if not reason:
            unexpected_events += n
        event_table.append(
            {
                "path": f"events[] ({world}) {kind}/{actor}",
                "count": n,
                "reason": reason,
                "example": ev_examples.get(cls),
            }
        )

    table = []
    unexpected = 0
    for p in sorted(counts):
        reason = ALLOWED.get(p, "")
        if p == "cues[] (engine-only confirm)":
            reason = "MSG_CONFIRM_CARDS for the opponent's Deck/Extra is not sent to this seat (filter rule)"
        if not reason:
            unexpected += counts[p]
        table.append(
            {
                "path": p,
                "count": counts[p],
                "frames": frames_bad[p],
                "reason": reason,
                "example": examples.get(p),
            }
        )
    report = {
        "frame_pairs": pairs,
        "frames_equal": equal,
        "paths_different": table,
        "events": event_table,
        "prompt_option_differences": prompt_bad,
        "unexpected_differences": unexpected + unexpected_events,
    }
    # Keep the first example small but useful: add the last few received messages when
    # the matching packet dumps are given.
    if packets:
        last = packet_tails(packets)
        for entry in table:
            ex = entry.get("example")
            if isinstance(ex, dict) and isinstance(ex.get("at"), dict):
                at = ex["at"]
                tail = last.get((at.get("g"), at.get("s"), at.get("pi")))
                if tail:
                    ex["last_messages"] = tail
    print(json.dumps(report, separators=(",", ":")))
    if prompt_bad:
        print(f"ALERT stream_compare: prompt options differ at {prompt_bad} decisions", file=sys.stderr)
        return 2
    if report["unexpected_differences"]:
        print(
            f"ALERT stream_compare: {report['unexpected_differences']} non-allowlisted differences",
            file=sys.stderr,
        )
        return 3
    return 0


_NAMES = {
    0x1: "RETRY",
    0x2: "HINT",
    0x4: "START",
    0x5: "WIN",
    0x6: "UPDATE_DATA",
    0x7: "UPDATE_CARD",
    0xA: "SELECT_BATTLECMD",
    0xB: "SELECT_IDLECMD",
    0xC: "SELECT_EFFECTYN",
    0xD: "SELECT_YESNO",
    0xE: "SELECT_OPTION",
    0xF: "SELECT_CARD",
    0x10: "SELECT_CHAIN",
    0x12: "SELECT_PLACE",
    0x13: "SELECT_POSITION",
    0x14: "SELECT_TRIBUTE",
    0x16: "SELECT_COUNTER",
    0x17: "SELECT_SUM",
    0x19: "SORT_CARD",
    0x1A: "SELECT_UNSELECT_CARD",
    0x1F: "CONFIRM_CARDS",
    0x20: "SHUFFLE_DECK",
    0x21: "SHUFFLE_HAND",
    0x23: "SWAP_GRAVE_DECK",
    0x25: "REVERSE_DECK",
    0x28: "NEW_TURN",
    0x29: "NEW_PHASE",
    0x32: "MOVE",
    0x35: "POS_CHANGE",
    0x36: "SET",
    0x37: "SWAP",
    0x3C: "SUMMONING",
    0x3D: "SUMMONED",
    0x3E: "SPSUMMONING",
    0x3F: "SPSUMMONED",
    0x40: "FLIPSUMMONING",
    0x41: "FLIPSUMMONED",
    0x46: "CHAINING",
    0x47: "CHAINED",
    0x48: "CHAIN_SOLVING",
    0x49: "CHAIN_SOLVED",
    0x4A: "CHAIN_END",
    0x5A: "DRAW",
    0x6E: "ATTACK",
    0x6F: "BATTLE",
}


def packet_tails(paths: list[Path]) -> dict[tuple, list[str]]:
    """(g,s,pi) -> last 4 packet message names, for report context."""
    out: dict[tuple, list[str]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("k") != "pkt":
                    continue
                raw = bytes.fromhex(r["msg"])
                t = raw[0] if raw else -1
                key = (r["g"], r["s"], r["i"] + 1)
                tail = out.setdefault(key, [])
                tail.append(_NAMES.get(t, f"0x{t:02X}"))
                if len(tail) > 4:
                    del tail[0]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", nargs=2, metavar=("PY", "RS"))
    ap.add_argument("--gate3", nargs="+", metavar="FRAMES")
    ap.add_argument("--packets", nargs="+", metavar="PACKETS", default=[])
    a = ap.parse_args(argv)
    if a.gate1:
        return gate1(Path(a.gate1[0]), Path(a.gate1[1]))
    if a.gate3:
        return gate3([Path(x) for x in a.gate3], [Path(x) for x in a.packets])
    ap.error("choose --gate1 PY RS or --gate3 FRAMES")
    return 2


if __name__ == "__main__":
    sys.exit(main())
