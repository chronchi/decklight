"""Per-turn game event summary from a stored game record's engine replay.

Raw material for the engine-eval plan's metric #7 ("interruptions shown"), tempo and
tech-card exposure metrics (docs/plans/2026-10-16-engine-eval-and-metrics.md, milestone N1).

A stored record (a ``games-*.jsonl`` line of search_match) holds ``decks``, ``seed``,
``choices`` and ``winner``. :func:`events` replays the choices in the Python reference
env (:meth:`PyEnv.replay <ygo_zero.engine.env.PyEnv.replay>`) and folds the duel's raw
core message trace (``Duel.trace``) into per-turn counters plus a chain-link list.

Binary layouts below are read off the message writers in ``third_party/ygopro-core/``
(see also the gotchas in ``../ygo-architect/CLAUDE.md``):

- MSG_NEW_TURN: player u8 (``duel.py`` keeps ``raw[1] & 1``).
- MSG_CHAINING (``processor.cpp``): code u32 + handler loc_info
  (con u8, loc u8, seq u32, pos u32) + triggering_controller u8 +
  triggering_location u8 + triggering_sequence u32 + description u64 + chain count u32
  = 33 bytes. The controller is the *triggering* controller, not necessarily the
  handler's current controller.
- MSG_CHAIN_NEGATED / MSG_CHAIN_DISABLED (``operations.cpp``): chain position u8.
- MSG_DRAW (``operations.cpp``): player u8 + count u32 + count x (code u32, pos u32).
- MSG_DAMAGE / MSG_RECOVER / MSG_PAY_LPCOST: player u8 + amount u32.
- MSG_LPUPDATE (``libduel.cpp`` ``SetLP`` only): player u8 + absolute LP u32.
- MSG_SUMMONING / MSG_SPSUMMONING / MSG_FLIPSUMMONING: code u32 (0 when face-down
  for SP) + loc_info; controller byte at offset 5. The bare MSG_SUMMONED /
  MSG_SPSUMMONED / MSG_FLIPSUMMONED carry no payload, so the summoner is taken from
  the immediately preceding SUMMONING message.
- MSG_ATTACK (``processor.cpp``): attacker loc_info + target loc_info (empty when
  direct); attacker controller at offset 1.
- MSG_MOVE: 29 bytes, code u32 + from/to loc_info + reason u32 (see
  ``knowledge.py``); a card leaving LOCATION_DECK counts as seen for ``from_con``.
"""

from __future__ import annotations

import struct
from pathlib import Path

from ygo_zero.engine import constants as C
from ygo_zero.engine import query as Q
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.engine.env import PyEnv

ROOT = Path(__file__).resolve().parents[3]
DECK_DIRS = ("pool", "gauntlet", "newcards")

# MSG_MOVE is always code + from loc_info + to loc_info + reason (see knowledge.py).
_MOVE_LEN = 1 + 4 + 10 + 10 + 4
# MSG_CHAINING fixed size: code + loc_info + tcon + tloc + tseq + desc + count.
_CHAINING_LEN = 1 + 4 + 10 + 1 + 1 + 4 + 8 + 4

COUNTERS = (
    "activations",
    "activations_opp_turn",
    "negated_by_p_suffered",
    "damage_taken",
    "lp_paid",
    "lp_recovered",
    "normal_summons",
    "special_summons",
    "attacks_declared",
    "draws",
)


def deck_path(name: str) -> Path:
    for d in DECK_DIRS:
        p = ROOT / "decks" / d / f"{name}.ydk"
        if p.is_file():
            return p
    raise FileNotFoundError(f"deck {name} not found in decks/pool|gauntlet|newcards")


def _parse_chaining(raw: bytes) -> tuple[int, int, int] | None:
    """(code, triggering controller, chain length at activation) or None."""
    if len(raw) != _CHAINING_LEN:
        return None
    (code,) = struct.unpack_from("<I", raw, 1)
    tcon = raw[1 + 4 + 10]
    (count,) = struct.unpack_from("<I", raw, 1 + 4 + 10 + 1 + 1 + 4 + 8)
    if tcon not in (0, 1):
        return None
    return code, tcon, count


def _parse_chain_pos(raw: bytes) -> int | None:
    """Chain position of a MSG_CHAIN_NEGATED / MSG_CHAIN_DISABLED."""
    if len(raw) != 2:
        return None
    return raw[1]


def _parse_draw(raw: bytes) -> tuple[int, list[int]] | None:
    """(player, drawn codes) or None."""
    if len(raw) < 6:
        return None
    player, (count,) = raw[1], struct.unpack_from("<I", raw, 2)
    if player not in (0, 1) or len(raw) != 6 + 8 * count:
        return None
    codes = [struct.unpack_from("<I", raw, 6 + 8 * i)[0] for i in range(count)]
    return player, codes


def _parse_lp(raw: bytes) -> tuple[int, int] | None:
    """(player, amount) of a MSG_DAMAGE / RECOVER / PAY_LPCOST / LPUPDATE or None."""
    if len(raw) != 6:
        return None
    player, (amount,) = raw[1], struct.unpack_from("<I", raw, 2)
    if player not in (0, 1):
        return None
    return player, amount


def _parse_move(raw: bytes) -> tuple[int, int, int] | None:
    """(code, from_con, from_loc) or None."""
    if len(raw) != _MOVE_LEN:
        return None
    (code,) = struct.unpack_from("<I", raw, 1)
    return code, raw[5], raw[6]


def _parse_attacker(raw: bytes) -> int | None:
    """Attacking player's seat from a MSG_ATTACK, or None."""
    if len(raw) != 1 + 10 + 10:
        return None
    con = raw[1]
    return con if con in (0, 1) else None


def _parse_summoner(raw: bytes) -> int | None:
    """Summoning player's seat from a MSG_*SUMMONING, or None."""
    if len(raw) != 1 + 4 + 10:
        return None
    con = raw[5]
    return con if con in (0, 1) else None


def _blank_turn(turn: int, turn_player: int) -> dict:
    return {
        "turn": turn,
        "turn_player": turn_player,
        "players": [{c: 0 for c in COUNTERS} for _ in range(2)],
    }


def events(rec: dict, engine: Engine) -> dict:
    """Replay one stored game record and return its per-turn event summary.

    ``rec`` needs ``decks`` (ydk names resolved under ``decks/`` or deck-like
    objects), ``seed``, ``choices`` and ``winner``. Raises on a truncated record,
    a replay whose winner differs from the record, or an LP accounting mismatch.
    """
    if rec.get("truncated"):
        raise ValueError(f"game {rec.get('game_idx')}: truncated record has no full trace")
    from ygo_zero.decks import load_ydk

    decks = [load_ydk(deck_path(d)) if isinstance(d, str) else d for d in rec["decks"]]
    cfg = DuelConfig(decks[0], decks[1], seed=rec["seed"])
    env = PyEnv.replay(engine, cfg, rec["choices"])
    if not env.done or env.winner != rec["winner"]:
        raise ValueError(
            f"game {rec.get('game_idx')}: replay winner {env.winner} "
            f"(done={env.done}) != record {rec['winner']}"
        )
    out = _extract(env, game_idx=rec.get("game_idx"))
    return out


def _extract(env: PyEnv, game_idx=None) -> dict:
    duel = env.duel
    assert duel is not None
    turns: list[dict] = []
    pending: list[bytes] = []  # messages before the first MSG_NEW_TURN (opening draws)
    cur: dict | None = None
    chain_links: list[dict] = []
    active: list[int] = []  # chain_links indices of the currently open chain
    seen: list[set[int]] = [set(), set()]
    tracked_lp = [env.duel.config.lp, env.duel.config.lp]
    lp_update_seen = False
    pending_summon: int | None = None  # summoner seat from the last *SUMMONING

    def turn_entry() -> dict:
        assert cur is not None
        return cur

    def apply(raw: bytes) -> None:
        nonlocal cur, lp_update_seen, pending_summon
        t = raw[0]
        if t == C.MSG_NEW_TURN:
            tp = raw[1] & 1
            cur = _blank_turn(len(turns) + 1, tp)
            turns.append(cur)
            active.clear()
            pending_summon = None
            return
        if cur is None:
            pending.append(raw)
            return
        e = turn_entry()
        if t == C.MSG_CHAINING:
            parsed = _parse_chaining(raw)
            if parsed is None:
                return
            code, con, count = parsed
            chain_links.append(
                {
                    "turn": e["turn"],
                    "turn_player": e["turn_player"],
                    "controller": con,
                    "code": code,
                    "chain_len_at_activation": count,
                    "negated": False,
                }
            )
            active.append(len(chain_links) - 1)
            e["players"][con]["activations"] += 1
            if con != e["turn_player"]:
                e["players"][con]["activations_opp_turn"] += 1
            seen[con].add(code)
        elif t in (C.MSG_CHAIN_NEGATED, C.MSG_CHAIN_DISABLED):
            pos = _parse_chain_pos(raw)
            if pos is None:
                return
            for idx in active:
                link = chain_links[idx]
                if link["chain_len_at_activation"] == pos and not link["negated"]:
                    link["negated"] = True
                    e["players"][link["controller"]]["negated_by_p_suffered"] += 1
                    break
        elif t == C.MSG_CHAIN_END:
            active.clear()
        elif t == C.MSG_DRAW:
            parsed = _parse_draw(raw)
            if parsed is None:
                return
            player, codes = parsed
            e["players"][player]["draws"] += len(codes)
            seen[player].update(c for c in codes if c)
        elif t == C.MSG_DAMAGE:
            parsed = _parse_lp(raw)
            if parsed is None:
                return
            player, amount = parsed
            e["players"][player]["damage_taken"] += amount
            tracked_lp[player] -= amount
        elif t == C.MSG_RECOVER:
            parsed = _parse_lp(raw)
            if parsed is None:
                return
            player, amount = parsed
            e["players"][player]["lp_recovered"] += amount
            tracked_lp[player] += amount
        elif t == C.MSG_PAY_LPCOST:
            parsed = _parse_lp(raw)
            if parsed is None:
                return
            player, amount = parsed
            e["players"][player]["lp_paid"] += amount
            tracked_lp[player] -= amount
        elif t == C.MSG_LPUPDATE:
            parsed = _parse_lp(raw)
            if parsed is None:
                return
            player, value = parsed
            tracked_lp[player] = value
            lp_update_seen = True
        elif t in (C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING):
            con = _parse_summoner(raw)
            if con is not None:
                pending_summon = con
        elif t == C.MSG_SUMMONED:
            if pending_summon is not None:
                e["players"][pending_summon]["normal_summons"] += 1
                pending_summon = None
        elif t == C.MSG_SPSUMMONED:
            if pending_summon is not None:
                e["players"][pending_summon]["special_summons"] += 1
                pending_summon = None
        elif t == C.MSG_FLIPSUMMONED:
            pending_summon = None  # tracked for pairing only; not a normal/special summon
        elif t == C.MSG_ATTACK:
            con = _parse_attacker(raw)
            if con is not None:
                e["players"][con]["attacks_declared"] += 1
        elif t == C.MSG_MOVE:
            parsed = _parse_move(raw)
            if parsed is None:
                return
            code, from_con, from_loc = parsed
            if from_loc == C.LOCATION_DECK and from_con in (0, 1) and code:
                seen[from_con].add(code)

    for raw in duel.trace:
        apply(raw)
        if cur is not None and pending:
            # First MSG_NEW_TURN just created turn 1: attribute the buffered
            # pre-turn messages (opening draws) to it, in order.
            buffered, pending = pending, []
            for b in buffered:
                apply(b)
    if not turns:
        raise ValueError(f"game {game_idx}: trace has no MSG_NEW_TURN")
    if pending:  # pragma: no cover - defensive; cannot happen when turns is non-empty
        raise ValueError(f"game {game_idx}: {len(pending)} messages before the first turn")

    # Q.lps reads unsigned u32, but a losing seat's LP goes below zero (damage
    # overshoot), so interpret as signed i32 like query.parse_card's _i32.
    final_lp = [v - (1 << 32) if v >= 1 << 31 else v for v in Q.lps(duel.query_field())]
    if tracked_lp != final_lp:
        raise ValueError(
            f"game {game_idx}: LP trace mismatch: tracked {tracked_lp} != queried {final_lp} "
            f"(lp_update_seen={lp_update_seen})"
        )
    if not lp_update_seen:
        start_lp = duel.config.lp
        for p in (0, 1):
            dmg = sum(t["players"][p]["damage_taken"] for t in turns)
            paid = sum(t["players"][p]["lp_paid"] for t in turns)
            rec = sum(t["players"][p]["lp_recovered"] for t in turns)
            if dmg + paid - rec != start_lp - final_lp[p]:
                raise ValueError(
                    f"game {game_idx}: seat {p} LP formula mismatch: "
                    f"{dmg}+{paid}-{rec} != {start_lp}-{final_lp[p]}"
                )
    return {
        "game_idx": game_idx,
        "seed": env.duel.config.seed,
        "winner": env.winner,
        "final_lp": final_lp,
        "turns": turns,
        "chain_links": chain_links,
        "cards_seen": [sorted(seen[0]), sorted(seen[1])],
    }
