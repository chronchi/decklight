"""Engine-message cues for the web client (server side).

The web client plays the game as a sequence of short animated cues (Master Duel
style). The board snapshots plus decision-`log` events (`view.build_events`) only
say what the players *decided*; cues say what the *engine* did: moves, chains,
attacks, damage, phases. The source is `Duel.trace` (every raw core message in
order); message layouts follow `../ygo-architect/CLAUDE.md` ("message-format
gotchas": `loc_info` = ctrl u8 + loc u8 + seq u32 + pos u32 = 10 bytes,
MSG_DRAW count u32 + 8 bytes per card, MSG_NEW_PHASE u16) and the core in
`third_party/ygopro-core` (read-only) with `../edopro/gframe/duelclient.cpp`
(`ClientAnalyze`) as the authoritative per-message reference.

`build_cues(env, viewer, since=0, reveal_all=False)` returns the cues for
`env.duel.trace[since:]` and the new trace length. Unknown/unsupported messages
never raise: they are skipped and counted (see `cues_for_trace`).

Two fields need care:

- Codes: MSG_MOVE / MSG_SUMMONING / MSG_CHAINING / MSG_CONFIRM_CARDS name the
  card; MSG_ATTACK / MSG_BATTLE / MSG_BECOME_TARGET carry only `loc_info`. Those
  resolve codes through a slot memory fed by the code-carrying messages seen
  earlier in the slice (empty on a miss: code 0, the zone still identifies the
  slot for the client). MZONE/SZONE slots are stable zones; pile sequences
  (hand/deck/...) shift without messages, so the memory only trusts arrivals.
- Hidden information: a code is shown iff `reveal_all`, the card is announced
  (activation/confirm: public by the rules), or it is public where the event
  leaves it (face-up field, GY, face-up banished, own hand and own face-downs).
  A card that ends hidden stays code 0 / name "" even when it came from a
  public zone (GY -> opponent's hand): no cue ever names an opponent hidden
  card (their face-down sets, hand, deck and face-down banished cards).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from decklight.engine import constants as C
from decklight.engine import messages as M
from decklight.web.view import DB, PHASE_NAMES, _uid

MSG_NAMES: dict[int, str] = {v: k for k, v in vars(C).items() if k.startswith("MSG_") and isinstance(v, int)}


def message_name(t: int) -> str:
    """`MSG_FOO` for a known message type, else `MSG_<hex>`."""
    return MSG_NAMES.get(t, f"MSG_{t:#x}")


KNOWN_CUE_TYPES = frozenset(
    {
        "turn",
        "phase",
        "draw",
        "move",
        "summon",
        "activate",
        "chain_solving",
        "chain_solved",
        "chain_negated",
        "chain_disabled",
        "chain_end",
        "target",
        "attack",
        "battle",
        "damage",
        "recover",
        "lp",
        "pay",
        "confirm",
        "shuffle",
        "win",
    }
)

# Brief short labels, via view.py's phase grouping (battle sub-phases -> "Battle").
_SHORT_PHASE = {
    "draw": "Draw",
    "standby": "Standby",
    "main1": "Main 1",
    "battle": "Battle",
    "main2": "Main 2",
    "end": "End",
}

_ZONE_NAMES = {
    C.LOCATION_HAND: "hand",
    C.LOCATION_DECK: "deck",
    C.LOCATION_EXTRA: "extra",
    C.LOCATION_MZONE: "mzone",
    C.LOCATION_SZONE: "szone",
    C.LOCATION_GRAVE: "grave",
    C.LOCATION_REMOVED: "banished",
    C.LOCATION_OVERLAY: "overlay",
}

# Summon-reason flags that mark a move as material (not the summon itself).
_MATERIAL_REASONS = (
    C.REASON_FUSION
    | C.REASON_SYNCHRO
    | C.REASON_XYZ
    | C.REASON_LINK
    | C.REASON_RITUAL
    | C.REASON_MATERIAL
    | C.REASON_RELEASE
)

# One-pile locations whose sequences shift without messages: arrivals update the
# slot memory, departures do not clear it.
_PILES = frozenset(
    {
        C.LOCATION_DECK,
        C.LOCATION_HAND,
        C.LOCATION_GRAVE,
        C.LOCATION_REMOVED,
        C.LOCATION_EXTRA,
    }
)


def _side(ctrl: int, viewer: int) -> str:
    return "me" if ctrl == viewer else "opp"


def _zone(ctrl: int, loc: int, seq: int, viewer: int) -> dict:
    name = _ZONE_NAMES.get(loc, "unknown")
    if loc == C.LOCATION_SZONE and seq == 5:
        name = "field"  # the field-spell zone (matches view.py `_where`)
    return {"side": _side(ctrl, viewer), "loc": name, "seq": seq}


def _end_visible(ctrl: int, loc: int, pos: int, viewer: int) -> bool:
    """Whether a card at one end of an event is public to `viewer`.

    Mirrors the model's own visibility (`obs.Board._public`) plus the owner
    rule: your own face-down cards and hand are visible to you; the opponent's
    face-down sets, hand, deck and face-down banished cards are not. GY and
    overlay units are always public.
    """
    if loc in (C.LOCATION_GRAVE, C.LOCATION_OVERLAY):
        return True
    if loc == 0:
        return False
    faceup = bool(pos & C.POS_FACEUP)
    if loc in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED, C.LOCATION_DECK, C.LOCATION_EXTRA):
        return faceup or ctrl == viewer
    if loc == C.LOCATION_HAND:
        return ctrl == viewer
    return ctrl == viewer


def _ref(
    code: int,
    ctrl: int,
    loc: int,
    seq: int,
    pos: int,
    viewer: int,
    reveal_all: bool,
    *,
    force_show: bool = False,
) -> dict:
    """A card reference. Hidden cards become code 0 / name "" (never leak).

    `owner` is the card's controller as the viewer sees it ("me"/"opp"); the
    core messages name no separate owner, so controlled == owned here. `uid`
    matches view.py `_uid`, so the client can find the board element.
    """
    show = reveal_all or force_show or _end_visible(ctrl, loc, pos, viewer)
    if not show:
        code = 0
    name = "" if code == 0 else DB.names().get(code, f"#{code}")
    return {
        "code": code,
        "name": name,
        "owner": _side(ctrl, viewer),
        "zone": _zone(ctrl, loc, seq, viewer),
        "uid": _uid(ctrl, loc, seq, code),
    }


def _loc_info(r: M._Reader) -> tuple[int, int, int, int]:
    return (r.u8(), r.u8(), r.u32(), r.u32())


def _move_reason(reason: int, prev_loc: int, cur_loc: int, cur_pos: int) -> str:
    if reason & C.REASON_DESTROY:
        return "destroyed"
    if reason & _MATERIAL_REASONS:
        return "material"
    if reason & C.REASON_SUMMON:
        return "summon"
    if reason & C.REASON_SPSUMMON:
        return "spsummon"
    if cur_loc in (C.LOCATION_MZONE, C.LOCATION_SZONE) and cur_pos & C.POS_FACEDOWN:
        return "set"
    if cur_loc == C.LOCATION_REMOVED:
        return "banished"
    if reason & C.REASON_RETURN:
        return "returned"
    if cur_loc in (C.LOCATION_DECK, C.LOCATION_EXTRA) and prev_loc != cur_loc:
        return "returned"
    if cur_loc == C.LOCATION_HAND:
        return "added" if prev_loc == C.LOCATION_DECK else "returned"
    if cur_loc == C.LOCATION_GRAVE:
        return "sent"
    return "other"


def _card_type(code: int, cards: dict | None) -> int:
    if cards is not None:
        cd = cards.get(code)
        if cd is not None:
            return int(getattr(cd, "type", 0))
    st = DB.stats().get(int(code))
    return int(st[5]) if st is not None else 0


def _sp_how(code: int, cards: dict | None) -> str:
    t = _card_type(code, cards)
    if t & C.TYPE_LINK:
        return "link"
    if t & C.TYPE_XYZ:
        return "xyz"
    if t & C.TYPE_SYNCHRO:
        return "synchro"
    if t & C.TYPE_FUSION:
        return "fusion"
    if t & C.TYPE_RITUAL:
        return "ritual"
    if t & C.TYPE_PENDULUM:
        return "pendulum"
    return "special"


def _i32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def cues_for_trace(
    trace: Sequence[bytes],
    *,
    viewer: int,
    since: int = 0,
    reveal_all: bool = False,
    lp0: int = 8000,
    cards: dict | None = None,
) -> tuple[list[dict], int, Counter]:
    """Cues for `trace[since:]`, the new trace length, and skipped-message counts.

    `viewer` is the absolute seat cues are filtered for; `lp0` is both players'
    starting LP (damage/recover/pay cues carry a running `lp_after`). `cards`
    maps codes to card data (for summon `how`); without it the DB is used.
    """
    cues: list[dict] = []
    skipped: Counter = Counter()
    turn = 0
    lp = [lp0, lp0]
    chain_card: dict[int, dict] = {}  # link -> activating card ref (this chain)
    last_link: int | None = None  # link of the most recent MSG_CHAINING
    pending: list[tuple[int, int]] = []  # (code, ctrl) summons awaiting their MOVE
    last_release: dict[int, int] = {}  # ctrl -> trace index of last RELEASE move
    slot_code: dict[tuple[int, int, int], int] = {}  # (ctrl, loc, seq) -> code

    def skip(raw: bytes) -> None:
        skipped[message_name(raw[0]) if raw else "MSG_empty"] += 1

    def forget(ctrl: int, loc: int, seq: int) -> None:
        if loc not in _PILES:  # stable zones: a departure empties the slot
            slot_code.pop((ctrl, loc, seq), None)

    for n in range(len(trace)):
        # The whole prefix replays so LP/turn/slot state is exact at `since`;
        # only messages at n >= since emit cues (the server advances a cursor).
        raw = trace[n]
        emit = n >= since
        if not raw:
            skip(raw)
            continue
        t = raw[0]
        try:
            if t == C.MSG_NEW_TURN:
                p = M._Reader(raw).u8()
                turn += 1
                if emit:
                    cues.append({"t": "turn", "n": n, "turn": turn, "player": _side(p & 1, viewer)})
            elif t == C.MSG_NEW_PHASE:
                phase = _SHORT_PHASE.get(PHASE_NAMES.get(M._Reader(raw).u16(), "main1"), "Main 1")
                if emit:
                    cues.append({"t": "phase", "n": n, "phase": phase})
            elif t == C.MSG_DRAW:
                r = M._Reader(raw)
                p, count = r.u8(), r.u32()
                refs = [
                    _ref(code, p, C.LOCATION_HAND, -1, pos, viewer, reveal_all)
                    for code, pos in ((r.u32(), r.u32()) for _ in range(count))
                ]
                if emit:
                    cues.append(
                        {"t": "draw", "n": n, "player": _side(p, viewer), "count": count, "cards": refs}
                    )
            elif t == C.MSG_MOVE:
                r = M._Reader(raw)
                code = r.u32()
                pc, pl, ps, pp = _loc_info(r)
                cc, cl, cs, cp = _loc_info(r)
                reason = r.u32()
                if reason & C.REASON_RELEASE:
                    last_release[pc] = n
                if pc == cc and pl == cl and pl in _PILES:
                    skip(raw)  # reorder bookkeeping inside one pile
                    continue
                if reason & (C.REASON_SUMMON | C.REASON_SPSUMMON | C.REASON_FLIP):
                    hit = next(
                        (
                            i
                            for i, (sc, ss) in enumerate(pending)
                            if ss == cc and (sc == code or sc == 0 or code == 0)
                        ),
                        None,
                    )
                    if hit is not None:
                        del pending[hit]  # the summon's own move; summon cue covers it
                        forget(pc, pl, ps)
                        slot_code[(cc, cl, cs)] = code
                        continue
                forget(pc, pl, ps)
                if code:
                    slot_code[(cc, cl, cs)] = code
                else:
                    slot_code.pop((cc, cl, cs), None)
                if not emit:
                    continue
                # Destination governs: a card that ends hidden (opponent's
                # hand/deck, face-down zones) stays code 0 even when it came
                # from a public zone (e.g. GY -> hand), so no cue ever names
                # an opponent hidden card. Location 0 (leaving the duel) has
                # no hidden after-state: fall back to the source.
                if cl == 0:
                    show = reveal_all or _end_visible(pc, pl, pp, viewer)
                else:
                    show = reveal_all or _end_visible(cc, cl, cp, viewer)
                cues.append(
                    {
                        "t": "move",
                        "n": n,
                        "card": _ref(code, cc, cl, cs, cp, viewer, reveal_all, force_show=show),
                        "from": _zone(pc, pl, ps, viewer),
                        "to": _zone(cc, cl, cs, viewer),
                        "reason": _move_reason(reason, pl, cl, cp),
                    }
                )
            elif t in (C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING):
                r = M._Reader(raw)
                code = r.u32()
                fc, fl, fs, _fp = _loc_info(r)
                pending.append((code, fc))
                if code:
                    slot_code[(fc, fl, fs)] = code
                if t == C.MSG_FLIPSUMMONING:
                    how = "flip"
                elif t == C.MSG_SUMMONING:
                    how = "tribute" if n - last_release.get(fc, -(10**9)) <= 8 else "normal"
                else:
                    how = _sp_how(code, cards)
                # The summon's own move may precede the announcement (normal
                # summons: hand -> mzone with reason 0) or follow it (special
                # summons with REASON_SPSUMMON, caught by `pending` above): a
                # summon cue replaces the matching move cue of the same card.
                dest: tuple[int, int] | None = None
                from_zone = _zone(fc, fl, fs, viewer)
                if code and emit:
                    for i in range(len(cues) - 1, -1, -1):
                        c = cues[i]
                        if n - c["n"] > 4:
                            break
                        if (
                            c["t"] == "move"
                            and c["card"]["code"] == code
                            and c["card"]["owner"] == _side(fc, viewer)
                            and c["to"]["loc"] == "mzone"
                        ):
                            to = c["to"]
                            dest = (viewer if to["side"] == "me" else 1 - viewer, to["seq"])
                            from_zone = c["from"]
                            del cues[i]
                            break
                if dest is not None:
                    card = _ref(
                        code,
                        dest[0],
                        C.LOCATION_MZONE,
                        dest[1],
                        C.POS_FACEUP_ATTACK,
                        viewer,
                        reveal_all,
                        force_show=True,
                    )
                else:
                    card = _ref(
                        code,
                        fc,
                        C.LOCATION_MZONE,
                        -1,
                        C.POS_FACEUP_ATTACK,
                        viewer,
                        reveal_all,
                        force_show=code != 0,
                    )  # summons arrive face-up
                if emit:
                    cues.append({"t": "summon", "n": n, "card": card, "how": how, "from": from_zone})
            elif t == C.MSG_CHAINING:
                r = M._Reader(raw)
                code = r.u32()
                hc, hl, hs, hp = _loc_info(r)
                r.u8(), r.u8(), r.u32()  # triggering controller/location/sequence
                desc = r.u64()
                link = r.u32()
                effect = DB.desc_text(desc >> 20, desc & 0xFFFFF) or None
                ref = _ref(
                    code, hc, hl, hs, hp, viewer, reveal_all, force_show=True
                )  # activation announces the card
                chain_card[link] = ref
                last_link = link
                if code:
                    slot_code[(hc, hl, hs)] = code
                if emit:
                    cues.append({"t": "activate", "n": n, "card": ref, "link": link, "effect": effect})
            elif t == C.MSG_CHAIN_SOLVING:
                link = M._Reader(raw).u8()
                if emit:
                    cues.append({"t": "chain_solving", "n": n, "link": link})
            elif t == C.MSG_CHAIN_SOLVED:
                link = M._Reader(raw).u8()
                if emit:
                    cues.append({"t": "chain_solved", "n": n, "link": link})
            elif t == C.MSG_CHAIN_NEGATED:
                link = M._Reader(raw).u8()
                if emit:
                    cues.append({"t": "chain_negated", "n": n, "link": link})
            elif t == C.MSG_CHAIN_DISABLED:
                link = M._Reader(raw).u8()
                if emit:
                    cues.append({"t": "chain_disabled", "n": n, "link": link})
            elif t == C.MSG_CHAIN_END:
                chain_card.clear()
                last_link = None
                if emit:
                    cues.append({"t": "chain_end", "n": n})
            elif t == C.MSG_BECOME_TARGET:
                r = M._Reader(raw)
                targets = [_loc_info(r) for _ in range(r.u32())]
                source = chain_card.get(last_link) if last_link is not None else None
                if emit:
                    cues.append(
                        {
                            "t": "target",
                            "n": n,
                            "source": source,
                            "cards": [
                                _ref(slot_code.get((tc, tl, ts), 0), tc, tl, ts, tp, viewer, reveal_all)
                                for tc, tl, ts, tp in targets
                            ],
                        }
                    )
            elif t == C.MSG_ATTACK:
                r = M._Reader(raw)
                ac, al, aseq, ap = _loc_info(r)
                tc, tl, ts, tp = _loc_info(r)
                if not emit:
                    continue
                attacker = _ref(slot_code.get((ac, al, aseq), 0), ac, al, aseq, ap, viewer, reveal_all)
                target = (
                    None
                    if tl == 0
                    else _ref(slot_code.get((tc, tl, ts), 0), tc, tl, ts, tp, viewer, reveal_all)
                )
                cues.append({"t": "attack", "n": n, "attacker": attacker, "target": target})
            elif t == C.MSG_BATTLE:
                r = M._Reader(raw)
                ac, al, aseq, ap = _loc_info(r)
                aatk, adef = _i32(r.u32()), _i32(r.u32())
                r.u8()
                tc, tl, ts, tp = _loc_info(r)
                datk, ddef = _i32(r.u32()), _i32(r.u32())
                r.u8()
                if not emit:
                    continue
                # Damage calculation is public: stats ride along even when a
                # face-down defender's code stays hidden.
                attacker = _ref(slot_code.get((ac, al, aseq), 0), ac, al, aseq, ap, viewer, reveal_all)
                attacker["atk"], attacker["def"] = aatk, adef
                if tl == 0:
                    defender = None
                else:
                    defender = _ref(slot_code.get((tc, tl, ts), 0), tc, tl, ts, tp, viewer, reveal_all)
                    defender["atk"], defender["def"] = datk, ddef
                cues.append({"t": "battle", "n": n, "attacker": attacker, "target": defender})
            elif t == C.MSG_DAMAGE:
                r = M._Reader(raw)
                p, amount = r.u8(), r.u32()
                lp[p] = max(0, lp[p] - amount)
                if emit:
                    cues.append(
                        {
                            "t": "damage",
                            "n": n,
                            "player": _side(p, viewer),
                            "amount": amount,
                            "lp_after": lp[p],
                        }
                    )
            elif t == C.MSG_RECOVER:
                r = M._Reader(raw)
                p, amount = r.u8(), r.u32()
                lp[p] += amount
                if emit:
                    cues.append(
                        {
                            "t": "recover",
                            "n": n,
                            "player": _side(p, viewer),
                            "amount": amount,
                            "lp_after": lp[p],
                        }
                    )
            elif t == C.MSG_LPUPDATE:
                r = M._Reader(raw)
                p, val = r.u8(), r.u32()
                lp[p] = _i32(val)
                if emit:
                    cues.append({"t": "lp", "n": n, "player": _side(p, viewer), "lp": lp[p]})
            elif t == C.MSG_PAY_LPCOST:
                r = M._Reader(raw)
                p, cost = r.u8(), r.u32()
                lp[p] = max(0, lp[p] - cost)
                if emit:
                    cues.append(
                        {"t": "pay", "n": n, "player": _side(p, viewer), "amount": cost, "lp_after": lp[p]}
                    )
            elif t == C.MSG_CONFIRM_CARDS:
                r = M._Reader(raw)
                p, count = r.u8(), r.u32()
                cards_out = []
                for _ in range(count):
                    code, cc, cl, cs = r.u32(), r.u8(), r.u8(), r.u32()
                    if code:
                        slot_code[(cc, cl, cs)] = code
                    if emit:
                        cards_out.append(
                            _ref(code, cc, cl, cs, C.POS_FACEUP, viewer, reveal_all, force_show=True)
                        )  # revealed
                if emit:
                    cues.append({"t": "confirm", "n": n, "player": _side(p, viewer), "cards": cards_out})
            elif t in (C.MSG_SHUFFLE_HAND, C.MSG_SHUFFLE_DECK):
                p = M._Reader(raw).u8()
                if emit:
                    cues.append(
                        {
                            "t": "shuffle",
                            "n": n,
                            "player": _side(p, viewer),
                            "loc": "hand" if t == C.MSG_SHUFFLE_HAND else "deck",
                        }
                    )
            elif t == C.MSG_WIN:
                r = M._Reader(raw)
                w, reason = r.u8(), r.u8()
                if emit:
                    cues.append(
                        {
                            "t": "win",
                            "n": n,
                            "winner": "draw" if w == 2 else _side(w, viewer),
                            "reason": reason,
                        }
                    )
            else:
                skip(raw)
        except Exception:  # noqa: BLE001 -- unknown/unsupported messages are skipped, never raised
            skip(raw)
    return cues, len(trace), skipped


def build_cues(env, viewer: int, since: int = 0, reveal_all: bool = False) -> tuple[list[dict], int]:
    """Cues for `env.duel.trace[since:]` and the new trace length.

    `viewer` is the absolute seat the cues are filtered for (hidden opponent
    cards become code 0 / name ""); `reveal_all` shows everything (spectator
    replays). Each cue carries `"t"` (type) and `"n"` (trace index).
    """
    trace = env.duel.trace
    lp0 = getattr(getattr(env, "config", None), "lp", 8000)
    cards = getattr(getattr(getattr(env, "duel", None), "engine", None), "cards", None)
    cues, new_len, _ = cues_for_trace(
        trace, viewer=viewer, since=since, reveal_all=reveal_all, lp0=lp0, cards=cards
    )
    return cues, new_len
