"""Board/prompt/event builders for the web client (W1 server side).

The human's view is built from `obs.Board(duel, human)` — the same visibility
rules as the model's own observation — over the full zone layout from `query.py`.
`uid` is `<controller>:<location>:<sequence>:<code>`: stable while a card stays
in one place (a move changes location/sequence; a different card reusing a slot
changes the code), so the client can animate moves. The core exposes no
per-card identity beyond (controller, location, sequence, code) and no
counters, so `counters` is always 0.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.obs import Board
from ygo_zero.engine.resources import default_cdbs

ROOT = Path(__file__).resolve().parents[3]
STRINGS_CONF = ROOT / "third_party" / "edopro-strings.conf"

PHASE_NAMES = {
    C.PHASE_DRAW: "draw", C.PHASE_STANDBY: "standby", C.PHASE_MAIN1: "main1",
    C.PHASE_BATTLE_START: "battle", C.PHASE_BATTLE_STEP: "battle",
    C.PHASE_DAMAGE: "battle", C.PHASE_DAMAGE_CAL: "battle", C.PHASE_BATTLE: "battle",
    C.PHASE_MAIN2: "main2", C.PHASE_END: "end",
}

_LOC_ZONES = {
    C.LOCATION_HAND: "hand", C.LOCATION_GRAVE: "gy", C.LOCATION_REMOVED: "banished",
    C.LOCATION_EXTRA: "extra", C.LOCATION_DECK: "deck", C.LOCATION_OVERLAY: "extra",
}

_VERBS = {
    Kind.SUMMON: "normal", Kind.SPSUMMON: "special", Kind.MSET: "set",
    Kind.SSET: "set_spell", Kind.REPOS: "position", Kind.ACTIVATE: "activate",
    Kind.TO_BATTLE: "to_battle", Kind.TO_MAIN2: "to_main2", Kind.TO_END: "end_turn",
    Kind.PASS: "pass", Kind.YES: "yes", Kind.NO: "no", Kind.OPTION: "option",
    Kind.SELECT: "select", Kind.UNSELECT: "unselect", Kind.FINISH: "finish",
    Kind.CANCEL: "cancel", Kind.PLACE: "place", Kind.POSITION: "position",
    Kind.NUMBER: "announce", Kind.ATTRIBUTE: "announce", Kind.RACE: "announce",
    Kind.DECLARE: "announce", Kind.SORT: "sort", Kind.COUNTER: "select",
    Kind.RPS: "select",
}

_SAY = {
    Kind.SUMMON: "Normal Summons", Kind.SPSUMMON: "Special Summons",
    Kind.REPOS: "changes the position of", Kind.MSET: "sets a monster",
    Kind.SSET: "sets a spell/trap", Kind.ACTIVATE: "activates",
    Kind.ATTACK: "attacks with", Kind.TO_BATTLE: "enters the Battle Phase",
    Kind.TO_MAIN2: "goes to Main Phase 2", Kind.TO_END: "ends the turn",
    Kind.YES: "uses the effect of", Kind.NO: "declines the effect of",
    Kind.OPTION: "chooses", Kind.SELECT: "selects", Kind.UNSELECT: "unselects",
    Kind.FINISH: "finishes the selection", Kind.CANCEL: "cancels",
    Kind.PLACE: "places in", Kind.POSITION: "chooses position for",
    Kind.NUMBER: "declares the number", Kind.ATTRIBUTE: "declares the attribute",
    Kind.RACE: "declares the type", Kind.DECLARE: "declares the card",
    Kind.SORT: "orders", Kind.COUNTER: "removes counters from",
    Kind.RPS: "plays rock-paper-scissors", Kind.PASS: "passes",
}

# Generic wording for the bot's hidden actions (the client learns nothing).
_HIDDEN_SAY = {
    Kind.ACTIVATE: "activates a face-down card", Kind.SELECT: "selects a card",
    Kind.UNSELECT: "selects a card", Kind.SUMMON: "summons a monster",
    Kind.SPSUMMON: "special summons a monster", Kind.MSET: "sets a monster",
    Kind.SSET: "sets a spell/trap", Kind.REPOS: "changes a monster's position",
    Kind.POSITION: "chooses a position", Kind.YES: "uses a hidden effect",
    Kind.NO: "declines a hidden effect", Kind.OPTION: "chooses an effect",
    Kind.ATTACK: "attacks with a hidden monster",
}


class _Db:
    """Card names, effect texts and description strings (lazy, cached)."""

    def __init__(self) -> None:
        self._names: dict[int, str] | None = None
        self._texts: dict[int, str] | None = None
        self._sys: dict[int, str] | None = None
        self._card_strs: dict[int, list[str]] | None = None
        self._stats: dict[int, tuple] | None = None

    def names(self) -> dict[int, str]:
        if self._names is None:
            out: dict[int, str] = {}
            for path in default_cdbs():
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    for code, name in con.execute("SELECT id, name FROM texts").fetchall():
                        out[int(code)] = str(name)
                finally:
                    con.close()
            self._names = out
        return self._names

    def stats(self) -> dict[int, tuple]:
        """code -> (attack, defense, level, rank, link, type) from the card DB."""
        if self._stats is None:
            out: dict[int, tuple] = {}
            for path in default_cdbs():
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    rows = con.execute(
                        "SELECT id, type, atk, def, level FROM datas").fetchall()
                finally:
                    con.close()
                for code, type_, atk, dfn, level in rows:
                    level = int(level)
                    lv = -(level & 0xFF) if level < 0 else level & 0xFF
                    rank = link_rating = 0
                    if type_ & C.TYPE_XYZ:
                        rank = lv
                        lv = 0
                    if type_ & C.TYPE_LINK:
                        link_rating = lv
                        lv = 0
                        dfn = 0
                    out[int(code)] = (int(atk), int(dfn), lv, rank, link_rating, int(type_))
            self._stats = out
        return self._stats

    def effect_text(self, code: int) -> str:
        if self._texts is None:
            out: dict[int, str] = {}
            for path in default_cdbs():
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    for c, desc in con.execute("SELECT id, desc FROM texts").fetchall():
                        out.setdefault(int(c), str(desc or ""))
                finally:
                    con.close()
            self._texts = out
        import re

        return re.sub(r"\s+", " ", self._texts.get(int(code), "")).strip()

    def desc_text(self, hi: int, lo: int) -> str:
        """Text of an engine description id (card str<lo+1> or system string lo)."""
        hi, lo = int(hi), int(lo)
        if hi <= 0:
            if self._sys is None:
                sys: dict[int, str] = {}
                if STRINGS_CONF.exists():
                    for line in STRINGS_CONF.read_text(encoding="utf-8").splitlines():
                        if line.startswith("!system "):
                            _, num, *rest = line.split(" ", 2)
                            if num.isdigit() and rest:
                                sys[int(num)] = rest[0].strip()
                self._sys = sys
            return self._sys.get(lo, "")
        if self._card_strs is None:
            out: dict[int, list[str]] = {}
            cols = ", ".join(f"str{i}" for i in range(1, 17))
            for path in default_cdbs():
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    for row in con.execute(f"SELECT id, {cols} FROM texts").fetchall():
                        out[int(row[0])] = [str(x or "") for x in row[1:]]
                finally:
                    con.close()
            self._card_strs = out
        strs = self._card_strs.get(hi, [])
        t = strs[lo].strip() if 0 <= lo < len(strs) else ""
        return "" if "%" in t else t


DB = _Db()


# --- cards ---


def _uid(controller: int, location: int, sequence: int, code: int) -> str:
    return f"{controller}:{location:#x}:{sequence}:{code}"


def _blank_card(uid: str) -> dict:
    return {"uid": uid, "code": 0, "name": None, "faceup": False, "pos": None,
            "atk": 0, "def": 0, "level": 0, "rank": 0, "link": 0,
            "counters": 0, "overlays": []}


def _pos_of(location: int, position: int) -> str | None:
    fd = bool(position & C.POS_FACEDOWN)
    if location == C.LOCATION_MZONE:
        if fd:
            return "set"
        if position & C.POS_FACEUP_ATTACK:
            return "atk"
        return "def"
    if location in (C.LOCATION_SZONE,):
        return "set" if fd else None
    return None


def _full_card(controller: int, location: int, sequence: int, code: int, *,
               faceup: bool, pos: str | None, atk: int = 0, dfn: int = 0,
               level: int = 0, rank: int = 0, link: int = 0,
               overlays: list | None = None) -> dict:
    names = DB.names()
    return {"uid": _uid(controller, location, sequence, code), "code": code,
            "name": names.get(code, f"#{code}"), "faceup": faceup, "pos": pos,
            "atk": atk, "def": dfn, "level": level, "rank": rank, "link": link,
            "counters": 0, "overlays": overlays or []}


def _state_card(c, *, visible: bool) -> dict:
    """A queried board card as the viewer sees it (hidden zone cards: code 0)."""
    uid = _uid(c.controller, c.location, c.sequence, c.code if visible else 0)
    if not visible:
        d = _blank_card(uid)
        d["pos"] = _pos_of(c.location, c.position)
        return d
    overlays = [_full_card(c.controller, C.LOCATION_OVERLAY, i, code, faceup=True, pos=None)
                for i, code in enumerate(c.overlay)]
    faceup = True if c.location == C.LOCATION_HAND else bool(c.position & C.POS_FACEUP)
    return _full_card(c.controller, c.location, c.sequence, c.code, faceup=faceup,
                      pos=_pos_of(c.location, c.position), atk=c.attack, dfn=c.defense,
                      level=c.level, rank=c.rank, link=c.link, overlays=overlays)


def _named_card(code: int, uid: str) -> dict:
    """A card known only by code (history entries): stats from the DB when known."""
    st = DB.stats().get(int(code))
    if st is None:
        return _full_card(0, 0, 0, int(code), faceup=True, pos=None)
    atk, dfn, level, rank, link, _type = st
    d = _full_card(0, 0, 0, int(code), faceup=True, pos=None,
                   atk=atk, dfn=dfn, level=level, rank=rank, link=link)
    d["uid"] = uid
    return d


def _visible(controller: int, c, me: int) -> bool:
    return controller == me or Board._public(c)


# --- view ---


def build_view(env, human_seat: int) -> dict:
    """The human's board view (hidden hand/banished cards are null entries)."""
    duel = env.duel
    me = human_seat
    q = {(p, loc): duel.query_location(p, loc) for p in (0, 1)
         for loc in (C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
                     C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA,
                     C.LOCATION_DECK)}
    field = duel.query_field()
    from ygo_zero.engine.query import lps

    lp = lps(field)
    from ygo_zero.engine.query import parse_chain

    chain = parse_chain(field)

    def side(p: int) -> dict:
        mine = p == me
        hand = []
        for c in q[(p, C.LOCATION_HAND)]:
            if c is None:
                continue
            hand.append(_state_card(c, visible=True) if mine or Board._public(c) else None)
        mz = q[(p, C.LOCATION_MZONE)]
        mzone = [_slot(c, mine, p) for c in list(mz[:5]) + [None] * (5 - len(mz[:5]))]
        emz = [_slot(c, mine, p) for c in list(mz[5:7]) + [None] * (2 - len(mz[5:7]))]
        sz = q[(p, C.LOCATION_SZONE)]
        szone = [_slot(c, mine, p) for c in list(sz[:5]) + [None] * (5 - len(sz[:5]))]
        fz = sz[5] if len(sz) > 5 else None
        fcard = _slot(fz, mine, p) if fz is not None else None
        gy = [_state_card(c, visible=True) for c in q[(p, C.LOCATION_GRAVE)] if c is not None]
        ban = []
        for c in q[(p, C.LOCATION_REMOVED)]:
            if c is None:
                continue
            ban.append(_state_card(c, visible=True)
                       if mine or Board._public(c) else None)
        extra_all = [c for c in q[(p, C.LOCATION_EXTRA)] if c is not None]
        if mine:
            extra = [_state_card(c, visible=True) for c in extra_all]
        else:
            extra = [_state_card(c, visible=True) for c in extra_all
                     if c.position & C.POS_FACEUP]
        deck_n = sum(c is not None for c in q[(p, C.LOCATION_DECK)])
        return {"lp": max(0, lp[p] - (1 << 32) if lp[p] >= 1 << 31 else lp[p]), "deck": deck_n, "extra_count": len(extra_all),
                "hand": hand, "mzone": mzone, "emz": emz, "szone": szone,
                "field": fcard, "gy": gy, "banished": ban, "extra": extra}

    def _slot(c, mine: bool, p: int) -> dict | None:
        if c is None:
            return None
        if mine or Board._public(c):
            return _state_card(c, visible=True)
        return _state_card(c, visible=False)

    chain_out = []
    for ch in chain:
        found = None
        for p in (0, 1):
            for loc in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_HAND,
                        C.LOCATION_GRAVE, C.LOCATION_REMOVED):
                for c in q[(p, loc)]:
                    if (c is not None and c.controller == ch.controller
                            and c.sequence == ch.sequence and c.code == ch.code):
                        found = c
                        break
        # Every chained card was announced, so its code is public even while
        # the card itself is still hidden (e.g. a hand trap); the slot lookup
        # only fills in live stats when the same card is still there.
        card = (_state_card(found, visible=True) if found is not None
                else _named_card(ch.code, _uid(ch.controller, ch.location,
                                               ch.sequence, ch.code)))
        chain_out.append({"card": card,
                          "controller": "you" if ch.controller == me else "bot"})
    return {"turn": duel.turn, "phase": PHASE_NAMES.get(duel.phase, "main1"),
            "turn_player": "you" if duel.turn_player == me else "bot",
            "you": side(me), "bot": side(1 - me), "chain": chain_out}


# --- prompt ---


def _prompt_kind(msg) -> str:
    if isinstance(msg, M.SelectIdleCmd):
        return "idle"
    if isinstance(msg, M.SelectBattleCmd):
        return "battle"
    if isinstance(msg, M.SelectChain):
        return "chain"
    if isinstance(msg, (M.SelectEffectYn,)):
        return "effectyn"
    if isinstance(msg, M.SelectYesNo):
        return "yesno"
    if isinstance(msg, M.SelectOption):
        return "option"
    if isinstance(msg, (M.SelectCard, M.SelectTribute, M.SelectSum, M.SelectUnselectCard)):
        return "card"
    if isinstance(msg, M.SelectPlace):
        return "place"
    if isinstance(msg, M.SelectPosition):
        return "position"
    if isinstance(msg, (M.AnnounceNumber, M.AnnounceRace, M.AnnounceAttrib,
                        M.AnnounceCard, M.RockPaperScissors)):
        return "announce"
    if isinstance(msg, M.SortCards):
        return "sort"
    if isinstance(msg, M.SelectCounter):
        return "counter"
    return "card"


def _select_hint(env, human_seat: int) -> str:
    """The engine's select hint (MSG_HINT HINT_SELECTMSG) for the pending decision."""
    for raw in reversed(env.duel.trace):
        if raw[0] != C.MSG_HINT or len(raw) < 11:
            continue
        if raw[1] == C.HINT_SELECTMSG and raw[2] == human_seat:
            import struct

            (data,) = struct.unpack_from("<Q", raw, 3)
            return DB.desc_text(data >> 20, data & 0xFFFFF)
    return ""


def _phase_label(phase: str) -> str:
    return {"draw": "Draw Phase", "standby": "Standby Phase", "main1": "Main Phase 1",
            "battle": "Battle Phase", "main2": "Main Phase 2",
            "end": "End Phase"}.get(phase, phase)


def _prompt_text(env, human_seat: int, kind: str, view: dict, msg) -> str:
    if kind == "idle":
        return f"Your move — {_phase_label(view['phase'])}"
    if kind == "battle":
        return "Battle Phase — choose your attack"
    if kind == "chain":
        top = view["chain"][-1]["card"] if view["chain"] else None
        nm = (top or {}).get("name")
        return f"Chain {nm}'s effect?" if nm else "Respond to the chain?"
    if kind == "effectyn":
        c = _ref_card(env, human_seat, msg.card)
        nm = (c or {}).get("name") or "this card"
        d = DB.desc_text(msg.description >> 20, msg.description & 0xFFFFF)
        return f"Use {nm}'s effect?" + (f' ("{d}")' if d else "")
    if kind == "yesno":
        d = DB.desc_text(msg.description >> 20, msg.description & 0xFFFFF)
        return d or "Answer yes or no"
    if kind == "option":
        return "Choose one of the effect's options"
    if kind == "card":
        hint = _select_hint(env, human_seat)
        base = hint or "Select the card(s)"
        if hasattr(msg, "min") and hasattr(msg, "max"):
            if msg.max > msg.min:
                base += f" (pick {msg.min}–{msg.max})"
            elif msg.min == 1 and msg.max == 1:
                base += " (pick 1)"
        return base
    if kind == "place":
        return "Choose a zone"
    if kind == "position":
        return "Choose a battle position"
    if kind == "announce":
        return "Declare a card" if isinstance(msg, M.AnnounceCard) else "Make your declaration"
    if kind == "sort":
        return "Order the cards"
    return "Choose"


def _prompt_card(env, human_seat: int, kind: str, view: dict, msg) -> dict | None:
    if kind == "effectyn":
        return _ref_card(env, human_seat, msg.card)
    if view["chain"]:
        return view["chain"][-1]["card"]
    return None


def _ref_card(env, human_seat: int, ref, board=None) -> dict | None:
    """An option's card as the human sees it (hidden field cards: code 0)."""
    if ref is None:
        return None
    board = board if board is not None else Board(env.duel, human_seat)
    r = board.row_of(ref)
    if r >= 0:
        row = board.rows[r]
        return _state_card(row.card, visible=row.visible)
    if ref.controller == human_seat or ref.location in (C.LOCATION_DECK, C.LOCATION_EXTRA):
        return _named_card(ref.code, _uid(ref.controller, ref.location,
                                          ref.sequence, ref.code))
    d = _blank_card(_uid(ref.controller, ref.location, ref.sequence, 0))
    d["pos"] = "set"
    return d


def _where(ref, human_seat: int) -> dict | None:
    if ref is None:
        return None
    if ref.location == C.LOCATION_MZONE:
        zone = "mzone" if ref.sequence < 5 else "emz"
    elif ref.location == C.LOCATION_SZONE:
        zone = "field" if ref.sequence == 5 else "szone"
    else:
        zone = _LOC_ZONES.get(ref.location, "deck")
    return {"side": "you" if ref.controller == human_seat else "bot",
            "zone": zone, "seq": ref.sequence}


def _verb(opt) -> str:
    if opt.kind == Kind.ATTACK:
        return "direct" if opt.arg else "attack"
    return _VERBS.get(opt.kind, opt.kind.name.lower())


def _effect(opt) -> str | None:
    if opt.kind in (Kind.ACTIVATE, Kind.OPTION, Kind.YES, Kind.NO) and opt.desc:
        return DB.desc_text(opt.desc >> 20, opt.desc & 0xFFFFF) or None
    return None


def _label(opt, card: dict | None, where: dict | None, effect: str | None) -> str:
    nm = (card or {}).get("name") if card else None
    if not nm:
        nm = "a face-down card" if card is not None else ""
    k = opt.kind
    if k == Kind.PASS:
        return "Pass"
    if k == Kind.YES:
        return f"Yes — use {nm}" if nm else "Yes"
    if k == Kind.NO:
        return f"No — decline {nm}" if nm else "No"
    if k == Kind.FINISH:
        return "Done"
    if k == Kind.CANCEL:
        return "Cancel"
    if k == Kind.TO_BATTLE:
        return "Go to Battle Phase"
    if k == Kind.TO_MAIN2:
        return "Go to Main Phase 2"
    if k == Kind.TO_END:
        return "End the turn"
    if k == Kind.ATTACK:
        return f"Direct attack with {nm}" if opt.arg else f"Attack with {nm}"
    if k == Kind.SUMMON:
        return f"Normal Summon {nm}"
    if k == Kind.SPSUMMON:
        return f"Special Summon {nm}"
    if k == Kind.MSET:
        return f"Set {nm} (monster)"
    if k == Kind.SSET:
        return f"Set {nm}"
    if k == Kind.ACTIVATE:
        s = f"Activate {nm}"
        return s + (f' ("{effect}")' if effect else "")
    if k == Kind.REPOS:
        return f"Change {nm}'s position"
    if k == Kind.POSITION:
        return {1: "Attack Position", 2: "Face-down Attack", 4: "Defense Position",
                8: "Face-down Defense"}.get(opt.arg, "Choose a position")
    if k == Kind.NUMBER:
        return f"Declare {opt.arg}"
    if k in (Kind.RACE, Kind.ATTRIBUTE):
        return f"Declare ({opt.arg:#x})"
    if k == Kind.DECLARE:
        return nm or "Declare a card"
    if k == Kind.SORT:
        return f"Put {nm} next"
    if k == Kind.OPTION:
        return effect or f"Option {opt.arg + 1}"
    if k == Kind.SELECT:
        loc = f" ({where['zone']})" if where and where["zone"] not in ("hand", "deck") else ""
        return f"{nm}{loc}" if nm else "Select a face-down card"
    if k == Kind.UNSELECT:
        return f"Unselect {nm}" if nm else "Unselect"
    if k == Kind.PLACE:
        return f"Place in zone {opt.arg}"
    if k == Kind.COUNTER:
        return f"Remove from {nm}" if nm else "Remove a counter"
    return f"{k.name} {nm}".strip()


def build_prompt(env, human_seat: int, prompt_id: int) -> dict | None:
    """The human's pending prompt (None when the human has nothing to answer)."""
    if env.done or env.player != human_seat:
        return None
    dec = env.pending_decision()
    if dec is None:
        return None
    opts = env.opts
    if len(opts) <= 1:
        return None
    view = build_view(env, human_seat)
    board = Board(env.duel, human_seat)
    kind = _prompt_kind(dec.msg)
    options = []
    for i, opt in enumerate(opts):
        card = _ref_card(env, human_seat, opt.card, board)
        where = _where(opt.card, human_seat)
        effect = _effect(opt)
        options.append({"index": i, "verb": _verb(opt),
                        "label": _label(opt, card, where, effect),
                        "card": card, "where": where, "effect": effect})
    min_n = int(getattr(dec.msg, "min", 0) or 0)
    max_n = int(getattr(dec.msg, "max", 0) or 0)
    return {"id": prompt_id, "kind": kind,
            "text": _prompt_text(env, human_seat, kind, view, dec.msg),
            "card": _prompt_card(env, human_seat, kind, view, dec.msg),
            "options": options, "min": min_n, "max": max_n}


# --- events ---


def build_events(env, human_seat: int, since: int = 0) -> list[dict]:
    """Plain-words public history for the human (the bot's private moves generic)."""
    names = DB.names()
    out = []
    for n, e in enumerate(env.history[since:], start=since + 1):
        try:
            kind = Kind(e.row[0])
        except ValueError:
            kind = None
        code = e.row[2]
        me = e.actor == human_seat
        who = "you" if me else "bot"
        say = _SAY.get(kind, "plays") if kind is not None else "plays"
        # the client shows the actor (you / bot) beside each event, so the text is the predicate only
        if not me and not e.public:
            text = _HIDDEN_SAY.get(kind, "makes a hidden move")
            card = None
        else:
            nm = names.get(code, f"#{code}") if code else ""
            text = f"{say} {nm}".strip()
            card = _named_card(code, f"e{n}") if code else None
        out.append({"n": n, "actor": who, "text": text[:1].upper() + text[1:], "card": card})
    return out
