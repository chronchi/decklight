"""Opponent deck primers from bot-vs-bot game records (web-ux-plan step 5, item 7).

A primer summarizes how one deck list plays, mined from stored ``games-*.jsonl``
records (see :mod:`ygo_zero.analysis.events` for the record format and message
layouts). Decks are keyed by a **content hash** (sha1 of the sorted main + extra
card codes; side deck ignored), so a pool deck with the same list as a gauntlet
deck shares one primer.

Per deck (from every game it played, either seat):
- ``games``, ``win_rate``, ``went_first_win_rate`` (bot vs bot, context only);
- ``archetype``: the largest setcode group of its main deck, named by the most
  common leading word of card names in that group (override map supported);
- ``key_cards``: top 8 cards by plays (activations + summons, once per game);
- ``openers``: the 3 most common first-3-play sequences on turn 1 going first;
- ``chokepoints``: the deck's cards opponents responded to (an opponent chain
  link right after the activation in the same chain, or the first opponent
  activation after a summon in the same turn) and how often negation hit;
- ``hand_traps_seen``: cards opponents activated from the hand during this
  deck's turn, and the deck card each answered.

:func:`primer_for` looks a primer up by deck list and adds ``your_answers``:
the human deck's cards that appear in any deck's ``hand_traps_seen`` or that
negated something anywhere, with the bot-deck cards they hit against this deck
(or the note ``"used against other decks"``).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from ygo_zero.analysis.events import deck_path, events
from ygo_zero.analysis.puzzles import PLAY_KINDS, replay_false
from ygo_zero.decks import load_ydk
from ygo_zero.engine import constants as C
from ygo_zero.engine.duel import DuelConfig, Engine

MIN_GAMES = 20  # a primer with fewer games is not served (primer_for -> None)
# A card counts as a negate for ``your_answers`` from this many credited negations (1-2 are attribution noise:
# Sky Striker Ace - Raye and S:P Little Knight had 1 each in the full build, Ash Blossom 2,837).
MIN_NEGATIONS = 3

OTHER_DECKS_NOTE = "used against other decks"


def deck_hash(main: Iterable[int], extra: Iterable[int] = ()) -> str:
    """sha1 over the sorted main codes + sorted extra codes (side ignored)."""
    m = sorted(int(c) for c in main)
    e = sorted(int(c) for c in extra)
    h = hashlib.sha1()
    h.update(",".join(map(str, m)).encode())
    h.update(b"|")
    h.update(",".join(map(str, e)).encode())
    return h.hexdigest()


def hash_of(deck) -> str:
    """Hash a Deck (main + extra), a ``{"main":..,"extra":..}`` mapping, or a flat code list."""
    if isinstance(deck, Mapping):
        main = deck.get("main", ())
        extra = deck.get("extra", ())
        return deck_hash(main, extra)
    if hasattr(deck, "main"):
        return deck_hash(deck.main, getattr(deck, "extra", ()))
    return deck_hash(deck, ())


_ALIAS: dict[int, int] | None = None


def canon(code: int) -> int:
    """The original card of an alternate artwork (its cdb ``alias``), else the code itself. Alt arts would otherwise
    split one card into two entries (Ash Blossom 14558127 / 14558128 in the first full build)."""
    global _ALIAS
    if _ALIAS is None:
        from ygo_zero.engine.resources import load_cards

        _ALIAS = {c: d.alias for c, d in load_cards().items() if d.alias}
    return _ALIAS.get(int(code), int(code))


def codes_of(deck) -> set[int]:
    """All card codes (alt arts folded, see :func:`canon`) of a Deck / mapping / flat list (for ``your_answers``)."""
    if isinstance(deck, Mapping):
        raw = (*deck.get("main", ()), *deck.get("extra", ()))
    elif hasattr(deck, "main"):
        raw = (*deck.main, *getattr(deck, "extra", ()))
    else:
        raw = deck
    return {canon(c) for c in raw}


def _canon_summary(s: dict) -> dict:
    """A per-game summary with every card code folded by :func:`canon` (in place; returns it)."""
    for l in s["links"]:
        if l.get("code"):
            l["code"] = canon(l["code"])
        sm = l.get("answers_summon")
        if sm and sm.get("code"):
            sm["code"] = canon(sm["code"])
    for sm in s["summons"]:
        if sm.get("code"):
            sm["code"] = canon(sm["code"])
        ans = sm.get("answered_by")
        if ans and ans.get("code"):
            ans["code"] = canon(ans["code"])
    for st in s["steps"]:
        if st.get("code"):
            st["code"] = canon(st["code"])
    return s


# --- per-game extraction ---


def _signed(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def _summarize_game(engine: Engine, rec: dict, cards: dict, names: dict[int, str]) -> dict:
    """Replay one record; per-seat plays, summons, chains and hand-trap answers.

    Raises on a truncated record or a replay whose winner differs (like
    :func:`events`), so bad records fail loud at build time, never silently.
    """
    ev = events(rec, engine)  # validates winner + LP accounting; chain links + turns
    if any(not isinstance(d, str) for d in rec["decks"]):
        raise ValueError(f"game {rec.get('game_idx')}: primer needs named decks, got {rec['decks']!r}")
    decks = [load_ydk(deck_path(d)) for d in rec["decks"]]
    cfg = DuelConfig(decks[0], decks[1], seed=rec["seed"])
    # Host-space replay (auto_single=False, singletons included): kinds/codes of
    # every play for openers, plus first_player. Record choices are
    # auto_single=True-space; replay_false consumes them at multi-option prompts.
    env, steps = replay_false(engine, cfg, [int(c) for c in rec["choices"]])
    if not env.done or env.winner != rec["winner"]:
        raise ValueError(f"game {rec.get('game_idx')}: host replay diverged from the record")
    first_player = int(env.duel.first_player)

    trace = env.duel.trace
    links, summons = _parse_trace(trace)  # engine-truth activations/summons with locations
    return {
        "game_idx": rec.get("game_idx"),
        "seed": rec["seed"],
        "decks": list(rec["decks"]),
        "winner": rec["winner"],
        "first_player": first_player,
        "turns": ev["turns"],
        "chain_links": ev["chain_links"],
        "links": links,
        "summons": summons,
        "steps": steps,
    }


def _parse_trace(trace: Sequence[bytes]) -> tuple[list[dict], list[dict]]:
    """Activations (with handler location) and summons (ok + negated) from a raw trace.

    Layouts are read off ``third_party/ygopro-core`` via
    :mod:`ygo_zero.analysis.events` (MSG_CHAINING = code + handler loc_info +
    triggering loc + desc + count; MSG_*SUMMONING = code + loc_info).
    """
    import struct

    links: list[dict] = []
    summons: list[dict] = []
    turn = 0
    group = 0  # chain group: bumped on every MSG_CHAIN_END / MSG_NEW_TURN
    pending: dict | None = None  # the last SUMMONING awaiting its SUMMONED
    pending_link: dict | None = None  # last SUMMONED awaiting an opponent answer

    def flush_pending(negated_by: dict | None) -> None:
        nonlocal pending
        if pending is None:
            return
        if negated_by is not None:
            summons.append({**pending, "ok": False, "answered_by": negated_by})
        pending = None

    for raw in trace:
        if not raw:
            continue
        t = raw[0]
        if t == C.MSG_NEW_TURN:
            turn += 1
            group += 1
            flush_pending(None)
            pending_link = None
        elif t == C.MSG_CHAIN_END:
            # An unmatched SUMMONING closed over by a chain that answered it was
            # negated on summon (e.g. Solemn Strike); else it was just an announcement.
            if pending is not None:
                answered = next(
                    (l for l in links if l["group"] == group and l["controller"] != pending["seat"]),
                    None,
                )
                flush_pending(answered)
            group += 1
        elif t in (C.MSG_SUMMONING, C.MSG_SPSUMMONING, C.MSG_FLIPSUMMONING):
            if len(raw) < 15:
                continue
            (code,) = struct.unpack_from("<I", raw, 1)
            con, loc = raw[5], raw[6]
            if con not in (0, 1):
                continue
            flush_pending(None)
            pending = {"turn": turn, "group": group, "seat": con, "code": int(code), "loc": int(loc)}
        elif t in (C.MSG_SUMMONED, C.MSG_SPSUMMONED):
            if pending is not None:
                pending_link = {**pending, "ok": True}
                summons.append(pending_link)
                pending = None
        elif t in (C.MSG_CHAIN_NEGATED, C.MSG_CHAIN_DISABLED):
            if len(raw) != 2:
                continue
            for l in links:
                if l["group"] == group and l["count"] == raw[1] and not l["negated"]:
                    l["negated"] = True
                    break
        elif t == C.MSG_CHAINING:
            if len(raw) != 1 + 4 + 10 + 1 + 1 + 4 + 8 + 4:
                continue
            (code,) = struct.unpack_from("<I", raw, 1)
            hcon, hloc = raw[5], raw[6]
            tcon = raw[1 + 4 + 10]
            (count,) = struct.unpack_from("<I", raw, 1 + 4 + 10 + 1 + 1 + 4 + 8)
            if tcon not in (0, 1):
                continue
            link = {
                "turn": turn,
                "group": group,
                "controller": int(tcon),
                "code": int(code),
                "handler_loc": int(hloc),
                "handler_con": int(hcon),
                "count": int(count),
                "negated": False,
            }
            # The card summoning answers: the first opponent activation after a
            # summon in the same turn (e.g. hand traps / Strike on summon).
            link["answers_summon"] = None
            if pending_link is not None and link["controller"] != pending_link["seat"]:
                if link["turn"] == pending_link["turn"]:
                    link["answers_summon"] = dict(pending_link)
                pending_link = None
            links.append(link)
    flush_pending(None)
    return links, summons


# --- aggregation ---


_SETNAMES: dict[int, str] | None = None


def _base_sets(code: int, cards: dict) -> set[int]:
    cd = cards.get(int(code))
    return {sc & 0x0FFF for sc in cd.setcodes if sc} if cd is not None and cd.setcodes else set()


def set_doc_freq(mains: Iterable[Sequence[int]], cards: dict) -> tuple[Counter, int]:
    """(base setcode -> number of decks using it, number of decks): the idf side of :func:`_archetype`."""
    df: Counter = Counter()
    n = 0
    for main in mains:
        n += 1
        df.update(set().union(*(_base_sets(c, cards) for c in main)) if main else set())
    return df, n


def _display_setname(setname: str, card_names: Sequence[str]) -> str:
    """Some ``!setname`` entries are name-search substrings, not names ("lswarm", "Qli", "True Draco|True King"):
    keep the first alternative, and when it is not a whole word of the group's card names, use the most common
    card-name word that contains it (Evilswarm, Qliphort)."""
    key = setname.split("|")[0].strip()
    words = [w.strip(",.\"'") for nm in card_names for w in nm.split()]
    if not key or key in words or " " in key:
        return key or setname
    hits = Counter(w for w in words if key.lower() in w.lower())
    return hits.most_common(1)[0][0] if hits else key


def _archetype(
    main: Sequence[int],
    cards: dict,
    names: dict[int, str],
    overrides: Mapping[str, str],
    h: str,
    df: tuple[Counter, int] | None = None,
) -> str:
    """The official archetype name (``!setname`` in edopro-strings.conf) of the main deck's best base-setcode group;
    sub-archetypes (0x1115 Sky Striker Ace) count toward their base (0x115 Sky Striker). Groups are scored by cards
    in the deck x log(decks / decks using the set) over ``df`` (:func:`set_doc_freq` of every deck in the store), so
    a splashed staple package (Mulcharmy, Dominus: 3-ofs in many decks) does not outscore the deck's own engine; raw
    counts named 15 of 220 decks "Mulcharmy". A winning setcode missing from the strings file (newer archetypes) is
    named by the most common first word of its card names (that rule alone named 4 decks "The")."""
    global _SETNAMES
    import math

    if h in overrides:
        return overrides[h]
    if _SETNAMES is None:
        from ygo_zero.cards.effect_strings import set_names

        try:
            _SETNAMES = set_names()
        except OSError:
            _SETNAMES = {}
    groups: dict[int, list[int]] = {}
    for c in main:
        for sc in _base_sets(c, cards):
            groups.setdefault(sc, []).append(int(c))
    if groups:
        freq, n = df if df is not None and df[1] else (Counter(), 1)

        def score(sc: int) -> tuple[float, int]:
            k = len(groups[sc])
            return (k * math.log((n + 1) / (freq.get(sc, 0) + 1)) if freq else k), -sc

        best = max(groups, key=score)
        # Label: the group's most common full setcode with a name (Spellbook 0x106e, Traptrix 0x108a), else the base
        # (whose names can be odd: 0x6e "Prophecy", 0x8a "Envy").
        full: Counter = Counter()
        for c in groups[best]:
            cd = cards.get(int(c))
            full.update(sc for sc in cd.setcodes if sc and sc & 0x0FFF == best and sc in _SETNAMES)
        label = max(full, key=lambda sc: (full[sc], sc == best, -sc)) if full else best
        if label in _SETNAMES:
            return _display_setname(_SETNAMES[label], [names.get(c, "") for c in groups[best]])
        pool = groups[
            best
        ]  # a setcode newer than edopro-strings.conf (Elfnote, Angelechy): name it from its cards
    else:
        pool = [int(c) for c in main]
    if not pool:
        return "unknown"
    words = Counter()
    for c in pool:
        nm = names.get(int(c), "")
        if nm:
            words[nm.split()[0]] += 1
    if not words:
        return "unknown"
    return words.most_common(1)[0][0]


def aggregate(
    summaries: list[dict], cards: dict, names: dict[int, str], overrides: Mapping[str, str] | None = None
) -> dict:
    """Fold per-game summaries into the primers store (``decks`` + ``negated_by``)."""
    overrides = dict(overrides or {})
    decks: dict[str, dict] = {}
    negated_by: dict[str, dict[str, int]] = {}  # negator code -> {victim code -> times}

    def entry(deck) -> dict:
        h = hash_of(deck)
        e = decks.get(h)
        if e is None:
            e = decks[h] = {
                "hash": h,
                "files": [],
                "main": sorted(int(c) for c in deck.main),
                "extra": sorted(int(c) for c in deck.extra),
                "games": 0,
                "wins": 0,
                "went_first_games": 0,
                "went_first_wins": 0,
                "plays": Counter(),  # code -> games with a play
                "openers": Counter(),  # first-3-plays tuple -> games
                "openers_n": 0,
                "responses": Counter(),  # card code -> opponent answers
                "resp_neg": Counter(),  # card code -> answers that negated it
                "hand_traps": Counter(),  # opp hand card -> times vs this deck
                "ht_answers": Counter(),  # (ht code, deck card) -> times
            }
        return e

    for s in summaries:
        _canon_summary(s)
        deck_names = s["decks"]
        loaded = [load_ydk(deck_path(d)) for d in deck_names]
        hashes = [hash_of(d) for d in loaded]
        for seat in (0, 1):
            e = entry(loaded[seat])
            if deck_names[seat] not in e["files"]:
                e["files"].append(deck_names[seat])
            e["games"] += 1
            e["wins"] += s["winner"] == seat
            if s["first_player"] == seat:
                e["went_first_games"] += 1
                e["went_first_wins"] += s["winner"] == seat
        # key cards: activations + summons, once per game per seat
        played: dict[int, set[int]] = {0: set(), 1: set()}
        for l in s["links"]:
            if l["code"]:
                played[l["controller"]].add(l["code"])
        for sm in s["summons"]:
            if sm.get("ok") and sm["code"]:
                played[sm["seat"]].add(sm["code"])
        for seat in (0, 1):
            e = decks[hashes[seat]]
            for c in played[seat]:
                e["plays"][c] += 1
        # openers: first 3 plays of turn 1 when going first
        for seat in (0, 1):
            if s["first_player"] != seat:
                continue
            seq = [
                st["code"]
                for st in s["steps"]
                if st["player"] == seat and st["turn"] == 1 and st["kind"] in PLAY_KINDS and st["code"]
            ][:3]
            if seq:
                e = decks[hashes[seat]]
                e["openers"][tuple(seq)] += 1
                e["openers_n"] += 1
        # chokepoints: opponent answers to activations (next link, same chain)
        by_group: dict[int, list[dict]] = {}
        for l in s["links"]:
            by_group.setdefault(l["group"], []).append(l)
        for ls in by_group.values():
            ls.sort(key=lambda l: l["count"])
            for i, l in enumerate(ls):
                nxt = [m for m in ls[i + 1 :] if m["controller"] != l["controller"]]
                if not nxt or not l["code"]:
                    continue
                rsp = nxt[0]
                e = decks[hashes[l["controller"]]]
                e["responses"][l["code"]] += 1
                # Negated = the activation's own link was negated (by the answer or later).
                if l["negated"]:
                    e["resp_neg"][l["code"]] += 1
                _ = rsp
                # Global negator index: credit the answering link only when it is the chain's last link and
                # directly follows the negated one, so nothing later could have negated it. (Crediting any next
                # opponent link listed Sky Striker Ace - Raye as a negate in the first full build.)
                if l["negated"] and rsp["code"] and rsp is ls[-1] and ls[i + 1] is rsp:
                    negated_by.setdefault(str(rsp["code"]), {}).setdefault(str(l["code"]), 0)
                    negated_by[str(rsp["code"])][str(l["code"])] += 1
        # chokepoints, summon half: first opponent activation after a summon
        for sm in s["summons"]:
            if not sm["code"]:
                continue
            ans = sm.get("answered_by")
            if ans is None:
                continue
            e = decks[hashes[sm["seat"]]]
            e["responses"][sm["code"]] += 1
            if not sm.get("ok"):
                e["resp_neg"][sm["code"]] += 1
            if ans["code"]:
                neg_key = str(ans["code"])
                if not sm.get("ok"):
                    negated_by.setdefault(neg_key, {}).setdefault(str(sm["code"]), 0)
                    negated_by[neg_key][str(sm["code"])] += 1
        # hand traps seen: opponent activations from the hand on this deck's turn
        turn_player = {t["turn"]: t["turn_player"] for t in s["turns"]}
        for l in s["links"]:
            if l["handler_loc"] != C.LOCATION_HAND or not l["code"]:
                continue
            tp = turn_player.get(l["turn"])
            if tp is None or l["controller"] == tp:
                continue
            e = decks[hashes[tp]]
            e["hand_traps"][l["code"]] += 1
            answered = _answered_card(s, l, tp)
            if answered:
                e["ht_answers"][(l["code"], answered)] += 1

    return {"decks": _finalize(decks, cards, names, overrides), "negated_by": negated_by}


def _answered_card(s: dict, link: dict, seat: int) -> int | None:
    """The deck card a hand trap answered: the link it chained to (same chain,
    controlled by the deck), else the deck's most recent summon code, else None."""
    same = [l for l in s["links"] if l["group"] == link["group"] and l["count"] == link["count"] - 1]
    if same and same[0]["controller"] == seat and same[0]["code"]:
        return same[0]["code"]
    if link.get("answers_summon"):
        sm = link["answers_summon"]
        if sm["seat"] == seat and sm["code"]:
            return sm["code"]
    return None


def _finalize(
    decks: dict[str, dict], cards: dict, names: dict[int, str], overrides: Mapping[str, str]
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    df = set_doc_freq([e["main"] for e in decks.values()], cards)
    for h, e in decks.items():
        games = e["games"]
        key_cards = [
            {
                "code": c,
                "name": names.get(c, f"#{c}"),
                "games": n,
                "per_game": round(n / games, 3),
            }
            for c, n in e["plays"].most_common(8)
        ]
        openers = [
            {
                "plays": list(seq),
                "names": [names.get(c, f"#{c}") for c in seq],
                "games": n,
                "share": round(n / e["openers_n"], 3) if e["openers_n"] else 0.0,
            }
            for seq, n in e["openers"].most_common(3)
        ]
        chokes = sorted(e["responses"], key=lambda c: (-e["responses"][c], c))
        chokepoints = [
            {
                "code": c,
                "name": names.get(c, f"#{c}"),
                "responses": e["responses"][c],
                "negated": e["resp_neg"].get(c, 0),
                "response_rate": round(e["responses"][c] / games, 3),
            }
            for c in chokes[:8]
        ]
        hts = sorted(e["hand_traps"], key=lambda c: (-e["hand_traps"][c], c))
        hand_traps = []
        for c in hts[:12]:
            answered = [
                {
                    "code": dc,
                    "name": names.get(dc, f"#{dc}"),
                    "times": e["ht_answers"][(c, dc)],
                }
                for dc in sorted({dc for (hc, dc) in e["ht_answers"] if hc == c})
            ]
            answered.sort(key=lambda d: (-d["times"], d["code"]))
            hand_traps.append(
                {
                    "code": c,
                    "name": names.get(c, f"#{c}"),
                    "times": e["hand_traps"][c],
                    "answered": answered[:6],
                }
            )
        out[h] = {
            "hash": h,
            "files": sorted(e["files"]),
            "main_size": len(e["main"]),
            "extra_size": len(e["extra"]),
            "games": games,
            "win_rate": round(e["wins"] / games, 3) if games else 0.0,
            "went_first_games": e["went_first_games"],
            "went_first_win_rate": (
                round(e["went_first_wins"] / e["went_first_games"], 3) if e["went_first_games"] else 0.0
            ),
            "archetype": _archetype(e["main"], cards, names, overrides, h, df),
            "key_cards": key_cards,
            "openers": openers,
            "chokepoints": chokepoints,
            "hand_traps_seen": hand_traps,
        }
    return out


# --- runtime lookup ---


def primer_for(store: Mapping, bot_deck, my_deck) -> dict | None:
    """The stored primer for the bot's deck plus ``your_answers`` from the human deck.

    ``bot_deck`` / ``my_deck`` are Deck objects, ``{"main","extra"}`` mappings
    or flat code lists. Returns None when the deck is unknown or has fewer than
    :data:`MIN_GAMES` games.
    """
    decks = store.get("decks", {})
    h = hash_of(bot_deck)
    p = decks.get(h)
    if p is None or p.get("games", 0) < MIN_GAMES:
        return None
    mine = codes_of(my_deck)
    ht_here = {h["code"]: h for h in p.get("hand_traps_seen", [])}
    ht_global: set[int] = set()
    for d in decks.values():
        for h2 in d.get("hand_traps_seen", []):
            ht_global.add(h2["code"])
    negated_by = store.get("negated_by", {})
    answers = []
    for c in sorted(mine):
        here = ht_here.get(c)
        hits = here["answered"] if here else []
        neg_victims = negated_by.get(str(c), {})
        if sum(neg_victims.values()) < MIN_NEGATIONS:
            neg_victims = {}
        if c not in ht_global and not neg_victims:
            continue
        if hits:
            answers.append({"code": c, "name": names_of(store, c), "hits": hits})
        elif neg_victims:
            victims = [
                {"code": int(v), "name": names_of(store, int(v)), "times": n}
                for v, n in sorted(neg_victims.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
            answers.append(
                {
                    "code": c,
                    "name": names_of(store, c),
                    "note": OTHER_DECKS_NOTE,
                    "hits_elsewhere": victims[:6],
                }
            )
        else:
            answers.append({"code": c, "name": names_of(store, c), "note": OTHER_DECKS_NOTE, "hits": []})
    answers.sort(key=lambda a: (not a.get("hits"), a["code"]))
    return {**p, "your_answers": answers}


def names_of(store: Mapping, code: int) -> str:
    """A card name from any primer entry that mentions the code, else ``#<code>``."""
    for d in store.get("decks", {}).values():
        for section in ("key_cards", "chokepoints", "hand_traps_seen"):
            for item in d.get(section, []):
                if item.get("code") == code:
                    return item.get("name", f"#{code}")
                for sub in item.get("answered", []):
                    if sub.get("code") == code:
                        return sub.get("name", f"#{code}")
    try:  # cards only seen as a key of negated_by: take the name from the card database
        from ygo_zero.web.view import DB

        nm = DB.names().get(int(code))
    except Exception:  # noqa: BLE001 -- no card db: keep the code
        nm = None
    return nm or f"#{code}"


def load_store(path) -> dict:
    """Load a primers JSON file (``{"decks", "negated_by"}``); ``.gz`` files are gzip (the committed web copy)."""
    import gzip
    import json
    from pathlib import Path

    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw)
