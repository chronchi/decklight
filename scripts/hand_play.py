"""Hand-play a full duel (both seats) through the engine and save it as a "gold" game record.

A person (or Claude) drives every decision by kind + card name, with a short reason; zone (PLACE) prompts are
answered automatically; positions are chosen with POS:attack / POS:defense. State lives in data/demos/gold_work/<game>/state.json,
so each command is a fresh process that replays the duel.

    uv run python scripts/hand_play.py new  g1 snake-eye tenpai-dragon 7
    uv run python scripts/hand_play.py show g1 [--texts]
    uv run python scripts/hand_play.py do   g1 "ACT:Pre-Preparation SEL:Saji P END" --why "search the starter"
    uv run python scripts/hand_play.py undo g1 3
    uv run python scripts/hand_play.py save g1      # -> data/demos/gold/<d0>-vs-<d1>.jsonl
    uv run python scripts/hand_play.py log  g1 [--turns N]  # compact public log of the duel so far

Name matching ignores punctuation and case (apostrophes, hyphens, quotes, stars, slashes).

Tokens: an option number; P (pass/cancel at a prompt — never answers YES/NO prompts, use YES or NO
there); KIND:name-substring where KIND is a prefix of the
option kind (ACT, SUM, SPS, REPOS, MSET, SSET, SEL, UNS (unselect), YES, NO, OPT (effect option),
ATK (attack with), FIN (finish), CAN (cancel), POS:attack / POS:defense (positions), NUM:value
(announce a number, e.g. NUM:5), DEC:name (declare a card), SORT:name (order a card first),
RACE:/ATTR: (announce race/attribute by name fragment), COUNT:/RPS:/PLA: (rare: counters,
rock-paper-scissors, places — numbers usually suffice)) or one of END (TO_END), BP (TO_BATTLE),
M2 (TO_MAIN2). `do` stops at the first token that matches no option and prints the state.
Commands are case-insensitive (`CHECK`, `Show`). `SEL:face-down` (also `opponents_face-down`,
`set`, `facedown`) picks the unique face-down option, else lists every matching option numbered with
controller + zone + face-up name or "face-down" for an option-number pick. One token picks exactly one
option: `SEL:own:<name|face-down>` / `SEL:opp:<name|face-down>` restrict to that controller (from the
acting seat's view); `SEL:<M/S><zone>` / `SEL:own:M3` / `SEL:opp:S2` pick by board zone (0-indexed, the
numbering `show` prints as `zone N`; `FZ` is the field zone); identical copies (same card, controller,
location kind) stay interchangeable and pick the first. A still-ambiguous token stops with the numbered
list. `ATK:<card>` at a card-selection prompt (no ATTACK option) applies as `SEL:<card>` when it selects
exactly one option, else stops saying the prompt is a selection.
`CAN`, `CANCEL` and `NO` answer the passive option at any prompt that offers one (YES/NO prompts
still need explicit YES/NO). `BP` works only where a Battle Phase option is on offer (the idle
prompt). Several `ATK:` tokens may share one `do` (one attack each; empty chain windows between
attacks are passed like --auto-pass).
A numeric token is only valid while the current prompt's option list still equals the option list of the
prompt the command started on (numbers re-index after every pick, so a stale number would hit the wrong
card); otherwise `do` stops with a prompt-changed message. Named tokens (SEL:name, P, ...) keep working
across prompts as today. PASS* passes through chain windows that offer nothing new (per seat: only
activations already on offer at that seat's first window of the run), for either seat, stopping at a
non-chain message, chain length change, turn or phase change, a new activation for either seat
(50 steps max); those answers are labeled `pass`. PASS* (and --auto-pass) continues across seat changes
while the next seat's window offers no live option besides those it already declined this turn.
A `--hold "Card A,Card B"` list (per seat, expires
on turn change) additionally lets PASS* pass an EMPTY-chain window whose every live option is a held
card; any window with chain length > 0 still stops. `do --auto-pass` appends PASS* after the last
token. `do` prints a compact view (Extra Deck as a count; GY/banished/deck-count lines only when
changed; `show` and each seat's first view of a turn stay full; `do --full` forces the full view).
`show`/`do --json` print one machine-readable object instead of the text view.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk
from ygo_zero.demos.game import (
    _effect_text,
    _opts,
    chain_names,
    desc_text,
    describe_state,
    hidden_card,
    option_desc,
)
from ygo_zero.demos.names import CardNames
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.engine.resources import load_cards

WORK = ROOT / "data" / "demos" / "gold_work"
OUT = ROOT / "data" / "demos" / "gold"
_ALIASES = {"END": Kind.TO_END, "BP": Kind.TO_BATTLE, "M2": Kind.TO_MAIN2, "ATK": Kind.ATTACK}
_PASSIVE = (Kind.PASS, Kind.NO, Kind.CANCEL, Kind.FINISH)
_MSGS = {v: k for k, v in vars(C).items() if k.startswith("MSG_")}
_PASS_STEPS = 50
_MULTI_MSGS = (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD, C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_SUM)
# Option-row location labels (obs action col 7; 0 = no card). Matches game._LOCS wording.
_LOC_NAMES = {1: "Deck", 2: "hand", 3: "Monster Zone", 4: "S/T Zone", 5: "GY",
              6: "banished", 7: "Extra Deck", 8: "Xyz"}  # fmt: skip
# Full zone names for the duplicate disambiguation suffix (e.g. "Monster Zone 2").
_ZONE_NAMES = {1: "Deck", 2: "hand", 3: "Monster Zone", 4: "Spell/Trap Zone", 5: "GY",
               6: "banished", 7: "Extra Deck", 8: "Xyz"}  # fmt: skip


def _load(game: str) -> dict:
    return json.loads((WORK / game / "state.json").read_text())


def _store(game: str, st: dict) -> None:
    (WORK / game).mkdir(parents=True, exist_ok=True)
    (WORK / game / "state.json").write_text(json.dumps(st))


def _config(st: dict) -> DuelConfig:
    d0 = load_ydk(ROOT / f"decks/pool/{st['decks'][0]}.ydk")
    d1 = load_ydk(ROOT / f"decks/pool/{st['decks'][1]}.ydk")
    return DuelConfig(d0, d1, seed=st["seed"])


# seat -> live options it passed on at its latest chain window, valid until the next real action or turn
_DECLINED: dict[int, set] = {}


def _note(env, obs, i: int) -> None:
    """Track declined chain windows for PASS*: a pass at a chain window records that seat's live options; any
    other choice (an action) or a new turn forgets all of them."""
    k = _opts(obs)[i][0]
    if int(obs["global"][21]) == C.MSG_SELECT_CHAIN and k in _PASSIVE:
        _DECLINED[env.player] = _live(obs) | {("turn", int(obs["global"][2]))}
    elif int(obs["global"][21]) != C.MSG_SELECT_PLACE:
        _DECLINED.clear()


def _replay(engine: Engine, st: dict):
    env = CppEnv(engine, _config(st))
    obs = env.reset()
    _DECLINED.clear()
    for i in st["choices"]:
        _note(env, obs, i)
        obs = env.step(i)
        if obs is None:
            break
    return env, obs


def _kind(k: int) -> str:
    try:
        return Kind(k).name
    except ValueError:
        return str(k)


def _loc_of(obs, i: int) -> str:
    """Location suffix of option `i` ("" when the option names no card), e.g. "GY" or "opponent's GY".
    From the option row's loc_id column (col 7) and relative controller (col 8: 1 = opponent)."""
    try:
        row = list(obs["actions"])[i]
        loc, ctrl = int(row[7]), int(row[8])
    except (IndexError, TypeError, KeyError):
        return ""
    name = _LOC_NAMES.get(loc, "")
    if not name:
        return ""
    return f"opponent's {name}" if ctrl == 1 else name


def _arg_of(obs, i: int) -> int:
    """Option row's arg column (col 5): the announced value for NUMBER, the direct-attack flag for
    ATTACK, the position bits for POSITION."""
    try:
        return int(list(obs["actions"])[i][5])
    except (IndexError, TypeError, KeyError):
        return 0


def _opt_row(obs, i: int) -> int:
    """Board row index behind option `i` (-1 when the option names no card)."""
    try:
        return int(list(obs["actions"])[i][1])
    except (IndexError, TypeError, KeyError):
        return -1


def _zone_of(obs, i: int) -> str:
    """Zone suffix of option `i`, e.g. "Monster Zone 2" ("" when the option names no board card)."""
    r = _opt_row(obs, i)
    if r < 0:
        return ""
    try:
        cards = list(obs["cards"][obs["card_mask"]])
        if not 0 <= r < len(cards):
            return ""
        loc, seq = int(cards[r][1]), int(cards[r][2])
    except (IndexError, TypeError, KeyError):
        return ""
    name = _ZONE_NAMES.get(loc, "")
    if not name:
        return ""
    zone = seq - 1 if seq > 0 else i  # deck rows carry no sequence: fall back to the option index
    return f"{name} {zone}"


_FACE_DOWN_POS = (2, 4, 6)  # obs position col: face-down ATK / face-down DEF / set
_CTRL_WORDS = {"own": 0, "opp": 1, "opponent": 1, "opponents": 1}  # SEL: controller qualifiers
_ATK_SEL_MSGS = (C.MSG_SELECT_CARD, C.MSG_SELECT_UNSELECT_CARD)  # ATK:<card> applies as SEL: here


def _is_face_down(obs, i: int) -> bool:
    """Whether option `i` names a face-down card in the acting seat's view (own set cards and the
    opponent's face-down cards, whose code is 0 and whose label says "face-down")."""
    try:
        row = list(obs["actions"])[i]
    except (IndexError, TypeError, KeyError):
        return False
    try:
        lbl = hidden_card(obs, i)
    except (IndexError, TypeError, KeyError):
        lbl = ""
    if lbl and "face-down" in lbl:
        return True
    try:
        r = int(row[1])
        cards = list(obs["cards"][obs["card_mask"]])
        pos = int(cards[r][5]) if 0 <= r < len(cards) else 0
    except (IndexError, TypeError, KeyError):
        return False
    return pos in _FACE_DOWN_POS


def _facedown_cands(obs, head: str) -> list[int]:
    """Option indices matching kind `head` whose card is face-down (for SEL:face-down)."""
    out = []
    for i, (k, _) in enumerate(_opts(obs)):
        kname = _kind(k)
        ok_kind = (k == _ALIASES[head]) if head in _ALIASES else kname.startswith(head)
        if ok_kind and _is_face_down(obs, i):
            out.append(i)
    return out


def _ctrl_of(obs, i: int) -> int:
    """Relative controller behind option `i` (0 = own, 1 = opponent, -1 = no card)."""
    try:
        return int(list(obs["actions"])[i][8])
    except (IndexError, TypeError, KeyError):
        return -1


def _zone_key(obs, i: int) -> tuple[str, int] | None:
    """Board zone behind option `i` as ("M"/"S", 0-indexed zone), e.g. ("S", 2); None when the
    option names no Monster/Spell-Trap Zone card. The numbering matches `show`'s "zone N"
    (and _zone_of's "Monster Zone N"); the field zone is ("S", 5)."""
    r = _opt_row(obs, i)
    if r < 0:
        return None
    try:
        cards = list(obs["cards"][obs["card_mask"]])
        if not 0 <= r < len(cards):
            return None
        loc, seq = int(cards[r][1]), int(cards[r][2])
    except (IndexError, TypeError, KeyError):
        return None
    if loc == 3:
        return ("M", seq - 1)
    if loc == 4:
        return ("S", seq - 1)
    return None


def _split_ctrl(raw_sub: str) -> tuple[int | None, str]:
    """`own:`/`opp:` controller qualifier of a SEL token (from the acting seat's view)."""
    parts = raw_sub.split(":")
    if len(parts) >= 2 and parts[0].strip().lower() in _CTRL_WORDS:
        return _CTRL_WORDS[parts[0].strip().lower()], ":".join(parts[1:])
    return None, raw_sub


def _parse_zone(rest: str) -> tuple[str, int] | None:
    """`M<n>` / `S<n>` zone token (0-indexed, the numbering `show` prints) or `FZ` (field zone)."""
    s = _norm(rest)
    if s == "fz":
        return ("S", 5)
    m = re.fullmatch(r"([ms])(\d+)", s)
    return (m.group(1).upper(), int(m.group(2))) if m else None


def _identical(obs, idxs: list[int]) -> bool:
    """Whether options are interchangeable copies: same card, same controller, same location kind."""
    keys = set()
    for i in idxs:
        try:
            row = list(obs["actions"])[i]
            keys.add((int(row[2]), int(row[8]), int(row[7])))
        except (IndexError, TypeError, KeyError):
            return False
    return len(keys) == 1


def _sel_cands(obs, head: str, raw_sub: str, names: dict[int, str]) -> tuple[str, list[tuple[int, int]]]:
    """SELECT-head token search: (mode, [(rank, index)]). Mode is "face" for face-down tokens,
    "zone" for a bare zone token, else "name". Controller (`own:`/`opp:`) and zone
    (`M<n>`/`S<n>`/`FZ`) qualifiers filter the pool before name ranking."""
    ctrl, rest = _split_ctrl(raw_sub)
    rn = _norm(rest)
    zone = _parse_zone(rest)
    face = rn == "set" or "facedown" in rn
    mode = "face" if face else "zone" if zone is not None else "name"
    out = []
    for i, (k, code) in enumerate(_opts(obs)):
        kname = _kind(k)
        ok_kind = (k == _ALIASES[head]) if head in _ALIASES else kname.startswith(head)
        if not ok_kind:
            continue
        if ctrl is not None and _ctrl_of(obs, i) != ctrl:
            continue
        if zone is not None and _zone_key(obs, i) != zone:
            continue
        if mode == "face":
            if _is_face_down(obs, i):
                out.append((0, i))
            continue
        if mode == "zone":
            out.append((0, i))
            continue
        if k == Kind.NUMBER:
            cands = [str(_arg_of(obs, i))]
        else:
            nm = _norm(names.get(code, "") if code else "")
            cands = [nm]
            loc = _loc_of(obs, i).lower()
            if nm and loc:
                cands.append(_norm(f"{names.get(code, '')} ({loc})"))
                short = loc.split()[-1]  # "opponent's GY" also matches " (GY)"
                if short != loc:
                    cands.append(_norm(f"{names.get(code, '')} ({short})"))
            zone_sfx = _zone_of(obs, i).lower()
            if nm and zone_sfx:
                cands.append(_norm(f"{names.get(code, '')} ({zone_sfx})"))
        desc = _norm(option_desc(obs, i))
        rank = None
        for cand in cands:
            r = (
                0
                if not rn or cand == rn
                else 1
                if cand.startswith(rn)
                else 2
                if rn in f"{cand} {desc}"
                else None
            )
            if r is not None and (rank is None or r < rank):
                rank = r
        if rank is not None:
            out.append((rank, i))
    return mode, out


def _opt_label(obs, i: int, names: dict[int, str]) -> str:
    """`i: controller name (zone)` label for the ambiguous-selection listing, e.g.
    `1: opponent's Mirror Force (Spell/Trap Zone 0)` or `0: your face-down card (S/T Zone 2)`.
    A known card that is face-down on the field keeps its name with a face-down marker (the
    acting seat knows its own set cards); an unknown card shows as "face-down card"."""
    try:
        row = list(obs["actions"])[i]
        code, ctrl = int(row[2]), int(row[8])
    except (IndexError, TypeError, KeyError):
        return f"{i}: option {i}"
    whose = "your" if ctrl == 0 else "opponent's" if ctrl == 1 else "a"
    known = names.get(code, f"#{code}") if code else ""
    if known:
        nm = known + (" (face-down)" if _is_face_down(obs, i) else "")
    else:
        nm = "face-down card"
    zone = _zone_of(obs, i) or _loc_of(obs, i) or "field"
    return f"{i}: {whose} {nm} ({zone})"


def _link_rating(code: int) -> int | None:
    """The Link rating of card `code` (None when not a Link monster or unknown)."""
    if not code:
        return None
    d = _card_db().get(code)
    if d is None or not (d.type & C.TYPE_LINK):
        return None
    return int(d.level)


def _display_names(obs, names: dict[int, str]) -> list[str]:
    """Base card name of each option (NUMBER options show their announced value). Duplicates get a
    " (loc)" suffix; duplicates that share the location also get the zone index, e.g. two Attack
    Position "Snake-Eye Ash" in the Monster Zone, or two face-down cards in one Spell/Trap Zone."""
    opts = _opts(obs)
    base = []
    for i, (k, code) in enumerate(opts):
        if k == Kind.NUMBER:
            base.append(str(_arg_of(obs, i)))
        elif code:
            base.append(names.get(code, f"#{code}"))
        else:
            base.append(hidden_card(obs, i) or "(no card)")
    counts: dict[str, int] = {}
    for nm in base:
        counts[nm.lower()] = counts.get(nm.lower(), 0) + 1
    locs = [_loc_of(obs, i) for i in range(len(opts))]
    sameloc: dict[tuple[str, str], int] = {}
    for nm, loc in zip(base, locs):
        key = (nm.lower(), loc)
        sameloc[key] = sameloc.get(key, 0) + 1
    out = []
    for i, nm in enumerate(base):
        if counts[nm.lower()] < 2:
            out.append(nm)
        elif locs[i] and sameloc[(nm.lower(), locs[i])] >= 2:
            zone = _zone_of(obs, i)
            out.append(f"{nm} ({zone})" if zone else f"{nm} ({locs[i]})")
        elif locs[i]:
            out.append(f"{nm} ({locs[i]})")
        else:
            out.append(f"{nm} #{i}")
    # last resort: options with no loc/zone info still sharing a name get the option index
    seen: dict[str, int] = {}
    for i, nm in enumerate(out):
        key = nm.lower()
        if key in seen and out[seen[key]].lower() == key:
            out[i] = f"{nm} #{i}"
            if "#" not in out[seen[key]]:
                out[seen[key]] = f"{out[seen[key]]} #{seen[key]}"
        else:
            seen.setdefault(key, i)
    return out


def _selected_names(obs, names: dict[int, str]) -> list[str]:
    """Names of already-picked cards in a multi-select prompt.

    MSG_SELECT_UNSELECT_CARD keeps the picked set in the message itself (its UNSELECT options) and the
    engine never flags those board rows selected (obs card col 20 stays 0), so the picked names come from
    the UNSELECT options there; other multi-select prompts use the selected flag (col 20 == 1)."""
    opts = _opts(obs)
    if int(obs["global"][21]) == C.MSG_SELECT_UNSELECT_CARD:
        out = []
        for i, (k, code) in enumerate(opts):
            if k != Kind.UNSELECT:
                continue
            out.append(names.get(code, f"#{code}") if code else (hidden_card(obs, i) or "(hidden)"))
        return out
    out = []
    for r in obs["cards"][obs["card_mask"]]:
        if int(r[20]) != 1:
            continue
        code = int(r[0])
        out.append(names.get(code, f"#{code}") if code else "(hidden)")
    return out


def _prompt_key(obs) -> tuple:
    """The option list of the current prompt as (kind, code, desc) keys (for the numeric-token guard)."""
    return tuple((k, code, option_desc(obs, i)) for i, (k, code) in enumerate(_opts(obs)))


_P_CHOICE = (Kind.PASS, Kind.CANCEL, Kind.FINISH)  # what the `P` token may answer (never YES/NO)
_YESNO_MSGS = (C.MSG_SELECT_YESNO, C.MSG_SELECT_EFFECTYN)

# Name matching ignores punctuation and case (both sides reduce to alphanumerics): apostrophes,
# hyphens, quotes, stars and slashes never break a match ("Fiendsmiths_Tract" finds "Fiendsmith's
# Tract", "Tenyi_Spirit_Adhara" finds "Tenyi Spirit - Adhara", "D_D" finds "D/D").
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _NON_ALNUM.sub("", s.lower())


def _match(obs, tok: str, names: dict[int, str]) -> int | None:
    opts = _opts(obs)
    if tok.isdigit():
        i = int(tok)
        return i if i < len(opts) else None
    if tok.upper() == "P":
        return next((i for i, (k, _) in enumerate(opts) if k in _P_CHOICE), None)
    head, _, raw_sub = tok.partition(":")
    head, sub = head.upper(), _norm(raw_sub)
    if head in ("CAN", "CANCEL", "NO"):
        # bare cancel words (and CAN:/CANCEL:/NO: forms) mean the passive option at any prompt that
        # offers one — except YES/NO prompts, where NO is a real decision and P still refuses.
        msg = int(obs["global"][21]) if "global" in obs else -1
        if msg not in _YESNO_MSGS:
            j = next((i for i, (k, _) in enumerate(opts) if k in _P_CHOICE), None)
            if j is not None:
                return j
        if head in ("CAN", "CANCEL"):
            return None
    if head in ("SEL", "SELECT"):
        # Selection tokens with controller (`own:`/`opp:`) and zone (`M<n>`/`S<n>`/`FZ`)
        # qualifiers: one token picks exactly one option. Several non-interchangeable matches
        # stop (the `!!` lists them); identical copies pick the first.
        mode, found = _sel_cands(obs, head, raw_sub, names)
        if not found:
            return None
        if mode == "face" and len(found) > 1:
            return None
        best_rank = min(r for r, _ in found)
        best = [i for r, i in found if r == best_rank]
        if len(best) == 1 or _identical(obs, best):
            return best[0]
        return None
    if sub == "set" or "facedown" in sub:
        # SEL:face-down (also opponents_face-down / opponent's face-down card / set / facedown):
        # the unique face-down option of that kind; several face-downs need the option number.
        cands = _facedown_cands(obs, head)
        if len(cands) == 1:
            return cands[0]
        return None
    best = None  # (rank, index): exact name < name prefix < substring of name/description
    for i, (k, code) in enumerate(opts):
        kname = _kind(k)
        ok_kind = (k == _ALIASES[head]) if head in _ALIASES else kname.startswith(head)
        if not ok_kind:
            continue
        if k == Kind.NUMBER:
            cands = [str(_arg_of(obs, i))]
        else:
            nm = _norm(names.get(code, "") if code else "")
            cands = [nm]
            loc = _loc_of(obs, i).lower()
            if nm and loc:
                cands.append(_norm(f"{names.get(code, '')} ({loc})"))
                short = loc.split()[-1]  # "opponent's GY" also matches " (GY)"
                if short != loc:
                    cands.append(_norm(f"{names.get(code, '')} ({short})"))
            zone = _zone_of(obs, i).lower()
            if nm and zone:
                cands.append(_norm(f"{names.get(code, '')} ({zone})"))
        desc = _norm(option_desc(obs, i))
        rank = None
        for cand in cands:
            r = (
                0
                if not sub or cand == sub
                else 1
                if cand.startswith(sub)
                else 2
                if sub in f"{cand} {desc}"
                else None
            )
            if r is not None and (rank is None or r < rank):
                rank = r
        if rank is not None and (best is None or (rank, i) < (best[0], best[1])):
            best = (rank, i)
    return best[1] if best else None


_SHORT_KIND = {
    Kind.SUMMON: "SUM", Kind.SPSUMMON: "SPS", Kind.REPOS: "REPOS", Kind.MSET: "MSET",
    Kind.SSET: "SSET", Kind.ACTIVATE: "ACT", Kind.ATTACK: "ATK", Kind.TO_BATTLE: "BP",
    Kind.TO_MAIN2: "M2", Kind.TO_END: "END", Kind.PASS: "P", Kind.YES: "YES",
    Kind.NO: "NO", Kind.OPTION: "OPT", Kind.SELECT: "SEL", Kind.UNSELECT: "UNS",
    Kind.FINISH: "FIN", Kind.CANCEL: "CAN", Kind.POSITION: "POS", Kind.NUMBER: "NUM",
    Kind.DECLARE: "DEC", Kind.SORT: "SORT", Kind.RACE: "RACE", Kind.ATTRIBUTE: "ATTR",
    Kind.COUNTER: "COUNT", Kind.RPS: "RPS",
}  # fmt: skip


def _available_kinds(obs) -> list[str]:
    """Short kind labels of the current options, in order (for the no-match message)."""
    out = []
    for k, _ in _opts(obs):
        try:
            s = _SHORT_KIND.get(Kind(k), _kind(k))
        except ValueError:
            s = str(k)
        if s not in out:
            out.append(s)
    return out


def _no_match_msg(obs, tok: str, names: dict[int, str] | None = None) -> str:
    """The `!!` line when `tok` matches no option: which kinds are on offer (and a hint for BP)."""
    if tok.upper() == "P" and "global" in obs and int(obs["global"][21]) in _YESNO_MSGS:
        return "!! P does not answer yes/no prompts: use YES or NO"
    msg = int(obs["global"][21]) if "global" in obs else -1
    head, _, raw_sub = tok.partition(":")
    head = head.upper()
    if head in ("SEL", "SELECT"):
        mode, found = _sel_cands(obs, head, raw_sub, names or {})
        if found:
            best_rank = min(r for r, _ in found)
            best = [i for r, i in found if r == best_rank]
            if len(best) > 1 or mode == "face":
                listed = "; ".join(_opt_label(obs, i, names or {}) for i in best)
                if mode == "face":
                    return (
                        f"!! token {tok!r} matches several face-down cards: {listed} — "
                        "pick one with SEL:own/opp:<card>, SEL:<M/S><zone> (e.g. SEL:opp:S2) "
                        "or the option number"
                    )
                return (
                    f"!! token {tok!r} matches several options: {listed} — qualify with "
                    "SEL:own/opp:<card> or SEL:<M/S><zone>, or pick the option number"
                )
        if (raw_sub and _norm(raw_sub) == "set") or "facedown" in _norm(raw_sub):
            return f"!! token {tok!r} matches no face-down card at this prompt"
    elif (raw_sub and _norm(raw_sub) == "set") or "facedown" in _norm(raw_sub):
        cands = _facedown_cands(obs, head)
        if len(cands) > 1:
            try:
                disp = _display_names(obs, names or {})
            except (IndexError, TypeError, KeyError):
                disp = [f"option {i}" for i in cands]
            listed = "; ".join(f"{i}: {disp[i]}" for i in cands)
            return (
                f"!! token {tok!r} matches several face-down cards: {listed} — "
                "pick one with SEL:<card> (zone suffix) or the option number"
            )
        if not cands:
            return f"!! token {tok!r} matches no face-down card at this prompt"
    avail = ", ".join(_available_kinds(obs)) or "none"
    extra = ""
    if head.upper() == "BP" and Kind.TO_BATTLE not in [k for k, _ in _opts(obs)]:
        extra = f" (no Battle Phase option at this prompt ({_MSGS.get(msg, msg)}); answer it first)"
    if head.upper() == "BP" and extra:
        return (
            f"!! BP: no Battle Phase option at this prompt ({_MSGS.get(msg, msg)}); "
            f"answer it first (available: {avail})"
        )
    return f"!! token {tok!r} matches no option; available: {avail}{extra}"


def _atk_fallback(obs, tok: str, names: dict[int, str]) -> tuple[int | None, str] | None:
    """`ATK:<card>` typed at a card-selection prompt with no ATTACK option: (index, note) when
    `<card>` selects exactly one option (applied as `SEL:<card>`), (None, `!!` message) when the
    prompt is a selection but the match is ambiguous or empty, None when not applicable (other
    prompts, or an ATTACK option is on offer). Never applies on an ambiguous match."""
    head, _, sub = tok.partition(":")
    if head.upper() != "ATK" or not sub.strip():
        return None
    if int(obs["global"][21]) not in _ATK_SEL_MSGS:
        return None
    if any(k == Kind.ATTACK for k, _ in _opts(obs)):
        return None
    j = _match(obs, "SEL:" + sub, names)
    if j is not None:
        return (j, f"(ATK at a selection prompt: applied as SEL:{sub})")
    opts = [i for i, (k, _) in enumerate(_opts(obs)) if k in (Kind.SELECT, Kind.UNSELECT)]
    listed = "; ".join(_opt_label(obs, i, names) for i in opts) or "none"
    msg = (
        f"!! ATK:{sub}: this prompt is a card selection "
        f"({_MSGS.get(int(obs['global'][21]), '?')}), not an attack — {sub!r} matches no "
        f"single option; options: {listed}"
    )
    return (None, msg)


def _auto(env, obs, st: dict):
    """Answer zone prompts with their first option (positions are real decisions and wait for a token)."""
    while obs is not None and int(obs["global"][21]) == C.MSG_SELECT_PLACE:
        st["choices"].append(0)
        st["labels"].append("auto")
        st["actors"].append(env.player)
        st["turns"].append(int(obs["global"][2]))
        obs = env.step(0)
    return obs


def _chain_len(obs) -> int:
    m = obs["chain_mask"]
    try:
        return int(m.sum())
    except AttributeError:
        return sum(1 for x in m if x)


def _live(obs) -> set:
    """Non-passive options of a prompt as (kind, code, effect description) keys."""
    return {(k, code, option_desc(obs, i)) for i, (k, code) in enumerate(_opts(obs)) if k not in _PASSIVE}


def _parse_hold(s: str) -> list[str]:
    """A `--hold "Card A,Card B"` list as normalized names."""
    return [x.replace("_", " ").strip().lower() for x in s.split(",") if x.strip()]


def _held(st: dict, seat: int, turn: int) -> set[str]:
    """Held card names of `seat` for `turn` (empty when none were set or the turn changed)."""
    box = (st.get("hold") or {}).get(str(seat))
    if not box or box.get("turn") != turn:
        return set()
    return set(box.get("names", []))


def _hold_covers(obs, held: set[str], names: dict[int, str]) -> bool:
    """Whether every live (non-passive) option of the prompt is a held card."""
    if not held:
        return False
    live = _live(obs)
    if not live:
        return False
    for _, code, _ in live:
        nm = (names.get(code, "") if code else "").strip().lower()
        if not nm or nm not in held:
            return False
    return True


def _pass_star(env, obs, st: dict, names=None) -> tuple:
    """Answer the passive option (the `P` rule) while the prompt is a MSG_SELECT_CHAIN window for either
    seat whose live options are a subset of what that
    seat already passed on at its latest chain window this turn with no action since (e.g. a held Big Bang
    through Draw and Standby). A `--hold` list additionally passes an EMPTY-chain window whose every live
    option is a held card; any window with chain length > 0 still stops there. Stops at a non-chain message,
    chain length change, turn change, any window of
    a seat that has not yet declined those options, or _PASS_STEPS answers (50 steps max). Optional-trigger prompts (EFFECTYN/YESNO) always stop: they are real
    decisions. Steps are labeled `pass` (zone prompts answered on the way keep the `auto` label).
    Returns (obs, stop reason naming the seat)."""
    turn0 = int(obs["global"][2]) if obs is not None else -1
    chain0 = _chain_len(obs) if obs is not None else -1
    first = True
    reason = f"cap ({_PASS_STEPS} steps)"
    for _ in range(_PASS_STEPS):
        if obs is None:
            reason = "duel over"
            break
        seat = env.player
        msg = int(obs["global"][21])
        if msg != C.MSG_SELECT_CHAIN:
            reason = f"P{seat} {_MSGS.get(msg, msg)} is not a response window"
            break
        if _chain_len(obs) != chain0:
            reason = f"chain changed (P{seat} to act)"
            break
        if int(obs["global"][2]) != turn0:
            reason = f"turn changed (P{seat} to act)"
            break
        live = _live(obs)
        if first:  # the window PASS* was issued at: an explicit decision
            first = False
        elif live <= _DECLINED.get(seat, set()) and ("turn", turn0) in _DECLINED.get(seat, set()):
            pass  # nothing new since that seat's latest decline
        elif (
            chain0 == 0
            and names is not None
            and _hold_covers(obs, _held(st, seat, turn0), names.names)
        ):
            pass  # empty chain and only held cards live (e.g. holding Kirin through your own combo)
        else:
            reason = f"P{seat} has something to decide (not declined since the last action)"
            break
        opts = _opts(obs)
        i = next((j for j, (k, _) in enumerate(opts) if k in _PASSIVE), None)
        if i is None:
            reason = f"P{seat} has no passive option"
            break
        _note(env, obs, i)
        st["choices"].append(i)
        st["labels"].append("pass")
        st["actors"].append(env.player)
        st["turns"].append(int(obs["global"][2]))
        obs = env.step(i)
        obs = _auto(env, obs, st)
    return obs, reason


def decision_json(
    env, obs, index: int, deck, names: CardNames, seen: set, stopped=None, texts: bool = False
) -> dict:
    """The current decision as a machine-readable object (for `--json` and the gold exporter). `seen` holds
    card codes whose effect text was already shown (their `text` is then ""); it is updated in place."""
    out = {
        "index": index,
        "actor": None,
        "deck": deck,
        "msg": None,
        "hint": "",
        "turn": None,
        "phase": None,
        "lp": None,
        "state": "",
        "chain": [],
        "options": [],
        "over": obs is None,
        "winner": env.winner if obs is None else None,
        "stopped": stopped,
    }
    if obs is None:
        return out
    p = env.player
    g = obs["global"]
    msg = int(g[21])
    disp = _display_names(obs, names.names)
    opts = []
    for i, (k, code) in enumerate(_opts(obs)):
        nm = disp[i]
        txt = ""
        if code and (texts or code not in seen):
            txt = _effect_text(code)
            seen.add(code)
        opts.append(
            {
                "i": i,
                "kind": _kind(k),
                "card": nm,
                "desc": option_desc(obs, i),
                "loc": _loc_of(obs, i),
                "text": txt,
            }
        )
    out.update(
        {
            "actor": p,
            "msg": _MSGS.get(msg, str(msg)),
            "hint": _hint(env, obs),
            "turn": int(g[2]),
            "phase": int(g[3]),
            "lp": [int(g[0]), int(g[1])],
            "state": describe_state(obs, names.names),
            "chain": chain_names(obs, names.names),
            "options": opts,
        }
    )
    return out


def _hint(env, obs) -> str:
    """What a card selection is for ("Select the card(s) to discard"), from the engine's select hint."""
    h = int(getattr(env, "select_hint", 0) or 0)
    if not h or obs is None or int(obs["global"][21]) in (C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD):
        return ""
    t = desc_text(h >> 20, h & 0xFFFFF)
    return "" if "%" in t else t


# describe_state line indices always shown in the compact view (turn/LP, hand, both fields);
# GY/banished/deck-count lines print only when changed since that seat's previous view.
_ALWAYS_LINES = (0, 1, 2, 3, 8, 9)
_ED_LINE = 6  # replaced by a count-only line in the compact view


def _extra_count(obs) -> int:
    """Own Extra Deck size (face-down cards are hidden from the opponent's view entirely)."""
    return sum(1 for r in obs["cards"][obs["card_mask"]] if int(r[3]) == 0 and int(r[1]) == 7)


def _compact_lines(full: list[str], prev: list[str] | None, extra_n: int) -> list[str]:
    """Compact view of the full state lines: Extra Deck as a count; GY/banished/deck-count lines only
    when they changed since the seat's previous view; hand, field and LP lines always."""
    out = []
    for idx, line in enumerate(full):
        if idx in _ALWAYS_LINES:
            out.append(line)
        elif idx == _ED_LINE:
            out.append(f"Your Extra Deck: {extra_n} cards")
        elif prev is None or idx >= len(prev) or prev[idx] != line:
            out.append(line)
    return out


def _show(env, obs, st: dict, names: CardNames, texts: bool, compact: bool = False) -> None:
    if obs is None:
        print(f"DUEL OVER: winner {env.winner}")
        return
    p = env.player
    g = obs["global"]
    hint = _hint(env, obs)
    print(
        f"[{len(st['choices'])}] P{p} {st['decks'][p]} to act — {_MSGS.get(int(g[21]))}"
        + (f": {hint}" if hint else "")
    )
    full = describe_state(obs, names.names).splitlines()
    lines = full
    if compact:
        box = (st.get("compact") or {}).get(str(p))
        if box is not None and box.get("turn") == int(g[2]) and isinstance(box.get("lines"), list):
            lines = _compact_lines(full, box["lines"], _extra_count(obs))
        # else: the seat's first view of the turn stays full
    st.setdefault("compact", {})[str(p)] = {"turn": int(g[2]), "lines": full}
    print(
        "\n".join(lines)
        .replace("Your banished: none\n", "")
        .replace("Opponent banished: none\n", "")
    )
    ch = chain_names(obs, names.names)
    if ch:
        print("CHAIN:", " -> ".join(ch))
    if int(g[21]) in _MULTI_MSGS:
        # min/max counts are not exposed by the obs or the env (C++ untouched), so only the picks so far.
        sel = _selected_names(obs, names.names)
        print(f"Selected so far: {', '.join(sel) if sel else 'none'}")
    seen = set(st.setdefault("seen", []))
    disp = _display_names(obs, names.names)
    material = int(g[21]) in _MULTI_MSGS  # card-pick prompts: every line names its card + zone
    for i, (k, code) in enumerate(_opts(obs)):
        nm = disp[i]
        if material and k in (Kind.SELECT, Kind.UNSELECT):
            # Link/material picks (e.g. Promethean Princess): always name the card with its
            # location/zone suffix (not just on duplicates) plus the Link rating, so name
            # tokens work and the agent need not guess indices.
            base = names.names.get(code, f"#{code}") if code else (hidden_card(obs, i) or "(no card)")
            zone = _zone_of(obs, i)
            loc = _loc_of(obs, i)
            suffix = zone or loc
            if suffix:
                nm = f"{base} ({suffix})"
            else:
                nm = base
            lr = _link_rating(code)
            if lr is not None:
                nm = f"{nm} [Link-{lr}]"
        what = option_desc(obs, i)
        nm = f'{nm} "{what}"' if what else nm
        txt = ""
        if code and (texts or code not in seen):
            txt = " | " + _effect_text(code)
            seen.add(code)
        print(f"  {i}: {_kind(k)} {nm}{txt}")
    st["seen"] = sorted(seen)


def _check(engine: Engine, env, obs, st: dict, names: CardNames) -> None:
    """Print a pre-END/BP self-review for the acting seat (no state change): unused options, effects this
    seat already activated this turn, and lethal math (opponent LP vs face-up ATK + blockers)."""
    if obs is None:
        print(f"DUEL OVER: winner {env.winner}")
        print("Unused options:\n  none (duel over)")
        print("Already activated this turn:\n  none (duel over)")
        print("Opponent LP vs your face-up ATK total: duel over")
        return
    p = env.player
    turn = int(obs["global"][2])
    print(f"[{len(st['choices'])}] P{p} {st['decks'][p]} check — turn {turn}")
    print("Unused options:")
    n = 0
    for i, (k, code) in enumerate(_opts(obs)):
        if k in _PASSIVE:
            continue
        if k == Kind.NUMBER:
            nm = str(_arg_of(obs, i))
        else:
            nm = names.names.get(code, f"#{code}") if code else (hidden_card(obs, i) or "(no card)")
        what = option_desc(obs, i)
        nm = f'{nm} "{what}"' if what else nm
        print(f"  {i}: {_kind(k)} {nm}")
        n += 1
    if not n:
        print("  none")
    # effects this seat already activated this turn: replay and record each chosen ACTIVATE of this seat+turn
    print(f"Already activated this turn (P{p}):")
    cfg_env = CppEnv(engine, _config(st))
    o2 = cfg_env.reset()
    done = []
    for k, ch in enumerate(st["choices"]):
        if o2 is None:
            break
        if k < len(st["actors"]) and k < len(st["turns"]) and st["actors"][k] == p and st["turns"][k] == turn:
            oo = _opts(o2)
            if 0 <= int(ch) < len(oo) and oo[int(ch)][0] == Kind.ACTIVATE:
                c = oo[int(ch)][1]
                nm = names.names.get(c, f"#{c}") if c else "(no card)"
                what = option_desc(o2, int(ch))
                done.append(f'{nm} "{what}"' if what else nm)
        o2 = cfg_env.step(int(ch))
    print(f"  {'; '.join(done) if done else 'none'}")
    g = obs["global"]
    rows = obs["cards"][obs["card_mask"]]
    atk = sum(int(r[14]) for r in rows if int(r[3]) == 0 and int(r[1]) == 3 and int(r[5]) == 1 and int(r[0]))
    opp_mons = any(int(r[3]) == 1 and int(r[1]) == 3 for r in rows)
    print(
        f"Opponent LP {int(g[1])} vs your face-up ATK total {atk} "
        f"({'lethal' if atk >= int(g[1]) and not opp_mons else 'not lethal' if not opp_mons else 'blocked'}); "
        f"opponent controls monsters: {'yes' if opp_mons else 'no'}"
    )
    for line in _battle_lines(obs, names.names, _probe_for_check(engine, obs, st)):
        print(line)


def _probe_for_check(engine: Engine, obs, st: dict) -> list[tuple[int, int, bool]] | None:
    """The engine attacker probe for `check`: only at a MSG_SELECT_IDLECMD offering TO_BATTLE."""
    if obs is None or int(obs["global"][21]) != C.MSG_SELECT_IDLECMD:
        return None
    if Kind.TO_BATTLE not in [k for k, _ in _opts(obs)]:
        return None
    return _probe_battle_attackers(engine, st)


def _why_at(notes: list, turns: list[int], actors: list[int], k: int) -> str:
    """The note for decision `k`: latest start <= k within the same turn and seat, else ""."""
    best, text = -1, ""
    for s, t in notes:
        if int(s) <= k and int(s) > best and turns[int(s)] == turns[k] and actors[int(s)] == actors[k]:
            best, text = int(s), str(t)
    return text


def _log(st: dict, names: CardNames, engine: Engine, last_n: int | None = None) -> None:
    """Compact public log of the duel so far (read-only): one turn header with LP per turn, one line per
    non-pass (`gold`) decision: `T3 P0 SUMMON Snake-Eye Ash — why...`. It is what a fresh per-turn session
    reads to continue. `--turns N` keeps only the last N turns."""
    env = CppEnv(engine, _config(st))
    obs = env.reset()
    entries = []  # (turn, actor, line)
    for k, ch in enumerate(st["choices"]):
        if obs is None:
            break
        actor = env.player
        g = obs["global"]
        turn = int(g[2])
        lp = (int(g[0]), int(g[1])) if actor == 0 else (int(g[1]), int(g[0]))  # absolute P0, P1
        label = st["labels"][k] if k < len(st.get("labels", [])) else ""
        if label == "gold" and obs is not None:
            oo = _opts(obs)
            kind, code = oo[int(ch)] if 0 <= int(ch) < len(oo) else (None, 0)
            try:
                kname = Kind(kind).name
            except ValueError:
                kname = str(kind)
            if kind == Kind.NUMBER:
                what = str(_arg_of(obs, int(ch)))
            elif code:
                what = names.names.get(code, f"#{code}")
                desc = option_desc(obs, int(ch))
                what = f'{what} "{desc}"' if desc else what
            else:
                what = option_desc(obs, int(ch))
            line = f"T{turn} P{actor} {kname}" + (f" {what}" if what else "")
            why = ""
            if k < len(st.get("turns", [])) and k < len(st.get("actors", [])):
                why = _why_at(st.get("notes", []), st["turns"], st["actors"], k)
            if why:
                line += f" — {why}"
            entries.append((turn, lp, line))
        obs = env.step(int(ch))
    if last_n is not None:
        turns_seen = sorted({t for t, _, _ in entries})
        keep = set(turns_seen[-last_n:]) if last_n > 0 else set()
        entries = [e for e in entries if e[0] in keep]
    if obs is None:
        print(f"DUEL OVER: winner {env.winner}")
    cur = None
    for turn, lp, line in entries:
        if turn != cur:
            cur = turn
            print(f"== Turn {turn} — LP P0 {lp[0]}, P1 {lp[1]}")
        print(line)
    if not entries:
        print("(no decisions yet)")


_POS_SHORT = {1: "ATK", 2: "face-down ATK", 3: "DEF", 4: "face-down DEF", 5: "face-up", 6: "set"}

_CARD_DB: dict | None = None


def _card_db() -> dict:
    """Card data by code (attack/type for the summon-then-attack scan), loaded once."""
    global _CARD_DB
    if _CARD_DB is None:
        try:
            _CARD_DB = load_cards()
        except Exception:  # noqa: BLE001 - without card data the scan is skipped
            _CARD_DB = {}
    return _CARD_DB


def _no_direct_attack(code: int) -> bool:
    return "cannot attack directly" in _effect_text(code).lower()


def _cannot_attack(code: int) -> bool:
    t = _effect_text(code).lower()
    return "cannot attack" in t and "cannot attack directly" not in t


def _summon_attack_lines(obs, names: dict[int, str], direct: int) -> list[str]:
    """Summon-then-attack scan (estimate): for each SUMMON/SPSUMMON option on offer whose card is a
    monster (Extra Deck Link/Xyz/Synchro/Fusion or a hand monster), its ATK from the card data, so the
    agent sees e.g. "Apollousa 2800 (+ remaining attackers) vs LP 3300". Materials are not modelled."""
    out = []
    cands = []  # (name, atk or None, loc)
    for i, (k, code) in enumerate(_opts(obs)):
        if k not in (Kind.SUMMON, Kind.SPSUMMON) or not code:
            continue
        d = _card_db().get(code)
        if d is None or not (d.type & C.TYPE_MONSTER):
            continue
        loc = _loc_of(obs, i)
        cands.append((names.get(code, f"#{code}"), d.attack if d.attack >= 0 else None, loc))
    if not cands:
        return out
    lp = int(obs["global"][1])
    out.append("  Summon-then-attack (estimate, materials not modelled):")
    for n, atk, loc in cands:
        where = f" ({loc})" if loc else ""
        if atk is None:
            out.append(f"    {n}{where}: ATK unknown")
            continue
        total = atk + direct
        out.append(f"    {n}{where} [ATK {atk}] + remaining attackers {direct} = {total} vs LP {lp}"
                   + (" (lethal)" if total >= lp else ""))
    return out


_PROBE_STEPS = 20  # max passive answers while stepping a scratch duel into its Battle Phase


def _probe_battle_attackers(engine: Engine, st: dict) -> list[tuple[int, int, bool]] | None:
    """Engine-true attackers behind a Main Phase TO_BATTLE option: replay the duel in a scratch env
    (read-only: `st` is never mutated), step TO_BATTLE, answer any chain/yes-no windows with the
    passive option, and read the MSG_SELECT_BATTLECMD ATTACK options.

    Returns [(code, atk, direct_ok)] (direct_ok = the engine's direct-attack flag, obs arg col 5),
    or None when there is no TO_BATTLE on offer, the Battle Phase cannot be reached passively, or
    the duel ends on the way. There is no hidden-info issue: nothing is printed from the other
    seat's view (only this seat already sees)."""
    try:
        env = CppEnv(engine, _config(st))
        obs = env.reset()
        # exact replay: recorded choices already include the auto-answered zone prompts
        for ch in st["choices"]:
            if obs is None:
                return None
            obs = env.step(int(ch))
        if obs is None or int(obs["global"][21]) != C.MSG_SELECT_IDLECMD:
            return None
        tb = next((i for i, (k, _) in enumerate(_opts(obs)) if k == Kind.TO_BATTLE), None)
        if tb is None:
            return None
        obs = env.step(tb)
        for _ in range(_PROBE_STEPS):
            while obs is not None and int(obs["global"][21]) == C.MSG_SELECT_PLACE:
                obs = env.step(0)
            if obs is None:
                return None
            if int(obs["global"][21]) == C.MSG_SELECT_BATTLECMD:
                break
            j = next((i for i, (k, _) in enumerate(_opts(obs)) if k in _PASSIVE), None)
            if j is None:
                return None
            obs = env.step(j)
        else:
            return None
        if obs is None or int(obs["global"][21]) != C.MSG_SELECT_BATTLECMD:
            return None
        rows = list(obs["cards"][obs["card_mask"]])
        out = []
        for i, (k, code) in enumerate(_opts(obs)):
            if k != Kind.ATTACK:
                continue
            r = _opt_row(obs, i)
            atk = int(rows[r][14]) if 0 <= r < len(rows) else 0
            out.append((code, atk, bool(_arg_of(obs, i))))
        return out
    except Exception:  # noqa: BLE001 - any replay failure falls back to the text estimate
        return None


def _battle_lines(
    obs, names: dict[int, str], engine_attackers: list[tuple[int, int, bool]] | None = None
) -> list[str]:
    """Battle helper: own attackers (ATK, position), opponent monsters, direct-damage total, a greedy
    attack assignment vs face-up monsters, a summon-then-attack scan, and warnings for Attack Position
    monsters below the opponent's best face-up ATK (or with 0 ATK).

    ATTACK options exist only in the Battle Phase. In the Main Phase `check` probes the engine
    (`engine_attackers`: the ATTACK options behind TO_BATTLE, with the engine's direct-attack flag),
    so per-turn "cannot attack directly" clauses applied by other cards are modelled; only when no
    TO_BATTLE is on offer do the attackers fall back to an estimate from this seat's face-up Attack
    Position monsters (text heuristic, clearly labelled). The obs does not expose whether a monster
    already attacked this turn, so the estimate assumes each one can still attack.
    """
    rows = list(obs["cards"][obs["card_mask"]])

    def nm(code: int) -> str:
        return names.get(code, f"#{code}") if code else "(hidden)"

    attackers = []  # (name, atk, direct_ok)
    estimated = False
    probed = False
    for i, (k, code) in enumerate(_opts(obs)):
        if k != Kind.ATTACK:
            continue
        r = _opt_row(obs, i)
        atk = int(rows[r][14]) if 0 <= r < len(rows) else 0
        attackers.append((nm(code), atk, bool(_arg_of(obs, i))))
    if not attackers and engine_attackers is not None:
        # Main Phase with TO_BATTLE on offer: the engine's own attacker list (u33 Cooky class).
        probed = True
        attackers = [(nm(code), atk, direct_ok) for code, atk, direct_ok in engine_attackers]
    if not attackers:
        # No ATTACK option and no probe (e.g. first turn, no Battle Phase): text-heuristic estimate.
        estimated = True
        for r in rows:
            if int(r[3]) != 0 or int(r[1]) != 3 or int(r[5]) != 1:
                continue
            code = int(r[0])
            if not code or _cannot_attack(code):
                continue
            attackers.append((nm(code), int(r[14]), not _no_direct_attack(code)))
    foes = []  # (name, atk, dfn, pos, faceup)
    for r in rows:
        if int(r[3]) != 1 or int(r[1]) != 3:
            continue
        code, pos = int(r[0]), int(r[5])
        if code and pos in (1, 3):
            foes.append((nm(code), int(r[14]), int(r[15]), pos, True))
        else:
            foes.append((nm(code), 0, 0, pos, False))
    if probed:
        out = ["Battle (engine: attackers and direct flags from the Battle Phase behind TO_BATTLE):"]
    else:
        out = ["Battle (estimate: no ATTACK option now — attackers from your face-up ATK monsters):"
               if estimated else "Battle:"]
    no_bp = estimated and int(obs["global"][2]) == 1
    if no_bp:
        out.append("  No Battle Phase on the first turn of the duel (attackers are hypothetical).")
    # The engine's direct flag is only meaningful when the opponent controls no monsters (with
    # blockers on board every flag reads 0); the text heuristic has no such caveat.
    engine_flags = not estimated
    field_empty = not any(int(r[3]) == 1 and int(r[1]) == 3 for r in rows)
    if attackers:
        bits = []
        for n, a, direct_ok in attackers:
            if direct_ok:
                tag = ""
            elif engine_flags and field_empty:
                tag = " (engine: cannot attack directly)"
            elif estimated:
                tag = " (text: cannot attack directly)"
            else:
                tag = ""
            bits.append(f"{n} [ATK {a}]{tag}")
        out.append("  Your attackers: " + "; ".join(bits))
    elif estimated:
        out.append("  Your attackers: none (no face-up Attack Position monsters)")
    else:
        out.append("  Your attackers: none (no ATTACK option right now)")
    if foes:
        bits = []
        for n, a, d, pos, up in foes:
            bits.append(f"{n} [{_POS_SHORT.get(pos, '?')}, {a}/{d}]" if up else f"{n} [face-down, unknown]")
        out.append("  Opponent monsters: " + "; ".join(bits))
    else:
        out.append("  Opponent monsters: none")
    if engine_flags and field_empty:
        # engine-true lethal math (u33 Cooky): attackers the engine bars from direct attacks
        # do not count toward the direct total
        direct = sum(a for _, a, ok in attackers if ok)
    else:
        direct = sum(a for _, a, _ in attackers)
    out.append(f"  Direct-damage total if the opponent's field were empty: {direct}")
    out.extend(_summon_attack_lines(obs, names, direct))
    faceup = [t for t in foes if t[4]]
    if attackers and faceup:
        taken: set[int] = set()
        total = 0
        out.append("  Attack plan (greedy, best damage first):")
        for n, a, direct_ok in sorted(attackers, key=lambda t: -t[1]):
            best = None  # (damage, target index)
            for j, (tn, ta, td, pos, _) in enumerate(faceup):
                if j in taken:
                    continue
                if pos == 1 and a > ta:
                    dmg = a - ta
                elif pos == 3 and a > td:
                    dmg = 0
                else:
                    continue
                if best is None or (dmg, ta + td) > (best[0], faceup[best[1]][1] + faceup[best[1]][2]):
                    best = (dmg, j)
            if best is not None:
                _, j = best
                taken.add(j)
                total += best[0]
                out.append(f"    {n} -> {faceup[j][0]} (deals {best[0]})")
            elif direct_ok:
                total += a
                out.append(f"    {n} -> direct ({a})")
            else:
                out.append(f"    {n} holds (beats nothing face-up)")
        out.append(f"  Plan damage total: {total} vs LP {int(obs['global'][1])}")
    elif attackers:
        if any(f[4] is False for f in foes):
            out.append("  No face-up targets: attacks into face-down monsters are a guess (assume 1500 DEF).")
        else:
            out.append(f"  All attackers direct: {direct} vs LP {int(obs['global'][1])}")
    opp_best = max([t[1] for t in faceup], default=0)
    for r in rows:
        if int(r[3]) != 0 or int(r[1]) != 3 or int(r[5]) != 1:
            continue
        code, a = int(r[0]), int(r[14])
        if a == 0:
            out.append(f"  WARNING: {nm(code)} has 0 ATK in Attack Position")
        elif faceup and a < opp_best:
            out.append(
                f"  WARNING: {nm(code)} (ATK {a}) in Attack Position below "
                f"the opponent's best face-up ATK {opp_best}"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", type=str.lower, choices=["new", "show", "do", "undo", "save", "check", "log"])
    ap.add_argument("game")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--why", default="")
    ap.add_argument("--texts", action="store_true")
    ap.add_argument("--turns", type=int, default=None,
                    help="with log: only the last N turns (default: all turns)")
    ap.add_argument("--json", action="store_true", help="print the state as one JSON object (show/do)")
    ap.add_argument("--auto-pass", action="store_true", help="append PASS* after the last token (do)")
    ap.add_argument("--cross", action="store_true", help="let tokens continue after the acting seat changes")
    ap.add_argument("--full", action="store_true", help="print the full view instead of the compact one (do)")
    ap.add_argument(
        "--hold",
        default=None,
        help='comma-separated card names held this turn, e.g. --hold "Kirin,Belle" (do); '
        "PASS* then also passes empty-chain windows whose only live options are held cards",
    )
    a = ap.parse_args()
    engine, names = Engine(), CardNames()
    if a.cmd == "new":
        d0, d1, seed = a.rest
        st = {
            "decks": [d0, d1],
            "seed": int(seed),
            "choices": [],
            "labels": [],
            "actors": [],
            "turns": [],
            "notes": [],
        }
        env, obs = _replay(engine, st)
        obs = _auto(env, obs, st)
        _show(env, obs, st, names, a.texts)
        _store(a.game, st)
        return
    st = _load(a.game)
    if a.cmd == "undo":
        n = int(a.rest[0]) if a.rest else 1
        for key in ("choices", "labels", "actors", "turns"):
            st[key] = st[key][:-n] if n else st[key]
        st["notes"] = [x for x in st["notes"] if x[0] < len(st["choices"])]
        # also drop trailing auto answers so the state ends on a real decision
        while st["labels"] and st["labels"][-1] == "auto":
            for key in ("choices", "labels", "actors", "turns"):
                st[key].pop()
    env, obs = _replay(engine, st)
    if a.cmd == "check":
        _check(engine, env, obs, st, names)
        return
    if a.cmd == "log":
        _log(st, names, engine, a.turns)
        return
    if a.cmd == "do":
        start = len(st["choices"])
        actor = env.player if obs is not None else None
        stopped = "ok"
        prev_attack = False  # the last choice applied in this command declared an attack
        first_opts = _prompt_key(obs) if obs is not None else ()
        toks = " ".join(a.rest).split()
        if a.auto_pass:
            toks.append("PASS*")
        if a.hold is not None and obs is not None and actor is not None:
            st.setdefault("hold", {})[str(actor)] = {
                "turn": int(obs["global"][2]),
                "names": _parse_hold(a.hold),
            }
        for tok in toks:
            if obs is None:
                stopped = "duel over"
                break
            # Attack bridge: several ATK: tokens in one `do` (one attack per token). After an
            # attack resolves the prompt may leave the seat (opponent chain windows) and come
            # back to MSG_SELECT_BATTLECMD for the same seat; pass those empty windows the same
            # way --auto-pass does. Every other safety stop still applies below.
            if (prev_attack and tok.partition(":")[0].upper() == "ATK" and obs is not None
                    and int(obs["global"][21]) == C.MSG_SELECT_CHAIN):
                n0 = len(st["choices"])
                obs, reason = _pass_star(env, obs, st, names)
                if len(st["choices"]) != n0:
                    print(f"attack bridge passed {len(st['choices']) - n0} windows: {reason}")
            # PASS* may continue across seat changes: _pass_star still stops at any new decision
            # (a seat with something undecided), a non-chain prompt, a chain change or a turn change.
            if env.player != actor and not a.cross and tok.upper() != "PASS*":
                print(f"!! player changed to P{env.player}; stopped before {tok!r}")
                stopped = f"player changed to P{env.player}"
                break
            if tok.upper() == "PASS*":
                n0 = len(st["choices"])
                obs, reason = _pass_star(env, obs, st, names)
                print(f"PASS* answered {len(st['choices']) - n0} windows: {reason}")
                stopped = f"PASS*: {reason}"
                prev_attack = False
                continue
            if tok.isdigit() and _prompt_key(obs) != first_opts:
                print(
                    f"!! prompt changed after the first choice; option numbers refer to the first prompt. "
                    f"Use names (SEL:<card>) or issue a new command. Stopped before {tok!r}"
                )
                stopped = f"prompt changed before {tok!r}"
                break
            i = _match(obs, tok, names.names)
            note = ""
            if i is None:
                fb = _atk_fallback(obs, tok, names.names)
                if fb is not None:
                    if fb[0] is not None:
                        i, note = fb
                    else:
                        print(fb[1])
                        stopped = f"ATK at a selection prompt: {tok!r}"
                        break
            if i is None:
                msg = _no_match_msg(obs, tok, names.names)
                print(msg)
                stopped = ("P on a yes/no prompt" if msg.startswith("!! P does not answer")
                           else f"no match for {tok!r}")
                break
            k, code = _opts(obs)[i]
            nm = names.names.get(code, f"#{code}") if code else (hidden_card(obs, i) or "(no card)")
            what = option_desc(obs, i)
            if not a.json:
                print(f'  [{len(st["choices"])}] P{env.player} {_kind(k)} {nm} "{what}"')
            if note and not a.json:
                print(note)
            _note(env, obs, i)
            st["choices"].append(i)
            st["labels"].append("gold")
            st["actors"].append(env.player)
            st["turns"].append(int(obs["global"][2]))
            prev_attack = (k == Kind.ATTACK)
            obs = env.step(i)
            obs = _auto(env, obs, st)
        if a.why and len(st["choices"]) > start:
            st["notes"].append([start, a.why])
    if a.cmd == "save":
        OUT.mkdir(parents=True, exist_ok=True)
        cfg = _config(st)
        rec = {
            "game_id": f"gold-{a.game}",
            "line_id": f"{st['decks'][0]}-vs-{st['decks'][1]}/gold-{a.game}",
            "level": "L4",
            "decks": st["decks"],
            "config": {
                "deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
                "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
                "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw, "lp": cfg.lp, "setup": [],
                "shuffle": cfg.shuffle,
            },
            "auto_single": True,
            "choices": st["choices"], "labels": st["labels"], "actors": st["actors"], "turns": st["turns"],
            "notes": st["notes"],
            "winner": env.winner if obs is None else None,
            "end_turn": st["turns"][-1] if st["turns"] else 0,
            "source": "hand-played (Claude), both seats",
        }  # fmt: skip
        path = OUT / f"{st['decks'][0]}-vs-{st['decks'][1]}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"saved {rec['game_id']} ({len(st['choices'])} choices, winner {rec['winner']}) -> {path}")
        return
    if a.json and a.cmd in ("show", "do"):
        seen = set(st.setdefault("seen", []))
        deck = st["decks"][env.player] if obs is not None else None
        print(
            json.dumps(
                decision_json(
                    env,
                    obs,
                    len(st["choices"]),
                    deck,
                    names,
                    seen,
                    stopped if a.cmd == "do" else None,
                    a.texts,
                )
            )
        )
        st["seen"] = sorted(seen)
        _store(a.game, st)
        return
    _show(env, obs, st, names, a.texts, compact=(a.cmd == "do" and not a.full))
    _store(a.game, st)


if __name__ == "__main__":
    main()
