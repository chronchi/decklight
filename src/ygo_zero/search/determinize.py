"""Honest (determinized) search worlds: the search must not see hidden cards.

`WindowSearch` clones a game by replaying the true config + choice prefix, so its playouts know
the opponent's hidden hand, both deck orders and future draws (an oracle). Before each search
sample, everything the searching player `me` has not seen is re-sampled, keeping the world
consistent with what `me` has observed: the codes of hidden card slots are permuted in the
initial deck orders, the prefix is replayed, and the world is accepted only if `me`'s observation
at the root is identical to the true one (rejection sampling).
"""

from __future__ import annotations

import time

import numpy as np

from ygo_zero.engine import constants as C
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig

# cardid of the first main-deck card is derived empirically (see Determinizer): duel creation
# consumes cardid 1 for something that never reaches the board, so team 0's main block starts at 2.
_DECK = C.LOCATION_DECK
_HAND = C.LOCATION_HAND
_MZONE = C.LOCATION_MZONE
_SZONE = C.LOCATION_SZONE


def obs_equal(a: dict | None, b: dict | None) -> bool:
    """Exact equality of two observations (every array, exactly)."""
    if a is None or b is None:
        return a is None and b is None
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not isinstance(va, np.ndarray) or not isinstance(vb, np.ndarray):
                return False
            if va.shape != vb.shape or va.dtype != vb.dtype:
                return False
            if not np.array_equal(va, vb):
                return False
        elif va != vb:
            return False
    return True


def _is_public(loc: int, pos: int, is_pub: bool) -> bool:
    """Mirror of `Board::is_public` (cpp/ygoenv/env.cpp): is this card public to a non-holder?"""
    if loc == C.LOCATION_GRAVE:
        return True
    if bool(is_pub):
        return True
    if loc == _HAND:
        return False
    return bool(pos & C.POS_FACEUP)


class SeenTracker:
    """Cardids player `me` can identify. `update(env)` after every step of the live game adds
    every cardid `me` can identify now; once seen, a card stays seen."""

    def __init__(self, me: int) -> None:
        self.me = me
        self.seen: set[int] = set()

    def update(self, env: CppEnv) -> None:
        cards = env.cards()
        me = self.me
        n = len(cards["code"])
        for i in range(n):
            p = int(cards["player"][i])
            loc = int(cards["location"][i])
            cid = int(cards["cardid"][i])
            pos = int(cards["position"][i])
            if p == me:
                if loc in (_HAND, _MZONE, _SZONE, C.LOCATION_GRAVE, C.LOCATION_REMOVED):
                    self.seen.add(cid)
                elif loc == C.LOCATION_EXTRA and (pos & C.POS_FACEUP):
                    self.seen.add(cid)
                # own deck (unknown order) and own face-down extra stay hidden
            elif loc == _HAND:
                if bool(cards["known"][i]):  # hand_known_: the other player (= me) knows it
                    self.seen.add(cid)
            elif _is_public(loc, pos, bool(cards["is_public"][i])):
                self.seen.add(cid)


class _Deck:
    """DeckLike with an explicit main order (for determinized worlds with shuffle=False)."""

    def __init__(self, main, extra) -> None:
        self.main = list(main)
        self.extra = list(extra)


class Determinizer:
    """Draws honest worlds for one searched decision by rejection sampling.

    `sample() -> DuelConfig | None`: permute the codes among each team's hidden slots (a
    face-down MZONE slot only swaps with monster codes, a face-down SZONE slot only with
    spell/trap codes), rebuild with `shuffle=False`, replay `prefix`, and accept only if the
    replay raises nothing and the obs for `me` at the end equals `root_obs` exactly. After
    `tries` rejections returns None (the caller falls back to the true config).
    """

    def __init__(self, engine, config, auto_single: bool, me: int, prefix: list[int],
                 seen: set[int], root_obs: dict, rng, tries: int = 8) -> None:
        self.engine = engine
        self.config = config
        self.auto_single = auto_single
        self.me = me
        self.prefix = list(prefix)
        self.seen = set(seen)
        self.root_obs = root_obs
        self.rng = rng
        self.tries = tries
        self.proposals = 0
        self.accepted = 0
        self.fallbacks = 0
        self.seconds = 0.0
        # Slot map: at reset, DECK cards are all main-origin, so the main block starts at the
        # lowest deck cardid; main cardids are consecutive in main_of(team) order (verified below).
        probe = CppEnv(engine, config, auto_single)
        probe.reset()
        table = probe.cards()
        self.base: dict[int, int] = {}
        self.true_main: dict[int, list[int]] = {}
        for team in (0, 1):
            main = list(config.main_of(team))
            m = (table["player"] == team) & (table["location"] == _DECK)
            deck_ids = sorted(table["cardid"][m].tolist())
            assert len(deck_ids) + int(((table["player"] == team) & (table["location"] == _HAND)).sum()) >= len(main)
            base = deck_ids[0] if deck_ids else None
            assert base is not None
            want = set(range(base, base + len(main)))
            got: dict[int, int] = {}
            for loc in (_DECK, _HAND):
                mm = (table["player"] == team) & (table["location"] == loc)
                for cid, code in zip(table["cardid"][mm].tolist(), table["code"][mm].tolist()):
                    if cid in want:
                        got[cid] = code
            assert set(got) == want, f"team {team}: main block is not consecutive"
            assert [got[i] for i in sorted(got)] == main, f"team {team}: codes != main_of"
            self.base[team] = base
            self.true_main[team] = main
        # Current positions of every main-deck card under the true config + prefix.
        live = CppEnv(engine, config, auto_single)
        live.reset()
        for c in self.prefix:
            live.step(c)
        cur = live.cards()
        self.cur: dict[int, dict[int, dict]] = {0: {}, 1: {}}
        for i in range(len(cur["code"])):
            p = int(cur["player"][i])
            cid = int(cur["cardid"][i])
            base = self.base.get(p)
            if base is None:
                continue
            slot = cid - base
            if 0 <= slot < len(self.true_main[p]):
                self.cur[p][slot] = {
                    "location": int(cur["location"][i]),
                    "position": int(cur["position"][i]),
                    "code": int(cur["code"][i]),
                    "known": bool(cur["known"][i]),
                    "is_public": bool(cur["is_public"][i]),
                }

    def _hidden_slots(self, team: int) -> list[int]:
        """Main-deck slots of `team` whose code `me` must not see: not in `seen` and still
        concealed (opponent: in the deck, in the hand unknown to me, or face-down on the
        field; me: still in the deck — my hand/field cards are always seen)."""
        base = self.base[team]
        out = []
        for slot, st in self.cur[team].items():
            if base + slot in self.seen:
                continue
            loc = st["location"]
            if team == self.me:
                if loc == _DECK:
                    out.append(slot)
            elif loc in (_DECK, _HAND):
                out.append(slot)
            elif loc in (_MZONE, _SZONE) and not _is_public(loc, st["position"], st["is_public"]):
                out.append(slot)
            # grave/removed/face-up extra are public, hence seen
        return out

    def _code_kind(self, code: int) -> str:
        cd = self.engine.cards.get(code)
        if cd is None:
            return "other"
        t = cd.type
        if t & C.TYPE_MONSTER:
            return "monster"
        if t & (C.TYPE_SPELL | C.TYPE_TRAP):
            return "spell"
        return "other"

    def _slot_need(self, team: int, slot: int) -> str | None:
        """Required code kind for a hidden slot (face-down field), else None."""
        st = self.cur[team][slot]
        if st["location"] == _MZONE:
            return "monster"
        if st["location"] == _SZONE:
            return "spell"
        return None

    def _propose_mains(self) -> dict[int, list[int]] | None:
        """Permuted main-deck orders, or None when no valid permutation was found (identity)."""
        out: dict[int, list[int]] = {}
        for team in (0, 1):
            hidden = self._hidden_slots(team)
            if not hidden:
                out[team] = list(self.true_main[team])
                continue
            codes = [self.cur[team][s]["code"] for s in hidden]
            needs = {s: self._slot_need(team, s) for s in hidden}
            # A face-down MZONE slot only swaps with monster codes, SZONE only with spell/trap.
            m_slots = [s for s in hidden if needs[s] == "monster"]
            s_slots = [s for s in hidden if needs[s] == "spell"]
            kinds = {co: self._code_kind(co) for co in set(codes)}
            m_codes = [co for co in codes if kinds[co] == "monster"]
            s_codes = [co for co in codes if kinds[co] == "spell"]
            if len(m_codes) < len(m_slots) or len(s_codes) < len(s_slots):
                return None  # keep identity (always a valid world)
            m_codes = list(m_codes)
            s_codes = list(s_codes)
            self.rng.shuffle(m_codes)
            self.rng.shuffle(s_codes)
            rest = [co for co in codes if kinds[co] not in ("monster", "spell")]
            # leftover monster/spell codes beyond the constrained slots join the free pool
            rest += m_codes[len(m_slots):] + s_codes[len(s_slots):]
            self.rng.shuffle(rest)
            free = [s for s in hidden if needs[s] is None]
            if len(rest) != len(free):
                return None
            assign = dict(zip(m_slots, m_codes[: len(m_slots)], strict=True))
            assign.update(zip(s_slots, s_codes[: len(s_slots)], strict=True))
            assign.update(zip(free, rest, strict=True))
            new_main = list(self.true_main[team])
            for s, co in assign.items():
                new_main[s] = co
            out[team] = new_main
        return out

    def _build_config(self, mains: dict[int, list[int]]) -> DuelConfig:
        cfg = self.config
        return DuelConfig(
            deck0=_Deck(mains[0], list(cfg.deck0.extra)),
            deck1=_Deck(mains[1], list(cfg.deck1.extra)),
            seed=cfg.seed, flags=cfg.flags, lp=cfg.lp, hand=cfg.hand, draw=cfg.draw,
            setup=cfg.setup, shuffle=False,
        )

    def sample(self):
        """One honest world, or None after `tries` rejections (caller falls back to truth)."""
        t0 = time.perf_counter()
        try:
            for _ in range(self.tries):
                self.proposals += 1
                mains = self._propose_mains()
                if mains is None:
                    mains = {0: list(self.true_main[0]), 1: list(self.true_main[1])}
                world = self._build_config(mains)
                try:
                    env = CppEnv(self.engine, world, self.auto_single)
                    obs = env.reset()
                    for c in self.prefix:
                        obs = env.step(c)
                except Exception:
                    continue
                if obs_equal(obs, self.root_obs):
                    self.accepted += 1
                    return world
            self.fallbacks += 1
            return None
        finally:
            self.seconds += time.perf_counter() - t0
