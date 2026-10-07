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


# Columns of an option row that describe what the option *means* (obs.py `option_row`):
# [kind, row_of, code, desc_hi, desc_lo, arg, msg_type, loc, ctrl, step]. Cols 1
# (position in the obs card table) and 9 (a counter) are excluded from the key.
_CMP_COLS = [0, 2, 3, 4, 5, 6, 7, 8]


def option_key(row) -> tuple[int, ...]:
    """Meaning of one option row (all columns except `row_of` and `step`)."""
    return (int(row[0]), int(row[2]), int(row[3]), int(row[4]),
            int(row[5]), int(row[6]), int(row[7]), int(row[8]))


def key_rank_of_choice(obs: dict, idx: int) -> tuple[tuple[int, ...], int]:
    """Key of the chosen option plus its rank among the legal options with the same key
    (0 for the first, 1 for the second identical copy, ...; ordered by option index)."""
    acts = obs["actions"]
    mask = obs["action_mask"].astype(bool)
    key = option_key(acts[int(idx)])
    rank = 0
    for j in range(int(idx)):
        if bool(mask[j]) and option_key(acts[j]) == key:
            rank += 1
    return key, rank


def find_by_key_rank(obs: dict, key: tuple[int, ...], rank: int) -> int | None:
    """Legal option index with this key and rank, or the last copy with the key when fewer
    copies exist, or None when no legal option has the key."""
    acts = obs["actions"]
    mask = obs["action_mask"].astype(bool)
    matches = [j for j in range(len(mask)) if bool(mask[j]) and option_key(acts[j]) == key]
    if not matches:
        return None
    if int(rank) < len(matches):
        return matches[int(rank)]
    return matches[-1]


def prefix_keys_for(engine, config, prefix: list[int], auto_single: bool = True,
                    ) -> list[tuple[tuple[int, ...], int]]:
    """(key, rank) of every choice of `prefix` in the true game (for the Determinizer)."""
    env = CppEnv(engine, config, auto_single)
    obs = env.reset()
    out = []
    for c in prefix:
        out.append(key_rank_of_choice(obs, int(c)))
        obs = env.step(int(c))
    return out


def root_obs_equal(a: dict | None, b: dict | None) -> bool:
    """`me`'s root-obs comparison for semantic replay: `actions` matches on the meaning
    columns only (`_CMP_COLS`), the mask and every other array match exactly."""
    if a is None or b is None:
        return a is None and b is None
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if k == "actions":
            if not isinstance(va, np.ndarray) or not isinstance(vb, np.ndarray):
                return False
            if va.shape != vb.shape:
                return False
            if not np.array_equal(va[:, _CMP_COLS], vb[:, _CMP_COLS]):
                return False
        elif isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
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

    `sample() -> (DuelConfig, resolved_prefix) | None`: permute the codes among each team's hidden slots (a
    face-down MZONE slot only swaps with monster codes, a face-down SZONE slot only with
    spell/trap codes), rebuild with `shuffle=False`, replay `prefix_keys` by meaning, and accept only if the
    replay raises nothing and `me`'s obs at the end matches `root_obs` on the meaning columns
    (`root_obs_equal`). After `tries` rejections returns None (the caller falls back to the true config).
    `resolved_prefix` is the world's own option indices, so branches extend it with the root candidate
    index (the root options are identical by the obs check). Without `prefix_keys` (None) replays the
    raw `prefix` indices with the exact `obs_equal` check (the old behaviour, for comparison).
    Rejections are counted by reason in `reasons`: `no_key` (no legal option has the recorded key),
    `replay_error` (the env ended early or a step raised), `obs_mismatch` (root obs differs).
    """

    def __init__(self, engine, config, auto_single: bool, me: int, prefix: list[int],
                 seen: set[int], root_obs: dict, rng, tries: int = 8,
                 prefix_keys: list[tuple[tuple[int, ...], int]] | None = None) -> None:
        self.engine = engine
        self.config = config
        self.auto_single = auto_single
        self.me = me
        self.prefix = list(prefix)
        self.prefix_keys = (None if prefix_keys is None
                            else [(tuple(k), int(r)) for k, r in prefix_keys])
        self.seen = set(seen)
        self.root_obs = root_obs
        self.rng = rng
        self.tries = tries
        self.proposals = 0
        self.accepted = 0
        self.fallbacks = 0
        self.seconds = 0.0
        self.reasons: dict[str, int] = {"no_key": 0, "replay_error": 0, "obs_mismatch": 0}
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
        """One honest world + its own option indices, or None after `tries` rejections."""
        t0 = time.perf_counter()
        try:
            semantic = self.prefix_keys is not None
            for _ in range(self.tries):
                self.proposals += 1
                mains = self._propose_mains()
                if mains is None:
                    mains = {0: list(self.true_main[0]), 1: list(self.true_main[1])}
                world = self._build_config(mains)
                try:
                    env = CppEnv(self.engine, world, self.auto_single)
                    obs = env.reset()
                    if semantic:
                        assert self.prefix_keys is not None
                        resolved: list[int] = []
                        failed: str | None = None
                        for key, rank in self.prefix_keys:
                            if obs is None:
                                failed = "replay_error"
                                break
                            idx = find_by_key_rank(obs, key, rank)
                            if idx is None:
                                failed = "no_key"
                                break
                            resolved.append(idx)
                            obs = env.step(idx)
                        if failed is not None:
                            self.reasons[failed] += 1
                            continue
                    else:
                        resolved = list(self.prefix)
                        for c in self.prefix:
                            obs = env.step(c)
                except Exception:
                    self.reasons["replay_error"] += 1
                    continue
                ok = root_obs_equal(obs, self.root_obs) if semantic else obs_equal(obs, self.root_obs)
                if ok:
                    self.accepted += 1
                    return world, resolved
                self.reasons["obs_mismatch"] += 1
            self.fallbacks += 1
            return None
        finally:
            self.seconds += time.perf_counter() - t0
