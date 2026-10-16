"""Map a raw EDOPro response back to env option-index sequences (EDOPro play D1).

The duel runs in our engine, so a human answer arrives as raw response bytes (exactly what
EDOPro's client sends to a server). The env is driven by option indices instead, where one
pending core message may take several choices (`Decision.choose` returns the bytes only when
the selection is complete). `choices_for_response` inverts that: at the env's current pending
decision, return the index list that, fed to `env.step` one by one, sends exactly `resp`.

Same bytes mean the same game state (the core only sees the bytes), so when several
sequences give the same bytes any of them plays identically; we return the one with the
fewest steps, ties broken by lexicographically smallest indices. In practice every
response determines a unique length (a card set is picked then optionally FINISHed, counters
and bits always take exactly `count` picks), so depth-first search in ascending index order
finds that sequence first.

The search mirrors `PyEnv._settle` exactly: single-option steps are taken automatically
(and take no index) when `env.auto_single`, otherwise options are cut at `O.MAX_OPTIONS`
(a response that needs a cut-off option is not expressible: None). The env is never
mutated: the search runs on cheap copies of the pending `Decision` (only the small mutable
pick-state lists are copied; the engine, card DB and message stay shared and untouched).
"""

import copy
import logging
import struct
import time
from collections import deque

from ygo_zero.engine import actions as A
from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine import obs as O
from ygo_zero.engine import responses as R

logger = logging.getLogger(__name__)

NODE_BUDGET = 200_000  # search nodes per call; exceeding it returns None + a warning

MSG_NAMES = {v: k for k, v in vars(C).items() if k.startswith("MSG_")}


def _clone(dec: A.Decision) -> A.Decision:
    """A copy of a pending decision with independent pick state.

    `Decision.choose` mutates only small attributes (`step`, `selected`, and per-class
    `picked` / `taken`+`zones` / `value` / `left` / `order`); copying every list attribute
    covers those without deep-copying the engine or card DB.
    """
    new = copy.copy(dec)
    for k, v in new.__dict__.items():
        if isinstance(v, list):
            new.__dict__[k] = list(v)
    return new


def _parse_cards(resp: bytes) -> tuple[list[int], frozenset[int]] | None:
    """SELECT_CARD/TRIBUTE/SUM indices, or None when malformed (CANCEL handled separately)."""
    if len(resp) < 8:
        return None
    (kind, n) = struct.unpack_from("<iI", resp)
    if kind != 2 or len(resp) != 8 + n:
        return None
    idx = list(resp[8 : 8 + n])
    return idx, frozenset(idx)


def _single_step(dec: A.Decision, auto_single: bool, resp: bytes) -> list[int] | None:
    """Decisions that complete in one `choose`: try every offered option on a copy."""
    full = _clone(dec).options()
    if auto_single and len(full) == 1:
        return [] if _clone(dec).choose(0) == resp else None
    for i in range(min(len(full), O.MAX_OPTIONS)):
        if _clone(dec).choose(i) == resp:
            return [i]
    return None


def _follow(dec: A.Decision, auto_single: bool, wants: list, resp: bytes) -> list[int] | None:
    """Verify a decoded pick order (`wants`: option predicates) step by step.

    Mirrors `PyEnv._settle`: auto-single steps take no index, otherwise the wanted option
    must be within the `MAX_OPTIONS` cut. Intermediate picks must not complete the
    decision; the last one must, with exactly `resp`.
    """
    d, seq, out = _clone(dec), [], None
    qi: deque = deque(wants)
    while True:
        opts = d.options()
        if not opts:
            return None
        if auto_single and len(opts) == 1:
            if not qi or not qi[0](opts[0]):
                return None
            qi.popleft()
            out = d.choose(0)
        else:
            if not qi:
                return None
            want = qi[0]
            i = next((k for k, o in enumerate(opts[: O.MAX_OPTIONS]) if want(o)), None)
            if i is None:
                return None
            qi.popleft()
            out = d.choose(i)
            seq.append(i)
        if out is not None:
            return seq if (not qi and out == resp) else None
        if not qi:
            return None


def _place_path(dec: A.Decision, auto_single: bool, resp: bytes) -> list[int] | None:
    """SELECT_PLACE/DISFIELD: the bytes list the zones in pick order, decode and follow."""
    msg = dec.msg
    assert isinstance(msg, M.SelectPlace)
    if msg.count <= 0 or len(resp) != 3 * msg.count:
        return None
    args = []
    for off in range(0, len(resp), 3):
        p, loc, s = resp[off], resp[off + 1], resp[off + 2]
        rel = p ^ msg.player
        if loc == C.LOCATION_MZONE and s < 7:
            args.append((rel << 4) + s)
        elif loc == C.LOCATION_SZONE and s < 8:
            args.append((rel << 4) + 8 + s)
        else:
            return None
    wants = [(lambda o, a=a: o.kind == A.Kind.PLACE and o.arg == a) for a in args]
    return _follow(dec, auto_single, wants, resp)


def _sort_path(dec: A.Decision, auto_single: bool, resp: bytes) -> list[int] | None:
    """SORT_CARD/SORT_CHAIN: the bytes are each card's rank; invert to pick order and follow."""
    msg = dec.msg
    assert isinstance(msg, M.SortCards)
    n = len(msg.cards)
    if len(resp) != n or set(resp) != set(range(n)):
        return None
    order = [0] * n
    for i, r in enumerate(resp):
        order[r] = i
    wants = [(lambda o, c=c: o.kind == A.Kind.SORT and o.card == msg.cards[c]) for c in order[:-1]]
    return _follow(dec, auto_single, wants, resp)


class _Prune:
    """Exact per-type pruning for the multi-step search (never prunes a valid path)."""

    def __init__(self, dec: A.Decision, resp: bytes) -> None:
        self.dec, self.resp = dec, resp
        self.ok = True
        if isinstance(dec, A._PickCards):
            self._init_cards()
        elif isinstance(dec, A._Bits):
            self._init_bits()
        elif isinstance(dec, A._Counter):
            self._init_counter()

    def _init_cards(self) -> None:
        if self.resp == R.CANCEL:
            self.cancel, self.cards, self.want = True, None, None
            return
        self.cancel = False
        parsed = _parse_cards(self.resp)
        if parsed is None:
            self.ok = False
            return
        _, self.want = parsed
        refs = self.dec.refs
        if any(i >= len(refs) for i in self.want):
            self.ok = False
            return
        self.cards = {refs[i] for i in self.want}

    def _init_bits(self) -> None:
        msg = self.dec.msg
        race = isinstance(msg, M.AnnounceRace)
        width = 8 if race else 4
        if len(self.resp) != width:
            self.ok = False
            return
        (self.value,) = struct.unpack_from("<Q" if race else "<I", self.resp)
        if self.value & ~msg.available or self.value.bit_count() != msg.count:
            self.ok = False  # each of the `count` picks adds exactly one new available bit

    def _init_counter(self) -> None:
        msg = self.dec.msg
        assert isinstance(msg, M.SelectCounter)
        if len(self.resp) != 2 * len(msg.cards):
            self.ok = False
            return
        self.counts = list(struct.unpack(f"<{len(msg.cards)}h", self.resp))
        if (
            sum(self.counts) != msg.count
            or any(c < 0 or c > have for c, (_, have) in zip(self.counts, msg.cards, strict=True))
        ):
            self.ok = False

    # --- option / state viability ---

    def option_ok(self, d: A.Decision, opt: A.Option) -> bool:
        dec = self.dec
        if isinstance(dec, A._PickCards):
            if self.cancel:
                return False  # CANCEL is only offered before any pick; handled directly
            if opt.kind == A.Kind.FINISH:
                return {dec.refs[i] for i in d.picked} == self.cards
            return opt.kind == A.Kind.SELECT and opt.card in self.cards
        if isinstance(dec, A._Bits):
            bit = (1 << opt.arg) if dec.race else opt.arg
            return bool(self.value & bit)
        if isinstance(dec, A._Counter):
            j = [k for k, (c, _) in enumerate(dec.msg.cards) if c == opt.card]
            taken = dec.msg.cards[j[0]][1] - d.left[j[0]] if j else 0
            return bool(j) and taken < self.counts[j[0]]
        return True

    def state_ok(self, d: A.Decision) -> bool:
        dec = self.dec
        if isinstance(dec, A._PickCards):
            return all(dec.refs[i] in self.cards for i in d.picked)
        if isinstance(dec, A._Bits):
            if d.value & ~self.value:
                return False
            return self.value.bit_count() - d.value.bit_count() == dec.msg.count - d.step
        if isinstance(dec, A._Counter):
            return all(
                have - left <= c
                for (_, have), left, c in zip(dec.msg.cards, d.left, self.counts, strict=True)
            )
        return True


def _search(dec: A.Decision, auto_single: bool, resp: bytes, stats: dict | None) -> list[int] | None:
    """Depth-first search over decision copies, ascending indices (first hit wins)."""
    if isinstance(dec, A._PickCards) and resp == R.CANCEL:
        # CANCEL is only offered before the first pick.
        opts = _clone(dec).options()
        if auto_single and len(opts) == 1:
            return [] if _clone(dec).choose(0) == resp else None
        i = next(
            (k for k, o in enumerate(opts[: O.MAX_OPTIONS]) if o.kind == A.Kind.CANCEL), None
        )
        return [i] if i is not None and _clone(dec).choose(i) == resp else None
    prune = _Prune(dec, resp)
    if not prune.ok:
        return None
    nodes = 0
    stack = [(_clone(dec), [])]
    while stack:
        d, seq = stack.pop()
        nodes += 1
        if nodes > NODE_BUDGET:
            logger.warning("choices_for_response: search budget %d exceeded at %s", NODE_BUDGET, dec.msg)
            if stats is not None:
                stats["bounded"] = True
            return None
        opts = d.options()
        if not opts:
            continue
        if auto_single and len(opts) == 1:
            if not prune.option_ok(d, opts[0]):
                continue
            out = d.choose(0)
            if out is not None:
                if out == resp:
                    if stats is not None:
                        stats["nodes"] = nodes
                    return seq
            elif prune.state_ok(d):
                stack.append((d, seq))
            continue
        trunc = opts[: O.MAX_OPTIONS]
        for i in range(len(trunc) - 1, -1, -1):
            if not prune.option_ok(d, trunc[i]):
                continue
            d2 = _clone(d)
            out = d2.choose(i)
            if out is not None:
                if out == resp:
                    if stats is not None:
                        stats["nodes"] = nodes + 1
                    return seq + [i]
            elif prune.state_ok(d2):
                stack.append((d2, seq + [i]))
    if stats is not None:
        stats["nodes"] = nodes
    return None


def choices_for_response(env, resp: bytes, stats: dict | None = None) -> list[int] | None:
    """Option indices that make the env's pending decision send exactly `resp`.

    `env` is not mutated. Returns None when no such sequence exists: the response is
    invalid, malformed, or not expressible in the env's action space (an option cut at
    `MAX_OPTIONS`, or a search over the node budget). `stats`, when given, is filled with
    the message name, searched nodes and milliseconds.
    """
    t0 = time.perf_counter()
    if stats is not None:
        stats.update(nodes=0, ms=0.0, bounded=False)
    dec = env.pending_decision()
    auto_single = env.auto_single
    if dec is None or not isinstance(resp, (bytes, bytearray)):
        return None
    resp = bytes(resp)
    if stats is not None:
        stats["msg"] = MSG_NAMES.get(dec.msg.type, hex(dec.msg.type))
    if isinstance(dec, (A._Fixed, A._UnselectCard)):
        seq = _single_step(dec, auto_single, resp)
    elif isinstance(dec, A._Place):
        seq = _place_path(dec, auto_single, resp)
    elif isinstance(dec, A._Sort):
        seq = _sort_path(dec, auto_single, resp)
    elif isinstance(dec, (A._PickCards, A._Bits, A._Counter)):
        seq = _search(dec, auto_single, resp, stats)
    else:  # pragma: no cover - make_decision covers every Select subclass
        seq = None
    if stats is not None:
        stats["ms"] = (time.perf_counter() - t0) * 1000.0
    return seq


def decision_groups(engine, config, choices, auto_single: bool = True):
    """Replay `choices` and split them into per-response groups.

    Returns [(msg type, index slice, raw response)] in decision order. Each group is one
    pending core decision: its slice holds the env choices that completed it (empty for
    decisions taken entirely by auto-single steps) and its raw response bytes. Raises on
    replay failure or trailing choices that produce no response. The replay uses a plain
    `PyEnv` subclass that only logs takes, so the grouping matches the env exactly.
    """
    from ygo_zero.engine.env import PyEnv

    class GroupEnv(PyEnv):
        def _take(self, index: int, record: bool) -> None:
            t = self.dec.msg.type
            n = len(self.duel.responses)
            super()._take(index, record)
            self.takes.append((t, index, record, len(self.duel.responses) > n))

    env = GroupEnv(engine, config, auto_single)
    env.takes = []
    env.reset()
    for c in choices:
        env.step(int(c))
    groups: list[tuple[int, list[int], bytes]] = []
    pending: list[tuple[int, int, bool]] = []
    ri = 0
    for t, index, record, completed in env.takes:
        pending.append((t, index, record))
        if completed:
            resp = env.duel.responses[ri]
            ri += 1
            groups.append((pending[0][0], [i for _, i, r in pending if r], resp))
            pending = []
    if ri != len(env.duel.responses) or pending:
        raise ValueError(
            f"grouping failed: {len(env.takes)} takes, {ri}/{len(env.duel.responses)} responses grouped"
        )
    return groups
