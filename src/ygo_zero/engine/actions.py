"""Turn each core decision into a sequence of single choices (docs/action_spec.md).

    dec = make_decision(msg, cards, announce_pool)
    while (resp := dec.choose(pick(dec.options()))) is None:
        pass
    duel.respond(resp)

`options()` lists every legal option in spec order (no truncation here; the env offers the first
MAX_OPTIONS). `choose(i)` returns the core response once the selection is complete, else None.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum

from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine import responses as R
from ygo_zero.engine.announce import is_declarable
from ygo_zero.engine.messages import CardRef
from ygo_zero.engine.random_agent import _sum_check_at_least, _sum_check_exact
from ygo_zero.engine.resources import CardData


class Kind(IntEnum):
    SUMMON = 1
    SPSUMMON = 2
    REPOS = 3
    MSET = 4
    SSET = 5
    ACTIVATE = 6
    ATTACK = 7
    TO_BATTLE = 8
    TO_MAIN2 = 9
    TO_END = 10
    PASS = 11
    YES = 12
    NO = 13
    OPTION = 14
    SELECT = 15
    UNSELECT = 16
    FINISH = 17
    CANCEL = 18
    PLACE = 19
    POSITION = 20
    NUMBER = 21
    ATTRIBUTE = 22
    RACE = 23
    DECLARE = 24
    SORT = 25
    COUNTER = 26
    RPS = 27


@dataclass(frozen=True, slots=True)
class Option:
    kind: Kind
    card: CardRef | None = None
    code: int = 0
    desc: int = 0
    arg: int = 0


class Decision:
    """One pending core message being answered, possibly over several choices."""

    def __init__(self, msg: M.Select) -> None:
        self.msg = msg
        self.step = 0  # picks made so far in this message
        self.selected: list[CardRef] = []

    def options(self) -> list[Option]:
        raise NotImplementedError

    def choose(self, index: int) -> bytes | None:
        opt = self.options()[index]
        self.step += 1
        return self._apply(opt, index)

    def _apply(self, opt: Option, index: int) -> bytes | None:
        raise NotImplementedError


class _Fixed(Decision):
    """Single-step message: a fixed option list, each with its response."""

    def __init__(self, msg: M.Select, pairs: list[tuple[Option, bytes]]) -> None:
        super().__init__(msg)
        self._opts = [o for o, _ in pairs]
        self._resps = [r for _, r in pairs]

    def options(self) -> list[Option]:
        return self._opts

    def _apply(self, opt: Option, index: int) -> bytes:
        return self._resps[index]


def _idle(msg: M.SelectIdleCmd) -> _Fixed:
    pairs = []
    for kind, act, refs in (
        (Kind.SUMMON, R.IDLE_SUMMON, msg.summonable),
        (Kind.SPSUMMON, R.IDLE_SPSUMMON, msg.spsummonable),
        (Kind.REPOS, R.IDLE_REPOS, msg.repositionable),
        (Kind.MSET, R.IDLE_MSET, msg.msetable),
        (Kind.SSET, R.IDLE_SSET, msg.ssetable),
    ):
        pairs += [(Option(kind, c, c.code), R.command(act, i)) for i, c in enumerate(refs)]
    for i, ch in enumerate(msg.activatable):
        opt = Option(Kind.ACTIVATE, ch.card, ch.card.code, ch.description, ch.client_mode)
        pairs.append((opt, R.command(R.IDLE_ACTIVATE, i)))
    if msg.can_battle_phase:
        pairs.append((Option(Kind.TO_BATTLE), R.command(R.IDLE_BATTLE)))
    if msg.can_end_phase:
        pairs.append((Option(Kind.TO_END), R.command(R.IDLE_END)))
    return _Fixed(msg, pairs)


def _battle(msg: M.SelectBattleCmd) -> _Fixed:
    pairs = []
    for i, ch in enumerate(msg.activatable):
        opt = Option(Kind.ACTIVATE, ch.card, ch.card.code, ch.description, ch.client_mode)
        pairs.append((opt, R.command(R.BATTLE_ACTIVATE, i)))
    for i, (c, direct) in enumerate(msg.attackable):
        pairs.append((Option(Kind.ATTACK, c, c.code, arg=int(direct)), R.command(R.BATTLE_ATTACK, i)))
    if msg.can_main2:
        pairs.append((Option(Kind.TO_MAIN2), R.command(R.BATTLE_MAIN2)))
    if msg.can_end_phase:
        pairs.append((Option(Kind.TO_END), R.command(R.BATTLE_END)))
    return _Fixed(msg, pairs)


def _chain(msg: M.SelectChain) -> _Fixed:
    pairs = [
        (Option(Kind.ACTIVATE, ch.card, ch.card.code, ch.description, ch.client_mode), R.int32(i))
        for i, ch in enumerate(msg.chains)
    ]
    if not msg.forced or not msg.chains:
        pairs.append((Option(Kind.PASS), R.CANCEL))
    return _Fixed(msg, pairs)


class _PickCards(Decision):
    """SELECT_CARD / SELECT_TRIBUTE / SELECT_SUM: pick a set of cards one at a time."""

    def __init__(self, msg: M.Select, refs: list[CardRef], params: list[int], cancelable: bool) -> None:
        super().__init__(msg)
        self.refs, self.params, self.cancelable = refs, params, cancelable
        self.picked: list[int] = []

    def _addable(self, i: int) -> bool:
        raise NotImplementedError

    def _can_finish(self) -> bool:
        raise NotImplementedError

    def _auto_send(self) -> bool:
        raise NotImplementedError

    def options(self) -> list[Option]:
        opts = [
            Option(Kind.SELECT, c, c.code, arg=self.params[i])
            for i, c in enumerate(self.refs)
            if i not in self.picked and self._addable(i)
        ]
        if self._can_finish():
            opts.append(Option(Kind.FINISH))
        if self.cancelable and not self.picked:
            opts.append(Option(Kind.CANCEL))
        return opts

    def _apply(self, opt: Option, index: int) -> bytes | None:
        if opt.kind == Kind.CANCEL:
            return R.CANCEL
        if opt.kind == Kind.FINISH:
            return R.cards(sorted(self.picked))
        offered = [i for i in range(len(self.refs)) if i not in self.picked and self._addable(i)]
        self.picked.append(offered[index])  # SELECT options come first, in message order
        self.selected.append(opt.card)
        return R.cards(sorted(self.picked)) if self._auto_send() else None


class _SelectCard(_PickCards):
    def __init__(self, msg: M.SelectCard) -> None:
        super().__init__(msg, msg.cards, [0] * len(msg.cards), msg.cancelable)
        self.min, self.max = msg.min, min(msg.max, len(msg.cards))

    def _addable(self, i: int) -> bool:
        return True

    def _can_finish(self) -> bool:
        return self.min <= len(self.picked) < self.max

    def _auto_send(self) -> bool:
        return len(self.picked) >= self.max


class _SelectTribute(_PickCards):
    def __init__(self, msg: M.SelectTribute) -> None:
        refs = [c for c, _ in msg.cards]
        super().__init__(msg, refs, [p for _, p in msg.cards], msg.cancelable)
        self.min, self.max = msg.min, msg.max

    def _total(self) -> int:
        return sum(self.params[i] for i in self.picked)

    def _addable(self, i: int) -> bool:
        if len(self.picked) + 1 > self.max:
            return False
        need = self.min - self._total() - self.params[i]
        if need <= 0:
            return True
        slots = self.max - len(self.picked) - 1
        rest = sorted(
            (self.params[j] for j in range(len(self.refs)) if j != i and j not in self.picked), reverse=True
        )
        return sum(rest[:slots]) >= need

    def _can_finish(self) -> bool:
        return self._total() >= self.min and len(self.picked) < self.max

    def _auto_send(self) -> bool:
        return len(self.picked) >= self.max


class _SelectSum(_PickCards):
    NODE_BUDGET = 200_000

    def __init__(self, msg: M.SelectSum) -> None:
        refs = [c for c, _ in msg.cards]
        super().__init__(msg, refs, [p for _, p in msg.cards], False)
        self.valid = self._enumerate()
        self.selected = [c for c, _ in msg.must]

    def _is_valid(self, chosen: list[int]) -> bool:
        msg = self.msg
        params = [p for _, p in msg.must] + [self.params[i] for i in chosen]
        if msg.at_least:
            return _sum_check_at_least(params, msg.acc)
        return msg.min <= len(chosen) <= msg.max and _sum_check_exact(params, msg.acc)  # min/max exclude must

    def _enumerate(self) -> list[frozenset[int]]:
        """All valid subsets (indices into msg.cards). Pruned DFS; bounded by NODE_BUDGET."""
        msg, n = self.msg, len(self.refs)
        must_min = sum(_min_param(p) for _, p in msg.must)
        out: list[frozenset[int]] = []
        nodes = 0

        def dfs(k: int, chosen: list[int], min_sum: int) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > self.NODE_BUDGET:
                return
            if chosen and self._is_valid(chosen):
                out.append(frozenset(chosen))
            if k == n:
                return
            if msg.at_least:
                if min_sum >= msg.acc:  # any superset has a redundant card
                    return
            elif len(chosen) >= msg.max or min_sum >= msg.acc:
                return
            for j in range(k, n):
                dfs(j + 1, chosen + [j], min_sum + _min_param(self.params[j]))

        dfs(0, [], must_min)
        if self._is_valid([]):
            out.append(frozenset())
        return out

    def _addable(self, i: int) -> bool:
        cur = set(self.picked) | {i}
        return any(cur <= s for s in self.valid)

    def _can_finish(self) -> bool:
        return frozenset(self.picked) in self.valid

    def _auto_send(self) -> bool:
        return False


def _min_param(p: int) -> int:
    o1, o2 = p & 0xFFFF, p >> 16
    return o2 if o2 and o2 < o1 else o1


class _UnselectCard(Decision):
    def options(self) -> list[Option]:
        msg = self.msg
        opts = [Option(Kind.SELECT, c, c.code) for c in msg.selectable]
        opts += [Option(Kind.UNSELECT, c, c.code) for c in msg.unselectable]
        if msg.finishable:
            opts.append(Option(Kind.FINISH))
        elif msg.cancelable:
            opts.append(Option(Kind.CANCEL))
        return opts

    def _apply(self, opt: Option, index: int) -> bytes:
        if opt.kind in (Kind.FINISH, Kind.CANCEL):
            return R.CANCEL
        return R.unselect_card(index)


def zone_bit(zone: int) -> tuple[int, int, int]:
    """Option arg (0-6 my MZ, 8-15 my SZ, +16 opp) -> (relative player, location, sequence)."""
    rel, z = zone >> 4, zone & 15
    return (rel, C.LOCATION_SZONE, z - 8) if z >= 8 else (rel, C.LOCATION_MZONE, z)


class _Place(Decision):
    def __init__(self, msg: M.SelectPlace) -> None:
        super().__init__(msg)
        self.taken = msg.flag
        self.zones: list[tuple[int, int, int]] = []

    def options(self) -> list[Option]:
        return [
            Option(Kind.PLACE, arg=z)
            for z in (*range(7), *range(8, 16), *range(16, 23), *range(24, 32))
            if not self.taken >> z & 1
        ]

    def _apply(self, opt: Option, index: int) -> bytes | None:
        self.taken |= 1 << opt.arg
        rel, loc, seq = zone_bit(opt.arg)
        self.zones.append((self.msg.player ^ rel, loc, seq))
        return R.place(self.zones) if len(self.zones) == self.msg.count else None


class _Bits(Decision):
    """ANNOUNCE_RACE / ANNOUNCE_ATTRIB: `count` distinct bits, one per step."""

    def __init__(self, msg: M.AnnounceRace | M.AnnounceAttrib) -> None:
        super().__init__(msg)
        self.race = isinstance(msg, M.AnnounceRace)
        self.value = 0

    def options(self) -> list[Option]:
        avail = self.msg.available & ~self.value
        if self.race:
            return [Option(Kind.RACE, arg=b) for b in range(64) if avail >> b & 1]
        return [Option(Kind.ATTRIBUTE, arg=1 << b) for b in range(32) if avail >> b & 1]

    def _apply(self, opt: Option, index: int) -> bytes | None:
        self.value |= (1 << opt.arg) if self.race else opt.arg
        if self.step < self.msg.count:
            return None
        return R.announce_race(self.value) if self.race else R.announce_attrib(self.value)


class _Counter(Decision):
    def __init__(self, msg: M.SelectCounter) -> None:
        super().__init__(msg)
        self.left = [have for _, have in msg.cards]

    def options(self) -> list[Option]:
        return [
            Option(Kind.COUNTER, c, c.code, arg=self.left[i])
            for i, (c, _) in enumerate(self.msg.cards)
            if self.left[i] > 0
        ]

    def _apply(self, opt: Option, index: int) -> bytes | None:
        i = [j for j in range(len(self.left)) if self.left[j] > 0][index]
        self.left[i] -= 1
        if self.step < self.msg.count:
            return None
        return R.counters([have - left for (_, have), left in zip(self.msg.cards, self.left)])


class _Sort(Decision):
    """SORT_CARD / SORT_CHAIN: pick the card that goes next; the response gives each card's rank."""

    def __init__(self, msg: M.SortCards) -> None:
        super().__init__(msg)
        self.order: list[int] = []

    def _left(self) -> list[int]:
        return [i for i in range(len(self.msg.cards)) if i not in self.order]

    def options(self) -> list[Option]:
        return [Option(Kind.SORT, self.msg.cards[i], self.msg.cards[i].code) for i in self._left()]

    def _apply(self, opt: Option, index: int) -> bytes | None:
        self.order.append(self._left()[index])
        self.selected.append(opt.card)
        left = self._left()
        if len(left) > 1:
            return None
        self.order += left
        rank = [0] * len(self.order)
        for r, i in enumerate(self.order):
            rank[i] = r
        return R.sort(rank)


def make_decision(
    msg: M.Select,
    cards: Mapping[int, CardData] | None = None,
    announce_pool: Callable[[], Iterable[int]] | None = None,
) -> Decision:
    """`cards` + `announce_pool` (codes the player may know of) are needed only for ANNOUNCE_CARD."""
    match msg:
        case M.SelectIdleCmd():
            return _idle(msg)
        case M.SelectBattleCmd():
            return _battle(msg)
        case M.SelectChain():
            return _chain(msg)
        case M.SelectEffectYn():
            c, d = msg.card, msg.description
            return _Fixed(
                msg,
                [(Option(Kind.YES, c, c.code, d), R.int32(1)), (Option(Kind.NO, c, c.code, d), R.int32(0))],
            )
        case M.SelectYesNo():
            d = msg.description
            return _Fixed(
                msg, [(Option(Kind.YES, desc=d), R.int32(1)), (Option(Kind.NO, desc=d), R.int32(0))]
            )
        case M.SelectOption():
            return _Fixed(
                msg, [(Option(Kind.OPTION, desc=d, arg=i), R.int32(i)) for i, d in enumerate(msg.options)]
            )
        case M.SelectCard():
            return _SelectCard(msg)
        case M.SelectTribute():
            return _SelectTribute(msg)
        case M.SelectSum():
            return _SelectSum(msg)
        case M.SelectUnselectCard():
            return _UnselectCard(msg)
        case M.SelectPlace():
            return _Place(msg)
        case M.SelectPosition():
            bits = [b for b in (1, 2, 4, 8) if msg.positions & b]
            return _Fixed(msg, [(Option(Kind.POSITION, code=msg.code, arg=b), R.int32(b)) for b in bits])
        case M.AnnounceNumber():
            return _Fixed(msg, [(Option(Kind.NUMBER, arg=n), R.int32(i)) for i, n in enumerate(msg.options)])
        case M.AnnounceRace() | M.AnnounceAttrib():
            return _Bits(msg)
        case M.AnnounceCard():
            codes = declare_options(msg.opcodes, cards or {}, announce_pool() if announce_pool else ())
            return _Fixed(msg, [(Option(Kind.DECLARE, code=c), R.int32(c)) for c in codes])
        case M.SortCards():
            return _Sort(msg)
        case M.SelectCounter():
            return _Counter(msg)
        case M.RockPaperScissors():
            return _Fixed(msg, [(Option(Kind.RPS, arg=v), R.int32(v)) for v in (1, 2, 3)])
    raise NotImplementedError(type(msg).__name__)


def declare_options(opcodes: list[int], cards: Mapping[int, CardData], pool: Iterable[int]) -> list[int]:
    """Declarable codes among the pool (sorted); else the first declarable code in the whole DB."""
    codes = sorted({c for c in pool if c in cards and is_declarable(cards[c], opcodes)})
    if codes:
        return codes
    for c in sorted(cards):
        if is_declarable(cards[c], opcodes):
            return [c]
    raise RuntimeError("no declarable card in the card DB")
