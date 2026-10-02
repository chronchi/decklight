"""Uniform-ish random policy that only sends legal responses. Baseline opponent and engine fuzzer."""

import random

from ygo_zero.engine import constants as C
from ygo_zero.engine import messages as M
from ygo_zero.engine import responses as R
from ygo_zero.engine.announce import declarable_codes
from ygo_zero.engine.resources import CardData


def _sum_check_exact(params: list[int], acc: int, i: int = 0) -> bool:
    """`select_sum_check1` from playerop.cpp: each card counts as its low or (if set) high value."""
    if acc == 0 or i == len(params):
        return False
    o1, o2 = params[i] & 0xFFFF, params[i] >> 16
    if i == len(params) - 1:
        return acc in (o1, o2)
    return (acc > o1 and _sum_check_exact(params, acc - o1, i + 1)) or (
        o2 > 0 and acc > o2 and _sum_check_exact(params, acc - o2, i + 1)
    )


def _sum_check_at_least(params: list[int], acc: int) -> bool:
    total = hi = 0
    lo = 0x7FFFFFFF
    for p in params:
        o1, o2 = p & 0xFFFF, p >> 16
        ms = o2 if o2 and o2 < o1 else o1
        total += ms
        hi += max(o1, o2)
        lo = min(lo, ms)
    return not (hi < acc or total - lo >= acc)


def sum_is_valid(msg: M.SelectSum, chosen: list[int]) -> bool:
    params = [p for _, p in msg.must] + [msg.cards[i][1] for i in chosen]
    if msg.at_least:
        return _sum_check_at_least(params, msg.acc)
    return msg.min <= len(chosen) <= msg.max and _sum_check_exact(params, msg.acc)


class RandomAgent:
    def __init__(self, seed: int = 0, cards: dict[int, CardData] | None = None) -> None:
        self.rng = random.Random(seed)
        self.cards = cards or {}

    def __call__(self, msg: M.Select) -> bytes:
        return self.respond(msg)

    def respond(self, msg: M.Select) -> bytes:
        rng = self.rng
        match msg:
            case M.SelectIdleCmd():
                acts = [(R.IDLE_SUMMON, i) for i in range(len(msg.summonable))]
                acts += [(R.IDLE_SPSUMMON, i) for i in range(len(msg.spsummonable))]
                acts += [(R.IDLE_REPOS, i) for i in range(len(msg.repositionable))]
                acts += [(R.IDLE_MSET, i) for i in range(len(msg.msetable))]
                acts += [(R.IDLE_SSET, i) for i in range(len(msg.ssetable))]
                acts += [(R.IDLE_ACTIVATE, i) for i in range(len(msg.activatable))]
                acts += [(R.IDLE_BATTLE, 0)] * msg.can_battle_phase + [(R.IDLE_END, 0)] * msg.can_end_phase
                return R.command(*rng.choice(acts))
            case M.SelectBattleCmd():
                acts = [(R.BATTLE_ACTIVATE, i) for i in range(len(msg.activatable))]
                acts += [(R.BATTLE_ATTACK, i) for i in range(len(msg.attackable))]
                acts += [(R.BATTLE_MAIN2, 0)] * msg.can_main2 + [(R.BATTLE_END, 0)] * msg.can_end_phase
                return R.command(*rng.choice(acts))
            case M.SelectChain():
                if msg.chains and (msg.forced or rng.random() < 0.5):
                    return R.int32(rng.randrange(len(msg.chains)))
                return R.CANCEL
            case M.SelectEffectYn() | M.SelectYesNo():
                return R.int32(rng.randrange(2))
            case M.SelectOption() | M.AnnounceNumber():
                return R.int32(rng.randrange(len(msg.options)))
            case M.SelectCard():
                k = rng.randint(msg.min, msg.max)
                return R.cards(sorted(rng.sample(range(len(msg.cards)), k)))
            case M.SelectUnselectCard():
                if (msg.finishable or msg.cancelable) and (not msg.selectable or rng.random() < 0.3):
                    return R.CANCEL
                if msg.selectable:
                    return R.unselect_card(rng.randrange(len(msg.selectable)))
                return R.unselect_card(len(msg.selectable) + rng.randrange(len(msg.unselectable)))
            case M.SelectTribute():
                order = list(range(len(msg.cards)))
                rng.shuffle(order)
                chosen, total = [], 0
                for i in order:
                    if total >= msg.min and (len(chosen) >= msg.max or rng.random() < 0.5):
                        break
                    if len(chosen) < msg.max:
                        chosen.append(i)
                        total += msg.cards[i][1]
                return R.cards(sorted(chosen))
            case M.SelectSum():
                return R.cards(self._random_sum(msg))
            case M.SelectCounter():
                counts = [0] * len(msg.cards)
                for _ in range(msg.count):
                    i = rng.choice([j for j, (_, have) in enumerate(msg.cards) if counts[j] < have])
                    counts[i] += 1
                return R.counters(counts)
            case M.SelectPlace():
                return R.place(rng.sample(self._free_zones(msg), msg.count))
            case M.SelectPosition():
                return R.int32(rng.choice([b for b in (1, 2, 4, 8) if msg.positions & b]))
            case M.SortCards():
                order = list(range(len(msg.cards)))
                rng.shuffle(order)
                return R.sort(order)
            case M.AnnounceRace():
                bits = [1 << i for i in range(64) if msg.available >> i & 1]
                return R.announce_race(sum(rng.sample(bits, msg.count)))
            case M.AnnounceAttrib():
                bits = [1 << i for i in range(32) if msg.available >> i & 1]
                return R.announce_attrib(sum(rng.sample(bits, msg.count)))
            case M.AnnounceCard():
                codes = declarable_codes(self.cards.values(), msg.opcodes)
                if not codes:
                    raise RuntimeError("no declarable card: pass the engine's card dict to RandomAgent")
                return R.int32(rng.choice(codes))
            case M.RockPaperScissors():
                return R.int32(rng.randint(1, 3))
        raise NotImplementedError(type(msg).__name__)

    @staticmethod
    def _free_zones(msg: M.SelectPlace) -> list[tuple[int, int, int]]:
        zones = []
        for half, player in ((0, msg.player), (16, 1 - msg.player)):
            zones += [(player, C.LOCATION_MZONE, s) for s in range(7) if not msg.flag >> (half + s) & 1]
            zones += [(player, C.LOCATION_SZONE, s) for s in range(8) if not msg.flag >> (half + 8 + s) & 1]
        return zones

    def _random_sum(self, msg: M.SelectSum, budget: int = 20_000) -> list[int]:
        """Randomized DFS over subsets; returns the first valid one."""
        order = list(range(len(msg.cards)))
        self.rng.shuffle(order)
        nodes = 0

        def dfs(k: int, chosen: list[int]) -> list[int] | None:
            nonlocal nodes
            nodes += 1
            if nodes > budget:
                return None
            if sum_is_valid(msg, chosen):
                return chosen
            if k == len(order) or (not msg.at_least and len(chosen) >= msg.max):
                return None
            return dfs(k + 1, chosen + [order[k]]) or dfs(k + 1, chosen)

        found = dfs(0, [])
        if found is None:
            raise RuntimeError(f"no valid SELECT_SUM subset found: {msg}")
        return sorted(found)
