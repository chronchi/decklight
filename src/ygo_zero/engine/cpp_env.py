"""The C++ env (`ygo_zero._ygoenv`, built by scripts/build_ext.sh) behind the same interface as `PyEnv`."""

from collections.abc import Sequence

from ygo_zero import _ygoenv
from ygo_zero.engine.duel import DuelConfig, Engine

assert _ygoenv.OBS_VERSION == 1


def cpp_engine(engine: Engine) -> "_ygoenv.Engine":
    """A C++ engine with the same card data and script index as `engine` (cached on it)."""
    cached = getattr(engine, "_cpp", None)
    if cached is None:
        cached = _ygoenv.Engine()
        for cd in engine.cards.values():
            cached.add_card(
                cd.code, cd.alias, list(cd.setcodes), cd.type, cd.level, cd.attribute, cd.race,
                cd.attack, cd.defense, cd.lscale, cd.rscale, cd.link_marker,
            )  # fmt: skip
        cached.set_scripts({name: str(path) for name, path in engine._script_paths.items()})
        cached.finalize()
        engine._cpp = cached
    return cached


def cpp_config(config: DuelConfig) -> "_ygoenv.DuelConfig":
    return _ygoenv.DuelConfig(
        main0=config.main_of(0), extra0=list(config.deck0.extra),
        main1=config.main_of(1), extra1=list(config.deck1.extra),
        seed=config.seed, flags=config.flags, lp=config.lp, hand=config.hand, draw=config.draw,
        setup=[tuple(p) for p in config.setup],
    )  # fmt: skip


class CppEnv:
    def __init__(self, engine: Engine, config: DuelConfig, auto_single: bool = True) -> None:
        self.config = config
        self._env = _ygoenv.Env(cpp_engine(engine), cpp_config(config), auto_single)

    def reset(self):
        return self._env.reset()

    def step(self, index: int):
        return self._env.step(index)

    def cards(self) -> dict:
        """Every card of both players (DECK/HAND/MZONE/SZONE/GRAVE/REMOVED/EXTRA) with its
        creation id: dict of numpy arrays `player, location, sequence, position, code, cardid,
        is_public, known` (`known` = the other player knows this HAND card's identity)."""
        import numpy as np

        rows = self._env.cards()
        n = len(rows)
        out = {
            "player": np.empty(n, np.int64),
            "location": np.empty(n, np.int64),
            "sequence": np.empty(n, np.int64),
            "position": np.empty(n, np.int64),
            "code": np.empty(n, np.int64),
            "cardid": np.empty(n, np.int64),
            "is_public": np.empty(n, np.bool_),
            "known": np.empty(n, np.bool_),
        }
        for i, c in enumerate(rows):
            out["player"][i] = c.controller
            out["location"][i] = c.location
            out["sequence"][i] = c.sequence
            out["position"][i] = c.position
            out["code"][i] = c.code
            out["cardid"][i] = c.cardid
            out["is_public"][i] = c.is_public
            out["known"][i] = c.known
        return out

    def __getattr__(self, name):  # done, winner, player, turn, errors, choices, decisions, truncated
        return getattr(self._env, name)

    @classmethod
    def replay(
        cls, engine: Engine, config: DuelConfig, choices: Sequence[int], auto_single: bool = True
    ) -> "CppEnv":
        env = cls(engine, config, auto_single)
        env.reset()
        for i in choices:
            env.step(i)
        return env


class VecEnv:
    """N C++ envs stepped by a thread pool. `buf` holds zero-copy views (N, ...) that each `reset`/`step`
    overwrites in place: the obs arrays plus `player` (to act), `done` and `winner` (episode just ended in that
    env; its obs row already shows the next episode). Finished games: `pop_finished()` → replayable records."""

    TRUNCATED = 3  # Episode.winner of a game cut by set_max_decisions

    def __init__(
        self,
        engine: Engine,
        decks: Sequence,
        num_envs: int,
        num_threads: int,
        seed: int,
        auto_single: bool = True,
    ) -> None:
        self.decks = list(decks)
        pairs = [(list(d.main), list(d.extra)) for d in self.decks]
        self._v = _ygoenv.VecEnv(cpp_engine(engine), pairs, num_envs, num_threads, seed, auto_single)
        self.buf = self._v.buffers()
        self.num_envs = num_envs

    def reset(self) -> dict:
        self._v.reset()
        return self.buf

    def step(self, actions) -> dict:
        self._v.step(actions)
        return self.buf

    def async_reset(self) -> dict:
        """Queue an episode start for every env and return at once; collect starts via `recv`."""
        self._v.async_reset()
        return self.buf

    def send(self, ids, actions) -> None:
        """Queue env `ids[k]` to step with `actions[k]` (no wait)."""
        import numpy as np

        self._v.send(np.asarray(ids, dtype=np.int32), np.asarray(actions, dtype=np.int32))

    def recv(self, min_ready: int = 1, max_ready=None):
        """Block until at least `min_ready` envs are ready, return up to `max_ready` ready env ids
        (completion order). Rows `buf[k][ids]` are valid until those ids are sent again; copy what you keep."""
        import numpy as np

        if max_ready is None:
            max_ready = self.num_envs
        out = self._v.recv(int(min_ready), int(max_ready))
        return np.asarray(out, dtype=np.int32)

    def set_schedule(self, games: Sequence[tuple[int, int, int]]) -> None:
        """Evaluation mode: play exactly these (deck0 index, deck1 index, seed) games, then idle (player -1).
        Call before `reset`. Finished episodes carry `.index` into this list."""
        self._v.set_schedule(list(games))

    def set_max_decisions(self, n: int) -> None:
        """Cut every episode after n decisions (0 = never); it finishes with winner `TRUNCATED` (RL rollouts)."""
        self._v.set_max_decisions(n)

    def set_ga_share(self, p: float, start: int, count: int) -> None:
        """Gauntlet pool (deck-aware RL league): with probability `p` an episode draws both
        decks uniformly from `self.decks[start:start+count]` (the gauntlet slice) and flags
        `buf["ga"][i]`; `buf["deck0"][i]`/`buf["deck1"][i]` carry the episode's deck indices
        (-1 for DAGS/idle). `p = 0` keeps behaviour bit-identical. Call before `reset`."""
        self._v.set_ga_share(float(p), int(start), int(count))

    def set_starts(self, starts, p: float, seed: int) -> None:
        """DAGS mid-game starts (RL v2): `starts` is a list of
        `(main0, extra0, main1, extra1, duel_seed, prefix[, hand, draw, lp, setup])` with the mains
        already in final (shuffled) order. With probability `p` a non-scheduled episode starts from
        one of them (uniform pick). `p = 0` keeps behaviour bit-identical. Call before `reset`."""
        self._v.set_starts(list(starts), float(p), int(seed))

    @property
    def dags_used(self) -> int:
        return self._v.dags_used

    @property
    def dags_bad(self) -> int:
        return self._v.dags_bad

    @property
    def remaining(self) -> int:
        return self._v.remaining

    def pop_finished(self) -> list:
        return self._v.pop_finished()

    def episode_config(self, ep) -> DuelConfig:
        return DuelConfig(self.decks[ep.deck0], self.decks[ep.deck1], seed=ep.seed)
