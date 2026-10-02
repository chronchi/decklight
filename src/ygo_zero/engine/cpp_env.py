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
        main0=list(config.deck0.main), extra0=list(config.deck0.extra),
        main1=list(config.deck1.main), extra1=list(config.deck1.extra),
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

    def set_schedule(self, games: Sequence[tuple[int, int, int]]) -> None:
        """Evaluation mode: play exactly these (deck0 index, deck1 index, seed) games, then idle (player -1).
        Call before `reset`. Finished episodes carry `.index` into this list."""
        self._v.set_schedule(list(games))

    @property
    def remaining(self) -> int:
        return self._v.remaining

    def pop_finished(self) -> list:
        return self._v.pop_finished()

    def episode_config(self, ep) -> DuelConfig:
        return DuelConfig(self.decks[ep.deck0], self.decks[ep.deck1], seed=ep.seed)
