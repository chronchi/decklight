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
