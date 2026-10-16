"""Python reference env: one duel, both seats driven through `step(option_index)`.

The observation is always for the player to act (`env.player`). A game is reproduced by
(DuelConfig, auto_single, the list of chosen indices), see `PyEnv.replay`.

    env = PyEnv(engine, config)
    obs = env.reset()
    while not env.done:
        obs = env.step(policy(obs))
"""

from collections.abc import Sequence

import numpy as np

from ygo_zero.engine import obs as O
from ygo_zero.engine.actions import Decision, Option, make_decision
from ygo_zero.engine.duel import Duel, DuelConfig, Engine

Obs = dict[str, np.ndarray]


class PyEnv:
    def __init__(self, engine: Engine, config: DuelConfig, auto_single: bool = True) -> None:
        self.engine, self.config, self.auto_single = engine, config, auto_single
        self.duel: Duel | None = None

    # --- public API ---

    def reset(self) -> Obs:
        if self.duel is not None:
            self.duel.close()
        self.duel = self.engine.new_duel(self.config)
        self.history: list[O.HistoryEntry] = []
        self.choices: list[int] = []  # agent choices (auto steps excluded): the replay
        self.decisions = 0  # all single choices, auto ones included
        self.truncated = 0  # decisions whose option list was cut at MAX_OPTIONS
        self.dec: Decision | None = None
        self._next_decision()
        return self._settle()

    @property
    def done(self) -> bool:
        return self.duel.ended

    @property
    def winner(self) -> int | None:
        return self.duel.winner

    @property
    def player(self) -> int:
        return self.dec.msg.player

    @property
    def knowledge_resyncs(self) -> int:
        return self.duel.knowledge_resyncs

    def pending_decision(self) -> Decision | None:
        """The decision the current player must answer (None once the duel ends)."""
        return self.dec

    def history_log(self) -> np.ndarray:
        """All recorded entries in absolute form [n, 14] (see `obs.history_log`)."""
        return O.history_log(self.history)

    def history_len(self) -> int:
        return len(self.history)

    def step(self, index: int) -> Obs | None:
        """Take option `index` of the current observation; returns the next one (None once the duel ends)."""
        if not 0 <= index < len(self.opts):
            raise IndexError(f"option {index} of {len(self.opts)}")
        self.choices.append(index)
        self._take(index, record=True)
        return self._settle()

    @classmethod
    def replay(
        cls, engine: Engine, config: DuelConfig, choices: Sequence[int], auto_single: bool = True
    ) -> "PyEnv":
        env = cls(engine, config, auto_single)
        env.reset()
        for i in choices:
            env.step(i)
        return env

    # --- internals ---

    def _next_decision(self) -> None:
        msg = self.duel.advance()
        self.dec = (
            None
            if msg is None
            else make_decision(msg, self.engine.cards, lambda p=msg.player: self._known_codes(p))
        )

    def _known_codes(self, p: int) -> set[int]:
        """ANNOUNCE_CARD pool: p's deck list, cards p sees now, and codes in p's visible history."""
        deck = self.config.deck0 if p == 0 else self.config.deck1
        codes = set(deck.main) | set(deck.extra) | O.Board(self.duel, p).visible_codes()
        for e in self.history:
            if e.actor == p or e.public:
                codes.add(e.row[2])
        codes.discard(0)
        return codes

    def _take(self, index: int, record: bool) -> None:
        dec, opt = self.dec, self.opts[index]
        if record:
            self.history.append(
                O.history_entry(self.board, opt, dec.msg.type, dec.step, self.duel.turn, self.duel.phase)
            )
        self.decisions += 1
        resp = dec.choose(index)
        if resp is not None:
            self.duel.respond(resp)
            self._next_decision()

    def _settle(self) -> Obs | None:
        """Take automatic steps until a real choice (returns its obs) or the end (None)."""
        while self.dec is not None:
            opts = self.dec.options()
            if not opts:
                raise RuntimeError(f"no legal option for {self.dec.msg}")
            if self.auto_single and len(opts) == 1:
                self.opts = opts
                self._take(0, record=False)
                continue
            self.opts: list[Option] = opts[: O.MAX_OPTIONS]
            self.truncated += len(opts) > O.MAX_OPTIONS
            self.board = O.Board(self.duel, self.player)
            return self._observe()
        return None

    def _observe(self) -> Obs:
        b, dec, me = self.board, self.dec, self.player
        cards, card_mask = b.cards(dec.selected)
        card_known, known_code = b.knowledge(self.duel.memory)
        chain, chain_mask = b.chain_array()
        actions, action_mask = O.actions_array(b, dec, self.opts)
        history, history_mask = O.history_array(self.history, me, self.duel.turn)
        hist_long, hist_long_mask = O.history_array(
            self.history, me, self.duel.turn, n=O.HISTORY_MAX, salient=True
        )
        return {
            "cards": cards, "card_mask": card_mask,
            "card_known": card_known, "known_code": known_code,
            "global": b.global_array(self.duel, dec.msg.type),
            "chain": chain, "chain_mask": chain_mask,
            "actions": actions, "action_mask": action_mask,
            "history": history, "history_mask": history_mask,
            "hist_long": hist_long, "hist_long_mask": hist_long_mask,
        }  # fmt: skip
