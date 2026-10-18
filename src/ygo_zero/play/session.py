"""Transport-independent duel session shared by the EDOPro host and the web server.

A DuelSession owns one duel: the authoritative host PyEnv (auto_single=False, so
every core prompt is a decision), the bot (a `Bot.next_choice` or a product
`ProductDriver` with its lockstep checks), the human's pending decision, and the
record. The EDOPro host (`edopro/host.py`) and the web server (`web/server.py`)
both drive it; only their transports differ.

Singleton rule: single-option prompts carry nothing to choose, so the session
auto-answers them for the bot always, and for the human when `human_auto` is set
(the web client, like the EDOPro client, is only asked multi-option prompts).
The EDOPro host leaves `human_auto` off: its client answers every prompt.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ygo_zero.engine.actions import Kind
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.engine.env import PyEnv

# Chain-window timings (EDOPro TIMING_*) that mark a bare phase-transition window.
_T_PHASES = (
    0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x1000000 | 0x4000000
)  # DP SP M-end BP-start/end EP, BP, BP-step-end
_T_CHAIN_END = 0x8000
_T_DRAW, _T_TOHAND = 0x10000, 0x200000
# What may happen since the human's last decision for a window to stay "quiet".
_QUIET_KINDS = {Kind.PASS, Kind.TO_BATTLE, Kind.TO_MAIN2, Kind.TO_END, Kind.NO}


@dataclass
class DuelResult:
    winner: int | None
    turns: int
    choices: list[int]
    actors: list[int]
    surrendered: bool = False
    disconnected: bool = False
    diverged: bool = False  # product driver lockstep broke: ALERTed, duel stopped


class DriverDiverged(Exception):
    """The product driver's env diverged from the host env (ALERTed): stop the duel."""


class DuelSession:
    """One duel from the host's side: `reset`, then `pump`/`step_human`/`step_bot`.

    Exactly one of `bot` (a `Bot`: `next_choice(env) -> int`) or `launcher` (a
    `ProductLauncher`: `start_duel(...) -> ProductDriver`, `record_info(seed)`)
    must be given. `cfg` fixes the decks and the deal seed; `human_seat` is the
    human's engine seat (the bot holds the other one).
    """

    def __init__(
        self,
        engine: Engine,
        cfg: DuelConfig,
        human_seat: int,
        *,
        bot=None,
        launcher=None,
        seed: int | None = None,
        human_side: tuple = (),
        human_name: str = "human",
        record_root: Path = Path("runs/web"),
        human_auto: bool = False,
        record_meta: dict | None = None,
        driver_prefix: Sequence[int] = (),
        driver_prefix_decisions: Sequence[dict] | None = None,
    ) -> None:
        if (bot is None) == (launcher is None):
            raise ValueError("DuelSession needs exactly one of bot= / launcher=")
        self.engine = engine
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed
        self.human_seat = human_seat
        self.bot_seat = 1 - human_seat
        self.bot = bot
        self.launcher = launcher
        self.human_side = tuple(human_side)
        self.human_name = human_name
        self.record_root = record_root
        self.human_auto = human_auto
        self.record_meta = dict(record_meta or {})  # extra record fields (deck ids, mode, ...)
        # What-if prefix (product-space outer choices + host-verified decisions):
        # the driver replays it verbatim instead of re-searching it.
        self.driver_prefix = tuple(int(c) for c in (driver_prefix or ()))
        self.driver_prefix_decisions = (
            [dict(d) for d in driver_prefix_decisions] if driver_prefix_decisions is not None else None
        )
        self.env = PyEnv(engine, cfg, auto_single=False)
        self.driver = None
        self.product_info: dict | None = None
        self.prod_len = 0  # non-singleton host steps so far (= the driver's outer decision index)
        self.prompt_id = 0  # human multi-option prompts presented so far
        self._prompt_dec = None  # the Decision the current prompt_id belongs to
        self.thinking_since: float | None = None  # set while blocked on the product
        self.result: DuelResult | None = None

    # --- lifecycle ---

    def reset(self) -> None:
        self.env.reset()
        self.prod_len = 0
        self.prompt_id = 0
        self._prompt_dec = None
        self.thinking_since = None
        self.result = None
        self.driver = None
        self.product_info = None
        if self.launcher is not None:
            try:
                self.driver = self.launcher.start_duel(
                    self.engine,
                    self.cfg,
                    self.bot_seat,
                    game_seed=self.seed,
                    ws_seed=self.seed,
                    prefix=self.driver_prefix,
                    prefix_decisions=self.driver_prefix_decisions,
                )
            except TypeError:  # test stubs without prefix support: plain start
                if self.driver_prefix:
                    raise
                self.driver = self.launcher.start_duel(
                    self.engine, self.cfg, self.bot_seat, game_seed=self.seed, ws_seed=self.seed
                )
            self.product_info = self.launcher.record_info(self.seed)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    @property
    def done(self) -> bool:
        return self.env.done

    @property
    def need_human(self) -> bool:
        """A multi-option prompt waits for the human (after singletons are settled)."""
        return not self.env.done and self.env.player == self.human_seat and len(self.env.opts) > 1

    # --- lockstep (driver mode) ---

    def check_driver_step(self, k: int, pre: tuple[int, int]) -> None:
        """Outer decision k must match the host env's pre-step (player, option count)."""
        from ygo_zero.edopro.product_bot import DriverError

        try:
            want = self.driver.decision(k)
        except DriverError as e:
            raise DriverDiverged(str(e)) from None
        if (want["player"], want["n_options"]) != pre:
            raise DriverDiverged(
                f"outer decision {k}: host (player={pre[0]}, options={pre[1]}) vs "
                f"product (player={want['player']}, options={want['n_options']})"
            )

    # --- stepping ---

    def settle(self, side: str = "all") -> int:
        """Auto-answer single-option prompts: the bot's always, the human's too when
        `side == "all"` (web) or `human_auto` is set. Dropped from the driver's
        product space, like the product's CppEnv skips them. Returns steps taken."""
        n = 0
        while not self.env.done and len(self.env.opts) == 1:
            if self.env.player == self.human_seat and side != "all" and not self.human_auto:
                break
            self.env.step(0)
            n += 1
        return n

    def step_bot(self) -> None:
        """One bot decision at a multi-option prompt (the caller sends MSG_WAITING first)."""
        from ygo_zero.edopro.product_bot import DriverError

        env = self.env
        pre = (env.player, len(env.opts))
        if self.driver is not None:
            self.thinking_since = time.monotonic()
            try:
                idx = self.driver.next_bot()
            except DriverError as e:
                raise DriverDiverged(str(e)) from None
            finally:
                self.thinking_since = None
            self.check_driver_step(self.prod_len, pre)
            env.step(idx)
            self.prod_len += 1
        else:
            env.step(self.bot.next_choice(env))

    def step_human(self, index: int) -> None:
        """One human pick: validate, step, and (driver mode, multi-option) feed the
        driver with the lockstep check. Single-option human steps are dropped from
        the product space."""
        from ygo_zero.edopro.product_bot import DriverError  # noqa: F401 (error type only)

        env = self.env
        pre = (env.player, len(env.opts))
        if env.player != self.human_seat:
            raise ValueError(f"not the human's decision (player {env.player})")
        if not 0 <= index < pre[1]:
            raise IndexError(f"option {index} of {pre[1]}")
        env.step(index)
        if self.driver is not None and pre[1] > 1:
            self.driver.feed_human([index])
            self.check_driver_step(self.prod_len, pre)
            self.prod_len += 1

    def pump(self, side: str = "all") -> None:
        """Settle singletons and take bot decisions until a human multi-option
        prompt waits or the duel ends. Raises DriverDiverged on lockstep breaks."""
        while not self.env.done:
            self.settle(side)
            if self.env.done:
                break
            if self.env.player == self.human_seat:
                if len(self.env.opts) > 1:
                    break
                # human singleton left standing (EDOPro mode): the client answers it.
                break
            self.step_bot()

    def current_prompt_id(self) -> int:
        """The prompt id for the waiting human decision (same id across the picks
        of one core message, a fresh id for each new message)."""
        if self.env.dec is not self._prompt_dec:
            self._prompt_dec = self.env.dec
            self.prompt_id += 1
        return self.prompt_id

    def auto_pass_index(self) -> int | None:
        """The pass option of a quiet chain window, else None (the "Auto" chain setting).

        Quiet = an optional SelectChain with an empty chain, opened only by a phase
        transition (draw/standby/main-end/battle/end timings, the draw-phase draw, or a
        chain end), with nothing but passes and phase moves since the human's last
        decision. Windows after a summon, activation, attack, set or any other event keep
        asking, and so do trigger windows (their timing carries the event bits)."""
        from ygo_zero.engine import messages as M
        from ygo_zero.engine.query import parse_chain

        env = self.env
        if not self.need_human:
            return None
        dec = env.pending_decision()
        msg = dec.msg if dec is not None else None
        if not isinstance(msg, M.SelectChain) or msg.forced or msg.spe_count:
            return None
        t = int(msg.hint_timing) | int(msg.hint_timing_other)
        allowed = _T_PHASES | _T_CHAIN_END | ((_T_DRAW | _T_TOHAND) if t & _T_DRAW else 0)
        if t == 0 or t & ~allowed:
            return None
        if parse_chain(env.duel.query_field()):
            return None
        last = -1
        for i in range(len(env.history) - 1, -1, -1):
            if env.history[i].actor == self.human_seat:
                last = i
                break
        for e in env.history[last + 1 :]:
            try:
                if Kind(e.row[0]) not in _QUIET_KINDS:
                    return None
            except ValueError:
                return None
        for i, o in enumerate(env.opts):
            if o.kind == Kind.PASS:
                return i
        return None

    def abandon(self) -> None:
        """Make a blocked `step_bot` return at once (surrender / new game while the
        product thinks): the driver's next_bot raises, its worker gets a poison pick.
        The session must not be stepped afterwards."""
        from ygo_zero.edopro.product_bot import DriverError

        d = self.driver
        if d is None:
            return
        try:
            with d.cond:
                if d.error is None:
                    d.error = DriverError("abandoned by the host")
                d.cond.notify_all()
            d.human_q.put(-1)
        except Exception:  # noqa: BLE001, S110 -- best effort; the duel is over either way
            pass

    # --- ending ---

    def finish(self) -> DuelResult:
        """The duel ended: join the driver (driver mode) and verify lockstep, then
        build the DuelResult. Raises DriverDiverged on mismatch."""
        from ygo_zero.edopro.product_bot import DriverError, host_to_product

        env = self.env
        if self.driver is not None:
            try:
                self.driver.join(timeout=600)
            except DriverError as e:
                raise DriverDiverged(str(e)) from None
            res = self.driver.res
            if res is None:
                raise DriverDiverged("product driver finished with no result")
            mine = host_to_product(self.engine, self.cfg, env.choices)
            if [int(c) for c in mine] != [int(c) for c in res.choices]:
                raise DriverDiverged(
                    f"choice history mismatch: host {len(mine)} vs product {len(res.choices)} steps"
                )
            if res.winner != env.duel.winner or res.truncated:
                raise DriverDiverged(
                    f"ending mismatch: product winner={res.winner} truncated={res.truncated} vs "
                    f"host winner={env.duel.winner}"
                )
        self.result = DuelResult(
            env.duel.winner, env.duel.turn, list(env.choices), [e.actor for e in env.history]
        )
        return self.result

    def surrender(self, by_human: bool = True) -> DuelResult:
        """The human surrenders (or disconnects): the other seat wins."""
        env = self.env
        winner = self.bot_seat if by_human else self.human_seat
        self.result = DuelResult(
            winner, env.duel.turn, list(env.choices), [e.actor for e in env.history], surrendered=by_human
        )
        return self.result

    # --- record ---

    def write_record(
        self, result: DuelResult, seed: int | None = None, auto_passed: set[int] | None = None
    ) -> Path:
        """The host's record format: game.jsonl (replayable) + human.ydk."""
        bot = self.launcher if self.launcher is not None else self.bot
        bot_name = getattr(getattr(bot, "bot_deck", None), "name", "") or "bot"
        return write_record(
            self.record_root,
            self.cfg,
            self.seed if seed is None else seed,
            result,
            human_seat=self.human_seat,
            human_side=self.human_side,
            bot_name=bot_name,
            product_info=self.product_info,
            meta=self.record_meta,
            auto_passed=auto_passed,
        )


def write_record(
    record_root: Path,
    cfg: DuelConfig,
    seed: int,
    result: DuelResult,
    *,
    human_seat: int,
    human_side: tuple = (),
    bot_name: str = "bot",
    product_info: dict | None = None,
    meta: dict | None = None,
    auto_passed: set[int] | None = None,
) -> Path:
    """Write one duel's record directory (game.jsonl + human.ydk + bot.ydk); returns
    game.jsonl. `meta` adds fields (e.g. human_deck, bot_deck, mode, chain)."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = record_root / ts
    for k in range(100):
        d = outdir if k == 0 else record_root / f"{ts}-{k}"
        try:
            d.mkdir(parents=True, exist_ok=False)
            outdir = d
            break
        except FileExistsError:
            continue
    rec = {
        "game_idx": 0,
        "decks": ["human" if i == human_seat else bot_name for i in (0, 1)],
        "seed": seed,
        "choices": result.choices,
        "auto_single": False,  # replay with PyEnv(..., auto_single=False)
        "actors": result.actors,
        "turns": result.turns,
        "winner": result.winner,
        "human_seat": human_seat,
        "surrendered": result.surrendered,
        "disconnected": result.disconnected,
        "diverged": result.diverged,
    }
    if product_info is not None:
        rec["product"] = product_info
    if auto_passed is not None:
        rec["auto_passed"] = sorted(int(x) for x in auto_passed)
    for k, v in (meta or {}).items():
        rec.setdefault(k, v)
    (outdir / "game.jsonl").write_text(json.dumps(rec) + "\n")
    human = cfg.deck0 if human_seat == 0 else cfg.deck1
    ydk = "#main\n" + "".join(f"{c}\n" for c in human.main)
    ydk += "#extra\n" + "".join(f"{c}\n" for c in human.extra)
    ydk += "!side\n" + "".join(f"{c}\n" for c in human_side)
    (outdir / "human.ydk").write_text(ydk)
    bot = cfg.deck1 if human_seat == 0 else cfg.deck0
    bydk = "#main\n" + "".join(f"{c}\n" for c in bot.main)
    bydk += "#extra\n" + "".join(f"{c}\n" for c in bot.extra)
    (outdir / "bot.ydk").write_text(bydk)
    print(
        f"record -> {outdir / 'game.jsonl'} (winner seat {result.winner}, {result.turns} turns)", flush=True
    )
    return outdir / "game.jsonl"
