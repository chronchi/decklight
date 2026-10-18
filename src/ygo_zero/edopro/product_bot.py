"""The product as the EDOPro host's bot (EDOPro play D3): the full product search as `search_match` plays it.

`ProductDriver` runs `search_game_gen` (S = the bot's seat, R = the human) in a worker
thread with search_match's routed forward: base logits for rollouts and S's outer
decisions, one-hot for R's. R's indices come from the human (a queue fed by the host
from D1's `choices_for_response`, or a `human_fn` in tests). The bot's choices are read
from the live `res.choices` (via the `inflight` mirror) and stepped into the host's
authoritative `PyEnv`; after every step both envs must agree on player/done/option
count, else ALERT and the duel stops (never continue silently).

auto_single mismatch (D2 finding): the host env runs `auto_single=False` (every core
prompt is a decision, the client is prompted too) while the product's `CppEnv` runs
`auto_single=True` (single-option steps are not in `res.choices`). The mapping lives in
exactly one place: `host_to_product` / `product_to_host` below (both replay a
`PyEnv(auto_single=False)` and differ only in direction). Live, the host auto-takes the
bot's singletons (index 0) between bot choices and drops the human steps taken at
single-option prompts; the unit test round-trips a recorded game both ways.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np

from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.engine.env import PyEnv
from ygo_zero.search.play import argmax_select, search_game_gen, softmax_conf
from ygo_zero.search.product import (
    RouteWindowSearch,  # noqa: F401 (re-exported for search_match-style use)
    build_search,
    load_policy,
    load_value,
    make_pool,
    product_search_kwargs,
    resolve_product_paths,
    seat_of,
    sha256_file,
    tag_game,
)


class DriverError(RuntimeError):
    """Lockstep divergence, a dead driver, or a human queue underflow: ALERT and stop the duel."""


# --- auto_single mapping (the one place) ---


def _replay_lens(engine: Engine, config: DuelConfig, host_choices: Sequence[int]) -> list[int]:
    """`len(env.opts)` before each step of replaying `host_choices` on PyEnv(auto_single=False)."""
    env = PyEnv(engine, config, auto_single=False)
    env.reset()
    lens = []
    for c in host_choices:
        if env.done:
            raise ValueError(f"host sequence overruns the duel ({len(lens)} steps replayed)")
        lens.append(len(env.opts))
        env.step(int(c))
    return lens


def host_to_product(engine: Engine, config: DuelConfig, host_choices: Sequence[int]) -> list[int]:
    """Host-env (auto_single=False) indices -> product CppEnv (auto_single=True) indices.

    Single-option steps take no index in the product env, so drop the steps taken at
    single-option prompts. This is what the host feeds the driver for human responses.
    """
    lens = _replay_lens(engine, config, host_choices)
    return [int(c) for c, n in zip(host_choices, lens, strict=True) if n > 1]


def product_to_host(engine: Engine, config: DuelConfig, product_choices: Sequence[int]) -> list[int]:
    """Product CppEnv indices -> host-env indices: re-insert index 0 at single-option prompts."""
    env = PyEnv(engine, config, auto_single=False)
    env.reset()
    out: list[int] = []
    it = iter(product_choices)
    pending = True
    while not env.done:
        if len(env.opts) == 1:
            out.append(0)
            env.step(0)
            continue
        if pending:
            try:
                c = int(next(it))
            except StopIteration:
                pending = False
                continue
            out.append(c)
            env.step(c)
        else:
            raise ValueError("product sequence ends before the duel does")
    if pending:
        try:
            next(it)
            raise ValueError("product sequence overruns the duel")
        except StopIteration:
            pass
    return out


def _replay_prefix_decisions(engine: Engine, config: DuelConfig, product_prefix: Sequence[int]) -> list[dict]:
    """Per-outer-decision (player, n_options) for a product-space prefix.

    Replays the product env (CppEnv, auto_single=True — singletons are not
    outer decisions) so a what-if driver started with ``prefix`` can pre-fill
    its decision log without re-searching the prefix.
    """
    from ygo_zero.engine.cpp_env import CppEnv

    env = CppEnv(engine, config, True)
    obs = env.reset()
    out: list[dict] = []
    for i, c in enumerate(product_prefix):
        if obs is None:
            raise ValueError(f"product prefix overruns the duel at step {i}")
        out.append({"player": int(seat_of(obs)), "n_options": int(obs["action_mask"].sum())})
        obs = env.step(int(c))
    return out


# --- the launcher (loads P + value once, spawns one driver per duel) ---


class ProductLauncher:
    """`--opponent product`: holds the loaded product; `start_duel` spawns its driver.

    `rps` is random (the host falls back to it for RPS/TP); per-duel seeds come from --seed.
    """

    def __init__(self, *, ckpt=None, value_ckpt=None, emb=None, device: str = "cpu",
                 batch: int = 512, threads: int = 1, fast: bool = False,
                 replay_threads: int = 1, search_kwargs: dict | None = None) -> None:
        import random

        ckpt_p, value_p, emb_p = resolve_product_paths(ckpt, value_ckpt, emb)
        self.policy = load_policy(ckpt_p, emb_p, device=device, batch=batch, threads=threads)
        self.value = None if fast else load_value(value_p, emb_p, device=device,
                                                  batch=batch, threads=threads)
        self.ckpt, self.value_ckpt, self.emb = ckpt_p, value_p, emb_p
        self.device, self.batch, self.threads = device, batch, threads
        self.fast, self.replay_threads = fast, replay_threads
        self.search_kwargs = dict(search_kwargs or {})
        self.ckpt_sha = sha256_file(ckpt_p)
        self.value_sha = sha256_file(value_p) if value_p is not None else None
        self._rng = random.Random()

    def rps(self) -> int:
        return self._rng.randint(1, 3)

    def base_logits(self, obs_list: list[dict]) -> np.ndarray:
        """The base policy batched forward (BCPolicy.logits takes a stacked batch dict)."""
        from ygo_zero.search.turn1 import _batch

        return np.asarray(self.policy.logits(_batch(obs_list)))

    def start_duel(
        self,
        engine: Engine,
        config: DuelConfig,
        bot_seat: int,
        *,
        game_seed: int,
        ws_seed: int,
        human_fn: Callable[[dict], int] | None = None,
        prefix: Sequence[int] = (),
        prefix_decisions: Sequence[dict] | None = None,
    ) -> ProductDriver:
        return ProductDriver(
            engine,
            config,
            bot_seat=bot_seat,
            base_logits=self.base_logits,
            value=self.value,
            search_kwargs=self.search_kwargs,
            game_seed=game_seed,
            ws_seed=ws_seed,
            fast=self.fast,
            replay_threads=self.replay_threads,
            human_fn=human_fn,
            prefix=prefix,
            prefix_decisions=prefix_decisions,
        ).start()

    def record_info(self, seed: int) -> dict:
        """Product settings for the duel record, so a game can be re-run."""
        return {
            "ckpt": str(self.ckpt),
            "ckpt_sha256": self.ckpt_sha,
            "value": str(self.value_ckpt),
            "value_sha256": self.value_sha,
            "emb": str(self.emb),
            "device": self.device,
            "game_seed": int(seed),
            "ws_seed": int(seed),
            "fast": self.fast,
            "replay_threads": self.replay_threads,
            "search": product_search_kwargs(fast=self.fast) | dict(self.search_kwargs),
        }


# --- the driver ---


class ProductDriver:
    """One duel's product search (S = `bot_seat`, R = the human seat), in a worker thread.

    `base_logits(obs_list) -> [n, A]` is the base policy batched forward (BCPolicy.logits
    in production, a tiny net in tests); `value` is a learned value `values(obs_batch)`
    or None. `search_kwargs` overrides product defaults (k/samples/horizon/conf_skip/...);
    `fast=True` plays raw P (conf-skip 0, no value needed). `human_fn(obs) -> int` answers
    R's decisions in-process (tests); production feeds `feed_human` from the socket loop.

    The host consumes bot choices in order via `next_bot()` (blocking) and checks lockstep
    via `decision(k)`; `feed_human` enqueues C-space (product) indices. `join()` waits for
    the game; `res`/`error` hold the outcome. `close()` frees the replay pool.
    """

    def __init__(
        self,
        engine: Engine,
        config: DuelConfig,
        *,
        bot_seat: int,
        base_logits: Callable[[list[dict]], np.ndarray],
        value=None,
        search_kwargs: dict | None = None,
        game_seed: int = 0,
        ws_seed: int = 0,
        fast: bool = False,
        replay_threads: int = 1,
        human_fn: Callable[[dict], int] | None = None,
        prefix: Sequence[int] = (),
        prefix_decisions: Sequence[dict] | None = None,
    ) -> None:
        self.engine, self.config, self.bot_seat = engine, config, bot_seat
        self.base_logits = base_logits
        self.value = None if fast else value
        self.human_fn = human_fn
        self.fast = fast
        # What-if prefix (product-space outer choices): replayed verbatim by
        # search_game_gen before the first live decision, never re-searched.
        self._prefix = tuple(int(c) for c in (prefix or ()))
        if prefix_decisions is not None:
            if len(prefix_decisions) != len(self._prefix):
                raise ValueError(
                    f"prefix_decisions has {len(prefix_decisions)} entries "
                    f"for {len(self._prefix)} prefix choices"
                )
            self._prefix_decisions = [
                {"player": int(d["player"]), "n_options": int(d["n_options"])} for d in prefix_decisions
            ]
        elif self._prefix:
            self._prefix_decisions = _replay_prefix_decisions(engine, config, self._prefix)
        else:
            self._prefix_decisions = []
        kw = dict(product_search_kwargs(fast=fast))
        kw.update(search_kwargs or {})
        # Split: search construction takes k/horizon/samples/rollout_temp; the game takes the rest.
        search_kw = {k: kw.pop(k) for k in ("k", "horizon", "samples", "rollout_temp")}
        self.gen_kw = {"conf_skip": kw.pop("conf_skip"), "own_frac": kw.pop("own_frac"),
                       "own_turn1": kw.pop("own_turn1"), "honest": kw.pop("honest"),
                       "det_tries": kw.pop("det_tries"), "det_fallback": kw.pop("det_fallback"),
                       "margin": kw.pop("margin"), "max_search_turn": kw.pop("max_search_turn"),
                       "max_searches": kw.pop("max_searches")}
        if kw:
            raise ValueError(f"ProductDriver: unknown search kwargs {sorted(kw)}")
        self.game_seed, self.ws_seed = game_seed, ws_seed
        self.route: dict[int, tuple[int, int] | None] = {}
        self.pool = make_pool(replay_threads)
        self.search = build_search(engine, seed=ws_seed, value=self.value,
                                   route=self.route, pool=self.pool, **search_kw)
        self.gen_kwargs = dict(self.gen_kw)
        self.human_q: queue.Queue[int] = queue.Queue()
        self.cond = threading.Condition()
        self.played: list[tuple[int, int]] = []  # (choice, actor) in decision order (mirror)
        # What-if prefix decisions are known before the thread starts (replayed,
        # never re-searched); live outer decisions append in _forward.
        self.decision_log: list[dict] = [dict(d) for d in self._prefix_decisions]
        self.lat = {"t0": 0.0, "outer": None}  # pending outer decision start
        self._rollouts_used = 0  # rollout batches yielded since the pending outer decision
        self._rollouts_calls = 0  # rollouts_gen calls since the pending outer decision
        self.res = None
        self.error: BaseException | None = None
        self.finished = threading.Event()
        self._bot_delivered = 0  # next_bot pointer into bot-actor entries of played
        self._thread: threading.Thread | None = None
        self._inflight: dict = {}
        self._wrap_rollouts()

    # -- setup --

    def _wrap_rollouts(self) -> None:
        """Count rollouts_gen calls/batches per outer decision (searched vs raw/skip reporting)."""
        search, drv = self.search, self
        orig = search.rollouts_gen

        def counting(*a, **k):
            drv._rollouts_calls += 1
            gen = orig(*a, **k)
            try:
                req = next(gen)
                while True:
                    drv._rollouts_used += 1
                    fb = yield req
                    req = gen.send(fb)
            except StopIteration as stop:
                return stop.value

        search.rollouts_gen = counting  # type: ignore[method-assign]

    def start(self) -> ProductDriver:
        self._thread = threading.Thread(target=self._run, daemon=True, name="product-driver")
        self._thread.start()
        return self

    # -- host -> driver --

    def feed_human(self, indices: Sequence[int]) -> None:
        """Enqueue the human's C-space (product) indices, in decision order."""
        for i in indices:
            self.human_q.put(int(i))

    # -- driver -> host --

    def next_bot(self, timeout: float | None = None) -> int:
        """The bot's next choice (blocking); raises DriverError on divergence/death/end mismatch."""
        with self.cond:
            while True:
                if self.error is not None:
                    raise DriverError(str(self.error))
                # played entries are in order; return the next undelivered bot entry, if any:
                for n_skip, (c, a) in enumerate(self.played[self._bot_delivered :]):
                    if a == self.bot_seat:
                        self._bot_delivered += n_skip + 1
                        return int(c)
                if self.finished.is_set():
                    raise DriverError(
                        f"product game ended with no further bot decision "
                        f"({len(self.played)} decisions played)")
                self.cond.wait(timeout=timeout if timeout is not None else 60.0)
                if timeout is not None:
                    raise DriverError("timed out waiting for the product's next decision")

    def decision(self, k: int, timeout: float = 300.0) -> dict:
        """Outer decision k's (player, n_options); waits until the driver reaches it."""
        t0 = time.monotonic()
        while True:
            with self.cond:
                if k < len(self.decision_log):
                    return self.decision_log[k]
                if self.error is not None:
                    raise DriverError(str(self.error))
                if self.finished.is_set():
                    raise DriverError(f"product game ended before outer decision {k}")
            if time.monotonic() - t0 > timeout:
                raise DriverError(f"timed out waiting for product outer decision {k}")
            time.sleep(0.01)

    def n_played(self) -> int:
        with self.cond:
            return len(self.played)

    def join(self, timeout: float | None = None) -> None:
        self.finished.wait(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.error is not None:
            raise DriverError(str(self.error))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    # -- the worker --

    def _next_human(self, obs: dict) -> int:
        if self.human_fn is not None:
            h = int(self.human_fn(dict(obs)))
            if not 0 <= h < int(obs["action_mask"].sum()):
                raise DriverError(f"human_fn returned illegal index {h}")
            return h
        try:
            return int(self.human_q.get(timeout=600))
        except queue.Empty:
            raise DriverError("timed out waiting for the human's choice") from None

    def _forward(self, obs_list: list[dict]) -> np.ndarray:
        missing = object()
        base_at, base_obs, r_at = [], [], []
        outer = None  # ("s"|"r", obs): a lone outer decision (rollout batches can also hold one obs)
        for i, o in enumerate(obs_list):
            tag = self.route.pop(id(o), missing)
            if tag is missing:
                raise DriverError("untagged obs in the batch: every request passes through tag_game")
            if tag is None:
                base_at.append(i)
                base_obs.append(o)
            else:
                if len(obs_list) != 1:
                    raise DriverError("mixed rollout/outer batch (search_game_gen never yields one)")
                _, s_seat = tag
                if seat_of(o) == s_seat:
                    base_at.append(i)
                    base_obs.append(o)
                    outer = ("s", o)
                else:
                    r_at.append(i)
                    outer = ("r", o)
        a_width = int(obs_list[0]["action_mask"].shape[0])
        lg = np.zeros((len(obs_list), a_width), np.float32)
        if base_obs:
            b = np.asarray(self.base_logits(base_obs))
            if b.shape[1] != a_width:
                raise DriverError(f"policy width {b.shape[1]} != action options {a_width}")
            for k, i in enumerate(base_at):
                lg[i] = b[k]
        if outer is not None:
            kind, o = outer
            with self.cond:
                self.decision_log.append({"player": seat_of(o),
                                          "n_options": int(o["action_mask"].sum())})
                self.lat["t0"] = time.perf_counter()
                self.lat["outer"] = kind
                if kind == "s":
                    # guard-masked logits like search_game_gen's conf_skip check (play.py)
                    z = np.array(lg[base_at[0]], dtype=np.float64)
                    z[~np.asarray(o["action_mask"], dtype=bool)] = -np.inf
                    self.lat["z"] = z
                else:
                    self.lat["z"] = None
                self._rollouts_used = 0
                self._rollouts_calls = 0
        if r_at:
            for i in r_at:
                lg[i, self._next_human(obs_list[i])] = 1.0
        return lg

    def _note_resolved(self) -> None:
        """After each send: mirror newly appended choices + print the bot decision's latency."""
        cur = self._inflight.get(self.game_seed, [])
        with self.cond:
            while len(self.played) < len(cur):
                k = len(self.played)
                c = int(cur[k])
                a = int(self.decision_log[k]["player"]) if k < len(self.decision_log) else -1
                self.played.append((c, a))
                if a == self.bot_seat and self.lat["outer"] == "s":
                    wall = time.perf_counter() - self.lat["t0"]
                    searched = self._rollouts_used > 0
                    if searched:
                        how = "searched"
                    elif (self.lat["z"] is not None
                            and softmax_conf(self.lat["z"]) >= self.gen_kwargs["conf_skip"]):
                        how = "conf-skip"
                    else:
                        how = "skip"  # unconfident but unsearched: det-skip or ineligible
                    print(f"bot decision k={k} {wall:.2f}s {how} "
                          f"(rollout batches {self._rollouts_used})", flush=True)
                    with_bot = dict(self.decision_log[k])
                    with_bot.update(searched=searched, wall=round(wall, 3))
                    self.decision_log[k] = with_bot
                    self.lat["outer"] = None
                self.cond.notify_all()

    def _run(self) -> None:
        try:
            gen = tag_game(0, self.bot_seat, search_game_gen(
                self.engine, self.config, self.search, self.bot_seat, argmax_select,
                game_seed=self.game_seed, guard_mode="per_seat",
                unguarded_seat=1 - self.bot_seat, latency=True,
                inflight=self._inflight, prefix=self._prefix, **self.gen_kwargs,
            ), self.route)  # fmt: skip
            req = next(gen)
            while True:
                fb = self._forward(req)
                req = gen.send(fb)
                self._note_resolved()
        except StopIteration as stop:
            self.res = stop.value
            self._note_resolved()
            with self.cond:
                self.finished.set()
                self.cond.notify_all()
        except BaseException as e:  # noqa: BLE001 -- surfaced to the host as ALERT + duel stop
            import traceback

            traceback.print_exc()
            self.error = e
            with self.cond:
                self.finished.set()
                self.cond.notify_all()
