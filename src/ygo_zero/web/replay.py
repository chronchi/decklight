"""Spectator replay frames for video rendering (game_video.py).

`replay_frames` replays a recorded game's choices through `PyEnv` (mirroring
`PyEnv.replay` exactly) and emits one client-shaped state frame per choice plus
a final board frame — the same dicts the live server's `state_message` sends,
with `prompt` always None and a `caption` (who decided + the chosen option's
label) in its place. Seat 0 is shown at the bottom; both hands are visible
(`build_view(..., reveal_all=True)`).
"""

from __future__ import annotations

from collections.abc import Sequence

from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.engine.env import PyEnv
from ygo_zero.web.view import build_events, build_prompt, build_view

# Narration-free log kinds in a video: the caption bar already says which
# card was chosen / which position was picked, so these add nothing.
_DROP_LOG_KINDS = {"select", "position"}


def _replay_log(env: PyEnv, since: int) -> list[dict]:
    return [e for e in build_events(env, 0, since=since)
            if (e.get("kind") or "").lower() not in _DROP_LOG_KINDS]


def _fallback_label(env: PyEnv, seat: int, index: int) -> str:
    """A non-empty label when there is no multi-option prompt (singletons)."""
    try:
        opt = env.opts[index]
    except (IndexError, AttributeError):
        return f"Option {index}"
    try:
        from ygo_zero.web.view import _effect, _label, _ref_card, _where

        card = _ref_card(env, seat, opt.card)
        where = _where(opt.card, seat)
        return _label(opt, card, where, _effect(opt)) or opt.kind.name
    except Exception:  # noqa: BLE001 -- never break a replay on a label
        return getattr(opt.kind, "name", str(opt.kind))


def replay_frames(
    engine: Engine,
    cfg: DuelConfig,
    choices: Sequence[int],
    names: tuple[str, str] = ("Player 1", "Player 2"),
    *,
    expect_winner: int | None = None,
    auto_single: bool = True,
    surrendered: bool = False,
) -> list[dict]:
    """Replay `choices` and return `len(choices)+1` spectator frames.

    A duel won in-engine must end `env.done` with the recorded winner;
    a surrendered record (`surrendered=True`) ends the choices before the
    in-engine duel is over, so the final frame takes the recorded winner
    with reason "surrender" (the web server records surrenders the same way).

    Raises `RuntimeError` when the replay does not reproduce the record.
    """
    n0, n1 = names
    env = PyEnv(engine, cfg, auto_single=auto_single)
    env.reset()
    frames: list[dict] = []
    prev_len = 0
    for k, idx in enumerate(choices):
        if env.done:
            raise RuntimeError(f"replay ended early at step {k} of {len(choices)}")
        seat = env.player
        prompt = build_prompt(env, seat, k)
        label = ""
        if prompt is not None:
            for o in prompt["options"]:
                if o["index"] == int(idx):
                    label = o["label"]
                    break
        if not label:
            label = _fallback_label(env, seat, int(idx))
        view = build_view(env, 0, reveal_all=True)
        log = _replay_log(env, prev_len)
        frame: dict = {
            "t": "state",
            "view": view,
            "prompt": None,
            "log": log,
            "thinking": None,
            "result": None,
            # Frame 0 is the title card: no caption bar over it. The first
            # decision's caption is still listed in the .frames.txt index.
            "caption": None if k == 0 else {"actor": names[seat], "text": label},
            "names": [n0, n1],
        }
        if k == 0:
            frame["title"] = f"{n0} vs {n1} · seed {cfg.seed} · ygo-zero"
        frames.append(frame)
        prev_len = len(env.history)
        env.step(int(idx))
    if env.done:
        if expect_winner is not None and env.winner != expect_winner:
            raise RuntimeError(
                f"replay winner {env.winner} != record winner {expect_winner}"
            )
        winner, reason = env.winner, "win"
    elif surrendered and expect_winner is not None:
        winner, reason = expect_winner, "surrender"
    else:
        raise RuntimeError(
            f"replay of {len(choices)} choices did not finish the duel "
            f"(done={env.done}, winner={env.winner})"
        )
    view = build_view(env, 0, reveal_all=True)
    log = _replay_log(env, prev_len)
    frames.append(
        {
            "t": "state",
            "view": view,
            "prompt": None,
            "log": log,
            "thinking": None,
            "result": {"winner": names[winner], "reason": reason},
            "caption": {"actor": names[winner], "text": "Wins the duel"},
            "names": [n0, n1],
        }
    )
    return frames
