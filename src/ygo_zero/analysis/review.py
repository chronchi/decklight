"""Post-game review and missed-response alerts (web-ux-plan step 3).

For each human decision with >= 2 options, compare the human's choice with the
agent's choice by one-step lookahead, using only the human's own observation
for every judgement. The bot side of a counterfactual plays greedy by its own
policy on its own obs (the opponent's own view: legitimate, never shown).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ygo_zero.engine import messages as M
from ygo_zero.engine.actions import Kind

MARK_GOOD = "Good"
MARK_INACC = "Inaccuracy"
MARK_MIST = "Mistake"
MARK_BLUND = "Blunder"
MARK_MISSED = "Missed response"

MAX_CANDS = 5
MAX_LOOKAHEAD = 32


def mark_for_delta(delta_pts: float) -> str:
    if delta_pts >= -3:
        return MARK_GOOD
    if delta_pts >= -7:
        return MARK_INACC
    if delta_pts >= -15:
        return MARK_MIST
    return MARK_BLUND


def _batch(obs_list: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([np.ascontiguousarray(o[k]) for o in obs_list]) for k in obs_list[0]}


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.array(logits[: len(mask)], dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    z[~m] = -np.inf
    z -= np.nanmax(z)
    e = np.exp(z)
    e[~m] = 0.0
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        out = np.zeros_like(e)
        out[m] = 1.0 / max(1, int(m.sum()))
        return out
    return e / s


def human_obs_dict(env, human_seat: int) -> dict:
    """Obs dict from the human's view at the env's current position.

    Mirrors PyEnv._observe but with viewer=human_seat even when it is not the
    human's turn (the estimate's value must never see the bot's hand).
    """
    from ygo_zero.engine.obs import Board, actions_array, history_array

    duel = env.duel
    board_h = Board(duel, human_seat)
    dec = env.dec
    if dec is None or env.done:
        import numpy as _np

        from ygo_zero.engine import obs as O

        cards, card_mask = board_h.cards(())
        card_known, known_code = board_h.knowledge(duel.memory)
        chain, chain_mask = board_h.chain_array()
        actions = _np.zeros((O.MAX_OPTIONS, O.ACTION_COLS), _np.int32)
        action_mask = _np.zeros(O.MAX_OPTIONS, bool)
        history, history_mask = history_array(env.history, human_seat, duel.turn)
        hl, hlm = history_array(env.history, human_seat, duel.turn, n=O.HISTORY_MAX, salient=True)
        glob = board_h.global_array(duel, 0)
        return {
            "cards": cards,
            "card_mask": card_mask,
            "card_known": card_known,
            "known_code": known_code,
            "global": glob,
            "chain": chain,
            "chain_mask": chain_mask,
            "actions": actions,
            "action_mask": action_mask,
            "history": history,
            "history_mask": history_mask,
            "hist_long": hl,
            "hist_long_mask": hlm,
        }
    opts = env.opts
    cards, card_mask = board_h.cards(dec.selected)
    card_known, known_code = board_h.knowledge(duel.memory)
    chain, chain_mask = board_h.chain_array()
    actions, action_mask = actions_array(board_h, dec, opts)
    from ygo_zero.engine import obs as O

    history, history_mask = history_array(env.history, human_seat, duel.turn)
    hl, hlm = history_array(env.history, human_seat, duel.turn, n=O.HISTORY_MAX, salient=True)
    glob = board_h.global_array(duel, dec.msg.type)
    return {
        "cards": cards,
        "card_mask": card_mask,
        "card_known": card_known,
        "known_code": known_code,
        "global": glob,
        "chain": chain,
        "chain_mask": chain_mask,
        "actions": actions,
        "action_mask": action_mask,
        "history": history,
        "history_mask": history_mask,
        "hist_long": hl,
        "hist_long_mask": hlm,
    }


def load_record(record_path: str | Path):
    """Load a web game record dir or game.jsonl: (rec, cfg, human_seat, record_dir)."""
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.duel import DuelConfig

    p = Path(record_path)
    d = p if p.is_dir() else p.parent
    rec = json.loads((d / "game.jsonl").read_text())
    human, bot = load_ydk(d / "human.ydk"), load_ydk(d / "bot.ydk")
    hs = int(rec["human_seat"])
    cfg = (
        DuelConfig(human, bot, seed=int(rec["seed"]))
        if hs == 0
        else DuelConfig(bot, human, seed=int(rec["seed"]))
    )
    return rec, cfg, hs, d


def collect_human_decisions(engine, cfg, rec) -> list[dict]:
    """Replay the record; one entry per human multi-option decision."""
    from ygo_zero.engine.env import PyEnv
    from ygo_zero.web.view import PHASE_NAMES, _prompt_kind

    choices = [int(c) for c in rec["choices"]]
    hs = int(rec["human_seat"])
    auto = {int(x) for x in (rec.get("auto_passed") or [])}
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    out: list[dict] = []
    for i, ch in enumerate(choices):
        if env.done:
            break
        if env.player == hs and len(env.opts) > 1:
            dec = env.pending_decision()
            kind = _prompt_kind(dec.msg) if dec is not None else "card"
            out.append(
                {
                    "prefix": list(env.choices),
                    "choice": int(ch),
                    "n_opts": len(env.opts),
                    "opt_kinds": [o.kind for o in list(env.opts)],
                    "turn": int(env.duel.turn),
                    "phase": PHASE_NAMES.get(env.duel.phase, "main1"),
                    "kind": kind,
                    "hist_n": len(env.choices) + 1,
                    "auto": (len(env.choices) + 1) in auto,
                    "is_chain": dec is not None and isinstance(dec.msg, M.SelectChain),
                }
            )
        env.step(int(ch))
    return out


def _launcher_logits(launcher, obs_list: list[dict]) -> np.ndarray:
    if hasattr(launcher, "base_logits"):
        return np.asarray(launcher.base_logits(obs_list))
    if hasattr(launcher, "forward"):
        return np.asarray(launcher.forward(obs_list))
    raise AttributeError("launcher has neither base_logits nor forward")


def _policy_probs(launcher, obs: dict) -> np.ndarray:
    logits = np.asarray(_launcher_logits(launcher, [obs])).reshape(-1)
    return _masked_softmax(logits, obs["action_mask"])


def _argmax_masked(logits: np.ndarray, mask: np.ndarray) -> int:
    z = np.array(logits[: len(mask)], dtype=np.float64)
    z[~np.asarray(mask, dtype=bool)] = -np.inf
    return int(np.argmax(z))


def estimate_candidates(
    engine,
    cfg,
    prefix: list[int],
    cands: list[int],
    human_seat: int,
    launcher,
    calib,
    *,
    max_steps: int = MAX_LOOKAHEAD,
) -> list[tuple[float, int]]:
    """Win-chance estimate per candidate by one-step lookahead.

    Returns [(p, turn)] in candidate order. Plays forward greedily (both sides
    argmax on the actor's own obs) until the human's next decision, the end of
    the starting turn, or game end.
    """
    from ygo_zero.engine.env import PyEnv

    value_fn = getattr(launcher, "value", None)
    if value_fn is None:
        raise ValueError("review needs the full bot (no value net)")
    out: list[tuple[float, int]] = []
    finals: list[tuple[PyEnv, int, bool, int | None]] = []
    for c in cands:
        env = PyEnv(engine, cfg, auto_single=False)
        env.reset()
        for pc in prefix:
            if env.done:
                break
            env.step(int(pc))
        start_turn = int(env.duel.turn)
        if env.done:
            finals.append((env, start_turn, True, None))
            continue
        env.step(int(c))
        steps = 0
        while not env.done and steps < max_steps:
            if env.duel.turn != start_turn:
                break
            if env.player == human_seat and len(env.opts) > 1:
                break
            obs = env._observe()
            lg = np.asarray(_launcher_logits(launcher, [obs])).reshape(-1)
            env.step(_argmax_masked(lg, obs["action_mask"]))
            steps += 1
        finals.append((env, start_turn, env.done, None))
    # Batch the value forwards over all non-terminal finals.
    need = [(i, e) for i, (e, _, done, _) in enumerate(finals) if not done]
    vmap: dict[int, float] = {}
    if need:
        obs_list = [human_obs_dict(e, human_seat) for _, e in need]
        # Keep only keys the value net wants is handled by the net; feed all.
        buf = _batch(obs_list)
        vals = np.asarray(value_fn(buf)).ravel()
        for (i, e), v in zip(need, vals, strict=True):
            vmap[i] = float(np.clip(v, -1, 1))
    for i, (e, _s, done, _x) in enumerate(finals):
        turn = int(e.duel.turn)
        if done:
            p = 1.0 if e.duel.winner == human_seat else 0.0
        else:
            p = calib.p(vmap[i], turn)
        out.append((float(p), turn))
    return out


def review_game(engine, cfg, rec, launcher, calib, *, max_steps: int = MAX_LOOKAHEAD, progress=None) -> dict:
    """Full review of a web record. Returns the review JSON dict."""
    from ygo_zero.web.view import build_prompt

    t0 = time.monotonic()
    from ygo_zero.engine.env import PyEnv

    decisions = collect_human_decisions(engine, cfg, rec)
    hs = int(rec["human_seat"])
    entries: list[dict] = []
    fwd_ms = 0.0
    val_ms = 0.0
    for k, d in enumerate(decisions):
        # Rebuild the env at decision k for labels + the human obs.
        env = PyEnv(engine, cfg, auto_single=False)
        env.reset()
        for pc in d["prefix"]:
            env.step(int(pc))
        obs = env._observe()
        t1 = time.monotonic()
        probs = _policy_probs(launcher, obs)
        fwd_ms += (time.monotonic() - t1) * 1000.0
        legal = [int(i) for i in np.flatnonzero(obs["action_mask"])]
        order = sorted(legal, key=lambda j: (-float(probs[j]), int(j)))
        cands: list[int] = [int(d["choice"])]
        for j in order:
            if int(j) not in cands:
                cands.append(int(j))
            if len(cands) >= 3:
                break
        if d["is_chain"]:
            for j, kd in enumerate(d["opt_kinds"]):
                if kd == Kind.ACTIVATE and int(j) not in cands and int(j) != int(d["choice"]):
                    cands.append(int(j))
                if len(cands) >= MAX_CANDS:
                    break
        cands = cands[:MAX_CANDS]
        t1 = time.monotonic()
        est = estimate_candidates(engine, cfg, d["prefix"], cands, hs, launcher, calib, max_steps=max_steps)
        val_ms += (time.monotonic() - t1) * 1000.0
        # Ties keep the policy order: among equal estimates pick the lower policy rank.
        best_p = max(p for p, _ in est)
        tied = [j for j, (p, _) in enumerate(est) if p == best_p]
        prank = {c: r for r, c in enumerate(order)}
        best_j = min(tied, key=lambda j: prank.get(cands[j], 999))
        agent_idx = int(cands[best_j])
        human_p = float(est[cands.index(int(d["choice"]))][0])
        agent_p = float(est[best_j][0])
        delta_pts = round((human_p - agent_p) * 100, 1)
        # Missed response: passed (incl. auto-passed) while an ACTIVATE existed,
        # the policy would activate (>= 0.5 mass on activating) and activating rates higher.
        had_act = any(kd == Kind.ACTIVATE for kd in d["opt_kinds"])
        human_kind = d["opt_kinds"][int(d["choice"])] if 0 <= int(d["choice"]) < len(d["opt_kinds"]) else None
        human_pass = human_kind in (Kind.PASS, Kind.NO) or (d["is_chain"] and human_kind == Kind.PASS)
        p_activate = float(sum(probs[j] for j, kd in enumerate(d["opt_kinds"]) if kd == Kind.ACTIVATE))
        act_cands = [
            j
            for j, c in enumerate(cands)
            if 0 <= c < len(d["opt_kinds"]) and d["opt_kinds"][c] == Kind.ACTIVATE
        ]
        act_better = any(est[j][0] > human_p for j in act_cands)
        missed = bool(d["is_chain"] and had_act and human_pass and p_activate >= 0.5 and act_better)
        mark = MARK_MISSED if missed else mark_for_delta(delta_pts)
        prompt = build_prompt(env, hs, 1)
        opt_by_idx = {o["index"]: o for o in (prompt["options"] if prompt else [])}
        your_o = opt_by_idx.get(int(d["choice"]), {})
        agent_o = opt_by_idx.get(agent_idx, {})
        your_card = your_o.get("card") or {}
        agent_card = agent_o.get("card") or {}
        entries.append(
            {
                "k": k,
                "turn": d["turn"],
                "phase": d["phase"],
                "kind": d["kind"],
                "mark": mark,
                "delta_pts": delta_pts,
                "your": {
                    "label": your_o.get("label", f"option {d['choice']}"),
                    "card": {"code": your_card.get("code", 0), "name": your_card.get("name")},
                },
                "agent": {
                    "label": agent_o.get("label", f"option {agent_idx}"),
                    "card": {"code": agent_card.get("code", 0), "name": agent_card.get("name")},
                },
                "policy_p_yours": round(float(probs[int(d["choice"])]), 4),
                "policy_p_agent": round(float(probs[agent_idx]), 4),
                "missed_response": missed,
                "auto_passed": bool(d["auto"]),
                "note": (
                    "You passed; the agent would activate "
                    + f"{agent_o.get('label', '')} ({delta_pts:+.1f} pts)"
                    if missed
                    else ""
                ),
                "est_yours": round(human_p, 4),
                "est_agent": round(agent_p, 4),
            }
        )
        if progress is not None:
            progress(k + 1, len(decisions))
    counts = {MARK_GOOD: 0, MARK_INACC: 0, MARK_MIST: 0, MARK_BLUND: 0, MARK_MISSED: 0}
    for e in entries:
        counts[e["mark"]] = counts.get(e["mark"], 0) + 1
    acc = (sum(max(0.0, 100.0 + e["delta_pts"]) for e in entries) / len(entries)) if entries else 100.0
    dt = time.monotonic() - t0
    return {
        "decisions": entries,
        "counts": counts,
        "accuracy": round(acc, 1),
        "missed": sum(1 for e in entries if e["missed_response"]),
        "timing": {
            "seconds": round(dt, 2),
            "fwd_ms": round(fwd_ms, 1),
            "value_ms": round(val_ms, 1),
            "decisions": len(entries),
        },
    }
