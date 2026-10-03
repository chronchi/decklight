"""Remote verification on the warm vast.ai session box.

The planner LLM calls always stay on the laptop (the OpenCode key must never leave it); only engine
verification batches go to the box. `verdict_to_json` / `verdict_from_json` carry Verdicts (including
their DuelConfig puzzle starts) over the wire; `remote_verify` ships items there and reads Verdicts back.

The `_push` / `_run` / `_pull` indirection exists so tests can fake the box without touching ssh.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import uuid
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.demos.verify import Verdict
from ygo_zero.engine.duel import DuelConfig, Placement
from ygo_zero.infra import vast

# (line_json, main, extra, side, name, seed, budget): the same tuples plan_lines._verify_job takes.
VerifyItem = tuple[dict, tuple, tuple, tuple, str, int, int]


def _deck_to_json(deck) -> dict:
    return {
        "main": list(deck.main),
        "extra": list(deck.extra),
        "side": list(getattr(deck, "side", ())),
        "name": getattr(deck, "name", ""),
    }


def _deck_from_json(d: dict) -> Deck:
    return Deck(
        tuple(d.get("main", ())), tuple(d.get("extra", ())), tuple(d.get("side", ())), d.get("name", "")
    )


def _config_to_json(config: DuelConfig | None) -> dict | None:
    if config is None:
        return None
    return {
        "deck0": _deck_to_json(config.deck0),
        "deck1": _deck_to_json(config.deck1),
        "seed": config.seed,
        "flags": config.flags,
        "lp": config.lp,
        "hand": config.hand,
        "draw": config.draw,
        "setup": [list(p) for p in config.setup],
    }


def _config_from_json(d: dict | None) -> DuelConfig | None:
    if d is None:
        return None
    return DuelConfig(
        _deck_from_json(d["deck0"]),
        _deck_from_json(d["deck1"]),
        seed=d["seed"],
        flags=d["flags"],
        lp=d["lp"],
        hand=d["hand"],
        draw=d["draw"],
        setup=tuple(Placement(*p) for p in d.get("setup", [])),
    )


def verdict_to_json(v: Verdict) -> dict:
    return {
        "line_id": v.line_id,
        "ok": v.ok,
        "reason": v.reason,
        "choices": list(v.choices),
        "labels": list(v.labels),
        "seed": v.seed,
        "nodes": v.nodes,
        "end_field_codes": list(v.end_field_codes),
        "detail": v.detail,
        "skipped_steps": None if v.skipped_steps is None else list(v.skipped_steps),
        "deck_source": v.deck_source,
        "executed_steps": v.executed_steps,
        "inferred_frac": v.inferred_frac,
        "hand_inferred": v.hand_inferred,
        "config": _config_to_json(v.config),
    }


def verdict_from_json(d: dict) -> Verdict:
    return Verdict(
        line_id=d["line_id"],
        ok=d["ok"],
        reason=d["reason"],
        choices=list(d["choices"]),
        labels=list(d["labels"]),
        seed=d["seed"],
        nodes=d["nodes"],
        end_field_codes=list(d["end_field_codes"]),
        detail=d["detail"],
        skipped_steps=None if d.get("skipped_steps") is None else list(d["skipped_steps"]),
        deck_source=d.get("deck_source", ""),
        executed_steps=d.get("executed_steps", 0),
        inferred_frac=d.get("inferred_frac", 0.0),
        hand_inferred=d.get("hand_inferred", False),
        config=_config_from_json(d.get("config")),
    )


def _item_to_json(item: VerifyItem) -> dict:
    line_json, main, extra, side, name, seed, budget = item
    return {
        "line": line_json,
        "main": list(main),
        "extra": list(extra),
        "side": list(side),
        "name": name,
        "seed": seed,
        "budget": budget,
    }


def _item_from_json(d: dict) -> VerifyItem:
    return (
        d["line"],
        tuple(d["main"]),
        tuple(d["extra"]),
        tuple(d["side"]),
        d["name"],
        d["seed"],
        d["budget"],
    )


def write_items(path: str | Path, items: list[VerifyItem]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(_item_to_json(it)) + "\n" for it in items)


def read_items(path: str | Path) -> list[VerifyItem]:
    out = []
    for raw in Path(path).read_text().splitlines():
        if raw.strip():
            out.append(_item_from_json(json.loads(raw)))
    return out


def require_session() -> dict:
    """The warm-box record, or a RuntimeError telling the user to start a session (plan_lines --remote)."""
    try:
        return vast.require_session()
    except vast.VastError as e:
        raise RuntimeError(str(e)) from e


def _push(local_path: str, remote_path: str) -> None:
    vast.session_push([local_path], str(Path(remote_path).parent))


def _run(cmd: str, timeout_s: float | None) -> tuple[int, str]:
    return vast.session_run(cmd, timeout_s)


def _pull(remote_path: str, local_path: str) -> None:
    vast.session_pull(remote_path, local_path)


def remote_verify(items: list[VerifyItem], timeout_per_item: float, jobs: int | None = None) -> list[Verdict]:
    """Verify items on the session box; Verdicts come back in input order.

    Items go to a uniquely-named jsonl (several planners may run at once), the box runs
    `scripts/verify_items.py` on them, and the verdict jsonl is pulled back. The box never sees any LLM key.
    """
    if not items:
        return []
    require_session()
    tag = uuid.uuid4().hex[:8]
    tmp = Path(tempfile.gettempdir())
    local_in = tmp / f"verify_in_{tag}.jsonl"
    local_out = tmp / f"verify_out_{tag}.jsonl"
    remote_in = f"/tmp/verify_in_{tag}.jsonl"
    remote_out = f"/tmp/verify_out_{tag}.jsonl"
    write_items(local_in, items)
    try:
        _push(str(local_in), remote_in)
        cmd = f"uv run --no-sync python scripts/verify_items.py --in {remote_in} --out {remote_out} --timeout {timeout_per_item}"  # fmt: skip
        if jobs is not None:
            cmd += f" --jobs {int(jobs)}"
        rc, out = _run(cmd, timeout_s=len(items) * timeout_per_item + 300)
        if rc != 0:
            raise RuntimeError(f"remote verify failed (exit {rc}): {out[-2000:]}")
        _pull(remote_out, str(local_out))
        verdicts = [verdict_from_json(json.loads(raw)) for raw in local_out.read_text().splitlines() if raw.strip()]  # fmt: skip
        if len(verdicts) != len(items):
            raise RuntimeError(f"remote verify returned {len(verdicts)} verdicts for {len(items)} items")
        return verdicts
    finally:
        local_in.unlink(missing_ok=True)
        local_out.unlink(missing_ok=True)
        with contextlib.suppress(Exception):  # best-effort remote cleanup
            _run(f"rm -f {remote_in} {remote_out}", timeout_s=60)
