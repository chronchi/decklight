"""LLM-planned turns from real mid-game states (retrain plan v3, stage M).

Samples own-turn Main Phase 1 starts from recorded games (pool train decks
only), asks the LLM to plan the rest of that turn, verifies the plan in the
engine from that exact state (strict verifier), and writes BC records.

Usage:
    uv run python scripts/plan_turns.py --games 'data/demos/sw-v/p0-a*.jsonl' --per-bucket 1 --dry-run

Split mode (2026-10-12: verification is ~4 min of one core per state, so the laptop's 4 cores took ~7 h per 10 decks):
the laptop makes the LLM calls (`--llm-only`, fills data/cache/llm), a vast box verifies from that cache with no key
(`--cache-only --proc i --nproc n`) and writes the repair prompts it could not answer to `--pending-out`; the laptop
answers them (`--llm-only --prompts <pending>`) and a second vast round finishes the repairs.
"""

from __future__ import annotations

import os

os.environ["YGO_STRICT_VERIFY"] = "1"

import argparse
import glob
import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk  # noqa: E402
from decklight.demos import llm as _llm  # noqa: E402
from decklight.demos import verify  # noqa: E402
from decklight.demos.dataset import record_config  # noqa: E402
from decklight.demos.explain import explain_failure  # noqa: E402
from decklight.demos.extract import _fill_codes, _parse_json  # noqa: E402
from decklight.demos.game import TURN_SYSTEM, describe_state, legal_actions, turn_prompt  # noqa: E402
from decklight.demos.names import CardNames  # noqa: E402
from decklight.demos.plan import card_sheet, compact_line_json, repair_prompt  # noqa: E402
from decklight.demos.schema import ComboLine  # noqa: E402
from decklight.demos.verify import plan_from_state  # noqa: E402
from decklight.engine import constants as C  # noqa: E402
from decklight.engine.cpp_env import CppEnv  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402
from decklight.model.bc_data import eval_only_decks  # noqa: E402

if not verify.STRICT:
    # Already imported without the env var (e.g. under pytest sharing one
    # process): strict only narrows the chain-window search, so force it.
    verify.STRICT = True
assert verify.STRICT is True

BUCKETS = ("second", "mid", "end")
PROMPT_VERSION = "plan_turns-v1"
MODEL_DEFAULT = "opencode/muse-spark-1.3-contributor-free"
PHASE_MAIN1 = 0x4
PRINCIPLES = ROOT / "docs" / "play_principles.md"  # plan v3 stage M: the prompt carries the general principles


def bucket_of(turn: int, went_first: bool, my_lp: int, opp_lp: int, my_deck: int) -> str | None:
    """Bucket for an own-turn Main Phase 1 state (first match wins: second, end, mid)."""
    if turn == 2 and not went_first:
        return "second"
    if turn >= 3 and (my_lp <= 3000 or opp_lp <= 3000 or my_deck <= 10):
        return "end"
    if 3 <= turn <= 6:
        return "mid"
    return None


def is_train_deck(name: str, pool_dir: Path, hold: set, new: set) -> bool:
    return name not in hold and name not in new and (pool_dir / f"{name}.ydk").exists()


def game_skipped(decks: list, hold: set, new: set) -> bool:
    return any(d in hold or d in new for d in decks)


def replay_actors_turns(engine: Engine, config, auto_single: bool,
                        choices: list[int]) -> tuple[list[int], list[int]]:
    """(acting player, duel turn) before each choice, by replay."""
    env = CppEnv(engine, config, auto_single)
    obs = env.reset()
    actors, turns = [], []
    for c in choices:
        if obs is None:
            raise ValueError("choices end the duel before the last decision")
        actors.append(env.player)
        turns.append(int(obs["global"][2]))
        obs = env.step(int(c))
    return actors, turns


def config_dict_of(config) -> dict:
    """The stored-config dict shape game records use (so iter_samples replays it)."""
    return {
        "deck0_main": list(config.deck0.main),
        "deck0_extra": list(config.deck0.extra),
        "deck1_main": list(config.deck1.main),
        "deck1_extra": list(config.deck1.extra),
        "seed": config.seed,
        "hand": config.hand,
        "draw": config.draw,
        "setup": [[p.team, p.code, p.location, p.sequence] for p in config.setup],
        "shuffle": config.shuffle,
    }


def sample_states(engine: Engine, records: list[dict], seed: int, per_bucket: int,
                  buckets: tuple[str, ...] = BUCKETS,
                  decks_allow: set[str] | None = None,
                  pool_dir: Path = ROOT / "decks" / "pool",
                  eval_sets: tuple[set, set] | None = None) -> list[dict]:
    """Own-turn Main Phase 1 idle starts, at most `per_bucket` per (deck, bucket).

    At most one state per game per bucket; games are spread by shuffling with
    `seed`. Each state holds the source record's stored config dict, the
    decision index `k`, the prefix `choices[:k]`, and the acting deck name.
    """
    hold, new = eval_sets if eval_sets is not None else eval_only_decks()
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    counts: Counter = Counter()
    out: list[dict] = []
    for ri in order:
        rec = records[ri]
        decks = rec.get("decks") or []
        if len(decks) < 2 or game_skipped(decks, hold, new):
            continue
        try:
            config = record_config(rec)
        except (KeyError, TypeError, ValueError):
            continue
        stored = rec.get("config")
        if not isinstance(stored, dict):
            continue
        choices = [int(c) for c in rec.get("choices", [])]
        if not choices:
            continue
        auto_single = rec.get("auto_single", True)
        game_id = str(rec.get("game_id") or rec.get("line_id") or f"rec{ri}")
        try:
            env = CppEnv(engine, config, auto_single)
            obs = env.reset()
        except Exception:  # noqa: BLE001 - one bad record never kills sampling
            continue
        seen: set[tuple[int, int]] = set()  # (turn, player) already examined
        took: set[str] = set()  # buckets already taken from this game
        try:
            for k in range(len(choices)):
                if obs is None:
                    break
                g = obs["global"]
                turn, phase = int(g[2]), int(g[3])
                my_turn, went_first = bool(int(g[5])), bool(int(g[4]))
                msg, chain = int(g[21]), int(g[20])
                player = env.player
                if (my_turn and phase == PHASE_MAIN1 and msg == C.MSG_SELECT_IDLECMD
                        and chain == 0 and (turn, player) not in seen):
                    seen.add((turn, player))
                    name = decks[player] if player < len(decks) else ""
                    if not is_train_deck(name, pool_dir, hold, new):
                        pass
                    elif decks_allow is not None and name not in decks_allow:
                        pass
                    else:
                        b = bucket_of(turn, went_first, int(g[0]), int(g[1]), int(g[6]))
                        if (b is not None and b in buckets and b not in took
                                and counts[(name, b)] < per_bucket):
                            took.add(b)
                            counts[(name, b)] += 1
                            out.append({
                                "deck": name, "bucket": b, "game_id": game_id, "k": k,
                                "prefix": choices[:k], "config": config,
                                "stored_config": stored,
                                "auto_single": auto_single, "decks": list(decks),
                                "turn": turn,
                            })
                obs = env.step(choices[k])
        except Exception:  # noqa: BLE001 - a replay error ends this game, not sampling
            continue
    return out


def state_obs(engine: Engine, state: dict):
    """The observation at a sampled state (replay the prefix)."""
    env = CppEnv(engine, state["config"], state["auto_single"])
    obs = env.reset()
    for c in state["prefix"]:
        if obs is None:
            return None
        obs = env.step(int(c))
    return obs


def build_prompt(engine: Engine, names: CardNames, sheets: dict[str, str],
                 state: dict, obs=None) -> tuple[str, dict]:
    """(turn prompt, info) for a sampled state."""
    obs = obs if obs is not None else state_obs(engine, state)
    state_text = describe_state(obs, names.names)
    legal, _ = legal_actions(obs, names.names)
    prompt = turn_prompt(state["deck"], sheets[state["deck"]], state_text,
                         can_attack=state["turn"] > 1, examples=[], legal=legal)
    if PRINCIPLES.is_file():
        prompt = PRINCIPLES.read_text() + "\n\n" + prompt
    return prompt, {"state_text": state_text, "legal": legal}


def ground_line(d: dict, names: CardNames, deck_name: str, line_id: str,
                went_first: bool) -> ComboLine:
    line = ComboLine.from_json(d)
    line.id = line_id
    line.deck = deck_name
    line.going = "first" if went_first else "second"
    line.source_file = "plan_turns"
    line.steps = [st for st in line.steps if st.card or st.action not in ("other", "pass")]
    for j, st in enumerate(line.steps):
        st.i = j
    _fill_codes(line, names)
    return line


def make_turn_record(engine: Engine, state: dict, line: ComboLine, v, model: str) -> dict:
    k = state["k"]
    labels = ["prefix"] * k + [str(x) for x in v.labels[k:]]
    actors, turns = replay_actors_turns(engine, state["config"], state["auto_single"], v.choices)
    return {
        "line_id": line.id,
        "game_id": state["game_id"],
        "bucket": state["bucket"],
        "deck": state["deck"],
        "decks": list(state["decks"]),
        "config": state["stored_config"],
        "auto_single": state["auto_single"],
        "choices": [int(c) for c in v.choices],
        "labels": labels,
        "actors": actors,
        "turns": turns,
        "plan": line.to_json(),
        "llm": {"model": model, "prompt_version": PROMPT_VERSION},
    }


def parse_verify(engine: Engine, names: CardNames, sheets: dict[str, str],
                 pool_decks: dict, state: dict, resp) -> tuple[dict | None, str, str]:
    """One parse + verify round. Returns (record|None, status, repair_prompt).

    Status is "ok", "unparsable", "malformed: ...", "pass" or "verify:<reason>".
    The repair prompt is "" unless another attempt is worthwhile.
    """
    obs = state_obs(engine, state)
    if obs is None:
        return None, "duel_over", ""
    went_first = bool(int(obs["global"][4]))
    base, _ = build_prompt(engine, names, sheets, state, obs)
    deck_name = state["deck"]
    line_id = f"{deck_name}/turn-{state['bucket']}-{state['game_id']}-{state['k']}"
    dicts = _parse_json(resp if isinstance(resp, str) else str(resp)) \
        if not isinstance(resp, BaseException) else []
    if not dicts:
        what = f"llm error: {resp}"[:200] if isinstance(resp, BaseException) else "unparsable"
        rp = base + "\n\nYour previous answer was not valid JSON with a 'lines' list. Answer again."
        return None, what, rp
    try:
        line = ground_line(dicts[0], names, deck_name, line_id, went_first)
    except Exception as e:  # noqa: BLE001
        return None, f"malformed: {e}"[:200], ""
    if not line.steps:
        return None, "pass", ""
    v = plan_from_state(engine, state["config"], state["prefix"], line,
                        pool_decks[deck_name], budget=state["budget"])
    if v.ok:
        return make_turn_record(engine, state, line, v, state["model"]), "ok", ""
    try:
        ex = explain_failure(engine, line, pool_decks[deck_name], v, names.names)
    except Exception:  # noqa: BLE001
        ex = ""
    rp = repair_prompt(base, compact_line_json(line), f"{v.reason}: {v.detail}", ex)
    return None, f"verify:{v.reason}", rp


def plan_states(engine: Engine, names: CardNames, sheets: dict[str, str],
                pool_decks: dict, states: list[dict], complete_many_fn,
                budget: int, repairs: int, model: str) -> list[tuple[dict | None, str]]:
    """Plan + verify every state; batched LLM rounds, in-process engine work."""
    for s in states:
        s["budget"] = budget
        s["model"] = model
    results: list[tuple[dict | None, str]] = [(None, "pending")] * len(states)
    pending = list(range(len(states)))
    prompts = [build_prompt(engine, names, sheets, states[i])[0] for i in pending]
    for attempt in range(repairs + 1):
        if not pending:
            break
        responses = complete_many_fn(prompts)
        nxt, nxt_prompts = [], []
        for i, resp in zip(pending, responses):
            rec, status, rp = parse_verify(engine, names, sheets, pool_decks, states[i], resp)
            if rec is not None:
                results[i] = (rec, "ok")
            elif attempt < repairs and rp:
                nxt.append(i)
                nxt_prompts.append(rp)
            else:
                results[i] = (None, status)
        pending, prompts = nxt, nxt_prompts
    return results


def load_game_records(patterns: list[str]) -> list[dict]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in sorted(glob.glob(pat, recursive=True)))
    seen, recs = set(), []
    for f in files:
        if not f.is_file() or str(f) in seen:
            continue
        seen.add(str(f))
        try:
            for raw in f.read_text().splitlines():
                if raw.strip():
                    recs.append(json.loads(raw))
        except (OSError, json.JSONDecodeError) as e:
            print(f"plan_turns: skipping {f}: {e}", flush=True)
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+", required=True, help="record globs (e.g. data/demos/sw-v/*.jsonl)")
    ap.add_argument("--per-bucket", type=int, default=4)
    ap.add_argument("--buckets", nargs="+", default=list(BUCKETS), choices=list(BUCKETS))
    ap.add_argument("--decks", nargs="*", default=None, help="optional allow-list of acting decks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--repairs", type=int, default=1)
    ap.add_argument("--llm-timeout", type=int, default=900)
    ap.add_argument("--out-dir", default="data/demos/turns")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--llm-only", action="store_true",
                    help="make the first-round LLM calls (fills the cache) and stop; with --prompts, answer those")
    ap.add_argument("--prompts", type=Path, default=None, help="jsonl of {'prompt'} (a --pending-out) for --llm-only")
    ap.add_argument("--cache-only", action="store_true",
                    help="never call the LLM: states with an uncached first prompt are skipped, uncached repair "
                         "prompts go to --pending-out")
    ap.add_argument("--pending-out", type=Path, default=None)
    ap.add_argument("--deck-groups", type=Path, default=None, help="file: one --decks chunk per line")
    ap.add_argument("--ids", type=Path, default=None, help="only the states whose line id is listed (one per line)")
    ap.add_argument("--ids-out", type=Path, default=None,
                    help="--llm-only: append the line ids of states whose first prompt was not cached before this run")
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=1)
    args = ap.parse_args(argv)
    if args.prompts and not args.llm_only:
        ap.error("--prompts needs --llm-only")
    if args.llm_only and args.cache_only:
        ap.error("--llm-only and --cache-only exclude each other")
    sfx = f".p{args.proc}" if args.nproc > 1 else ""

    def llm_many(prompts: list[str]) -> list:
        return _llm.complete_many(prompts, model=args.model, system=TURN_SYSTEM,
                                  max_workers=args.workers, return_exceptions=True,
                                  timeout=args.llm_timeout, variant=args.variant)

    def cached(prompt: str) -> str | None:
        return _llm.cached(prompt, model=args.model, system=TURN_SYSTEM, variant=args.variant)

    if args.prompts:  # answer the pending repair prompts of a --cache-only round
        prompts = sorted({json.loads(x)["prompt"] for x in args.prompts.read_text().splitlines() if x.strip()})
        todo = [p for p in prompts if cached(p) is None]
        print(f"plan_turns: {len(prompts)} pending prompts, {len(todo)} not cached", flush=True)
        res = llm_many(todo)
        bad = sum(isinstance(r, BaseException) for r in res)
        print(f"plan_turns: answered {len(todo) - bad}/{len(todo)} ({bad} llm errors)", flush=True)
        return 2 if todo and bad == len(todo) else 0

    engine, names = Engine(), CardNames()
    recs = load_game_records(args.games)
    print(f"plan_turns: {len(recs)} games", flush=True)
    # the sampled states depend on the deck allow-list (one state per game per bucket), so a split run samples each
    # original chunk (--deck-groups, one space-separated chunk per line) exactly as the laptop sweep did
    groups = ([ln.split() for ln in args.deck_groups.read_text().splitlines() if ln.strip()] if args.deck_groups
              else [args.decks])
    states = [s for g in groups for s in sample_states(engine, recs, args.seed, args.per_bucket, tuple(args.buckets),
                                                       set(g) if g else None)]
    print(f"plan_turns: {len(states)} states "
          f"({dict(Counter((s['bucket'], s['deck']) for s in states))})", flush=True)
    if not states:
        print("ALERT plan_turns: 0 states sampled", flush=True)
        return 2

    pool_names = sorted({s["deck"] for s in states})
    pool_decks, sheets = {}, {}
    for n in pool_names:
        try:
            pool_decks[n] = load_ydk(ROOT / "decks" / "pool" / f"{n}.ydk")
            sheets[n] = card_sheet(engine, pool_decks[n], names)
        except OSError as e:
            print(f"plan_turns: no pool deck {n}: {e}", flush=True)
    states = [s for s in states if s["deck"] in pool_decks]
    if not states:
        print("ALERT plan_turns: 0 states sampled", flush=True)
        return 2

    if args.dry_run:
        for s in states[:2]:
            prompt, _ = build_prompt(engine, names, sheets, s)
            print(f"===== {s['deck']}/{s['bucket']} game={s['game_id']} k={s['k']} =====")
            print(prompt)
        return 0

    pending: list[str] = []

    def cache_many(prompts: list[str]) -> list:
        out: list = []
        for p in prompts:
            r = cached(p)
            if r is None:
                pending.append(p)
                out.append(RuntimeError("not cached"))
            else:
                out.append(r)
        return out

    complete_many_fn = cache_many if args.cache_only else llm_many

    out_dir = Path(args.out_dir)
    done = {json.loads(x)["line_id"] for f in out_dir.glob("*/*.jsonl") for x in f.read_text().splitlines() if x.strip()}
    states = [s for s in states if f"{s['deck']}/turn-{s['bucket']}-{s['game_id']}-{s['k']}" not in done]
    if args.ids:
        allow = {x.strip() for x in args.ids.read_text().splitlines() if x.strip()}
        states = [s for s in states if f"{s['deck']}/turn-{s['bucket']}-{s['game_id']}-{s['k']}" in allow]
    states = [s for s in states
              if zlib.crc32(f"{s['deck']}/turn-{s['bucket']}-{s['game_id']}-{s['k']}".encode()) % args.nproc == args.proc]
    print(f"plan_turns: {len(states)} states to plan ({len(done)} lines already in {out_dir})", flush=True)
    if args.llm_only:
        prompts = [build_prompt(engine, names, sheets, s)[0] for s in states]
        if args.ids_out:
            args.ids_out.parent.mkdir(parents=True, exist_ok=True)
            with args.ids_out.open("a") as f:
                for s, p in zip(states, prompts):
                    if cached(p) is None:
                        f.write(f"{s['deck']}/turn-{s['bucket']}-{s['game_id']}-{s['k']}\n")
        res = llm_many(prompts)
        bad = sum(isinstance(r, BaseException) for r in res)
        print(f"plan_turns: llm-only: {len(res) - bad}/{len(res)} answered ({bad} llm errors)", flush=True)
        return 2 if res and bad == len(res) else 0
    if args.cache_only:
        n0 = len(states)
        states = [s for s in states if cached(build_prompt(engine, names, sheets, s)[0]) is not None]
        print(f"plan_turns: cache-only: {len(states)}/{n0} states have a cached first answer", flush=True)
        if n0 and not states:
            print(f"ALERT plan_turns: 0 of {n0} states have a cached first answer (cache not shipped?)", flush=True)
            return 2
    results = plan_states(engine, names, sheets, pool_decks, states, complete_many_fn,
                          args.budget, args.repairs, args.model)
    report: dict = {"per": {}, "states": len(states), "verified": 0, "attempts": 0}
    for s, (rec, status) in zip(states, results):
        key = f"{s['bucket']}/{s['deck']}"
        e = report["per"].setdefault(key, {"states": 0, "parsed": 0, "verified": 0,
                                           "reasons": Counter()})
        e["states"] += 1
        report["attempts"] += 1
        if rec is not None:
            e["parsed"] += 1
            e["verified"] += 1
            report["verified"] += 1
            p = out_dir / s["bucket"] / f"{s['deck']}{sfx}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        else:
            if status.startswith("verify:"):
                e["parsed"] += 1
            e["reasons"][status] += 1
    serial = {k: {**v, "reasons": dict(v["reasons"])} for k, v in report["per"].items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"report{sfx}.json").write_text(json.dumps({**report, "per": serial}, indent=1))
    if args.pending_out:
        args.pending_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pending_out.open("a") as f:
            for p in pending:
                f.write(json.dumps({"prompt": p}) + "\n")
        print(f"plan_turns: {len(pending)} uncached prompts -> {args.pending_out}", flush=True)
    print(f"plan_turns: verified={report['verified']}/{report['states']}", flush=True)
    for k in sorted(serial):
        print(f"  {k}: {serial[k]}", flush=True)
    if report["verified"] == 0 and report["attempts"] >= 10:
        print(f"ALERT plan_turns: 0 plans verified out of {report['attempts']} attempts", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
