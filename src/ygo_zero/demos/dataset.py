"""Behaviour-cloning samples from engine-verified combo lines (ADR 0002).

Only replay records are stored (`data/demos/verified/*.jsonl`: config + choices + labels); observations are
regenerated here by replaying each record in the C++ env (ADR 0001 storage rule).

    for obs, action, weight, meta in iter_samples(engine, Path("data/demos/verified")):
        ...  # obs: dict of arrays for the player to act; action: option index

Weights: choices stated by the guide count fully; choices the verifier found by search are plausible but
arbitrary among equivalent ones (zones, which copy), so they count less; the opponent's automatic moves are
not samples at all.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine, Placement

LABEL_WEIGHT = {"stated": 1.0, "inferred": 0.5, "interrupt": 1.0, "end": 1.0, "battle": 1.0, "step": 1.0}  # battle: LLM Battle Phase picks; step: LLM decision-by-decision turn play
# hand-played games (data/demos/gold, scripts/hand_play.py) use their own labels: gold = a chosen move, pass = a PASS*
# auto-pass (rule-applied after a deliberate decline: half weight), auto = zone pick (not trained)
GOLD_LABEL_WEIGHT = {"gold": 1.0, "pass": 0.5}
# turn-1 search teacher records (scripts/gen_search.py, ADR 0006 M2): search = player 0's searched move, opp = the
# passive opponent (not trained)
SEARCH_LABEL_WEIGHT = {"search": 1.0}
# mid-game search teacher (scripts/gen_window.py, M3): searched response windows and own-turn plays; skip = not trained
WINDOW_LABEL_WEIGHT = {"window": 1.0, "own": 1.0}
_PASSIVE_KINDS = (Kind.PASS, Kind.NO, Kind.FINISH, Kind.CANCEL)


def end_turn_sample(env, obs, me: int, max_steps: int = 60) -> tuple[dict, int] | None:
    """After a line's last choice: answer pending prompts passively until player `me` is back at an open idle state in
    the same turn, and return (that obs, the index of its "end turn" option). Line records stop at the last combo step,
    so without this the model never learns to stop (playtest 2026-10-03: it reproduced a Mikanko line exactly, then
    kept acting and wrecked its board). None when the turn ends on its own or no end option exists."""
    if obs is None:
        return None
    turn = int(obs["global"][2])
    for _ in range(max_steps):
        if obs is None or int(obs["global"][2]) != turn:
            return None
        n = int(obs["action_mask"].sum())
        kinds = [int(obs["actions"][i][0]) for i in range(n)]
        if env.player == me and int(obs["global"][21]) == C.MSG_SELECT_IDLECMD:
            for want in (Kind.TO_END, Kind.TO_MAIN2):  # turn 1 has no battle: Main 1 -> End
                if want in kinds:
                    return {k: v.copy() for k, v in obs.items()}, kinds.index(want)
            return None
        a = next((i for i, k in enumerate(kinds) if k in _PASSIVE_KINDS), 0)
        obs = env.step(a)
    return None

ROOT = Path(__file__).resolve().parents[3]
JUDGED_DIR = ROOT / "data" / "demos" / "games_judged"

# Realism-judge multiplier by turn/interrupt score (Task C): bad turns are
# dropped, mediocre ones down-weighted, good ones unchanged.
JUDGE_WEIGHT = {5: 1.0, 4: 1.0, 3: 0.5, 2: 0.0, 1: 0.0}

_judge_cache: dict[str, dict[str, dict]] = {}


def sample_kind(source: str | Path, label: str) -> str:
    """BC data-mix kind tag from the record file path and the sample label.

    - files under `expanded*` -> line_expanded (wins over improved)
    - files under `improved/` (not expanded) -> line_improved
    - files under `games/` -> game_interrupt if label == "interrupt" else game_plan
    - files under `gold/` -> gold_pass if label == "pass" else gold
    - files under `teacher/` -> teacher_pass if label == "pass" else teacher
    - files under `dagger/` -> dagger_pass if label == "pass" else dagger
    - files under `search/` -> search
    - files under `window/` -> window_own if label == "own" else window
    - files under `sw-<name>/` (relabelled self-play window search, ADR 0007) -> sw-<name>_own if label == "own"
      else sw-<name>, so each labelling is its own kind in --kind-frac
    - everything else -> line
    """
    parts = Path(source).parts
    sw = next((p for p in parts if p.startswith("sw-")), None)
    if sw is not None:
        return f"{sw}_own" if label == "own" else sw
    if "gold" in parts:
        return "gold_pass" if label == "pass" else "gold"
    if "teacher" in parts:
        return "teacher_pass" if label == "pass" else "teacher"
    if "dagger" in parts:
        return "dagger_pass" if label == "pass" else "dagger"
    if "search" in parts:
        return "search"
    if "window" in parts:
        return "window_own" if label == "own" else "window"
    if any(p.startswith("expanded") for p in parts):
        return "line_expanded"
    if "improved" in parts:
        return "line_improved"
    if "games" in parts:
        return "game_interrupt" if label == "interrupt" else "game_plan"
    return "line"


def load_judgements(source: str | Path) -> dict[str, dict]:
    """Judgements for a record file, keyed by line_id (and game_id): {} when unjudged.

    The judgement file is `JUDGED_DIR/<same file name>.jsonl`, one line per
    game (see scripts/judge_games.py). Results are cached per source path.
    """
    key = str(source)
    if key not in _judge_cache:
        out: dict[str, dict] = {}
        p = JUDGED_DIR / Path(source).name
        if p.exists():
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(j, dict):
                    continue
                for k in (j.get("line_id"), j.get("game_id")):
                    if k and k not in out:
                        out[k] = j
        _judge_cache[key] = out
    return _judge_cache[key]


def _valid_score(v) -> int | None:
    try:
        s = int(v)
    except (TypeError, ValueError):
        return None
    return s if s in JUDGE_WEIGHT else None


def judge_turn_score(judgement: dict | None, turn: int) -> int | None:
    """The judged realism score for a turn, or None when unjudged/invalid."""
    if not judgement:
        return None
    entry = (judgement.get("turns") or {}).get(str(turn))
    if isinstance(entry, dict):
        entry = entry.get("score")
    return _valid_score(entry)


def judge_interrupt_scores(judgement: dict | None) -> dict[tuple[int, int], list[int]]:
    """Valid interrupt scores grouped by (turn, player), in judgement order."""
    out: dict[tuple[int, int], list[int]] = {}
    for e in (judgement or {}).get("interrupts") or []:
        if not isinstance(e, dict):
            continue
        s = _valid_score(e.get("score"))
        if s is None:
            continue
        try:
            out.setdefault((int(e["turn"]), int(e["player"])), []).append(s)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def record_config(rec: dict, seed: int | None = None) -> DuelConfig:
    """The DuelConfig stored in a verified record (optionally with another seed from `seeds_ok`)."""
    c = rec["config"]
    deck0 = Deck(tuple(c["deck0_main"]), tuple(c["deck0_extra"]))
    deck1 = Deck(tuple(c["deck1_main"]), tuple(c["deck1_extra"]))
    setup = tuple(Placement(*p) for p in c.get("setup", []))
    return DuelConfig(
        deck0, deck1, seed=c["seed"] if seed is None else seed, lp=c.get("lp", 8000), hand=c.get("hand", 0),
        draw=c.get("draw", 1), setup=setup, shuffle=c.get("shuffle", False),  # no key: made before decks were shuffled
    )  # fmt: skip


def load_records(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def starved_opening(rec: dict) -> bool:
    """An opening game recorded before 2026-10-03 21:30: the line record's config gave the opponent no opening hand
    (0 cards, so 1 card on its turn 2). Unrealistic for both players: never trained on (NOTEBOOK 2026-10-03 21:30)."""
    if not rec.get("opening") or "game_id" not in rec:
        return False
    c = rec.get("config", {})
    return c.get("hand", 5) == 0 and not any(int(p[0]) == 1 and int(p[2]) == C.LOCATION_HAND for p in c.get("setup", []))


def iter_samples(engine: Engine, path: Path) -> Iterator[tuple[dict, int, float, dict]]:
    """Yield (obs, action, weight, meta) for every agent decision of every verified record.

    Only the record's own seed is replayed: `choices` come from that seed's search, and other seeds in
    `seeds_ok` were verified with their own (unstored) choices. The replay always steps `choices`; when the
    record carries `targets` (DAgger: the labeler's option index at student decisions, null elsewhere) the
    yielded action is the target, and the sample is skipped when the target is not a legal option there.
    Records without `targets` yield their choices unchanged."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        judges = load_judgements(f)
        weights = GOLD_LABEL_WEIGHT if ("gold" in f.parts or "teacher" in f.parts or "dagger" in f.parts) else LABEL_WEIGHT
        weights = SEARCH_LABEL_WEIGHT if "search" in f.parts else weights
        weights = WINDOW_LABEL_WEIGHT if ("window" in f.parts or any(q.startswith("sw-") for q in f.parts)) else weights
        for rec in load_records(f):
            if starved_opening(rec):
                continue
            out: list = []  # buffered: a record that fails to replay is skipped whole (never half-yielded)
            try:
                judge = judges.get(rec.get("line_id")) or judges.get(rec.get("game_id"))
                iscores = judge_interrupt_scores(judge)
                used: dict[tuple[int, int], int] = {}
                for seed in [rec["config"]["seed"]]:
                    env = CppEnv(engine, record_config(rec, seed), rec.get("auto_single", True))
                    obs = env.reset()
                    targets = rec.get("targets")  # DAgger records: the labeler's move at student decisions
                    target_sets = rec.get("target_sets")
                    target_conf = rec.get("target_conf")
                    for k, (choice, label) in enumerate(zip(rec["choices"], rec["labels"], strict=True)):
                        if obs is None:
                            raise ValueError(f"{rec['line_id']} seed {seed}: duel ended at choice {k}")
                        target = targets[k] if targets is not None else None
                        action = choice if target is None else int(target)
                        if target is not None:  # the target must be legal here, else there is no sample
                            mask = obs.get("action_mask")
                            if mask is None or not 0 <= action < mask.shape[0] or not bool(mask[action]):
                                obs = env.step(choice)
                                continue
                        weight, score = weights.get(label, 0.0), None
                        if weight and judge is not None and "turns" in rec and "actors" in rec:
                            turn, actor = rec["turns"][k], rec["actors"][k]
                            if label == "interrupt":
                                key = (turn, actor)
                                pool = iscores.get(key, [])
                                i = used.get(key, 0)
                                if i < len(pool):
                                    score = pool[i]
                                used[key] = i + 1
                            else:
                                score = judge_turn_score(judge, turn)
                            if score is not None:
                                weight *= JUDGE_WEIGHT[score]
                        alt: list[int] | None = None
                        if target_sets is not None and k < len(target_sets) and target_sets[k] is not None:
                            mask = obs.get("action_mask")
                            alt = [int(a) for a in target_sets[k]
                                   if 0 <= int(a) < len(mask) and bool(mask[int(a)])] if mask is not None else \
                                [int(a) for a in target_sets[k]]
                            if target_conf is not None and k < len(target_conf) and target_conf[k] is not None:
                                weight *= float(target_conf[k])
                        if weight:
                            # game records (ADR 0004) carry both decks and who acted; lines are named <deck>/<id>
                            deck = (
                                rec["decks"][rec["actors"][k]]
                                if "actors" in rec
                                else rec["line_id"].split("/")[0]
                            )
                            meta = {
                                "line_id": rec["line_id"],
                                "seed": seed,
                                "k": k,
                                "label": label,
                                "deck": deck,
                                "kind": sample_kind(f, label),
                                "judge": score,
                                "hlog_end": env.history_len(),  # recorded entries before this decision
                                "me": env.player,  # acting seat, absolute
                                "turn": int(obs["global"][2]),
                            }
                            if alt is not None:
                                meta["alt"] = alt
                            out.append(({key: v.copy() for key, v in obs.items()}, action, weight, meta))
                        obs = env.step(choice)
                    if "actors" not in rec:  # a combo line: teach the stop (records end at the last combo step)
                        end = end_turn_sample(env, obs, rec.get("me", 0))
                        if end is not None:
                            obs_end, a_end = end
                            meta = {"line_id": rec["line_id"], "seed": seed, "k": len(rec["choices"]), "label": "end",
                                    "deck": rec["line_id"].split("/")[0], "kind": sample_kind(f, "end"), "judge": None,
                                    "hlog_end": env.history_len(), "me": env.player,
                                    "turn": int(obs_end["global"][2])}
                            out.append((obs_end, a_end, LABEL_WEIGHT["end"], meta))
                log = env.history_log()  # the record's full log, shared by all its samples (shards slice [:hlog_end])
                for _, _, _, m in out:
                    m["hlog"] = log
            except Exception as e:  # noqa: BLE001 - one bad record must not kill a shard build
                print(f"iter_samples: skipping {rec.get('line_id')} in {f.name}: {e!r}"[:300], flush=True)
                continue
            yield from out
