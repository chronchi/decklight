"""LLM realism judge for full-game turns and interruption decisions (Task C).

For each game record in data/demos/games not yet judged, replay it in the
engine to build a compact per-turn narrative (acting deck, hand at turn
start, plan title/steps, actions taken, interruptions with their `why`, LP
after the turn) and ask muse (default effort, one call per game) to rate
every turn 1-5 and every interruption decision 1-5. Judgements go to
data/demos/games_judged/<same file name>.jsonl, one line per game, and
`demos.dataset.iter_samples` uses them to down-weight/drop bad decisions.

Run: YGO_OPENCODE_SERVER=http://127.0.0.1:4097 nice -n 19 taskset -c 0-3 \\
     uv run python scripts/judge_games.py --limit 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.demos import llm as _llm
from ygo_zero.demos.dataset import JUDGE_WEIGHT, record_config
from ygo_zero.demos.names import CardNames
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine

MODEL = "opencode-go/muse-spark-1.3-contributor"
JUDGED_DIR = ROOT / "data" / "demos" / "games_judged"

# Actions worth naming in the narrative; everything else is counted, not listed.
INTERESTING = {Kind.SUMMON, Kind.SPSUMMON, Kind.MSET, Kind.SSET, Kind.ACTIVATE, Kind.ATTACK, Kind.REPOS}
_MAX_NARR_CHARS = 9000

PROMPT_TMPL = """You are a top competitive Yu-Gi-Oh! player. Below is a full duel between "{d0}" (goes first) and "{d1}", summarized turn by turn from the engine replay. "Plan" is what the acting player intended (title + steps); "Actions" is what it actually did (summons, activations, sets, attacks); "Interrupts" are hand-trap/quick-effect decisions by the NON-turn player with their stated reason.

{GAME}

Rate each turn 1-5 for how a strong player with this hand and board would play (5 = what a strong player would do, 3 = playable but clearly suboptimal, 1 = nonsensical: e.g. set-and-pass with a live combo hand, wasting removal on nothing, never attacking for lethal), and each interruption decision 1-5 (activation or pass) for timing/target (5 = stops a key play or correctly saves the card, 1 = pointless use or missing an obvious stop). Be strict: most turns of a decent player are 3-4, not 5.
JSON only, no prose: {{"turns": {{"<turn>": {{"score": 1-5, "why": "..."}}}}, "interrupts": [{{"turn": N, "player": P, "card": "...", "score": 1-5, "why": "..."}}]}}.
Rate EVERY turn shown above and EVERY interrupt shown above.
If no interrupts are shown for the whole game, return "interrupts": []. Never invent interrupts."""


def _kind_name(v: int) -> str:
    try:
        return Kind(v).name
    except ValueError:
        return str(v)


def _hand_names(obs, names: dict[int, str]) -> list[str]:
    out = []
    for r in obs["cards"][obs["card_mask"]]:
        if int(r[3]) == 0 and int(r[1]) == 2 and int(r[0]):
            out.append(names.get(int(r[0]), f"#{int(r[0])}"))
    return out


def _lp_absolute(obs, player: int) -> list[int]:
    g = obs["global"]
    return [int(g[0]), int(g[1])] if player == 0 else [int(g[1]), int(g[0])]


def _plan_summary(plans: list[dict], turn: int, max_steps: int = 10) -> str:
    bits = []
    for p in plans:
        if p.get("turn") != turn:
            continue
        ok = [a for a in p.get("attempts", []) if a.get("reason") == "ok" and isinstance(a.get("line"), dict)]
        if not ok:
            reasons = [str(a.get("reason", "?")) for a in p.get("attempts", [])]
            bits.append(f"[{p.get('why')}: no verified plan ({', '.join(reasons)})]")
            continue
        line = ok[-1]["line"]
        steps = [f"{s.get('action', '?')} {s.get('card') or ''}".strip() for s in line.get("steps", [])]
        if len(steps) > max_steps:
            steps = steps[:max_steps] + [f"... +{len(line['steps']) - max_steps} more"]
        title = line.get("title") or ""
        bits.append(f"[{p.get('why')}] {title}: {' // '.join(steps)}"[:600])
    return " | ".join(bits) or "none"


def build_narrative(rec: dict, engine: Engine, names: CardNames) -> str:
    """Compact per-turn narrative of a game record, via an engine replay."""
    nm = names.names
    env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
    obs = env.reset()
    decks = rec["decks"]
    interrupts = rec.get("interrupts", []) or []
    by_turn: dict[int, dict] = {}
    order: list[int] = []

    def turn_block(t: int) -> dict:
        if t not in by_turn:
            by_turn[t] = {"hand": None, "deck": None, "player": None, "actions": [], "other": 0, "lp": None}
            order.append(t)
        return by_turn[t]

    for k, (choice, label) in enumerate(zip(rec["choices"], rec["labels"], strict=True)):
        if obs is None:
            break
        turn, actor = rec["turns"][k], rec["actors"][k]
        b = turn_block(turn)
        if b["hand"] is None:
            b.update(hand=_hand_names(obs, nm), deck=decks[actor], player=actor)
        row = obs["actions"][choice]
        kind, code = int(row[0]), int(row[2])
        card = nm.get(code, f"#{code}") if code else ""
        try:
            kk = Kind(kind)
        except ValueError:
            kk = None
        if kk in INTERESTING:
            b["actions"].append(f"{_kind_name(kind)} {card}".strip())
        else:
            b["other"] += 1
        obs = env.step(choice)
        # LP after the turn: first obs of a new turn, or the final state.
        nxt = rec["turns"][k + 1] if k + 1 < len(rec["turns"]) else None
        if obs is None:
            if env.winner is not None:
                b["lp"] = "game over"
        elif nxt != turn:
            lp = _lp_absolute(obs, env.player)
            b["lp"] = f"{lp[0]}-{lp[1]}"

    int_by_turn: dict[int, list[str]] = defaultdict(list)
    for e in interrupts:
        if isinstance(e, dict):
            int_by_turn[e.get("turn")].append(
                f"p{e.get('player')} {e.get('card', '?')} ({e.get('why', '')[:120]})"
            )
    lines = [
        (
            f"Game {rec.get('game_id')}: {decks[0]} vs {decks[1]}, winner={rec.get('winner')}, "
            f"final LP={rec.get('lp')}"
        )
    ]
    for t in order:
        b = by_turn[t]
        acts = b["actions"][:40]
        if len(b["actions"]) > 40:
            acts.append(f"... +{len(b['actions']) - 40} more")
        if b["other"]:
            acts.append(f"[{b['other']} minor decisions]")
        lines.append(
            f"Turn {t} — {b['deck']} (player {b['player']}):\n"
            f"  Hand: {', '.join(b['hand'] or []) or 'none'}\n"
            f"  Plan: {_plan_summary(rec.get('plans', []), t)}\n"
            f"  Actions: {'; '.join(acts) or 'none'}\n"
            f"  Interrupts: {'; '.join(int_by_turn.get(t, [])) or 'none'}\n"
            f"  LP after: {b['lp']}"
        )
    text = "\n".join(lines)
    return text[:_MAX_NARR_CHARS] + (" ..." if len(text) > _MAX_NARR_CHARS else "")


def parse_judgement(text: str) -> dict:
    """Parse the judge reply into {"turns", "interrupts"} with valid 1-5 scores."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    turns, interrupts = {}, []
    if isinstance(obj, dict):
        raw_turns = obj.get("turns") or {}
        if isinstance(raw_turns, dict):
            for k, v in raw_turns.items():
                v = v.get("score") if isinstance(v, dict) else v
                try:
                    s = int(v)
                except (TypeError, ValueError):
                    continue
                if s in JUDGE_WEIGHT:
                    try:
                        turns[str(int(k))] = {
                            "score": s,
                            "why": str(obj["turns"][k].get("why", ""))[:300]
                            if isinstance(obj["turns"][k], dict)
                            else "",
                        }
                    except (KeyError, TypeError, ValueError):
                        continue
        raw_int = obj.get("interrupts") or []
        if isinstance(raw_int, list):
            for e in raw_int:
                if not isinstance(e, dict):
                    continue
                try:
                    s = int(e.get("score", 0))
                    interrupts.append(
                        {
                            "turn": int(e["turn"]),
                            "player": int(e["player"]),
                            "card": str(e.get("card", ""))[:120],
                            "score": s,
                            "why": str(e.get("why", ""))[:300],
                        }
                    )
                except (TypeError, ValueError, KeyError):
                    continue
                if interrupts[-1]["score"] not in JUDGE_WEIGHT:
                    interrupts.pop()
    return {"turns": turns, "interrupts": interrupts}


def filter_interrupts(interrupts: list[dict], rec: dict) -> list[dict]:
    """Drop rated interrupts with no matching logged decision (the judge invents
    some when a game shows none). Match by (turn, player) as a multiset: at most
    as many kept as the record logged for that key, in order."""
    budget: dict[tuple[int, int], int] = defaultdict(int)
    for e in rec.get("interrupts", []) or []:
        if isinstance(e, dict):
            try:
                budget[(int(e["turn"]), int(e["player"]))] += 1
            except (KeyError, TypeError, ValueError):
                continue
    out = []
    for e in interrupts:
        try:
            key = (int(e["turn"]), int(e["player"]))
        except (KeyError, TypeError, ValueError):
            continue
        if budget.get(key, 0) > 0:
            budget[key] -= 1
            out.append(e)
    return out


def turn_decks(rec: dict) -> dict[str, str]:
    """Acting deck per turn: the plan owner (authoritative), else the majority
    actor. (The first choice of a turn can belong to the responding opponent.)"""
    decks = rec.get("decks", [])
    out: dict[str, str] = {}
    for p in rec.get("plans", []) or []:
        try:
            out.setdefault(str(int(p["turn"])), p.get("deck") or decks[int(p["player"])])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    if len(out) < len(set(rec.get("turns", []))):
        votes: dict[str, Counter] = defaultdict(Counter)
        for t, a in zip(rec.get("turns", []), rec.get("actors", []), strict=False):
            votes[str(t)][int(a)] += 1
        for t, c in votes.items():
            if t not in out:
                try:
                    out[t] = decks[c.most_common(1)[0][0]]
                except IndexError:
                    continue
    return out


def judged_ids(path: Path) -> set[str]:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.add(j.get("line_id") or j.get("game_id"))
    return out


def summarize(judged_dir: Path) -> dict:
    """Corpus stats over all judgement files."""
    deck_scores: dict[str, list[int]] = defaultdict(list)
    n_turns = n_bad = n_int = 0
    int_sum = 0
    n_games = 0
    for f in sorted(judged_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_games += 1
            for v in (j.get("turns") or {}).values():
                s = v.get("score") if isinstance(v, dict) else v
                if s not in JUDGE_WEIGHT:
                    continue
                n_turns += 1
                n_bad += s <= 2
                deck = (v.get("deck") if isinstance(v, dict) else None) or "?"
                deck_scores[deck].append(s)
            for e in j.get("interrupts") or []:
                if isinstance(e, dict) and e.get("score") in JUDGE_WEIGHT:
                    n_int += 1
                    int_sum += e["score"]
    per_deck = {d: round(sum(v) / len(v), 2) for d, v in deck_scores.items() if v}
    ranked = sorted(per_deck.items(), key=lambda kv: kv[1])
    return {
        "games": n_games,
        "turns": n_turns,
        "share_turns_le2": round(n_bad / n_turns, 3) if n_turns else None,
        "mean_interrupt": round(int_sum / n_int, 2) if n_int else None,
        "per_deck_mean": per_deck,
        "bottom5": ranked[:5],
        "top5": ranked[-5:][::-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", type=Path, default=ROOT / "data" / "demos" / "games")
    ap.add_argument("--judged-dir", type=Path, default=JUDGED_DIR)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="max new games to judge (0 = all)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    engine, names = Engine(), CardNames()
    args.judged_dir.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[Path, dict, str]] = []
    for f in sorted(args.games_dir.glob("*.jsonl")):
        if args.limit and len(todo) >= args.limit:
            break
        done = judged_ids(args.judged_dir / f.name)
        for line in f.read_text().splitlines():
            if args.limit and len(todo) >= args.limit:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            if (rec.get("line_id") or rec.get("game_id")) in done:
                continue
            try:
                narr = build_narrative(rec, engine, names)
            except Exception as e:  # noqa: BLE001 - one bad replay must not kill the batch
                print(f"skip {rec.get('line_id')}: narrative failed: {e}", flush=True)
                continue
            decks = rec["decks"]
            prompt = PROMPT_TMPL.format(d0=decks[0], d1=decks[1], GAME=narr)
            if len(prompt) > 12000:
                prompt = prompt[:12000] + ' ..."'
            todo.append((f, rec, prompt))
    print(f"{len(todo)} games to judge", flush=True)
    if not todo:
        print(json.dumps(summarize(args.judged_dir), indent=1))
        return
    resps = _llm.complete_many(
        [p for _, _, p in todo], model=args.model, max_workers=args.workers, return_exceptions=True
    )
    new = 0
    for (f, rec, _), r in zip(todo, resps, strict=True):
        if isinstance(r, Exception):
            print(f"LLM failed for {rec.get('line_id')}: {r}", flush=True)
            continue
        j = parse_judgement(r)
        # attach the acting deck per turn for the per-deck summary
        decks_by_turn = turn_decks(rec)
        for t, v in j["turns"].items():
            v["deck"] = decks_by_turn.get(t, "?")
        out = {
            "game_id": rec.get("game_id"),
            "line_id": rec.get("line_id"),
            "turns": j["turns"],
            "interrupts": filter_interrupts(j["interrupts"], rec),
        }
        with (args.judged_dir / f.name).open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        new += 1
    print(f"judged {new}/{len(todo)} new games", flush=True)
    print(json.dumps(summarize(args.judged_dir), indent=1))


if __name__ == "__main__":
    main()
