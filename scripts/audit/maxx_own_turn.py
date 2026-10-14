"""Where does a policy fire Maxx "C" on its own turn with no opponent chain link? (Pc 2026-10-13: 251 of P's 420
own-T1 hand-trap fires.) For each such fire in search_match games: turn, phase, message, how many own decisions
came before it this turn, own field/hand size, and the other legal options. Prints a summary and the first --show cases.

    uv run python scripts/audit/maxx_own_turn.py runs/<job>/p [--seat s] [--max-games N] [--show 8]
    uv run python scripts/audit/maxx_own_turn.py --records 'data/demos/teacher/*.jsonl' --max-games 1500
      (teacher records: every actor's decisions, stats split by the record's `source`, e.g. sampled vs argmax)
"""

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from principles import LOC_HAND, iter_games  # noqa: E402

from ygo_zero.cards.embed import card_texts  # noqa: E402

from ygo_zero.engine.actions import Kind  # noqa: E402
from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import DuelConfig, Engine  # noqa: E402

MAXX = 23434538


def iter_records(pattern: str, engine: Engine, max_games: int):
    """Teacher records (`choices` + `config`, already-shuffled mains): every decision, tagged by the record's source."""
    from types import SimpleNamespace

    files = sorted(glob.glob(pattern, recursive=True))
    random.Random(0).shuffle(files)
    n = 0
    for f in files:
        for raw in open(f):
            r = json.loads(raw)
            c = r["config"]
            cfg = DuelConfig(SimpleNamespace(main=c["deck0_main"], extra=c["deck0_extra"]),
                             SimpleNamespace(main=c["deck1_main"], extra=c["deck1_extra"]), int(c["seed"]),
                             lp=c.get("lp", 8000), hand=c.get("hand", 5), draw=c.get("draw", 1), shuffle=bool(c.get("shuffle", False)))
            env = CppEnv(engine, cfg, bool(r.get("auto_single", True)))
            obs = env.reset()
            tag = "sampled" if "sampled" in r.get("source", "") else "argmax"
            for ch in r["choices"]:
                if obs is None:
                    break
                yield obs, int(ch), f"{tag}|{r['line_id']}", None
                obs = env.step(int(ch))
            n += 1
            if max_games and n >= max_games:
                return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--records", default="", help="glob of teacher record files instead of game dirs")
    ap.add_argument("--seat", choices=("s", "ya"), default="s")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    engine = Engine()
    texts = card_texts()
    name = lambda c: texts.get(int(c), (str(c), ""))[0] or str(c)  # noqa: E731
    stats, shown, games, last_line, k_turn = Counter(), 0, set(), None, 0
    src = iter_records(a.records, engine, a.max_games) if a.records else iter_games(a.dirs, engine, a.seat)
    for obs, ch, line, _ in src:
        if a.max_games and len(games) >= a.max_games and line not in games:
            break
        games.add(line)
        g, acts, mask = obs["global"], obs["actions"], obs["action_mask"].astype(bool)
        own, turn, phase, msg = int(g[5]) == 1, int(g[2]), int(g[3]), int(g[21])
        key = (line, turn)
        if key != last_line:
            last_line, k_turn = key, 0
        k_turn += 1
        cm = obs["chain_mask"].astype(bool)
        opp_link = bool((obs["chain"][cm][:, 4] == 1).any()) if cm.any() else False
        legal = [i for i in range(len(mask)) if mask[i]]
        is_maxx = lambda i: int(acts[i, 0]) == Kind.ACTIVATE and int(acts[i, 7]) == LOC_HAND and int(acts[i, 2]) == MAXX  # noqa: E731
        if not own or opp_link or not any(is_maxx(i) for i in legal):
            continue
        t = ("T1" if turn == 1 else "T2+") + (f" {line.split('|')[0]}" if a.records else "")
        stats[f"{t} offered"] += 1
        if not is_maxx(ch):
            continue
        stats[f"{t} fired"] += 1
        stats[f"{t} fired, chain {'open' if cm.any() else 'empty'}"] += 1
        stats[f"{t} fired, own decision #{min(k_turn, 4)}{'+' if k_turn >= 4 else ''} of the turn"] += 1
        stats[f"{t} fired, {len(legal)} legal options" if len(legal) <= 3 else f"{t} fired, 4+ legal options"] += 1
        if shown < a.show:
            shown += 1
            others = [f"{Kind(int(acts[i, 0])).name}:{name(int(acts[i, 2]))}" for i in legal if i != ch]
            print(f"{line} {t} phase 0x{phase:x} msg {msg} own decision #{k_turn} chain {int(cm.sum())}; "
                  f"others: {others[:6]}")
    print(f"games {len(games)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
