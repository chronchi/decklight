"""Run engine games under the AddressSanitizer build (scripts/build_asan.sh) to find core memory bugs.

  replay FILES...   replay flight files ($YGO_FLIGHT_DIR leftovers of a segfaulted process), each alone
  inter FILES...    the same files interleaved one step at a time through one shared Engine (as search_match runs them)
  fuzz --games N    uniformly random play on random deck pairs from decks/{pool,newcards,gauntlet}

Run through the wrapper so ASan is preloaded:
  bash scripts/audit/asan_check.sh replay runs/<job>/<variant>/flight/*.jsonl
ASan aborts with a report on the first error (exit 1); exit 0 means no memory error was seen.
Engine only (no model): fine on the laptop under taskset -c 0-3.
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if "--release" not in sys.argv:  # --release: the normal build (fast; for hangs, not memory errors)
    _ext = sorted((ROOT / "build/asan/ext").glob("_ygoenv*.so"))
    if not _ext:
        sys.exit("no ASan build: run scripts/build_asan.sh")
    _spec = importlib.util.spec_from_file_location("ygo_zero._ygoenv", _ext[0])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["ygo_zero._ygoenv"] = _mod
    _spec.loader.exec_module(_mod)

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine, Placement


def load_flight(f: Path) -> tuple[DuelConfig, list[int]]:
    lines = f.read_text().splitlines()
    h = json.loads(lines[0])
    cfg = DuelConfig(SimpleNamespace(main=h["deck0_main"], extra=h["deck0_extra"]),
                     SimpleNamespace(main=h["deck1_main"], extra=h["deck1_extra"]), h["seed"], flags=h["flags"],
                     lp=h["lp"], hand=h["hand"], draw=h["draw"], setup=tuple(Placement(*p) for p in h["setup"]),
                     shuffle=False)  # fmt: skip
    return cfg, [int(x) for x in lines[1:]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("replay", "inter", "fuzz"))
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument(
        "--verbose", action="store_true", help="fuzz: print each game's decks and seed before playing it"
    )
    ap.add_argument("--release", action="store_true", help="use the normal build instead of the ASan one")
    ap.add_argument(
        "--trace", type=int, default=-1, help="fuzz: print every choice of this game (to find a hang)"
    )
    a = ap.parse_args()
    eng = Engine()
    if a.mode == "replay":
        for f in a.files:
            cfg, ch = load_flight(f)
            env = CppEnv(eng, cfg)
            env.reset()
            for i in ch:
                env.step(i)
            print(f"ok {f} {len(ch)}", flush=True)
    elif a.mode == "inter":
        games = []
        for f in a.files:
            cfg, ch = load_flight(f)
            env = CppEnv(eng, cfg)
            env.reset()
            games.append((env, ch))
        for k in range(max(len(ch) for _, ch in games)):
            for env, ch in games:
                if k < len(ch):
                    env.step(ch[k])
        print(f"inter ok {len(games)} games", flush=True)
    else:
        decks = [
            p for d in ("pool", "newcards", "gauntlet") for p in sorted((ROOT / "decks" / d).glob("*.ydk"))
        ]
        rng = random.Random(a.seed)
        steps = 0
        for g in range(a.games):
            d0, d1 = rng.sample(decks, 2)
            seed = rng.randrange(1 << 31)
            if a.verbose:
                print(f"game {g}: {d0.stem} vs {d1.stem} seed {seed}", flush=True)
            env = CppEnv(eng, DuelConfig(load_ydk(d0), load_ydk(d1), seed=seed))
            obs = env.reset()
            for _ in range(a.max_steps):
                if obs is None:
                    break
                ch = rng.randrange(int(obs["action_mask"].sum()))
                if g == a.trace:
                    print(f"step {steps} choice {ch}", flush=True)
                obs = env.step(ch)
                steps += 1
            if (g + 1) % 50 == 0:
                print(f"{g + 1}/{a.games} games, {steps} steps", flush=True)
        if steps == 0:
            sys.exit("ALERT asan_check fuzz: 0 steps played")
        print(f"fuzz ok {a.games} games {steps} steps", flush=True)


if __name__ == "__main__":
    main()
