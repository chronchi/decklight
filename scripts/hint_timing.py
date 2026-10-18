"""Real-net hint timing (brief hints-whatif acceptance).

Loads the product nets (scripts/product.sh paths: P + value_p2), replays the
Mitsurugi fixture to its first 3 multi-option human decisions and prints the
hint (top-3 + estimates) and the wall ms for each. Exits non-zero when a hint
is empty or slower than 3 s (fail fast and loud).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.analysis.hints import compute_hint
from ygo_zero.analysis.review import load_record
from ygo_zero.engine.duel import Engine
from ygo_zero.engine.env import PyEnv
from ygo_zero.eval.winchance import load_calib

FIXTURE = ROOT / "tests" / "fixtures" / "web" / "mitsurugi-20261016"
BUDGET_MS = 3000.0


def main() -> int:
    from ygo_zero.edopro.product_bot import ProductLauncher
    from ygo_zero.search.product import PRODUCT_CKPT, PRODUCT_VALUE, resolve_product_paths

    ckpt_p, value_p, _emb_p = resolve_product_paths()
    assert ckpt_p == PRODUCT_CKPT and value_p == PRODUCT_VALUE, (ckpt_p, value_p)
    t0 = time.monotonic()
    launcher = ProductLauncher(device="cpu")
    print(f"loaded product in {time.monotonic() - t0:.1f}s: {ckpt_p} + {value_p}", flush=True)
    calib = load_calib()

    engine = Engine()
    rec, cfg, hs, _d = load_record(FIXTURE / "game.jsonl")
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    done = 0
    for ch in rec["choices"]:
        if env.done or done >= 3:
            break
        if env.player == hs and len(env.opts) > 1:
            out = compute_hint(engine, cfg, env, hs, launcher, calib)
            if not out["options"]:
                print("ALERT hint_timing: empty hint options", flush=True)
                return 1
            ms = out["ms"]
            opts = ", ".join(f"#{o['index']} p={o['p']:.2f} est={o['est']}" for o in out["options"])
            flag = "" if ms < BUDGET_MS else "  OVER BUDGET"
            print(f"decision {done}: ms={ms:.0f} [{opts}]{flag}", flush=True)
            if ms >= BUDGET_MS:
                print(f"ALERT hint_timing: hint took {ms:.0f} ms >= {BUDGET_MS:.0f} ms", flush=True)
                return 2
            done += 1
        env.step(int(ch))
    if done < 3:
        print(f"ALERT hint_timing: only {done} human multi-option decisions", flush=True)
        return 1
    print(f"ok: 3 hints, all under {BUDGET_MS:.0f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
