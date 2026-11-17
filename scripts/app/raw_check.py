"""Check the Rust raw policy decisions against the torch policy the ONNX was exported from (A2c).

    uv run python scripts/app/raw_check.py --dump data/cache/app/raw_rs.jsonl.gz --seed 31000 \\
        --pairs blue-eyes:branded branded:burn ... [--onnx data/cache/onnx/r1]

Replays every recorded game in PyEnv with the recorded choices, computes the checkpoint's logits at every
decision, and requires the Rust chosen index to equal torch's argmax wherever the top-2 torch gap is >= TIE.
Reports the number of near-ties and the max |dlogit| over legal options.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIE = 1e-4


def resolve_deck(name: str) -> Path:
    p = Path(name)
    if p.is_file():
        return p
    q = ROOT / "decks" / "pool" / f"{name}.ydk"
    return q if q.is_file() else p


def manifest_inputs(onnx: Path) -> tuple[Path, Path]:
    man = json.loads((onnx / "manifest.json").read_text())
    ckpt = emb = None
    for k in man.get("inputs", {}):
        if k.endswith(".pt") and "forge" in k or k.endswith("best.pt") and ckpt is None:
            ckpt = ROOT / k
        if k.endswith(".npz") and emb is None:
            emb = ROOT / k
    if ckpt is None or emb is None:
        raise SystemExit(f"manifest inputs do not name a .pt and a .npz: {man.get('inputs')}")
    return ckpt, emb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--onnx", type=Path, default=ROOT / "data/cache/onnx/r1")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--emb", type=Path, default=None)
    a = ap.parse_args(argv)

    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig, Engine
    from decklight.engine.env import PyEnv
    from decklight.search.product import load_policy

    ckpt, emb = manifest_inputs(a.onnx)
    ckpt = a.ckpt or ckpt
    emb = a.emb or emb
    policy = load_policy(ckpt, emb)

    pairs = []
    for item in a.pairs:
        x, sep, y = item.partition(":")
        if not sep:
            raise SystemExit(f"--pairs entry {item!r} needs deck0:deck1")
        pairs.append((resolve_deck(x), resolve_deck(y)))
    if not pairs:
        raise SystemExit("--pairs is required")
    decks = [(load_ydk(p), load_ydk(q)) for p, q in pairs]
    engine = Engine()

    decisions = games = near_ties = disagree = diverged = 0
    max_dlogit = 0.0
    rows: list[dict] = []
    with gzip.open(a.dump, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    i = 0
    while i < len(rows):
        first = rows[i]
        if first.get("end"):
            i += 1
            continue
        g = int(first["g"])
        block = []
        while i < len(rows) and not rows[i].get("end"):
            block.append(rows[i])
            i += 1
        # rows[i] is the block's end marker
        i += 1
        seed = int(first["seed"])
        d0, d1 = decks[g % len(decks)]
        env = PyEnv(engine, DuelConfig(d0, d1, seed=seed), auto_single=False)
        env.reset()
        games += 1
        for k, row in enumerate(block):
            obs = env._observe()
            n = len(env.opts)
            if n != int(row["n"]) or int(env.player) != int(row["player"]):
                diverged += 1
                print(
                    f"ALERT raw_check: game {g} decision {k}: python n/player {n}/{env.player} "
                    f"vs rust {row['n']}/{row['player']}",
                    file=sys.stderr,
                )
                break
            batch = {key: np.ascontiguousarray(v)[None] for key, v in obs.items()}
            z = np.asarray(policy.logits(batch)[0], dtype=np.float64)[:n]
            legal = np.asarray(obs["action_mask"])[:n].astype(bool)
            got = np.asarray(row["logits"], dtype=np.float64)[:n]
            if legal.any():
                max_dlogit = max(max_dlogit, float(np.abs(z[legal] - got[legal]).max()))
            idx = np.flatnonzero(legal)
            if len(idx) >= 2:
                vals = z[idx]
                top = np.sort(vals)
                if top[-1] - top[-2] < TIE:
                    near_ties += 1
                    nice = True
                else:
                    nice = False
            else:
                nice = False
            want = int(idx[np.argmax(z[idx])]) if len(idx) else -1
            if not nice and int(row["choice"]) != want:
                disagree += 1
                if disagree <= 5:
                    print(
                        f"ALERT raw_check: game {g} decision {k}: rust {row['choice']} != torch {want} "
                        f"(gap {float(np.sort(z[idx])[-1] - np.sort(z[idx])[-2]):.3g})",
                        file=sys.stderr,
                    )
            decisions += 1
            env.step(int(row["choice"]))
        env.duel.close()

    summary = {
        "games": games,
        "decisions": decisions,
        "near_ties": near_ties,
        "disagreements": disagree,
        "diverged": diverged,
        "max_abs_dlogit": max_dlogit,
        "tie": TIE,
        "ok": disagree == 0 and diverged == 0 and decisions > 0,
    }
    print(json.dumps(summary))
    if not summary["ok"]:
        print("ALERT raw_check: raw policy parity failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
