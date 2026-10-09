"""ExIt pilot label filter (PLAN §0a "Plan after step 3"): keep only clean, decorrelated search labels.

    python scripts/exit_filter.py --records 'runs/<job>/records/*.jsonl' --out data/demos/exit-p/r0 \
        [--z 2.0] [--per-game 3] [--max-turn 2] [--seed 0]

Teacher records (search_match --save-records, with per-sample scores) are rewritten with every searched decision
that fails the filter turned into "skip" (target None, search entry dropped), so the shard builder never sees it:
- clean: (best - runner-up mean score) / SE >= --z, SE from the per-sample scores (sqrt((var_b + var_r) / n));
- turns: only decisions on turns <= --max-turn (0 = any);
- decorrelated: at most --per-game labels per game, drawn at random (seeded) from the clean ones.
Prints kept/dropped counts by reason and the override share among kept labels. Exits 1 if nothing is kept, 2 if no
searched decision carries samples (records written before per-sample logging).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics as st
import sys
from pathlib import Path


def clean_z(samples: list[list[float]]) -> float:
    """(best - runner-up) / SE of the gap, by candidate means; inf when the SE is 0 and the gap > 0."""
    means = [sum(x) / len(x) for x in samples]
    order = sorted(range(len(means)), key=lambda j: (-means[j], j))
    b, r = order[0], order[1]
    n = min(len(samples[b]), len(samples[r]))
    se = math.sqrt((st.pvariance(samples[b]) + st.pvariance(samples[r])) / max(n, 1))
    gap = means[b] - means[r]
    if se == 0:
        return math.inf if gap > 0 else 0.0
    return gap / se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="glob of teacher record jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--max-turn", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(a.records))
    if not files:
        sys.exit(f"no files match {a.records}")
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    cnt = {"searched": 0, "no_samples": 0, "turn": 0, "unclean": 0, "decorrelate": 0, "kept": 0, "kept_override": 0}
    games = kept_games = 0
    for f in files:
        with open(f) as fh:
            recs = [json.loads(x) for x in fh if x.strip()]
        out = []
        for rec in recs:
            games += 1
            ok = []
            for e in rec.get("search") or []:
                cnt["searched"] += 1
                sm = e.get("samples") or []
                if len(sm) < 2 or not all(sm):
                    cnt["no_samples"] += 1
                    continue
                if a.max_turn and rec["turns"][e["k"]] > a.max_turn:
                    cnt["turn"] += 1
                    continue
                if clean_z(sm) < a.z:
                    cnt["unclean"] += 1
                    continue
                ok.append(e)
            rng.shuffle(ok)
            keep = sorted(ok[: a.per_game], key=lambda e: e["k"])
            cnt["decorrelate"] += len(ok) - len(keep)
            ks = {e["k"] for e in keep}
            for k, lab in enumerate(rec["labels"]):
                if lab != "skip" and k not in ks:
                    rec["labels"][k] = "skip"
                    rec["targets"][k] = None
            rec["search"] = keep
            cnt["kept"] += len(keep)
            cnt["kept_override"] += sum(e["chosen"] != e["played"] for e in keep)
            if keep:
                kept_games += 1
                out.append(rec)
        (a.out / Path(f).name).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"games {games} (with kept labels {kept_games}); " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
    if cnt["searched"] and cnt["no_samples"] == cnt["searched"]:
        print("ALERT exit_filter: no searched decision carries per-sample scores")
        sys.exit(2)
    if cnt["kept"] == 0:
        print("ALERT exit_filter: no label kept")
        sys.exit(1)
    print(f"kept labels: overrides {cnt['kept_override'] / cnt['kept']:.1%}")


if __name__ == "__main__":
    main()
