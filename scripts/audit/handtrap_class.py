"""Hand-trap class from reference play, for the stage D per-line drop (retrain plan v2 §Stage D; NOTEBOOK 2026-10-11).

The Lua class in principles.py ("monster with a quick effect usable from the hand") also holds combo pieces that strong
players fire on their own turn (Elfnote Regina, Labrynth Stovie Torbie, Thunder Dragondark). Dropping every line that
fires such a card on its own turn would drop good combo lines. So the class is taken from how the reference players
use each card (ygo-agent teacher games, hand-played gold games, DAgger teacher labels): per card, the share of its
hand activations made on its own turn with no opponent chain link to answer.

- `hand_trap`: ≥ `--min-uses` reference activations and own-turn share ≤ `--max-own` (fired to answer the opponent);
- `combo`: ≥ `--min-uses` and own-turn share ≥ `--min-combo` (fired to build the own turn);
- cards without enough reference uses get the class of their nearest classified cards by card-text vector (k nearest
  with cosine ≥ `--min-sim`, majority), else `unknown` (kept, listed for review).
The reference players waste too (ygo-agent fires Ash at its own effects and Maxx "C" on its own turns 2+), which
inflates real hand traps' own-turn share; the default cut sits in the observed gap between Maxx "C" (0.27) and the
Bystials / Lady Labrynth (0.44, real own-turn use). Purely data: no card is named here.

Output: `--out` (code -> name, class, own, opp, source) and, per `--drops` file of principles.py, the number of lines
the class drops and the line ids (`<drops stem>.final.jsonl`).

    uv run python scripts/audit/handtrap_class.py --refs data/cache/audit/principles-{teacher_big,gold,dagger}.json \
        --drops data/cache/audit/drops-{muse,claude,guides}.jsonl --out data/cache/audit/handtraps.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OWN = ("ht_own_t1", "ht_own_t2p")
OPP = ("ht_opp", "ht_own_answer")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", type=Path, required=True)
    ap.add_argument("--drops", nargs="*", type=Path, default=[])
    ap.add_argument("--min-uses", type=int, default=5)
    ap.add_argument("--max-own", type=float, default=0.35)  # the gap: Maxx "C" 0.27 .. Bystials 0.44 (NOTEBOOK 10-11 22:30)
    ap.add_argument("--min-combo", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-sim", type=float, default=0.8)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from ygo_zero.cards.embed import card_texts

    texts = card_texts()
    own, opp = Counter(), Counter()
    for f in args.refs:
        cc = json.loads(f.read_text()).get("card_counts")
        if cc is None:
            print(f"ALERT handtrap_class: {f} has no card_counts (rerun principles.py)")
            return 2
        for m in OWN:
            own.update({int(c): v for c, v in cc.get(m, {}).items()})
        for m in OPP:
            opp.update({int(c): v for c, v in cc.get(m, {}).items()})
    cls: dict[int, dict] = {}
    for c in set(own) | set(opp):
        n = own[c] + opp[c]
        if n < args.min_uses:
            continue
        share = own[c] / n
        k = "hand_trap" if share <= args.max_own else "combo" if share >= args.min_combo else "mixed"
        cls[c] = {"class": k, "own": own[c], "opp": opp[c], "source": "reference"}
    if not cls:
        print("ALERT handtrap_class: no card reaches --min-uses in the references")
        return 2

    # cards fired in the audited sources but without reference evidence -> nearest classified cards by text vector
    name_code: dict[str, int] = {}
    for c, (nm, _) in texts.items():
        name_code.setdefault(nm, c)
    fired: dict[str, Counter] = {}
    for f in args.drops:
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        fired[f.stem] = Counter({nm: 1 for r in rows for nm in r["cards"]})
    z = np.load(args.emb)
    index = {int(c): i for i, c in enumerate(z["codes"])}
    vec = z["card"].astype(np.float32)
    known = [c for c in cls if c in index and cls[c]["class"] != "mixed"]
    kv = vec[[index[c] for c in known]] if known else np.zeros((0, vec.shape[1]), np.float32)
    for nm in sorted({nm for cnt in fired.values() for nm in cnt}):
        c = name_code.get(nm)
        if c is None or c in cls:
            continue
        entry = {"class": "unknown", "own": 0, "opp": 0, "source": "none"}
        if c in index and len(known):
            sims = kv @ vec[index[c]]
            top = np.argsort(-sims)[: args.k]
            top = [t for t in top if sims[t] >= args.min_sim]
            if top:
                votes = Counter(cls[known[t]]["class"] for t in top)
                k, nv = votes.most_common(1)[0]
                if nv * 2 > len(top):
                    entry = {"class": k, "own": 0, "opp": 0,
                             "source": "knn:" + ",".join(f"{texts[known[t]][0]}={sims[t]:.2f}" for t in top)}  # fmt: skip
        cls[c] = entry
    out = {str(c): {"name": texts.get(c, (str(c), ""))[0], **v} for c, v in sorted(cls.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    by = Counter(v["class"] for v in out.values())
    print(f"classes: {dict(by)}")
    for k in ("hand_trap", "combo", "mixed", "unknown"):
        names = [f"{v['name']} ({v['own']}/{v['opp']}{'' if v['source'] in ('reference', 'none') else ' knn'})"
                 for v in out.values() if v["class"] == k]  # fmt: skip
        print(f"{k}: {', '.join(names[:40])}")
    ht = {v["name"] for v in out.values() if v["class"] == "hand_trap"}
    for f in args.drops:
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        drop = [r for r in rows if any(nm in ht for nm in r["cards"])]
        kept = Counter(nm for r in rows if r not in drop for nm in r["cards"])
        f.with_suffix(".final.jsonl").write_text("".join(json.dumps(r) + "\n" for r in drop))
        print(f"{f.stem}: {len(drop)} of {len(rows)} flagged lines dropped; kept (not hand traps): "
              f"{kept.most_common(8)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
