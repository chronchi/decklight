import sys, random
from collections import Counter
from pathlib import Path
ROOT = Path("~/Documents/ygo-projects/ygo-zero"); sys.path.insert(0, str(ROOT / "src"))
from ygo_zero.demos.dataset import iter_samples
from ygo_zero.demos.game import _opts
from ygo_zero.engine.duel import Engine
HT = {23434538: "MaxxC", 14558127: "Ash", 59438930: "Ogre"}
src = sys.argv[1]; files = sorted((ROOT/"data/demos"/src).rglob("*.jsonl")); random.Random(0).shuffle(files)
eng = Engine(); lab = Counter(); kinds = Counter(); ex = []; ws = Counter()
for f in files[:150]:
    for obs, a, w, meta in iter_samples(eng, f):
        if int(obs["global"][5]) != 1 or int(obs["global"][2]) != 1: continue
        o = _opts(obs); kd, code = o[a]
        if kd == 6 and code in HT:
            h = HT[code]; ph = int(obs["global"][3])
            lab[(h, ph, meta["label"])] += 1; ws[(h, meta['label'])] += w; kinds[meta["kind"]] += 1
            if len(ex) < 6 and h == "MaxxC": ex.append((meta["line_id"], meta["k"], meta["label"], w))
for k, v in sorted(lab.items()): print(k, v)
print(kinds); print(ws); print(*ex, sep="\n")
