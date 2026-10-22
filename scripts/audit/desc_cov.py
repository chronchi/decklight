"""Engine-only: share of ACTIVATE/YES options (and chain links) whose desc points at a card string (str1-16 of a
card) vs a system string, in teacher games."""
import sys, random
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ygo_zero.demos.dataset import iter_samples
from ygo_zero.engine.duel import Engine
files = sorted((ROOT / "data/demos/teacher").glob("*.jsonl")); random.Random(0).shuffle(files)
eng = Engine(); c = Counter(); n = 0
import numpy as np
z=np.load(ROOT/"data/cards/emb/qwen3-0.6b-d512.npz"); NE=dict(zip(z["codes"].tolist(), z["n_effects"].tolist())); sysn=Counter(); syscodes=Counter()
for f in files[:60]:
    for obs, a, w, meta in iter_samples(eng, f):
        n += 1
        m = int(obs["action_mask"].sum()); A = obs["actions"][:m]
        for r in A:
            k = int(r[0])
            if k not in (6, 12): continue
            code, ec, ei = int(r[2]), int(r[3]), int(r[4])
            if ec == 0:
                c[(k, "system")] += 1
                if k == 6: sysn[min(NE.get(code,0),4)] += 1; syscodes[code] += 1
            elif ec == code: c[(k, "own str")] += 1
            else: c[(k, "other card str")] += 1
        if n > 60000: break
    if n > 60000: break
print(n, c)
print("n_effects of cards with system-desc ACTIVATE:", sorted(sysn.items()))
print(syscodes.most_common(10))
