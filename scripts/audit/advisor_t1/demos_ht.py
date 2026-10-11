"""Per BC source: at decisions where ACTIVATE Maxx C (or other hand traps) is offered, label fire rate split by
own turn / opponent turn and phase. Uses dataset.iter_samples (the training samples, with their labels/targets)."""
import sys, random, time
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path("~/Documents/ygo-projects/ygo-zero")
sys.path.insert(0, str(ROOT / "src"))
from ygo_zero.demos.dataset import iter_samples
from ygo_zero.demos.game import _opts
from ygo_zero.engine.duel import Engine
HT = {23434538: "MaxxC", 24224830: "CBTG", 14558127: "Ash", 10045474: "Imperm", 97268402: "Veiler", 94145021: "Droll", 59438930: "Ogre", 73642296: "Belle", 44146295: "Mirrorjade"}
PH = {1: "draw", 2: "standby", 4: "main1", 512: "end"}
src, nfiles, tmax = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
files = sorted((ROOT / "data/demos" / src).rglob("*.jsonl")); random.Random(0).shuffle(files); files = files[:nfiles]
eng = Engine(); st = defaultdict(Counter); t0 = time.time(); nf = 0; ns = 0
for f in files:
    if time.time() - t0 > tmax: break
    nf += 1
    try:
        for obs, a, w, meta in iter_samples(eng, f):
            ns += 1
            o = _opts(obs)
            own = "own" if int(obs["global"][5]) == 1 else "opp"
            t1 = "T1" if int(obs["global"][2]) == 1 else "T2+"
            ph = PH.get(int(obs["global"][3]), "other")
            kd, code = o[a] if a < len(o) else (0, 0)
            for (k2, c2) in set(o):
                if k2 == 6 and c2 in HT:
                    h = HT[c2]
                    key = (h, own, ph if h == "MaxxC" else "*", t1 if own == "own" else "")
                    st[key]["off"] += 1
                    if kd == 6 and code == c2: st[key]["fire"] += 1
    except Exception as e:
        print("ERR", f, e, file=sys.stderr)
print(f"## {src}: files {nf}, samples {ns}")
for k in sorted(st):
    c = st[k]
    if c["off"] >= 5 or c["fire"]:
        print(src, *k, "offered", c["off"], "fired", c["fire"], f"rate {c['fire']/c['off']:.2f}")
