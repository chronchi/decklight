"""Engine-only replay: for line sources, per record, find inferred ACTIVATE labels at MSG_SELECT_CHAIN (the decisions
a strict verifier would refuse to try) and inferred labels by kind/message. Estimates how many lines would change."""
import sys, random, time, json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ygo_zero.demos.dataset import iter_samples
from ygo_zero.demos.game import _opts
from ygo_zero.engine.duel import Engine
from ygo_zero.engine import constants as C
src, nfiles, tmax = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
files = sorted((ROOT / "data/demos" / src).rglob("*.jsonl")); random.Random(1).shuffle(files); files = files[:nfiles]
eng = Engine(); t0 = time.time()
kinds = Counter(); recs = {}; chain_act = Counter(); st_act = Counter(); n = 0
KN = {}
for f in files:
    if time.time() - t0 > tmax: break
    try:
        for obs, a, w, meta in iter_samples(eng, f):
            n += 1
            o = _opts(obs)
            kd, code = o[a] if a < len(o) else (-1, 0)
            msg = int(obs["global"][21])
            lab = meta["label"]
            kinds[(lab, kd, msg == C.MSG_SELECT_CHAIN)] += 1
            rid = meta["line_id"]
            r = recs.setdefault(rid, [0, 0, 0, 0])
            r[0] += 1
            if lab == "stated" and msg == C.MSG_SELECT_CHAIN and kd == 6:
                st_act[code] += 1
            if lab == "inferred":
                r[1] += 1
                if msg == C.MSG_SELECT_CHAIN and kd == 6:
                    r[2] += 1; chain_act[code] += 1
                    if code not in {42141493,23434538,14558127,84192580,59438930,91800273,24224830,10045474,97268402,94145021,73642296}: r[3] += 1
    except Exception as e:
        print("ERR", f, e, file=sys.stderr)
print(f"## {src}: files scanned, samples {n}, records {len(recs)}, {time.time()-t0:.0f}s")
tot = sum(kinds.values())
for k, v in kinds.most_common(25):
    print(k, v, f"{v/tot:.3f}")
nr = len(recs); w = sum(1 for r in recs.values() if r[2] > 0)
print(f"records with >=1 inferred ACTIVATE at SELECT_CHAIN: {w}/{nr} = {w/max(1,nr):.3f}")
print("top cards:", chain_act.most_common(15))
print("stated chain act:", st_act.most_common(15))
HT={42141493,23434538,14558127,84192580,59438930,91800273,24224830,10045474,97268402,94145021,73642296}
print("inferred chain ACTIVATE share that are hand traps:", sum(v for k,v in chain_act.items() if k in HT)/max(1,sum(chain_act.values())))
w2=sum(1 for r in recs.values() if r[3]>0); print(f"records with >=1 non-handtrap inferred chain ACTIVATE: {w2}/{nr} = {w2/max(1,nr):.3f}")
