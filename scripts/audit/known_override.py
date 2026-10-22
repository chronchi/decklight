"""History-review Stage-0 measurement (docs/reviews/2026-10-09-history-arch-opus.md, appendix): at searched decisions of
300 sw-t1 records, how often a remembered opponent card exists (card_known) and whether the override rate differs; how far
back the v1 32-entry history window reaches. Usage: taskset -c 0-3 uv run python scripts/audit/known_override.py 300
"""
import json, glob, random, sys
import numpy as np
from decklight.engine.duel import Engine
from decklight.engine.cpp_env import CppEnv
from decklight.demos.dataset import record_config
random.seed(0)
files = sorted(glob.glob("data/demos/sw-t1/*.jsonl"))
recs = []
for f in files:
    for line in open(f):
        if line.strip():
            recs.append(json.loads(line))
random.shuffle(recs)
recs = recs[: int(sys.argv[1])]
eng = Engine()
rows = []  # per searched decision
allwin = []  # per decision: does v1 window reach previous turn
for rec in recs:
    S = {e["k"]: e for e in rec.get("search", [])}
    env = CppEnv(eng, record_config(rec), rec.get("auto_single", True))
    obs = env.reset()
    for k, c in enumerate(rec["choices"]):
        if obs is None:
            break
        if int(obs["action_mask"].sum()) >= 2:
            hm = obs["history_mask"].astype(bool); ta = obs["history"][hm, 11]
            t = int(obs["global"][2])
            allwin.append((t, int(ta.max()) if len(ta) else 0, int(obs["hist_long_mask"].sum())))
        if k in S:
            e = S[k]
            cards = obs["cards"]; m = obs["card_mask"].astype(bool)
            opp_hand_hidden = int(((cards[:, 1] == 2) & (cards[:, 3] == 1) & (cards[:, 0] == 0) & m).sum())
            known = int((obs["card_known"] == 1).sum())
            sc = e["scores"]; cands = e["cands"]
            best = max(sc)
            pm = best - sc[cands.index(e["played"])] if e["played"] in cands else None
            rows.append(dict(turn=int(obs["global"][2]), kind=e["kind"], ovr=int(e["chosen"] != e["played"]),
                             known=known, ohh=opp_hand_hidden, margin=pm))
        obs = env.step(int(c))
n = len(rows); print("games", len(recs), "searched", n)
kn = [r for r in rows if r["known"] > 0]; nk = [r for r in rows if r["known"] == 0]
hh = [r for r in rows if r["ohh"] > 0]
print("frac searched with >=1 remembered opp hidden card: %.3f" % (len(kn)/n))
print("frac searched with opp hidden hand cards: %.3f; mean hidden %.2f" % (len(hh)/n, np.mean([r['ohh'] for r in rows])))
print("override rate known>0 %.3f (n %d) vs known=0 %.3f (n %d)" % (np.mean([r['ovr'] for r in kn]) if kn else -1, len(kn), np.mean([r['ovr'] for r in nk]), len(nk)))
for lo, hi in ((1,1),(2,2),(3,4),(5,6),(7,99)):
    s = [r for r in rows if lo <= r["turn"] <= hi]
    if s: print(f"turn {lo}-{hi}: n {len(s)} known>0 {np.mean([r['known']>0 for r in s]):.3f} ovr {np.mean([r['ovr'] for r in s]):.3f}")
a = np.array(allwin)
later = a[a[:,0] >= 2]
print("decisions turn>=2:", len(later), " v1 window reaches an earlier turn: %.3f" % np.mean(later[:,1] >= 1),
      " reaches 2+ turns back: %.3f" % np.mean(later[:,1] >= 2), " salient log len median %d p90 %d" % (np.median(later[:,2]), np.percentile(later[:,2],90)))
