# Audit from docs/reviews/2026-10-09-search-plan-opus.md (Opus 5.5 subagent, 2026-10-09). Run from ygo-zero/: python3 scripts/audit/harness_swstats.py
import json,glob,collections,statistics as st
gaps=[];ov=[];n=0;games=0;decks=collections.Counter();turnov=collections.Counter();turnall=collections.Counter()
top2=[]
for f in glob.glob('data/demos/sw-t1/*.jsonl'):
    for line in open(f):
        r=json.loads(line); games+=1
        for s in r['search']:
            sc=s['scores']; n+=1
            srt=sorted(sc,reverse=True)
            g=srt[0]-srt[1]  # top-2 margin
            pj=s['cands'].index(s['played']) if s['played'] in s['cands'] else None
            o=s['chosen']!=s['played']
            m = (max(sc)-sc[pj]) if pj is not None else None
            gaps.append(g); ov.append((o,m,s.get('turn', r['turns'][s['k']]),s['kind']))
print('games',games,'searched decisions',n)
def q(xs,ps=(10,25,50,75,90)):
    xs=sorted(xs); return [round(xs[int(p/100*(len(xs)-1))],3) for p in ps]
print('top2 margin quantiles',q(gaps))
ovm=[m for o,m,t,k in ov if o and m is not None]
print('overrides',len(ovm),'share',round(len(ovm)/n,3))
print('override margin (best - policy pick) quantiles',q(ovm))
for th in (0.02,0.05,0.1,0.2,0.4):
    print(f'overrides with margin>={th}:',sum(m>=th for m in ovm), round(sum(m>=th for m in ovm)/n,3))
t=collections.Counter(); tt=collections.Counter()
for o,m,tu,k in ov:
    key=min(tu,7); tt[key]+=1
    if o and m is not None and m>=0.2: t[key]+=1
print('decisive overrides (>=0.2) by turn', {k:(t[k],tt[k]) for k in sorted(tt)})
