# Audit from docs/reviews/2026-10-09-search-plan-opus.md (Opus 5.5 subagent, 2026-10-09). Run from ygo-zero/: python3 scripts/audit/harness_pairh.py
import json,glob,collections
e2={}
for f in glob.glob('runs/e2-20261007-2129/raw/games-*.jsonl'):
    for l in open(f):
        r=json.loads(l)
        s=r['s_seat']; sd=r['decks'][s]; od=r['decks'][1-s]
        first = (s==0)
        w = 1.0 if r['winner']==s else (0.5 if r['winner']==2 else 0.0)
        e2[(sd,od,r['seed'],first)]=(w,len(r['choices']),r['turns'])
hp={}
for f in glob.glob('runs/e2-20261007-2129/hppo/games-*.jsonl'):
    for l in open(f):
        r=json.loads(l)
        s=r['s_seat']; sd=r['decks'][s]; od=r['decks'][1-s]
        hp[(sd,od,r['seed'],s==0)]=1.0 if r['winner']==s else 0.0
pt={}
for l in open('runs/rl5ga-20261008-2015/it0/games.jsonl'):
    r=json.loads(l)
    pt[(r['deck_a'][:-4],r['deck_b'][:-4],r['seed'],r['a_first'])]=(r['a_win'],r['decisions'],r['turns'])
print(len(e2),len(pt),len(hp))
common=set(e2)&set(pt); print('common',len(common))
if len(common)<700:
    print(list(e2)[:3]); print(list(pt)[:3])
c=collections.Counter()
samelen=0
for k in common:
    a=e2[k][0]; b=pt[k][0]
    c[(a,b)]+=1
    samelen += (e2[k][2]==pt[k][2])
print('E2raw,playtest outcome table',c)
print('E2raw mean',sum(e2[k][0] for k in common)/len(common),'playtest mean',sum(pt[k][0] for k in common)/len(common))
print('same #turns',samelen)
for first in (True,False):
    ks=[k for k in common if k[3]==first]
    a=sum(e2[k][0] for k in ks)/len(ks); b=sum(pt[k][0] for k in ks)/len(ks)
    wl=sum(1 for k in ks if e2[k][0]==1 and pt[k][0]==0); lw=sum(1 for k in ks if e2[k][0]==0 and pt[k][0]==1)
    print('first' if first else 'second', 'E2raw',round(a,3),'playtest',round(b,3),'E2-only wins',wl,'pt-only wins',lw)
from math import comb
n=83+55; k=55
p=2*sum(comb(n,i) for i in range(0,k+1))/2**n; print('sign test p',p)
