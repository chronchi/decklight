# Audit from docs/reviews/2026-10-09-search-plan-opus.md (Opus 5.5 subagent, 2026-10-09). Run from ygo-zero/: python3 scripts/audit/harness_div.py
import json,glob,collections,os
e2={}
for f in glob.glob('runs/e2-20261007-2129/raw/games-*.jsonl'):
    for l in open(f):
        r=json.loads(l); e2[(r['decks'][0],r['decks'][1],r['seed'])]=r
pt={}
for f in glob.glob('runs/rl5ga-20261008-2015/it0/games/*'):
    f = f+'/state.json' if os.path.isdir(f) else f
    r=json.loads(open(f).readline())
    d=[os.path.basename(x)[:-4] for x in r['decks']]
    pt[(d[0],d[1],r['seed'])]=r
print(len(e2),len(pt),len(set(e2)&set(pt)))
ident=0; div=collections.Counter(); divturn=collections.Counter(); first_pos=[]
ex=[]
for k in set(e2)&set(pt):
    a=e2[k]['choices']; b=pt[k]['choices']
    if a==b: ident+=1; continue
    i=next((j for j in range(min(len(a),len(b))) if a[j]!=b[j]), min(len(a),len(b)))
    s=e2[k]['s_seat']
    actor = e2[k]['actors'][i] if i < len(e2[k]['actors']) else None
    who = 'S(us)' if actor==s else ('ygo-agent' if actor is not None else 'end')
    div[who]+=1; first_pos.append(i)
    if len(ex)<6: ex.append((k,i,who,a[max(0,i-2):i+3],b[max(0,i-2):i+3]))
print('identical',ident)
print('first divergence by actor',div)
fp=sorted(first_pos); print('first-div index quantiles',[fp[int(q*(len(fp)-1))] for q in (0.1,0.25,0.5,0.75,0.9)])
for e in ex: print(e)
print('pt keys',list(pt[next(iter(pt))].keys()))
pairs=collections.Counter(); turnc=collections.Counter(); wl=collections.Counter()
for k in set(e2)&set(pt):
    a=e2[k]['choices']; b=pt[k]['choices']
    if a==b: continue
    i=next((j for j in range(min(len(a),len(b))) if a[j]!=b[j]), None)
    if i is None: continue
    pairs[(a[i],b[i])]+=1
print('(E2 choice, playtest choice) at first divergence',pairs.most_common(12))
