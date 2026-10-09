# Audit from docs/reviews/2026-10-09-search-plan-opus.md (Opus 5.5 subagent, 2026-10-09). Run from ygo-zero/: python3 scripts/audit/harness_divr.py
import json,glob,collections,os,sys
sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/audit')
from render_games import _opts, option_desc, H
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.demos.names import CardNames
e2={}
for f in glob.glob('runs/e2-20261007-2129/raw/games-*.jsonl'):
    for l in open(f):
        r=json.loads(l); e2[(r['decks'][0],r['decks'][1],r['seed'])]=r
pt={}
for f in glob.glob('runs/rl5ga-20261008-2015/it0/games/*'):
    f=f+'/state.json' if os.path.isdir(f) else f
    r=json.loads(open(f).readline()); d=[os.path.basename(x)[:-4] for x in r['decks']]
    pt[(d[0],d[1],r['seed'])]=r
eng=Engine(); names=CardNames().names
tal=collections.Counter(); shown=0
for k in sorted(set(e2)&set(pt)):
    a=e2[k]['choices']; st=pt[k]; b=st['choices']
    if a==b: continue
    i=next((j for j in range(min(len(a),len(b))) if a[j]!=b[j]), None)
    if i is None: continue
    env=CppEnv(eng,H._config(st)); obs=env.reset()
    for c in b[:i]: obs=env.step(int(c))
    msg=H._MSGS.get(int(obs['global'][21])); o=_opts(obs)
    def d(c):
        kk,code=o[int(c)]; return f"{H._kind(kk)} {names.get(code,'') if code else ''}"
    tal[(msg, d(a[i]).split()[0], d(b[i]).split()[0])]+=1
    if shown<12:
        shown+=1; print(k, i, msg, '| E2 ygo-agent:', d(a[i]), '| playtest ygo-agent:', d(b[i]), '| n opts', len(o))
for t,c in tal.most_common(12): print(c,t)
