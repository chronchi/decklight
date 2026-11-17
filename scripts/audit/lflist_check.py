#!/usr/bin/env python3
"""Check .ydk lists against the first banlist in data/cache/lflists/0TCG.lflist.conf (EDOPro "2026.10 TCG").

    python3 scripts/audit/lflist_check.py decks/custom/invoked-tcg2026-10.ydk decks/pool/invoked-meta2.ydk

Prints each card over its limit (main + extra + side). Exits 1 when any list is illegal.
"""
import sys,collections,re,sqlite3,glob
lf=open('data/cache/lflists/0TCG.lflist.conf').read().split('\n')
# first list only
lim={};started=False
for l in lf:
    if l.startswith('!'):
        if started: break
        started=True; print('list',l); continue
    m=re.match(r'^(\d+)\s+(-?\d+)',l)
    if m: lim[int(m[1])]=int(m[2])
names={}
for db in glob.glob('third_party/BabelCDB/*.cdb'):
    for c,n in sqlite3.connect(db).execute('select id,name from texts'): names.setdefault(c,n)
rc = False
for f in sys.argv[1:]:
    cnt=collections.Counter(int(x) for x in open(f).read().split() if x.isdigit())
    bad=[(names.get(c,c),n,lim[c]) for c,n in cnt.items() if c in lim and n>lim[c]]
    print(f, "ILLEGAL" if bad else "legal", bad); rc = rc or bool(bad)
sys.exit(1 if rc else 0)
