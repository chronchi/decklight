# Game audit (NOTEBOOK 2026-10-06 17:20): engine-free stats over rendered game.txt files.
# Usage: python scripts/audit/badatk.py <games dir> [out]
import re, sys, glob, sqlite3, collections as C
G = sys.argv[1]
st = {}
for db in ["third_party/BabelCDB/cards.cdb", "data/cards.cdb"]:
    try:
        for n, a, d, ty in sqlite3.connect(db).execute("select t.name, d.atk, d.def, d.type from texts t join datas d on t.id=d.id"):
            if ty & 1: st.setdefault(n, (a, d))
    except Exception as e: print(db, e)
res = C.defaultdict(C.Counter); ex = []
for f in glob.glob(f"{G}/*/*/game.txt"):
    L = open(f).read().splitlines(); ps = int(re.search(r"policy seat P(\d)", L[0]).group(1)); match = L[0].split()[1].rstrip(":")
    body = L[:L.index("--- decisions (non-forced) ---")]
    for i, l in enumerate(body):
        m = re.match(r"T(\d+) P([01]) ATTACK (.+)$", l)
        if not m: continue
        who = "pol" if int(m.group(2)) == ps else "opp"
        att = m.group(3).strip()
        nxt = body[i+1] if i+1 < len(body) else ""
        t = re.match(rf"T\d+ P{m.group(2)} SELECT (.+)$", nxt)
        r = res[(match, who)]; r["attacks"] += 1
        if not t: r["direct_or_auto"] += 1; continue
        tgt = t.group(1).strip()
        if att not in st or tgt not in st: r["unknown"] += 1; continue
        a = st[att][0]; ta, td = st[tgt]
        if a < 0 or ta < 0: r["unknown"] += 1; continue
        r["vs_monster"] += 1
        if a < min(ta, td): r["loses_both_pos"] += 1; who == "pol" and ex.append((f, l, tgt, a, ta, td))
        elif a <= ta: r["not_above_atk"] += 1
for k in sorted(res):
    r = res[k]; v = max(r["vs_monster"], 1)
    print(k, dict(r), "bad_rate %.3f" % (r["loses_both_pos"] / v))
import random; random.seed(0); [print(e) for e in random.sample(ex, min(12, len(ex)))]
print("--- bad attacks by policy deck")
dk = C.Counter()
for e in ex:
    L = open(e[0]).readline(); dk[re.search(r"\((\S+)\)", L).group(1)] += 1
[print(v, k) for k, v in dk.most_common(8)]
