# Game audit (NOTEBOOK 2026-10-06 17:20): engine-free stats over rendered game.txt files.
# Usage: python scripts/audit/audit.py <games dir> [out]
import re, sys, glob, collections as C
G = sys.argv[1]
dec = re.compile(r"^\[\d+\] T(\d+) P([01]) (\S+?)(?::\S.*?)? (\S+) (.*?) '(.*)' /(\d+)$")
stats = {k: C.Counter() for k in ("pol", "opp")}
runs = []
for f in sorted(glob.glob(f"{G}/*/*/game.txt")):
    txt = open(f).read().splitlines()
    m = re.search(r"policy seat P(\d)", txt[0]); ps = int(m.group(1)); match = txt[0].split()[1].rstrip(":")
    won = "WIN" in txt[0]
    i = txt.index("--- decisions (non-forced) ---")
    turns = C.defaultdict(list)
    for l in txt[i+1:]:
        d = dec.match(l)
        if not d: continue
        t, p, kind, act, card, desc, n = d.groups()
        turns[int(t)].append((int(p), kind, act, card, desc, int(n)))
    for t, ds in turns.items():
        tp = None
        for p, kind, act, card, desc, n in ds:
            if kind == "IDLECMD": tp = p; break
        if tp is None: continue
        who = "pol" if tp == ps else "opp"
        s = stats[who]; s["turns"] += 1
        mine = [x for x in ds if x[0] == tp]
        acts = [x for x in mine if x[1] == "IDLECMD" and x[2] in ("ACTIVATE","SUMMON","SPSUMMON","SSET","MSET")]
        first = next(x for x in mine if x[1] == "IDLECMD")
        if not acts and first[5] >= 3: s["idle_turn_with_options"] += 1; runs.append((match, f, t, who, "did nothing, first idle /%d" % first[5]))
        s["attacks"] += sum(1 for x in mine if x[2] == "ATTACK")
        s["battle_end_noattack"] += sum(1 for x in mine if x[1]=="BATTLECMD" and x[2] in ("TO_END","TO_MAIN") and x[5] > 2)
        s["battlecmd"] += sum(1 for x in mine if x[1]=="BATTLECMD")
        # loops: same (kind,act,card,desc) repeated > 5 in one turn
        cnt = C.Counter((x[1],x[2],x[3],x[4]) for x in mine if x[1] in ("IDLECMD",))
        top = cnt.most_common(1)
        if top and top[0][1] >= 6: s["loopy_turns"] += 1; runs.append((match, f, t, who, f"repeat x{top[0][1]} {top[0][0]}"))
        s["idle_decisions"] += sum(1 for x in mine if x[1]=="IDLECMD")
    for p, kind, act, card, desc, n in [x for ds in turns.values() for x in ds]:
        who = "pol" if p == ps else "opp"
        s = stats[who]
        if kind == "CHAIN": s["chain_"+act] += 1
        if kind == "EFFECTYN": s["yn_"+act] += 1
for who, s in stats.items():
    t = s["turns"]
    print(who, dict(s), "| per turn: idle_dec %.1f attacks %.2f nothing %.3f loopy %.3f | chain act rate %.2f | effyn NO %.3f" % (
        s["idle_decisions"]/t, s["attacks"]/t, s["idle_turn_with_options"]/t, s["loopy_turns"]/t,
        s["chain_ACTIVATE"]/(s["chain_ACTIVATE"]+s["chain_PASS"]), s["yn_NO"]/(s["yn_NO"]+s["yn_YES"])))
by = C.Counter((r[0], r[3], r[4].split()[0]) for r in runs)
for k, v in sorted(by.items()): print(k, v)
open(sys.argv[2], "w").write("\n".join("\t".join(map(str, r)) for r in runs))
