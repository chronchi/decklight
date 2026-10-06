# Game audit (NOTEBOOK 2026-10-06 17:20): engine-free stats over rendered game.txt files.
# Usage: python scripts/audit/selfdmg.py <games dir> [out]
import re, sys, glob, collections as C
G = sys.argv[1]; hdr = re.compile(r"^== Turn (\d+) — LP P0 (\d+), P1 (\d+)")
res = C.defaultdict(lambda: [0,0,0,0])  # turns, turns_with_self_loss, total_loss, turns_with_attack
worst = []
for f in glob.glob(f"{G}/*/*/game.txt"):
    L = open(f).read().splitlines(); ps = int(re.search(r"policy seat P(\d)", L[0]).group(1)); match = L[0].split()[1].rstrip(":")
    body = L[:L.index("--- decisions (non-forced) ---")]
    hs = [(i, *map(int, hdr.match(l).groups())) for i, l in enumerate(body) if hdr.match(l)]
    for (i, t, a0, a1), (j, _, b0, b1) in zip(hs, hs[1:]):
        lines = body[i+1:j]
        att = [l for l in lines if " ATTACK " in l]
        if not att: continue
        tp = int(att[0].split()[1][1])
        who = "pol" if tp == ps else "opp"
        loss = (a0, a1)[tp] - (b0, b1)[tp]
        r = res[(match, who)]; r[0] += 1; r[1] += loss > 0; r[2] += max(loss, 0)
        if who == "pol" and loss >= 1000: worst.append((loss, f, t))
for k in sorted(res): t, n, s, _ = res[k]; print(k, "turns_with_attack", t, "own-LP-loss turns %.3f" % (n/t), "avg loss/turn %.0f" % (s/t))
worst.sort(reverse=True); print(len(worst)); [print(w) for w in worst[:15]]
