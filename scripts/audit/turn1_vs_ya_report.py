"""Reports on scripts/audit/turn1_lines_vs_ya.py output (ours vs ygo-agent going first on the same opening hands).

    python scripts/audit/turn1_vs_ya_report.py <dump.jsonl> decomp|waste|pairs|decks [D1,D2]|read D1,D2
decomp: LP/field/hand at the start of turns 2-5 and win-from-there; waste: disruption fired on our own turn 1;
pairs: hand-matched means; decks: per-deck end boards / plays; read: hand-matched lines. Laptop-safe.
"""
import sys

MODE, sys.argv = sys.argv[2], [sys.argv[0], sys.argv[1]] + sys.argv[3:]

if MODE == "decomp":
    import json, sys, statistics as st
    R = [json.loads(l) for l in open(sys.argv[1])]
    M = lambda xs: round(st.mean(xs), 2) if xs else None
    for P in ("ours", "ya"):
        rs = [r for r in R if r["pilot"] == P]; n = len(rs)
        print(f"== {P} going first (opponent = {'ygo-agent' if P == 'ours' else 'our base_r3'}), n={n}, win {M([r['win'] for r in rs])}")
        end2 = [r for r in rs if r["turns"] <= 2]
        print(f"   game over during turn 2: {len(end2)} (seat0 won {sum(r['win'] for r in end2)})")
        for t in ("2", "3", "4", "5"):
            z = [r for r in rs if t in r["snaps"]]
            if not z: continue
            s = [r["snaps"][t] for r in z]
            print(f"   start T{t}: n={len(z)} LP first {M([x['lp'][0] for x in s])} second {M([x['lp'][1] for x in s])} | field first {M([x['f0'] for x in s])} second {M([x['f1'] for x in s])} | hand first {M([x["h0"] for x in s])} second {M([x["h1"] for x in s])} | win from here {M([r['win'] for r in z])}")
        z = [r for r in rs if "3" in r["snaps"]]
        lost_board = [r for r in z if r["snaps"]["3"]["f0"] <= 1]
        print(f"   board wiped (≤1 card) by start T3: {len(lost_board)}/{len(z)} -> win {M([r['win'] for r in lost_board])}; kept: win {M([r['win'] for r in z if r['snaps']['3']['f0'] > 1])}")
        lp_hit = [r for r in z if r["snaps"]["3"]["lp"][0] <= 4000]
        print(f"   first player LP ≤4000 at T3: {len(lp_hit)}/{len(z)}")

if MODE == "waste":
    import json, sys, statistics as st
    from collections import Counter
    R = [json.loads(l) for l in open(sys.argv[1])]
    HT = ('Maxx "C"', "Ash Blossom", "Effect Veiler", "Nibiru", "Droll & Lock", "Ghost Ogre", "Ghost Belle", "Called by the Grave",
          "Crossout Designator", "Infinite Impermanence", "Mirrorjade the Iceblade Dragon Banish", "Widow Anchor Negate", "Twin Twisters", "Harpie's Feather Duster", "Lightning Storm")
    for P in ("ours", "ya"):
        rs = [r for r in R if r["pilot"] == P]; n = len(rs)
        c = Counter(); g = 0
        for r in rs:
            hit = [h for m in r["t1"] if m.startswith("ACTIVATE") for h in HT if h in m]
            c.update(hit); g += bool(hit)
        held = Counter(h for r in rs for card in r["hand"] for h in HT[:9] if h in card)
        print(P, f"games with a hand-trap/disruption fired on own turn 1: {g}/{n}", dict(c.most_common()))
        print("   in opening hand:", dict(held.most_common()))
        print("   t1 decisions", round(st.mean(len(r["t1"]) for r in rs), 1), " end-of-T1 hand", round(st.mean(r["snaps"]["2"]["h0"] for r in rs), 2) if all("2" in r["snaps"] for r in rs) else "")

if MODE == "pairs":
    import json, sys, statistics as st
    R = [json.loads(l) for l in open(sys.argv[1])]
    O = {(r["deck"], r["seed"], r["opp"]): r for r in R if r["pilot"] == "ours"}
    Y = {}
    for r in R:
        if r["pilot"] == "ya": Y.setdefault((r["deck"], r["seed"]), []).append(r)
    same = diff = 0; pairs = []
    for (d, s, o), r in O.items():
        for y in Y.get((d, s), []):
            if sorted(y["hand"]) == sorted(r["hand"]): same += 1; pairs.append((r, y))
            else: diff += 1
    print("ours-first games with a ya-first game on the same deck+seed:", same + diff, "same opening hand:", same)
    M = lambda xs: round(st.mean(xs), 3)
    print("hand-matched pairs", len(pairs), "| field ours", M([len(r["end"]) for r, _ in pairs]), "ya", M([len(y["end"]) for _, y in pairs]),
          "| T1-end hand ours", M([r["snaps"]["2"]["h0"] for r, _ in pairs]), "ya", M([y["snaps"]["2"]["h0"] for _, y in pairs]),
          "| win ours", M([r["win"] for r, _ in pairs]), "ya", M([y["win"] for _, y in pairs]))
    mc = [(r, y) for r, y in pairs if any('Maxx "C"' in m for m in r["t1"] if m.startswith("ACTIVATE"))]
    print("pairs where ours fired Maxx C on own T1:", len(mc), "win ours", M([r["win"] for r, _ in mc]), "ya", M([y["win"] for _, y in mc]),
          "| ya fired it too:", sum(any('Maxx "C"' in m for m in y["t1"] if m.startswith("ACTIVATE")) for _, y in mc))
    nomc = [(r, y) for r, y in pairs if (r, y) not in mc]
    print("other pairs:", len(nomc), "win ours", M([r["win"] for r, _ in nomc]), "ya", M([y["win"] for _, y in nomc]))

if MODE == "decks":
    import json, sys
    from collections import Counter
    R = [json.loads(l) for l in open(sys.argv[1])]
    for d in sys.argv[2].split(","):
        print("=" * 30, d)
        for P in ("ours", "ya"):
            rs = [r for r in R if r["deck"] == d and r["pilot"] == P]
            e = Counter(c for r in rs for c in set(r["end"])); a = Counter(m.split(" ", 1)[1][:40] for r in rs for m in r["t1"] if m.startswith(("ACTIVATE", "SPSUMMON", "SUMMON")))
            t2 = Counter(m.split(" ", 1)[1][:40] for r in rs for m in r["t2"] if m.startswith("ACTIVATE"))
            print(f"-- {P} n={len(rs)} win {sum(r['win'] for r in rs)/len(rs):.2f} mean end {sum(len(r['end']) for r in rs)/len(rs):.2f}")
            print("   end board (games with):", ", ".join(f"{k} {v}" for k, v in e.most_common(12)))
            print("   t1 plays:", ", ".join(f"{k} {v}" for k, v in a.most_common(14)))
            print("   t2 activations:", ", ".join(f"{k} {v}" for k, v in t2.most_common(8)))

if MODE == "read":
    import json, sys
    from collections import defaultdict
    R = [json.loads(l) for l in open(sys.argv[1])]
    G = defaultdict(lambda: {"ours": [], "ya": []})
    for r in R: G[(r["deck"], r["seed"])][r["pilot"]].append(r)
    KEEP = ("ACTIVATE", "SUMMON", "SPSUMMON", "SSET", "MSET", "SELECT", "TO_END", "YES")
    for (d, s), v in sorted(G.items()):
        if d not in sys.argv[2].split(","): continue
        print(f"##### {d} seed {s} hand: {', '.join(v['ours'][0]['hand'])}")
        for P in ("ours", "ya"):
            r = v[P][0]
            mv = [m[:60] for m in r["t1"] if m.startswith(KEEP)]
            print(f"  {P:4s} wins {sum(x['win'] for x in v[P])}/{len(v[P])}  T1-end hand {r['snaps']['2']['h0']}  end: {', '.join(r['end'])}")
            print("       " + " | ".join(mv))
