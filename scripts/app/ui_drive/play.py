import os, random, re
MODE = os.environ.get("MODE", "duel"); TAG = os.environ.get("TAG", MODE)
VW, VH = int(os.environ.get("VW", 384)), int(os.environ.get("VH", 775))
SEED = int(os.environ.get("SEED", 1)); MAXS = int(os.environ.get("MAXS", 900))
YOUDECK = os.environ.get("YOUDECK", ""); BOTDECK = os.environ.get("BOTDECK", "")
random.seed(SEED)
AUD = open(os.path.join(os.path.dirname(os.path.abspath("scripts/app/ui_drive/play.py")), "audit.js")).read()
LASTP = {}
def _ws(ws):
    def got(f):
        try:
            m = json.loads(f) if isinstance(f, str) else None
            if m and m.get("t") == "state" and m.get("prompt"): LASTP["p"] = m["prompt"]
        except Exception: pass
    ws.on("framereceived", got)
pg.on("websocket", _ws)
GRIDBAD = "[...document.querySelectorAll('.selectgrid .cbox')].map(b=>{const c=b.querySelector('.card'); if(!c) return 'nocard'; const r=c.getBoundingClientRect(), q=b.getBoundingClientRect(); return (Math.abs(r.top-q.top)>3||Math.abs(r.height-q.height)>3) ? c.className+' '+Math.round(r.width)+'x'+Math.round(r.height)+' vs '+Math.round(q.width)+'x'+Math.round(q.height) : ''}).filter(Boolean)"
nbad = 0
pg.set_viewport_size({"width": VW, "height": VH})
URL = os.environ.get("URL", "http://127.0.0.1:8766/?t=devtok")
pg.goto(URL); time.sleep(3)
if not pg.evaluate("!!document.querySelector('[data-act=lobbytab]')"):
    if pg.evaluate("!!document.querySelector('[data-act=newgame]')"): tap('[data-act=newgame]')
    else: tap('[data-act=gamemenu]'); tap('[data-act=backlobby]')
    time.sleep(2)
def pick(side, want):
    tap(f'[data-pickopen={side}]'); time.sleep(0.8)
    shot(TAG + "_picker_" + side)
    names = pg.evaluate("[...document.querySelectorAll('[data-" + side + "deck]')].filter(b=>b.getBoundingClientRect().width>2).map(b=>'|'+b.textContent.trim().slice(0,30))")
    print(side, "choices:", len(names), names[:6])
    i = next((k for k, n in enumerate(names) if want.lower() in n.lower()), random.randrange(len(names))) if names else None
    if i is not None: tap('[data-pick]', 0) if False else pg.evaluate("i => { const v=[...document.querySelectorAll('.dkpick')].filter(b=>b.getBoundingClientRect().width>2); v[i].scrollIntoView({block:'center'}); v[i].click(); }", i)
    time.sleep(0.6)
if MODE == "puzzle":
    tap('[data-act=lobbytab][data-tab=puzzles]'); time.sleep(1)
    n = pg.evaluate("document.querySelectorAll('#lobby .prow').length")
    tap('#lobby .prow', random.randrange(n))
else:
    tap('[data-act=lobbytab][data-tab=play]')
    if YOUDECK or BOTDECK:
        try:
            if YOUDECK: pick("you", YOUDECK)
            if BOTDECK: pick("bot", BOTDECK)
        except Exception as e: print("pick failed", e)
    tap('[data-play=%s]' % ("practice" if MODE == "practice" else "duel"))
    print("decks:", pg.evaluate("[...document.querySelectorAll('[data-pickopen]')].map(b=>b.textContent.trim())"))
    tap('[data-act=start]')
for i in range(90):
    time.sleep(1)
    if pg.evaluate("!document.querySelector('#starting') && document.querySelectorAll('#hand .hcard, #sheet').length>0"): break
state_js = """() => {
  const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 2 && r.height > 2; };
  const q = (s) => [...document.querySelectorAll(s)].filter(vis);
  return {
    over: !!q('[data-act=newgame], [data-act=pznext], [data-act=pzretry]').length,
    sheetOpts: q('#sheet:not(.pilesheet) [data-opt]:not([disabled])').map(b => b.className + '|' + b.textContent.trim().slice(0, 24)),
    cards: q('.card.actionable').length,
    closers: q('[data-act=closerecap],[data-act=closereveal],[data-act=closekw],[data-act=closedetail],[data-act=closemenu]').map(b => b.getAttribute('data-act')),
    menuOpts: q('#menu [data-opt]:not([disabled])').length,
    phase: q('.phasebtns [data-opt]').map(b => b.textContent.trim()),
    sig: (document.querySelector('.phase, .turn')||{}).innerText + '|' + ((document.querySelector('#sheet .stitle')||{}).innerText||'') ,
    toast: (document.querySelector('#toast, .toast, #banner:not([hidden])')||{}).innerText || '',
  };
}"""
def click(sel, i=0):
    return pg.evaluate("([s,i]) => { const v=[...document.querySelectorAll(s)].filter(e=>{const r=e.getBoundingClientRect();return r.width>2&&r.height>2}); const e=v[i]; if(!e) return null; e.click(); return e.textContent.trim().slice(0,30); }", [sel, i])
issues = {}; sigs = set(); shots = 0; t0 = time.time(); last = None; same = 0; toasts = set(); usedhint = 0; zshots = set()
for n in range(MAXS):
    s = pg.evaluate(state_js)
    a = pg.evaluate(AUD)
    for it in a["issues"]:
        k = (it["kind"], re.sub(r"\d+", "#", it["el"]), re.sub(r"\d+", "#", it.get("by", "")))
        if k not in issues:
            issues[k] = (it, n); shot(f"{TAG}_iss{len(issues):02d}")
            print("ISSUE", len(issues), n, it)
    gb = pg.evaluate(GRIDBAD)
    if gb and nbad < 3:
        nbad += 1; shot(f"{TAG}_gridbad{nbad}"); print("GRIDBAD", gb)
        json.dump(LASTP.get("p"), open(f"{OUT}/{TAG}_gridbad{nbad}.json", "w"), indent=1)
        print(pg.evaluate("document.querySelector('.selectgrid').outerHTML.slice(0,1500)"))
    if s["toast"] and s["toast"] not in toasts: toasts.add(s["toast"]); print("TOAST:", s["toast"][:100])
    if s["sig"] not in sigs and shots < 40:
        sigs.add(s["sig"]); shots += 1; shot(f"{TAG}_s{shots:02d}")
    if s["over"]:
        print("OVER at", n, round(time.time()-t0), "s"); shot(f"{TAG}_over"); break
    act = None
    if MODE == "practice" and usedhint < 4 and pg.evaluate("!!document.querySelector('[data-act=hint]:not([disabled])')") and random.random() < 0.15:
        click('[data-act=hint]'); usedhint += 1
        for k in range(40):
            time.sleep(1)
            if pg.evaluate("!!document.querySelector('.hintlist') || !document.querySelector('.hintwait')"): break
        shot(f"{TAG}_hint{usedhint}"); a = pg.evaluate(AUD)
        for it in a["issues"]: print("HINT-ISSUE", it)
    zp = pg.evaluate("document.querySelectorAll('.zpick').length")
    if zp and n not in zshots:
        zshots.add(n); shot(f"{TAG}_zone{len(zshots)}")
    if s["closers"]: act = ("close", click("[data-act=" + s["closers"][0] + "]"))
    elif zp: act = ("zone", click(".zpick", random.randrange(zp)))
    elif s["sheetOpts"]:
        prim = [i for i, c in enumerate(s["sheetOpts"]) if "primary" in c.split("|")[0]]
        if prim and random.random() < 0.5: act = ("confirm", click("#sheet:not(.pilesheet) [data-opt].primary:not([disabled])"))
        else: act = ("sheet", click("#sheet:not(.pilesheet) [data-opt]:not([disabled])", random.randrange(min(3, len(s["sheetOpts"])))))
    elif s["menuOpts"]: act = ("menu", click("#menu [data-opt]:not([disabled])", random.randrange(s["menuOpts"])))
    elif s["cards"] and random.random() < 0.6: act = ("card", click(".card.actionable", random.randrange(s["cards"])))
    elif s["phase"]:
        act = ("phase", pg.evaluate("() => { const v=[...document.querySelectorAll('.phasebtns [data-opt]')].filter(b=>b.getBoundingClientRect().width>2); const b=v[v.length-1]; b.click(); return b.textContent.trim(); }"))
    key = (s["sig"], act); same = same + 1 if key == last else 0; last = key
    if same > 200: print("STUCK?", s); shot(f"{TAG}_stuck"); break
    time.sleep(0.5 if act else 1.0)
print("steps", n, "time", round(time.time()-t0), "issues", len(issues), "tiny-font sample", a["tiny"][:6], a["ntiny"])
