"""WCAG 2.2 AA check of the web client against its mock states.

Serves web/ (as web_screens.py does) and, per state (lobby, board, sheets, menus, dialogs,
recap, frozen cue scenes; phone 412x915, small phone 360x780, desktop 1280x800, desktop at
200% zoom = 640x400 CSS px, and a phone with large system text = 412 px at 130% zoom):

- axe-core (injected from cdnjs by Playwright; never vendored under web/) with the WCAG 2.0/2.1/
  2.2 A+AA rules; fails on any serious or critical violation;
- a contrast table from computed colours: every visible text style (foreground composited over the
  painted stack under it) against 4.5:1, or 3:1 for large text (>=24 px, or >=18.66 px bold), and
  the meaningful non-text marks (state rings, LP bars, toggles, chain marks, focus ring) against 3:1;
- target sizes: every tappable >= 24x24 CSS px (2.5.8), the primary buttons >= 44;
- keyboard: Tab reaches the actionable cards and the bar buttons, Enter opens, Escape closes
  and focus returns to the opener, a visible >= 2 px focus ring;
- reduced motion: cue scenes become fades (no transforms move), pace stays at the calm default;
- screen reader: the polite live region announces cue text and new prompts.

Exit 1 on any failure. Prints the axe summary, the contrast table and the target report.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_screens import launch_kwargs, serve

AXE_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.12.1/axe.min.js"
AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]

# (name, mock query, viewport) — viewport keys below.
STATES = [
    ("lobby", "0", "phone"),
    ("lobby-picker", "0&picker=you", "phone"),
    ("main-phase", "1", "phone"),
    ("chain-prompt", "2", "phone"),
    ("select-cards", "3", "phone"),
    ("select-picked", "3&sel=1,3", "phone"),
    ("bot-thinking", "4", "phone"),
    ("battle", "5", "phone"),
    ("position", "6", "phone"),
    ("game-over", "7", "phone"),
    ("gy-sheet", "8", "phone"),
    ("chain-3", "9", "phone"),
    ("focus", "1&focus=ym2", "phone"),
    ("action-menu", "1&menu=yh2", "phone"),
    ("game-menu", "1&gamemenu=1", "phone"),
    ("reveal", "1&reveal=bm1", "phone"),
    ("recap", "15&recap=1", "phone"),
    ("cue-activate", "10&cue=0&t=0.6", "phone"),
    ("cue-negated", "11&cue=9&t=0.8", "phone"),
    ("cue-attack", "13&cue=1&t=0.75", "phone"),
    ("cue-turn", "12&cue=0&t=0.5", "phone"),
    ("main-small", "1", "small"),
    ("chain-prompt-small", "2", "small"),
    ("main-desktop", "1", "desktop"),
    ("recap-desktop", "15&recap=1", "desktop"),
    ("main-zoom200", "1", "zoom200"),
    ("chain-prompt-zoom200", "2", "zoom200"),
    ("lobby-zoom200", "0", "zoom200"),
    ("main-bigtext", "1", "bigtext"),
    ("select-bigtext", "3", "bigtext"),
    ("log-tab", "16&log=1&logtab=log", "phone"),
    ("stats-tab", "16&log=1&logtab=stats", "phone"),
    ("cards-tab", "16&log=1&logtab=cards", "phone"),
    ("stats-small", "16&log=1&logtab=stats", "small"),
    ("stats-desktop", "16&logtab=stats", "desktop"),
    # Win chance on: the board bar, the game-menu switch, the Stats line.
    ("winchance-board", "18&winchance=1", "phone"),
    ("winchance-menu", "1&gamemenu=1&winchance=1", "phone"),
    ("winchance-stats", "19&log=1&logtab=stats&winchance=1", "phone"),
    # Post-game review: summary + list, mistakes filter, decision board, game over.
    ("review-tab", "7&log=1&logtab=review&review=1", "phone"),
    ("review-mistakes", "7&log=1&logtab=review&review=1&reviewfilter=mistakes", "phone"),
    ("review-frame", "7&log=1&logtab=review&review=1&reviewframe=1", "phone"),
    ("review-small", "7&log=1&logtab=review&review=1", "small"),
    ("gameover-review", "7&review=1", "phone"),
    # Practice hints + what-if (step 4): lobby switch, badged prompts, banner.
    ("practice-lobby", "0&play=practice", "phone"),
    ("hint-prompt", "2&hint=1", "phone"),
    ("hint-cards", "3&hint=1", "phone"),
    ("hint-idle", "1&hint=1", "phone"),
    ("hint-small", "2&hint=1", "small"),
    ("whatif-banner", "1&whatif=1", "phone"),
    ("review-whatif", "7&log=1&logtab=review&review=1&reviewframe=1", "phone"),
]
VIEWPORTS = {
    "phone": {
        "viewport": {"width": 412, "height": 915},
        "device_scale_factor": 2,
        "is_mobile": True,
        "has_touch": True,
    },
    "small": {
        "viewport": {"width": 360, "height": 780},
        "device_scale_factor": 2,
        "is_mobile": True,
        "has_touch": True,
    },
    "desktop": {"viewport": {"width": 1280, "height": 800}},
    # 200% browser zoom on a 1280x800 window: 640x400 CSS px at device scale 2.
    "zoom200": {"viewport": {"width": 640, "height": 400}, "device_scale_factor": 2},
    # Large system font on a phone (Android Chrome scales the page): 412 px at 130%.
    "bigtext": {
        "viewport": {"width": 317, "height": 704},
        "device_scale_factor": 2.6,
        "is_mobile": True,
        "has_touch": True,
    },
}

# Shared colour helpers + the text contrast scan (runs in the page).
COLOR_JS = r"""
window.__a11y = (() => {
  const parse = (s) => {
    const m = String(s || '').match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  };
  const over = (top, bot, a) => ({r: top.r * a + bot.r * (1 - a), g: top.g * a + bot.g * (1 - a),
                                  b: top.b * a + bot.b * (1 - a), a: 1});
  const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = (c) => 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const opacityOf = (el) => { let o = 1; for (let e = el; e && e.nodeType === 1; e = e.parentElement) o *= +getComputedStyle(e).opacity; return o; };
  const hex = (c) => '#' + [c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');
  // Painted background under (x, y), up to and including `el` (elements above it are ignored).
  const bgAt = (el, x, y) => {
    const stack = document.elementsFromPoint(x, y);
    const i = stack.indexOf(el);
    const below = (i >= 0 ? stack.slice(i) : stack).reverse();
    let c = {r: 20, g: 19, b: 16, a: 1};
    let image = false;
    for (const e of below) {
      const cs = getComputedStyle(e);
      if (e.tagName === 'IMG' && e !== el) image = true;
      if (cs.backgroundImage && cs.backgroundImage !== 'none' && e !== el && !/gradient/.test(cs.backgroundImage)) image = true;
      const b = parse(cs.backgroundColor);
      if (b && b.a > 0) c = over(b, c, b.a * opacityOf(e));
    }
    return {c, image};
  };
  const covered = (el, x, y) => {
    const top = document.elementFromPoint(x, y);
    return !!top && top !== el && !el.contains(top) && !top.contains(el);
  };
  const sig = (el) => {
    const part = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') +
      [...e.classList].filter((k) => !/^(sel|on|new)$/.test(k) || true).slice(0, 3).map((k) => '.' + k).join('');
    const p = el.parentElement;
    return (p && p !== document.body ? part(p) + ' > ' : '') + part(el);
  };
  function textScan() {
    const out = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const seen = new Set();
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!n.nodeValue.trim()) continue;
      const el = n.parentElement;
      if (!el || seen.has(el)) continue;
      seen.add(el);
      if (el.closest('[aria-hidden="true"], .negt, .sr-only, #sr-live, [inert]')) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === 'hidden' || cs.display === 'none') continue;
      const range = document.createRange(); range.selectNodeContents(n);
      const r = range.getBoundingClientRect();
      if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
      const op = opacityOf(el);
      if (op < 0.15) continue;   // fading out / not shown
      const x = Math.min(innerWidth - 1, Math.max(0, r.left + Math.min(r.width / 2, 6))), y = Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2));
      if (covered(el, x, y)) continue;
      const fg0 = parse(cs.color);
      const {c: bg, image} = bgAt(el, x, y);
      const fg = over(fg0, bg, fg0.a * op);
      const size = parseFloat(cs.fontSize), weight = +cs.fontWeight || 400;
      const large = size >= 24 || (size >= 18.66 && weight >= 700);
      const disabled = !!el.closest('[aria-disabled="true"], :disabled');
      out.push({sig: sig(el), text: n.nodeValue.trim().slice(0, 28), fg: hex(fg), bg: hex(bg), size, weight,
                ratio: +ratio(fg, bg).toFixed(2), need: large ? 3 : 4.5, image, disabled});
    }
    return out;
  }
  return {parse, over, ratio, hex, bgAt, opacityOf, textScan};
})();
"""

# Meaningful non-text marks: (label, selector, colour source) against the painted background
# beside them. Colour source: 'bg' (background-color), 'shadow' (first box-shadow colour),
# 'outline', 'stroke', 'border-bottom'.
MARKS = [
    ("actionable card ring", ".card.actionable", "shadow"),
    ("selected card ring", ".card.selected", "shadow"),
    ("focused card ring", ".card.focus", "shadow"),
    ("chain: newest link ring", ".link.new .thumb", "shadow"),
    ("chain: link number ring", ".link.bot .cn", "shadow"),
    ("chain: negated slash", ".link.neg .negx", "bg"),
    ("LP bar (you)", ".pbar.you .lpbar > i", "bg"),
    ("LP bar (bot)", ".pbar.bot .lpbar > i", "bg"),
    ("segmented: selected", ".seg button.sel", "border-bottom"),
    ("game menu seg: selected", "#gamemenu .seg button.sel", "shadow-or-border"),
    ("toggle on: knob", ".sw.on i", "bg"),
    ("toggle off: knob", ".sw:not(.on) i", "bg"),
    ("toggle track outline", ".sw:not(.on)", "shadow"),
    ("option selected border", ".opt.sel", "border-left"),
    ("negated tag mark", ".negtag i", "shadow"),
    ("cue target ring", "#ov-cues .ring", "stroke"),
    ("cue negation mark", "#ov-cues .negmark .mark", "stroke"),
    ("duel tabs: selected underline", ".dtabs [role=tab].on", "shadow"),
]

MARKS_JS = r"""(marks) => {
  const A = window.__a11y; const out = [];
  for (const [label, sel, src] of marks) {
    const el = document.querySelector(sel);
    if (!el) continue;
    const cs = getComputedStyle(el);
    let c = null;
    if (src === 'bg') c = A.parse(cs.backgroundColor);
    else if (src === 'shadow' || src === 'shadow-or-border') {
      // a ring is the shadow layer that carries the state: the strongest colour in the list
      const cols = (cs.boxShadow.match(/rgba?\([^)]+\)/g) || []).map(A.parse);
      const r0 = el.getBoundingClientRect();
      const {c: bg0} = A.bgAt(document.elementFromPoint(Math.max(0, r0.left - 3), r0.top + r0.height / 2) || document.body,
                              Math.max(0, r0.left - 3), r0.top + r0.height / 2);
      cols.sort((a, b) => A.ratio(A.over(b, bg0, b.a), bg0) - A.ratio(A.over(a, bg0, a.a), bg0));
      c = cols[0] || null;
    }
    else if (src === 'border-bottom') c = A.parse(cs.borderBottomColor);
    else if (src === 'border-left') c = A.parse(cs.borderLeftColor);
    else if (src === 'stroke') c = A.parse(cs.stroke);
    if ((!c || c.a === 0) && src === 'shadow-or-border') c = A.parse(cs.borderColor);
    if (!c) { out.push({label, sel, error: 'no colour from ' + src}); continue; }
    const r = el.getBoundingClientRect();
    // background: just outside the element (what the mark sits against)
    const x = Math.max(0, Math.min(innerWidth - 1, r.left - 3)), y = Math.max(0, Math.min(innerHeight - 1, r.top + r.height / 2));
    const top = document.elementFromPoint(x, y) || document.body;
    const {c: bg} = A.bgAt(top, x, y);
    const op = A.opacityOf(el) * (src === 'stroke' ? (+cs.strokeOpacity || 1) : 1);
    const fg = A.over(c, bg, c.a * op);
    out.push({label, sel, fg: A.hex(fg), bg: A.hex(bg), ratio: +A.ratio(fg, bg).toFixed(2)});
  }
  return out;
}"""

TARGETS_JS = r"""() => {
  const sel = 'button, a[href], input, select, textarea, summary, [role=button], [role=radio], [role=switch], [tabindex]:not([tabindex="-1"]), [data-act], [data-opt], .card.actionable, [data-pile], [data-recap], [data-focusuid], [data-detailuid]';
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    if (el.closest('[aria-hidden="true"], [inert]') || el.id === 'scrim' || /scrim$/.test(el.id || '')) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || +cs.opacity === 0 || cs.pointerEvents === 'none') continue;
    // pseudo-element hit areas count (e.g. .pile::after); use the larger box
    let w = r.width, h = r.height;
    for (const ps of ['::before', '::after']) {
      const p = getComputedStyle(el, ps);
      if (p.content && p.content !== 'none' && p.position === 'absolute') {
        const ins = ['top', 'right', 'bottom', 'left'].map((k) => parseFloat(p[k]) || 0);
        w = Math.max(w, r.width - ins[1] - ins[3]); h = Math.max(h, r.height - ins[0] - ins[2]);
      }
    }
    const name = (el.getAttribute('aria-label') || el.textContent || el.className || el.tagName).trim().replace(/\s+/g, ' ').slice(0, 32);
    out.push({name, cls: String(el.className || el.tagName).slice(0, 40), w: Math.round(w), h: Math.round(h)});
  }
  return out;
}"""


def run_axe(page) -> list[dict]:
    if not page.evaluate("!!window.axe"):
        page.add_script_tag(url=AXE_URL)
    res = page.evaluate(
        """async (tags) => {
      const r = await axe.run(document, {runOnly: {type: 'tag', values: tags}, resultTypes: ['violations']});
      return r.violations.map((v) => ({id: v.id, impact: v.impact, help: v.help,
        nodes: v.nodes.slice(0, 4).map((n) => (n.target || []).join(' ') + ' :: ' + (n.failureSummary || '').split('\\n').slice(1, 2).join(''))}));
    }""",
        AXE_TAGS,
    )
    return res


def main() -> int:
    from playwright.sync_api import sync_playwright

    httpd, port = serve()
    base = f"http://127.0.0.1:{port}/index.html"
    errors: list[str] = []
    axe_all: dict[str, list] = {}
    text_rows: dict[tuple, dict] = {}
    mark_rows: dict[str, dict] = {}
    small_targets: dict[str, dict] = {}
    primary_small: dict[str, dict] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            ctxs = {}

            def page_for(vp):
                if vp not in ctxs:
                    ctx = browser.new_context(**VIEWPORTS[vp])
                    p = ctx.new_page()
                    p.on("pageerror", lambda e, vp=vp: errors.append(f"pageerror ({vp}): {e}"))
                    ctxs[vp] = (ctx, p)
                return ctxs[vp][1]

            for name, q, vp in STATES:
                page = page_for(vp)
                page.goto(f"{base}?mock=1&i={q}", wait_until="networkidle")
                page.wait_for_timeout(450)
                page.add_script_tag(content=COLOR_JS)
                if vp in ("zoom200", "bigtext") and page.evaluate(
                    "document.documentElement.scrollWidth > window.innerWidth + 1"
                ):
                    errors.append(f"{name}: horizontal scroll at {vp}")
                viol = run_axe(page)
                axe_all[name] = viol
                for v in viol:
                    if v["impact"] in ("serious", "critical"):
                        errors.append(
                            f"axe {v['impact']} in {name}: {v['id']} ({v['help']}) {v['nodes'][:2]}"
                        )
                for row in page.evaluate("window.__a11y.textScan()"):
                    key = (row["sig"], row["fg"], row["bg"], round(row["size"]))
                    if key not in text_rows or row["ratio"] < text_rows[key]["ratio"]:
                        row["state"] = name
                        text_rows[key] = row
                for row in page.evaluate(MARKS_JS, MARKS):
                    if row["label"] not in mark_rows or row.get("ratio", 0) < mark_rows[row["label"]].get(
                        "ratio", 99
                    ):
                        row["state"] = name
                        mark_rows[row["label"]] = row
                if vp in ("phone", "small", "desktop"):
                    for t in page.evaluate(TARGETS_JS):
                        k = t["cls"] + "|" + t["name"]
                        if min(t["w"], t["h"]) < 24:
                            small_targets.setdefault(k, dict(t, state=name))
                        if (
                            any(p in t["cls"] for p in ("bigbtn", "phbtn", "startbtn", "iconbtn", "dx"))
                            and min(t["w"], t["h"]) < 44
                        ):
                            primary_small.setdefault(k, dict(t, state=name))

            # Detail + log drawer opened from the keyboard: Enter opens, Escape closes, focus returns.
            page = page_for("phone")
            page.goto(f"{base}?mock=1&i=1", wait_until="networkidle")
            page.wait_for_timeout(400)
            kb = keyboard_checks(page, errors)
            page.add_script_tag(content=COLOR_JS)
            for name, opener in (("log-drawer", '[data-act="log"].iconbtn'), ("card-detail", None)):
                page.goto(f"{base}?mock=1&i=1", wait_until="networkidle")
                page.wait_for_timeout(300)
                if opener:
                    page.locator(opener).first.click()
                else:
                    page.locator(".field .card[data-uid]:not(.back):not(.actionable)").first.click()
                page.wait_for_timeout(350)
                viol = run_axe(page)
                axe_all[name] = viol
                for v in viol:
                    if v["impact"] in ("serious", "critical"):
                        errors.append(
                            f"axe {v['impact']} in {name}: {v['id']} ({v['help']}) {v['nodes'][:2]}"
                        )

            # Reduced motion + live region.
            rm = browser.new_context(reduced_motion="reduce", **VIEWPORTS["phone"])
            p = rm.new_page()
            p.on("pageerror", lambda e: errors.append(f"pageerror (reduced): {e}"))
            motion = reduced_motion_checks(p, base, errors)
            rm.close()
            live = live_region_checks(page_for("phone"), base, errors)
            for ctx, _ in ctxs.values():
                ctx.close()
            browser.close()
    finally:
        httpd.shutdown()

    # ---- report ----
    print("\n== axe-core (WCAG 2.0/2.1/2.2 A+AA) ==")
    total = 0
    for name, viol in axe_all.items():
        if viol:
            total += len(viol)
            print(f"  {name}: " + ", ".join(f"{v['id']}[{v['impact']}]x{len(v['nodes'])}" for v in viol))
    print(f"  states: {len(axe_all)}, violations: {total}")

    print("\n== text contrast (computed colours; fg composited over the painted stack) ==")
    print(f"  {'ratio':>6} {'need':>4}  {'fg':7} {'bg':7} {'px':>4}  style  [example · state]")
    fails = 0
    for row in sorted(text_rows.values(), key=lambda r: r["ratio"]):
        bad = row["ratio"] < row["need"] and not row["disabled"]
        fails += bad
        flag = "FAIL" if bad else ("img " if row["image"] else "    ")
        print(
            f"  {row['ratio']:6.2f} {row['need']:4}  {row['fg']} {row['bg']} {row['size']:4.0f}  {flag} {row['sig']}"
            f"  [{row['text']} · {row['state']}]" + ("  (disabled)" if row["disabled"] else "")
        )
        if bad and not row["image"]:
            errors.append(
                f"text contrast {row['ratio']} < {row['need']}: {row['sig']} '{row['text']}' ({row['state']})"
            )
    print(f"  styles: {len(text_rows)}, below threshold: {fails}")

    print("\n== non-text marks (>= 3:1 against what they sit on) ==")
    for row in mark_rows.values():
        if "error" in row:
            print(f"  ----  {row['label']}: {row['error']}")
            continue
        bad = row["ratio"] < 3
        print(
            f"  {row['ratio']:5.2f}  {'FAIL' if bad else '    '} {row['label']:28} {row['fg']} on {row['bg']}  ({row['state']})"
        )
        if bad:
            errors.append(f"mark contrast {row['ratio']} < 3: {row['label']} ({row['state']})")

    print("\n== target sizes ==")
    for t in small_targets.values():
        print(f"  <24: {t['w']}x{t['h']} {t['cls']} '{t['name']}' ({t['state']})")
        errors.append(f"target under 24px: {t['w']}x{t['h']} {t['cls']} '{t['name']}' ({t['state']})")
    for t in primary_small.values():
        print(f"  primary <44: {t['w']}x{t['h']} {t['cls']} '{t['name']}' ({t['state']})")
    print(f"  under 24: {len(small_targets)}, primary under 44: {len(primary_small)}")

    print("\n== keyboard ==\n  " + json.dumps(kb))
    print("== reduced motion ==\n  " + json.dumps(motion))
    print("== live region ==\n  " + json.dumps(live))

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  {e}")
        return 1
    print("\nok: WCAG 2.2 AA checks passed")
    return 0


def keyboard_checks(page, errors) -> dict:
    out: dict[str, object] = {}
    # Tab order reaches the bar buttons and the actionable cards.
    reached = []
    for _ in range(60):
        page.keyboard.press("Tab")
        d = page.evaluate("""() => { const a = document.activeElement; if (!a || a === document.body) return null;
          const cs = getComputedStyle(a); return {cls: String(a.className), act: a.getAttribute('data-act'),
          uid: a.getAttribute('data-uid'), ow: parseFloat(cs.outlineWidth) || 0, os: cs.outlineStyle,
          oc: cs.outlineColor, name: a.getAttribute('aria-label') || a.textContent.trim().slice(0, 20)}; }""")
        if d:
            reached.append(d)
    acts = {r["act"] for r in reached}
    cards = [r for r in reached if "card" in r["cls"].split() and "actionable" in r["cls"]]
    out["tab_stops"] = len(reached)
    out["actionable_cards_reached"] = len(cards)
    out["menu_button"] = "gamemenu" in acts
    out["log_button"] = "log" in acts
    if not cards:
        errors.append("keyboard: Tab never reaches an actionable card")
    if "gamemenu" not in acts or "log" not in acts:
        errors.append("keyboard: Tab does not reach the game menu / log buttons")
    rings = [r for r in reached if r["os"] != "none" and r["ow"] >= 2]
    out["focus_ring_2px"] = f"{len(rings)}/{len(reached)}"
    if len(rings) < len(reached):
        bad = [r["cls"][:30] for r in reached if not (r["os"] != "none" and r["ow"] >= 2)][:5]
        errors.append(f"keyboard: focus ring under 2 px on {bad}")
    # Enter on an actionable card opens its menu; Escape closes and focus returns to the card.
    page.goto(page.url, wait_until="networkidle")
    page.wait_for_timeout(300)
    page.locator(".card.actionable").first.focus()
    uid = page.evaluate("document.activeElement.getAttribute('data-uid')")
    page.keyboard.press("Enter")
    page.wait_for_timeout(250)
    out["enter_opens_menu"] = page.evaluate("!!document.querySelector('#menu')")
    out["focus_in_menu"] = page.evaluate("!!document.activeElement.closest('#menu')")
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)
    out["escape_closes_menu"] = page.evaluate("!document.querySelector('#menu')")
    out["focus_returned_card"] = page.evaluate("document.activeElement.getAttribute('data-uid')") == uid
    # Game menu dialog: Space opens, focus moves in, Escape closes, focus returns.
    page.locator('[data-act="gamemenu"]').first.focus()
    page.keyboard.press(" ")
    page.wait_for_timeout(250)
    out["gamemenu_dialog"] = page.evaluate(
        "(() => { const d = document.querySelector('#gamemenu'); return !!d && d.getAttribute('role') === 'dialog' && d.getAttribute('aria-modal') === 'true'; })()"
    )
    out["focus_in_gamemenu"] = page.evaluate("!!document.activeElement.closest('#gamemenu')")
    # Tab stays inside the modal.
    inside = True
    for _ in range(14):
        page.keyboard.press("Tab")
        inside = inside and page.evaluate("!!document.activeElement.closest('#gamemenu')")
    out["focus_trapped"] = inside
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)
    out["escape_closes_gamemenu"] = page.evaluate("!document.querySelector('#gamemenu')")
    out["focus_returned_menu_btn"] = (
        page.evaluate("document.activeElement.getAttribute('data-act')") == "gamemenu"
    )
    # Card detail from the keyboard (a non-actionable face-up card).
    page.locator(".field .card[role=button]:not(.actionable)").first.focus()
    page.keyboard.press("Enter")
    page.wait_for_timeout(250)
    out["detail_dialog"] = page.evaluate(
        "(() => { const d = document.querySelector('#detail'); return !!d && d.getAttribute('aria-modal') === 'true' && !!document.activeElement.closest('#detail'); })()"
    )
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)
    out["escape_closes_detail"] = page.evaluate("!document.querySelector('#detail')")
    for k, v in out.items():
        if v is False:
            errors.append(f"keyboard: {k} failed")
    return out


def reduced_motion_checks(page, base, errors) -> dict:
    out: dict[str, object] = {}
    page.goto(f"{base}?mock=1&i=11", wait_until="networkidle")
    # Sample cue scene transforms while the chain mock plays: nothing may slide or scale.
    samples = page.evaluate("""async () => {
      const seen = new Map(); let frames = 0;
      const t0 = performance.now();
      while (performance.now() - t0 < 8000) {
        await new Promise((r) => requestAnimationFrame(r));
        const st = document.querySelector('#ov-cues .cue-stage');
        if (!document.querySelector('#ov-cues.playing')) break;
        if (!st || !st.children.length) continue;
        frames++;
        for (const el of st.querySelectorAll('*')) {
          const t = (el.style.transform || '') + '|' + (el.style.left || '') + '|' + (el.style.top || '') + '|' +
            (el.style.width || '') + '|' + (el.style.strokeDashoffset || '') + '|' + (el.getAttribute('d') || '').length;
          if (!seen.has(el)) seen.set(el, new Set());
          seen.get(el).add(t);
        }
      }
      const moving = {};
      for (const [el, v] of seen) if (v.size > 1) {
        const k = (el.getAttribute('class') || el.tagName);
        moving[k] = Math.max(moving[k] || 0, v.size);
      }
      return {frames, moving};
    }""")
    moving = samples["moving"]
    out["frames"] = samples["frames"]
    out["moving_elements"] = moving
    if samples["frames"] == 0:
        errors.append("reduced motion: no cue scene played")
    if moving:
        errors.append(f"reduced motion: cue elements still move: {list(moving)[:6]}")
    out["pace"] = page.evaluate("document.documentElement.dataset.motion || ''")
    return out


def live_region_checks(page, base, errors) -> dict:
    out: dict[str, object] = {}
    page.goto(f"{base}?mock=1&i=11", wait_until="networkidle")
    page.evaluate("""() => { window.__said = []; const r = document.getElementById('sr-live');
      if (!r) return; new MutationObserver((ms) => { for (const m of ms) for (const n of m.addedNodes) {
        const t = (n.textContent || '').trim(); if (t) window.__said.push(t); } })
        .observe(r, {childList: true}); }""")
    page.wait_for_function("!document.querySelector('#ov-cues').classList.contains('playing')", timeout=30000)
    page.wait_for_timeout(300)
    said = page.evaluate("window.__said || []")
    region = page.evaluate("""(() => { const r = document.getElementById('sr-live');
      return r ? {live: r.getAttribute('aria-live'), role: r.getAttribute('role')} : null; })()""")
    out["region"] = region
    out["announcements"] = said[:12]
    if not region or region["live"] != "polite":
        errors.append("live region: no #sr-live with aria-live=polite")
    if not any("activates" in s.lower() and "chain" in s.lower() for s in said):
        errors.append("live region: no 'activates … chain N' announcement during the chain mock")
    if not any("negated" in s.lower() for s in said):
        errors.append("live region: the negation was not announced")
    if page.evaluate("document.getElementById('app').getAttribute('aria-live')"):
        errors.append("live region: #app itself is aria-live (announces every re-render)")
    # A new prompt is announced.
    page.goto(f"{base}?mock=1&i=2", wait_until="networkidle")
    page.wait_for_timeout(500)
    txt = page.evaluate(
        "((document.getElementById('sr-live') || {}).lastElementChild || {}).textContent || ''"
    )
    out["prompt_announcement"] = txt.strip()[:80]
    if not txt.strip():
        errors.append("live region: a new prompt was not announced")
    return out


if __name__ == "__main__":
    sys.exit(main())
