"""Screenshot the phone-first web client against mock states.

Serves web/ with a tiny static server, opens ?mock=1&i=N at 412x915
(device scale 2, isMobile) saving one PNG per mock to
data/cache/web-screens/, plus a 360x780 small-phone variant of 02
and one desktop shot at 1280x800.

Fails (exit 1) on any page error or console error. Prints measured
card width and board-to-hand gap at both phone sizes.
"""

import functools
import http.server
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "data" / "cache" / "web-screens"

MOCKS = [
    ("01-lobby", 0),
    ("02-your-main-phase", 1),
    ("03-chain-prompt", 2),
    ("04-select-cards", 3),
    ("05-bot-thinking", 4),
    ("06-battle", 5),
    ("07-position", 6),
    ("08-game-over", 7),
    ("09-gy-sheet", 8),
    ("10-chain-3", 9),
    ("11-focus", "1&focus=ym2"),
    ("12-menu", "1&menu=yh2"),
    ("13-lobby-picker", "0&picker=you"),
    ("14-select-picked", "3&sel=1,3"),
    ("15-gamemenu", "1&gamemenu=1"),
    ("16-reveal", "1&reveal=bm1"),
    ("17-duel-stats", 16),
    ("18-duel-early", 17),
    ("19-winchance-board", "18&winchance=1"),
    ("20-winchance-stats", "19&winchance=1"),
    ("21-winchance-menu", "1&gamemenu=1&winchance=1"),
    # Practice + hints + what-if (step 4): the lobby switch preselected to
    # Practice, chain/grid/idle prompts with hint badges, the what-if banner.
    ("49-practice-lobby", "0&play=practice"),
    ("50-hint-prompt", "2&hint=1"),
    ("51-hint-cards", "3&hint=1"),
    ("52-hint-idle", "1&hint=1"),
    ("53-whatif-banner", "1&whatif=1"),
    # Rules in context + undo (step 5, items 6 and 9): card text with tappable
    # keywords, the keyword popover over it, the Rule notes switch, the practice
    # prompt bar with Hint + Undo.
    ("58-keywords-detail", "1&detailcode=14558128"),
    ("59-kw-popover", "1&detailcode=14558128&kw=quick-effect"),
    ("60-rule-notes-menu", "1&gamemenu=1"),
    ("61-practice-undo", "1&hint=1&canundo=1"),
    # Puzzles (step 5, item 8): the lobby list grouped by difficulty, the in-puzzle
    # banner, the Solved dialog and the Not-solved dialog.
    ("63-puzzle-list", "0&puzzles=1"),
    ("64-puzzle-banner", "1&puzzlebanner=1"),
    ("65-puzzle-solved", "1&puzzlesolved=1"),
    ("66-puzzle-failed", "1&puzzlefailed=1"),
    # Opponent deck primer (step 5, item 7): lobby sheet, not-enough-games
    # sheet, and the in-game sheet from the game menu.
    ("67-primer-lobby", "0&primer=1"),
    ("68-primer-empty", "0&primer=null"),
    ("69-primer-game", "1&primer=1"),
    # Setting-up status right after Start (the start button disabled, no second new_game).
    ("70-starting", "0&starting=1"),
]

# Pacing cues frozen mid-scene: ?cue=K&t=F freezes visible scene K at fraction F
# (web/cues.js). Indices follow planSteps() over each mock's cue list.
CUE_SHOTS = [
    ("20-cue-activate", "10&cue=0&t=0.6"),
    ("21-cue-target", "10&cue=1&t=0.6"),
    ("22-cue-summon-gy", "10&cue=2&t=0.55"),
    ("23-cue-chain2", "11&cue=3&t=0.6"),
    ("24-cue-chain3", "11&cue=4&t=0.6"),
    ("25-cue-resolve", "11&cue=5&t=0.6"),
    ("26-cue-negated", "11&cue=9&t=0.8"),
    ("27-cue-turn", "12&cue=0&t=0.5"),
    ("28-cue-materials", "12&cue=3&t=0.5"),
    ("29-cue-link-summon", "12&cue=4&t=0.6"),
    ("30-cue-phase", "13&cue=0&t=0.5"),
    ("31-cue-attack", "13&cue=1&t=0.75"),
    ("32-cue-destroyed", "13&cue=2&t=0.5"),
    ("33-cue-damage", "13&cue=3&t=0.45"),
    ("34-cue-set", "14&cue=0&t=0.5"),
    ("35-cue-added", "14&cue=1&t=0.6"),
    ("36-cue-confirm", "14&cue=2&t=0.6"),
    ("37-cue-mill", "14&cue=4&t=0.5"),
    # A cue with a glossary rule note (chain link 2 on the activate scene).
    ("62-cue-rule-note", "11&cue=3&t=0.6"),
]


# Duel-log sheet tabs (mock 17 carries realistic stats): Log, Stats (top and
# scrolled to bottom) and Cards, opened with ?log=1&logtab=.
LOG_SHOTS = [
    ("39-log-tab", "16&log=1&logtab=log"),
    ("40-stats-tab", "16&log=1&logtab=stats"),
    ("41-stats-bottom", "16&log=1&logtab=stats"),
    ("42-cards-tab", "16&log=1&logtab=cards"),
    # Win chance on (mock 20): the Stats tab with the dashed kincha line.
    ("44-winchance-line", "19&log=1&logtab=stats&winchance=1"),
    # Post-game review (mock 08 game over + mock/review.json): summary + list,
    # the mistakes filter, the decision board, and the game-over Review button.
    # 45-47 mirror the real flow (Review tapped: the game-over card dismissed,
    # the sheet open on Review); 48 is the game-over card itself.
    ("45-review-tab", "7&log=1&logtab=review&review=1"),
    ("46-review-mistakes", "7&log=1&logtab=review&review=1&reviewfilter=mistakes"),
    ("47-review-frame", "7&log=1&logtab=review&review=1&reviewframe=1"),
    ("48-gameover-review", "7&review=1"),
]


# Small-phone variants of the cue scenes whose lines must read at 360 px.
CUE_SHOTS_360 = [
    ("26-small-cue-negated", "11&cue=9&t=0.8"),
    ("31-small-cue-attack", "13&cue=1&t=0.75"),
    ("21-small-cue-target", "10&cue=1&t=0.6"),
]

# Contrast of the negation mark, from screenshot pixels. Two clips of the stamp's box, one with
# the mark hidden; pixels that change are the mark (light) and its casing (dark). WCAG relative
# luminance on the medians: mark vs the dimmed card under it, and mark vs its adjacent casing.
NEG_CONTRAST_JS = """async ([a, b]) => {
  const load = async (b64) => { const img = new Image(); img.src = 'data:image/png;base64,' + b64;
    await img.decode(); const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0); return g.getImageData(0, 0, c.width, c.height).data; };
  const A = await load(a), B = await load(b);
  const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const L = (d, i) => 0.2126 * lin(d[i]) + 0.7152 * lin(d[i + 1]) + 0.0722 * lin(d[i + 2]);
  const changed = [];
  for (let i = 0; i < A.length; i += 4) {
    const dd = Math.abs(A[i] - B[i]) + Math.abs(A[i + 1] - B[i + 1]) + Math.abs(A[i + 2] - B[i + 2]);
    if (dd > 30) changed.push(i);
  }
  if (changed.length < 50) return {error: 'only ' + changed.length + ' changed pixels'};
  changed.sort((x, y) => L(A, x) - L(A, y));
  const n = changed.length, mark = changed.slice(Math.floor(n * 0.8)), casing = changed.slice(0, Math.floor(n * 0.2));
  const med = (d, idx) => { const v = idx.map((i) => L(d, i)).sort((x, y) => x - y); return v[v.length >> 1]; };
  const rgb = (d, idx) => { const i = idx[idx.length >> 1]; return [d[i], d[i + 1], d[i + 2]]; };
  const lm = med(A, mark), lc = med(A, casing), lb = med(B, mark);
  const lbMax = mark.map((i) => L(B, i)).sort((x, y) => x - y)[Math.floor(mark.length * 0.95)];
  const cr = (x, y) => (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
  return {mark_vs_card: +cr(lm, lb).toFixed(2), mark_vs_casing: +cr(lm, lc).toFixed(2),
          mark_vs_lightest_card: +cr(lm, lbMax).toFixed(2),
          mark_rgb: rgb(A, mark), card_rgb: rgb(B, mark), casing_rgb: rgb(A, casing), pixels: n};
}"""


def neg_contrast(page) -> dict:
    """Pixel-measured contrast of the frozen negation stamp (see NEG_CONTRAST_JS)."""
    import base64

    box = page.evaluate("""() => { const m = document.querySelector('#ov-cues .negmark circle.case');
      if (!m) return null; const r = m.getBoundingClientRect();
      return {x: Math.max(0, r.x - 4), y: Math.max(0, r.y - 4), width: r.width + 8, height: r.height + 8}; }""")
    if not box:
        return {"error": "no negation stamp on screen"}
    shown = page.screenshot(clip=box)
    page.evaluate("""() => { const s = document.createElement('style'); s.id = 'neg-probe';
      s.textContent = '#ov-cues .negmark{visibility:hidden}'; document.head.appendChild(s); }""")
    page.wait_for_timeout(50)
    hidden = page.screenshot(clip=box)
    page.evaluate("() => document.getElementById('neg-probe').remove()")
    return page.evaluate(
        NEG_CONTRAST_JS, [base64.b64encode(shown).decode(), base64.b64encode(hidden).decode()]
    )


REVIEW_TOPMOST_JS = """() => {
  // The game-over card must not cover the review: at the review list's centre
  // the topmost element has to be inside the drawer (or the decision board).
  const list = document.querySelector('#logdrawer .rlist');
  if (!list) return 'no-rlist';
  if (document.querySelector('#gameover')) return 'gameover-present';
  const r = list.getBoundingClientRect();
  const x = Math.min(window.innerWidth - 1, Math.max(0, r.left + r.width / 2));
  const y = Math.min(window.innerHeight - 1, Math.max(0, r.top + r.height / 2));
  const el = document.elementFromPoint(x, y);
  if (!el) return 'no-element';
  if (el.closest('#reviewframe')) return 'ok-frame';
  return el.closest('#logdrawer') ? 'ok' : 'covered-by:' + (el.id || el.className || el.tagName);
}"""


def review_topmost(page) -> str:
    """Topmost check for the review shots (see REVIEW_TOPMOST_JS)."""
    return page.evaluate(REVIEW_TOPMOST_JS)


HINT_TOPMOST_JS = """() => {
  // Every VISIBLE hint badge's centre must land on its own option (never
  // covered). Badges on board cards behind an open sheet are covered by
  // design (the sheet itself is modal), so only the sheet's badges count
  // then; badges clipped away by an overflow container do not count either.
  const sheet = document.querySelector('#sheet:not(.pilesheet)');
  const vis = (r) => r.width > 2 && r.height > 2 && r.bottom > 0 && r.top < innerHeight;
  let badges = [...document.querySelectorAll('.hintbadge')].filter((b) => vis(b.getBoundingClientRect()));
  if (sheet) badges = badges.filter((b) => sheet.contains(b));
  badges = badges.filter((b) => {
    const clip = b.closest('#info');
    if (!clip) return true;
    const r = b.getBoundingClientRect(), c = clip.getBoundingClientRect();
    return r.top >= c.top - 1 && r.bottom <= c.bottom + 1;
  });
  if (!badges.length) return 'no-badges';
  const bad = [];
  for (const b of badges) {
    const r = b.getBoundingClientRect();
    const x = Math.min(window.innerWidth - 1, Math.max(0, r.left + r.width / 2));
    const y = Math.min(window.innerHeight - 1, Math.max(0, r.top + r.height / 2));
    const el = document.elementFromPoint(x, y);
    if (!el) { bad.push('no-element'); continue; }
    const host = b.closest('[data-opt], .card, .hintbar, .hintlist, .phasebtns');
    if (el === b || (host && (host.contains(el) || el.contains(host) || el === host))) continue;
    bad.push(el.id || el.className || el.tagName);
  }
  return bad.length ? 'covered-by:' + bad.slice(0, 3).join(',') : 'ok:' + badges.length;
}"""


WHATIF_TOPMOST_JS = """() => {
  // The what-if banner and every "Try another move" button must be uncovered:
  // at each centre the topmost element stays inside the banner / the dialog.
  const out = [];
  const banner = document.querySelector('#whatifbanner');
  if (banner) {
    const r = banner.getBoundingClientRect();
    const el = document.elementFromPoint(
      Math.min(window.innerWidth - 1, Math.max(0, r.left + r.width / 2)),
      Math.min(window.innerHeight - 1, Math.max(0, r.top + r.height / 2)));
    if (!el || (!banner.contains(el) && el !== banner)) out.push('banner-covered-by:' + (el && (el.id || el.className || el.tagName)));
  }
  const btns = [...document.querySelectorAll('#reviewframe .wopt')].filter((b) => {
    const r = b.getBoundingClientRect();
    return r.width > 2 && r.height > 2 && r.bottom > 0 && r.top < innerHeight;
  });
  for (const b of btns) {
    const r = b.getBoundingClientRect();
    const el = document.elementFromPoint(
      Math.min(window.innerWidth - 1, Math.max(0, r.left + r.width / 2)),
      Math.min(window.innerHeight - 1, Math.max(0, r.top + r.height / 2)));
    if (!el || !(el === b || b.contains(el) || el.contains(b))) out.push('wopt-covered-by:' + (el && (el.id || el.className || el.tagName)));
  }
  if (!banner && !btns.length) return 'none-present';
  return out.length ? out.slice(0, 3).join(',') : 'ok';
}"""


def hint_topmost(page) -> str:
    """Topmost check for hint badges (see HINT_TOPMOST_JS)."""
    return page.evaluate(HINT_TOPMOST_JS)


def whatif_topmost(page) -> str:
    """Topmost check for the what-if banner and option list (see WHATIF_TOPMOST_JS)."""
    return page.evaluate(WHATIF_TOPMOST_JS)


KW_TOPMOST_JS = """() => {
  // The keyword popover and the Undo button must be uncovered: at each centre
  // the topmost element stays inside the popover / on the button.
  const out = [];
  const pop = document.querySelector('#kwpop');
  if (pop) {
    const r = pop.getBoundingClientRect();
    const el = document.elementFromPoint(
      Math.min(innerWidth - 1, Math.max(0, r.left + r.width / 2)),
      Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2)));
    if (!el || !(el === pop || pop.contains(el) || el.contains(pop))) {
      out.push('popover-covered-by:' + (el && (el.id || el.className || el.tagName)));
    }
  }
  const undoBtn = document.querySelector('[data-act="undo"]');
  if (undoBtn) {
    const r = undoBtn.getBoundingClientRect();
    const el = document.elementFromPoint(
      Math.min(innerWidth - 1, Math.max(0, r.left + r.width / 2)),
      Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2)));
    if (!el || !(el === undoBtn || undoBtn.contains(el) || el.contains(undoBtn))) {
      out.push('undo-covered-by:' + (el && (el.id || el.className || el.tagName)));
    }
  }
  return out.length ? out.slice(0, 3).join(',') : 'ok';
}"""


def kw_topmost(page) -> str:
    """Topmost check for the keyword popover and the Undo button (KW_TOPMOST_JS)."""
    return page.evaluate(KW_TOPMOST_JS)


PUZZLE_TOPMOST_JS = """() => {
  // Every puzzle row's centre must land on its own row (never covered).
  const rows = [...document.querySelectorAll('#puzzlesheet .prow')].filter((b) => {
    const r = b.getBoundingClientRect();
    return r.width > 2 && r.height > 2 && r.bottom > 0 && r.top < innerHeight;
  });
  if (!rows.length) return 'none-present';
  const bad = [];
  for (const b of rows) {
    const r = b.getBoundingClientRect();
    const el = document.elementFromPoint(
      Math.min(window.innerWidth - 1, Math.max(0, r.left + r.width / 2)),
      Math.min(window.innerHeight - 1, Math.max(0, r.top + r.height / 2)));
    if (!el || !(el === b || b.contains(el) || el.contains(b))) bad.push(el && (el.id || el.className || el.tagName));
  }
  return bad.length ? 'covered-by:' + bad.slice(0, 3).join(',') : 'ok';
}"""


def puzzle_topmost(page) -> str:
    return page.evaluate(PUZZLE_TOPMOST_JS)


PUZZLE_BANNER_TOPMOST_JS = """() => {
  const b = document.querySelector('#puzzlebanner');
  if (!b) return 'none-present';
  const r = b.getBoundingClientRect();
  const el = document.elementFromPoint(
    Math.min(window.innerWidth - 1, Math.max(0, r.left + r.width / 2)),
    Math.min(window.innerHeight - 1, Math.max(0, r.top + r.height / 2)));
  if (!el || !(el === b || b.contains(el) || el.contains(b))) return 'banner-covered-by:' + (el && (el.id || el.className || el.tagName));
  return 'ok';
}"""


def puzzle_banner_topmost(page) -> str:
    return page.evaluate(PUZZLE_BANNER_TOPMOST_JS)


PUZZLE_DIALOG_TOPMOST_JS = """() => {
  const d = document.querySelector('#puzzleresult');
  if (!d) return 'none-present';
  const r = d.getBoundingClientRect();
  const el = document.elementFromPoint(
    Math.min(window.innerWidth - 1, Math.max(0, r.left + r.width / 2)),
    Math.min(window.innerHeight - 1, Math.max(0, r.top + r.height / 2)));
  if (!el || !(el === d || d.contains(el) || el.contains(d))) return 'dialog-covered-by:' + (el && (el.id || el.className || el.tagName));
  return 'ok';
}"""


def puzzle_dialog_topmost(page) -> str:
    return page.evaluate(PUZZLE_DIALOG_TOPMOST_JS)


PRIMER_TOPMOST_JS = """() => {
  // The primer sheet's close button and its last section must be uncovered.
  // Scroll the body first so the last section is in view even on a short phone.
  const sheet = document.querySelector('#primersheet');
  if (!sheet) return 'none-present';
  const out = [];
  const check = (el, label) => {
    if (!el) { out.push(label + '-missing'); return; }
    const r = el.getBoundingClientRect();
    const x = Math.min(innerWidth - 1, Math.max(0, r.left + r.width / 2));
    const y = Math.min(innerHeight - 1, Math.max(0, r.top + r.height / 2));
    const top = document.elementFromPoint(x, y);
    if (!top || !(top === el || el.contains(top) || top.contains(el))) {
      out.push(label + '-covered-by:' + (top && (top.id || top.className || top.tagName)));
    }
  };
  check(sheet.querySelector('[data-act="closeprimer"]'), 'close');
  const body = sheet.querySelector('.mbody');
  const heads = sheet.querySelectorAll('.mbody h3');
  if (body) body.scrollTop = body.scrollHeight;
  const last = heads.length ? heads[heads.length - 1] : (body ? body.querySelector(':scope > :last-child') : null);
  check(last, 'last-section');
  return out.length ? out.slice(0, 3).join(',') : 'ok';
}"""


def primer_topmost(page) -> str:
    return page.evaluate(PRIMER_TOPMOST_JS)


GAMEOVER_GAP_JS = """() => {
  // Vertical gap between the game-over dialog's two button rows.
  const rows = [...document.querySelectorAll('#gameover .row')];
  if (rows.length < 2) return 'need-2-rows:' + rows.length;
  const a = rows[0].getBoundingClientRect(), b = rows[1].getBoundingClientRect();
  return b.top - a.bottom;
}"""


def gameover_gap(page):
    """Gap between the two #gameover .row button rows (see GAMEOVER_GAP_JS)."""
    return page.evaluate(GAMEOVER_GAP_JS)


DTILE_OVERFLOW_JS = """() => {
  // Every stat tile's content must fit: no tile scrolls horizontally.
  return [...document.querySelectorAll('.dtile')].map((el) => ({
    cls: (el.querySelector('.dk') || {}).textContent || '?',
    sw: el.scrollWidth, cw: el.clientWidth,
  }));
}"""


def dtile_overflow(page) -> list:
    """Per-tile scrollWidth/clientWidth pairs (see DTILE_OVERFLOW_JS)."""
    return page.evaluate(DTILE_OVERFLOW_JS)


def check_no_dtile_overflow(page, name, errors) -> None:
    for t in dtile_overflow(page):
        if t["sw"] > t["cw"] + 1:
            errors.append(
                f"{name}: stat tile overflows ({t['cls'].strip()}: scroll {t['sw']} > client {t['cw']})"
            )


def launch_kwargs():
    """Point at the pre-installed Chromium (brief: already in ~/.cache/ms-playwright)."""
    env = os.environ.get("CHROME_EXE")
    if env:
        return {"executable_path": env}
    ms = Path.home() / ".cache" / "ms-playwright"
    for cand in sorted(ms.glob("chromium-*/chrome-linux*/chrome"), reverse=True):
        return {"executable_path": str(cand)}
    return {}


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(WEB))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


MEASURE_JS = """() => {
  const r = (el) => el ? el.getBoundingClientRect() : null;
  const round = (x) => Math.round(x * 10) / 10;
  const main = r(document.querySelector('.field.you [data-row="mzone"] .slot:not(.side)'));
  const side = r(document.querySelector('.field.you [data-row="szone"] .slot.side'));
  const hc = r(document.querySelector('#hand .hcard'));
  // Vertical blocks top to bottom; dead gap = space between consecutive blocks.
  const blocks = ['.pbar.bot', '#board', '.pbar.you', '#info', '#hand'].map((q) => [q, r(document.querySelector(q))]);
  const gaps = [];
  for (let i = 0; i + 1 < blocks.length; i++) gaps.push(round(blocks[i + 1][1].top - blocks[i][1].bottom));
  // Empty space inside the hand block above the cards (minus the 10px lift room)
  // and below the last feed row in the info panel.
  const hand = r(document.querySelector('#hand'));
  const handSlack = hc ? round(hc.top - hand.top - 14) : -1;
  const evs = document.querySelectorAll('#info .ev');
  const info = r(document.querySelector('#info'));
  const infoSlack = evs.length ? round(info.bottom - evs[evs.length - 1].getBoundingClientRect().bottom) : -1;
  const docW = document.documentElement.scrollWidth;
  return {boardW: main && round(main.width), sideW: side && round(side.width), handW: hc && round(hc.width),
    handH: hc && round(hc.height), infoH: info && round(info.height), gaps, handSlack, infoSlack,
    overflowX: docW > window.innerWidth, bottom: round(hand.bottom), vh: window.innerHeight};
}"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    httpd, port = serve()
    base = f"http://127.0.0.1:{port}/index.html"
    numbers: dict[str, object] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs())
            # Phone shots, one per mock.
            ctx = browser.new_context(
                viewport={"width": 412, "height": 915}, device_scale_factor=2, is_mobile=True
            )
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: errors.append(f"console-{m.type}: {m.text}") if m.type in ("error",) else None,
            )
            for name, i in MOCKS:
                page.goto(f"{base}?mock=1&i={i}", wait_until="networkidle")
                page.wait_for_timeout(400)
                page.screenshot(path=str(OUT / f"{name}.png"))
                print(f"shot {name}.png", flush=True)
                if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                    errors.append(f"horizontal overflow at 412px in {name}")
                if name == "02-your-main-phase":
                    numbers["m412"] = page.evaluate(MEASURE_JS)
                if name == "15-gamemenu" and not page.evaluate("!!document.querySelector('#gamemenu')"):
                    errors.append("15-gamemenu has no game menu")
                if name == "21-winchance-menu":
                    if not page.evaluate("!!document.querySelector('#gamemenu')"):
                        errors.append("21-winchance-menu has no game menu")
                    if not page.evaluate(
                        "/Show win chance/.test(document.querySelector('#gamemenu').textContent)"
                    ):
                        errors.append("21-winchance-menu: no win-chance switch")
                if name == "19-winchance-board" and not page.evaluate(
                    "!!document.querySelector('#winchance')"
                ):
                    errors.append("19-winchance-board has no win-chance bar")
                if (
                    name == "19-winchance-board"
                    and page.evaluate(
                        "document.querySelector('#winchance') && "
                        "document.querySelector('#winchance').getAttribute('aria-label')"
                    )
                    != "Win chance 62%"
                ):
                    errors.append("19-winchance-board: bar aria-label is not 'Win chance 62%'")
                if name == "16-reveal" and not page.evaluate("!!document.querySelector('#reveal')"):
                    errors.append("16-reveal has no reveal overlay")
                if name == "49-practice-lobby" and not page.evaluate(
                    "!!document.querySelector('[data-play=\\'practice\\'].sel')"
                ):
                    errors.append("49-practice-lobby: Practice is not selected")
                if name in ("50-hint-prompt", "51-hint-cards", "52-hint-idle"):
                    if not page.evaluate("!!document.querySelector('.hintbadge')"):
                        errors.append(f"{name}: no hint badges on screen")
                    if not page.evaluate("!!document.querySelector('[data-act=hint]')"):
                        errors.append(f"{name}: no Hint button on screen")
                    top = hint_topmost(page)
                    if not top.startswith("ok:"):
                        errors.append(f"{name}: hint badges not topmost at their centre ({top})")
                if name == "53-whatif-banner":
                    if not page.evaluate("!!document.querySelector('#whatifbanner')"):
                        errors.append("53-whatif-banner: no what-if banner on screen")
                    top = whatif_topmost(page)
                    if top != "ok":
                        errors.append(f"53-whatif-banner: what-if banner covered ({top})")
                if name == "58-keywords-detail" and not page.evaluate(
                    "document.querySelectorAll('#detail .dtext .kw').length"
                ):
                    errors.append("58-keywords-detail: no tappable keywords in the card text")
                if name == "59-kw-popover":
                    if not page.evaluate("!!document.querySelector('#kwpop')"):
                        errors.append("59-kw-popover: no rule popover on screen")
                    if not page.evaluate("!!document.querySelector('#detail')"):
                        errors.append("59-kw-popover: the card detail closed under the popover")
                    top = kw_topmost(page)
                    if top != "ok":
                        errors.append(f"59-kw-popover: popover covered ({top})")
                if name == "60-rule-notes-menu":
                    if not page.evaluate(
                        "/Rule notes/.test(document.querySelector('#gamemenu').textContent)"
                    ):
                        errors.append("60-rule-notes-menu: no Rule notes switch")
                    if not page.evaluate("!!document.querySelector('[data-act=rulenotes]')"):
                        errors.append("60-rule-notes-menu: no Rule notes toggle control")
                if name == "61-practice-undo":
                    if not page.evaluate("!!document.querySelector('[data-act=undo]')"):
                        errors.append("61-practice-undo: no Undo button on screen")
                    if not page.evaluate(
                        "!!document.querySelector('.hintbar [data-act=hint] ~ [data-act=undo]')"
                    ):
                        errors.append("61-practice-undo: no Hint button next to Undo")
                    if page.evaluate(
                        "!!document.querySelector('.phasebtns [data-act=hint], .phasebtns [data-act=undo]')"
                    ):
                        errors.append("61-practice-undo: Hint/Undo back in the bar (no room at phone width)")
                    top = kw_topmost(page)
                    if top != "ok":
                        errors.append(f"61-practice-undo: Undo button covered ({top})")
                if name == "63-puzzle-list":
                    if not page.evaluate("!!document.querySelector('#puzzlesheet')"):
                        errors.append("63-puzzle-list: no puzzles sheet on screen")
                    if page.evaluate("document.querySelectorAll('#puzzlesheet .prow').length") < 3:
                        errors.append("63-puzzle-list: fewer than 3 puzzle rows")
                    if not page.evaluate("/Solved/.test(document.querySelector('#puzzlesheet').textContent)"):
                        errors.append("63-puzzle-list: no Solved mark on screen")
                    top = puzzle_topmost(page)
                    if top != "ok":
                        errors.append(f"63-puzzle-list: puzzle rows not topmost ({top})")
                if name == "64-puzzle-banner":
                    if not page.evaluate("!!document.querySelector('#puzzlebanner')"):
                        errors.append("64-puzzle-banner: no puzzle banner on screen")
                    top = puzzle_banner_topmost(page)
                    if top != "ok":
                        errors.append(f"64-puzzle-banner: puzzle banner covered ({top})")
                if name in ("65-puzzle-solved", "66-puzzle-failed"):
                    if not page.evaluate("!!document.querySelector('#puzzleresult')"):
                        errors.append(f"{name}: no puzzle result dialog on screen")
                    if name == "65-puzzle-solved" and not page.evaluate(
                        "/Solved in/.test(document.querySelector('#puzzleresult').textContent)"
                    ):
                        errors.append("65-puzzle-solved: no 'Solved in' text")
                    if name == "66-puzzle-failed" and not page.evaluate(
                        "/Not solved/.test(document.querySelector('#puzzleresult').textContent)"
                    ):
                        errors.append("66-puzzle-failed: no 'Not solved' text")
                    top = puzzle_dialog_topmost(page)
                    if top != "ok":
                        errors.append(f"{name}: puzzle dialog not topmost ({top})")
                if name in ("67-primer-lobby", "68-primer-empty", "69-primer-game"):
                    if not page.evaluate("!!document.querySelector('#primersheet')"):
                        errors.append(f"{name}: no primer sheet on screen")
                    sheet = page.evaluate("document.querySelector('#primersheet').textContent")
                    if name in ("67-primer-lobby", "69-primer-game"):
                        for want in (
                            "Key cards",
                            "How it opens",
                            "Your answers",
                            "pts a game",
                            "first in",
                            "Hand traps and negates in your deck",
                        ):
                            if want not in sheet:
                                errors.append(f"{name}: no {want!r} in the sheet")
                        if " → " in sheet:
                            errors.append(f"{name}: old ' → ' opener lines are back")
                        if "used on" not in sheet:
                            errors.append(f"{name}: no 'used on' in Your answers")
                        if "— hit " in sheet:
                            errors.append(f"{name}: old '— hit' wording is back")
                        # Hidden: not validated on held-out games (user, 2026-10-18).
                        if "Where it gets stopped" in sheet:
                            errors.append(f"{name}: 'Where it gets stopped' is back")
                        if "Ash Blossom & Joyous Spring" not in sheet:
                            errors.append(f"{name}: no matching hand trap in Your answers")
                        if "win_rate" in sheet or "Win rate" in sheet:
                            errors.append(f"{name}: bot win rate leaked into the sheet")
                    if name == "68-primer-empty":
                        if "Not enough bot games with this deck yet." not in sheet:
                            errors.append("68-primer-empty: no not-enough-games line")
                        if page.evaluate("document.querySelectorAll('#primersheet h3').length"):
                            errors.append("68-primer-empty: empty sheet still shows sections")
                    if name == "69-primer-game" and page.evaluate("!!document.querySelector('#gamemenu')"):
                        errors.append("69-primer-game: the game menu is still open under the primer")
                    top = primer_topmost(page)
                    if top != "ok":
                        errors.append(f"{name}: primer close/last section not topmost ({top})")
                    if name in ("67-primer-lobby", "69-primer-game"):
                        # A card name opens the card detail; Escape returns to the sheet.
                        page.locator("#primersheet .pname").first.click()
                        page.wait_for_timeout(300)
                        if not page.evaluate("!!document.querySelector('#detail')"):
                            errors.append(f"{name}: tapping a card name opened no detail")
                        # A late card-text arrival re-renders the lobby: the detail
                        # must stay open (it used to close itself the first time).
                        page.evaluate("window.render && window.render()")
                        page.wait_for_timeout(1500)
                        if not page.evaluate("!!document.querySelector('#detail')"):
                            errors.append(f"{name}: the detail closed itself on re-render")
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(300)
                        if page.evaluate("!!document.querySelector('#detail')"):
                            errors.append(f"{name}: Escape did not close the card detail")
                        if not page.evaluate("!!document.querySelector('#primersheet')"):
                            errors.append(f"{name}: the primer sheet closed with the detail")
                        if not page.evaluate(
                            "!!(document.activeElement && document.activeElement.closest('#primersheet'))"
                        ):
                            errors.append(f"{name}: focus left the primer sheet")
                    if name == "67-primer-lobby":
                        # Opened by a real tap (not the query hook): focus moves in,
                        # Escape closes, and focus returns to the About button.
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(250)
                        if page.evaluate("!!document.querySelector('#primersheet')"):
                            errors.append("67-primer-lobby: Escape did not close the primer sheet")
                        page.locator('[data-act="primer"]').click()
                        page.wait_for_timeout(300)
                        if not page.evaluate("!!document.querySelector('#primersheet')"):
                            errors.append("67-primer-lobby: About this deck opened no sheet")
                        if not page.evaluate(
                            "!!(document.activeElement && document.activeElement.closest('#primersheet'))"
                        ):
                            errors.append("67-primer-lobby: focus did not move into the sheet")
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(300)
                        if page.evaluate("!!document.querySelector('#primersheet')"):
                            errors.append("67-primer-lobby: Escape did not close the primer sheet")
                        if not page.evaluate(
                            "!!(document.activeElement && document.activeElement.getAttribute('data-act') === 'primer')"
                        ):
                            errors.append("67-primer-lobby: focus did not return to the About button")
                if name == "01-lobby":
                    if page.evaluate("document.body.innerText.includes('ygo-zero')"):
                        errors.append("01-lobby: old name still visible in the lobby text")
                    if page.evaluate("document.title.includes('ygo-zero')"):
                        errors.append("01-lobby: old name still in the page title")
                    if not page.evaluate("!!document.querySelector('.hero .mark svg')"):
                        errors.append("01-lobby: no decklight mark in the hero")
                if name == "08-game-over":
                    gap = gameover_gap(page)
                    if not isinstance(gap, (int, float)) or gap < 8:
                        errors.append(f"08-game-over: button rows touch (gap {gap})")
                if name in ("17-duel-stats", "20-winchance-stats"):
                    # Board shots carry this mock's stats: the drawer states
                    # below (40/44) assert the tiles; here just check nothing
                    # already overflows in the DOM, if tiles are present.
                    check_no_dtile_overflow(page, name, errors)
                if name == "70-starting":
                    if not page.evaluate("!!document.querySelector('#starting')"):
                        errors.append("70-starting: no loading screen after Start")
                    if not page.evaluate(
                        "/Setting up the duel/.test(document.querySelector('#starting').textContent)"
                    ):
                        errors.append("70-starting: no 'Setting up the duel' kicker")
                    if not page.evaluate(
                        "/decklight \\(Full\\)/.test(document.querySelector('#starting').textContent)"
                    ):
                        errors.append("70-starting: no bot strength/decks line")
                    btn = page.evaluate("""() => { const b = document.querySelector('#startbtn');
                      return b ? {disabled: !!b.disabled, busy: b.getAttribute('aria-busy'),
                                 label: b.textContent.trim()} : null; }""")
                    if not btn or not btn["disabled"] or btn["busy"] != "true":
                        errors.append(f"70-starting: start button not disabled+busy ({btn})")
                    if not btn or btn["label"] != "Starting…":
                        errors.append(f"70-starting: start button label is not 'Starting…' ({btn})")
                    # A second click sends no second new_game: the button is
                    # disabled, so the loading screen just stays put.
                    page.locator("#startbtn").click(force=True)
                    page.wait_for_timeout(300)
                    if not page.evaluate("!!document.querySelector('#starting')"):
                        errors.append("70-starting: a second Start click left the loading screen")
            for name, i in CUE_SHOTS:
                page.goto(f"{base}?mock=1&i={i}", wait_until="networkidle")
                page.wait_for_timeout(250)
                page.screenshot(path=str(OUT / f"{name}.png"))
                print(f"shot {name}.png", flush=True)
                if not page.evaluate(
                    "window.cueFrozen === true && "
                    "document.querySelector('#ov-cues .cue-stage').children.length > 0"
                ):
                    errors.append(f"{name}: no cue scene on screen")
                if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                    errors.append(f"horizontal overflow at 412px in {name}")
                if name == "26-cue-negated":
                    numbers["neg412"] = nc = neg_contrast(page)
                    if "error" in nc or nc["mark_vs_casing"] < 3 or nc["mark_vs_card"] < 3:
                        errors.append(f"26-cue-negated: negation mark contrast below 3:1 ({nc})")
                    if not page.evaluate(
                        "/negated/i.test(document.querySelector('#ov-cues .cue-neg').textContent)"
                    ):
                        errors.append("26-cue-negated: no 'negated' text label")
                if name == "62-cue-rule-note" and not page.evaluate(
                    "document.querySelector('#ov-cues .cue-rule.on') && "
                    "document.querySelector('#ov-cues .cue-rule').textContent.trim().length > 0"
                ):
                    errors.append("62-cue-rule-note: no rule note shown on the cue")
            # Duel-log sheet tabs (Log | Stats | Cards) against the stats mock.
            for name, i in LOG_SHOTS:
                page.goto(f"{base}?mock=1&i={i}", wait_until="networkidle")
                page.wait_for_timeout(400)
                # 48 is the game-over card itself: the drawer stays shut until Review is tapped.
                if name != "48-gameover-review" and not page.evaluate(
                    "!!document.querySelector('#logdrawer')"
                ):
                    errors.append(f"{name}: no log drawer on screen")
                if name == "41-stats-bottom":
                    page.evaluate("document.querySelector('#logdrawer .dbody').scrollTop = 1e6")
                    page.wait_for_timeout(200)
                page.screenshot(path=str(OUT / f"{name}.png"))
                print(f"shot {name}.png", flush=True)
                if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                    errors.append(f"horizontal overflow at 412px in {name}")
                if name == "40-stats-tab" and not page.evaluate(
                    "!!document.querySelector('#logdrawer .dtiles')"
                ):
                    errors.append("40-stats-tab: no stat tiles on screen")
                if name in ("40-stats-tab", "41-stats-bottom"):
                    check_no_dtile_overflow(page, name, errors)
                if name == "42-cards-tab":
                    if not page.evaluate("!!document.querySelector('#logdrawer .dcards')"):
                        errors.append("42-cards-tab: no card lists on screen")
                    # Tapping a cards-tab cell opens that card's details.
                    page.locator("#logdrawer .dcell").first.click()
                    page.wait_for_timeout(300)
                    if not page.evaluate("!!document.querySelector('#detail')"):
                        errors.append("42-cards-tab: tapping a card opened no detail")
                    page.locator("#detail .dx").first.click()
                    page.wait_for_timeout(300)
                if name == "45-review-tab" and not page.evaluate(
                    "!!document.querySelector('#logdrawer .rlist')"
                ):
                    errors.append("45-review-tab: no review list on screen")
                if (
                    name == "46-review-mistakes"
                    and page.evaluate("document.querySelectorAll('#logdrawer .rrow').length") < 2
                ):
                    errors.append("46-review-mistakes: mistakes filter shows fewer than 2 rows")
                if name == "47-review-frame" and not page.evaluate(
                    "!!document.querySelector('#reviewframe')"
                ):
                    errors.append("47-review-frame: no decision board on screen")
                if name == "47-review-frame":
                    if page.evaluate("document.querySelectorAll('#reviewframe .wopt').length") < 2:
                        errors.append("47-review-frame: no 'Try another move' options")
                    top = whatif_topmost(page)
                    if top != "ok":
                        errors.append(f"47-review-frame: what-if list covered ({top})")
                if name in ("45-review-tab", "46-review-mistakes", "47-review-frame"):
                    top = review_topmost(page)
                    if top not in ("ok", "ok-frame"):
                        errors.append(f"{name}: review list not topmost at its centre ({top})")
                if name == "48-gameover-review":
                    if not page.evaluate("!!document.querySelector('#gameover')"):
                        errors.append("48-gameover-review: no game-over modal")
                    if not page.evaluate(
                        "/Review game/.test(document.querySelector('#gameover').textContent)"
                    ):
                        errors.append("48-gameover-review: no Review game button")
                    gap = gameover_gap(page)
                    if not isinstance(gap, (int, float)) or gap < 8:
                        errors.append(f"48-gameover-review: button rows touch (gap {gap})")
            # A negated link stays marked in the chain strip (shape + word, not colour alone).
            neg_html = page.evaluate("""async () => { const m = await import('./render/board.js');
              return m.chainHTML([{card: {name: 'A'}, controller: 'bot'}, {card: {name: 'B'}, controller: 'you'}],
                                 true, new Set([1])); }""")
            if "link bot neg" not in neg_html or "negated" not in neg_html or neg_html.count(' neg"') != 1:
                errors.append("chainHTML does not mark exactly the negated link")
            # Recap at the start of your turn, and a cue list played in real time.
            page.goto(f"{base}?mock=1&i=15&recap=1", wait_until="networkidle")
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / "38-recap.png"))
            print("shot 38-recap.png", flush=True)
            # Tapping a recap line opens that card's details.
            page.locator("#info .recap [data-recap]").first.click()
            page.wait_for_timeout(300)
            if not page.evaluate("!!document.querySelector('#detail')"):
                errors.append("38-recap: tapping a recap line opened no card detail")
            if page.evaluate("document.querySelectorAll('#info .recap .rrow').length") < 4:
                errors.append("38-recap: recap has fewer than 4 lines")
            page.goto(f"{base}?mock=1&i=13", wait_until="networkidle")
            page.wait_for_function(
                "!document.querySelector('#ov-cues').classList.contains('playing')", timeout=15000
            )
            if page.evaluate("document.querySelector('.pbar.you .lp').textContent.trim()") != "6200":
                errors.append("live cue playback did not end on the mock's board (LP != 6200)")
            # Detail overlay via a real tap (field monster -> detail X).
            page.goto(f"{base}?mock=1&i=1", wait_until="networkidle")
            page.wait_for_timeout(400)
            card = page.locator(
                ".field [data-row=mzone] .slot:not(.side) .card[data-uid]:not(.back):not(.actionable)"
            ).first
            if card.count():
                card.click()
                page.wait_for_timeout(300)
            page.screenshot(path=str(OUT / "17-detail-x.png"))
            print("shot 17-detail-x.png", flush=True)
            if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                errors.append("horizontal overflow at 412px in 17-detail-x")
            if not page.evaluate("!!document.querySelector('#detail .dx')"):
                errors.append("17-detail-x has no detail overlay with an X button")
            ctx.close()
            # Small-phone variant of 02.
            ctx3 = browser.new_context(
                viewport={"width": 360, "height": 780}, device_scale_factor=2, is_mobile=True
            )
            s = ctx3.new_page()
            s.on("pageerror", lambda e: errors.append(f"pageerror-small: {e}"))
            s.on(
                "console",
                lambda m: (
                    errors.append(f"console-small-{m.type}: {m.text}") if m.type in ("error",) else None
                ),
            )
            for nm, i in [
                ("02-small", 1),
                ("03-small", 2),
                ("01-small", 0),
                ("07-small", 6),
                ("40-small-stats", "16&log=1&logtab=stats"),
                ("45-small-review", "7&log=1&logtab=review&review=1"),
            ]:
                s.goto(f"{base}?mock=1&i={i}", wait_until="networkidle")
                s.wait_for_timeout(400)
                s.screenshot(path=str(OUT / f"{nm}.png"))
                print(f"shot {nm}.png", flush=True)
                if nm == "02-small":
                    numbers["m360"] = s.evaluate(MEASURE_JS)
                if nm == "40-small-stats" and not s.evaluate(
                    "!!document.querySelector('#logdrawer .dtiles')"
                ):
                    errors.append("40-small-stats: no stat tiles on screen")
                if nm == "40-small-stats":
                    check_no_dtile_overflow(s, nm, errors)
                    # Larger system text: the tiles must still fit at 130%.
                    s.evaluate("document.body.style.zoom = '1.3'")
                    s.wait_for_timeout(300)
                    for t in dtile_overflow(s):
                        if t["sw"] > t["cw"] + 1:
                            errors.append(
                                f"{nm}@130%: stat tile overflows ({t['cls'].strip()}: scroll {t['sw']} > client {t['cw']})"
                            )
                    s.evaluate("document.body.style.zoom = ''")
                    s.wait_for_timeout(200)
                if nm == "45-small-review" and not s.evaluate(
                    "!!document.querySelector('#logdrawer .rlist')"
                ):
                    errors.append("45-small-review: no review list on screen")
                if nm == "45-small-review":
                    top = review_topmost(s)
                    if top not in ("ok", "ok-frame"):
                        errors.append(f"45-small-review: review list not topmost at its centre ({top})")
                if s.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                    errors.append(f"horizontal overflow at 360px in {nm}")
            for nm, i in CUE_SHOTS_360:
                s.goto(f"{base}?mock=1&i={i}", wait_until="networkidle")
                s.wait_for_timeout(250)
                s.screenshot(path=str(OUT / f"{nm}.png"))
                print(f"shot {nm}.png", flush=True)
                if not s.evaluate("window.cueFrozen === true"):
                    errors.append(f"{nm}: no frozen cue scene")
                if s.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
                    errors.append(f"horizontal overflow at 360px in {nm}")
                if nm == "26-small-cue-negated":
                    numbers["neg360"] = nc = neg_contrast(s)
                    if "error" in nc or nc["mark_vs_casing"] < 3 or nc["mark_vs_card"] < 3:
                        errors.append(f"{nm}: negation mark contrast below 3:1 ({nc})")
                numbers.setdefault(
                    "lw360",
                    s.evaluate(
                        "getComputedStyle(document.querySelector('#ov-cues')).getPropertyValue('--cue-lw')"
                    ),
                )
            ctx3.close()
            # Desktop shot of the main-phase board (log drawer open).
            ctx2 = browser.new_context(viewport={"width": 1280, "height": 800})
            d = ctx2.new_page()
            d.on("pageerror", lambda e: errors.append(f"pageerror-desktop: {e}"))
            d.on(
                "console",
                lambda m: (
                    errors.append(f"console-desktop-{m.type}: {m.text}") if m.type in ("error",) else None
                ),
            )
            d.goto(f"{base}?mock=1&i=1", wait_until="networkidle")
            d.wait_for_timeout(400)
            d.screenshot(path=str(OUT / "09-desktop.png"))
            print("shot 09-desktop.png", flush=True)
            # Desktop side panel with the Stats tab (the drawer stays shut on desktop).
            d.goto(f"{base}?mock=1&i=16&logtab=stats", wait_until="networkidle")
            d.wait_for_timeout(400)
            d.screenshot(path=str(OUT / "43-desktop-stats.png"))
            print("shot 43-desktop-stats.png", flush=True)
            if not d.evaluate("!!document.querySelector('#logside .dtiles')"):
                errors.append("43-desktop-stats: no stat tiles in the side panel")
            ctx2.close()
            browser.close()
    finally:
        httpd.shutdown()
    print(f"measure412: {numbers.get('m412')}", flush=True)
    print(f"measure360: {numbers.get('m360')}", flush=True)
    print(f"negation contrast 412: {numbers.get('neg412')}", flush=True)
    print(f"negation contrast 360: {numbers.get('neg360')}", flush=True)
    print(f"cue line weight 360: {numbers.get('lw360')}", flush=True)
    if errors:
        print("ERRORS:", flush=True)
        for e in errors:
            print(f"  {e}", flush=True)
        return 1
    print(
        f"ok: {len(MOCKS) + len(CUE_SHOTS) + len(LOG_SHOTS) + len(CUE_SHOTS_360) + 8} PNGs in {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
