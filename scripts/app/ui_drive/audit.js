() => {
  const st = document.createElement('style');
  st.textContent = '* { pointer-events: auto !important; }';
  document.head.appendChild(st);
  const out = [];
  const W = innerWidth, H = innerHeight;
  const name = (e) => {
    if (!e) return '?';
    let s = e.tagName.toLowerCase();
    if (e.id) s += '#' + e.id;
    if (typeof e.className === 'string' && e.className.trim()) s += '.' + e.className.trim().split(/\s+/).slice(0, 2).join('.');
    return s;
  };
  const path = (e) => { const p = []; for (let x = e, i = 0; x && i < 3; x = x.parentElement, i++) p.unshift(name(x)); return p.join(' > '); };
  const visible = (e) => {
    for (let x = e; x; x = x.parentElement) {
      const cs = getComputedStyle(x);
      if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity) < 0.05) return false;
    }
    return true;
  };
  const LAYERS = '#sheet, #ov-cues, #ov-menu, #menu, #gamemenu, #primersheet, #pscrim, #toast, .toast, #layer > *, [role=dialog], #botbanner, #detail, #kwpop';
  const layer = (e) => e.closest(LAYERS) || 'base';
  const inScroller = (e) => { for (let x = e.parentElement; x && x !== document.body; x = x.parentElement) { const c = getComputedStyle(x); if (/auto|scroll/.test(c.overflowY + c.overflowX)) return true; } return false; };
  const walker = document.createTreeWalker(document.getElementById('app') || document.body, NodeFilter.SHOW_TEXT);
  const seen = new Set();
  let tiny = [];
  for (let t = walker.nextNode(); t; t = walker.nextNode()) {
    const txt = t.textContent.trim();
    if (!txt) continue;
    const el = t.parentElement;
    if (!el || seen.has(el)) continue;
    if (el.closest('.sr-only, [aria-hidden="true"] .sr-only, svg')) continue;
    if (el.classList.contains('sr-only')) continue;
    const rg = document.createRange(); rg.selectNodeContents(t);
    const rects = [...rg.getClientRects()].filter(r => r.width > 1 && r.height > 1);
    if (!rects.length || !visible(el)) continue;
    seen.add(el);
    const cs = getComputedStyle(el);
    const fs = parseFloat(cs.fontSize);
    if (fs < 9) tiny.push(txt.slice(0, 20) + '@' + fs.toFixed(1));
    // clipping ancestors
    for (let a = el.parentElement, self = el; a && a !== document.body; a = a.parentElement) {
      const acs = getComputedStyle(a);
      const clips = /hidden|clip/.test(acs.overflowX + acs.overflowY);
      if (!clips) continue;
      const scrolls = /auto|scroll/.test(acs.overflowX + acs.overflowY);
      const masked = (acs.maskImage && acs.maskImage !== 'none') || (acs.webkitMaskImage && acs.webkitMaskImage !== 'none');
      const clamped = acs.webkitLineClamp && acs.webkitLineClamp !== 'none';
      const ell = acs.textOverflow === 'ellipsis';
      const ar = a.getBoundingClientRect();
      for (const r of rects) {
        const inside = r.left >= ar.left - 1 && r.right <= ar.right + 1 && r.top >= ar.top - 1 && r.bottom <= ar.bottom + 1;
        const outside = r.right <= ar.left || r.left >= ar.right || r.bottom <= ar.top || r.top >= ar.bottom;
        if (!inside && !outside && !scrolls && !masked && !clamped && !ell)
          out.push({kind: 'clipped', text: txt.slice(0, 40), el: path(el), by: name(a)});
      }
      break;
    }
    for (const r of rects) {
      if ((r.right > W + 1 || r.left < -1 || r.bottom > H + 1) && !inScroller(el)) { out.push({kind: 'offscreen', text: txt.slice(0, 40), el: path(el)}); break; }
      // probe points inside the line box
      const ys = [r.top + r.height * 0.5];
      const xs = [r.left + 2, r.left + r.width * 0.5, r.right - 2];
      for (const x of xs) for (const y of ys) {
        if (x < 0 || y < 0 || x > W || y > H) continue;
        const top = document.elementFromPoint(x, y);
        if (!top || top === el || el.contains(top) || top.contains(el)) continue;
        if (layer(top) !== layer(el)) continue;
        const tr = top.getBoundingClientRect();
        if (!(x >= tr.left && x <= tr.right && y >= tr.top && y <= tr.bottom)) continue; // pseudo hit-area
        const tcs = getComputedStyle(top);
        const paints = (tcs.backgroundColor !== 'rgba(0, 0, 0, 0)' || tcs.backgroundImage !== 'none' || top.tagName === 'IMG' || (top.textContent || '').trim());
        if (!paints) continue;
        out.push({kind: 'covered', text: txt.slice(0, 40), el: path(el), by: path(top)});
        break;
      }
    }
  }
  st.remove();
  return {issues: out, tiny: tiny.slice(0, 12), ntiny: tiny.length};
}
