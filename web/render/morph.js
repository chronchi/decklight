// Minimal keyed DOM morph: patch container children in place so images,
// focus, scroll and CSS entry animations survive re-renders.
// Children are matched by data-key, else data-uid, else position+tag.
export function morph(container, html) {
  if (!container) return;
  const tpl = document.createElement("template");
  tpl.innerHTML = html || "";
  patchChildren(container, Array.from(tpl.content.childNodes));
}

function keyOf(node) {
  if (!node || node.nodeType !== 1) return null;
  return node.getAttribute("data-key") || node.getAttribute("data-uid") || null;
}

function patchChildren(parent, fresh) {
  const live = Array.from(parent.childNodes);
  const byKey = new Map();
  for (const el of live) {
    if (el.nodeType !== 1) continue;
    const k = keyOf(el);
    if (k) {
      const mk = el.tagName + "\0" + k;
      if (!byKey.has(mk)) byKey.set(mk, el);
    }
  }
  const used = new Set();
  // Unkeyed live elements available for positional matching, per tag.
  const byTag = new Map();
  for (const el of live) {
    if (el.nodeType !== 1 || keyOf(el)) continue;
    const t = el.tagName;
    if (!byTag.has(t)) byTag.set(t, []);
    byTag.get(t).push(el);
  }
  const unkeyedText = live.filter((n) => n.nodeType === 3 && !used.has(n));
  let ti = 0;
  for (const node of fresh) {
    let match = null;
    if (node.nodeType === 1) {
      const k = keyOf(node);
      if (k) {
        const cand = byKey.get(node.tagName + "\0" + k);
        if (cand && !used.has(cand)) match = cand;
      }
      if (!match) {
        // Positional fallback, but never steal a keyed node that a later
        // fresh sibling claims; check that first.
        const pool = byTag.get(node.tagName) || [];
        // Prefer a live node whose key is not claimed by any fresh sibling.
        const claimed = new Set();
        for (const f of fresh) {
          if (f.nodeType === 1) {
            const fk = keyOf(f);
            if (fk) claimed.add(f.tagName + "\0" + fk);
          }
        }
        while (pool.length) {
          const cand = pool[0];
          if (used.has(cand)) { pool.shift(); continue; }
          const ck = keyOf(cand);
          if (ck && claimed.has(cand.tagName + "\0" + ck)) break;
          match = cand;
          pool.shift();
          break;
        }
      }
    } else if (node.nodeType === 3) {
      while (ti < unkeyedText.length) {
        const cand = unkeyedText[ti++];
        if (!used.has(cand)) { match = cand; break; }
      }
    }
    if (match) {
      used.add(match);
      if (match.nodeType === 1 && node.nodeType === 1) morphEl(match, node);
      else if (match.nodeType === 3 && node.nodeType === 3) {
        if (match.nodeValue !== node.nodeValue) match.nodeValue = node.nodeValue;
      }
    } else {
      match = node;
    }
    // Place in order.
    const cur = parent.childNodes[fresh.indexOf(node)] || null;
    if (match.parentNode !== parent) parent.insertBefore(match, cur);
    else if (match !== cur) parent.insertBefore(match, cur);
  }
  for (const el of live) {
    if (!used.has(el) && el.parentNode === parent) parent.removeChild(el);
  }
}

function morphEl(live, fresh) {
  if (live.tagName !== fresh.tagName) {
    live.replaceWith(fresh.cloneNode(true));
    return;
  }
  const isFocusedInput = live.tagName === "INPUT" && document.activeElement === live;
  for (const a of Array.from(fresh.attributes)) {
    if (isFocusedInput && a.name === "value") continue;
    if (live.tagName === "IMG" && a.name === "src" && live.getAttribute("src") === a.value) continue;
    if (live.getAttribute(a.name) !== a.value) live.setAttribute(a.name, a.value);
  }
  for (const a of Array.from(live.attributes)) {
    if (!fresh.hasAttribute(a.name)) {
      if (isFocusedInput && a.name === "value") continue;
      live.removeAttribute(a.name);
    }
  }
  // Never touch an <img> whose src is unchanged, so images never reload.
  if (live.tagName === "IMG") return;
  patchChildren(live, Array.from(fresh.childNodes));
}
