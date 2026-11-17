// The play screen as a replay player (docs/plans/export.md): the app itself in an iframe (index.html?replay=embed, the
// same app.js, CSS and card sizing as a duel), fed `games:watch` frames. Watch shows it full screen; exports and the
// chat's stills snapshot it: the document is copied into an SVG image (all its CSS, the card sizes app.js set on
// <html>, every image inlined as a data URL) and drawn on a canvas.

const XHTML = "http://www.w3.org/1999/xhtml";
const art = new Map();   // url -> data URL ("" when it failed: the card shows without its art)

// A fetch that hangs (a stalled connection) must not stall the picture: after ART_MS the card shows without its art,
// and only art that arrived is kept (2026-11-16: the chat's boards stayed on "Drawing the board…" for good).
const ART_MS = 8000;
async function dataUrl(url) {
  if (art.has(url)) return art.get(url);
  let v = "";
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ART_MS);
  try {
    const r = await fetch(url, {signal: ctl.signal});
    if (r.ok) {
      const b = await r.blob();
      v = await new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.onerror = () => res(""); fr.readAsDataURL(b); });
    }
  } catch (e) { /* no art */ } finally { clearTimeout(timer); }
  if (v) art.set(url, v);
  return v;
}

// Every url(...) in a CSS or markup string as a data URL (an SVG image loads nothing from outside).
async function inlineUrls(text, base) {
  const urls = new Map();
  for (const m of text.matchAll(/url\((&quot;|"|')?(.*?)\1\)/g)) {
    const u = m[2].replace(/&amp;/g, "&");
    if (!u.startsWith("data:")) urls.set(m[0], new URL(u, base).href);
  }
  // All at once: each waits at most ART_MS, so a picture waits ART_MS at most, not ART_MS per image.
  const got = await Promise.all([...urls.values()].map(dataUrl));
  [...urls.keys()].forEach((whole, k) => { text = text.split(whole).join(got[k] ? "url(&quot;" + got[k] + "&quot;)" : "none"); });
  return text;
}

export class Player {
  // host: the element the iframe goes in; o.width/o.height: its CSS size (a fixed size for exports and stills, so a
  // file looks the same on every phone); o.hidden: off screen (exports, stills).
  constructor(host, o) {
    o = o || {};
    const f = document.createElement("iframe");
    const t = new URLSearchParams(location.search).get("t");
    f.src = "index.html?replay=embed" + (t ? "&t=" + encodeURIComponent(t) : "");
    f.title = o.title || "Replay";
    f.setAttribute("tabindex", "-1");   // the board only shows; the page around it has the controls
    if (o.hidden) f.setAttribute("aria-hidden", "true");
    f.className = "player" + (o.hidden ? " player-off" : "");
    if (o.width) f.style.width = o.width + "px";
    if (o.height) f.style.height = o.height + "px";
    host.appendChild(f);
    this.frame = f;
    this.css = null;
    this.ready = new Promise((res, rej) => {
      const t0 = Date.now();
      const tick = () => {
        let w = null;
        try { w = f.contentWindow; } catch (e) { /* not loaded */ }
        if (w && w.playerReady) { res(w); return; }
        if (Date.now() - t0 > 20000) { rej(new Error("the replay player did not start: close the replay and open it again")); return; }
        setTimeout(tick, 50);
      };
      tick();
    });
  }
  async load(frames, o) {
    const w = await this.ready;
    w.loadFrames(frames, o || {});
    this.n = frames.length;
  }
  // i: the frame; o: {cues: false} shows the board at once, {cue: k, t: 0..1} freezes cue scene k (see app.js).
  async speed(s) { const w = await this.ready; w.setSpeed(s); }
  async show(i, o) { const w = await this.ready; await w.showFrame(i, o || {cues: false}); }
  async plan(i) { const w = await this.ready; return w.cuePlan(i) || []; }
  destroy() { this.frame.remove(); }

  async styles(doc) {
    if (this.css != null) return this.css;
    const parts = [];
    for (const sh of doc.styleSheets) {
      try { for (const r of sh.cssRules) parts.push(r.cssText); } catch (e) { /* another origin's sheet */ }
    }
    // :root is the SVG inside an image: the app's tokens go on its <html> instead.
    this.css = await inlineUrls(parts.join("\n").replace(/:root\b/g, "html"), doc.baseURI);
    return this.css;
  }

  // The screen as it is now: a canvas `scale` times its CSS size.
  async snap(scale) {
    const w = await this.ready;
    const doc = w.document;
    const W = w.innerWidth, H = w.innerHeight;
    const css = await this.styles(doc);
    const root = doc.documentElement.cloneNode(true);
    for (const el of root.querySelectorAll("script, link, style, meta, title, iframe, noscript")) el.remove();
    const ims = [...root.querySelectorAll("img[src]")];
    const got = await Promise.all(ims.map((im) => dataUrl(new URL(im.getAttribute("src"), doc.baseURI).href)));
    ims.forEach((im, k) => { if (got[k]) im.setAttribute("src", got[k]); else im.removeAttribute("src"); });
    const head = root.querySelector("head") || root.insertBefore(doc.createElementNS(XHTML, "head"), root.firstChild);
    const st = doc.createElementNS(XHTML, "style");
    st.textContent = css;
    head.appendChild(st);
    root.setAttribute("style", (root.getAttribute("style") || "") + ";width:" + W + "px;height:" + H + "px;overflow:hidden");
    let xml = new XMLSerializer().serializeToString(root);
    xml = await inlineUrls(xml, doc.baseURI);
    const cw = Math.round(W * scale), ch = Math.round(H * scale);
    // The image is the player's CSS size (its media queries see that width, not the file's) and the canvas scales it:
    // an SVG image is drawn as vectors at the canvas size.
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '" height="' + H + '"><foreignObject x="0" y="0" width="' + W + '" height="' + H + '">' + xml + "</foreignObject></svg>";
    const img = new Image();
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    await img.decode();
    const c = document.createElement("canvas");
    c.width = cw; c.height = ch;
    const g = c.getContext("2d");
    g.fillStyle = getComputedStyle(doc.body).backgroundColor || "#141310";
    g.fillRect(0, 0, cw, ch);
    g.drawImage(img, 0, 0, cw, ch);
    return c;
  }
}
