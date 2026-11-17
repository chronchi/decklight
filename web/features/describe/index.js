// "Describe a card" model (docs/design/deck-builder.md, Search): the optional 1.2 GB download that turns on semantic search.
// Settings shows this panel; the deck builder's search asks `ready()`. Server messages: describe_status /
// describe_download / describe_cancel / describe_remove (docs/design/api/cards-and-decks.md). Actions are `dm:*`.
import {esc} from "../../render/card.js";

const GB = 1e9;
const size = (b) => (b >= GB ? (b / GB).toFixed(1) + " GB" : Math.round(b / 1e6) + " MB");

export function describeModel(ctx) {
  // st: the last describe_status ({installed, downloading, bytes, total, error}) or null before the first one.
  let st = null;
  let arm = null;          // "data" (download on mobile data) | "remove": the second tap confirms
  let armTimer = 0;
  let mockTimer = 0;

  function setStatus(s) {
    st = s;
    if (ctx.onReady) ctx.onReady(!!(s && s.installed));
  }
  function armFor(kind) {
    arm = kind;
    clearTimeout(armTimer);
    armTimer = setTimeout(() => { arm = null; ctx.render(); }, 5000);
  }

  // Mock mode: a fake download that moves 4% per tick.
  const MOCK_TOTAL = 1200510818;
  function mockSend(m) {
    if (m.t === "describe_status" && !st) setStatus({installed: false, downloading: false, bytes: 0, total: MOCK_TOTAL, error: null});
    else if (m.t === "describe_download") {
      setStatus(Object.assign({}, st, {downloading: true, error: null}));
      clearInterval(mockTimer);
      mockTimer = setInterval(() => {
        const bytes = Math.min(st.total, st.bytes + st.total * 0.04);
        const done = bytes >= st.total;
        setStatus(Object.assign({}, st, {bytes, downloading: !done, installed: done}));
        if (done) clearInterval(mockTimer);
        ctx.render();
      }, 400);
    } else if (m.t === "describe_cancel") { clearInterval(mockTimer); setStatus(Object.assign({}, st, {downloading: false})); }
    else if (m.t === "describe_remove") setStatus({installed: false, downloading: false, bytes: 0, total: MOCK_TOTAL, error: null});
  }
  const send = (m) => (ctx.mock ? mockSend(m) : ctx.send(m));

  function html() {
    const intro = '<p class="dmnote">Find cards from a description, like “a tuner that special summons itself and ' +
      "searches”. A one-time download, then it works offline.</p>";
    let body;
    if (!st) {
      body = '<p class="dmline" role="status">Checking…</p>';
    } else if (st.installed) {
      body = '<p class="dmline" role="status"><span class="dmok" aria-hidden="true">✓</span> Installed · ' + size(st.total) +
        " · works offline</p>" +
        '<div class="artbtns"><button class="artbtn danger' + (arm === "remove" ? " confirm" : "") + '" data-act="dm:remove">' +
        (arm === "remove" ? "Remove " + size(st.total) + "? Tap again" : "Remove download") + "</button></div>";
    } else {
      const pct = st.total ? Math.min(100, Math.round((st.bytes / st.total) * 100)) : 0;
      const line = st.error ? '<span class="dmerr">' + esc(st.error) + "</span>"
        : st.downloading ? "Downloading · " + size(st.bytes) + " of " + size(st.total) + " · " + pct + "%"
        : st.bytes > 0 ? "Paused at " + size(st.bytes) + " of " + size(st.total)
        : "Not downloaded · " + size(st.total);
      const cell = !st.downloading && ctx.network() === "cellular";
      const go = st.bytes > 0 ? "Resume" : "Download " + size(st.total);
      const btn = st.downloading
        ? '<button class="artbtn" data-act="dm:pause">Pause</button>'
        : '<button class="artbtn' + (cell && arm === "data" ? " confirm" : "") + '" data-act="dm:download">' +
          (cell ? (arm === "data" ? "Use mobile data? Tap again" : go + " on mobile data") : go) + "</button>";
      body = '<p class="dmline" role="status" aria-live="polite">' + line + "</p>" +
        (st.downloading || st.bytes > 0 ? '<div class="artbar" aria-hidden="true"><i style="width:' + pct + '%"></i></div>' : "") +
        (cell ? '<p class="dmnote">You are on mobile data. Wi-Fi is better for 1.2 GB.</p>' : "") +
        '<div class="artbtns">' + btn + "</div>";
    }
    return '<div class="field-group dmpanel" data-feature="describe" id="set-describe" role="group" aria-labelledby="lb-describe">' +
      '<h3 class="flabel" id="lb-describe">Describe a card</h3>' + intro + body + "</div>";
  }

  function act(kind) {
    const a = kind.slice(3);
    if (a === "download") {
      if (ctx.network() === "cellular" && arm !== "data") { armFor("data"); ctx.render(); return; }
      arm = null;
      send({t: "describe_download"});
    } else if (a === "pause") send({t: "describe_cancel"});
    else if (a === "remove") {
      if (arm !== "remove") { armFor("remove"); ctx.render(); return; }
      arm = null;
      send({t: "describe_remove"});
    }
    ctx.render();
  }

  // True when the message was a describe_status (pushed during a download, or a reply).
  function onMessage(m) {
    if (!m || m.t !== "describe_status") return false;
    setStatus(m);
    ctx.render();
    return true;
  }

  return {
    html, act, onMessage,
    refresh() { send({t: "describe_status"}); },
    ready: () => !!(st && st.installed),
  };
}
