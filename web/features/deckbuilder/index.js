// Deck builder feature (docs/design/deck-builder.md): the Decks lobby tab and the deck editor. It owns its state, talks to the
// server through ctx.send (docs/design/api/cards-and-decks.md), and plugs into app.js through the feature registry hooks:
// tab panel, full-screen view, sheet, actions (`db:*`), messages (`req` = "db:…"), Android back.
import {parseYdk, toYdk, toYdke} from "../../lib/ydk.js";
import {FORMATS, validate, counts, isExtraDeckType} from "../../lib/deck.js";
import {searchState, searchRequest} from "../cards/search.js";
import {listHTML, editorHTML, sheetHTML} from "./view.js";
import {mockBackend} from "./mock.js";

const FORMAT = FORMATS.standard;
const SAVE_MS = 600;     // autosave after the last edit
const SEARCH_MS = 160;   // names & text search while typing

export function deckBuilder(ctx) {
  // ctx = {send(msg), render(), say(text), mock: bool, openSettings()}
  const send = ctx.mock ? mockBackend((m) => onMessage(m)) : ctx.send;
  const st = {
    decks: null,          // [{id, name, updated, cover, counts, legal}] from deck_list, newest first
    edit: null,           // {id, name, main, extra, side, cover, dirty, saving, saveErr}
    view: "deck",         // editor body: "deck" | "add"
    sel: null,            // {sec, code}: the picked card in the deck grid
    sheet: null,          // {kind: export|import|menu|delete|card, ...}
    search: searchState(),
    cards: new Map(),     // code -> {code, name, kind, line, extra, type?, text?, alias?}
    canShare: !!(window.DecklightShare && window.DecklightShare.shareYdk),
  };
  let seq = 0;
  let afterLoad = null;   // screenshot hook: what to open once the deck arrives
  let saveTimer = 0;
  let searchTimer = 0;
  const pending = new Set();    // card_get codes in flight
  let describeReady = false;    // "Describe a card" model installed (Settings tells us; kept across decks)

  function req(kind) { seq += 1; return "db:" + kind + ":" + seq; }
  function card(code) {
    const c = st.cards.get(code);
    // Search rows are cached without text, so a row alone does not count as loaded.
    if ((!c || c.text == null) && !pending.has(code)) { pending.add(code); send({t: "card_get", code, req: req("card")}); }
    return c || null;
  }
  const kindOf = (code) => (st.cards.get(code) || {}).kind || "monster";
  // Alternate artworks are one card: search rows carry the original code, a deck may hold an alt-art code.
  const rootOf = (code) => (st.cards.get(code) || {}).alias || code;
  const deckCodes = () => (st.edit ? [...st.edit.main, ...st.edit.extra, ...st.edit.side] : []);
  const copies = (code) => {
    const r = rootOf(code);
    return deckCodes().filter((c) => rootOf(c) === r).length;
  };
  const copiesIn = (code, sec) => (st.edit ? st.edit[sec].filter((c) => rootOf(c) === rootOf(code)).length : 0);
  // The code the deck already uses for this card (keeps alt-art copies on one tile), else the code itself.
  const inDeckCode = (code) => deckCodes().find((c) => rootOf(c) === rootOf(code)) || code;
  function legal() {
    if (!st.edit) return null;
    return validate(st.edit, FORMAT, {
      nameOf: (c) => (st.cards.get(c) || {}).name || String(c),
      aliasOf: (c) => (st.cards.get(c) || {}).alias || c,
    });
  }
  const viewState = () => Object.assign({}, st, {card, kindOf, copies, copiesIn, isExtra, counts, legal: legal()});

  // ---- editing ----
  function touch() {
    st.edit.dirty = true;
    st.edit.saveErr = null;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(save, SAVE_MS);
  }
  function save() {
    const d = st.edit;
    if (!d || !d.dirty) return;
    d.dirty = false;
    d.saving = true;
    send({t: "deck_save", id: d.id || undefined, name: d.name, main: d.main, extra: d.extra, side: d.side,
      cover: d.cover || undefined, req: req("save")});
  }
  function isExtra(code) {
    const c = st.cards.get(code);
    return !!(c && (c.extra || (c.type != null && isExtraDeckType(c.type))));
  }
  function add(code, sec) {
    code = inDeckCode(code);
    const d = st.edit;
    const to = sec || (isExtra(code) ? "extra" : "main");
    d[to].push(code);
    if (!d.cover && to !== "side") d.cover = code;
    touch();
    const c = st.cards.get(code);
    ctx.say("Added " + (c ? c.name : "card") + " to the " + (to === "main" ? "Main" : to === "extra" ? "Extra" : "Side") +
      " Deck, " + copies(code) + " in the deck.");
  }
  function removeOne(code, sec) {
    code = inDeckCode(code);
    const d = st.edit;
    const secs = sec ? [sec] : ["side", "extra", "main"];
    for (const s of secs) {
      const i = d[s].lastIndexOf(code);
      if (i >= 0) {
        d[s].splice(i, 1);
        if (st.sel && st.sel.code === code && !d[s].includes(code)) st.sel = null;
        // The cover must stay a card in the deck: the last copy gone, the first Main (or Extra) card takes over.
        if (d.cover === code && !d.main.includes(code) && !d.extra.includes(code)) d.cover = d.main[0] || d.extra[0] || 0;
        touch();
        return true;
      }
    }
    return false;
  }
  function openDeck(id) {
    st.edit = {id, name: "", main: [], extra: [], side: [], cover: 0, dirty: false, saving: false, saveErr: null, loading: true};
    st.view = "deck";
    st.sel = null;
    send({t: "deck_get", id, req: req("get")});
  }
  function newDeck(name, deck) {
    st.edit = Object.assign({id: null, name: name || "New deck", main: [], extra: [], side: [], cover: 0,
      dirty: false, saving: false, saveErr: null}, deck || {});
    st.view = deck ? "deck" : "add";
    st.sel = null;
    st.search = Object.assign(searchState(), {describeReady});  // the model's state outlives a deck
    touch();
  }
  function runSearch() {
    const m = searchRequest(st.search);
    if (!m) { st.search.rows = null; st.search.loading = false; return; }
    if (st.search.mode === "describe" && !st.search.describeReady) return;
    st.search.loading = true;
    st.search.sent = req("search");
    send(Object.assign(m, {req: st.search.sent}));
  }
  function exportSheet(note) {
    st.sheet = {kind: "export", ydk: toYdk(st.edit), note: note || ""};
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall through */ }
    const ta = document.querySelector(".dbydk");
    if (ta) { ta.select(); try { return document.execCommand("copy"); } catch (e) { return false; } }
    return false;
  }
  function fileName() { return (st.edit.name || "deck").replace(/[^\w .-]+/g, "").trim().replace(/\s+/g, "-") + ".ydk"; }

  // ---- actions ----
  function act(kind, el) {
    const a = kind.slice(3);
    const code = el && el.dataset.code ? Number(el.dataset.code) : 0;
    if (a === "new") newDeck();
    else if (a === "import") st.sheet = {kind: "import", name: "", text: "", note: ""};
    else if (a === "open") openDeck(el.dataset.id);
    else if (a === "close") {
      if (st.edit && st.edit.dirty) { clearTimeout(saveTimer); save(); }
      st.edit = null; st.sheet = null; st.sel = null;
      send({t: "deck_list", req: req("list")});
    } else if (a === "view") {
      // The search bar is shared: its query and chips carry over (Deck filters the deck, Add cards searches all cards).
      st.view = el.dataset.view; st.sel = null; st.search.menu = false;
      if (st.view === "add" && st.search.mode === "names_text") runSearch();
    }
    else if (a === "pick") {
      const s = {sec: el.dataset.sec, code};
      st.sel = st.sel && st.sel.sec === s.sec && st.sel.code === s.code ? null : s;
    } else if (a === "minus" && st.sel) removeOne(st.sel.code, st.sel.sec);
    else if (a === "plus" && st.sel) add(st.sel.code, st.sel.sec);
    else if (a === "move" && st.sel) {
      const {sec, code: c} = st.sel;
      const to = sec === "side" ? (isExtra(c) ? "extra" : "main") : "side";
      removeOne(c, sec);
      st.edit[to].push(c);
      st.sel = {sec: to, code: c};
      touch();
    } else if (a === "details" && st.sel) { st.sheet = {kind: "card", code: st.sel.code}; card(st.sel.code); }
    else if (a === "export") exportSheet();
    else if (a === "menu") st.sheet = {kind: "menu"};
    else if (a === "askdelete") st.sheet = {kind: "delete"};
    else if (a === "delete") {
      if (st.edit.id) send({t: "deck_delete", id: st.edit.id, req: req("delete")});
      ctx.say("Deleted " + st.edit.name + ".");
      st.edit = null; st.sheet = null;
      send({t: "deck_list", req: req("list")});
    } else if (a === "duplicate") {
      const d = st.edit;
      st.sheet = null;
      newDeck(d.name + " (copy)", {main: [...d.main], extra: [...d.extra], side: [...d.side], cover: d.cover});
      ctx.say("Duplicated. You are now editing the copy.");
    } else if (a === "closesheet") st.sheet = null;
    else if (a === "copy" || a === "copyydke") {
      const text = a === "copy" ? st.sheet.ydk : toYdke(st.edit);
      copyText(text).then((ok) => {
        if (st.sheet && st.sheet.kind === "export") st.sheet.note = ok ? (a === "copy" ? "Copied the deck list." : "Copied the ydke:// link.")
          : "Could not copy. Select the text and copy it by hand.";
        ctx.say(st.sheet ? st.sheet.note : "");
        ctx.render();
      });
      return;
    } else if (a === "share") {
      const text = st.sheet.ydk;
      if (st.canShare) window.DecklightShare.shareYdk(fileName(), text);
      else {
        const url = URL.createObjectURL(new Blob([text], {type: "text/plain"}));
        const link = Object.assign(document.createElement("a"), {href: url, download: fileName()});
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        st.sheet.note = "Saved " + fileName() + " to your downloads.";
      }
    } else if (a === "pickfile") { const f = document.getElementById("dbfile"); if (f) f.click(); return; }
    else if (a === "doimport") {
      const s = st.sheet;
      const p = parseYdk(s.text || "");
      if (!p.main.length && !p.extra.length && !p.side.length) { s.note = "No card codes found in that text."; ctx.render(); return; }
      send({t: "deck_import", ydk: s.text, name: (s.name || "").trim() || undefined, req: req("import")});
      s.note = "Importing…";
    } else if (a === "cplus") { if (copies(code) < 3) add(code, el.dataset.sec); }
    else if (a === "cminus") removeOne(code, el.dataset.sec);
    // search component (prefix db:s-)
    else if (a === "s-add") { if (copies(code) < 3) add(code, el.dataset.sec); }
    else if (a === "s-card") { st.sheet = {kind: "card", code}; card(code); }
    else if (a === "s-chip") {
      const k = el.dataset.chip;
      const ch = st.search.chips;
      st.search.chips = ch.includes(k) ? ch.filter((x) => x !== k) : [...ch, k];
      if (st.view === "add" && st.search.mode === "names_text") runSearch();
    } else if (a === "s-clear") {
      st.search.q = ""; st.sel = null;
      if (st.view === "add") runSearch();
    } else if (a === "s-modemenu") st.search.menu = !st.search.menu;
    else if (a === "s-mode") {
      st.search.mode = el.dataset.smode; st.search.menu = false; st.search.rows = null;
      if (st.search.mode === "names_text") runSearch();
    } else if (a === "s-go") runSearch();
    else if (a === "s-filters") st.sheet = {kind: "filters"};
    else if (a === "s-f") {
      const k = el.dataset.fk, v = el.dataset.fv, f = st.search.f;
      const cur = f[k] || [];
      f[k] = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
    } else if (a === "s-fclear") st.search.f = {};
    else if (a === "s-fapply") { st.sheet = null; if (st.view === "add") runSearch(); }
    else if (a === "s-ex") { st.search.q = el.dataset.q || ""; runSearch(); }
    else if (a === "s-stop") { st.search.loading = false; st.search.sent = null; }  // the late reply is ignored
    else if (a === "s-getdescribe") { ctx.openSettings(); return; }
    ctx.render();
  }

  // Inputs: deck name, search text, import fields and the file picker.
  function onInput(ev) {
    const t = ev.target;
    const k = t && t.dataset && t.dataset.dbinput;
    if (!k) return;
    if (k === "name" && st.edit) {
      if (ev.type === "change") { st.edit.name = t.value.trim() || "Untitled deck"; touch(); ctx.render(); }
    } else if (k === "q") {
      st.search.q = t.value;
      if (st.view === "deck") { st.sel = null; ctx.render(); }
      else if (st.search.mode === "names_text") { clearTimeout(searchTimer); searchTimer = setTimeout(() => { runSearch(); ctx.render(); }, SEARCH_MS); }
    } else if (k.startsWith("f:")) {
      const [, rk, ix] = k.split(":");
      const r = (st.search.f[rk] || [null, null]).slice();
      r[Number(ix)] = t.value === "" ? null : Math.max(0, Number(t.value));
      st.search.f[rk] = r;
      if (ev.type === "change") ctx.render();
    } else if (k === "impname" && st.sheet) st.sheet.name = t.value;
    else if (k === "imptext" && st.sheet) {
      const had = !!(st.sheet.text || "").trim();
      st.sheet.text = t.value;
      if (had !== !!t.value.trim()) ctx.render();
    } else if (k === "file" && t.files && t.files[0] && ev.type === "change") {
      const f = t.files[0];
      f.text().then((text) => {
        if (!st.sheet || st.sheet.kind !== "import") return;
        st.sheet.text = text;
        if (!st.sheet.name) st.sheet.name = f.name.replace(/\.ydk$/i, "").replace(/[-_]+/g, " ");
        ctx.render();
      });
    }
  }
  document.addEventListener("input", onInput);
  document.addEventListener("change", onInput);
  document.addEventListener("keydown", (ev) => {
    const t = ev.target;
    if (ev.key === "Enter" && t && t.dataset && t.dataset.dbinput === "q" && st.search.mode === "describe") {
      runSearch(); ctx.render();
    }
  });

  // ---- messages (true when consumed) ----
  function onMessage(m) {
    const mine = typeof m.req === "string" && m.req.startsWith("db:");
    if (m.t === "error" && mine) {
      if (m.req.startsWith("db:save") && st.edit) { st.edit.saving = false; st.edit.saveErr = m.error; }
      else if (m.req.startsWith("db:search")) { st.search.loading = false; st.search.err = m.error; }
      else if (m.req.startsWith("db:import") && st.sheet) st.sheet.note = m.error;
      else ctx.say(m.error);
      ctx.render();
      return true;
    }
    if (mine && m.req.startsWith("db:import")) {
      // deck_import replies like deck_saved, plus unknown codes and lines.
      const unknown = m.unknown || [];
      st.sheet = null;
      openDeck(m.id);
      ctx.say("Imported." + (unknown.length ? " " + unknown.length + " unknown card codes were kept: " + unknown.join(", ") + "." : ""));
      ctx.render();
      return true;
    }
    if (m.t === "user_decks") {
      st.decks = m.decks || [];
      for (const d of st.decks) if (d.cover) card(d.cover);
    } else if (m.t === "user_deck") {
      if (!st.edit || st.edit.id !== m.deck.id) return true;
      Object.assign(st.edit, m.deck, {loading: false, dirty: false, saving: false});
      for (const c of new Set([...m.deck.main, ...m.deck.extra, ...m.deck.side])) card(c);
      if (afterLoad) { const f = afterLoad; afterLoad = null; f(); }
    } else if (m.t === "deck_saved") {
      if (st.edit) {
        if (!st.edit.id) st.edit.id = m.id;
        st.edit.saving = false;
        if (st.edit.dirty) save();
      }
    } else if (m.t === "card") {
      pending.delete(m.code);
      st.cards.set(m.code, m);
    } else if (m.t === "cards") {
      if (m.req !== st.search.sent) return true;
      st.search.loading = false;
      st.search.err = null;
      st.search.rows = m.rows || [];
      st.search.total = m.total || 0;
      for (const r of st.search.rows) if (!st.cards.has(r.code)) st.cards.set(r.code, r);
    } else if (m.t === "deck_ydk") {
      return true;
    } else if (!mine) return false;
    ctx.render();
    return true;
  }

  return {
    id: "db",
    tab: {key: "decks", label: "Decks"},
    // Lobby hooks
    enter() { if (st.decks === null) send({t: "deck_list", req: req("list")}); },
    panelHTML: () => listHTML(viewState()),
    fullView: () => (st.edit ? editorHTML(viewState()) : null),
    sheetOpen: () => !!st.sheet,
    sheetHTML: () => sheetHTML(viewState()),
    editing: () => !!st.edit,
    // Android back: the sheet first, then the editor.
    back(name) {
      if (name === "dbsheet") st.sheet = null;
      else if (name === "dbedit") act("db:close", null);
    },
    act,
    onMessage,
    // Android back: the open search-mode menu is the top layer.
    menuOpen: () => !!st.search.menu,
    closeMenu(rerender = true) { if (!st.search.menu) return false; st.search.menu = false; if (rerender) ctx.render(); return true; },
    setDescribeReady(v) { describeReady = st.search.describeReady = !!v; },
    // Screenshot hooks (?ltab=decks&dbopen=<id>&dbview=add&dbq=<text>&dbmode=describe&dbchip=trap&dbsheet=…&dbcard=<code>&dbpick=main:<code>).
    mockInit(q) {
      if (st.decks === null) send({t: "deck_list", req: req("list")});
      const sheet = q.get("dbsheet");
      if (sheet === "import") st.sheet = {kind: "import", name: "", text: "", note: ""};
      const id = q.get("dbopen");
      if (!id) return;
      openDeck(id);
      afterLoad = () => {
        if (q.get("dbview")) st.view = q.get("dbview");
        if (q.get("dbmode")) st.search.mode = q.get("dbmode");
        if (q.get("dbchip")) st.search.chips = q.get("dbchip").split(",");
        if (q.get("dbq") || q.get("dbchip")) { st.search.q = q.get("dbq") || ""; runSearch(); }
        const pk = q.get("dbpick");
        if (pk) { const [sec, c] = pk.split(":"); st.sel = {sec, code: Number(c)}; }
        if (sheet === "export") exportSheet();
        else if (sheet === "menu") st.sheet = {kind: "menu"};
        else if (sheet === "delete") st.sheet = {kind: "delete"};
        else if (sheet === "filters") st.sheet = {kind: "filters"};
        else if (sheet === "card") { const c = Number(q.get("dbcard")); st.sheet = {kind: "card", code: c}; card(c); }
        ctx.render();
      };
    },
  };
}
