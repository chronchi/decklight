// Ask feature (docs/plans/ask-tab.md step 7): the Ask tab's chat over the server's ask_* messages
// (rust/decklight-server/src/ask/ws.rs). It owns its state and plugs into app.js like the deck builder: tab panel,
// composer, sheet, actions (`ask:*`), messages (`ask_*`), Android back.
import {esc} from "../../render/card.js";
import {I, DMARK, mdHTML, trailHTML, errHTML, branchHTML, toolArg, cardsIn, plainText} from "./view.js";
import {blockSpecs, blockHTML, framesNeeded, stepSeqs, seqAct} from "./blocks.js";
import {retryStill} from "../../lib/stills.js";
import {MAX as MAX_ATTACH, chipHTML, attachSheetHTML, readPhoto, IA} from "./attach.js";

// Shown while the model works and no tool is running; a new one every 2.5 s, never the same twice in a row.
const WORDS = ["Shuffling", "Drawing", "Reading the field", "Checking the ruling", "Counting outs", "Building the chain",
  "Resolving the chain", "Looking at the graveyard", "Setting a card", "Searching the deck", "Thinking a turn ahead",
  "Weighing the hand traps"];
const SUGGEST = [
  ["What does Maxx \"C\" do?", "Card text and rulings"],
  ["Odds of opening a starter in my deck", "Draw odds over your deck"],
  ["How do I do going second against the bot?", "Your games against the bot"],
];

export function askFeature(ctx) {
  // ctx = {send(msg), render(), say(text), mock: bool, openCard(code, name), watch(game id, choice)}
  const st = {
    settings: null,      // the last ask_settings frame
    models: {},          // preset -> {list} | {pending} | {err}
    chats: null,         // [chat] newest first
    running: null,       // chat id with a turn in flight (server's view)
    chat: null,          // the open chat (chat_json) or null for a new one
    rows: [],            // every row of the open chat
    live: null,          // {chat, base, q, text, calls, limit, retry, word, t0}
    err: null,           // an error not stored in a chat ({message, action})
    view: "thread",      // thread | list | settings
    sheet: null,         // null | "model" | {menu: user row id}
    sheetPreset: null,   // the provider tab shown in the model sheet
    device: null,        // the last device_status frame (the On this device tab: phone check, model downloads)
    devErr: null,        // a refused device_* request ({message})
    draft: "",
    attach: [],          // the composer's attachments: {kind: deck|game|card, id|code, label} | {kind: photo, data, thumb}
    edit: null,          // {id, parent} when editing a sent question
    trailOpen: new Set(),
    outs: new Set(),
    q: "",               // chat list search
    delArm: null,        // chat id armed for delete
    keyMsg: null,        // {ok|err} after a key save
    lastSend: null,
    up: false,           // the thread is scrolled up from its end (shows the jump-to-latest button)
    frames: new Map(),   // game block boards by key (t:<game>:<turn> | d:<game>:<decision>): {pending} | {error} | ask_frames
    blocks: new Map(),   // game block play state by key: {i, playing, watch, auto}
  };
  let wordTimer = 0;
  let seqTimer = 0;
  let findSeq = 0;
  let attTimer = 0;
  let frame = 0;
  let entered = false;
  const send = (m) => { if (!ctx.mock) ctx.send(m); };
  const render = () => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; ctx.render(); }); };

  // ---- the branch on screen -------------------------------------------------------------------------------------
  const byId = () => new Map(st.rows.map((r) => [r.id, r]));
  const kids = (id) => st.rows.filter((r) => (r.parent || null) === (id || null)).sort((a, b) => (a.id < b.id ? -1 : 1));
  function pathTo(leaf) {
    const m = byId();
    const out = [];
    for (let r = m.get(leaf); r; r = r.parent ? m.get(r.parent) : null) out.push(r);
    return out.reverse();
  }
  // The newest leaf under a row: what a branch switch moves the head to.
  function leafOf(id) {
    let cur = id;
    for (;;) {
      const k = kids(cur);
      if (!k.length) return cur;
      cur = k[k.length - 1].id;
    }
  }
  const liveHere = () => st.live && st.chat && st.live.chat === st.chat.id;

  // Rows grouped into turns: a question and everything after it up to the next question.
  function turns(rows) {
    const out = [];
    for (const r of rows) {
      if (r.role === "user") out.push({user: r, rest: []});
      else if (out.length) out[out.length - 1].rest.push(r);
    }
    return out;
  }
  function callsOf(rest) {
    const results = new Map(rest.filter((r) => r.role === "tool").map((r) => [r.body.tool_call_id, r.body.content]));
    const calls = [];
    for (const r of rest) {
      if (r.role !== "assistant") continue;
      for (const tc of r.body.tool_calls || []) {
        const fn = tc.function || {};
        const res = results.get(tc.id);
        let parsed = null;
        try { parsed = JSON.parse(res); } catch (e) { /* plain text */ }
        calls.push({id: tc.id, name: fn.name, args: fn.arguments, result: res, done: true,
          ok: !(parsed && typeof parsed === "object" && parsed.error), arg: toolArg(fn.name, fn.arguments, res)});
      }
    }
    return calls;
  }
  const textOf = (rest) => rest.filter((r) => r.role === "assistant" && r.body.content).map((r) => r.body.content).join("\n\n");

  // ---- server messages ------------------------------------------------------------------------------------------
  function onMessage(m) {
    // The "which card is this?" sheet's search results (cards_search with an ask: req).
    if (m && m.t === "cards" && String(m.req || "").startsWith("ask:")) {
      if (st.sheet && (st.sheet.find || st.sheet.attach) && st.sheet.req === m.req) st.sheet.rows = m.rows || [];
      render();
      return true;
    }
    if (m && m.t === "user_decks" && m.req === "ask:att:decks") {
      if (st.sheet && st.sheet.attach === "deck") st.sheet.rows = m.decks || [];
      render();
      return true;
    }
    if (m && (m.t === "ask_data" || m.t === "ask_data_pending") && String(m.key || "").startsWith("ask:att:")) {
      if (m.t === "ask_data" && st.sheet && st.sheet.attach === "game" && st.sheet.req === m.key) {
        if (m.error) st.sheet.err = m.error;
        else {
          const r = m.result || {};
          st.sheet.rows = (r.games || []).concat((r.samples || []).filter((g) => !st.sheet.q ||
            (g.your_deck + " " + g.opponent_deck).toLowerCase().includes(st.sheet.q.toLowerCase())).map((g) => Object.assign({sample: true}, g)));
        }
      }
      render();
      return true;
    }
    if (m && m.t === "device_status") {
      st.device = m;
      if (m.req === "ask:dev") st.devErr = null;
      render();
      return true;
    }
    if (m && m.t === "error" && m.req === "ask:dev") {
      st.devErr = {message: m.error || m.message || m.text || "The phone refused the request: try again"};
      render();
      return true;
    }
    if (m && m.t === "error" && String(m.req || "").startsWith("ask:")) {
      // The server's error frames carry `error` (with_req errors) or `text`.
      if (st.sheet && (st.sheet.find || st.sheet.attach) && st.sheet.req === m.req) {
        st.sheet.err = m.error || m.message || m.text || "The card search failed: try again";
      }
      render();
      return true;
    }
    if (!m || typeof m.t !== "string" || !m.t.startsWith("ask_")) return false;
    if (m.t === "ask_settings") {
      st.settings = m;
      if (!st.sheetPreset) st.sheetPreset = m.preset;
    } else if (m.t === "ask_models_pending") st.models[m.preset] = {pending: true};
    else if (m.t === "ask_models") st.models[m.preset] = {list: (m.models || []).map((x) => (typeof x === "string" ? x : x.id || x.name))};
    else if (m.t === "ask_chats") {
      st.chats = m.chats || [];
      st.running = m.running || null;
      if (st.chat && !st.chats.some((c) => c.id === st.chat.id) && !liveHere()) { st.chat = null; st.rows = []; }
    } else if (m.t === "ask_chat") {
      // A chat asked for before "Ask about this game" made a new one: not shown over the drafted question.
      if (st.fresh && !(st.chat && st.chat.id === m.chat.id)) { render(); return true; }
      st.chat = m.chat;
      st.rows = m.rows || [];
      if (m.running) st.running = m.chat.id;
    } else if (m.t === "ask_started") {
      st.running = m.chat;
      if (st.live) st.live.chat = m.chat;
      if (st.lastSend) st.lastSend.started = true;
      if (!st.chat) { st.chat = {id: m.chat, title: (st.live && st.live.q) || "New chat", head: null, preset: m.preset, model: m.model}; st.rows = []; }
    } else if (m.t === "ask_frames") {
      st.frames.set(m.key, m.error ? {error: m.error} : m);
      const u = st.blocks.get(m.key);
      if (u && u.auto && !m.error && (m.frames || []).length > 1 && !reducedMotion() && seqAct(st.blocks, m.key, m.frames.length, "play")) playTimer();
    } else if (m.t === "ask_frames_pending") { /* the frames follow */ }
    else if (m.t === "ask_event") onEvent(m);
    else if (m.t === "ask_cancel") { /* the turn ends with its own "Stopped." error */ }
    else if (m.t === "ask_error") {
      // A models request failed: show it in the sheet; anything else under the thread or in Settings.
      if (st.sheet === "model" && st.sheetPreset && st.models[st.sheetPreset] && st.models[st.sheetPreset].pending) {
        st.models[st.sheetPreset] = {err: {message: m.message, action: m.action}};
      } else {
        if (st.live && !st.running) {
          // Refused before it started (a photo too big, too many attachments): the question goes back to the composer.
          if (st.lastSend && !st.lastSend.started && st.live.q === st.lastSend.text && !st.attach.length) {
            st.attach = st.lastSend.attach || [];
            if (!st.draft) st.draft = st.lastSend.text;
          }
          stopLive();
        }
        if (st.view === "settings") st.keyMsg = {err: m.message};
        else st.err = {message: m.message, action: m.action};
        ctx.say(m.message);
      }
    }
    render();
    return true;
  }

  function onEvent(m) {
    const L = st.live && st.live.chat === m.chat ? st.live : null;
    if (m.kind === "done" || m.kind === "error") {
      st.running = null;
      if (L) stopLive();
      if (m.kind === "error" && !m.user) st.err = {message: m.message, action: m.action};
      if (m.kind === "error") ctx.say(m.message);
      // The stored rows (answer or error row) come back with the chat.
      if (st.chat && st.chat.id === m.chat) send({t: "ask_open", chat: m.chat});
      send({t: "ask_list"});
      return;
    }
    if (!L) return;
    if (m.kind === "status") L.status = m.text || null;   // "Loading Gemma 4 E4B on this phone's GPU"; "" clears it
    else if (m.kind === "text") { L.text += m.text || ""; L.retry = null; L.status = null; }
    else if (m.kind === "retry") L.retry = {attempt: m.attempt, secs: Math.round((m.delay_ms || 0) / 1000), reason: m.reason};
    else if (m.kind === "tool_start") {
      // Text written before a call is its own paragraph, as in the stored answer.
      if (L.text && !L.text.endsWith("\n\n")) L.text += "\n\n";
      L.calls.push({id: m.id, name: m.name, args: m.args, arg: toolArg(m.name, m.args), done: false});
      L.status = null;
    } else if (m.kind === "tool_done") {
      const c = L.calls.find((x) => x.id === m.id);
      if (c) Object.assign(c, {done: true, ok: m.ok !== false, result: m.result, arg: toolArg(c.name, c.args, m.result)});
    } else if (m.kind === "limit") L.limit = true;
  }

  function startLive(base, q, attach) {
    st.err = null;
    st.live = {chat: st.chat ? st.chat.id : null, base, q, attach: attach || [], text: "", calls: [], limit: false, retry: null, word: pickWord(""), t0: Date.now()};
    // On Android the answer runs under the background job (DownloadJob.kt, kind "answer"): it goes on with the app in
    // the background and says "Answer ready" when done. It can only start from this tap, while the app is on screen.
    if (window.DecklightJobs && window.DecklightJobs.answer) {
      const e = window.DecklightJobs.answer();
      if (e) st.err = {message: e + ": keep the app open until the answer ends"};
    }
    clearInterval(wordTimer);
    wordTimer = setInterval(() => { if (st.live) { st.live.word = pickWord(st.live.word); render(); } }, 2500);
  }
  function stopLive() {
    st.live = null;
    clearInterval(wordTimer);
  }
  function pickWord(prev) {
    const pool = WORDS.filter((w) => w !== prev);
    return pool[Math.floor(Math.random() * pool.length)];
  }

  // The attach sheet's search: games through the Games tab's list (ask_data), cards through cards_search.
  function attachSearch() {
    const s = st.sheet;
    if (!s || (s.attach !== "game" && s.attach !== "card")) return;
    findSeq += 1;
    s.req = "ask:att:" + findSeq;
    s.err = null;
    const q = (s.q || "").trim();
    if (s.attach === "game") {
      s.rows = null;
      send({t: "ask_data", key: s.req, what: "list", args: {filter: "all", q, limit: 20}});
    } else if (q) {
      s.rows = null;
      send({t: "cards_search", q, limit: 12, req: s.req});
    } else s.rows = [];
    render();
  }

  // ---- actions --------------------------------------------------------------------------------------------------
  function sendText(text, attach) {
    st.fresh = false;
    text = (text || "").trim();
    attach = attach || st.attach;
    if (!text && !attach.length) return;
    if (st.running) { st.err = {message: "An answer is still coming: wait for it or stop it first", action: "cancel"}; render(); return; }
    const msg = {t: "ask_send", text};
    if (attach.length) msg.attach = attach;
    let base = null;
    if (st.chat) {
      msg.chat = st.chat.id;
      base = st.edit ? st.edit.parent : st.chat.head;
      if (base) msg.parent = base;
      else if (st.edit) msg.parent = null;
    }
    st.lastSend = {text, attach, chat: msg.chat || null, parent: base};
    st.edit = null;
    st.draft = "";
    st.attach = [];
    const box = document.getElementById("akin");
    if (box) box.value = "";
    startLive(base, text, attach);
    send(msg);
    render();
  }
  function again(user) {
    if (!st.chat || !user) return;
    if (st.running) { st.err = {message: "An answer is still coming: wait for it or stop it first", action: "cancel"}; render(); return; }
    startLive(user, null);
    send({t: "ask_send", chat: st.chat.id, again: user});
    render();
  }
  function openChat(id) {
    st.view = "thread";
    if (!(st.chat && st.chat.id === id)) st.attach = [];
    st.err = null;
    st.edit = null;
    st.trailOpen.clear();
    st.outs.clear();
    if (st.chat && st.chat.id === id) { send({t: "ask_open", chat: id}); render(); return; }
    st.chat = {id, title: ((st.chats || []).find((c) => c.id === id) || {}).title || "", head: null};
    st.rows = [];
    send({t: "ask_open", chat: id});
    render();
  }
  function newChat() {
    st.view = "thread";
    st.chat = null;
    st.rows = [];
    st.err = null;
    st.edit = null;
    st.attach = [];
    if (st.live && !st.running) stopLive();
    render();
  }
  function openModels(preset) {
    st.sheet = "model";
    st.sheetPreset = preset || (st.settings && st.settings.preset) || null;
    const p = st.sheetPreset;
    const cached = st.models[p];
    // The device tab checks the phone and the downloads each time it opens (mock: local-check).
    if (p === "device") send({t: "device_status", req: "ask:dev"});
    else if (p && (!cached || cached.err)) {
      st.models[p] = {pending: true};
      const msg = {t: "ask_settings", op: "models", preset: p};
      if (p === "custom" && st.settings) msg.base_url = st.settings.base_url;
      send(msg);
    }
    render();
  }

  function act(kind, el) {
    const a = kind.slice(4);
    const id = el && el.getAttribute("data-id");
    // A still of the play screen: tapping it opens Watch at that play (Games' watch.js); Try again redraws it.
    if (a === "watchat") { ctx.watch(el.getAttribute("data-id"), +el.getAttribute("data-choice")); return; }
    if (a === "still") { retryStill(el.getAttribute("data-key")); render(); return; }
    if (a === "seq" || a === "frames" || a === "watch") {
      const key = el.getAttribute("data-key");
      const u = ui(key);
      if (a === "frames") st.frames.delete(key);
      else if (a === "watch") u.watch = !u.watch;
      else {
        const fr = st.frames.get(key);
        if (seqAct(st.blocks, key, fr && fr.frames ? fr.frames.length : 0, el.getAttribute("data-do"))) playTimer();
        u.auto = false;
      }
      // A key the cache dropped (Try again) or a watch just opened asks the server on this render.
      render();
      return;
    }
    if (a === "list") { st.view = st.view === "list" ? "thread" : "list"; if (st.view === "list") send({t: "ask_list"}); }
    else if (a === "new") { st.fresh = false; newChat(); }
    else if (a === "open") { st.fresh = false; openChat(id); }
    else if (a === "settings") { st.view = "settings"; st.sheet = null; st.keyMsg = null; send({t: "ask_settings", op: "get"}); }
    else if (a === "back") st.view = "thread";
    else if (a === "model") openModels();
    else if (a === "ptab") openModels(el.getAttribute("data-preset"));
    else if (a === "provider") { st.sheet = "provider"; st.provQ = ""; send({t: "device_status", req: "ask:dev"}); requestAnimationFrame(() => { const i = document.getElementById("akprovq"); if (i) i.focus(); }); }
    else if (a === "provback") openModels(st.sheetPreset);
    else if (a === "pick") {
      send({t: "ask_settings", op: "set", preset: el.getAttribute("data-preset"), model: el.getAttribute("data-model")});
      st.sheet = null;
      st.err = null;
    } else if (a === "pickother") {
      const inp = document.getElementById("akmodel");
      const v = inp ? inp.value.trim() : "";
      if (v) { send({t: "ask_settings", op: "set", preset: st.sheetPreset, model: v}); st.sheet = null; st.err = null; }
    } else if (a === "dev") {
      const d = el.getAttribute("data-do");
      const msg = {t: "device_" + d, req: "ask:dev", model: el.getAttribute("data-model") || ""};
      if (d === "wifi") msg.on = !(st.device && st.device.wifi_only);
      st.devErr = null;
      // On Android the download runs under the background job (DownloadJob.kt): it goes on with the app closed and
      // shows in one notification. It can only start from this tap, while the app is on screen.
      if (d === "download" && window.DecklightJobs) {
        const e = window.DecklightJobs.start();
        if (e) st.devErr = {message: e + ": the download runs only while the app is open"};
      }
      send(msg);
    } else if (a === "attach") st.sheet = {attach: "menu"};
    else if (a === "attach-deck") { st.sheet = {attach: "deck", rows: null}; send({t: "deck_list", req: "ask:att:decks"}); }
    else if (a === "attach-game") { st.sheet = {attach: "game", q: "", rows: null}; attachSearch(); }
    else if (a === "attach-card") st.sheet = {attach: "card", q: "", rows: []};
    else if (a === "attach-photo") st.sheet = {attach: "photo"};
    else if (a === "photo-take" || a === "photo-pick") {
      // The file chooser (RustWebChromeClient.kt): `capture` opens the camera, without it the photo picker.
      const inp = document.createElement("input");
      inp.type = "file";
      inp.accept = "image/*";
      if (a === "photo-take") inp.setAttribute("capture", "environment");
      inp.addEventListener("change", () => {
        const f = inp.files && inp.files[0];
        if (!f) return;
        st.sheet = {attach: "photo", busy: true};
        render();
        readPhoto(f).then((ph) => {
          st.attach = st.attach.filter((x) => x.kind !== "photo").concat([ph]);
          if (st.sheet && st.sheet.attach === "photo") st.sheet = null;
          render();
        }, (e) => {
          if (st.sheet && st.sheet.attach === "photo") st.sheet = {attach: "photo", err: e.message || String(e)};
          render();
        });
      });
      inp.click();
      return;
    } else if (a === "attach-pick") {
      const kind = el.getAttribute("data-kind");
      const item = {kind, label: el.getAttribute("data-label") || ""};
      if (kind === "card") item.code = +el.getAttribute("data-code");
      else item.id = el.getAttribute("data-id");
      const same = (x) => x.kind === kind && (kind === "card" ? x.code === item.code : x.id === item.id);
      if (!st.attach.some(same) && st.attach.length < MAX_ATTACH) st.attach = st.attach.concat([item]);
      st.sheet = null;
      requestAnimationFrame(() => { const box = document.getElementById("akin"); if (box) box.focus(); });
    } else if (a === "unattach") {
      const i = +el.getAttribute("data-i");
      st.attach = st.attach.filter((_, j) => j !== i);
    } else if (a === "closesheet") st.sheet = null;
    else if (a === "send") sendText(document.getElementById("akin") ? document.getElementById("akin").value : st.draft);
    else if (a === "stop") send({t: "ask_cancel"});
    else if (a === "sugg") sendText(el.getAttribute("data-q"));
    else if (a === "trail") { const k = el.getAttribute("data-key"); st.trailOpen.has(k) ? st.trailOpen.delete(k) : st.trailOpen.add(k); }
    else if (a === "out") {
      // Collapsed, a tap on a call opens the whole trail; open, it shows that call's output.
      const tr = el.closest(".ak-trail");
      const sum = tr && tr.querySelector(".sum");
      const k = sum && sum.getAttribute("data-key");
      if (k && !st.trailOpen.has(k)) { st.trailOpen.add(k); st.outs.add(id); } else if (st.outs.has(id)) st.outs.delete(id); else st.outs.add(id);
    } else if (a === "copy") {
      const t = el.getAttribute("data-text") || "";
      try { navigator.clipboard.writeText(t).then(() => ctx.say("Copied"), () => ctx.say("Copy failed: select the text instead")); }
      catch (e) { ctx.say("Copy failed: select the text instead"); }
      return;
    } else if (a === "find") {
      const name = el.getAttribute("data-name") || el.textContent;
      findSeq += 1;
      st.sheet = {find: name, req: "ask:find:" + findSeq, rows: null, err: null};
      send({t: "cards_search", q: name, limit: 8, req: st.sheet.req});
    } else if (a === "card") {
      if (st.sheet && st.sheet.find) st.sheet = null;
      const code = +el.getAttribute("data-code");
      if (code && ctx.openCard) ctx.openCard(code, el.textContent);
      return;
    } else if (a === "bottom") {
      jumpToEnd();
      return;
    } else if (a === "again") again(id);
    else if (a === "qmenu") st.sheet = {menu: id};
    else if (a === "editq") {
      const r = byId().get(id);
      if (r) {
        st.edit = {id, parent: r.parent || null};
        st.draft = r.body.content || "";
        // The edited question keeps its attachments (its photo by its saved name, with the thumbnail).
        const thumbs = (r.meta && r.meta.thumbs) || {};
        st.attach = (r.body.attach || []).map((x) => (x.kind === "photo" ? {kind: "photo", file: x.file, thumb: thumbs[x.file]} : Object.assign({}, x)));
        st.sheet = null;
        ctx.render();
        const box = document.getElementById("akin");
        if (box) { box.value = st.draft; box.focus(); }
        return;
      }
    } else if (a === "canceledit") { st.edit = null; st.draft = ""; st.attach = []; const box = document.getElementById("akin"); if (box) box.value = ""; }
    else if (a === "qbranch" || a === "abranch") {
      // Switch to the previous/next version of a question (its siblings) or of an answer (the question's children).
      const r = byId().get(id);
      const dir = +el.getAttribute("data-dir");
      if (r && st.chat) {
        const sibs = a === "qbranch" ? kids(r.parent).filter((x) => x.role === "user") : kids(r.id);
        const cur = a === "qbranch" ? r.id : (pathTo(st.chat.head).find((x) => x.parent === r.id) || {}).id;
        const i = sibs.findIndex((x) => x.id === cur);
        const to = sibs[i + dir];
        if (to) send({t: "ask_fork", chat: st.chat.id, message: leafOf(to.id)});
      }
    } else if (a === "del") {
      if (st.delArm === id) { send({t: "ask_delete", chat: id}); st.delArm = null; if (st.chat && st.chat.id === id) { st.chat = null; st.rows = []; } }
      else st.delArm = id;
    } else if (a === "preset") send({t: "ask_settings", op: "set", preset: el.getAttribute("data-preset")});
    else if (a === "keyset") {
      const inp = document.getElementById("akkey");
      const v = inp ? inp.value.trim() : "";
      if (!v) { st.keyMsg = {err: "Paste the key first."}; render(); return; }
      send({t: "ask_settings", op: "key_set", preset: el.getAttribute("data-preset"), key: v});
      if (inp) inp.value = "";
      st.keyMsg = {ok: "Key saved."};
      st.err = null;
    } else if (a === "keydel") {
      send({t: "ask_settings", op: "key_delete", preset: el.getAttribute("data-preset")});
      st.keyMsg = {ok: "Key removed."};
    } else if (a === "baseurl") {
      const inp = document.getElementById("akbase");
      send({t: "ask_settings", op: "set", base_url: inp ? inp.value.trim() : ""});
      st.keyMsg = {ok: "Address saved."};
    } else if (a === "do") {
      // An error's action button.
      const d = el.getAttribute("data-do");
      st.err = null;
      if (d === "retry") {
        const u = el.getAttribute("data-user");
        if (u) again(u);
        else if (st.lastSend) sendText(st.lastSend.text, st.lastSend.attach);
        return;
      }
      if (d === "settings") { st.view = "settings"; st.sheet = null; send({t: "ask_settings", op: "get"}); }
      else if (d === "model") openModels();
      else if (d === "new") newChat();
      else if (d === "cancel") send({t: "ask_cancel"});
    }
    render();
  }

  // Typing is kept in state so a re-render (a streamed answer) never loses it; Enter sends on a keyboard, Shift+Enter
  // is a new line.
  document.addEventListener("input", (ev) => {
    const t = ev.target;
    if (t.id === "akin") {
      st.draft = t.value;
      const b = document.querySelector(".ak-send:not(.stop)");
      if (b) b.disabled = !t.value.trim() && !st.attach.length;
    } else if (t.id === "akq") { st.q = t.value; render(); }
    else if (t.id === "akprovq") { st.provQ = t.value; render(); }
    else if (t.id === "akattq" && st.sheet && st.sheet.attach) {
      st.sheet.q = t.value;
      clearTimeout(attTimer);
      attTimer = setTimeout(attachSearch, 250);
    }
  });
  // The jump-to-latest button shows once the thread is scrolled up more than a screen's third from its end. It is
  // shown and hidden in place: a render here would restore the saved scroll position and stop the button's smooth
  // scroll part-way (2026-11-15: "it never goes to the bottom").
  // The jump glides to the thread's end in its own frames, aiming at the end as it is in each frame: a native smooth
  // scroll aims at the end when tapped and stops wherever a render restores the scroll position or the content grows
  // (images, a streaming answer), which left it part-way (2026-11-16: "it goes up to the middle sometimes").
  let jumpRun = 0;
  const jumpToEnd = () => {
    const run = ++jumpRun, t0 = performance.now(), still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const stop = () => { jumpRun += run === jumpRun ? 1 : 0; removeEventListener("wheel", stop, true); removeEventListener("touchstart", stop, true); };
    addEventListener("wheel", stop, true); addEventListener("touchstart", stop, true);
    const step = (now) => {
      if (run !== jumpRun) return;
      const sc = document.querySelector("#lobby.askview .lscroll");
      if (!sc) return stop();
      const end = sc.scrollHeight - sc.clientHeight, left = end - sc.scrollTop;
      if (left <= 1 && now - t0 > 250) return stop();
      sc.scrollTop = still || left < 4 ? end : sc.scrollTop + Math.max(4, left * 0.22);
      if (now - t0 > 3000) { sc.scrollTop = end; return stop(); }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  document.addEventListener("scroll", (ev) => {
    const sc = ev.target;
    if (!sc.matches || !sc.matches("#lobby.askview .lscroll")) return;
    const up = sc.scrollHeight - sc.scrollTop - sc.clientHeight > Math.max(240, sc.clientHeight / 3);
    if (up === st.up) return;
    st.up = up;
    const b = document.querySelector(".ak-jump");
    if (b) b.hidden = !up;
  }, true);
  document.addEventListener("keydown", (ev) => {
    if (ev.target.id === "akin" && ev.key === "Enter" && !ev.shiftKey && !ev.isComposing && matchMedia("(hover: hover)").matches) {
      ev.preventDefault();
      sendText(ev.target.value);
    }
  });

  // ---- markup ---------------------------------------------------------------------------------------------------
  const presetOf = (id) => ((st.settings && st.settings.presets) || []).find((p) => p.id === id) || null;
  const modelLine = () => {
    if (!st.settings) return "";
    const p = presetOf(st.settings.preset);
    return (p ? p.name : st.settings.preset) + " · " + (st.settings.model || "no model");
  };

  function headHTML() {
    const title = st.chat ? st.chat.title || "Chat" : "New chat";
    return '<div class="ak-head"><button class="ak-ibtn" data-act="ask:list" aria-label="Chats" aria-expanded="' + (st.view === "list") + '">' + I.list +
      '</button><h2 class="ak-title">' + esc(title) + '</h2><button class="ak-ibtn" data-act="ask:new" aria-label="New chat">' + I.plus + "</button></div>";
  }

  // A sent question's attachments above its text; a photo shows its thumbnail (kept in the row's meta).
  function sentChips(attach, thumbs) {
    if (!attach || !attach.length) return "";
    return '<span class="ak-att">' + attach.map((x) => chipHTML(x, null, x.thumb || (thumbs && thumbs[x.file]))).join("") + "</span>";
  }

  function turnHTML(t, last, live) {
    const u = t.user;
    let h = "";
    const qs = kids(u.parent).filter((x) => x.role === "user");
    h += '<div class="ak-qwrap">' + branchHTML(qs.findIndex((x) => x.id === u.id), qs.length, "ask:qbranch", u.id) +
      '<button class="ak-q" data-act="ask:qmenu" data-id="' + esc(u.id) + '" aria-haspopup="menu">' + sentChips(u.body.attach, (u.meta || {}).thumbs) +
      esc(u.body.content || "") + "</button></div>";
    if (live) return h;
    const calls = callsOf(t.rest);
    const meta = (t.rest.find((r) => r.meta && r.meta.model) || {}).meta || {};
    h += trailHTML(u.id, calls, {open: st.trailOpen.has(u.id), outs: st.outs, running: false,
      secs: meta.latency_ms && calls.length ? Math.max(1, Math.round(meta.latency_ms / 1000)) : 0});
    const text = textOf(t.rest);
    if (text) {
      const cards = cardsIn(calls.map((c) => c.result));
      for (const r of t.rest) for (const [n, code] of Object.entries((r.meta && r.meta.cards) || {})) cards.set(n, code);
      h += '<div class="ak-a">' + mdHTML(text, cards) + "</div>";
    }
    if (text) h += blocksHTML(calls, last);
    const er = t.rest.find((r) => r.role === "error");
    if (er) h += errHTML({message: er.body.error, action: er.body.action, user: u.id});
    if (text || er) {
      const answers = kids(u.id);
      const curA = t.rest[0] ? t.rest[0].id : null;
      const tok = meta.tokens ? (meta.tokens.prompt || 0) + (meta.tokens.completion || 0) : 0;
      const info = [meta.model, tok ? (tok >= 1000 ? (tok / 1000).toFixed(1) + "k" : tok) + " tokens" : ""].filter(Boolean).join(" · ");
      h += '<div class="ak-meta">' + branchHTML(answers.findIndex((x) => x.id === curA), answers.length, "ask:abranch", u.id) +
        '<button class="act" data-act="ask:again" data-id="' + esc(u.id) + '" aria-label="Retry">' + I.retry + "</button>" +
        (text ? '<button class="act" data-act="ask:copy" data-text="' + esc(plainText(text)) + '" aria-label="Copy">' + I.copy + "</button>" : "") +
        '<span class="info">' + esc(info) + "</span></div>";
    }
    return h;
  }

  // ---- game blocks (blocks.js) ----------------------------------------------------------------------------------
  const reducedMotion = () => { try { return matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; } };
  const ui = (key) => { if (!st.blocks.has(key)) st.blocks.set(key, {i: 0, playing: false}); return st.blocks.get(key); };
  // Asks the server for a block's boards once; a failure stays until Try again.
  function need(req, auto) {
    if (!req || st.frames.has(req.key)) return;
    st.frames.set(req.key, {pending: true});
    if (auto) ui(req.key).auto = true;
    const m = {t: "ask_frames", key: req.key, record: req.record};
    if (req.turn != null) m.turn = req.turn; else m.decision = req.decision;
    queueMicrotask(() => send(m));
  }
  function blocksHTML(calls, last) {
    const bs = blockSpecs(calls);
    if (!bs.length) return "";
    let h = '<div class="ab-blocks">';
    for (const b of bs) {
      need(framesNeeded(b), last && b.kind === "seq");
      if (b.kind === "bot" && ui(b.key).watch) need({key: "t:" + b.r.id + ":" + b.r.turn, record: b.r.id, turn: b.r.turn}, false);
      h += blockHTML(b, (k) => st.frames.get(k), ui);
    }
    return h + "</div>";
  }
  // One timer steps every playing sequence (1.6 s a play) and stops at each one's last play.
  function playTimer() {
    if (seqTimer) return;
    seqTimer = setInterval(() => {
      const any = stepSeqs(st.blocks, (k) => st.frames.get(k));
      if (!any) { clearInterval(seqTimer); seqTimer = 0; }
      render();
    }, 1600);
  }

  function liveHTML(L) {
    let h = "";
    if (L.q != null) h += '<div class="ak-qwrap"><div class="ak-q">' + sentChips(L.attach) + esc(L.q) + "</div></div>";
    h += trailHTML("live", L.calls, {open: st.trailOpen.has("live"), outs: st.outs, running: true, limit: L.limit});
    if (L.text) h += '<div class="ak-a">' + mdHTML(L.text, cardsIn(L.calls.map((c) => c.result)), true) + "</div>";
    const word = L.retry ? "Provider busy: trying again in " + L.retry.secs + " s (try " + L.retry.attempt + ")" : L.status || L.word;
    h += '<div class="ak-think" role="status">' + DMARK + '<span class="w" data-key="' + esc(word) + '">' + esc(word) + "…</span></div>";
    return h;
  }

  function threadHTML() {
    let h = '<div class="ak-thread" aria-live="off">';
    const L = liveHere() || (st.live && !st.chat) ? st.live : null;
    const leaf = L ? L.base : st.chat ? st.chat.head : null;
    const rows = leaf ? pathTo(leaf) : [];
    const ts = turns(rows);
    if (!ts.length && !L) {
      h += '<div class="ak-empty"><div class="k">Ask</div><p>Questions about cards, rulings, your decks and your games.<br>' +
        "Answers use the app's card data and your history.</p></div>";
      if (st.settings && !st.err) {
        h += '<div class="ak-sugg">' + SUGGEST.map(([q, d]) => '<button data-act="ask:sugg" data-q="' + esc(q) + '">' + esc(q) +
          "<span>" + d + "</span></button>").join("") + "</div>";
      }
    }
    ts.forEach((t, i) => { h += '<div class="ak-turn" data-key="' + esc(t.user.id) + '">' + turnHTML(t, i === ts.length - 1, L && L.q == null && L.base === t.user.id) + "</div>"; });
    if (L) h += '<div class="ak-turn" data-key="live">' + liveHTML(L) + "</div>";
    if (st.err) h += errHTML(st.err);
    if (st.chat && !L && st.chat.head == null && st.rows.length) h += '<p class="ak-help">This chat was moved back to its start: ask again.</p>';
    return h + "</div>";
  }

  function listHTML() {
    const q = st.q.trim().toLowerCase();
    const chats = (st.chats || []).filter((c) => !q || (c.title || "").toLowerCase().includes(q));
    let h = '<div class="ak-list"><label class="ak-search">' + '<span class="sr-only">Search chats</span><input id="akq" type="search" placeholder="Search chats" value="' +
      esc(st.q) + '" autocomplete="off"></label>';
    if (st.chats === null) h += '<p class="ak-help">Loading chats…</p>';
    else if (!chats.length) h += '<p class="ak-help">' + (q ? "No chat matches." : "No chats yet: ask something and it is kept here.") + "</p>";
    const day = (s) => (s || "").slice(0, 10);
    const today = new Date().toISOString().slice(0, 10);
    let grp = null;
    for (const c of chats) {
      const g = day(c.updated) === today ? "Today" : "Earlier";
      if (g !== grp) { grp = g; h += '<div class="ak-grp">' + g + "</div>"; }
      const tok = c.tokens ? (c.tokens >= 1000 ? (c.tokens / 1000).toFixed(1) + "k" : c.tokens) + " tokens" : "";
      const armed = st.delArm === c.id;
      h += '<div class="ak-chat' + (st.chat && st.chat.id === c.id ? " sel" : "") + '"><button class="open" data-act="ask:open" data-id="' + esc(c.id) +
        '"><span class="ti">' + esc(c.title || "Chat") + '</span><span class="mi">' + esc([c.model, tok].filter(Boolean).join(" · ")) +
        (st.running === c.id ? ' · <span class="live">answering</span>' : "") + "</span></button>" +
        '<button class="del' + (armed ? " arm" : "") + '" data-act="ask:del" data-id="' + esc(c.id) + '" aria-label="' +
        (armed ? "Tap again to delete" : "Delete chat") + '">' + (armed ? "Delete?" : I.trash) + "</button></div>";
    }
    return h + "</div>";
  }

  function settingsHTML() {
    const s = st.settings;
    let h = '<div class="ak-set"><div class="ak-sh"><button class="ak-ibtn" data-act="ask:back" aria-label="Back to the chat">' + I.back +
      "</button><h2>Ask settings</h2></div>";
    if (!s) return h + '<p class="ak-help">Loading…</p></div>';
    h += '<p class="ak-help">Ask uses your own AI provider. Keys stay in this device\'s keychain and are never shown again.</p>';
    if (s.key_store && !s.key_store.persistent) {
      h += errHTML({message: "No keychain found on this device: a key you add lasts only until the app closes.", action: "none"});
    }
    h += '<div class="ak-sec" id="ak-provs">Provider</div><div role="ak-radiogroup" aria-labelledby="ak-provs">';
    for (const p of s.presets || []) {
      const on = p.id === s.preset;
      h += '<button class="ak-prov" role="radio" aria-checked="' + on + '" data-act="ask:preset" data-preset="' + esc(p.id) + '"><span class="ak-radio' +
        (on ? " on" : "") + '"></span><span><span class="pn">' + esc(p.name) + '</span><span class="pd">' + esc(p.note || "") + '</span></span><span class="kh">' +
        (p.key ? "key " + esc(p.key) : p.needs_key ? "no key" : "") + "</span></button>";
    }
    h += "</div>";
    const p = presetOf(s.preset);
    if (p) {
      h += '<div class="ak-sec">' + esc(p.name) + "</div>";
      if (p.id === "custom") {
        h += '<label class="ak-field"><span class="fl">Address (OpenAI-compatible, e.g. http://192.0.2.20:11434/v1)</span><input id="akbase" type="url" value="' +
          esc(s.base_url || "") + '" autocomplete="off" spellcheck="false"></label><div class="ak-rowbtns"><button class="ak-btn" data-act="ask:baseurl">Save address</button></div>';
      }
      if (p.needs_key || p.id === "custom") {
        h += '<label class="ak-field"><span class="fl">' + (p.key ? "Replace the key (now " + esc(p.key) + ")" : "API key") +
          '</span><input id="akkey" type="password" autocomplete="off" spellcheck="false" placeholder="Paste the key"></label>' +
          '<div class="ak-rowbtns"><button class="ak-btn pri" data-act="ask:keyset" data-preset="' + esc(p.id) + '">Save key</button>' +
          (p.key ? '<button class="ak-btn" data-act="ask:keydel" data-preset="' + esc(p.id) + '">Remove key</button>' : "") + "</div>";
        if (p.keys_url) h += '<p class="ak-help">Get a key at <a href="' + esc(p.keys_url) + '" target="_blank" rel="noopener">' + esc(p.keys_url.replace(/^https?:\/\//, "")) + "</a></p>";
      }
      if (st.keyMsg) h += '<p class="ak-help ' + (st.keyMsg.err ? "bad" : "good") + '" role="status">' + esc(st.keyMsg.err || st.keyMsg.ok) + "</p>";
      h += '<div class="ak-sec">Model</div><button class="ak-field btnf" data-act="ask:model"><span>' + esc(s.model || "Pick a model") +
        '</span><span class="r">Change</span></button>';
    }
    return h + "</div>";
  }

  // A provider's state in one line: whether it can be used now.
  function provState(p) {
    if (p.id === "device") {
      const ready = ((st.device && st.device.models) || []).filter((m) => m.state === "ready").map((m) => m.name);
      return ready.length ? ready.join(", ") + " installed" : "Runs on your phone: download a model first";
    }
    if (p.id === "custom") return st.settings && st.settings.base_url ? st.settings.base_url : "Any OpenAI-compatible address";
    return p.needs_key ? (p.key ? "Key saved" : "No key yet") : "No key needed";
  }
  function providerHTML() {
    const s = st.settings || {};
    const q = (st.provQ || "").trim().toLowerCase();
    const list = (s.presets || []).filter((p) => !q || (p.name + " " + (p.note || "")).toLowerCase().includes(q));
    let h = '<div class="ak-scrim" data-act="ask:closesheet"></div><div class="ak-bsheet" role="dialog" aria-modal="true" aria-label="Provider" data-close="ask:provback">' +
      '<div class="grab"></div><h2>Provider</h2><label class="ak-search ak-asearch"><span class="sr-only">Search providers</span><input id="akprovq" type="search" placeholder="Search providers" value="' +
      esc(st.provQ || "") + '" autocomplete="off" spellcheck="false"></label><div class="ak-models">';
    if (!list.length) h += '<p class="ak-help">No provider matches “' + esc(st.provQ) + "”.</p>";
    for (const p of list) {
      const on = p.id === st.sheetPreset;
      h += '<button class="ak-mrow" data-act="ask:ptab" data-preset="' + esc(p.id) + '" aria-pressed="' + on + '"><span class="ak-radio' + (on ? " on" : "") +
        '"></span><span><span class="mn">' + esc(p.name) + '</span><span class="ml">' + esc(provState(p) + (p.note && p.needs_key ? " · " + p.note : "")) + "</span></span></button>";
    }
    return h + "</div></div>";
  }

  function sheetHTML() {
    if (!st.sheet) return "";
    if (st.sheet === "provider") return providerHTML();
    if (st.sheet.attach) return attachSheetHTML(st.sheet, st.attach.length >= MAX_ATTACH, st.attach.some((x) => x.kind === "photo"));
    if (st.sheet.find) {
      const f = st.sheet;
      let h = '<div class="ak-scrim" data-act="ask:closesheet"></div><div class="ak-bsheet" role="dialog" aria-modal="true" aria-label="Which card" data-close="ask:closesheet">' +
        '<div class="grab"></div><h2>Which card is “' + esc(f.find) + '”?</h2><p class="ak-help">The answer named a card that is not an exact card name. Pick it to see its details.</p><div class="ak-models">';
      if (f.err) h += errHTML({message: f.err, action: "none"});
      else if (f.rows === null) h += '<div class="ak-think">' + DMARK + '<span class="w">Searching the cards…</span></div>';
      else if (!f.rows.length) h += '<p class="ak-help">No card matches “' + esc(f.find) + '”.</p>';
      for (const r of f.rows || []) {
        h += '<button class="ak-mrow" data-act="ask:card" data-code="' + r.code + '"><span><span class="mn">' + esc(r.name) +
          '</span><span class="ml">' + esc(r.line || "") + "</span></span></button>";
      }
      return h + "</div></div>";
    }
    if (st.sheet.menu) {
      const r = byId().get(st.sheet.menu);
      if (!r) return "";
      return '<div class="ak-scrim" data-act="ask:closesheet"></div><div class="ak-bsheet" role="menu" aria-label="Your question" data-close="ask:closesheet">' +
        '<div class="grab"></div><button role="menuitem" data-act="ask:editq" data-id="' + esc(r.id) + '">' + I.edit +
        "Edit and ask again<span>keeps this version: switch with ‹ ›</span></button>" +
        '<button role="menuitem" data-act="ask:copy" data-text="' + esc(r.body.content || "") + '">' + I.copy + "Copy</button></div>";
    }
    const s = st.settings;
    let h = '<div class="ak-scrim" data-act="ask:closesheet"></div><div class="ak-bsheet" role="dialog" aria-modal="true" aria-label="Model" data-close="ask:closesheet"><div class="grab"></div><h2>Model</h2>';
    if (!s) return h + '<p class="ak-help">Loading…</p></div>';
    // The provider is a select row opening a searchable picker, not pills (user, 2026-11-15: "a selection menu with
    // search in case we add many more providers"; mocks model-provider / provider-pick).
    const cur = presetOf(st.sheetPreset);
    h += '<button class="ak-prow" data-act="ask:provider" aria-haspopup="dialog" aria-label="Provider: ' + esc(cur ? cur.name : "none") +
      '. Change"><span class="pk">Provider</span><span class="pv"><span class="pn">' + esc(cur ? cur.name : "Pick a provider") +
      '</span><span class="ps">' + esc(cur ? provState(cur) : "") + "</span></span>" + I.next + "</button>";
    if (st.sheetPreset === "device") return h + deviceHTML(s) + "</div>";
    const p = presetOf(st.sheetPreset);
    const m = st.models[st.sheetPreset] || {};
    h += '<div class="ak-models">';
    if (p && p.needs_key && !p.key) h += errHTML({message: "No " + p.name + " key yet: add it in Settings", action: "add_key"});
    else if (m.pending) h += '<div class="ak-think">' + DMARK + '<span class="w">Fetching the model list…</span></div>';
    else if (m.err) h += errHTML(m.err);
    for (const name of m.list || []) {
      const on = s.preset === st.sheetPreset && s.model === name;
      h += '<button class="ak-mrow" data-act="ask:pick" data-preset="' + esc(st.sheetPreset) + '" data-model="' + esc(name) + '" aria-pressed="' + on +
        '"><span class="ak-radio' + (on ? " on" : "") + '"></span><span class="mn">' + esc(name) + "</span>" +
        (p && name === p.default_model ? '<span class="badge">default</span>' : "") + "</button>";
    }
    h += '</div><label class="ak-field"><span class="fl">Or type a model name</span><span class="ak-inrow"><input id="akmodel" autocomplete="off" spellcheck="false" placeholder="' +
      esc((p && p.default_model) || "model id") + '"><button class="ak-btn" data-act="ask:pickother">Use</button></span></label>' +
      '<button class="ak-btn wide" data-act="ask:settings">' + I.key + "Providers and keys</button></div>";
    return h;
  }

  // On this device (mocks local-check / local-downloading): the phone check, each model with its download, Wi-Fi only.
  function deviceHTML(s) {
    const d = st.device;
    if (!d) return '<div class="ak-think">' + DMARK + '<span class="w">Checking this phone…</span></div>';
    const c = d.device || {};
    const gb = (b) => (b / 1e9).toFixed(1);
    const row = (ok, t) => '<div class="dr ' + (ok ? "ok" : "no") + '">' + (ok ? I.check : I.warn) + "<span>" + esc(t) + "</span></div>";
    let h = '<div class="ak-dcheck"><div class="dh">This phone' + (c.model ? " · " + esc(c.model) : "") + "</div>";
    h += row(c.gpu, c.gpu ? "GPU: OpenCL found: Gemma runs on the GPU" : "GPU: no OpenCL driver found: Gemma cannot run on this phone. Keep a cloud provider");
    if (c.memory) h += row(c.memory >= 6e9, "Memory: " + Math.round(c.memory / 1e9) + " GB (E4B needs about 4 GB while it answers)");
    if (c.free != null) h += row(c.free >= 5e9, "Storage: " + gb(c.free) + " GB free");
    h += "</div>";
    if (st.devErr) h += errHTML({message: st.devErr.message, action: "none"});
    for (const m of d.models || []) {
      const on = s.preset === "device" && s.model === m.id;
      const ready = m.state === "installed";
      const pct = m.total ? Math.floor((100 * m.bytes) / m.total) : 0;
      let sub = esc(gb(m.total)) + " GB · " + esc(m.note);
      if (m.state === "downloading") sub = gb(m.bytes) + " of " + gb(m.total) + " GB";
      else if (m.state === "waiting_wifi") sub = gb(m.bytes) + " of " + gb(m.total) + " GB · waiting for Wi-Fi";
      else if (m.state === "paused") sub = "Paused at " + gb(m.bytes) + " of " + gb(m.total) + " GB";
      const btn = (dv, label) => '<button class="ak-btn" data-act="ask:dev" data-do="' + dv + '" data-model="' + esc(m.id) + '">' + label + "</button>";
      const act = ready ? btn("remove", "Remove") : m.state === "downloading" || m.state === "waiting_wifi" ? btn("pause", "Pause")
        : c.gpu ? btn("download", m.state === "paused" || m.error ? "Resume" : "Download") : "";
      const name = '<span class="mn">' + esc(m.name) + (ready ? '<span class="badge">installed</span>' : "") + "</span>";
      const bar = m.bytes && !ready ? '<span class="ak-bar" role="progressbar" aria-label="' + esc(m.name) + ' download" aria-valuenow="' + pct +
        '" aria-valuemin="0" aria-valuemax="100"><i style="width:' + pct + '%"></i></span>' : "";
      h += '<div class="ak-mrow dev">' + (ready ? '<button class="ak-pick" data-act="ask:pick" data-preset="device" data-model="' + esc(m.id) +
        '" aria-pressed="' + on + '"><span class="ak-radio' + (on ? " on" : "") + '"></span>' : '<span class="ak-pick">') +
        "<span>" + name + '<span class="ml">' + sub + "</span>" + bar + "</span>" + (ready ? "</button>" : "</span>") + act + "</div>";
      if (m.error) h += errHTML({message: m.error, action: "none"});
    }
    h += '<button class="ak-mrow ak-tog" role="switch" aria-checked="' + !!d.wifi_only + '" data-act="ask:dev" data-do="wifi">' +
      '<span class="mn">Download on Wi-Fi only</span><span class="sw' + (d.wifi_only ? " on" : "") + '"><i></i></span></button>';
    h += '<p class="ak-help">Runs on your phone. The first question after a while loads Gemma on the GPU (about 20 s).</p>';
    return h;
  }

  function composerHTML() {
    const busy = !!st.running;
    const editing = st.edit ? '<div class="ak-editing">Editing your question: the old version and its answer stay ' +
      '<button class="ak-link" data-act="ask:canceledit">Cancel</button></div>' : "";
    // Always in the thread, shown by the scroll listener without a render (see "scroll" below).
    const jump = st.view === "thread" ? '<button class="ak-jump" data-act="ask:bottom" aria-label="Jump to the latest message"' +
      (st.up ? "" : " hidden") + ">" + I.down + "</button>" : "";
    const chips = st.attach.length ? '<div class="ak-att">' + st.attach.map((x, i) => chipHTML(x, i, x.thumb)).join("") + "</div>" : "";
    // The model in use, under the question (user 2026-11-16: "add the pill for the model selection like in the attached
    // figure, that makes it more intuitive which model is being used and how to change it").
    const s = st.settings;
    const model = s ? '<button class="ak-model" data-act="ask:model" aria-haspopup="dialog" aria-label="Model: ' + esc(modelLine()) + ', change">' +
      '<span>' + esc(s.model || "Pick a model") + "</span>" + I.down + "</button>" : "";
    return '<div class="ak-composer" data-feature="ask">' + jump + editing + '<div class="ak-box">' + chips + model +
      '<button class="ak-plus" data-act="ask:attach" aria-haspopup="dialog" aria-label="Attach a deck, game, card or photo">' + I.plus + "</button>" +
      '<label class="sr-only" for="akin">Ask</label>' +
      '<textarea id="akin" rows="1" placeholder="Ask about cards" enterkeyhint="send">' + esc(st.draft) + "</textarea>" +
      (busy ? '<button class="ak-send stop" data-act="ask:stop" aria-label="Stop the answer">' + I.stop + "</button>"
        : '<button class="ak-send" data-act="ask:send" aria-label="Send"' + (st.draft.trim() || st.attach.length ? "" : " disabled") + ">" + I.send + "</button>") + "</div></div>";
  }

  return {
    onMessage,
    act,
    // The tab opened (or the socket came back): settings, chats and the open chat from the server.
    enter() {
      send({t: "ask_settings", op: "get"});
      send({t: "ask_list"});
      if (st.chat && st.chat.id) send({t: "ask_open", chat: st.chat.id});
      // A board asked for on the old socket never arrives: ask again.
      for (const [k, v] of st.frames) if (v.pending) st.frames.delete(k);
      entered = true;
    },
    entered: () => entered,
    // A question from another screen (Games: "Ask about this game"): a new chat with the question in the composer,
    // for the user to edit and send (nothing is sent, so no tokens are spent until they do).
    compose(text) {
      st.fresh = true;
      st.chat = null;
      st.rows = [];
      st.view = "thread";
      st.sheet = null;
      st.edit = null;
      st.attach = [];
      st.draft = text;
      requestAnimationFrame(() => { const box = document.getElementById("akin"); if (box) { box.value = text; box.focus(); } });
    },
    // The scrolling part of the tab, the composer under it, and the sheet over it.
    panelHTML() {
      const body = st.view === "settings" ? settingsHTML() : st.view === "list" ? listHTML() : threadHTML();
      return '<div data-feature="ask" class="ak">' + (st.view === "settings" ? "" : headHTML()) + body + "</div>";
    },
    footHTML: () => (st.view === "thread" ? composerHTML() : ""),
    sheetHTML: () => '<div data-feature="ask">' + sheetHTML() + "</div>",
    sheetOpen: () => !!st.sheet,
    viewOpen: () => st.view !== "thread",
    // Android back: the sheet, then Settings or the chat list.
    back() {
      if (st.sheet === "provider") { openModels(st.sheetPreset); return true; }
      if (st.sheet) { st.sheet = null; return true; }
      if (st.view !== "thread") { st.view = "thread"; return true; }
      return false;
    },
    // Keep the thread at the bottom while an answer streams (unless the user scrolled up to read).
    following: () => st.view === "thread" && !!st.live,
  };
}
