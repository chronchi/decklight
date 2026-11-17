// Games (Play > Games; mocks docs/ui/mocks/ask/mocks.html games, games-empty, game, game-decision, approved 2026-11-15):
// the user's saved games, their stats, each game's turns and the bot's view of every decision, without asking (user
// 2026-11-15: "whatever the llm has access to the user should also have access without having to ask"). The data
// comes from the server's ask_data (rust/decklight-server/src/ask/games.rs: the same reads as the replay, bot_view and
// my_stats tools) and ask_frames (the boards); the blocks are the chat's (../ask/blocks.js) under `games:` actions.
import {esc} from "../../render/card.js";
import {gameHTML, seqHTML, stepSeqs, seqAct, botHTML, grade, miniCard, phaseName} from "../ask/blocks.js";
import {retryStill} from "../../lib/stills.js";
import {exportSheet, SHARE} from "./export.js";
import {watchView} from "./watch.js";
import {morph} from "../../render/morph.js";

const FILTERS = [["all", "All"], ["bot", "Bot"], ["people", "People"], ["practice", "Practice"]];
const PAGE = 30;
// Zone and position choices: listed under a turn only on request (or when the check marked them).
const MINOR = new Set(["place", "position"]);
const STEP_MS = 1600;
const SAMPLES_KEY = "ygo-games-samples";
const KEY = "games:";      // ask_data keys
const FKEY = "games|";     // ask_frames keys (the Ask tab keeps its own)
const ICO = (d) => '<svg class="ak-ico" viewBox="0 0 24 24" aria-hidden="true">' + d + "</svg>";
const TRASH = ICO('<path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/>'), TICK = ICO('<path d="m6 12.5 4 4 8-9"/>');
const SEARCH_MS = 250;
const PLAYI = ICO('<path d="M8 5.5v13l10-6.5z"/>'), CHECKI = ICO('<circle cx="11" cy="11" r="6"/><path d="m15.5 15.5 4 4M8.5 11l2 2 3.5-4"/>');
const ASKI = ICO('<path d="M5 6h14v9H10l-4 3.5V15H5z"/>');
const BACK = ICO('<path d="m14.5 6-6 6 6 6"/>'), CH = ICO('<path d="m9.5 6 6 6-6 6"/>'), CD = ICO('<path d="m6 9.5 6 6 6-6"/>');

function ago(when) {
  const t = Date.parse(when);
  if (!isFinite(t)) return "";
  const s = (Date.now() - t) / 1000;
  if (s < 90) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " h ago";
  if (s < 172800) return "yesterday";
  return new Date(t).toLocaleDateString(undefined, {month: "short", day: "numeric"});
}
const went = (w) => (w === "first" ? "went 1st" : w === "second" ? "went 2nd" : "");
const vs = (g) => (g.your_deck || "Your deck") + " vs " + (g.opponent_deck || "opponent");
const kindWord = (g) => (g.kind === "practice" ? "Practice" : g.kind === "puzzle" ? "Puzzle" : "Duel");
const oppWord = (g) => (g.opponent === "remote" ? "Multiplayer" : "Bot");
const pts = (d) => "−" + Math.abs(Math.round(d));

// The small LP chart of a list row: you solid, the opponent dashed (the chat's game card in miniature).
function lpMini(lp) {
  if (!Array.isArray(lp) || lp.length < 2) return '<span class="gm-nolp"></span>';
  const top = Math.max(8000, ...lp.map((p) => Math.max(p[0] || 0, p[1] || 0)));
  const x = (i) => 2 + i * (60 / (lp.length - 1)), y = (v) => 2 + (1 - Math.max(0, v || 0) / top) * 22;
  const path = (k) => "M" + lp.map((p, i) => x(i).toFixed(1) + " " + y(p[k]).toFixed(1)).join(" L");
  return '<svg viewBox="0 0 64 26" aria-hidden="true"><path class="gm-o" d="' + path(1) + '"/><path class="gm-y" d="' + path(0) + '"/></svg>';
}

// Two panes from the expanded width class up (840 px + the 84 px rail).
const TWO = typeof matchMedia === "function" ? matchMedia("(min-width: 924px)") : {matches: false, addEventListener() {}};

export function gamesFeature(ctx) {
  TWO.addEventListener("change", () => ctx.render());
  // ctx = {send(msg), render(), mock: bool, play(), ask(text)}
  const st = {
    filter: "all",
    limit: PAGE,
    list: new Map(),     // "<filter>:<limit>" -> {pending} | {error} | result
    pages: new Map(),    // game id -> {pending} | {error} | page
    checks: new Map(),   // game id -> {progress} | {error} | result
    bots: new Map(),     // "<id>:<n>" -> {pending} | {error} | bot_view
    frames: new Map(),   // FKEY + "t:<id>:<turn>" | FKEY + "d:<id>:<n>" -> {pending} | {error} | ask_frames
    ui: new Map(),       // a sequence's play state by frames key: {i, playing}
    game: null,          // the open game's id
    turn: null,          // its open turn
    minor: false,        // the open turn lists zone and position choices too
    decision: null,      // the open decision (1-based) or null
    hideSamples: false,
    q: "",               // the search applied to the list (deck names)
    qInput: "",          // the search box as typed (applied after a pause)
    selecting: false,    // Select mode: rows tick instead of opening
    sel: new Set(),      // ticked game ids
    confirm: false,      // the "Delete N games?" dialog is open
    del: null,           // {pending} | {error} | {deleted} for the last delete; {pending} | {error} for Select all
    all: null,
  };
  let qTimer = 0, seq = 0;
  const xp = exportSheet({send: (m) => ctx.send(m), render: () => ctx.render(), mock: ctx.mock});
  // Watch (watch.js) sits over every tab; its Export opens the sheet inside Watch's own layer.
  const wv = watchView({send: (m) => ctx.send(m), mock: ctx.mock, act: (k, el) => act(k, el),
    exportFrom: (info) => { xp.open({ids: [info.id], info, turn: info.turn, play: info.play, what: "turn"}); ctx.render(); focusSoon(".gx h2"); }});
  try { st.hideSamples = localStorage.getItem(SAMPLES_KEY) === "hide"; } catch (e) { /* ignore */ }
  let timer = 0;
  const send = (m) => { if (!ctx.mock) queueMicrotask(() => ctx.send(m)); };
  const reduced = () => { try { return matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; } };
  const ui = (k) => { if (!st.ui.has(k)) st.ui.set(k, {i: 0, playing: false}); return st.ui.get(k); };

  function data(map, mkey, what, args) {
    if (map.has(mkey)) return map.get(mkey);
    map.set(mkey, {pending: true});
    send({t: "ask_data", key: KEY + what + ":" + mkey, what, args});
    return map.get(mkey);
  }
  const listNow = () => data(st.list, st.filter + ":" + st.limit + ":" + st.q, "list", {filter: st.filter, limit: st.limit, q: st.q});
  const pageOf = (id) => data(st.pages, id, "game", {record: id});
  function frames(key, req) {
    if (!st.frames.has(key)) {
      st.frames.set(key, {pending: true});
      send(Object.assign({t: "ask_frames", key}, req));
    }
    return st.frames.get(key);
  }
  function playTimer() {
    if (timer) return;
    timer = setInterval(() => {
      const any = stepSeqs(st.ui, (k) => st.frames.get(k));
      if (!any) { clearInterval(timer); timer = 0; }
      ctx.render();
    }, STEP_MS);
  }

  // ---- server messages ------------------------------------------------------------------------------------------
  function onMessage(m) {
    if (!m || typeof m.key !== "string") return false;
    if ((m.t === "ask_frames" || m.t === "ask_frames_pending") && m.key.startsWith(FKEY)) {
      if (m.t === "ask_frames") st.frames.set(m.key, m.error ? {error: m.error} : m);
      ctx.render();
      return true;
    }
    if ((m.t !== "ask_data" && m.t !== "ask_data_pending") || !m.key.startsWith(KEY)) return false;
    if (m.t === "ask_data_pending") return true;
    const rest = m.key.slice(KEY.length);
    const what = rest.slice(0, rest.indexOf(":"));
    const mkey = rest.slice(what.length + 1);
    if (what === "ids" || what === "delete") { onSelectReply(what, mkey, m); ctx.render(); return true; }
    if (what === "export") { xp.onReply(mkey, m); ctx.render(); return true; }
    if (what === "watch") { wv.onReply(mkey, m); ctx.render(); return true; }
    const map = {list: st.list, game: st.pages, check: st.checks, bot: st.bots}[what];
    if (!map) return true;
    if (m.progress) map.set(mkey, {progress: m.progress});
    else if (m.error) map.set(mkey, {error: m.error});
    else {
      map.set(mkey, m.result);
      // A finished check marks the decisions on the page and fills the bot's views.
      if (what === "check") {
        const p = st.pages.get(mkey);
        if (p && p.id) p.check = m.result;
      }
    }
    ctx.render();
    return true;
  }

  // Replies to Select all (ids) and Delete; only the latest request of each counts.
  function onSelectReply(what, mkey, m) {
    if (mkey !== String(seq)) return;
    if (what === "ids") {
      if (m.error) st.all = {error: m.error};
      else { st.all = null; st.sel = new Set(m.result.ids); }
      return;
    }
    // A delete that stopped midway may have deleted some: the list is asked again and the ticks cleared.
    if (m.error) { st.del = {error: m.error}; st.confirm = false; st.sel.clear(); st.list.clear(); return; }
    for (const id of st.sel) { st.pages.delete(id); st.checks.delete(id); }
    st.del = {deleted: m.result.deleted};
    st.sel.clear();
    st.selecting = false;
    st.confirm = false;
    st.list.clear();
    focusSoon(".gm-done");
  }
  function ask(what, args) {
    seq += 1;
    ctx.send({t: "ask_data", key: KEY + what + ":" + seq, what, args});
  }

  // ---- the list -------------------------------------------------------------------------------------------------
  const errHTML = (e, act, what) => '<div class="ab-fail" role="alert">Could not load ' + what + ": " + esc(e) +
    ' <button class="ak-btn" data-act="games:' + act + '">Try again</button></div>';
  const waitHTML = (t) => '<div class="ab-wait" role="status">' + t + "</div>";

  function row(g, sample) {
    const res = sample ? ["SAMPLE", "s"] : g.result === "you won" ? ["WON", "w"] : g.result === "you lost" ? ["LOST", "l"] : ["—", "u"];
    const meta = sample ? ["Bot vs bot", g.turns + " turns", went(g.you_went)]
      : [oppWord(g), kindWord(g), g.turns + " turns", went(g.you_went), ago(g.when)];
    const tag = !sample && g.kind === "practice" && st.filter !== "practice" ? '<span class="gm-tag">PRACTICE</span>' : "";
    const label = esc((sample ? "Sample game: " : res[0] + ": ") + vs(g) + ", " + meta.filter(Boolean).join(", "));
    // Select mode: the row ticks (a toggle button) instead of opening the game.
    const on = st.selecting && st.sel.has(g.id);
    const head = st.selecting && !sample
      ? '<button class="gm-row gm-pick' + (on ? " sel" : "") + '" data-act="games:tick" data-id="' + esc(g.id) + '" aria-pressed="' + on + '" aria-label="' + label + '">' +
        '<span class="gm-cb" aria-hidden="true">' + (on ? TICK : "") + "</span>"
      : '<button class="gm-row' + (String(g.id) === String(st.game) ? " cur" : "") + '" data-act="games:open"' + (sample ? "" : ' data-hold="games:hold"') +
        ' data-id="' + esc(g.id) + '" aria-label="' + label + '"' + (String(g.id) === String(st.game) ? ' aria-current="true"' : "") + ">";
    return head + '<span class="gm-res gm-' + res[1] + '" aria-hidden="true">' + res[0] + '</span><span class="gm-tx"><span class="gm-ti">' + esc(vs(g)) + tag +
      '</span><span class="gm-mi">' + esc(meta.filter(Boolean).join(" · ")) + "</span></span>" + lpMini(g.lp) + "</button>";
  }
  function statRow(label, t, odds) {
    const pct = t.win_rate == null ? 0 : t.win_rate;
    return '<div class="gm-sr"><span>' + esc(label) + '</span><span class="gm-n">' + t.wins + "–" + t.losses + '</span><span class="gm-bar" role="img" aria-label="' +
      (t.win_rate == null ? "no decided games" : pct + "% won") + '"><i style="width:' + pct + '%"></i></span></div>';
  }
  function statsHTML(s) {
    const t = s.total || {};
    const kind = st.filter === "practice" ? "practice games" : "practice and samples not counted";
    let h = '<div class="gm-stat"><div class="gm-big"><b>' + (t.wins || 0) + "–" + (t.losses || 0) + "</b><span>" +
      esc([t.win_rate != null ? t.win_rate + "% won" : "", "last " + s.days + " days", kind].filter(Boolean).join(" · ")) + "</span></div>";
    const group = (title, rows, name) => {
      const r = (rows || []).filter((x) => x.games > 0).sort((a, b) => b.games - a.games).slice(0, 5);
      if (!r.length) return "";
      return '<div class="gm-sub">' + title + "</div>" + r.map((x) => statRow(name(x.key), x)).join("");
    };
    h += group("Turn order", s.first, (k) => (k === "first" ? "Going first" : k === "second" ? "Going second" : "Not known"));
    h += group("Your decks", s.deck, (k) => k);
    h += group("Opponents", s.opponent_deck, (k) => k);
    return h + "</div>";
  }
  function samplesHTML(samples) {
    if (!samples || !samples.length) return "";
    if (st.hideSamples) return '<button class="ak-link gm-show" data-act="games:samples">Show the sample games</button>';
    return '<h3 class="gm-lbl">Sample games</h3><p class="gm-note">Example games that come with the app, so you can try the replay and the bot\'s view. They never count in your stats.</p>' +
      samples.map((g) => row(g, true)).join("") + '<button class="ak-btn gm-hide" data-act="games:samples">Hide sample games</button>';
  }
  // The search box and Select (mock games-search), or the selection bar in Select mode (mock games-select).
  function toolsHTML(L) {
    if (st.selecting) {
      const n = st.sel.size, total = L && L.total || 0;
      const allBtn = st.all && st.all.pending ? '<button class="ak-btn" disabled>Selecting…</button>'
        : n < total ? '<button class="ak-btn" data-act="games:selall">Select all ' + total + "</button>"
        : '<button class="ak-btn" data-act="games:selnone">Clear</button>';
      return '<div class="gm-selbar"><button class="ak-ibtn" data-act="games:selcancel" aria-label="Stop selecting">' + ICO('<path d="M7 7l10 10M17 7 7 17"/>') +
        '</button><span class="gm-selt" role="status">' + n + " selected</span>" + allBtn + "</div>" +
        (st.all && st.all.error ? '<div class="ab-fail" role="alert">Could not select all: ' + esc(st.all.error) + "</div>" : "");
    }
    return '<div class="gm-tools"><label class="gm-search"><span class="gm-sico" aria-hidden="true">' + ICO('<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5"/>') +
      '</span><input id="gsearch" type="search" autocomplete="off" spellcheck="false" enterkeyhint="search" placeholder="Search by deck or opponent" aria-label="Search your games by deck or opponent" value="' +
      esc(st.qInput) + '"></label></div>';
  }
  // Select sits on the "Your games" row, next to what it acts on (user 2026-11-15); a long press on a row also starts it.
  const selectBtn = (L) => (!st.selecting && L.games.length ? '<button class="ak-btn gm-selb" data-act="games:select">Select</button>' : "");
  function deleteHTML() {
    if (!st.selecting) return "";
    const n = st.sel.size;
    const word = n === 1 ? "1 game" : n + " games";
    let h = '<div class="gm-delbar"><button class="ak-btn" data-act="games:xpsel" aria-label="Export ' + word + '"' + (n ? "" : " disabled") + ">" + SHARE + " Export " + n + "</button>" +
      '<button class="ak-btn gm-dz" data-act="games:delask" aria-label="Delete ' + word + '"' + (n ? "" : " disabled") + ">" + TRASH + " Delete " + n + "</button></div>";
    if (st.confirm && n) {
      const busy = st.del && st.del.pending;
      h += '<div class="gm-scrim" data-act="games:delno"></div><div class="gm-dlg" role="alertdialog" aria-modal="true" aria-labelledby="gmdlgt" aria-describedby="gmdlgd" data-close="games:delno">' +
        '<h2 id="gmdlgt">Delete ' + word + "?</h2>" +
        '<p id="gmdlgd" class="gm-note">' + (n === 1 ? "It leaves" : "They leave") + " the list and stop counting in your stats, and Ask can no longer open " + (n === 1 ? "it" : "them") +
        ". The files move to a trash folder in the app's data on this device; they are not sent anywhere.</p>" +
        '<div class="gm-acts"><button class="ak-btn gm-dz" data-act="games:delyes"' + (busy ? " disabled" : "") + ">" + (busy ? "Deleting…" : "Delete") +
        '</button><button class="ak-btn" data-act="games:delno">Cancel</button></div></div>';
    }
    return h;
  }
  function listHTML() {
    const L = listNow();
    const ready = !L.pending && !L.error;
    let h = toolsHTML(ready ? L : null);
    if (!st.selecting) {
      h += '<div class="gm-chips" role="group" aria-label="Which games">' + FILTERS.map(([k, l]) =>
        '<button data-act="games:filter" data-f="' + k + '" aria-pressed="' + (st.filter === k) + '"' + (st.filter === k ? ' class="sel"' : "") + ">" + l + "</button>").join("") + "</div>";
    }
    if (st.del && st.del.deleted != null && !st.selecting) {
      h += '<p class="gm-note gm-done" role="status" tabindex="-1">Deleted ' + (st.del.deleted === 1 ? "1 game" : st.del.deleted + " games") + ".</p>";
    }
    if (st.del && st.del.error) {
      h += '<div class="ab-fail" role="alert">The delete stopped: ' + esc(st.del.error) + ". The list below is up to date: tick what is left and try again.</div>";
    }
    if (L.pending) return h + waitHTML("Loading your games…");
    if (L.error) return h + errHTML(L.error, "reload", "your games");
    // A search lists the matches only: no stat card, no samples.
    if (st.q) {
      h += '<div class="gm-lblrow"><p class="gm-note" role="status">' + (L.total ? (L.total === 1 ? "1 of your games matches" : L.total + " of your games match") : "None of your games match") +
        " “" + esc(st.q) + "” (your deck or the opponent's)." + "</p>" + selectBtn(L) + "</div>";
      h += L.games.map((g) => row(g, false)).join("");
      if (L.more) h += '<button class="ak-btn gm-more" data-act="games:more">Show ' + PAGE + " more</button>";
      return h + deleteHTML();
    }
    const none = !L.games.length && !(L.stats.total || {}).games;
    if (none && st.filter === "all") {
      h += '<div class="gm-stat"><div class="gm-big"><b>0–0</b><span>no games yet</span></div><p class="gm-note">Play the bot or a friend: every game is saved here with its replay, and your win rate by deck, turn order and opponent fills in.</p>' +
        '<div class="gm-acts"><button class="ak-btn pri" data-act="games:play">Play the bot</button></div></div>';
      return h + samplesHTML(L.samples);
    }
    h += statsHTML(L.stats);
    if ((L.stats.total || {}).games) h += '<div class="gm-acts"><button class="ak-btn" data-act="games:askstats">Ask about my stats</button></div>';
    h += '<div class="gm-lblrow"><h3 class="gm-lbl">Your games</h3>' + selectBtn(L) + "</div>";
    h += L.games.length ? L.games.map((g) => row(g, false)).join("") : '<p class="gm-note">No games here yet.</p>';
    if (L.more) h += '<button class="ak-btn gm-more" data-act="games:more">Show ' + PAGE + " more</button>";
    // Samples come with the app: they can be hidden, not deleted, so Select mode leaves them out.
    return h + (st.selecting ? "" : samplesHTML(L.samples)) + deleteHTML();
  }

  // ---- a game ---------------------------------------------------------------------------------------------------
  function headHTML(title, sub, act, extra) {
    return '<div class="gm-head"><button class="ak-ibtn" data-act="games:' + act + '" aria-label="Back">' + BACK + '</button><div class="gm-ht"><h2>' + esc(title) +
      '</h2><div class="gm-hs">' + esc(sub) + "</div></div>" + (extra || "") + "</div>";
  }
  const subOf = (p) => (p.sample ? "Sample game · bot vs bot" : [p.opponent === "remote" ? "Multiplayer" : "Bot", ago(p.when)].filter(Boolean).join(" · "));
  function plays(t) {
    const ev = t.events || [];
    const last = ev[ev.length - 1];
    const m = typeof last === "string" && /^\.\.\. (\d+) more/.exec(last);
    return m ? ev.length - 1 + +m[1] : ev.length;
  }
  function marks(p) {
    const out = new Map();
    for (const m of (p.check && p.check.decisions) || []) out.set(m.decision, m);
    return out;
  }
  function decRow(d, mk) {
    const bad = mk && grade(mk.delta);
    const tag = bad ? '<span class="gm-mk bad">' + pts(mk.delta) + " · " + bad + "</span>" : mk ? '<span class="gm-mk">OK</span>' : CH;
    return '<button class="gm-dec" data-act="games:decision" data-n="' + d.decision + '" aria-label="' +
      esc("Decision " + d.decision + ": " + (d.text || d.prompt) + ". You: " + d.you_picked + (bad ? ". " + bad + ", " + pts(mk.delta) + " points" : "")) + '">' +
      '<span class="gm-dn" aria-hidden="true">D' + d.decision + '</span><span class="gm-tx"><span>' + esc(d.text || phaseName(d.phase)) +
      '</span><span class="gm-pk">You: ' + esc(d.you_picked) + "</span></span>" + tag + "</button>";
  }
  // The game's actions as equal tiles under the game card (user 2026-11-16 picked mock D, docs/ui/mocks/watch-button: "you
  // can go from the game to the ask tab at the same time, that's very powerful"): Watch (the accent), Check my decisions
  // (its progress, then done) and Ask about this game. A game with no decisions of yours has no check.
  function actsHTML(p) {
    const c = st.checks.get(p.id);
    const n = (p.decisions || []).length;
    // aria: a longer name that starts with the visible label (WCAG 2.5.3, label in name).
    const tile = (attrs, ico, label, aria, cls) => '<button class="gm-tile' + (cls ? " " + cls : "") + '" ' + attrs +
      (aria ? ' aria-label="' + esc(aria) + '"' : "") + ">" + ico + "<span>" + label + "</span></button>";
    let h = '<div class="gm-tiles">' + tile('data-act="games:watch" data-id="' + esc(p.id) + '" data-turn="1"', PLAYI, "Watch", "Watch the replay", "pri");
    if (n && p.check) h += tile("disabled", TICK, "Checked", "");
    else if (n && c && c.progress) h += tile("disabled", CHECKI, "Checking… <small>" + c.progress.done + " / " + c.progress.total + "</small>", "");
    else if (n) h += tile('data-act="games:check"', CHECKI, "Check decisions", "");
    h += tile('data-act="games:askgame"', ASKI, "Ask", "Ask about this game") + "</div>";
    if (c && c.error) h += '<div class="ab-fail" role="alert">The check stopped: ' + esc(c.error) + "</div>";
    if (p.check) {
      const bad = p.check.decisions.filter((m) => grade(m.delta)).length;
      h += '<p class="gm-note" role="status">Checked: ' + (bad ? bad + " of your " + n + " decisions lost win chance (marked below)." : "none of your " + n + " decisions lost much win chance.") + "</p>";
    } else if (n && !(c && c.progress)) {
      h += '<p class="gm-note">Check decisions runs the bot\'s view on each of your ' + n + " decisions and marks the ones that lost the most win chance.</p>";
    }
    return h;
  }
  function gameHTMLPage() {
    const p = pageOf(st.game);
    if (p.pending) return headHTML("Game", "", "back") + waitHTML("Replaying the game…");
    if (p.error) return headHTML("Game", "", "back") + errHTML(p.error, "reloadgame", "this game");
    let h = headHTML(vs(p), subOf(p), "back", '<button class="ak-ibtn gm-xp" data-act="games:xpgame" aria-label="Export this game">' + SHARE + "</button>");
    const open = st.turn;
    h += gameHTML({r: p, hot: new Set(open != null ? [open] : [])}, "the open turn");
    // Watch opens the game in the play screen from turn 1 (a turn's still opens it at that play).
    h += actsHTML(p);
    const mk = marks(p);
    h += '<h3 class="gm-lbl">Turns</h3>';
    for (const t of p.log || []) {
      const ds = (p.decisions || []).filter((d) => d.turn === t.turn);
      const isOpen = open === t.turn;
      const worst = ds.map((d) => mk.get(d.decision)).filter((m) => m && grade(m.delta)).length;
      const lp = t.lp_at_start || [];
      h += '<button class="gm-turn' + (isOpen ? " open" : "") + '" data-act="games:turn" data-turn="' + t.turn + '" aria-expanded="' + isOpen + '">' +
        '<span class="gm-tn">T' + t.turn + "</span><span>" + (t.player === "you" ? "You" : "Opponent") + ' <span class="gm-who">· ' + plays(t) + " plays" +
        (ds.length ? " · decision" + (ds.length > 1 ? "s " + ds[0].decision + "–" + ds[ds.length - 1].decision : " " + ds[0].decision) : "") +
        (worst ? " · " + worst + " marked" : "") + '</span></span><span class="gm-lp">' + lp[0] + " · " + lp[1] + " " + (isOpen ? CD : CH) + "</span></button>";
      if (isOpen) {
        const fk = FKEY + "t:" + p.id + ":" + t.turn;
        const fr = frames(fk, {record: p.id, turn: t.turn});
        h += seqHTML({kind: "seq", key: fk, id: p.id, turn: t.turn, xp: "games:xpturn", xpIcon: SHARE}, fr.pending ? null : fr, ui(fk), "games");
        const shown = st.minor ? ds : ds.filter((d) => !MINOR.has(d.prompt) || grade((mk.get(d.decision) || {}).delta || 0));
        const hidden = ds.length - shown.length;
        if (shown.length) h += '<div class="gm-decs">' + shown.map((d) => decRow(d, mk.get(d.decision))).join("") + "</div>";
        if (hidden || st.minor) {
          h += '<button class="ak-link gm-minor" data-act="games:minor">' + (st.minor ? "Hide zone and position choices"
            : "Show " + hidden + " zone and position choice" + (hidden > 1 ? "s" : "")) + "</button>";
        }
      }
    }
    return h;
  }

  // ---- a decision -----------------------------------------------------------------------------------------------
  function decisionHTML() {
    const p = pageOf(st.game);
    if (!p.id) return gameHTMLPage();
    const ds = p.decisions || [];
    const n = st.decision;
    const d = ds[n - 1];
    if (!d) { st.decision = null; return gameHTMLPage(); }
    let h = headHTML("Decision " + n, "Turn " + d.turn + " · " + phaseName(d.phase) + " · " + vs(p), "backgame");
    if (d.text) h += '<p class="gm-q">' + esc(d.text) + "</p>";
    const fk = FKEY + "d:" + p.id + ":" + n;
    const fr = frames(fk, {record: p.id, decision: n});
    const fromCheck = marks(p).get(n);
    const bv = fromCheck ? fromCheck.view : data(st.bots, p.id + ":" + n, "bot", {record: p.id, decision: n});
    const acts = '<button class="ak-btn" data-act="games:askwhy">Ask why</button>';
    if (bv.pending) h += waitHTML("The bot is looking at this decision…");
    else if (bv.error) h += errHTML(bv.error, "reloadbot", "the bot's view");
    else h += botHTML({key: "gb:" + p.id + ":" + n, r: bv}, fr.pending ? null : fr, {}, null, {ns: "games", acts, key: fk});
    h += '<h3 class="gm-lbl">All your options here</h3><div class="gm-opts">';
    for (const o of d.options || []) {
      const you = o.index === d.picked_index;
      h += '<div class="gm-opt' + (you ? " you" : "") + '"><span class="gm-th' + (o.code ? "" : " none") + '">' + (o.code ? miniCard(o.code, o.label) : CH) + "</span><span>" + esc(o.label) +
        "</span>" + (you ? '<span class="ab-tag ab-you">YOUR PICK</span>' : "<span></span>") + "</div>";
    }
    h += "</div>";
    h += '<div class="gm-acts gm-nav">' + (n > 1 ? '<button class="ak-btn" data-act="games:decision" data-n="' + (n - 1) + '">‹ Decision ' + (n - 1) + "</button>" : "<span></span>") +
      (n < ds.length ? '<button class="ak-btn" data-act="games:decision" data-n="' + (n + 1) + '">Decision ' + (n + 1) + " ›</button>" : "") + "</div>";
    return h;
  }

  // ---- actions --------------------------------------------------------------------------------------------------
  function act(kind, el) {
    const a = kind.slice(KEY.length);
    const attr = (k) => el && el.getAttribute(k);
    if (a.startsWith("xp")) { exportAct(a, el); ctx.render(); return; }
    // Escape and back on Watch: the export sheet over it first.
    if (a === "wclose") { if (!xp.close()) wv.close(); ctx.render(); return; }
    if (a === "watch" || a === "watchat") {
      const id = attr("data-id") || st.game;
      wv.open(id, a === "watchat" ? {choice: +attr("data-choice")} : {turn: +attr("data-turn") || 1});
      ctx.render();
      return;
    }
    if (a === "still") { retryStill(attr("data-key")); ctx.render(); return; }
    if (a === "filter") { st.filter = attr("data-f") || "all"; st.limit = PAGE; seq += 1; st.all = null; }
    else if (a === "more") st.limit += PAGE;
    else if (a === "reload") st.list.delete(st.filter + ":" + st.limit + ":" + st.q);
    else if (a === "select" || a === "hold") {
      // A new Select session: replies to the last one's Select all are dropped (seq).
      seq += 1;
      st.selecting = true; st.confirm = false; st.sel.clear(); st.del = null; st.all = null;
      if (a === "hold") st.sel.add(attr("data-id"));
    } else if (a === "selcancel") { seq += 1; st.selecting = false; st.confirm = false; st.sel.clear(); st.all = null; }
    else if (a === "tick") {
      const id = attr("data-id");
      if (st.sel.has(id)) st.sel.delete(id); else st.sel.add(id);
    } else if (a === "selall") { st.all = {pending: true}; ask("ids", {filter: st.filter, q: st.q}); }
    else if (a === "selnone") st.sel.clear();
    else if (a === "delask") { st.confirm = true; st.del = null; focusSoon('[data-act="games:delno"].ak-btn'); }
    else if (a === "delno") { st.confirm = false; focusSoon('[data-act="games:delask"]'); }
    else if (a === "delyes") { st.del = {pending: true}; ask("delete", {ids: [...st.sel]}); }
    else if (a === "samples") {
      st.hideSamples = !st.hideSamples;
      try { localStorage.setItem(SAMPLES_KEY, st.hideSamples ? "hide" : "show"); } catch (e) { /* ignore */ }
    } else if (a === "open") {
      st.game = attr("data-id");
      st.turn = null;
      st.decision = null;
    } else if (a === "back") { st.game = null; st.decision = null; }
    else if (a === "backgame") st.decision = null;
    else if (a === "minor") st.minor = !st.minor;
    else if (a === "reloadgame") st.pages.delete(st.game);
    else if (a === "reloadbot") st.bots.delete(st.game + ":" + st.decision);
    else if (a === "turn") {
      const t = +attr("data-turn");
      st.turn = st.turn === t ? null : t;
      const fk = FKEY + "t:" + st.game + ":" + t;
      const fr = st.frames.get(fk);
      if (st.turn != null && !reduced() && !(fr && fr.error)) {
        // The turn plays as soon as it opens; its frames may still be on their way (any number of plays until then).
        ui(fk).i = 0;
        if (seqAct(st.ui, fk, fr && fr.frames ? fr.frames.length : Infinity, "play")) playTimer();
      }
    } else if (a === "decision") st.decision = +attr("data-n");
    else if (a === "check") {
      st.checks.set(st.game, {progress: {done: 0, total: (pageOf(st.game).decisions || []).length}});
      ctx.send({t: "ask_data", key: KEY + "check:" + st.game, what: "check", args: {record: st.game}});
    } else if (a === "play") { ctx.play(); return; }
    else if (a === "askstats") { ctx.ask("How am I doing in my games? Look at my stats by deck and by going first or second."); return; }
    else if (a === "askgame") { ctx.ask("Look at my game " + st.game + ": what decided it, and what could I have done better?"); return; }
    else if (a === "askwhy") { ctx.ask("In my game " + st.game + ", decision " + st.decision + ": why does the bot prefer its pick over mine?"); return; }
    else if (a === "seq" || a === "frames") {
      const key = attr("data-key");
      if (a === "frames") st.frames.delete(key);
      else {
        ui(key);
        const fr = st.frames.get(key);
        if (seqAct(st.ui, key, fr && fr.frames ? fr.frames.length : 0, attr("data-do"))) playTimer();
      }
    }
    ctx.render();
  }

  // Export (export.js): from the game's header (the whole game), the open turn's sequence (that turn, or the play on
  // screen) or Select mode (the ticked games).
  function exportAct(a, el) {
    if (a === "xpsel") { if (st.sel.size) xp.open({ids: [...st.sel]}); focusSoon(".gx h2"); return; }
    if (a === "xpgame" || a === "xpturn") {
      const p = pageOf(st.game);
      if (!p.id) return;
      let play = null;
      if (st.turn != null) {
        const fk = FKEY + "t:" + p.id + ":" + st.turn;
        const fr = st.frames.get(fk);
        const u = ui(fk);
        const f = fr && fr.frames && fr.frames[u.i];
        if (f && f.choice != null) { play = {k: u.i, n: fr.frames.length, choice: f.choice, text: f.text}; u.playing = false; }
      }
      xp.open({ids: [p.id], game: p, turn: st.turn, play, what: a === "xpturn" ? "turn" : "game"});
      focusSoon(".gx h2");
      return;
    }
    xp.act(a, el);
  }

  // After the next render, focus moves into the dialog or back to the control that opened it (WCAG 2.4.3).
  function focusSoon(sel) {
    if (typeof document === "undefined") return;
    setTimeout(() => { const el = document.querySelector('[data-feature="games"] ' + sel); if (el) el.focus(); }, 0);
  }

  return {
    onMessage,
    act,
    // The search box (app.js routes its input events here): applied after a short pause, from the first page.
    search(v) {
      st.qInput = v;
      clearTimeout(qTimer);
      qTimer = setTimeout(() => {
        const q = v.trim();
        if (q === st.q) return;
        st.q = q;
        st.limit = PAGE;
        ctx.render();
      }, SEARCH_MS);
    },
    // The tab opened, or the socket came back: anything asked on the old socket is asked again.
    enter() {
      xp.enter();
      wv.enter();
      for (const m of [st.list, st.pages, st.bots, st.frames]) for (const [k, v] of m) if (v && v.pending) m.delete(k);
      st.list.clear();   // new games may have been saved since
      for (const [k, v] of st.checks) if (v && v.progress) st.checks.delete(k);
      // Select all / Delete asked on the old socket: their replies are lost.
      if (st.all && st.all.pending) st.all = null;
      if (st.del && st.del.pending) {
        st.del = {error: "the connection dropped before the delete was confirmed; some games may already be gone"};
        st.confirm = false;
        st.sel.clear();
      }
      seq += 1;
    },
    panelHTML() {
      const page = st.game == null ? "" : st.decision ? decisionHTML() : gameHTMLPage();
      // Wide screens show the list beside the open game (docs/plans/responsive.md §4); narrow ones one at a time.
      if (page && TWO.matches) {
        return '<div data-feature="games" class="gm two"><div class="gm-pane gm-plist">' + listHTML() + '</div><div class="gm-pane gm-ppage">' + page +
          "</div>" + (wv.isOpen ? "" : xp.html()) + "</div>";
      }
      return '<div data-feature="games" class="gm">' + (page || listHTML()) + (wv.isOpen ? "" : xp.html()) + "</div>";
    },
    // Watch over any tab (the chat's stills open it too): opened by app.js for Ask, repainted on every render.
    watch(id, at) { wv.open(id, at); },
    get watching() { return wv.isOpen; },
    paintWatch() {
      if (!wv.isOpen || typeof document === "undefined") return;
      const host = document.querySelector("#watch .w-sheet");
      if (host) morph(host, '<div data-feature="games" class="gm">' + xp.html() + "</div>");
    },
    // Android back: the export sheet, Watch, a decision, then the game.
    back() {
      if (xp.close()) return true;
      if (wv.close()) return true;
      if (st.confirm) { st.confirm = false; return true; }
      if (st.selecting) { st.selecting = false; st.sel.clear(); return true; }
      if (st.decision) { st.decision = null; return true; }
      if (st.game != null) { st.game = null; return true; }
      return false;
    },
  };
}
