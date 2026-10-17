import {connectWS} from "./ws.js";
import {barHTML, sideHTML, midHTML, phaseBtnsHTML, chainHTML} from "./render/board.js";
import {handHTML, sheetHTML, pileSheetHTML, infoHTML, actionableMap, statsLine} from "./render/hand.js";
import {logHTML} from "./render/log.js";
import {esc, imgUrl, isHidden, phHTML} from "./render/card.js";

const MOCKS = [
  "mock/01-lobby.json",
  "mock/02-your-main-phase.json",
  "mock/03-chain-prompt.json",
  "mock/04-select-cards.json",
  "mock/05-bot-thinking.json",
  "mock/06-battle.json",
  "mock/07-position.json",
  "mock/08-game-over.json",
  "mock/09-gy-sheet.json",
  "mock/10-chain-3.json",
];

const qs = new URLSearchParams(location.search);
const MOCK = qs.has("mock");
let mockIdx = Math.max(0, parseInt(qs.get("i") || "0", 10) || 0);
const TOKEN = qs.get("t") || "";

const app = document.getElementById("app");
const logside = document.getElementById("logside");
const layer = document.getElementById("layer");
const banner = document.getElementById("banner");

let msg = null;          // latest server/mock message
let locked = false;      // input disabled until the next state
let sel = [];            // selected option indices (multi-select)
let expanded = null;     // option index whose effect text is expanded
let menuUid = null;      // uid with an open action menu
let menuRect = null;
let focusUid = null;     // card shown in the info panel
let logOpen = false;
let detail = null;       // card for the full-screen detail view
let lobby = null;
let lobbyPick = {you: null, bot: null, first: "rps", mode: "product"};
let pickerOpen = null;   // "you" | "bot" | null
let pileOpen = null;     // "gy:you" | "banished:bot" | "extra:you" ...
let ws = null;
let lpShown = {you: null, bot: null};
let events = [];         // accumulated log (the server sends only new events; dedupe by n)
const texts = new Map(); // code -> {text, type_str} | "pending" | "failed"

// ---- sizing ----
// The board is width-bound: 5 main columns plus 2 narrower pile columns.
// Height left after board + bars goes to the hand (up to 1.7x the board card),
// then to the info panel, which absorbs the rest (flex), so nothing is dead space.
const R = 86 / 59;
const SIDE = 0.8;
function fit() {
  const root = document.documentElement;
  const W = Math.min(window.innerWidth, 520);
  const H = window.innerHeight;
  const PAD = 6, GAP = 3;
  let w = Math.min((W - 2 * PAD - 6 * GAP) / (5 + 2 * SIDE), 76);
  let hw = Math.min(w * 1.7, 112);
  const infoMin = window.innerWidth >= 900 ? 0 : 64;   // desktop: card info lives in the side panel
  const need = (w, hw) => 44 + 54 + (5 * w * R + 36) + (hw * R + 26) + infoMin;
  for (let i = 0; i < 200 && need(w, hw) > H; i++) {
    if (hw > w * 1.25) hw -= 1;
    else w -= 0.5;
  }
  w = Math.max(30, w);
  root.style.setProperty("--card-w", w.toFixed(1) + "px");
  root.style.setProperty("--side-w", (w * SIDE).toFixed(1) + "px");
  root.style.setProperty("--hand-w", hw.toFixed(1) + "px");
  // Sheets stop below your bar unless their content needs more (small phones).
  const bar = document.querySelector(".pbar.you");
  if (bar) root.style.setProperty("--sheet-max", Math.max(H * 0.48, H - bar.getBoundingClientRect().bottom + 6).toFixed(0) + "px");
  const hand = document.getElementById("hand");
  if (hand) {
    const n = hand.children.length;
    const avail = hand.clientWidth - 24;
    let g = 6;
    if (n > 1 && n * hw + (n - 1) * 6 > avail) g = (avail - n * hw) / (n - 1);
    hand.style.setProperty("--hand-gap", g.toFixed(1) + "px");
  }
}
window.addEventListener("resize", () => { fit(); });

const reducedMotion = () =>
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---- data helpers ----
function allCards() {
  const out = [];
  if (!msg || !msg.view) return out;
  for (const side of [msg.view.you, msg.view.bot]) {
    out.push(...(side.hand || []), ...(side.mzone || []), ...(side.emz || []), ...(side.szone || []),
      ...(side.gy || []), ...(side.banished || []), ...(side.extra || []), side.field);
  }
  for (const l of msg.view.chain || []) out.push(l.card);
  if (msg.prompt) {
    out.push(msg.prompt.card);
    for (const o of msg.prompt.options || []) out.push(o.card);
  }
  for (const e of events) out.push(e.card);
  return out;
}

function findCard(uid) {
  if (!uid) return null;
  for (const c of allCards()) {
    if (c && c.uid === uid) return c;
    for (const o of (c && c.overlays) || []) if (o && o.uid === uid) return o;
  }
  return null;
}

function optionsForUid(uid) {
  if (!msg || !msg.prompt || !msg.prompt.options) return [];
  return msg.prompt.options.filter((o) => o.card && o.card.uid === uid && o.verb !== "select" && o.where);
}

// Card text from the server's /card/{code} endpoint, cached by code.
function cardText(code) {
  if (!code || MOCK) return null;
  const t = texts.get(code);
  if (t) return typeof t === "object" ? t : null;
  texts.set(code, "pending");
  fetch("/card/" + code + (TOKEN ? "?t=" + encodeURIComponent(TOKEN) : ""))
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((j) => { texts.set(code, {text: j.text || "", type_str: j.type_str || ""}); render(); })
    .catch(() => { texts.set(code, "failed"); });
  return null;
}

function addEvents(list, reset) {
  if (reset) events = [];
  for (const e of list || []) {
    const at = events.findIndex((x) => x.n === e.n);
    if (at >= 0 && events[at].text === e.text) continue;
    if (at >= 0) events = [];   // same n, different text: a new game restarted the numbering
    events.push(e);
  }
  events.sort((a, b) => (a.n || 0) - (b.n || 0));
}

// ---- outbound ----
function choose(index) {
  if (locked || !msg || !msg.prompt) return;
  locked = true;
  menuUid = null;
  expanded = null;
  render();
  if (MOCK) {
    setTimeout(() => { locked = false; render(); }, 350);
    return;
  }
  if (ws) ws.send({t: "choose", prompt: msg.prompt.id, index});
}

function onOption(index) {
  const p = msg && msg.prompt;
  if (!p || locked) return;
  const opt = (p.options || []).find((o) => o.index === index);
  if (!opt) return;
  // the env takes one pick per step: picked cards come back as "unselect" options
  choose(index);
}

// ---- motion ----
function snapshotRects() {
  const m = new Map();
  if (reducedMotion()) return m;
  for (const el of app.querySelectorAll(".card[data-uid]")) {
    const u = el.getAttribute("data-uid");
    if (u && !m.has(u)) m.set(u, el.getBoundingClientRect());
  }
  return m;
}

function flipAnimate(before) {
  if (reducedMotion() || !before.size) return;
  for (const el of app.querySelectorAll(".card[data-uid]")) {
    const r0 = before.get(el.getAttribute("data-uid"));
    if (!r0) continue;
    const r1 = el.getBoundingClientRect();
    const dx = r0.left + r0.width / 2 - (r1.left + r1.width / 2);
    const dy = r0.top + r0.height / 2 - (r1.top + r1.height / 2);
    if (Math.abs(dx) < 2 && Math.abs(dy) < 2) continue;
    const base = getComputedStyle(el).transform;
    const end = base === "none" ? "none" : base;
    el.animate([{transform: "translate(" + dx + "px," + dy + "px) " + (end === "none" ? "" : end)}, {transform: end}],
      {duration: 200, easing: "cubic-bezier(.2,.7,.3,1)"});
  }
}

// ---- render ----
function pileCards(which) {
  if (!msg || !msg.view || !which) return [];
  const [kind, who] = which.split(":");
  const side = who === "bot" ? msg.view.bot : msg.view.you;
  return (side && side[kind === "gy" ? "gy" : kind]) || [];
}

function hintFor(p) {
  if (msg.result) return "";
  if (msg.thinking) return "";   // the bot bar already shows it
  if (!p) return "Waiting for ygo-zero…";
  if (p.kind === "idle") return "Your move · tap a glowing card";
  if (p.kind === "battle") return "Battle · tap a glowing monster to attack";
  return "";
}

function renderGame() {
  const v = msg.view;
  const p = msg.prompt;
  const act = locked ? {} : actionableMap(p);
  if (focusUid && !findCard(focusUid)) focusUid = null;
  const yourTurn = v.turn_player === "you";

  let h = barHTML(v.bot, "bot", {thinking: msg.thinking, active: !yourTurn});
  h += '<div id="board">';
  h += '<div class="field bot">' + sideHTML(v.bot, {mirror: true, mock: MOCK, actionable: act, focus: focusUid}) + "</div>";
  h += '<div id="mid"><div id="midstrip" class="zrow">' + midHTML(v, {mock: MOCK, actionable: act, focus: focusUid}) + "</div>";
  const chain = chainHTML(v.chain, MOCK);
  if (chain) h += '<div id="chain">' + chain + "</div>";
  h += "</div>";
  h += '<div class="field you">' + sideHTML(v.you, {mirror: false, mock: MOCK, actionable: act, focus: focusUid}) + "</div>";
  h += "</div>";
  h += barHTML(v.you, "you", {phaseBtns: phaseBtnsHTML(p, locked), active: yourTurn});

  let info = null;
  if (focusUid) {
    const c = findCard(focusUid);
    const t = c && !isHidden(c) ? cardText(c.code) : null;
    info = {card: c, text: t && t.text, typeStr: t && t.type_str};
  }
  h += '<div id="info">' + infoHTML(info, events, MOCK, hintFor(p)) + "</div>";
  h += '<div id="hand" class="' + (locked ? "locked" : "") + '">' + handHTML(v.you.hand, act, MOCK, focusUid) + "</div>";

  const sheet = pileOpen ? pileSheetHTML(pileOpen, pileCards(pileOpen), MOCK)
    : (p && !msg.result ? sheetHTML(p, MOCK, sel, expanded) : "");
  if (sheet) h += '<div id="sheet" class="' + (locked && !pileOpen ? "locked" : "") + (pileOpen ? " pile" : "") + '">' + sheet + "</div>";
  app.innerHTML = h;
  app.classList.toggle("busy", locked);
  fit();

  for (const who of ["you", "bot"]) {
    const el = app.querySelector(".pbar." + who + " .lp");
    const target = v[who].lp;
    const prev = lpShown[who] == null ? target : lpShown[who];
    lpShown[who] = target;
    if (!el || prev === target || reducedMotion()) continue;
    if (target < prev) el.classList.add("flash");
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 400);
      el.textContent = String(Math.round(prev + (target - prev) * (1 - Math.pow(1 - k, 3))));
      if (k < 1) requestAnimationFrame(step);
      else setTimeout(() => el.classList.remove("flash"), 300);
    };
    el.textContent = String(prev);
    requestAnimationFrame(step);
  }

  const logh = logHTML(events, MOCK);
  logside.innerHTML = '<div class="sidecard">' + (info && info.card ? infoHTML(info, null, MOCK, "")
    : '<p class="emptytx">Tap any card to read it here.</p>') + "</div>" +
    '<div class="loghead">Duel log</div><div class="loglist">' + logh + "</div>";
  renderLayer(logh);
}

function renderLayer(logh) {
  let h = "";
  if (logOpen) {
    h += '<div id="scrim" data-act="closelog"></div>';
    h += '<div id="logdrawer" role="dialog" aria-label="Duel log">' +
      '<div class="loghead">Duel log<button class="iconbtn" data-act="closelog" aria-label="Close log">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
      '<div class="loglist">' + logh + "</div></div>";
  }
  if (menuUid && !locked) {
    const opts = optionsForUid(menuUid);
    const r = menuRect || {left: window.innerWidth / 2 - 40, top: 300, width: 80};
    const mw = Math.min(260, window.innerWidth - 16);
    const mh = opts.length * 48 + 56;
    const top = Math.max(8, Math.min(window.innerHeight - mh - 8, r.top - mh - 8));
    const left = Math.max(8, Math.min(window.innerWidth - mw - 8, r.left + (r.width || 0) / 2 - mw / 2));
    h += '<div id="menu" style="top:' + top + "px;left:" + left + "px;width:" + mw + 'px" role="menu">' +
      opts.map((o) => '<button data-opt="' + o.index + '" class="' +
        (["activate", "normal", "special", "attack", "direct"].includes(o.verb) ? "primary" : "") + '">' +
        esc(o.label || o.verb) + "</button>").join("") +
      '<button class="ghost" data-act="detail" data-uid="' + esc(menuUid) + '">Card details</button></div>';
  }
  if (detail) {
    const c = detail;
    const t = cardText(c.code);
    const face = !MOCK && c.code
      ? '<div class="dface">' + phHTML(c) + '<img alt="" src="' + imgUrl(c.code, "full") + '" onerror="this.remove()"></div>'
      : '<div class="dface">' + phHTML(c) + "</div>";
    const meta = [t && t.type_str, statsLine(c)].filter(Boolean).join(" · ");
    h += '<div id="detail" data-act="closedetail" role="dialog" aria-label="Card detail">' + face +
      "<h2>" + esc(c.name || "Hidden card") + "</h2>" +
      (meta ? '<div class="typeline">' + esc(meta) + "</div>" : "") +
      '<div class="dtext">' + esc((t && t.text) || (MOCK ? "Card text loads from the server in a live game." : "")) + "</div>" +
      '<div class="dclose">Tap anywhere to close</div></div>';
  }
  if (msg && msg.view && msg.result) {
    const win = msg.result.winner === "you";
    h += '<div id="gameover" role="alertdialog" aria-label="Game over"><div class="box ' + (win ? "win" : "loss") + '">' +
      '<div class="kicker">Duel over · turn ' + msg.view.turn + "</div>" +
      "<h2>" + (win ? "Victory" : "Defeat") + "</h2>" +
      '<p class="reason">' + esc(msg.result.reason === "win" ? "" : msg.result.reason || "") + "</p>" +
      '<div class="score"><span class="you">You<b>' + msg.view.you.lp + "</b></span>" +
      '<span class="bot">ygo-zero<b>' + msg.view.bot.lp + "</b></span></div>" +
      '<div class="row"><button class="bigbtn" data-act="newgame">New game</button>' +
      '<button class="bigbtn primary" data-act="rematch">Rematch</button></div></div></div>';
  }
  layer.innerHTML = h;
  const ll = layer.querySelector("#logdrawer .loglist") || logside.querySelector(".loglist");
  if (ll) ll.scrollTop = ll.scrollHeight;
}

// ---- lobby ----
function deckName(decks, id) {
  const d = (decks || []).find((x) => x.id === id);
  return d ? d.name : "Choose a deck";
}

const GROUPS = {pool: "Pool", newcards: "New cards", user: "Yours"};

function pickerHTML(decks, side, label) {
  const cur = lobbyPick[side];
  const open = pickerOpen === side;
  const d = decks.find((x) => x.id === cur);
  let h = '<section class="field-group"><div class="flabel">' + esc(label) + "</div>";
  h += '<button class="pickbtn' + (open ? " open" : "") + '" data-pickopen="' + side + '" aria-expanded="' + open + '">' +
    '<span class="dot ' + (side === "you" ? "you" : "bot") + '"></span>' +
    '<span class="nm">' + esc(deckName(decks, cur)) + "</span>" +
    (d ? '<span class="g">' + esc(GROUPS[d.group] || d.group) + "</span>" : "") +
    '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></button>';
  if (open) {
    h += '<div class="pickpanel"><input class="decksearch" type="search" placeholder="Search ' + decks.length +
      ' decks" aria-label="Search decks" data-search="' + side + '"><div class="decklist">';
    for (const x of decks) {
      h += '<button class="deckrow' + (cur === x.id ? " sel" : "") + '" data-' + side + 'deck="' + esc(x.id) +
        '" data-dname="' + esc(x.name.toLowerCase()) + '"><span class="nm">' + esc(x.name) + '</span><span class="g">' +
        esc(GROUPS[x.group] || x.group) + "</span></button>";
    }
    h += "</div></div>";
  }
  return h + "</section>";
}

function segHTML(label, key, items) {
  return '<section class="field-group"><div class="flabel">' + esc(label) + '</div><div class="seg" role="radiogroup">' +
    items.map(([k, t]) => '<button role="radio" aria-checked="' + (lobbyPick[key] === k) + '" class="' +
      (lobbyPick[key] === k ? "sel" : "") + (t.length > 10 ? " wide" : "") + '" data-' + key + '="' + k + '">' + t + "</button>").join("") +
    "</div></section>";
}

function renderLobby() {
  const decks = (lobby && lobby.decks) || [];
  if (!lobbyPick.you && decks.length) lobbyPick.you = decks[0].id;
  if (!lobbyPick.bot && decks.length) lobbyPick.bot = decks[Math.min(1, decks.length - 1)].id;
  let h = '<div id="lobby"><div class="lscroll">';
  h += '<header class="hero"><div class="fan" aria-hidden="true"><i></i><i></i><i></i></div>' +
    '<h1>ygo<span>-zero</span></h1><p>Duel the bot from your phone.</p></header>';
  h += pickerHTML(decks, "you", "Your deck");
  h += pickerHTML(decks, "bot", "ygo-zero plays");
  h += segHTML("Going first", "first", [["rps", "Rock-paper-scissors"], ["me", "Me"], ["bot", "Bot"]]);
  h += segHTML("Bot strength", "mode", [["product", "Full"], ["fast", "Fast"], ["random", "Random"]]);
  h += '<details class="ydk"><summary>Paste a .ydk instead<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></summary>' +
    '<textarea id="ydk" placeholder="#main&#10;14558128&#10;…" spellcheck="false"></textarea></details>';
  if (MOCK) h += '<div class="mocknav">mock ' + (mockIdx + 1) + "/" + MOCKS.length + " · ← → to step</div>";
  h += '</div><div id="startwrap"><button id="startbtn" data-act="start">Start duel</button></div></div>';
  app.innerHTML = h;
  app.classList.remove("busy");
  logside.innerHTML = '<div class="loghead">Duel log</div><div class="loglist"><p class="emptytx">No duel yet.</p></div>';
  layer.innerHTML = "";
}

function render() {
  const before = snapshotRects();
  if (!msg || msg.t === "lobby") {
    renderLobby();
    return;
  }
  if (msg.t === "state") {
    renderGame();
    flipAnimate(before);
    document.title = msg.prompt && msg.prompt.text ? msg.prompt.text + " · ygo-zero" : "ygo-zero";
  } else if (msg.t === "error") {
    banner.hidden = false;
    banner.textContent = msg.text || "error";
  }
}

// ---- events (delegated) ----
function cardTap(el) {
  const uid = el.getAttribute("data-uid");
  const c = findCard(uid);
  if (el.classList.contains("actionable") && optionsForUid(uid).length && !locked) {
    menuUid = menuUid === uid ? null : uid;
    menuRect = el.getBoundingClientRect();
    if (c && !isHidden(c)) focusUid = uid;
  } else {
    menuUid = null;
    if (c && !isHidden(c)) focusUid = focusUid === uid ? null : uid;
  }
  render();
}

app.addEventListener("click", (ev) => {
  const t = ev.target;
  const ef = t.closest(".opt .ef");
  if (ef) {
    // Tapping the effect text expands it; tapping the rest of the row chooses.
    const i = parseInt(ef.closest(".opt").getAttribute("data-opt"), 10);
    expanded = expanded === i ? null : i;
    render();
    return;
  }
  const opt = t.closest("[data-opt]");
  if (opt) {
    if (!opt.disabled) onOption(parseInt(opt.getAttribute("data-opt"), 10));
    return;
  }
  const pick = t.closest("[data-pickopen]");
  if (pick) {
    const side = pick.getAttribute("data-pickopen");
    pickerOpen = pickerOpen === side ? null : side;
    renderLobby();
    const inp = app.querySelector(".decksearch");
    if (inp && !("ontouchstart" in window)) inp.focus();
    return;
  }
  const pile = t.closest("[data-pile]");
  if (pile) {
    pileOpen = pile.getAttribute("data-pile");
    menuUid = null;
    render();
    return;
  }
  const det = t.closest("[data-detailuid]");
  if (det) {
    const c = findCard(det.getAttribute("data-detailuid"));
    if (c && !isHidden(c)) { detail = c; render(); }
    return;
  }
  const fu = t.closest("[data-focusuid]");
  if (fu) {
    focusUid = fu.getAttribute("data-focusuid");
    render();
    return;
  }
  for (const key of ["youdeck", "botdeck"]) {
    const el = t.closest("[data-" + key + "]");
    if (el) {
      lobbyPick[key === "youdeck" ? "you" : "bot"] = el.getAttribute("data-" + key);
      pickerOpen = null;
      renderLobby();
      return;
    }
  }
  for (const key of ["first", "mode"]) {
    const el = t.closest("[data-" + key + "]");
    if (el) {
      lobbyPick[key] = el.getAttribute("data-" + key);
      renderLobby();
      return;
    }
  }
  const act = t.closest("[data-act]");
  if (act) {
    onAction(act.getAttribute("data-act"), act);
    return;
  }
  const c = t.closest(".card[data-uid]");
  if (c) {
    cardTap(c);
    return;
  }
  if (menuUid) {
    menuUid = null;
    render();
  }
});

layer.addEventListener("click", (ev) => {
  const opt = ev.target.closest("[data-opt]");
  if (opt) {
    if (!opt.disabled) onOption(parseInt(opt.getAttribute("data-opt"), 10));
    return;
  }
  const det = ev.target.closest("[data-detailuid]");
  if (det) {
    const c = findCard(det.getAttribute("data-detailuid"));
    if (c && !isHidden(c)) { detail = c; logOpen = false; render(); }
    return;
  }
  const act = ev.target.closest("[data-act]");
  if (act) onAction(act.getAttribute("data-act"), act);
});

function onAction(kind, el) {
  if (kind === "log") logOpen = true;
  else if (kind === "closelog") logOpen = false;
  else if (kind === "closepile") pileOpen = null;
  else if (kind === "unfocus") focusUid = null;
  else if (kind === "detail") {
    detail = findCard(el.getAttribute("data-uid"));
    menuUid = null;
  } else if (kind === "closedetail") detail = null;
  else if (kind === "start" || kind === "rematch") {
    if (MOCK) { loadMock(1); return; }
    const ydk = kind === "start" ? ((document.getElementById("ydk") || {}).value || "").trim() : "";
    events = [];
    if (ws) ws.send({t: "new_game", deck: ydk ? {ydk} : lobbyPick.you, bot_deck: lobbyPick.bot,
      first: lobbyPick.first, mode: lobbyPick.mode});
    return;
  } else if (kind === "newgame") {
    msg = lobby || {t: "lobby", decks: []};
    detail = null;
  } else return;
  render();
}

// Long-press opens the full card detail.
let pressTimer = 0;
document.addEventListener("pointerdown", (ev) => {
  const c = ev.target.closest && ev.target.closest(".card[data-uid]");
  if (!c) return;
  const uid = c.getAttribute("data-uid");
  clearTimeout(pressTimer);
  pressTimer = setTimeout(() => {
    const card = findCard(uid);
    if (card && !isHidden(card)) {
      detail = card;
      menuUid = null;
      render();
    }
  }, 500);
});
for (const evt of ["pointerup", "pointercancel", "pointermove"]) {
  document.addEventListener(evt, () => clearTimeout(pressTimer));
}

// Announce search filter + lobby deck search (DOM filter, keeps focus).
document.addEventListener("input", (ev) => {
  const t = ev.target;
  if (t.matches(".announce input")) {
    const q = t.value.toLowerCase();
    for (const b of t.closest("#sheet").querySelectorAll(".opt")) {
      b.style.display = b.textContent.toLowerCase().includes(q) ? "" : "none";
    }
  }
  if (t.matches(".decksearch")) {
    const q = t.value.toLowerCase();
    for (const b of t.closest(".pickpanel").querySelectorAll(".deckrow")) {
      b.style.display = (b.getAttribute("data-dname") || "").includes(q) ? "" : "none";
    }
  }
});

// ---- boot ----
async function loadMock(i) {
  mockIdx = ((i % MOCKS.length) + MOCKS.length) % MOCKS.length;
  try {
    const r = await fetch(MOCKS[mockIdx] + "?v=2");
    msg = await r.json();
  } catch (e) {
    msg = {t: "error", text: "mock load failed: " + MOCKS[mockIdx]};
  }
  if (msg.t === "lobby") lobby = msg;
  if (msg.t === "state") addEvents(msg.log, true);
  locked = false;
  sel = [];
  expanded = null;
  menuUid = null;
  detail = null;
  focusUid = qs.get("focus") || null;
  pileOpen = msg.openPile || qs.get("pile") || null;
  if (qs.get("menu") && msg.t === "state") {
    // Screenshot hook: open the action menu on a card.
    render();
    const el = app.querySelector('.card[data-uid="' + qs.get("menu") + '"]');
    if (el) { menuUid = qs.get("menu"); menuRect = el.getBoundingClientRect(); }
  }
  if (qs.get("sel") && msg.t === "state") sel = qs.get("sel").split(",").map(Number);
  if (qs.get("picker") && msg.t === "lobby") pickerOpen = qs.get("picker");
  render();
}

document.addEventListener("keydown", (ev) => {
  if (!MOCK || ev.target.matches("input, textarea")) return;
  if (ev.key === "ArrowRight") loadMock(mockIdx + 1);
  if (ev.key === "ArrowLeft") loadMock(mockIdx - 1);
});

async function boot() {
  if (MOCK) {
    await loadMock(mockIdx);
    return;
  }
  if (!TOKEN) {
    try {
      lobby = await (await fetch(MOCKS[0])).json();
    } catch (e) {
      lobby = {t: "lobby", decks: []};
    }
    msg = lobby;
    renderLobby();
    return;
  }
  ws = connectWS(TOKEN, {
    onOpen() { banner.hidden = true; },
    onDrop() {
      banner.hidden = false;
      banner.textContent = "Reconnecting…";
    },
    onMessage(m) {
      if (m.t === "lobby") {
        lobby = m;
        if (!msg || msg.t === "lobby") msg = m;
      } else if (m.t === "state") {
        if (msg && msg.t === "state" && msg.prompt && (!m.prompt || m.prompt.id !== msg.prompt.id)) {
          sel = [];
          expanded = null;
        }
        if (!m.prompt) sel = [];
        addEvents(m.log, false);
        msg = m;
        locked = false;
        menuUid = null;
      } else if (m.t === "error") {
        banner.hidden = false;
        banner.textContent = m.text || "error";
        locked = false;
        setTimeout(() => { banner.hidden = true; }, 4000);
      } else {
        msg = m;
      }
      render();
    },
  });
}

boot();
