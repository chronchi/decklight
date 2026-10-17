import {connectWS} from "./ws.js";
import {barHTML, sideHTML, midHTML, phaseBtnsHTML, chainHTML} from "./render/board.js";
import {handHTML, sheetHTML, pileSheetHTML, infoHTML, actionableMap, statsLine} from "./render/hand.js";
import {logHTML} from "./render/log.js";
import {esc, imgUrl, isHidden, phHTML} from "./render/card.js";
import {morph} from "./render/morph.js";

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
const banner = document.getElementById("banner");
// Persistent overlay containers: each is morphed separately so an open
// drawer/menu/detail is never re-created (no re-animation, no lost focus).
function ov(id) {
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement("div");
    el.id = id;
    document.getElementById("layer").appendChild(el);
  }
  return el;
}
const ovLog = ov("ov-log");
const ovMenu = ov("ov-menu");
const ovGame = ov("ov-gamemenu");
const ovDetail = ov("ov-detail");
const ovReveal = ov("ov-reveal");
const ovOver = ov("ov-over");

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

// In-game menu + surrender confirm.
let gameMenu = false;
let surrenderArm = false;
let surrenderTimer = 0;
let chainMode = "auto";
try {
  const saved = localStorage.getItem("ygo-chain");
  if (saved === "auto" || saved === "all") chainMode = saved;
} catch (e) { /* private mode: stay on auto */ }
let inLobby = false;     // after Back to lobby: ignore states until new_game

// Bot activation reveal queue + top-of-board banner.
let revealQueue = [];
let revealCur = null;
let revealTimer = 0;
let revealHold = false;
let revealLeaving = false;
let bannerText = null;
let bannerTimer = 0;

// Overlay history (Android back closes the topmost overlay).
let ovStack = [];
let popGuard = false;
function openOvNames() {
  const o = [];
  if (logOpen) o.push("log");
  if (menuUid) o.push("menu");
  if (gameMenu) o.push("gamemenu");
  if (pileOpen) o.push("pile");
  if (detail) o.push("detail");
  if (revealCur) o.push("reveal");
  return o;
}
function pushOv(name) {
  if (ovStack.includes(name)) return;
  ovStack.push(name);
  try {
    history.pushState({ov: name}, "");
  } catch (e) { /* ignore */ }
}
function syncOv() {
  const cur = openOvNames();
  for (const n of cur) pushOv(n);
  ovStack = ovStack.filter((n) => cur.includes(n));
}
function closeState(name) {
  if (name === "log") logOpen = false;
  else if (name === "menu") menuUid = null;
  else if (name === "gamemenu") { gameMenu = false; surrenderArm = false; }
  else if (name === "pile") pileOpen = null;
  else if (name === "detail") detail = null;
  else if (name === "reveal") dismissReveal();
}
// X/scrim close: update state, then pop our own history entry so a later
// back press does not land on a dead overlay state.
function closeOv(name) {
  const i = ovStack.indexOf(name);
  if (i < 0) return;
  ovStack.splice(i, 1);
  try {
    if (history.state && history.state.ov) {
      popGuard = true;
      history.back();
      setTimeout(() => { popGuard = false; }, 800);
    }
  } catch (e) { popGuard = false; }
}
window.addEventListener("popstate", () => {
  if (popGuard) return;
  const top = ovStack.pop();
  if (!top) return;
  closeState(top);
  render();
});

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

// Resolve the card behind a sheet ⓘ button: the option's card, else the prompt card.
function detailCardForOpt(index) {
  const p = msg && msg.prompt;
  if (!p) return null;
  const o = (p.options || []).find((x) => x.index === index);
  const c = (o && o.card) || p.card;
  return c && !isHidden(c) ? c : null;
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
  const wasEmpty = events.length === 0;
  if (reset) events = [];
  const added = [];
  for (const e of list || []) {
    const at = events.findIndex((x) => x.n === e.n);
    if (at >= 0 && events[at].text === e.text) continue;
    if (at >= 0) events = [];   // same n, different text: a new game restarted the numbering
    events.push(e);
    added.push(e);
  }
  events.sort((a, b) => (a.n || 0) - (b.n || 0));
  // No reveals for the initial full-log replay or a reset batch.
  if (!reset && !wasEmpty && added.length) queueNotices(added);
}

// Bot notices: activations get a centered reveal (queued), other
// board-changing bot events get a small top-of-board banner. Only the
// newest 3 events of a batch can trigger anything.
function queueNotices(added) {
  const fresh = added.slice().sort((a, b) => (a.n || 0) - (b.n || 0)).slice(-3);
  for (const e of fresh) {
    if (e.actor !== "bot") continue;
    if (e.kind === "activate" && e.card && !isHidden(e.card)) {
      revealQueue.push(e);
    } else if (["summon", "spsummon", "attack", "mset", "sset"].includes(e.kind)) {
      showBanner("ygo-zero " + (e.text || ""));
    }
  }
  pumpReveal();
}

function showBanner(text) {
  bannerText = text;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { bannerText = null; render(); }, 1400);
  render();
}

function pumpReveal() {
  if (revealCur || revealLeaving || !revealQueue.length) return;
  revealCur = revealQueue.shift();
  render();
  if (!MOCK) revealTimer = setTimeout(() => dismissReveal(), 1600);
}

function dismissReveal() {
  if (!revealCur || revealLeaving) return;
  clearTimeout(revealTimer);
  const el = ovReveal.querySelector("#reveal");
  if (!el || reducedMotion() || MOCK) {
    revealCur = null;
    render();
    pumpReveal();
    return;
  }
  revealLeaving = true;
  const face = el.querySelector(".rface");
  const chain = document.getElementById("chain");
  if (face && chain) {
    const r1 = face.getBoundingClientRect(), r2 = chain.getBoundingClientRect();
    const dx = (r1.left + r1.width / 2) - (r2.left + r2.width / 2);
    const dy = (r1.top + r1.height / 2) - (r2.top + r2.height / 2);
    el.classList.add("leave");
    el.style.transform = "translate(" + (-dx * 0.5).toFixed(0) + "px," + (-dy * 0.5).toFixed(0) + "px) scale(.35)";
  } else if (el) {
    el.classList.add("leave");
  }
  setTimeout(() => {
    revealCur = null;
    revealLeaving = false;
    render();
    pumpReveal();
  }, 200);
}

function revealHTML(ev) {
  const c = ev.card || {};
  const t = c.code ? cardText(c.code) : null;
  const face = (!MOCK && c.code)
    ? '<div class="rface">' + phHTML(c) + '<img alt="" src="' + imgUrl(c.code, "full") + '" onerror="this.remove()"></div>'
    : '<div class="rface">' + phHTML(c) + "</div>";
  const txt = (t && t.text) || (MOCK ? "Card text loads from the server in a live game." : "");
  return '<div id="reveal" data-key="reveal" data-act="closereveal" role="alertdialog" aria-label="ygo-zero activates">' +
    face + '<div class="rkicker">ygo-zero activates</div>' +
    "<h2>" + esc(c.name || "") + "</h2>" +
    '<div class="rtext">' + esc(txt) + "</div></div>";
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

// Remember scroll positions across a morph, then restore them.
function saveScrolls(root, selectors) {
  const saved = selectors.map((s) => {
    const el = root.querySelector(s);
    return el ? el.scrollTop : null;
  });
  return () => {
    selectors.forEach((s, i) => {
      if (saved[i] == null) return;
      const el = root.querySelector(s);
      if (el) el.scrollTop = saved[i];
    });
  };
}

// Morph a log list: follow the bottom only if it was at the bottom (within 24px).
function morphBottom(container, html, listSel) {
  const sc = container.querySelector(listSel);
  const atBottom = !sc || (sc.scrollHeight - sc.scrollTop - sc.clientHeight < 24);
  const st = sc ? sc.scrollTop : 0;
  morph(container, html);
  const sc2 = container.querySelector(listSel);
  if (sc2) sc2.scrollTop = atBottom ? sc2.scrollHeight : Math.min(st, sc2.scrollHeight);
}

function gameMenuHTML() {
  const seg = (v, label) =>
    '<button role="radio" aria-checked="' + (chainMode === v) + '" class="' + (chainMode === v ? "sel" : "") +
    '" data-act="chainset" data-v="' + v + '">' + label + "</button>";
  return '<div id="gscrim" data-key="gscrim" data-act="closegamemenu"></div>' +
    '<div id="gamemenu" data-key="gamemenu" role="dialog" aria-label="Game menu">' +
    '<div class="mhead">Game menu<button class="iconbtn" data-act="closegamemenu" aria-label="Close game menu">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
    '<div class="mbody">' +
    '<button class="mrow danger' + (surrenderArm ? " confirm" : "") + '" data-act="surrender">' +
    (surrenderArm ? "Tap again to surrender" : "Surrender") + "</button>" +
    '<button class="mrow" data-act="backlobby">Back to lobby</button>' +
    '<div class="msep"></div>' +
    '<div class="mrow" style="pointer-events:none">Chain responses</div>' +
    '<div class="seg" role="radiogroup">' + seg("auto", "Auto") + seg("all", "All") + "</div>" +
    '<div class="mhint">Auto skips empty windows with nothing to respond to.</div>' +
    '<div class="msep"></div>' +
    '<div class="mrow disabled" aria-disabled="true">Sound<span class="sub">Off</span></div>' +
    "</div></div>";
}

function renderGame() {
  const v = msg.view;
  const p = msg.prompt;
  const act = locked ? {} : actionableMap(p);
  if (focusUid && !findCard(focusUid)) focusUid = null;
  const yourTurn = v.turn_player === "you";

  let h = barHTML(v.bot, "bot", {thinking: msg.thinking, active: !yourTurn});
  if (bannerText) h += '<div id="botbanner" data-key="botbanner">' + esc(bannerText) + "</div>";
  h += '<div id="board" data-key="board">';
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
  h += '<div id="info" data-key="info">' + infoHTML(info, events, MOCK, hintFor(p)) + "</div>";
  h += '<div id="hand" data-key="hand" class="' + (locked ? "locked" : "") + '">' + handHTML(v.you.hand, act, MOCK, focusUid) + "</div>";

  const sheet = pileOpen ? pileSheetHTML(pileOpen, pileCards(pileOpen), MOCK)
    : (p && !msg.result ? sheetHTML(p, MOCK, sel, expanded) : "");
  if (pileOpen) h += '<div id="pscrim" data-key="pscrim" data-act="closepile"></div>';
  if (sheet) h += '<div id="sheet" data-key="sheet" class="' + (locked && !pileOpen ? "locked" : "") + (pileOpen ? " pile" : "") + '">' + sheet + "</div>";
  const restore = saveScrolls(app, ["#sheet .sbody", "#info .ftext"]);
  morph(app, h);
  restore();
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
  morphBottom(logside,
    '<div class="sidecard" data-key="side">' + (info && info.card ? infoHTML(info, null, MOCK, "")
    : '<p class="emptytx">Tap any card to read it here.</p>') + "</div>" +
    '<div class="loghead" data-key="lhead">Duel log</div><div class="loglist" data-key="llist">' + logh + "</div>",
    ".loglist");
  renderOverlays(logh);
  syncOv();
}

function renderOverlays(logh) {
  let lh = "";
  if (logOpen) {
    lh = '<div id="scrim" data-key="scrim" data-act="closelog"></div>' +
      '<div id="logdrawer" data-key="logdrawer" role="dialog" aria-label="Duel log">' +
      '<div class="loghead">Duel log<button class="iconbtn" data-act="closelog" aria-label="Close log">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
      '<div class="loglist">' + logh + "</div></div>";
  }
  morphBottom(ovLog, lh, "#logdrawer .loglist");

  let mh = "";
  if (menuUid && !locked) {
    const opts = optionsForUid(menuUid);
    const r = menuRect || {left: window.innerWidth / 2 - 40, top: 300, width: 80};
    const mw = Math.min(260, window.innerWidth - 16);
    const mhh = opts.length * 48 + 56;
    const top = Math.max(8, Math.min(window.innerHeight - mhh - 8, r.top - mhh - 8));
    const left = Math.max(8, Math.min(window.innerWidth - mw - 8, r.left + (r.width || 0) / 2 - mw / 2));
    mh = '<div id="menu" data-key="menu" style="top:' + top + "px;left:" + left + "px;width:" + mw + 'px" role="menu">' +
      opts.map((o) => '<button data-opt="' + o.index + '" class="' +
        (["activate", "normal", "special", "attack", "direct"].includes(o.verb) ? "primary" : "") + '">' +
        esc(o.label || o.verb) + "</button>").join("") +
      '<button class="ghost" data-act="detail" data-uid="' + esc(menuUid) + '">Card details</button></div>';
  }
  morph(ovMenu, mh);

  morph(ovGame, gameMenu ? gameMenuHTML() : "");

  let dh = "";
  if (detail) {
    const c = detail;
    const t = cardText(c.code);
    const face = !MOCK && c.code
      ? '<div class="dface">' + phHTML(c) + '<img alt="" src="' + imgUrl(c.code, "full") + '" onerror="this.remove()"></div>'
      : '<div class="dface">' + phHTML(c) + "</div>";
    const meta = [t && t.type_str, statsLine(c)].filter(Boolean).join(" · ");
    dh = '<div id="detail" data-key="detail" data-act="closedetail" role="dialog" aria-label="Card detail">' +
      '<button class="dx" data-act="closedetail" aria-label="Close card details">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button>' + face +
      "<h2>" + esc(c.name || "Hidden card") + "</h2>" +
      (meta ? '<div class="typeline">' + esc(meta) + "</div>" : "") +
      '<div class="dtext">' + esc((t && t.text) || (MOCK ? "Card text loads from the server in a live game." : "")) + "</div>" +
      '<div class="dclose">Tap anywhere to close</div></div>';
  }
  morph(ovDetail, dh);

  if (!revealLeaving) morph(ovReveal, revealCur ? revealHTML(revealCur) : "");

  let oh = "";
  if (msg && msg.view && msg.result) {
    const win = msg.result.winner === "you";
    oh = '<div id="gameover" data-key="gameover" role="alertdialog" aria-label="Game over"><div class="box ' + (win ? "win" : "loss") + '">' +
      '<div class="kicker">Duel over · turn ' + msg.view.turn + "</div>" +
      "<h2>" + (win ? "Victory" : "Defeat") + "</h2>" +
      '<p class="reason">' + esc(msg.result.reason === "win" ? "" : msg.result.reason || "") + "</p>" +
      '<div class="score"><span class="you">You<b>' + msg.view.you.lp + "</b></span>" +
      '<span class="bot">ygo-zero<b>' + msg.view.bot.lp + "</b></span></div>" +
      '<div class="row"><button class="bigbtn" data-act="newgame">New game</button>' +
      '<button class="bigbtn primary" data-act="rematch">Rematch</button></div></div></div>';
  }
  morph(ovOver, oh);
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
  morph(ovLog, "");
  morph(ovMenu, "");
  morph(ovGame, "");
  morph(ovDetail, "");
  morph(ovReveal, "");
  morph(ovOver, "");
  ovStack = [];
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
  } else if (c && !isHidden(c)) {
    // Not actionable: open the full detail (field monsters included, both sides).
    menuUid = null;
    focusUid = uid;
    detail = c;
  } else {
    menuUid = null;
  }
  render();
}

function onInfoBtn(el) {
  const idx = el.getAttribute("data-detailcard");
  if (idx != null) {
    const c = detailCardForOpt(parseInt(idx, 10));
    if (c) {
      detail = c;
      render();
      return true;
    }
    return true; // a card grid cell without a resolvable card: never choose
  }
  const uid = el.getAttribute("data-detailuid");
  if (uid) {
    const c = findCard(uid);
    if (c && !isHidden(c)) {
      detail = c;
      render();
    }
    return true;
  }
  return false;
}

app.addEventListener("click", (ev) => {
  const t = ev.target;
  const info = t.closest("[data-detailcard], .infobtn");
  if (info && (info.getAttribute("data-detailcard") != null || info.getAttribute("data-detailuid") != null)) {
    if (onInfoBtn(info)) return;
  }
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

document.getElementById("layer").addEventListener("click", (ev) => {
  const info = ev.target.closest("[data-detailcard], .infobtn");
  if (info && (info.getAttribute("data-detailcard") != null || info.getAttribute("data-detailuid") != null)) {
    if (onInfoBtn(info)) {
      ev.stopPropagation();
      return;
    }
  }
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

function sendGame(o) {
  inLobby = false;
  if (MOCK) return;
  if (ws) ws.send(o);
}

function goLobby() {
  msg = lobby || {t: "lobby", decks: []};
  detail = null;
  menuUid = null;
  pileOpen = null;
  gameMenu = false;
  surrenderArm = false;
  revealCur = null;
  revealQueue = [];
  bannerText = null;
  inLobby = true;
  render();
}

function onAction(kind, el) {
  if (kind === "log") logOpen = true;
  else if (kind === "closelog") { closeState("log"); render(); closeOv("log"); return; }
  else if (kind === "closepile") { closeState("pile"); render(); closeOv("pile"); return; }
  else if (kind === "unfocus") focusUid = null;
  else if (kind === "detail") {
    detail = findCard(el.getAttribute("data-uid"));
    closeState("menu");
    ovStack = ovStack.filter((n) => n !== "menu");
  } else if (kind === "closedetail") { closeState("detail"); render(); closeOv("detail"); return; }
  else if (kind === "closereveal") { dismissReveal(); return; }
  else if (kind === "gamemenu") { gameMenu = true; surrenderArm = false; }
  else if (kind === "closegamemenu") { closeState("gamemenu"); render(); closeOv("gamemenu"); return; }
  else if (kind === "surrender") {
    if (!surrenderArm) {
      surrenderArm = true;
      clearTimeout(surrenderTimer);
      surrenderTimer = setTimeout(() => { surrenderArm = false; if (gameMenu) render(); }, 3000);
    } else {
      clearTimeout(surrenderTimer);
      surrenderArm = false;
      gameMenu = false;
      ovStack = ovStack.filter((n) => n !== "gamemenu");
      try { if (history.state && history.state.ov === "gamemenu") { popGuard = true; history.back(); setTimeout(() => { popGuard = false; }, 800); } } catch (e) { popGuard = false; }
      if (!MOCK && ws) ws.send({t: "surrender"});
      else { msg = lobby || {t: "lobby", decks: []}; }
    }
  } else if (kind === "backlobby") {
    if (!MOCK && ws && msg && msg.view && !msg.result) ws.send({t: "surrender"});
    goLobby();
    return;
  } else if (kind === "chainset") {
    const v = el.getAttribute("data-v");
    if (v === "auto" || v === "all") {
      chainMode = v;
      try { localStorage.setItem("ygo-chain", v); } catch (e) { /* ignore */ }
      if (!MOCK && ws) ws.send({t: "settings", chain: v});
    }
  } else if (kind === "start" || kind === "rematch") {
    if (MOCK) { loadMock(1); return; }
    const ydk = kind === "start" ? ((document.getElementById("ydk") || {}).value || "").trim() : "";
    events = [];
    revealQueue = [];
    revealCur = null;
    bannerText = null;
    sendGame({t: "new_game", deck: ydk ? {ydk} : lobbyPick.you, bot_deck: lobbyPick.bot,
      first: lobbyPick.first, mode: lobbyPick.mode, chain: chainMode});
    return;
  } else if (kind === "newgame") {
    goLobby();
    return;
  } else return;
  render();
}

// Long-press opens the full card detail (board, hand and sheet cells).
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
// Press-and-hold on the reveal keeps it open until release.
ovReveal.addEventListener("pointerdown", () => {
  if (revealCur) { revealHold = true; clearTimeout(revealTimer); }
});
for (const evt of ["pointerup", "pointercancel"]) {
  ovReveal.addEventListener(evt, () => {
    if (revealCur && revealHold) { revealHold = false; dismissReveal(); }
  });
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
  gameMenu = false;
  surrenderArm = false;
  revealCur = null;
  revealQueue = [];
  bannerText = null;
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
  if (qs.get("gamemenu") && msg.t === "state") gameMenu = true;
  render();
  if (qs.get("reveal") && msg.t === "state") {
    const c = findCard(qs.get("reveal"));
    if (c && !isHidden(c)) {
      revealQueue = [];
      revealCur = {card: c, actor: "bot", kind: "activate", n: -1, text: "Activates " + (c.name || "")};
      render();
    }
  }
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
        if (inLobby) return;   // Back to lobby: ignore states until the next new_game
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
