import {connectWS} from "./ws.js";
import {barHTML, sideHTML, midHTML, phaseBtnsHTML, chainHTML, winchanceHTML} from "./render/board.js";
import {handHTML, sheetHTML, pileSheetHTML, infoHTML, actionableMap, statsLine, pileTitle} from "./render/hand.js";
import {logHTML} from "./render/log.js";
import {LOG_TABS, tabStripHTML, tabPanelOpen, statsPanelHTML, cardsPanelHTML, reviewPanelHTML} from "./render/stats.js";
import {esc, imgUrl, isHidden, phHTML} from "./render/card.js";
import {morph} from "./render/morph.js";
import {createCuePlayer, recapItem, chainTracker, planSteps, stepText} from "./cues.js";
import {kwHTML, glossaryEntry} from "./glossary.js";

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
  "mock/11-cues-activate.json",
  "mock/12-cues-chain.json",
  "mock/13-cues-summon.json",
  "mock/14-cues-battle.json",
  "mock/15-cues-moves.json",
  "mock/16-cues-recap.json",
  "mock/17-duel-stats.json",
  "mock/18-duel-early.json",
  "mock/19-winchance-board.json",
  "mock/20-winchance-stats.json",
];

const qs = new URLSearchParams(location.search);
const MOCK = qs.has("mock");
let mockIdx = Math.max(0, parseInt(qs.get("i") || "0", 10) || 0);
const TOKEN = qs.get("t") || "";
// Spectator replay (game_video.py): ?replay=<frames-json-url>. Fetches the
// frames once, renders each through the mock/live render path, no inputs.
const REPLAY_URL = qs.get("replay");
const IS_REPLAY = REPLAY_URL != null && REPLAY_URL !== "";
let replayFrames = [];
let replayIdx = -1;
if (IS_REPLAY) {
  window.frameCount = 0;
  window.showFrame = async () => {};
}

const app = document.getElementById("app");
// Screen readers: one polite live region for cue lines, prompts and results (never #app itself,
// which would read every re-render).
const srLive = document.getElementById("sr-live");
function say(text) {
  if (!srLive || !text || IS_REPLAY) return;
  const p = document.createElement("p");
  p.textContent = text;
  srLive.appendChild(p);
  while (srLive.children.length > 4) srLive.firstChild.remove();
}
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
const ovCues = ov("ov-cues");
const ovKw = ov("ov-kw");

let msg = null;          // latest server/mock message
let locked = false;      // input disabled until the next state
let sel = [];            // selected option indices (multi-select)
let expanded = null;     // option index whose effect text is expanded
let menuUid = null;      // uid with an open action menu
let menuRect = null;
let focusUid = null;     // card shown in the info panel
let logOpen = false;
let logAll = false;      // show passes / auto-passes / zone picks too (drawer toggle)
const QUIET = new Set(["pass", "auto_pass", "place"]);
const shownEvents = () => (logAll ? events : events.filter((e) => !QUIET.has(e.kind)));
let logTab = "log";      // "log" | "stats" | "cards" | "review" (the duel-log sheet tabs)
try {
  const savedTab = localStorage.getItem("ygo-logtab");
  if (savedTab === "log" || savedTab === "stats" || savedTab === "cards" || savedTab === "review") logTab = savedTab;
} catch (e) { /* private mode: stay on log */ }
function setLogTab(t) {
  if (t !== "log" && t !== "stats" && t !== "cards" && t !== "review") return;
  logTab = t;
  try { localStorage.setItem("ygo-logtab", t); } catch (e) { /* ignore */ }
}
let detail = null;       // card for the full-screen detail view
let lobby = null;
let lobbyPick = {you: null, bot: null, first: "rps", mode: "product", play: "duel"};
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
let gameMode = null;     // the running game's bot strength (for the win-chance note)

// Win chance (the bot's calibrated estimate of your chance to win): a switch in
// the game menu, off by default so it never steers play unless asked for.
let winChance = false;
try {
  winChance = localStorage.getItem("ygo-winchance") === "1";
} catch (e) { /* private mode: stay off */ }

// Pacing cues (web/cues.js): speed "1" | "2" | "instant", pause after each bot
// action, and the "what happened" recap at the start of your turn.
let cueSpeed = "1";
let cuePause = false;
let cueRecap = true;
try {
  const sp = localStorage.getItem("ygo-cue-speed");
  if (sp === "1" || sp === "2" || sp === "instant") cueSpeed = sp;
  cuePause = localStorage.getItem("ygo-cue-pause") === "1";
  cueRecap = localStorage.getItem("ygo-cue-recap") !== "0";
} catch (e) { /* private mode: defaults */ }
function saveSetting(k, v) {
  try { localStorage.setItem(k, v); } catch (e) { /* ignore */ }
}
let recap = null;        // {turn, who, items:[{text, sub, card}]} shown at the start of your turn
let recapAcc = null;     // collecting during the opponent's turn
let lastCueN = -1;       // cues are deduped by trace index (peek states may repeat them)
let lastCueTurn = 0;

// Post-game review (web-ux-plan step 3): opened only after the game ends.
let review = null;         // the server's {"decisions", "counts", "accuracy", "missed"} or null
let reviewLoading = false;
let reviewProgress = null; // {done, total}
let reviewFilter = "all";  // "all" | "mistakes" | "missed"
let reviewFrame = null;    // {"k","view","options","entry"} for the tapped decision
let overDismissed = false; // "Review game" dismisses the game-over card; "Result" brings it back

// Practice mode + "Ask the bot" hints + what-if (step 4).
let practiceMode = false;  // the running game is practice (hints allowed)
let hint = null;           // the server's {"prompt","options":[{index,label,p,est}],"ms"} or null
let hintLoading = false;
let hintPrompt = null;     // prompt id the current hint belongs to
let whatIf = null;         // {"source","k","option","turn"} for a what-if game

// Puzzles (step 5, item 8): list from the server, current puzzle banner + result,
// solved marks remembered per browser, in-page solution replay.
let puzzles = null;        // [{id,decks,turn,difficulty,goal,n_plays}] or null (not loaded yet)
let puzzlesOpen = false;   // the puzzles list sheet is open (lobby)
let puzzle = null;         // the running puzzle {"id","goal","n_plays","status"} or null
let solvedIds = {};        // id -> true (localStorage ygo-puzzles-solved)
try {
  solvedIds = JSON.parse(localStorage.getItem("ygo-puzzles-solved") || "{}") || {};
} catch (e) { solvedIds = {}; }
function isSolved(id) { return !!(id && solvedIds[id]); }
function markSolved(id) {
  if (!id) return;
  solvedIds[id] = true;
  try { localStorage.setItem("ygo-puzzles-solved", JSON.stringify(solvedIds)); } catch (e) { /* ignore */ }
}
let solutionFrames = null; // puzzle_solution frames while showing the solution
let solutionIdx = -1;
let inSolution = false;
let liveEnd = null;       // the puzzle end state saved while showing the solution
let hidePuzzleResult = false; // suppress the Solved/Not-solved dialog during solution playback

// Rules in context (step 5, item 6) + undo (item 9).
let kwPop = null;          // {"id","term","long"} for the open keyword popover, or null
let kwOpener = null;       // the keyword element to return focus to on close
let ruleNotesOn = true;    // "Rule notes" switch (cue notes), on by default
try {
  ruleNotesOn = localStorage.getItem("ygo-rule-notes") !== "0";
} catch (e) { /* private mode: stay on */ }
let canUndo = false;       // the server says a practice undo is possible right now
let undoPending = false;   // an undo was sent: announce it when the rewind state lands

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
let popGuard = 0;        // pops we triggered ourselves (history.back after an X/scrim close)
function openOvNames() {
  const o = [];
  if (logOpen) o.push("log");
  if (menuUid) o.push("menu");
  if (gameMenu) o.push("gamemenu");
  if (pileOpen) o.push("pile");
  if (detail) o.push("detail");
  if (kwPop) o.push("kw");
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
  else if (name === "detail") { detail = null; kwPop = null; }
  else if (name === "kw") kwPop = null;
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
      popGuard += 1;
      history.back();
    }
  } catch (e) { popGuard = 0; }
}
window.addEventListener("popstate", () => {
  if (popGuard > 0) { popGuard -= 1; return; }
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
  const w0 = w, hw0 = hw;
  for (let i = 0; i < 200 && need(w, hw) > H; i++) {
    if (hw > w * 1.25) hw -= 1;
    else w -= 0.5;
  }
  w = Math.max(30, w);
  // Short viewports (200% zoom on a laptop, landscape phones): rather than shrinking cards
  // below legibility, keep them readable and let the page scroll (WCAG 1.4.4 / 1.4.10).
  const short = w < Math.min(w0, 44);
  if (short) {
    w = Math.min(w0, 48);
    hw = Math.min(hw0, w * 1.4);
    root.style.setProperty("--app-h", Math.ceil(need(w, hw) + 120) + "px");
  }
  root.classList.toggle("short", short);
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
  if (!code || IS_REPLAY) return null;
  const t = texts.get(code);
  if (t) return typeof t === "object" ? t : null;
  if (MOCK) return null;   // mock text (if any) is preloaded from mock/card-text.json
  texts.set(code, "pending");
  fetch("/card/" + code + (TOKEN ? "?t=" + encodeURIComponent(TOKEN) : ""))
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((j) => { texts.set(code, {text: j.text || "", type_str: j.type_str || ""}); render(); })
    .catch(() => { texts.set(code, "failed"); });
  return null;
}

function addEvents(list, reset, notices) {
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
  if (!reset && !wasEmpty && added.length && notices !== false) queueNotices(added);
}

// Bot notices: activations get a centered reveal (queued), other
// board-changing bot events get a small top-of-board banner. Only the
// newest 3 events of a batch can trigger anything.
function queueNotices(added) {
  if (IS_REPLAY) return;   // the video shows every frame; no pop-up reveals or banners
  const fresh = added.slice().sort((a, b) => (a.n || 0) - (b.n || 0)).slice(-3);
  for (const e of fresh) {
    if (e.actor !== "bot") continue;
    if (e.kind === "activate" && e.card && !isHidden(e.card)) {
      revealQueue.push(e);
      if (revealQueue.length > 2) revealQueue.shift();  // a long combo: show the latest, never a wall of reveals
    } else if (["summon", "spsummon", "attack", "mset", "sset"].includes(e.kind)) {
      const t = e.text || "";
      showBanner("ygo-zero " + t.charAt(0).toLowerCase() + t.slice(1));
    }
  }
  pumpReveal();
}

function showBanner(text) {
  say(text);
  bannerText = text;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { bannerText = null; render(); }, 1400);
  render();
}

function pumpReveal() {
  if (revealCur || revealLeaving || !revealQueue.length) return;
  revealCur = revealQueue.shift();
  say("ygo-zero activates " + ((revealCur.card && revealCur.card.name) || "a card"));
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
  return '<div id="reveal" data-key="reveal" data-act="closereveal" aria-hidden="true">' +
    face + '<div class="rkicker">ygo-zero activates</div>' +
    "<h2>" + esc(c.name || "") + "</h2>" +
    '<div class="rtext">' + esc(txt) + "</div></div>";
}

// ---- outbound ----
function choose(index) {
  if (locked || !msg || !msg.prompt) return;
  locked = true;
  recap = null;   // the recap stays until your first move
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

// Practice-mode undo: rewrite to the previous decision; the server re-sends the
// log and cues from the rewound position (see web/server.py `on_undo`).
function doUndo() {
  if (!msg || msg.result || !practiceMode || !canUndo || locked || msg.thinking) return;
  if (MOCK) { say("Undid your last decision"); return; }
  if (!ws) return;
  undoPending = true;
  events = [];
  hint = null;
  hintLoading = false;
  hintPrompt = null;
  resetCues();
  render();
  ws.send({t: "undo"});
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
  if (IS_REPLAY) return "";   // the caption bar carries the narration instead
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

function speedSeg(v, label) {
  return '<button role="radio" aria-checked="' + (cueSpeed === v) + '" class="' + (cueSpeed === v ? "sel" : "") +
    '" data-act="cuespeed" data-v="' + v + '">' + label + "</button>";
}

function toggleRow(act, on, label, hint) {
  return '<button class="mrow tog" role="switch" aria-checked="' + on + '" data-act="' + act + '">' +
    '<span class="tl"><span>' + esc(label) + '</span><span class="th">' + esc(hint) + "</span></span>" +
    '<span class="sw' + (on ? " on" : "") + '"><i></i></span></button>';
}

// "While you waited": the opponent's turn, in the info panel under your bar
// (non-modal: the board stays usable; it closes with its X or your first move).
function recapHTML(hint) {
  if (!recap) return "";
  const rows = recap.items.map((it, i) => {
    const c = it.card;
    const thumb = c && c.code ? '<span class="rth">' + phHTML(c) +
      (!MOCK ? '<img alt="" src="' + imgUrl(c.code, "small") + '" onerror="this.remove()">' : "") + "</span>"
      : '<span class="rth none"></span>';
    const tag = c && c.code ? "button" : "div";
    return "<" + tag + ' class="rrow' + (c && c.owner === "opp" ? " opp" : " me") + (it.neg ? " neg" : "") + '"' +
      (c && c.code ? ' data-recap="' + i + '"' : "") + ">" + thumb +
      '<span class="rtx"><span>' + esc(it.text) + "</span>" + (it.sub ? '<span class="rsub">' + esc(it.sub) + "</span>" : "") +
      "</span></" + tag + ">";
  }).join("");
  return '<div class="recap" data-key="recap" role="region" aria-label="What happened">' +
    '<div class="rhead"><span class="kicker">While you waited</span><span class="rt">' + esc(recap.who) +
    " · turn " + recap.turn + '</span><button class="iconbtn sm" data-act="closerecap" aria-label="Close recap">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
    '<div class="rlist">' + rows + "</div>" + (hint ? '<div class="hint">' + esc(hint) + "</div>" : "") + "</div>";
}

// Rule popover: the glossary `long` text, stacked above the card detail dialog.
function kwPopHTML() {
  if (!kwPop) return "";
  return '<div id="kwscrim" data-key="kwscrim" data-act="closekw"></div>' +
    '<div id="kwpop" data-key="kwpop" role="dialog" aria-modal="true" tabindex="-1" aria-label="' +
    esc("Rule: " + kwPop.term) + '">' +
    '<div class="mhead">' + esc(kwPop.term) + '<button class="iconbtn" data-act="closekw" aria-label="Close rule">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
    '<div class="mbody"><p>' + esc(kwPop.long) + '</p></div></div>';
}

function openKw(id, el) {
  const g = glossaryEntry(id);
  if (!g) return;
  kwOpener = el || null;
  kwPop = {id: g.id, term: g.term, long: g.long};
  render();
}

function closeKw() {
  if (!kwPop) return;
  const back = kwOpener;
  const id = kwPop.id;
  kwPop = null;
  kwOpener = null;
  render();
  let el = back && back.isConnected && visible(back) ? back : null;
  if (!el) el = [...document.querySelectorAll('[data-act="kw"][data-kw="' + id + '"]')].find(visible) || null;
  if (el) el.focus({preventScroll: true});
}

function gameMenuHTML() {
  const seg = (v, label) =>
    '<button role="radio" aria-checked="' + (chainMode === v) + '" class="' + (chainMode === v ? "sel" : "") +
    '" data-act="chainset" data-v="' + v + '">' + label + "</button>";
  const wcNote = (gameMode === "random" || gameMode === "fast")
    ? '<div class="mhint">Win chance needs the full bot.</div>'
    : "";
  return '<div id="gscrim" data-key="gscrim" data-act="closegamemenu"></div>' +
    '<div id="gamemenu" data-key="gamemenu" role="dialog" aria-modal="true" aria-label="Game menu" tabindex="-1">' +
    '<div class="mhead">Game menu<button class="iconbtn" data-act="closegamemenu" aria-label="Close game menu">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
    '<div class="mbody">' +
    '<button class="mrow danger' + (surrenderArm ? " confirm" : "") + '" data-act="surrender">' +
    (surrenderArm ? "Tap again to surrender" : "Surrender") + "</button>" +
    '<button class="mrow" data-act="backlobby">Back to lobby</button>' +
    '<div class="msep"></div>' +
    toggleRow("winchanceset", winChance, "Show win chance",
      "The bot's estimate of your chance to win, calibrated on ~3,000 games. Off by default so it doesn't steer your play.") +
    wcNote +
    '<div class="msep"></div>' +
    '<div class="mrow" style="pointer-events:none" id="gm-chain">Chain responses</div>' +
    '<div class="seg" role="radiogroup" aria-labelledby="gm-chain">' + seg("auto", "Auto") + seg("all", "All") + "</div>" +
    '<div class="mhint">Auto skips empty windows with nothing to respond to.</div>' +
    '<div class="msep"></div>' +
    '<div class="mrow" style="pointer-events:none" id="gm-pace">Pacing</div>' +
    '<div class="seg" role="radiogroup" aria-labelledby="gm-pace">' + speedSeg("1", "1×") + speedSeg("2", "2×") + speedSeg("instant", "Instant") + "</div>" +
    toggleRow("cuepause", cuePause, "Pause after each bot action", "Tap anywhere or press Enter to continue.") +
    toggleRow("cuerecap", cueRecap, "Turn recap", "What the bot did, at the start of your turn.") +
    toggleRow("rulenotes", ruleNotesOn, "Rule notes", "A one-line rule tip on the first of each kind of play.") +
    '<div class="msep"></div>' +
    '<div class="mrow disabled" aria-disabled="true">Sound<span class="sub">Off</span></div>' +
    "</div></div>";
}

function replayName(who) {
  if (!msg.names) return undefined;
  return who === "you" ? msg.names[0] : msg.names[1];
}

// Quiet narration bar where the prompt sheet normally sits (replay only).
function captionHTML(m) {
  const c = m.caption || {};
  return '<div class="grab"></div><div class="stitle"><div class="tt"><span class="t">' +
    esc(c.actor ? c.actor + " — " + (c.text || "") : (c.text || "")) + "</span></div></div>";
}

// "Ask the bot" hint maps for the current prompt: by option index (sheets,
// phase buttons) and by card uid (idle/battle glowing cards).
function hintMaps(p) {  const byIndex = {}, byUid = {};
  if (!hint || !p || hintPrompt !== p.id) return {byIndex, byUid};
  const opts = p.options || [];
  (hint.options || []).forEach((ho, i) => {
    const rank = i + 1;
    const pct = Math.round((ho.p || 0) * 100);
    const entry = {rank, pct, est: ho.est != null ? ho.est : null, label: ho.label || ""};
    byIndex[ho.index] = entry;
    const o = opts.find((x) => x.index === ho.index);
    if (o && o.card && o.card.uid) byUid[o.card.uid] = entry;
  });
  return {byIndex, byUid};
}

// Hint bar: the returned top picks in words (badges on the options carry them
// visually). Sheet prompts keep Hint/Undo in the sheet header; idle/battle
// prompts get them in this row under your bar, which is always visible.
function hintBarHTML(p, withButton, idle, undo) {
  if (!practiceMode || !p || msg.result || IS_REPLAY) return "";
  let inner = "";
  if (hintLoading) {
    inner = '<span class="hintwait" role="status">Asking the bot…</span>';
  } else if (hint && hintPrompt === p.id && (hint.options || []).length) {
    inner = '<ol class="hintlist">' + hint.options.map((ho, i) => {
      const pct = Math.round((ho.p || 0) * 100);
      return '<li><span class="hintbadge" aria-hidden="true">' + (i + 1) + "</span> " +
        esc(ho.label || ("Option " + ho.index)) +
        ' <span class="hintconf">' + pct + "%</span><span class=\"sr-only\">, bot's confidence " +
        pct + " percent" + (ho.est != null ? ", win chance " + Math.round(ho.est * 100) + " percent" : "") + "</span>" +
        (ho.est != null ? '<span class="hintest"> · win ' + Math.round(ho.est * 100) + "%</span>" : "") + "</li>";
    }).join("") + "</ol>";
  } else {
    inner = '<span class="hintsub">bot\u2019s top picks</span>';
  }
  // Idle/battle prompts: Hint + Undo sit in this row under your bar (the bar
  // itself has no room for them next to the phase buttons at phone width).
  const undoBtn = undo && undo.show
    ? '<button class="hintbtn" data-act="undo" aria-label="Undo your last decision"' +
      (undo.enabled ? "" : ' disabled aria-disabled="true"') + ">Undo</button>" : "";
  return '<div class="hintbar" data-key="hintbar">' +
    (withButton ? '<button class="hintbtn" data-act="hint"' +
      ' aria-label="Ask the bot for a hint"' + (hintLoading || locked ? " disabled" : "") + ">Hint</button>" : "") +
    undoBtn + inner + "</div>";
}

function renderGame() {
  const v = msg.view;
  const p = msg.prompt;
  const act = locked ? {} : actionableMap(p);
  const hm = hintMaps(p);
  const undo = {
    show: practiceMode && !msg.result,
    enabled: practiceMode && canUndo && !locked && !msg.thinking && !msg.result,
  };
  const replayLike = IS_REPLAY || inSolution;
  app.classList.toggle("replay", replayLike);
  if (focusUid && !findCard(focusUid)) focusUid = null;
  const yourTurn = v.turn_player === "you";

  let h = barHTML(v.bot, "bot", {thinking: msg.thinking, active: !yourTurn, name: replayName("bot")});
  if (whatIf) h += '<div id="whatifbanner" data-key="whatifbanner">What if · from turn ' + esc(String(whatIf.turn != null ? whatIf.turn : "?")) + "</div>";
  if (puzzle && puzzle.status === "playing") h += '<div id="puzzlebanner" data-key="puzzlebanner">Puzzle · ' + esc(String(puzzle.goal || "Win this turn")) + "</div>";
  if (IS_REPLAY) {
    // Spectator: seat 1's hand face-up under the top header (frames are reveal_all).
    h += '<div id="thand" data-key="thand">' + handHTML(v.bot.hand, {}, MOCK, null) + "</div>";
  }
  if (bannerText) h += '<div id="botbanner" data-key="botbanner">' + esc(bannerText) + "</div>";
  h += '<div id="board" data-key="board">';
  h += '<div class="field bot">' + sideHTML(v.bot, {mirror: true, mock: MOCK, actionable: act, focus: focusUid, hint: hm.byUid}) + "</div>";
  h += '<div id="mid"><div id="midstrip" class="zrow">' + midHTML(v, {mock: MOCK, actionable: act, focus: focusUid, names: msg.names || null, hint: hm.byUid}) + "</div>";
  if (winChance && msg.winchance) h += winchanceHTML(msg.winchance);
  const chain = chainHTML(v.chain, MOCK, chainNeg);
  if (chain) h += '<div id="chain">' + chain + "</div>";
  h += "</div>";
  h += '<div class="field you">' + sideHTML(v.you, {mirror: false, mock: MOCK, actionable: act, focus: focusUid, hint: hm.byUid}) + "</div>";
  h += "</div>";
  h += barHTML(v.you, "you", {phaseBtns: phaseBtnsHTML(p, locked, hm.byIndex, false, null), active: yourTurn, name: replayName("you")});

  let info = null;
  let sheet = "";
  if (replayLike) {
    // Bottom order: header → seat 0's hand → last 5 log lines → caption bar.
    // The caption is a static block (see #app.replay #sheet): it never overlaps the hand.
    h += '<div id="hand" data-key="hand">' + handHTML(v.you.hand, {}, MOCK, null) + "</div>";
    h += '<div id="info" data-key="info">' +
      (inSolution ? '<div class="solbar" data-key="solbar"><button class="hintbtn" data-act="exitsolution">Exit solution</button></div>' : "") +
      infoHTML(null, shownEvents().slice(-5), MOCK, "", msg.names || null) + "</div>";
    sheet = msg.caption && !msg.result ? captionHTML(msg) : "";
  } else {
    if (focusUid) {
      const c = findCard(focusUid);
      const t = c && !isHidden(c) ? cardText(c.code) : null;
      info = {card: c, text: t && t.text, typeStr: t && t.type_str};
    }
    const isSheet = p && p.kind !== "idle" && p.kind !== "battle";
    h += '<div id="info" data-key="info">' + (isSheet ? "" : hintBarHTML(p, true, true, undo)) +
      (recap && !info ? recapHTML(hintFor(p))
      : infoHTML(info, shownEvents(), MOCK, hintFor(p), msg.names || null)) + "</div>";
    h += '<div id="hand" data-key="hand" class="' + (locked ? "locked" : "") + '">' + handHTML(v.you.hand, act, MOCK, focusUid, hm.byUid) + "</div>";

    sheet = pileOpen ? pileSheetHTML(pileOpen, pileCards(pileOpen), MOCK)
      : (p && !msg.result && isSheet
        ? sheetHTML(p, MOCK, sel, expanded, {hintByIndex: hm.byIndex, showHint: practiceMode, showUndo: undo.show, undoEnabled: undo.enabled}) + hintBarHTML(p, false)
        : "");
  }
  if (pileOpen) h += '<div id="pscrim" data-key="pscrim" data-act="closepile"></div>';
  if (sheet) {
    const lab = pileOpen ? ' role="dialog" aria-modal="true" tabindex="-1" aria-label="' + esc(pileTitle(pileOpen, msg.names || null)) + '"'
      : replayLike ? "" : ' role="region" aria-label="' + esc((p && p.text) || "Choose") + '"';
    h += '<div id="sheet" data-key="sheet" class="' + (locked && !pileOpen ? "locked" : "") + (pileOpen ? " pilesheet" : "") + '"' + lab + ">" + sheet + "</div>";
  }
  const restore = saveScrolls(app, ["#sheet .sbody", "#info .ftext"]);
  morph(app, h);
  restore();
  app.classList.toggle("busy", locked);
  fit();

  for (const who of ["you", "bot"]) {
    const el = app.querySelector(".pbar." + who + " .lp");
    const ov = cue.lpOverride(who);   // a damage cue already ticked the counter down
    const target = ov != null ? ov : v[who].lp;
    const prev = lpShown[who] == null ? target : lpShown[who];
    lpShown[who] = target;
    if (!el) continue;
    if (IS_REPLAY || ov != null) {
      el.textContent = String(target);   // static screenshot / cue-driven: no count animation
      const bar = app.querySelector(".pbar." + who + " .lpbar > i");
      if (bar && ov != null) bar.style.width = (Math.max(0, Math.min(1, target / 8000)) * 100).toFixed(1) + "%";
      continue;
    }
    if (target < prev) el.classList.add("flash");
    const t0 = performance.now();
    const step = (t) => {
      if (cue.lpOverride(who) != null) return;   // a damage cue owns the counter now
      const k = Math.min(1, (t - t0) / 400);
      el.textContent = String(Math.round(prev + (target - prev) * (1 - Math.pow(1 - k, 3))));
      if (k < 1) requestAnimationFrame(step);
      else setTimeout(() => el.classList.remove("flash"), 300);
    };
    el.textContent = String(prev);
    requestAnimationFrame(step);
  }

  const logh = logHTML(shownEvents(), MOCK);
  const sideTabs = tabStripHTML(logTab, "s");
  const sideBody = tabPanelOpen("s", logTab) + sidePanelInner(logTab, logh) + "</div>";
  morph(logside,
    '<div class="sidecard" data-key="side">' + (info && info.card ? infoHTML(info, null, MOCK, "")
    : recap && !IS_REPLAY ? recapHTML("") : '<p class="emptytx">Tap any card to read it here.</p>') + "</div>" +
    sideTabs + sideBody);
  // Keep the desktop log list pinned to the bottom only while on the Log tab.
  if (logTab === "log") {
    const sc = logside.querySelector("#spanel-log .loglist");
    if (sc) sc.scrollTop = sc.scrollHeight;
  }
  renderOverlays(logh);
  syncOv();
}

function logAllBtn() {
  return '<button class="logall" data-act="logall" aria-pressed="' + logAll + '">' +
    (logAll ? "Key moves" : "Show all") + "</button>";
}

// Inner HTML of one duel-log tab panel (the tab strip lives outside).
function sidePanelInner(which, logh) {
  const st = msg && msg.stats;
  if (which === "stats") return statsPanelHTML(st, MOCK);
  if (which === "cards") return cardsPanelHTML(st, MOCK);
  if (which === "review") {
    if (reviewLoading) {
      const p = reviewProgress ? "Reviewing… " + reviewProgress.done + "/" + reviewProgress.total : "Reviewing…";
      return '<p class="emptytx" role="status">' + esc(p) + "</p>";
    }
    return reviewPanelHTML(review, winChance, reviewFilter, MOCK,
      !!(msg && msg.result && overDismissed));
  }
  return '<div class="loghead" data-key="lhead">Duel log' + logAllBtn() + '</div><div class="loglist" data-key="llist">' + logh + "</div>";
}

function renderOverlays(logh) {
  let lh = "";
  if (logOpen) {
    const body = tabPanelOpen("d", logTab) + sidePanelInner(logTab, logh) + "</div>";
    lh = '<div id="scrim" data-key="scrim" data-act="closelog"></div>' +
      '<div id="logdrawer" data-key="logdrawer" role="dialog" aria-modal="true" aria-label="This duel" tabindex="-1">' +
      '<div class="dhead">This duel<button class="iconbtn" data-act="closelog" aria-label="Close duel log">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
      tabStripHTML(logTab, "d") + body + "</div>";
  }
  if (logTab === "log") morphBottom(ovLog, lh, "#logdrawer .dbody");
  else morph(ovLog, lh);

  let mh = "";
  if (menuUid && !locked) {
    const opts = optionsForUid(menuUid);
    const r = menuRect || {left: window.innerWidth / 2 - 40, top: 300, width: 80};
    const mw = Math.min(260, window.innerWidth - 16);
    const mhh = opts.length * 48 + 56;
    const top = Math.max(8, Math.min(window.innerHeight - mhh - 8, r.top - mhh - 8));
    const left = Math.max(8, Math.min(window.innerWidth - mw - 8, r.left + (r.width || 0) / 2 - mw / 2));
    const mc = findCard(menuUid);
    mh = '<div id="menu" data-key="menu" style="top:' + top + "px;left:" + left + "px;width:" + mw + 'px" role="menu" tabindex="-1"' +
      ' aria-label="' + esc("Actions" + (mc && mc.name ? ": " + mc.name : "")) + '">' +
      opts.map((o) => '<button role="menuitem" data-opt="' + o.index + '" class="' +
        (["activate", "normal", "special", "attack", "direct"].includes(o.verb) ? "primary" : "") + '">' +
        esc(o.label || o.verb) + "</button>").join("") +
      '<button role="menuitem" class="ghost" data-act="detail" data-uid="' + esc(menuUid) + '">Card details</button></div>';
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
    dh = '<div id="detail" data-key="detail" data-act="closedetail" role="dialog" aria-modal="true" tabindex="-1" aria-label="' +
      esc("Card details: " + (c.name || "hidden card")) + '">' +
      '<button class="dx" data-act="closedetail" aria-label="Close card details">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button>' + face +
      "<h2>" + esc(c.name || "Hidden card") + "</h2>" +
      (meta ? '<div class="typeline">' + esc(meta) + "</div>" : "") +
      '<div class="dtext">' + kwHTML(esc((t && t.text) || (MOCK ? "Card text loads from the server in a live game." : ""))) + "</div>" +
      '<div class="dclose" aria-hidden="true">Tap anywhere to close</div></div>';
  }
  morph(ovDetail, dh);

  morph(ovKw, kwPopHTML());

  if (!revealLeaving) morph(ovReveal, revealCur ? revealHTML(revealCur) : "");

  let oh = "";
  if (IS_REPLAY && msg && msg.title && replayIdx === 0) {
    oh += '<div id="retitle" data-key="retitle"><div class="box"><div class="kicker">ygo-zero replay</div>' +
      "<h2>" + esc(msg.title) + "</h2></div></div>";
  }
  const pz = puzzle || (msg && msg.puzzle) || null;
  if (pz && (pz.status === "solved" || pz.status === "failed") && msg && msg.view && !hidePuzzleResult && !inSolution) {
    const n = pz.n_plays != null ? pz.n_plays : (pz.nPlays || "");
    if (pz.status === "solved") {
      oh += '<div id="puzzleresult" data-key="puzzleresult" role="alertdialog" aria-modal="true" tabindex="-1" aria-label="Puzzle solved"><div class="box">' +
        '<div class="kicker">Puzzle · turn ' + msg.view.turn + "</div>" +
        "<h2>Solved</h2>" +
        '<p class="reason">Solved in ' + esc(String(n)) + " plays</p>" +
        '<div class="row"><button class="bigbtn primary" data-act="puzzlenext">Next puzzle</button>' +
        '<button class="bigbtn" data-act="puzzlesback">Back to puzzles</button></div></div></div>';
    } else {
      oh += '<div id="puzzleresult" data-key="puzzleresult" role="alertdialog" aria-modal="true" tabindex="-1" aria-label="Puzzle not solved"><div class="box">' +
        '<div class="kicker">Puzzle · turn ' + msg.view.turn + "</div>" +
        "<h2>Not solved</h2>" +
        '<p class="reason">' + esc(String(pz.goal || "")) + "</p>" +
        '<div class="row"><button class="bigbtn primary" data-act="puzzleretry">Retry</button>' +
        '<button class="bigbtn" data-act="puzzlesolution">Show solution</button></div>' +
        '<div class="row"><button class="bigbtn" data-act="puzzlesback">Back to puzzles</button></div></div></div>';
    }
  } else if (msg && msg.view && msg.result && !overDismissed) {
    const missed = review ? (review.missed || 0) : 0;
    const reviewLabel = reviewLoading
      ? ("Reviewing… " + (reviewProgress ? reviewProgress.done + "/" + reviewProgress.total : ""))
      : (review ? ("Review game" + (missed ? " · " + missed + " missed response" + (missed === 1 ? "" : "s") : ""))
        : "Review game");
    const reviewBtn = '<button class="bigbtn" data-act="review" aria-label="' + esc(reviewLabel) + '"' +
      (reviewLoading ? " disabled" : "") + ">" + esc(reviewLabel) + "</button>";
    if (msg.names) {
      const win = msg.result.winner;
      const loser = msg.names[0] === win ? msg.names[1] : msg.names[0];
      const reason = msg.result.reason === "surrender" && loser ? loser + " surrenders" : "";
      oh += '<div id="gameover" data-key="gameover" role="alertdialog" aria-label="Game over"><div class="box">' +
        '<div class="kicker">Duel over · turn ' + msg.view.turn + "</div>" +
        "<h2>" + esc(win) + " wins</h2>" +
        '<p class="reason">' + esc(reason) + "</p>" +
        '<div class="score"><span class="you">' + esc(msg.names[0]) + "<b>" + msg.view.you.lp + "</b></span>" +
        '<span class="bot">' + esc(msg.names[1]) + "<b>" + msg.view.bot.lp + "</b></span></div>" +
        '<div class="row"><button class="bigbtn primary" data-act="rematch">Rematch</button>' + reviewBtn + "</div></div></div>";
    } else {
      const win = msg.result.winner === "you";
      oh = '<div id="gameover" data-key="gameover" role="alertdialog" aria-modal="true" tabindex="-1" aria-label="' + (win ? "Victory" : "Defeat") + '"><div class="box ' + (win ? "win" : "loss") + '">' +
        '<div class="kicker">Duel over · turn ' + msg.view.turn + "</div>" +
        "<h2>" + (win ? "Victory" : "Defeat") + "</h2>" +
        '<p class="reason">' + esc(msg.result.reason === "win" ? "" : msg.result.reason || "") + "</p>" +
        '<div class="score"><span class="you">You<b>' + msg.view.you.lp + "</b></span>" +
        '<span class="bot">ygo-zero<b>' + msg.view.bot.lp + "</b></span></div>" +
        '<div class="row"><button class="bigbtn" data-act="newgame">New game</button>' +
        '<button class="bigbtn primary" data-act="rematch">Rematch</button></div>' +
        '<div class="row">' + reviewBtn + "</div></div></div>";
    }
  }
  morph(ovOver, oh);
  let rh = "";
  if (reviewFrame && reviewFrame.entry) {
    const e = reviewFrame.entry;
    const opts = (reviewFrame.options || []).map((o) =>
      '<div class="opt' + (o.label === (e.your && e.your.label) ? " you" : "") +
      (o.label === (e.agent && e.agent.label) ? " agent" : "") + '">' +
      '<span class="tx"><span class="lb">' + esc(o.label || "") + "</span></span>" +
      (o.label === (e.your && e.your.label) ? '<span class="tag">You</span>' : "") +
      (o.label === (e.agent && e.agent.label) ? '<span class="tag">Agent</span>' : "") + "</div>").join("");
    const whatif = (reviewFrame.options || []).map((o) =>
      '<div role="listitem"><button class="wopt" data-act="whatif" data-k="' + reviewFrame.k + '" data-option="' + o.index + '"' +
      ' aria-label="Try another move: ' + esc(o.label || ("option " + o.index)) + '">' +
      esc(o.label || ("Option " + o.index)) + "</button></div>").join("");
    rh = '<div id="rscrim" data-key="rscrim" data-act="closereviewframe"></div>' +
      '<div id="reviewframe" data-key="reviewframe" role="dialog" aria-modal="true" tabindex="-1" aria-label="' +
      esc("Decision " + reviewFrame.k + ": " + (e.mark || "")) + '">' +
      '<div class="mhead">Turn ' + esc(String(e.turn)) + " · " + esc(e.mark || "") +
      "<button class=\"iconbtn\" data-act=\"closereviewframe\" aria-label=\"Close decision board\">" +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
      '<div class="mbody"><div class="rduo"><span>You: ' + esc((e.your && e.your.label) || "") + "</span>" +
      "<span>Agent: " + esc((e.agent && e.agent.label) || "") + "</span></div>" +
      '<div class="sbody">' + opts + "</div>" +
      '<div class="whead">Try another move</div><div class="wlist" role="list" aria-label="Try another move">' +
      whatif + "</div></div></div>";
  }
  let rEl = document.getElementById("ov-review");
  if (!rEl) {
    rEl = document.createElement("div");
    rEl.id = "ov-review";
    document.getElementById("layer").appendChild(rEl);
  }
  morph(rEl, rh);
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
      ' decks" aria-label="Search decks" data-search="' + side + '"><div class="decklist" role="listbox" aria-label="' + esc(label) + '">';
    for (const x of decks) {
      h += '<button role="option" aria-selected="' + (cur === x.id) + '" class="deckrow' + (cur === x.id ? " sel" : "") + '" data-' + side + 'deck="' + esc(x.id) +
        '" data-dname="' + esc(x.name.toLowerCase()) + '"><span class="nm">' + esc(x.name) + '</span><span class="g">' +
        esc(GROUPS[x.group] || x.group) + "</span></button>";
    }
    h += "</div></div>";
  }
  return h + "</section>";
}

function segHTML(label, key, items) {
  return '<section class="field-group"><div class="flabel" id="lb-' + key + '">' + esc(label) + '</div><div class="seg" role="radiogroup" aria-labelledby="lb-' + key + '">' +
    items.map(([k, t]) => '<button role="radio" aria-checked="' + (lobbyPick[key] === k) + '" class="' +
      (lobbyPick[key] === k ? "sel" : "") + (t.length > 10 ? " wide" : "") + '" data-' + key + '="' + k + '">' + t + "</button>").join("") +
    "</div></section>";
}

// Puzzles list sheet (lobby): grouped by difficulty, quiet Solved text marks.
function puzzlesSheetHTML() {
  if (!puzzlesOpen) return "";
  const list = Array.isArray(puzzles) ? puzzles : [];
  const groups = [["easy", "Easy"], ["medium", "Medium"], ["hard", "Hard"]];
  let body = "";
  for (const [key, label] of groups) {
    const rows = list.filter((p) => (p.difficulty || "medium") === key);
    body += '<div class="msep"></div><div class="mrow" style="pointer-events:none">' + esc(label) + "</div>";
    if (!rows.length) {
      body += '<p class="emptytx">No ' + esc(label.toLowerCase()) + " puzzles yet.</p>";
      continue;
    }
    body += '<div class="plist" role="list" aria-label="' + esc(label) + ' puzzles">' + rows.map((p) => {
      const title = esc(p.decks[0] + " vs " + p.decks[1] + " · Turn " + p.turn);
      return '<div role="listitem"><button class="prow" data-act="puzzlestart" data-id="' + esc(p.id) + '">' +
        '<span class="pt"><span class="ptitle">' + title + "</span>" +
        '<span class="psub">' + esc(p.goal || "") + "</span></span>" +
        (isSolved(p.id) ? '<span class="psolved">Solved</span>' : "") +
        "</button></div>";
    }).join("") + "</div>";
  }
  return '<div id="pscrim" data-key="pscrim" data-act="closepuzzles"></div>' +
    '<div id="puzzlesheet" data-key="puzzlesheet" role="dialog" aria-modal="true" tabindex="-1" aria-label="Puzzles">' +
    '<div class="mhead">Puzzles<button class="iconbtn" data-act="closepuzzles" aria-label="Close puzzles">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
    '<div class="mbody">' + body + "</div></div>";
}

function renderLobby() {
  const decks = (lobby && lobby.decks) || [];
  if (!lobbyPick.you && decks.length) lobbyPick.you = decks[0].id;
  if (!lobbyPick.bot && decks.length) lobbyPick.bot = decks[Math.min(1, decks.length - 1)].id;
  let h = '<div id="lobby"><div class="lscroll">';
  h += '<header class="hero"><div class="fan" aria-hidden="true"><i></i><i></i><i></i></div>' +
    '<h1>ygo<span>-zero</span></h1><p>Duel the bot from your phone.</p></header>';
  h += '<h2 class="sr-only">New duel</h2>';
  h += pickerHTML(decks, "you", "Your deck");
  h += pickerHTML(decks, "bot", "ygo-zero plays");
  h += segHTML("Mode", "play", [["duel", "Duel"], ["practice", "Practice"]]);
  h += segHTML("Going first", "first", [["rps", "Rock-paper-scissors"], ["me", "Me"], ["bot", "Bot"]]);
  h += segHTML("Bot strength", "mode", [["product", "Full"], ["fast", "Fast"], ["random", "Random"]]);
  h += '<details class="ydk"><summary>Paste a .ydk instead<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></summary>' +
    '<textarea id="ydk" aria-label="Deck list in .ydk format" placeholder="#main&#10;14558128&#10;…" spellcheck="false"></textarea></details>';
  h += '<section class="field-group"><div class="flabel">Puzzles</div>' +
    '<button class="pickbtn" data-act="puzzles" aria-haspopup="dialog"><span class="nm">Win this turn puzzles</span>' +
    '<span class="g">' + esc(puzzles ? String(puzzles.length) : "…") + '</span>' +
    '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></button></section>';
  if (MOCK) h += '<div class="mocknav">mock ' + (mockIdx + 1) + "/" + MOCKS.length + " · ← → to step</div>";
  h += '</div><div id="startwrap"><button id="startbtn" data-act="start">Start duel</button></div></div>';
  app.innerHTML = h;
  app.classList.remove("busy");
  logside.innerHTML = '<div class="loghead">Duel log</div><div class="loglist"><p class="emptytx">No duel yet.</p></div>';
  morph(ovLog, "");
  morph(ovMenu, "");
  morph(ovGame, puzzlesSheetHTML());
  morph(ovDetail, "");
  morph(ovReveal, "");
  morph(ovOver, "");
  ovStack = [];
}

function render() {
  // Static screenshots in replay mode: no entry animations, no transitions.
  document.documentElement.classList.toggle("replay", IS_REPLAY);
  const before = snapshotRects();
  if (!msg || msg.t === "lobby") {
    renderLobby();
    manageFocus();
    return;
  }
  if (msg.t === "state") {
    renderGame();
    if (!IS_REPLAY) flipAnimate(before);
    document.title = msg.prompt && msg.prompt.text ? msg.prompt.text + " · ygo-zero"
      : msg.caption && msg.caption.text ? msg.caption.text + " · ygo-zero" : "ygo-zero";
    manageFocus();
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
  // "Pause after each bot action": a tap anywhere steps to the next action (the
  // menu and log stay reachable). While cues play, taps reach the board as usual.
  if (cue.waiting() && !t.closest('[data-act="gamemenu"], [data-act="log"]')) {
    cue.resume();
    return;
  }
  const info = t.closest("[data-detailcard], .infobtn");
  if (info && (info.getAttribute("data-detailcard") != null || info.getAttribute("data-detailuid") != null)) {
    if (onInfoBtn(info)) return;
  }
  if (onRecapRow(t)) return;
  const kw = t.closest('[data-act="kw"]');
  if (kw) {
    // A keyword inside an option row: open the rule, never choose the option.
    openKw(kw.getAttribute("data-kw"), kw);
    return;
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
  for (const key of ["first", "mode", "play"]) {
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

// A recap line opens its card's details.
function onRecapRow(target) {
  const rr = target.closest("[data-recap]");
  if (!rr || !recap) return false;
  const it = recap.items[parseInt(rr.getAttribute("data-recap"), 10)];
  if (it && it.card && it.card.code) {
    detail = Object.assign({uid: "recap", pos: null, atk: 0, def: 0, level: 0, rank: 0, link: 0,
      counters: 0, overlays: []}, it.card, {faceup: true});
    render();
  }
  return true;
}

// A stats-cards cell opens that card's details (synthetic card: code + name).
function statName(code) {
  const st = msg && msg.stats;
  if (!st) return "";
  for (const e of [...(st.opp_seen || []), ...(st.you_used || [])]) {
    if (e && e.code === code) return e.name || "";
  }
  return "";
}

function openStatDetail(code) {
  if (!code) return;
  detail = {uid: "stat:" + code, code, name: statName(code) || ("#" + code),
    faceup: true, pos: null, atk: 0, def: 0, level: 0, rank: 0, link: 0,
    counters: 0, overlays: []};
  if (logOpen) {  // the detail takes over the drawer's history entry (back closes the detail)
    logOpen = false;
    const i = ovStack.indexOf("log");
    if (i >= 0) {
      ovStack[i] = "detail";
      try { history.replaceState({ov: "detail"}, ""); } catch (e) { /* ignore */ }
    }
  }
  render();
}

function onOverlayClick(ev) {
  const tab = ev.target.closest("[data-logtab]");
  if (tab) {
    setLogTab(tab.getAttribute("data-logtab"));
    render();
    return;
  }
  const rf = ev.target.closest("[data-reviewfilter]");
  if (rf) {
    reviewFilter = rf.getAttribute("data-reviewfilter") || "all";
    render();
    return;
  }
  const rk = ev.target.closest("[data-reviewk]");
  if (rk) {
    const k = parseInt(rk.getAttribute("data-reviewk"), 10);
    if (MOCK) {
      fetch("mock/review-frame.json?v=2").then((r) => r.json()).then((j) => {
        reviewFrame = j;
        render();
      }).catch(() => {});
    } else if (ws) {
      ws.send({t: "review_frame", k});
    }
    return;
  }
  const sc = ev.target.closest("[data-statcode]");
  if (sc) {
    openStatDetail(parseInt(sc.getAttribute("data-statcode"), 10));
    return;
  }
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
    if (c && !isHidden(c)) {
      detail = c;
      if (logOpen) {  // the detail takes over the drawer's history entry (back closes the detail)
        logOpen = false;
        const i = ovStack.indexOf("log");
        if (i >= 0) {
          ovStack[i] = "detail";
          try { history.replaceState({ov: "detail"}, ""); } catch (e) { /* ignore */ }
        }
      }
      render();
    }
    return;
  }
  if (onRecapRow(ev.target)) return;
  const act = ev.target.closest("[data-act]");
  if (act) onAction(act.getAttribute("data-act"), act);
}
document.getElementById("layer").addEventListener("click", onOverlayClick);
logside.addEventListener("click", onOverlayClick);

function sendGame(o) {
  inLobby = false;
  if (MOCK) return;
  if (ws) ws.send(o);
}

// Push both client settings (the server keeps the win-chance switch per
// connection, off by default, so reconnects re-send it).
function sendSettings() {
  if (MOCK || !ws) return;
  ws.send({t: "settings", chain: chainMode, winchance: winChance});
}

function goLobby() {
  msg = lobby || {t: "lobby", decks: []};
  detail = null;
  menuUid = null;
  pileOpen = null;
  gameMenu = false;
  gameMode = null;
  surrenderArm = false;
  revealCur = null;
  revealQueue = [];
  bannerText = null;
  review = null;
  reviewLoading = false;
  reviewProgress = null;
  reviewFrame = null;
  overDismissed = false;
  practiceMode = false;
  hint = null;
  hintLoading = false;
  hintPrompt = null;
  whatIf = null;
  puzzle = null;
  puzzlesOpen = false;
  inSolution = false;
  solutionFrames = null;
  solutionIdx = -1;
  liveEnd = null;
  hidePuzzleResult = false;
  inLobby = true;
  resetCues();
  render();
}

function onAction(kind, el) {
  if (kind === "logall") logAll = !logAll;
  else if (kind === "log") logOpen = true;
  else if (kind === "closelog") { closeState("log"); render(); closeOv("log"); return; }
  else if (kind === "closepile") { closeState("pile"); render(); closeOv("pile"); return; }
  else if (kind === "unfocus") focusUid = null;
  else if (kind === "detail") {
    detail = findCard(el.getAttribute("data-uid"));
    closeState("menu");
    ovStack = ovStack.filter((n) => n !== "menu");
  }   else if (kind === "closedetail") { closeState("detail"); render(); closeOv("detail"); return; }
  else if (kind === "closereveal") { dismissReveal(); return; }
  else if (kind === "kw") { openKw(el.getAttribute("data-kw"), el); return; }
  else if (kind === "closekw") { closeKw(); return; }
  else if (kind === "rulenotes") {
    ruleNotesOn = !ruleNotesOn;
    saveSetting("ygo-rule-notes", ruleNotesOn ? "1" : "0");
  } else if (kind === "undo") {
    doUndo();
    return;
  } else if (kind === "gamemenu") { gameMenu = true; surrenderArm = false; }
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
      closeOv("gamemenu");
      if (!MOCK && ws) ws.send({t: "surrender"});
      else { msg = lobby || {t: "lobby", decks: []}; }
    }
  } else if (kind === "backlobby") {
    if (!MOCK && ws && msg && msg.view && !msg.result) ws.send({t: "surrender"});
    goLobby();
    return;
  } else if (kind === "cuespeed") {
    const v = el.getAttribute("data-v");
    if (v === "1" || v === "2" || v === "instant") {
      cueSpeed = v;
      saveSetting("ygo-cue-speed", v);
      if (v === "instant") cue.flush();
    }
  } else if (kind === "cuepause") {
    cuePause = !cuePause;
    saveSetting("ygo-cue-pause", cuePause ? "1" : "0");
    if (!cuePause && cue.waiting()) cue.resume();
  } else if (kind === "cuerecap") {
    cueRecap = !cueRecap;
    saveSetting("ygo-cue-recap", cueRecap ? "1" : "0");
  } else if (kind === "closerecap") {
    recap = null;
  } else if (kind === "chainset") {
    const v = el.getAttribute("data-v");
    if (v === "auto" || v === "all") {
      chainMode = v;
      try { localStorage.setItem("ygo-chain", v); } catch (e) { /* ignore */ }
      if (!MOCK && ws) sendSettings();
    }
  } else if (kind === "winchanceset") {
    winChance = !winChance;
    saveSetting("ygo-winchance", winChance ? "1" : "0");
    if (!MOCK && ws) sendSettings();
  } else if (kind === "start" || kind === "rematch") {
    if (MOCK) { loadMock(1); return; }
    const ydk = kind === "start" ? ((document.getElementById("ydk") || {}).value || "").trim() : "";
    events = [];
    revealQueue = [];
    revealCur = null;
    bannerText = null;
    review = null;
    reviewLoading = false;
    reviewProgress = null;
    reviewFrame = null;
    overDismissed = false;
    hint = null;
    hintLoading = false;
    hintPrompt = null;
    whatIf = null;
    gameMode = lobbyPick.mode;
    practiceMode = lobbyPick.play === "practice";
    resetCues();
    sendGame({t: "new_game", deck: ydk ? {ydk} : lobbyPick.you, bot_deck: lobbyPick.bot,
      first: lobbyPick.first, mode: lobbyPick.mode, chain: chainMode,
      practice: lobbyPick.play === "practice"});
    return;
  } else if (kind === "newgame") {
    goLobby();
    return;
  } else if (kind === "review") {
    if (MOCK) {
      fetch("mock/review.json?v=2").then((r) => r.json()).then((r) => {
        review = r.review || r;
        overDismissed = true;
        logOpen = true;
        setLogTab("review");
        render();
      }).catch(() => {
        review = review || {decisions: [], counts: {}, accuracy: 0, missed: 0};
        render();
      });
      return;
    }
    if (!msg || !msg.result || !ws) return;
    overDismissed = true;
    logOpen = true;
    setLogTab("review");
    reviewLoading = true;
    reviewProgress = null;
    render();
    ws.send({t: "review"});
    return;
  } else if (kind === "showresult") {
    overDismissed = false;
  } else if (kind === "closereviewframe") {
    reviewFrame = null;
    render();
    return;
  } else if (kind === "hint") {
    if (!msg || !msg.prompt || hintLoading) return;
    if (MOCK) {
      const opts = (msg.prompt.options || []).slice(0, 3);
      const ps = [0.62, 0.25, 0.08];
      hint = {t: "hint", prompt: msg.prompt.id,
        options: opts.map((o, i) => ({index: o.index, label: o.label, p: ps[i] || 0.05, est: null})),
        ms: 12.3};
      hintPrompt = msg.prompt.id;
      say("Hint 1, " + (opts[0] ? opts[0].label : "no option") + ", bot's confidence 62%.");
      render();
      return;
    }
    if (!ws || !practiceMode) return;
    hintLoading = true;
    render();
    ws.send({t: "hint"});
    return;
  } else if (kind === "whatif") {
    const k = el.getAttribute("data-k");
    const option = el.getAttribute("data-option");
    if (k == null || option == null) return;
    if (MOCK) {
      reviewFrame = null;
      render();
      return;
    }
    if (!ws) return;
    reviewFrame = null;
    events = [];
    revealQueue = [];
    revealCur = null;
    bannerText = null;
    review = null;
    reviewLoading = false;
    reviewProgress = null;
    overDismissed = false;
    hint = null;
    hintLoading = false;
    hintPrompt = null;
    inLobby = false;
    render();
    ws.send({t: "whatif", k: parseInt(k, 10), option: parseInt(option, 10)});
    return;
  } else if (kind === "puzzles") {
    puzzlesOpen = true;
    if (MOCK && puzzles === null) puzzles = mockPuzzles();
    if (!MOCK && puzzles === null && ws) ws.send({t: "puzzles"});
    render();
    return;
  } else if (kind === "closepuzzles") {
    puzzlesOpen = false;
    render();
    return;
  } else if (kind === "puzzlestart") {
    const id = el.getAttribute("data-id");
    if (!id) return;
    if (MOCK) { loadMock(1); puzzle = {id, goal: "Win this turn · opponent has 6,700 LP", n_plays: 4, status: "playing"}; puzzlesOpen = false; render(); return; }
    startPuzzle(id);
    return;
  } else if (kind === "puzzleretry") {
    const id = puzzle && puzzle.id;
    if (!id) return;
    if (MOCK) { render(); return; }
    hidePuzzleResult = false;
    inSolution = false;
    solutionFrames = null;
    liveEnd = null;
    startPuzzle(id);
    return;
  } else if (kind === "puzzlenext") {
    const nxt = nextPuzzleId();
    if (!nxt) return;
    if (MOCK) { puzzle = {id: nxt, goal: "Win this turn · opponent has 6,700 LP", n_plays: 4, status: "playing"}; render(); return; }
    hidePuzzleResult = false;
    inSolution = false;
    solutionFrames = null;
    liveEnd = null;
    startPuzzle(nxt);
    return;
  } else if (kind === "puzzlesback") {
    hidePuzzleResult = false;
    inSolution = false;
    solutionFrames = null;
    liveEnd = null;
    goLobby();
    puzzlesOpen = true;
    if (MOCK && puzzles === null) puzzles = mockPuzzles();
    render();
    return;
  } else if (kind === "puzzlesolution") {
    if (MOCK) { say("Showing the solution."); render(); return; }
    if (!ws || !puzzle || !puzzle.id) return;
    ws.send({t: "puzzle_solution", id: puzzle.id});
    return;
  } else if (kind === "exitsolution") {
    exitSolution();
    return;
  } else return;
  render();
}

// Puzzles: start one, step to the next, mock data for screenshots, solution replay.
function startPuzzle(id) {
  events = [];
  revealQueue = [];
  revealCur = null;
  bannerText = null;
  review = null;
  reviewLoading = false;
  reviewProgress = null;
  reviewFrame = null;
  overDismissed = false;
  hint = null;
  hintLoading = false;
  hintPrompt = null;
  whatIf = null;
  puzzlesOpen = false;
  inSolution = false;
  solutionFrames = null;
  solutionIdx = -1;
  liveEnd = null;
  hidePuzzleResult = false;
  gameMode = "fast";
  practiceMode = true;
  inLobby = false;
  resetCues();
  render();
  if (ws) ws.send({t: "puzzle", id});
}

function nextPuzzleId() {
  if (!puzzles || !puzzles.length) return puzzle ? puzzle.id : null;
  const ids = puzzles.map((p) => p.id);
  const i = ids.indexOf(puzzle && puzzle.id);
  if (i < 0) return ids[0];
  return ids[(i + 1) % ids.length];
}

function mockPuzzles() {
  return [
    {id: "p-easy", decks: ["Blue Eyes", "Branded"], turn: 2, difficulty: "easy", goal: "Win this turn · opponent has 6,700 LP", n_plays: 3},
    {id: "p-medium", decks: ["Mitsurugi", "Snake-Eye"], turn: 3, difficulty: "medium", goal: "Win this turn · opponent has 4,200 LP", n_plays: 5},
    {id: "p-hard", decks: ["Branded", "Blue Eyes"], turn: 4, difficulty: "hard", goal: "Win this turn · opponent has 7,900 LP", n_plays: 8},
  ];
}

function startSolution(frames) {
  if (!Array.isArray(frames) || !frames.length) return;
  liveEnd = msg;
  hidePuzzleResult = true;
  inSolution = true;
  solutionFrames = frames;
  solutionIdx = 0;
  // Play with the normal cue pacing: first board at once, the rest through the cue player.
  msg = frames[0];
  locked = false;
  sel = [];
  menuUid = null;
  render();
  say("Showing the solution.");
  for (let i = 1; i < frames.length; i++) {
    const f = frames[i];
    cue.enqueue(Array.isArray(f.cues) ? f.cues : [], f);
  }
}

function exitSolution() {
  inSolution = false;
  solutionFrames = null;
  solutionIdx = -1;
  hidePuzzleResult = false;
  if (liveEnd) msg = liveEnd;
  liveEnd = null;
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
      (b.closest(".optw") || b).style.display = b.textContent.toLowerCase().includes(q) ? "" : "none";
    }
  }
  if (t.matches(".decksearch")) {
    const q = t.value.toLowerCase();
    for (const b of t.closest(".pickpanel").querySelectorAll(".deckrow")) {
      b.style.display = (b.getAttribute("data-dname") || "").includes(q) ? "" : "none";
    }
  }
});

// ---- keyboard and focus (WCAG 2.1.1, 2.4.3, 2.4.7) ----
// Modal layers, topmost first. Focus moves into the top one when it opens, Tab stays inside,
// Escape closes it, and focus returns to the control that opened it.
const FOCUSABLE = 'button:not([disabled]), [href], input, textarea, select, summary, [tabindex]:not([tabindex="-1"])';
const visible = (el) => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
function topModal() {
  if (IS_REPLAY) return null;
  for (const sel of ["#kwpop", "#detail", "#menu", "#gamemenu", "#puzzlesheet", "#sheet.pilesheet", "#reviewframe", "#logdrawer", "#puzzleresult[role=alertdialog]", "#gameover[role=alertdialog]"]) {
    const el = document.querySelector(sel);
    if (el && visible(el)) return el;
  }
  return null;
}
function focusables(root) {
  return [...root.querySelectorAll(FOCUSABLE)].filter((el) => visible(el) && !el.closest("[aria-hidden=true]"));
}
// Where focus was before a modal opened: the element, plus a way to find its re-rendered twin.
let modalTop = null;
let opener = null;
function describe(el) {
  if (!el || el === document.body) return null;
  for (const a of ["data-uid", "data-act", "data-pile", "data-opt", "data-pickopen", "data-recap"]) {
    const v = el.getAttribute(a);
    if (v != null) return {el, sel: "[" + a + '="' + CSS.escape(v) + '"]'};
  }
  return {el, sel: el.id ? "#" + CSS.escape(el.id) : null};
}
function restoreFocus() {
  const o = opener;
  opener = null;
  if (!o) return;
  let el = o.el && o.el.isConnected && visible(o.el) ? o.el : null;
  if (!el && o.sel) el = [...document.querySelectorAll(o.sel)].find(visible) || null;
  if (el) el.focus({preventScroll: true});
}
function manageFocus() {
  const top = topModal();
  if (top === modalTop) {
    if (top && !top.contains(document.activeElement)) (focusables(top)[0] || top).focus({preventScroll: true});
    return;
  }
  const prev = modalTop;
  modalTop = top;
  if (top) {
    if (!prev || !prev.isConnected) opener = describe(document.activeElement) || opener;
    (focusables(top)[0] || top).focus({preventScroll: true});
  } else {
    restoreFocus();
  }
  if (!top && document.activeElement === document.body) {
    // A new prompt sheet after the old focus target went away: start at its first choice.
    const first = document.querySelector("#sheet:not(.pilesheet) [data-opt]");
    if (first && visible(first) && !locked) first.focus({preventScroll: true});
  }
}
function closeTop() {
  if (kwPop) { closeKw(); return true; }
  if (menuUid) { closeState("menu"); render(); closeOv("menu"); return true; }
  if (detail) { closeState("detail"); render(); closeOv("detail"); return true; }
  if (reviewFrame) { reviewFrame = null; render(); return true; }
  if (gameMenu) { closeState("gamemenu"); render(); closeOv("gamemenu"); return true; }
  if (puzzlesOpen) { puzzlesOpen = false; render(); return true; }
  if (inSolution) { exitSolution(); return true; }
  if (pileOpen) { closeState("pile"); render(); closeOv("pile"); return true; }
  if (logOpen) { closeState("log"); render(); closeOv("log"); return true; }
  if (revealCur) { dismissReveal(); return true; }
  if (pickerOpen) { const side = pickerOpen; pickerOpen = null; renderLobby(); const b = app.querySelector('[data-pickopen="' + side + '"]'); if (b) b.focus(); return true; }
  if (recap && document.activeElement && document.activeElement.closest(".recap")) { recap = null; render(); return true; }
  return false;
}
document.addEventListener("keydown", (ev) => {
  const t = ev.target;
  const typing = t.matches && t.matches("input, textarea");
  if (ev.key === "Escape") {
    if (closeTop()) ev.preventDefault();
    return;
  }
  // Practice undo: "u" when no dialog is open.
  if ((ev.key === "u" || ev.key === "U") && !typing && !ev.ctrlKey && !ev.metaKey && !ev.altKey &&
      !topModal() && msg && msg.t === "state" && msg.prompt && !msg.result && practiceMode &&
      canUndo && !locked && !msg.thinking) {
    ev.preventDefault();
    doUndo();
    return;
  }
  // "Pause after each bot action": Enter continues, like a tap.
  if (cue.waiting() && (ev.key === "Enter" || ev.key === " ") && !typing && !topModal()) {
    ev.preventDefault();
    cue.resume();
    return;
  }
  if (ev.key === "Tab") {
    const m = topModal();
    if (!m) return;
    const f = focusables(m);
    if (!f.length) { ev.preventDefault(); m.focus(); return; }
    const i = f.indexOf(document.activeElement);
    if (ev.shiftKey && (i <= 0)) { ev.preventDefault(); f[f.length - 1].focus(); }
    else if (!ev.shiftKey && (i < 0 || i === f.length - 1)) { ev.preventDefault(); f[0].focus(); }
    return;
  }
  if ((ev.key === "ArrowDown" || ev.key === "ArrowUp") && t.closest && t.closest("#menu")) {
    const items = focusables(document.getElementById("menu"));
    const i = items.indexOf(t);
    const n = items[(i + (ev.key === "ArrowDown" ? 1 : -1) + items.length) % items.length];
    if (n) { ev.preventDefault(); n.focus(); }
    return;
  }
  // Duel-log tabs: arrow keys move between Log | Stats | Cards (ARIA tabs pattern).
  if ((ev.key === "ArrowRight" || ev.key === "ArrowLeft" || ev.key === "Home" || ev.key === "End") &&
      t.getAttribute && t.getAttribute("role") === "tab" && t.closest && t.closest(".dtabs")) {
    const tabs = [...t.closest(".dtabs").querySelectorAll('[role="tab"]')];
    let i = tabs.indexOf(t);
    if (ev.key === "Home") i = 0;
    else if (ev.key === "End") i = tabs.length - 1;
    else i = (i + (ev.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    ev.preventDefault();
    const next = tabs[i];
    if (next) {
      setLogTab(next.getAttribute("data-logtab"));
      render();
      const id = next.id;
      const el = (id && document.getElementById(id)) ||
        [...document.querySelectorAll('[data-logtab="' + next.getAttribute("data-logtab") + '"]')].find(visible);
      if (el) el.focus({preventScroll: true});
    }
    return;
  }
  // Non-native controls (cards, log lines): Enter and Space activate, as on a button.
  if ((ev.key === "Enter" || ev.key === " ") && !typing && t.getAttribute && t.getAttribute("role") === "button" &&
      !t.matches("button, summary, a")) {
    ev.preventDefault();
    t.click();
  }
});
if (reducedMotion()) document.documentElement.dataset.motion = "reduced";

// ---- cues ----
const cue = createCuePlayer({
  layer: ovCues,
  getView: () => (msg && msg.view) || null,
  apply: (m) => applyState(m),
  mock: MOCK,
  names: () => (msg && msg.names) || null,
  speed: () => (cueSpeed === "2" ? 2 : 1),
  pause: () => cuePause && !IS_REPLAY,
  announce: (t) => say(t),
  reduced: () => reducedMotion() && !IS_REPLAY,
  ruleNotes: () => ruleNotesOn,
  textFor: (code) => {
    const t = cardText(code);
    return t && t.text ? t.text : null;
  },
});

function resetCues() {
  cue.reset();
  recap = null;
  recapAcc = null;
  liveChain = chainTracker();
  chainNeg = new Set();
  lastCueN = -1;
  lastCueTurn = 0;
}

// New cues of a state message (peek/think states may repeat them: dedupe by trace index).
function freshCues(m) {
  if (!m || !Array.isArray(m.cues)) return [];
  if (m.view && m.view.turn < lastCueTurn) lastCueN = -1;   // a new duel restarted the trace
  if (m.view) lastCueTurn = m.view.turn;
  const out = m.cues.filter((c) => c && (typeof c.n !== "number" || c.n > lastCueN));
  for (const c of out) if (typeof c.n === "number" && c.n > lastCueN) lastCueN = c.n;
  return out;
}

// Collect the opponent's turn for the recap; attach it to the message where your turn starts.
// Also tracks the chain, so negations know their cause (m._negs, applied with the board).
let liveChain = chainTracker();
let chainNeg = new Set();   // chain links shown negated in the chain strip until the chain ends
function collectRecap(m, cues) {
  const oppName = (m.names && m.names[1]) || "ygo-zero";
  m._negs = m._negs || [];
  for (const c of cues) {
    const neg = liveChain.feed(c);
    if (neg) m._negs.push(neg);
    if (c.t === "chain_end") m._negs.push({end: true});
    if (c.t === "turn") {
      if (c.player === "opp") recapAcc = {turn: c.turn, who: oppName, items: []};
      else if (recapAcc) {
        if (recapAcc.items.length) m._recap = recapAcc;
        recapAcc = null;
      }
      continue;
    }
    if (!recapAcc) continue;
    const it = recapItem(c, neg);
    if (it) {
      recapAcc.items.push(it);
      if (recapAcc.items.length > 40) recapAcc.items.shift();
    }
  }
}

function receiveState(m) {
  const cues = freshCues(m);
  for (const c of cues) if (c.t === "activate" && c.card && c.card.code) cardText(c.card.code);   // warm the text cache
  collectRecap(m, cues);
  if (cueSpeed === "instant" && cues.length) sayInstant(cues);
  cue.enqueue(cueSpeed === "instant" ? [] : cues, m);
}

// Instant pacing plays no scenes; screen readers still hear what happened (the last few lines).
function sayInstant(cues) {
  const lines = planSteps(cues, (msg && msg.view && msg.view.chain) || []).filter((s) => s.dur > 0)
    .map((s) => stepText(s, msg && msg.names)).filter(Boolean);
  if (lines.length > 6) say((lines.length - 5) + " earlier moves. " + lines.slice(-5).join(". "));
  else if (lines.length) say(lines.join(". "));
}

// New prompts and the result, in words.
function announceState(prev, m) {
  if (IS_REPLAY) return;
  const p = m.prompt;
  if (p && (!prev || !prev.prompt || prev.prompt.id !== p.id)) {
    if (p.kind === "idle" || p.kind === "battle") {
      const n = Object.keys(actionableMap(p)).length;
      say((p.kind === "battle" ? "Battle Phase, your move. " : "Your move. ") + n + (n === 1 ? " card can act." : " cards can act."));
    } else {
      const pick = p.kind === "card" ? (p.min === p.max ? " Pick " + p.min + "." : " Pick " + p.min + " to " + p.max + ".") : "";
      const t = p.text || "Choose";
      say(t + (/[.?!…]$/.test(t) ? "" : ".") + pick);
    }
  }
  const pz = m.puzzle || null;
  const ppz = prev ? prev.puzzle : null;
  if (pz && pz.status && (!ppz || ppz.status !== pz.status)) {
    if (pz.status === "solved") say("Puzzle solved.");
    else if (pz.status === "failed") say("Puzzle not solved.");
  }
  if (m.result && !(prev && prev.result)) {
    const w = m.result.winner;
    say("Duel over. " + (w === "you" ? "Victory." : w === "bot" ? "Defeat." : String(w) + " wins."));
  }
  // Win-chance swings of 10 points or more, in words (the bar is never colour alone).
  const wp = m.winchance && typeof m.winchance.p === "number" ? m.winchance.p : null;
  const pp = prev && prev.winchance && typeof prev.winchance.p === "number" ? prev.winchance.p : null;
  if (winChance && wp != null && pp != null && Math.abs(wp - pp) >= 0.10) {
    say("Win chance " + Math.round(wp * 100) + "%.");
  }
}

// Negated chain links stay marked: in the chain strip until the chain ends, and on
// the log line of the negated activation (with what negated it).
function markNegations(m) {
  for (const ng of m._negs || []) {
    if (ng.end) { chainNeg = new Set(); continue; }
    chainNeg.add(ng.link);
    const code = ng.card && ng.card.code;
    if (!code) continue;
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (e.kind === "activate" && e.card && e.card.code === code && !e.neg) {
        e.neg = ng.by && ng.by.name ? "by " + ng.by.name : (ng.disabled ? "effect" : "activation");
        break;
      }
    }
  }
  if (m.view && (!m.view.chain || !m.view.chain.length)) chainNeg = new Set();
}

function applyState(m) {
  if (!m) return;
  if (msg && msg.t === "state" && msg.prompt && (!m.prompt || m.prompt.id !== msg.prompt.id)) {
    sel = [];
    expanded = null;
    hint = null;
    hintLoading = false;
    hintPrompt = null;
  }
  if (!m.prompt) {
    sel = [];
    if (!m.result) { hint = null; hintLoading = false; hintPrompt = null; }
  }
  practiceMode = !!m.practice;
  canUndo = !!m.can_undo;
  whatIf = m.what_if || null;
  puzzle = m.puzzle || null;
  if (puzzle && puzzle.status === "solved") markSolved(puzzle.id);
  if (inSolution && !(puzzle && puzzle.status)) { inSolution = false; solutionFrames = null; solutionIdx = -1; }
  if (undoPending) {
    say("Undid your last decision");
    undoPending = false;
  }
  // A cue-aware server replaces the old reveal/banner notices.
  addEvents(m.log, false, !("cues" in m));
  markNegations(m);
  announceState(msg && msg.t === "state" ? msg : null, m);
  msg = m;
  locked = false;
  menuUid = null;
  if (m._recap && cueRecap && !IS_REPLAY && !m.result) recap = m._recap;
  render();
}

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
  // Mock screenshots of rules-in-context need real card text (the live path
  // fetches /card/{code}; mocks preload a small map instead).
  try {
    const ct = await (await fetch("mock/card-text.json?v=2")).json();
    for (const [code, v] of Object.entries(ct)) texts.set(Number(code), v);
  } catch (e) { /* no mock card text */ }
  if (msg.t === "state") addEvents(msg.log, true);
  resetCues();
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
  review = null;
  reviewLoading = false;
  reviewProgress = null;
  reviewFilter = "all";
  reviewFrame = null;
  overDismissed = false;
  practiceMode = false;
  hint = null;
  hintLoading = false;
  hintPrompt = null;
  whatIf = null;
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
  if (qs.get("play") === "practice" && msg.t === "lobby") lobbyPick.play = "practice";
  if (qs.get("gamemenu") && msg.t === "state") gameMenu = true;
  // Screenshot hooks for the duel-log sheet: ?log=1 opens it, ?logtab=log|stats|cards|review picks the tab.
  if (msg.t === "state") {
    const lt = qs.get("logtab");
    if (lt === "log" || lt === "stats" || lt === "cards" || lt === "review") setLogTab(lt);
    if (qs.get("log") === "1") logOpen = true;
    // Screenshot hook for the win-chance switch: ?winchance=1 turns it on.
    if (qs.get("winchance") === "1") winChance = true;
    // Rules-in-context screenshots: ?detail=<uid> / ?detailcode=<code> open the
    // card detail, ?kw=<id> opens its rule popover, ?canundo=1 shows Undo ready.
    if (qs.get("detail")) {
      const c = findCard(qs.get("detail"));
      if (c && !isHidden(c)) detail = c;
    }
    if (qs.get("detailcode")) {
      const code = parseInt(qs.get("detailcode"), 10);
      const ct = texts.get(code);
      detail = {uid: "mock:" + code, code, name: (ct && ct.name) || ("#" + code), faceup: true, pos: null,
        atk: 0, def: 0, level: 0, rank: 0, link: 0, counters: 0, overlays: []};
    }
    if (qs.get("kw")) {
      const g = glossaryEntry(qs.get("kw"));
      if (g) kwPop = {id: g.id, term: g.term, long: g.long};
    }
    if (qs.get("canundo") === "1") { practiceMode = true; canUndo = true; }
    // Screenshot hooks for the review tab: ?review=1 loads mock/review.json,
    // ?reviewfilter=mistakes|missed picks the filter, ?reviewframe=1 opens the decision board.
    if (qs.get("review") === "1") {
      try {
        const r = await fetch("mock/review.json?v=2");
        const j = await r.json();
        review = j.review || j;
        reviewLoading = false;
        const rf = qs.get("reviewfilter");
        if (rf === "mistakes" || rf === "missed") reviewFilter = rf;
        if (qs.get("reviewframe") === "1") {
          try {
            const fr = await fetch("mock/review-frame.json?v=2");
            reviewFrame = await fr.json();
          } catch (e) { /* leave closed */ }
        }
        if (qs.get("log") === "1") {
          // The real flow: tapping "Review game" dismisses the game-over card
          // and opens the sheet on the Review tab.
          overDismissed = true;
          logOpen = true;
          setLogTab("review");
        }
      } catch (e) { /* leave empty */ }
    }
  }
  // Screenshot hooks for practice hints + the what-if banner (mock state only).
  if (msg.t === "state") {
    if (qs.get("whatif") === "1") {
      practiceMode = true;
      whatIf = {source: "mock", k: 1, option: 1, turn: msg.view ? msg.view.turn : 2};
    }
    if (qs.get("hint") === "1" && msg.prompt) practiceMode = true;
    // Puzzles screenshots: ?puzzlebanner=1 shows the in-puzzle banner,
    // ?puzzlesolved=1 / ?puzzlefailed=1 show the result dialogs.
    if (qs.get("puzzlebanner") === "1") {
      practiceMode = true;
      puzzle = {id: "p-mock", goal: "Win this turn · opponent has 6,700 LP", n_plays: 4, status: "playing"};
    }
    if (qs.get("puzzlesolved") === "1") {
      practiceMode = true;
      puzzle = {id: "p-mock", goal: "Win this turn · opponent has 6,700 LP", n_plays: 4, status: "solved"};
      msg = Object.assign({}, msg, {result: {winner: "you", reason: "win"}, prompt: null, puzzle});
    }
    if (qs.get("puzzlefailed") === "1") {
      practiceMode = true;
      puzzle = {id: "p-mock", goal: "Win this turn · opponent has 6,700 LP", n_plays: 4, status: "failed"};
      msg = Object.assign({}, msg, {result: {winner: "bot", reason: "puzzle failed"}, prompt: null, puzzle});
    }
  }
  if (msg.t === "lobby" && qs.get("puzzles") === "1") {
    puzzles = mockPuzzles();
    // Mark one as solved to show the quiet Solved mark (text, not colour alone).
    try {
      if (!solvedIds["p-easy"]) { solvedIds["p-easy"] = true; }
    } catch (e) { /* ignore */ }
    puzzlesOpen = true;
  }
  render();
  if (msg.t === "state" && !(Array.isArray(msg.cues) && msg.cues.length)) announceState(null, msg);
  if (msg.t === "state" && Array.isArray(msg.cues) && msg.cues.length) mockCues();
  if (qs.get("hint") === "1" && msg.t === "state" && msg.prompt) {
    const opts = (msg.prompt.options || []).slice(0, 3);
    const ps = [0.62, 0.25, 0.08];
    hint = {t: "hint", prompt: msg.prompt.id,
      options: opts.map((o, i) => ({index: o.index, label: o.label, p: ps[i] || 0.05, est: null})), ms: 12.3};
    hintPrompt = msg.prompt.id;
    render();
  }
  if (qs.get("reveal") && msg.t === "state") {
    const c = findCard(qs.get("reveal"));
    if (c && !isHidden(c)) {
      revealQueue = [];
      revealCur = {card: c, actor: "bot", kind: "activate", n: -1, text: "Activates " + (c.name || "")};
      render();
    }
  }
}

// Mock cue playback. The mock's `prev_view` is the board before the cues
// (default: the same view). ?cue=K&t=F freezes visible scene K at fraction F
// (screenshots); ?recap=1 applies the whole list at once and opens the recap;
// otherwise the cues play in real time and the mock's state applies after.
function mockCues() {
  const real = msg;
  const base = Object.assign({}, real, {view: real.prev_view || real.view, prompt: null, result: null, cues: undefined});
  const cues = real.cues;
  if (qs.get("recap")) {
    collectRecap(real, cues);
    applyState(real);
    return;
  }
  msg = base;
  render();
  if (qs.get("cue") != null) {
    const k = parseInt(qs.get("cue"), 10) || 0;
    const f = parseFloat(qs.get("t") || "0.5");
    cue.freeze(cues, base.view.chain, k, isFinite(f) ? f : 0.5, "mock" + mockIdx);
    window.cueFrozen = true;
    return;
  }
  collectRecap(real, cues);
  cue.enqueue(cueSpeed === "instant" ? [] : cues, real);
}

document.addEventListener("keydown", (ev) => {
  if (!MOCK || ev.target.matches("input, textarea")) return;
  if (ev.key === "ArrowRight") loadMock(mockIdx + 1);
  if (ev.key === "ArrowLeft") loadMock(mockIdx - 1);
});

// Wait for the card images visible in the current frame (bounded wait:
// a missing image must never stall the video).
function waitReplayImages(timeoutMs) {
  const imgs = [...document.querySelectorAll("#app img")].filter((im) => !im.complete);
  if (!imgs.length) return Promise.resolve();
  return Promise.race([
    Promise.all(imgs.map((im) => new Promise((res) => {
      im.addEventListener("load", res, {once: true});
      im.addEventListener("error", res, {once: true});
    }))),
    new Promise((res) => setTimeout(res, timeoutMs || 2500)),
  ]);
}

async function loadReplay() {
  let url = REPLAY_URL;
  if (TOKEN && !url.includes("t=")) url += (url.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN);
  const r = await fetch(url);
  if (!r.ok) {
    msg = {t: "error", text: "replay load failed: " + REPLAY_URL};
    render();
    return;
  }
  const j = await r.json();
  replayFrames = Array.isArray(j) ? j : (j.frames || []);
  window.frameCount = replayFrames.length;
  const baseChain = (i) => (i > 0 && replayFrames[i - 1].view && replayFrames[i - 1].view.chain) || [];
  const frameCues = (i) => (i > 0 && Array.isArray(replayFrames[i].cues) ? replayFrames[i].cues : []);
  // Visible cue scenes of frame i (ms at 1x): game_video.py samples each one.
  window.cuePlan = (i) => cue.plan(frameCues(i), baseChain(i));
  let frozenAt = -1;
  // Board j with the log up to `upto` (the cues of frame upto play over board upto-1).
  const setBoard = (j, upto) => {
    replayIdx = upto;
    events = [];
    for (let k = 0; k <= j; k++) {
      for (const e of replayFrames[k].log || []) {
        if (!events.some((x) => x.n === e.n)) events.push(e);
      }
    }
    events.sort((a, b) => (a.n || 0) - (b.n || 0));
    msg = replayFrames[j];
    locked = false;
    sel = [];
    expanded = null;
    menuUid = null;
    detail = null;
    gameMenu = false;
    pileOpen = null;
    revealCur = null;
    revealQueue = [];
    bannerText = null;
    render();
  };
  // showFrame(i): play frame i's cues over board i-1 in real time, then show board i.
  // showFrame(i, {cue: k, t: f}): board i-1 with visible cue scene k frozen at fraction f.
  // showFrame(i, {cues: false}): board i at once.
  window.showFrame = async (i, o) => {
    i = Math.max(0, Math.min(replayFrames.length - 1, i));
    o = o || {};
    const cues = frameCues(i);
    if (o.cue != null) {
      if (frozenAt !== i) {
        cue.reset();
        setBoard(Math.max(0, i - 1), i);
        frozenAt = i;
      }
      cue.freeze(cues, baseChain(i), o.cue, o.t || 0, "frame" + i);
      await waitReplayImages(2500);
      await cue.waitImages(2500);
      return;
    }
    frozenAt = -1;
    cue.reset();
    if (o.cues !== false && cues.length && replayIdx === i - 1 && cue.plan(cues, baseChain(i)).length) {
      setBoard(i - 1, i);
      await cue.playOnce(cues, baseChain(i));
    }
    setBoard(i, i);
    await waitReplayImages(2500);
  };
  if (replayFrames.length) await window.showFrame(0);
}

async function boot() {
  if (IS_REPLAY) {
    await loadReplay();
    return;
  }
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
    onOpen() { banner.hidden = true; sendSettings(); },
    onDrop() {
      banner.hidden = false;
      banner.textContent = "Reconnecting…";
    },
    onMessage(m) {
      if (m.t === "lobby") {
        lobby = m;
        if (!msg || msg.t === "lobby") msg = m;
        if (puzzles === null && !MOCK && ws) ws.send({t: "puzzles"});
      } else if (m.t === "review_progress") {
        reviewLoading = true;
        reviewProgress = {done: m.done || 0, total: m.total || 0};
        say("Reviewing… " + reviewProgress.done + " of " + reviewProgress.total);
        render();
        return;
      } else if (m.t === "review") {
        review = m;
        reviewLoading = false;
        reviewProgress = null;
        overDismissed = true;
        logOpen = true;
        setLogTab("review");
        const missed = m.missed || 0;
        say("Review ready. " + ((m.decisions || []).length) + " decisions" +
          (missed ? ", " + missed + " missed responses." : "."));
        render();
        return;
      } else if (m.t === "review_frame") {
        reviewFrame = m;
        render();
        return;
      } else if (m.t === "hint") {
        hint = m;
        hintLoading = false;
        hintPrompt = (msg && msg.prompt) ? msg.prompt.id : (m.prompt != null ? m.prompt : null);
        const first = (m.options || [])[0];
        if (first) {
          const pct = Math.round((first.p || 0) * 100);
          say("Hint 1, " + (first.label || "option") + ", bot's confidence " + pct + "%.");
        }
        render();
        return;
      } else if (m.t === "puzzles") {
        puzzles = Array.isArray(m.items) ? m.items : [];
        render();
        return;
      } else if (m.t === "puzzle_solution") {
        startSolution(m.frames || []);
        return;
      } else if (m.t === "state") {
        if (inLobby) return;   // Back to lobby: ignore states until the next new_game
        receiveState(m);
        return;
        banner.hidden = false;
        banner.textContent = m.text || "error";
        locked = false;
        setTimeout(() => { banner.hidden = true; }, 4000);
      } else {
        if (m.t === "error") undoPending = false;   // the undo was refused: don't announce it
        msg = m;
      }
      render();
    },
  });
}

boot();
