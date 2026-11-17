import {connectWS} from "./ws.js";
import {deckBuilder} from "./features/deckbuilder/index.js";
import {askFeature} from "./features/ask/index.js";
import {gamesFeature} from "./features/games/index.js";
import {describeModel} from "./features/describe/index.js";
import {onStill} from "./lib/stills.js";
import {barHTML, sideHTML, midHTML, phaseBtnsHTML, chainHTML, winchanceHTML} from "./render/board.js";
import {handHTML, sheetHTML, pileSheetHTML, infoHTML, actionableMap, statsLine, pileTitle} from "./render/hand.js";
import {logHTML, negTag} from "./render/log.js";
import {LOG_TABS, tabStripHTML, tabPanelOpen, statsPanelHTML, cardsPanelHTML, reviewPanelHTML} from "./render/stats.js";
import {artImg, esc, isHidden, isMonster, phHTML, setCardArt} from "./render/card.js";
import {morph} from "./render/morph.js";
import {deckSheetHTML} from "./render/decks.js";
import {peopleHTML, roomHTML, roomOverHTML, HANDS, HAND_ICON} from "./render/room.js";
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
  "mock/21-lobby-models.json",
  "mock/75-context-actions.json",
  "mock/76-place-cancel.json",
  "mock/77-select-cancel.json",
  "mock/78-art-running.json",
  "mock/79-art-wifi.json",
  "mock/80-art-done.json",
  "mock/81-settings.json",
  "mock/82-settings-cards.json",
  "mock/83-settings-ingame.json",
  "mock/84-card-actions.json",
  "mock/85-card-actions-pass.json",
  "mock/86-rps-pick.json",
  "mock/87-rps-result.json",
  "mock/88-prompt-subject-info.json",
  "mock/89-paused.json",
  "mock/90-cue-then-prompt.json",
  "mock/91-reveal-over-sheet.json",
  "mock/92-lobby-loading-bot.json",
];

const qs = new URLSearchParams(location.search);
const MOCK = qs.has("mock");
let mockIdx = Math.max(0, parseInt(qs.get("i") || "0", 10) || 0);
const TOKEN = qs.get("t") || "";
// Spectator replay (game_video.py): ?replay=<frames-json-url>. Fetches the
// frames once, renders each through the mock/live render path, no inputs.
const REPLAY_URL = qs.get("replay");
const IS_REPLAY = REPLAY_URL != null && REPLAY_URL !== "";
// The player inside Watch, exports and the chat's stills: the page around it has the controls.
const EMBED = REPLAY_URL === "embed";
// It only shows the board: focusing anything in it would pull the keyboard (Escape, Back) away from the page around it.
if (EMBED) HTMLElement.prototype.focus = function () {};
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
const readside = document.getElementById("readside");
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
const ovSettings = ov("ov-settings");

let msg = null;          // latest server/mock message
let locked = false;      // input disabled until the next state
let sel = [];            // selected option indices (multi-select)
let expanded = null;     // option index whose effect text is expanded
let menuUid = null;      // uid selected in the context bar
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
// Wide screens read a tapped card in the card reader (#readside from 1100 px, the side column's card from 900 px)
// instead of the full-screen detail, which would cover the board (docs/ui/mocks/table/, user 2026-11-16: "as shown").
const WIDE = matchMedia("(min-width: 900px)");
let readCard = null;
let lobby = null;
let lobbyPick = {you: null, bot: null, first: "rps", mode: "product", play: "duel", model: null};
// The bot duel's rock-paper-scissors, played here before the new_game message:
// {phase: "pick" | "result", mine, theirs} (hands use EDOPro's 1 scissors,
// 2 rock, 3 paper, as the room RPS does). null when no game is being set up.
let rpsGame = null;
let rpsTestHand = null;   // mock-mode stub for the random bot hand (acceptance checks)
if (MOCK) window.__rpsSetHand = (h) => { rpsTestHand = h == null ? null : Number(h); };
let pickerOpen = null;   // "model" | null (the deck pickers open the deck browser sheet instead)
let deckBrowser = null;  // {side, q, f: {group: [values]}, sort, view: deckId|null, top} while the deck sheet is open
let deckIndex;           // /deck_index.json: undefined = not fetched, null = loading
let optsOpen = false;     // lobby Options section expanded
let pasteOpen = false;    // lobby .ydk paste box shown (from the "Your deck" picker)
let modelNotice = null;  // status line under the model field (auto-switch), cleared on the next lobby change
let packBusy = false;    // a card-pack update/rollback request is running
let gameModel = null;    // running game's model label (starting screen + menu fallback)
let savedModel = null;
try {
  savedModel = localStorage.getItem("ygo-model");
} catch (e) { /* private mode: no remembered model */ }
let pileOpen = null;     // "gy:you" | "banished:bot" | "extra:you" ...
let ws = null;
let lpShown = {you: null, bot: null};
let events = [];         // accumulated log (the server sends only new events; dedupe by n)
const texts = new Map(); // code -> {text, type_str} | "pending" | "failed"

// In-game menu + surrender confirm.
let gameMenu = false;
// App-wide Settings screen (lobby gear + "All settings…" in the game menu).
let settingsOpen = false;
let surrenderArm = false;
let surrenderTimer = 0;
let chainMode = "auto";
// The chain stack while cues play (cues.js trackChain): null = draw it from the board state.
let cueChain = null;
let cueSolving = 0;
try {
  const saved = localStorage.getItem("ygo-chain");
  if (saved === "auto" || saved === "all") chainMode = saved;
} catch (e) { /* private mode: stay on auto */ }
let inLobby = false;     // after Back to lobby: ignore states until new_game
let gameMode = null;     // the running game's bot strength (for the win-chance note)
// New game sent, first state not yet applied: the start button is disabled and
// the lobby shows the setting-up status instead of the form (no second new_game).
let starting = false;
// App startup (the Android gate serves this client before the bot is ready):
// the lobby is up with a calm status line and Start disabled until it is.
let startupPending = false;
let startupFailed = null;

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

// Duel pause (bot duels): while paused, incoming state frames wait in
// pauseQueue and the cue player freezes; Resume replays both in order. The
// stage below holds every narration/decision behind a pause, and every prompt
// that arrives during narration waits its turn (no per-surface timers).
let paused = false;
let pauseQueue = [];

// Card art (web/render/card.js): on by default (user 2026-11-12), so a new
// player sees faces without digging into Settings. An explicit choice always
// wins. The server may send a default in the lobby message (--card-art); it
// applies only when the user never chose (no ygo-card-art key).
let cardArt = true;
let cardArtChosen = false;
try {
  const saved = localStorage.getItem("ygo-card-art");
  cardArtChosen = saved !== null;
  cardArt = saved === null ? true : saved === "1";
} catch (e) { /* private mode: stay on */ }
setCardArt(cardArt);
// Background art sync (Rust server only): null until its first `art_sync` frame, so the
// Python server's client hides the whole section. `artDeleteArm` is the two-tap delete.
let artSync = null;
let artDeleteArm = false;
function saveSetting(k, v) {
  try { localStorage.setItem(k, v); } catch (e) { /* ignore */ }
}

// Appearance (user 2026-11-15: "a toggle in the settings between light and dark with dark being the default", then
// "also add the option of follow the phone"): "dark" | "light" | "system". index.html applies the saved choice before
// the first paint; this keeps it current. On Android the web view's prefers-color-scheme does not follow the phone (it reported light in dark mode), so
// "system" asks the DecklightTheme bridge, which also recolours the status and navigation bars and the next launch's
// splash, and calls __decklightUiMode when the phone switches.
const themeBridge = window.DecklightTheme || null;
let themeMode = "dark";
{
  let t = null;
  try { t = localStorage.getItem("ygo-theme"); } catch (e) { /* private mode: the app's saved choice below */ }
  if (!t && themeBridge) { try { t = themeBridge.mode(); } catch (e) { /* older app build */ } }
  if (t === "light" || t === "system") themeMode = t;
}
const themeMedia = window.matchMedia ? matchMedia("(prefers-color-scheme: light)") : null;
function systemTheme() {
  if (themeBridge) { try { return themeBridge.system() === "light" ? "light" : "dark"; } catch (e) { /* fall through */ } }
  return themeMedia && themeMedia.matches ? "light" : "dark";
}
function applyTheme() {
  const t = themeMode === "system" ? systemTheme() : themeMode;
  const root = document.documentElement;
  if (t === "light") root.dataset.theme = "light"; else delete root.dataset.theme;
  root.style.background = getComputedStyle(root).getPropertyValue("--bg").trim();
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = root.style.background;
  const cs = document.querySelector('meta[name="color-scheme"]');
  if (cs) cs.content = t;
  if (themeBridge) { try { themeBridge.apply(themeMode, t); } catch (e) { /* older app build */ } }
}
window.__decklightUiMode = () => { if (themeMode === "system") applyTheme(); };
if (themeMedia && themeMedia.addEventListener) themeMedia.addEventListener("change", window.__decklightUiMode);
applyTheme();

// Pause vs the bot: hold incoming frames and freeze the cue player. Prompts are
// not shown (or answerable) until Resume; card details, the log and the menu stay live.
function setPaused(on) {
  on = !!on;
  if (on === paused) { render(); return; }
  if (on && (!msg || msg.t !== "state" || msg.room || IS_REPLAY || inSolution || msg.result)) return;
  paused = on;
  stage.hold(paused);
  if (paused) {
    clearTimeout(revealTimer);   // Pause holds the narration's clock too
    say("Paused. Tap Resume to continue.");
  } else {
    say("Resumed.");
    const q = pauseQueue;
    pauseQueue = [];
    for (const m of q) {
      if (m.t === "state") handleState(m);
      else msg = m;
    }
    if (cue.waiting()) cue.resume();
    cue.kick();
  }
  render();
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
let puzzleQ = "";           // the Puzzles tab search text
const puzzleF = {diff: new Set(), prog: new Set(), len: new Set(), deck: new Set()};  // selected filter chips
let puzzleAllDecks = false; // the Decks chips show every deck, not the top few
let puzzleMore = false;     // "More filters" (Length, Decks) is open
let puzzle = null;         // the running puzzle {"id","goal","n_plays","status"} or null
let solvedIds = {};        // id -> true (localStorage ygo-puzzles-solved)
// Lobby tabs (user 2026-11-03): Multiplayer ("people", rooms) | Bot ("play") | Puzzles. The last one used is remembered;
// a first visit opens Bot, the one that works with nobody else around. Since 2026-11-14 (user: "Yes let's go with bottom
// bar") they sit in a bottom bar, Play · Puzzles · Decks · Ask, and Play holds Multiplayer | Bot as sub-tabs.
const LOBBY_TABS = ["people", "play", "games", "puzzles", "decks", "ask"];
const BAR_TABS = [["play", "Play"], ["puzzles", "Puzzles"], ["decks", "Decks"], ["ask", "Ask"]];
const barIcons = {};   // web/assets/icons/<tab>.svg, inlined so the selected tab can colour its node
Promise.all(BAR_TABS.map(([k]) => fetch("assets/icons/" + k + ".svg").then((r) => (r.ok ? r.text() : "")).then((t) => {
  barIcons[k] = t.replace(/<\?xml[^>]*>/, "").replace(/<!--[\s\S]*?-->/g, "").replace(/<title>[^<]*<\/title>/, "").replace("<svg", '<svg focusable="false"');
}).catch(() => {}))).then(() => { if (typeof render === "function" && !inGameView()) render(); });
let lobbyTab = "play";
try { const t = localStorage.getItem("ygo-lobbytab"); if (LOBBY_TABS.includes(t)) lobbyTab = t; } catch (e) { /* ignore */ }
if (LOBBY_TABS.includes(qs.get("ltab"))) lobbyTab = qs.get("ltab");   // screenshot hook
let playSub = lobbyTab === "people" || lobbyTab === "games" ? lobbyTab : "play";   // the Play tab's last sub-tab
// Deck builder (docs/design/deck-builder.md): a self-contained feature; app.js only routes tabs, actions, messages, the
// stage and the back button to it.
const db = deckBuilder({
  send: (m) => { if (ws) ws.send(m); },
  render: () => render(),
  say: (t) => say(t),
  mock: MOCK,
  openSettings: () => openSettingsAt("set-describe"),
});
// Ask (docs/plans/ask-tab.md): chats with the user's own AI provider over the ask_* messages.
const ask = askFeature({send: (m) => { if (ws) ws.send(m); }, render: () => render(), say: (t) => say(t), mock: MOCK,
  openCard: (code, name) => openCodeDetail(code, name), watch: (id, choice) => { games.watch(id, {choice}); render(); }});
// Games (Play > Games): saved games, stats and the bot's view of each decision, the same data Ask's tools read.
const games = gamesFeature({send: (m) => { if (ws) ws.send(m); }, render: () => render(), mock: MOCK,
  play: () => { lobbyTab = playSub = "play"; saveSetting("ygo-lobbytab", lobbyTab); render(); },
  ask: (text) => { lobbyTab = "ask"; saveSetting("ygo-lobbytab", lobbyTab); ask.compose(text); ask.enter(); render(); }});
// The chat's and the Games screen's stills of the play screen redraw their block when ready.
onStill(() => render());
// "Describe a card" model download (Settings > Cards); the deck builder's describe mode waits for it.
const dm = describeModel({
  send: (m) => { if (ws) ws.send(m); },
  render: () => render(),
  mock: MOCK,
  network: () => artNetwork(),
  onReady: (v) => db.setDescribeReady(v),
});
// Popovers that close on a tap anywhere outside them (user 2026-11-13: "It should just close if I click anywhere
// else in the app"). One rule for the whole app: when it is open, what counts as inside, how to close. The tap still
// does what it does. Sheets and the keyword popup close through their scrims; the card action menu in onClick.
const POPOVERS = [
  {open: () => pickerOpen === "model", inside: ".pickpanel, [data-pickopen]", close: () => { pickerOpen = null; }},
  {open: () => db.menuOpen(), inside: ".csch-menu, .csch-mode", close: () => db.closeMenu(false)},
];
document.addEventListener("click", (ev) => {
  let closed = false;
  for (const p of POPOVERS) {
    if (p.open() && !ev.target.closest(p.inside)) { p.close(); closed = true; }
  }
  if (closed) setTimeout(() => render(), 0);
}, true);
// Opens Settings scrolled to one panel (the deck builder's "Download in Settings").
function openSettingsAt(id) {
  settingsOpen = true;
  render();
  // Like Android's settings search: scroll the panel into view and highlight it briefly so it is clear where to look.
  requestAnimationFrame(() => {
    const e = document.getElementById(id);
    if (!e) return;
    const smooth = !reducedMotion();
    e.scrollIntoView({block: "center", behavior: smooth ? "smooth" : "auto"});
    // The glow starts once the scroll has settled, so none of it is spent while the page moves. It is a Web
    // Animation on the element, not a class: Settings re-renders about once a second (art sync, download progress)
    // and a re-render resets classes, which cut the glow off abruptly (user 2026-11-13: "it should fade away and not
    // just disappear"). Holds ~3 s, then fades over 1.5 s; reduced motion: holds, then clears without fading.
    setTimeout(() => {
      const on = {backgroundColor: "rgba(var(--accent-rgb), 0.14)", boxShadow: "0 0 0 10px rgba(var(--accent-rgb), 0.14)",
        outline: "1.5px solid var(--accent)", outlineOffset: "10px"};
      const off = {backgroundColor: "rgba(var(--accent-rgb), 0)", boxShadow: "0 0 0 10px rgba(var(--accent-rgb), 0)",
        outline: "1.5px solid rgba(var(--accent-rgb), 0)", outlineOffset: "10px"};
      const rm = reducedMotion();
      e.animate([on, Object.assign({offset: 0.67, easing: rm ? "steps(1, end)" : "ease-out"}, on), off],
        {duration: 4500, fill: "none"});
    }, smooth ? 450 : 0);
    const h = e.querySelector("h3, h2");
    if (h) { h.setAttribute("tabindex", "-1"); h.focus({preventScroll: true}); }
  });
}
let room = null;           // the last {"t":"room"} frame while in a room, else null
let rooms = null;          // nearby rooms from the last scan (null = never scanned)
let scanning = false;
let scanErr = null;
let roomForm = {address: "", password: "", showPw: false, err: null, busy: null};  // busy: "host" | "join" | null
let roomUi = {closeArm: false, copied: null, note: null, deckLabel: null};
let roomCloseTimer = 0;
let duelist = "Duelist";   // the name others see (localStorage ygo-duelist)
try { duelist = localStorage.getItem("ygo-duelist") || "Duelist"; } catch (e) { /* ignore */ }
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

// Opponent deck primer (step 5, item 7): the sheet, its deck name, and loading.
let primerOpen = false;
let primer = null;         // trimmed primer reply, or null (not enough games)
let primerDeck = "";       // pretty bot deck name shown as the title
let primerForge;           // lobby: the bot deck's Forge coverage (null = not trained); undefined in a game
let primerHash = null;     // hash of the asked bot list (matches study_status.hash)
let primerLoading = false;
let canStudy = false;      // the primer reply said no primer but the list can be studied
let studyState = null;     // last study_status for the open sheet's deck, or null

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

// ---- the stage (docs/design/ui-attention.md) ----
// One visible layer-1/2/3 surface at a time. Layer 3 (what the user opened) goes
// first; narration (1) is read before a decision (2); Pause holds 1 and 2.
// `sync` reconciles the open-state variables with the controller at the start of
// every render, so no surface can decide on its own to cover another.
const STAGE_LAYER = {
  detail: 3, kw: 3, review: 3, settings: 3, deck: 3, log: 3, primer: 3, gamemenu: 3, pile: 3, dbsheet: 3,
  cues: 1, reveal: 1, recap: 1,
  rps: 2, menu: 2, zone: 2, sheet: 2,
};
// Lower wins. A recap is narration but a passive summary, so a decision may take
// the stage from it (and it returns afterwards).
const STAGE_PRIO = {
  detail: 10, kw: 11, review: 12, settings: 13, deck: 14, log: 15, primer: 16, gamemenu: 17, pile: 18, dbsheet: 19,
  cues: 20, reveal: 21,
  rps: 30, menu: 31, zone: 32, sheet: 33,
  recap: 34,
};
const stage = (() => {
  let active = null;
  let opts = null;
  let queue = [];
  let held = false;
  let fallback = 0;
  let activated = null;
  const layer = (k) => STAGE_LAYER[k] || 0;
  function clearFallback() {
    if (fallback) { clearTimeout(fallback); fallback = 0; }
  }
  // A decision queued behind narration never waits forever, but narration that
  // is playing normally is never cut: this is a stall watchdog, re-armed at every
  // new scene (progress). Only when no scene starts for STALL_MS (longer than the
  // longest scene at 1x) is the rest applied at once. A flat 2.5 s ceiling from
  // the prompt's arrival skipped a whole bot turn at once (user 2026-11-12: "10
  // moves happened almost all at once by the bot").
  const STALL_MS = 4000;
  function armFallback() {
    if (fallback) return;
    fallback = setTimeout(() => {
      fallback = 0;
      if (active !== "cues" || cue.waiting() || held) return;   // an explicit hold is not a ceiling
      try { cue.flush(); } catch (e) { /* the queue still drains below */ }
      if (active === "cues") { active = null; opts = null; }
      render();
    }, STALL_MS);
  }
  function activate(kind, o) {
    active = kind;
    opts = o || null;
    clearFallback();
    if (activated) activated(kind);
  }
  function request(kind, o) {
    if (!kind) return false;
    if (active === kind) { if (o) opts = o; return false; }
    if (kind === "kw" && active === "detail") return false;   // the gloss is part of the card view
    queue = queue.filter((q) => q.kind !== kind);
    const ln = layer(kind);
    if (held && ln <= 2) { queue.push({kind, opts: o}); return false; }
    if (!active) { activate(kind, o); return true; }
    if (STAGE_PRIO[kind] < STAGE_PRIO[active]) { activate(kind, o); return true; }
    queue.push({kind, opts: o});
    if (ln === 2 && active === "cues" && !cue.waiting()) armFallback();
    return false;
  }
  function done(kind) {
    queue = queue.filter((q) => q.kind !== kind);
    if (active !== kind) return;
    active = null;
    opts = null;
    clearFallback();
  }
  function sync(intents) {
    const want = new Set((intents || []).filter(Boolean));
    queue = queue.filter((q) => want.has(q.kind));
    if (active && !want.has(active)) { active = null; opts = null; }
    if (!active) clearFallback();
    const ordered = [...want].sort((a, b) => (STAGE_PRIO[a] || 0) - (STAGE_PRIO[b] || 0));
    for (const k of ordered) request(k);
  }
  return {
    request, done, sync,
    // A new narration scene started: the watchdog restarts from now.
    progress() { if (fallback) { clearFallback(); armFallback(); } },
    active: () => active,
    layer,
    lookup: () => !!(active && layer(active) === 3),
    held: () => held,
    hold(on) {
      on = !!on;
      if (on === held) return;
      held = on;
      if (held && active && layer(active) <= 2) { active = null; opts = null; clearFallback(); }
    },
    reset() { clearFallback(); active = null; opts = null; queue = []; held = false; },
    onActivate(fn) { activated = fn; },
  };
})();
stage.onActivate((kind) => {
  if (kind !== "reveal") return;
  clearTimeout(revealTimer);
  if (!MOCK) revealTimer = setTimeout(() => dismissReveal(), 1600);
});
// The stage for the current state, highest priority first.
function stageIntents() {
  const out = [];
  if (detail) out.push("detail");
  else if (kwPop) out.push("kw");
  if (reviewFrame) out.push("review");
  if (settingsOpen) out.push("settings");
  if (deckBrowser) out.push("deck");
  if (db.sheetOpen()) out.push("dbsheet");
  if (logOpen) out.push("log");
  if (primerOpen) out.push("primer");
  if (gameMenu) out.push("gamemenu");
  if (pileOpen) out.push("pile");
  if (cue.busy() || cue.frozen()) out.push("cues");
  if (revealCur) out.push("reveal");
  if (rpsGame) out.push("rps");
  if (menuUid && !locked) out.push("menu");
  if (msg && msg.t === "state" && !msg.result && !paused && !pileOpen) {
    const p = msg.prompt;
    const zp = p ? zonePicks(p, null) : null;
    if (p && p.kind !== "idle" && p.kind !== "battle" && !zp) out.push("sheet");
    else if (zp) out.push("zone");
  }
  if (recap) out.push("recap");
  return out;
}
function syncStage() { stage.sync(stageIntents()); }
// Focus inside a surface that steps aside (or is queued): remembered by its
// descriptor, restored when that surface takes the stage again.
let stageFocus = null;
function describeIn(root, el) {
  for (const a of ["data-opt", "data-act", "data-uid", "data-detailcard"]) {
    const v = el.getAttribute(a);
    if (v != null) return {el, sel: root + " [" + a + '="' + CSS.escape(v) + '"]'};
  }
  return null;
}
function rememberStageFocus() {
  const a = document.activeElement;
  if (!a || !a.closest) return;
  if (a.closest("#sheet")) stageFocus = describeIn("#sheet", a) || stageFocus;
  else if (a.closest("#menu") && a.hasAttribute("data-opt")) stageFocus = describeIn("#menu", a) || stageFocus;
}

// Android back / browser back (user 2026-11-13: swiping back on the deck picker or Multiplayer closed the app; it
// should step back towards the home page). While anything can be stepped back from, one history entry sits above the
// page; a back press pops it, goBack() takes one step, and the next render re-arms the entry if more steps remain.
// Only on the home page (Bot tab, nothing open) does back leave the app.
const HOME_TAB = "play";
let backArmed = false;
let popGuard = 0;        // pops we triggered ourselves (disarming after a state closed by tap)
function openOvNames() {
  const o = [];
  if (rpsGame) o.push("rps");
  if (logOpen) o.push("log");
  if (menuUid) o.push("menu");
  if (gameMenu) o.push("gamemenu");
  if (primerOpen) o.push("primer");
  if (pileOpen) o.push("pile");
  if (detail) o.push("detail");
  if (kwPop) o.push("kw");
  if (revealCur) o.push("reveal");
  if (settingsOpen) o.push("settings");
  if (deckBrowser) o.push("deck");
  if (pickerOpen) o.push("picker");
  if (reviewFrame) o.push("review");
  if (inSolution) o.push("solution");
  if (db.editing()) o.push("dbedit");
  if (db.sheetOpen()) o.push("dbsheet");
  if (lobbyTab === "ask" && (ask.sheetOpen() || ask.viewOpen())) o.push("ask");
  return o;
}
const inGameView = () => !!(msg && msg.t === "state" && !IS_REPLAY);
function backWanted() {
  return openOvNames().length > 0 || inGameView() || !!room || lobbyTab !== HOME_TAB;
}
function syncOv() {
  const want = backWanted();
  if (want && !backArmed) {
    try { history.pushState({ov: 1}, ""); backArmed = true; } catch (e) { /* ignore */ }
  } else if (!want && backArmed) {
    backArmed = false;
    try { popGuard += 1; history.back(); } catch (e) { popGuard = 0; }
  }
}
function closeState(name) {
  if (name === "rps") rpsGame = null;
  else if (name === "log") logOpen = false;
  else if (name === "menu") menuUid = null;
  else if (name === "gamemenu") { gameMenu = false; surrenderArm = false; }
  else if (name === "primer") { primerOpen = false; }
  else if (name === "pile") pileOpen = null;
  else if (name === "detail") { detail = null; kwPop = null; }
  else if (name === "kw") kwPop = null;
  else if (name === "reveal") dismissReveal();
  else if (name === "settings") settingsOpen = false;
  else if (name === "dbsheet" || name === "dbedit") db.back(name);
}
// Kept for the X/scrim close paths: the next render's syncOv() disarms the entry when nothing is left to close.
function closeOv(_name) {}
// One back step: the top layer, then the deck editor, then game -> game menu, room -> leave, other tab -> home tab.
function goBack() {
  if (db.closeMenu()) return;
  if (db.sheetOpen()) { db.back("dbsheet"); render(); return; }
  if (games.watching && games.back()) { render(); return; }
  if (lobbyTab === "ask" && ask.back()) { render(); return; }
  if (lobbyTab === "games" && games.back()) { render(); return; }
  if (closeTop()) return;
  if (db.editing()) { db.back("dbedit"); render(); return; }
  if (inGameView()) { gameMenu = true; render(); return; }
  if (room) { onRoomAction("roomleave", null); render(); return; }
  if (lobbyTab !== HOME_TAB) { lobbyTab = HOME_TAB; saveSetting("ygo-lobbytab", lobbyTab); render(); }
}
window.addEventListener("popstate", () => {
  if (popGuard > 0) { popGuard -= 1; return; }
  backArmed = false;
  goBack();
  syncOv();
});

// ---- sizing ----
// The board is width-bound: 5 main columns plus 2 narrower pile columns.
// Height left after board + bars goes to the hand (up to 1.7x the board card),
// then to the info panel, which absorbs the rest (flex), so nothing is dead space.
const R = 86 / 59;
const SIDE = 0.8;
function fit(again = true) {
  const root = document.documentElement;
  // The table grows with the screen (docs/plans/responsive.md §5): a phone's table on a phone, larger cards on a
  // tablet or a laptop. Wide screens keep the card reader and log beside it; text keeps its size.
  const iw = window.innerWidth, ih = window.innerHeight;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  // From 1100 px the reader and the log are columns of at least 290 px each side of the table: the table takes the rest.
  const z = iw >= 1100 ? clamp(Math.min(iw / 1000, ih / 860, (iw - 580) / 520), 1, 1.5)
    : iw >= 900 ? clamp(Math.min(iw / 1000, ih / 860), 1, 1.5) : clamp(Math.min(iw / 520, ih / 880), 1, 2);
  root.style.setProperty("--app-w", Math.round(520 * z) + "px");
  const W = Math.min(iw, 520 * z);
  const H = ih;
  const PAD = 6, GAP = 3;
  let w = Math.min((W - 2 * PAD - 6 * GAP) / (5 + 2 * SIDE), 76 * z);
  let hw = Math.min(w * 1.7, 112 * z);
  // Desktop: card info lives in the side panel. Practice puts Hint/Undo in the info panel
  // (44px buttons + padding + the lifted hand's 10px), and a puzzle/what-if banner sits above the board.
  const infoMin = window.innerWidth >= 900 ? 0 : (practiceMode && !IS_REPLAY ? 80 : 64);
  const banner = document.querySelector("#puzzlebanner, #whatifbanner");
  // The win-chance row sits under the turn panel: count it, or the info line slides under the hand
  // (user 2026-11-12, phone screenshot with win chance on).
  const wcRow = document.getElementById("winchance");
  // A replay also shows the opponent's hand (#thand) and a caption bar under the hand (#app.replay #sheet): count them,
  // or Watch cut your hand off at the bottom (2026-11-16, 16" laptop).
  const th = IS_REPLAY ? document.getElementById("thand") : null;
  const cap = IS_REPLAY ? document.querySelector("#app.replay #sheet") : null;
  const extra = (banner ? banner.offsetHeight + 2 : 0) + (wcRow ? wcRow.offsetHeight : 0) +
    (th ? th.offsetHeight : 0) + (cap ? cap.offsetHeight : 0);
  // The info panel needs its real lines (status + last action, two lines on a phone), not a fixed guess:
  // measured from the context bar's children, capped so cards never shrink for a long feed.
  let infoNeed = infoMin;
  const cb = window.innerWidth >= 900 ? null : document.querySelector("#info .ctxbar:not(.sel)");
  if (cb) {
    const kids = [...cb.children].filter((c) => c.offsetHeight);
    const hb = document.querySelector("#info .hintbar");
    const natural = kids.reduce((a, c) => a + c.offsetHeight, 0) + 6 * Math.max(0, kids.length - 1) + 24 +
      (hb ? hb.offsetHeight : 0);
    infoNeed = Math.min(Math.max(infoMin, natural), 140);
  }
  // The strip between the fields holds the turn panel, whose text keeps its size (and the phone's font scale): on a
  // small board it is taller than a card. Count its real height, or the panel and the Extra Monster Zones ran into
  // your field (2026-11-16, Watch on the user's phone).
  const panelH = () => [...document.querySelectorAll("#midstrip .phase, #midstrip .turnno")].reduce((a, e) => Math.max(a, e.offsetHeight), 0);
  const panel = panelH();
  const need = (w, hw) => 44 + 54 + (4 * w * R + Math.max(w * R, panel) + 36) + (hw * R + 26) + infoNeed + extra;
  const w0 = w, hw0 = hw;
  for (let i = 0; i < 200 && need(w, hw) > H; i++) {
    if (hw > w * 1.25) hw -= 1;
    else w -= 0.5;
  }
  w = Math.max(30, w);
  // Short viewports (200% zoom on a laptop, landscape phones): rather than shrinking cards
  // below legibility, keep them readable and let the page scroll (WCAG 1.4.4 / 1.4.10).
  // Watch's player and the export stills (EMBED) are a fixed frame that cannot scroll: the cards shrink to fit
  // instead (2026-11-16, the user's phone: Watch left 619 px and short mode cut the hand off).
  const short = !EMBED && w < Math.min(w0, 44);
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
  // A sheet whose top lands inside your bar cut the LP digits in half: cover the whole bar instead.
  // Layout offsets, not the client rect: the sheet's entry animation translates it.
  const sh = document.getElementById("sheet");
  if (sh && bar && sh.offsetParent) {
    sh.style.minHeight = "";
    const pr = sh.offsetParent.getBoundingClientRect();
    const top = pr.top + sh.offsetTop, bottom = top + sh.offsetHeight, b = bar.getBoundingClientRect();
    if (top > b.top - 2 && top < b.bottom) sh.style.minHeight = Math.ceil(bottom - b.top + 4) + "px";
    // Nor may it stop across the names on your hand cards (a short sheet, e.g. "Choose a zone"):
    // it then covers the hand whole.
    const cards = [...document.querySelectorAll("#hand .hcard")].map((c) => c.getBoundingClientRect().top);
    const ht = cards.length ? Math.min(...cards) : null;
    if (ht != null && top > ht - 2 && top < ht + 48) sh.style.minHeight = Math.ceil(bottom - ht + 6) + "px";
  }
  const hand = document.getElementById("hand");
  if (hand) {
    const n = hand.children.length;
    const avail = hand.clientWidth - 24;
    let g = 6;
    if (n > 1 && n * hw + (n - 1) * 6 > avail) g = (avail - n * hw) / (n - 1);
    hand.style.setProperty("--hand-gap", g.toFixed(1) + "px");
  }
  // Smaller cards narrow the panel's columns, so its text may wrap onto another line: fit once more with that height.
  if (again && panelH() > panel + 1) fit(false);
}
window.addEventListener("resize", () => { fit(); markInfoOverflow(); });

// Card-text fade + scroll (item 5): the bottom fade (.over) and the tab stop
// exist only while the text actually overflows, so it reads as scrollable.
// Touch scrolls natively (overflow-y); the keyboard reaches it via tabindex.
function markInfoOverflow() {
  const el = app.querySelector("#info .ftext");
  if (!el) return;
  const over = el.scrollHeight > el.clientHeight + 1;
  el.classList.toggle("over", over);
  if (over) {
    el.setAttribute("tabindex", "0");
    el.setAttribute("role", "region");
    el.setAttribute("aria-label", "Card text");
  } else {
    el.removeAttribute("tabindex");
    el.removeAttribute("role");
    el.removeAttribute("aria-label");
  }
}

const reducedMotion = () =>
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---- data helpers ----
function allCards() {
  const out = [];
  if (!msg || !msg.view) return out;
  for (const [side, who] of [[msg.view.you, "you"], [msg.view.bot, "bot"]]) {
    out.push(...(side.hand || []), ...(side.mzone || []), ...(side.emz || []), ...(side.szone || []),
      ...(side.gy || []), ...(side.banished || []), ...pileCards("extra:" + who), side.field);
  }
  for (const l of cueChain || msg.view.chain || []) out.push(l.card);
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
  // A replay file played offline (game_video.py) has no server; Watch's player (?replay=embed) has the app's.
  if (!code || (IS_REPLAY && REPLAY_URL !== "embed")) return null;
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
      showBanner(oppName() + " " + t.charAt(0).toLowerCase() + t.slice(1));
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
  say(oppName() + " activates " + ((revealCur.card && revealCur.card.name) || "a card"));
  stage.request("reveal");
  render();
}

function dismissReveal() {
  if (!revealCur || revealLeaving) return;
  // The reveal waits behind a lookup or another narration; its clock restarts
  // when the stage makes it active again (stage.onActivate).
  if (stage.active() !== "reveal") return;
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
    ? '<div class="rface">' + phHTML(c) + artImg(c.code, "full", 'alt=""') + "</div>"
    : '<div class="rface">' + phHTML(c) + "</div>";
  const txt = (t && t.text) || (MOCK ? "Card text loads from the server in a live game." : "");
  return '<div id="reveal" data-key="reveal" data-act="closereveal" aria-hidden="true">' +
    face + '<div class="rkicker">' + esc(oppName()) + " activates</div>" +
    "<h2>" + esc(c.name || "") + "</h2>" +
    '<div class="rtext">' + esc(txt) + "</div></div>";
}

// ---- outbound ----
function choose(index) {
  if (locked || paused || !msg || !msg.prompt) return;
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
  if (!p || locked || paused) return;
  const opt = (p.options || []).find((o) => o.index === index);
  if (!opt) return;
  // the env takes one pick per step: picked cards come back as "unselect" options
  choose(index);
}

// Practice-mode undo: rewrite to the previous decision; the server re-sends the
// log and cues from the rewound position (see web/server.py `on_undo`).
function doUndo() {
  if (!msg || msg.result || !practiceMode || !canUndo || locked || paused || msg.thinking) return;
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
  if (kind === "chain") {
    // Chain links as cards, labelled "n · name" (and whose), so the sheet reads like the strip.
    return (cueChain || msg.view.chain || []).map((l, i) => l.card && Object.assign({}, l.card, {name: (i + 1) + " · " +
      (l.card.name || "a card") + (l.controller === "bot" ? " · " + oppName() : " · you")}));
  }
  const side = who === "bot" ? msg.view.bot : msg.view.you;
  const cards = (side && side[kind === "gy" ? "gy" : kind]) || [];
  // Extra Deck cards lie face-down in the engine, but the views only send the ones you may see (all of yours, the
  // bot's face-up pendulums): show them face-up so the sheet draws them and a tap opens their details.
  return kind === "extra" ? cards.map((c) => (c && (c.code || c.name) ? {...c, faceup: true} : c)) : cards;
}

function hintFor(p) {
  if (IS_REPLAY) return "";   // the caption bar carries the narration instead
  if (msg.result) return "";
  if (msg.thinking) return "";   // the bot bar already shows it
  if (!p) return "Waiting for " + oppName() + "…";
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
      (!MOCK ? artImg(c.code, "small", 'alt=""') : "") + "</span>"
      : '<span class="rth none"></span>';
    const tag = c && c.code ? "button" : "div";
    return "<" + tag + ' class="rrow' + (c && c.owner === "opp" ? " opp" : " me") + (it.neg ? " neg" : "") + '"' +
      (c && c.code ? ' data-recap="' + i + '"' : "") + ">" + thumb +
      '<span class="rtx"><span>' + esc(it.text) + "</span>" + (it.sub ? '<span class="rsub">' + esc(it.sub) + "</span>" : "") +
      "</span></" + tag + ">";
  }).join("");
  return '<div class="recap" data-key="recap" role="region" aria-label="What happened">' +
    '<div class="rhead"><span class="kicker">While you waited</span><span class="rt">' + esc(recap.who) +
    " · turn " + recap.turn + "</span>" +
    // A short info panel (phones) has no room for the rows: the count opens the log instead.
    '<button class="rmore" data-act="log">' + recap.items.length + (recap.items.length === 1 ? " play" : " plays") +
    ' <span aria-hidden="true">›</span></button>' +
    '<button class="iconbtn sm" data-act="closerecap" aria-label="Close recap">' +
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

// The other side's name: the room opponent, else the bot.
function oppName() {
  return (msg && msg.room && msg.room.opponent) || "decklight";
}

// Card-art sync (Rust server): the device's network kind, sent with the art commands.
// The Android app answers through DecklightNet, a real metered check (a metered Wi-Fi hotspot
// counts as cellular). Browsers expose only the network type, if that. "unknown" means Wi-Fi,
// except "none" (no network): the sync waits for one, so keep it from starting on mobile data.
function artNetwork() {
  try {
    if (window.DecklightNet && DecklightNet.network) {
      const n = DecklightNet.network();
      return n === "none" ? "cellular" : n;
    }
  } catch (e) {
    // Fall through to the browser API.
  }
  const c = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (!c) return "unknown";
  if (c.type === "wifi" || c.type === "ethernet") return "wifi";
  if (c.type === "cellular" || c.type === "wimax") return "cellular";
  return "unknown";
}
function sendArtNetwork() {
  if (!MOCK && ws) ws.send({t: "art_sync", network: artNetwork()});
}
function sendArtToggle() {
  if (!MOCK && ws) ws.send({t: cardArt ? "art_sync_resume" : "art_sync_pause"});
}
function fmtNum(n) {
  return String(Math.max(0, Math.round(n || 0))).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
function artMB(bytes) {
  const mb = (bytes || 0) / 1e6;
  return (mb >= 100 ? Math.round(mb) : Math.round(mb * 10) / 10) + " MB";
}
function artStateWord(a) {
  if (!a) return "";
  if (a.state === "running") return "Downloading";
  if (a.state === "paused") return "Paused";
  if (a.state === "waiting_wifi") return "Waiting for Wi-Fi";
  if (a.state === "low_space") return "Low storage: stopped";
  if (a.state === "rate_limited") {
    const min = Math.max(1, Math.ceil((a.resume_in_s || 0) / 60));
    return "YGOPRODeck asked us to slow down: resumes in " + min + " min";
  }
  if (a.state === "done") return "All saved";
  if (a.state === "error") return "Some cards could not be fetched: retrying";
  return "Ready";
}
// One line, a thin bar and the actions, under the Card art switch in both Settings panels.
function artSyncHTML() {
  const a = artSync;
  if (!a) return "";
  const have = a.have || 0;
  const total = a.total || 0;
  const saved = "Card art: " + fmtNum(have) + " of " + fmtNum(total) + " saved · " + artMB(a.bytes);
  const unavailable = (a.unavailable || 0) > 0 ? " · " + fmtNum(a.unavailable) + " unavailable" : "";
  const pct = total > 0 ? Math.min(100, Math.round((have / total) * 1000) / 10) : 0;
  const running = a.state === "running" || a.state === "waiting_wifi" || a.state === "error";
  // Card art is the master switch: with it off there is nothing to resume (the server was told to pause).
  const pauseBtn = !cardArt
    ? '<span class="artoff">Turn on Card art to download the rest.</span>'
    : running
      ? '<button class="artbtn" data-act="artpause">Pause</button>'
      : '<button class="artbtn" data-act="artresume">Resume</button>';
  const dataBtn = cardArt && a.state === "waiting_wifi"
    ? '<button class="artbtn" data-act="artdata">Download on mobile data</button>'
    : "";
  const delText = artDeleteArm
    ? "Delete " + artMB(a.bytes) + " of saved card art? It downloads again while Card art is on."
    : "Delete saved art";
  const delBtn = (have > 0 || a.bytes > 0)
    ? '<button class="artbtn danger' + (artDeleteArm ? " confirm" : "") + '" data-act="artdelete">' +
      esc(delText) + "</button>"
    : "";
  return '<div class="artsync" role="group" aria-label="Card art download">' +
    '<div class="artsync-line" role="status" aria-live="polite">' + esc(saved + unavailable) +
    '<span class="artstate">' + esc(artStateWord(a)) + "</span></div>" +
    '<div class="artbar" aria-hidden="true"><i style="width:' + pct + '%"></i></div>' +
    '<div class="artbtns">' + pauseBtn + dataBtn + delBtn + "</div></div>";
}

function gameMenuHTML() {
  const wcNote = (gameMode === "random" || gameMode === "fast")
    ? '<div class="mhint">Win chance needs the full bot.</div>'
    : "";
  return '<div id="gscrim" data-key="gscrim" data-act="closegamemenu"></div>' +
    '<div id="gamemenu" data-key="gamemenu" role="dialog" aria-modal="true" aria-label="Game menu" tabindex="-1" data-close="closegamemenu">' +
    '<div class="grab" aria-hidden="true"></div><div class="mhead">Game menu<button class="iconbtn" data-act="closegamemenu" aria-label="Close game menu">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
    '<div class="mbody">' + (msg && msg.room ? roomMenuHTML() :
    '<div class="mrow disabled" aria-disabled="true"><span class="muted">Bot: ' +
    esc((msg && msg.model_label) || gameModel || "Base model") + " · " +
    esc({product: "Full", fast: "Fast", random: "Random"}[gameMode] || "Full") + "</span></div>" +
    '<button class="mrow danger' + (surrenderArm ? " confirm" : "") + '" data-act="surrender">' +
    (surrenderArm ? "Tap again to surrender" : "Surrender") + "</button>" +
    '<button class="mrow" data-act="backlobby">Back to lobby</button>' +
    "<button class=\"mrow\" data-act=\"primer\">Opponent's deck</button>" +
    '<div class="msep"></div>' +
    '<div class="mrow" style="pointer-events:none" id="gm-pace">Pacing</div>' +
    '<div class="seg" role="radiogroup" aria-labelledby="gm-pace">' + speedSeg("1", "1×") + speedSeg("2", "2×") + speedSeg("instant", "Instant") + "</div>" +
    toggleRow("winchanceset", winChance, "Show win chance",
      "The bot's estimate of your chance to win, calibrated on ~3,000 games. Off by default so it doesn't steer your play.") +
    wcNote +
    toggleRow("cardart", cardArt, "Card art", "Card images from YGOPRODeck, saved on this device once downloaded. Not bundled.")) +
    '<div class="msep"></div>' +
    '<button class="mrow" data-act="opensettings">All settings…</button>' +
    "</div></div>";
}

// App-wide Settings screen (lobby gear + "All settings…" in the game menu). One
// implementation per control: the rows reuse the same data-act handlers as the
// lobby and the game menu, so a change anywhere shows everywhere at once.
function settingsHTML() {
  if (!settingsOpen) return "";
  const chainSeg = (v, label) =>
    '<button role="radio" aria-checked="' + (chainMode === v) + '" class="' + (chainMode === v ? "sel" : "") +
    '" data-act="chainset" data-v="' + v + '">' + label + "</button>";
  const themeSeg = (v, label) =>
    '<button role="radio" aria-checked="' + (themeMode === v) + '" class="' + (themeMode === v ? "sel" : "") +
    '" data-act="themeset" data-v="' + v + '">' + label + "</button>";
  const back = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12.5 4.5 7 10l5.5 5.5"/></svg>';
  return '<div id="settings" role="dialog" aria-modal="true" aria-labelledby="settings-title" tabindex="-1">' +
    '<header class="shead"><button class="iconbtn" data-act="closesettings" aria-label="Back">' + back + "</button>" +
    '<h1 id="settings-title" tabindex="-1" data-autofocus>Settings</h1></header>' +
    '<div class="sbody">' +
    '<section id="set-appearance" aria-labelledby="set-appearance-h"><h2 id="set-appearance-h">Appearance</h2>' +
    '<div class="setlbl" id="set-theme">Theme</div>' +
    '<div class="seg" role="radiogroup" aria-labelledby="set-theme">' +
    themeSeg("dark", "Dark") + themeSeg("light", "Light") + themeSeg("system", "System") + "</div>" +
    '<div class="mhint">System follows your phone\'s dark mode.</div>' +
    "</section>" +
    '<section id="set-gameplay" aria-labelledby="set-gameplay-h"><h2 id="set-gameplay-h">Gameplay</h2>' +
    '<div class="setlbl" id="set-chain">Chain responses</div>' +
    '<div class="seg" role="radiogroup" aria-labelledby="set-chain">' +
    chainSeg("auto", "Auto") + chainSeg("all", "All") + "</div>" +
    '<div class="mhint">Auto skips empty windows with nothing to respond to.</div>' +
    '<div class="setlbl" id="set-pace">Pacing</div>' +
    '<div class="seg" role="radiogroup" aria-labelledby="set-pace">' +
    speedSeg("1", "1×") + speedSeg("2", "2×") + speedSeg("instant", "Instant") + "</div>" +
    toggleRow("cuepause", cuePause, "Pause after each bot action", "Tap anywhere or press Enter to continue.") +
    "</section>" +
    '<section id="set-learning" aria-labelledby="set-learning-h"><h2 id="set-learning-h">Learning aids</h2>' +
    toggleRow("winchanceset", winChance, "Show win chance",
      "The bot's estimate of your chance to win, calibrated on ~3,000 games. Off by default so it doesn't steer your play.") +
    toggleRow("cuerecap", cueRecap, "Turn recap", "What the bot did, at the start of your turn.") +
    toggleRow("rulenotes", ruleNotesOn, "Rule notes", "A one-line rule tip on the first of each kind of play.") +
    "</section>" +
    '<section id="set-cards" aria-labelledby="set-cards-h"><h2 id="set-cards-h">Cards</h2>' +
    toggleRow("cardart", cardArt, "Card art", "Card images from YGOPRODeck, saved on this device once downloaded. Not bundled.") +
    artSyncHTML() + packPanelHTML() + dm.html() +
    "</section>" +
    '<section id="set-profile" aria-labelledby="set-profile-h"><h2 id="set-profile-h">Profile</h2>' +
    '<div class="field-group"><label class="flabel" for="sname">Duelist name</label>' +
    '<input id="sname" class="rinput" maxlength="20" autocomplete="nickname" spellcheck="false" value="' + esc(duelist) + '"></div>' +
    "</section>" +
    '<section id="set-sound" aria-labelledby="set-sound-h"><h2 id="set-sound-h">Sound</h2>' +
    '<div class="mrow disabled" aria-disabled="true">Sound<span class="sub">Off</span></div>' +
    "</section>" +
    '<section id="set-about" aria-labelledby="set-about-h"><h2 id="set-about-h">About</h2>' +
    '<p class="setnote">Card images from YGOPRODeck, saved on this device once downloaded. Not bundled.</p>' +
    "</section>" +
    "</div></div>";
}

// The game menu's top rows in a room duel (P5): the opponent, Surrender, Leave room, and the learning
// features greyed with their reason (fair play: they stay off against people).
function roomMenuHTML() {
  const off = (label) => '<div class="mrow disabled" aria-disabled="true">' + label + '<span class="sub">Off in rooms</span></div>';
  return '<div class="mrow disabled" aria-disabled="true"><span class="muted">Playing ' +
    esc(msg.room.opponent || "a person") + "</span></div>" +
    '<button class="mrow danger' + (surrenderArm ? " confirm" : "") + '" data-act="surrender">' +
    (surrenderArm ? "Tap again to surrender" : "Surrender") + "</button>" +
    '<button class="mrow" data-act="roomleave">Leave room</button>' +
    '<div class="msep"></div>' + off("Hint") + off("Undo") + off("What-if") + off("Win chance") +
    '<div class="msep"></div><button class="mrow" data-act="opensettings">All settings…</button>';
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
// "Choose a zone": the options as board slots you tap, from their labels (both servers word
// them the same: "Your Monster Zone 3", "Opponent's Spell & Trap Zone 1", "Extra Monster Zone
// (left)", "Your Field Zone", "Your Pendulum Zone (right)"). Columns count from your left; the
// bot's rows are mirrored. Any label that does not parse keeps the whole prompt a list.
function zonePicks(p, hintByIndex) {
  if (!p || p.kind !== "place") return null;
  const zp = {you: {m: {}, s: {}}, bot: {m: {}, s: {}}, emz: {}};
  let n = 0;
  for (const o of p.options || []) {
    if (o.verb === "cancel" || o.verb === "pass") continue;
    const z = {index: o.index, label: o.label, hint: (hintByIndex || {})[o.index] || null};
    const L = o.label || "";
    let r;
    if ((r = /^(Your|Opponent's) (Monster|Spell & Trap) Zone ([1-5])$/.exec(L))) {
      const mine = r[1] === "Your", col = +r[3];
      zp[mine ? "you" : "bot"][r[2] === "Monster" ? "m" : "s"][mine ? col - 1 : 5 - col] = z;
    } else if ((r = /^(Your|Opponent's) Field Zone$/.exec(L))) {
      zp[r[1] === "Your" ? "you" : "bot"].f = z;
    } else if ((r = /^(Your|Opponent's) Pendulum Zone \((left|right)\)$/.exec(L))) {
      const mine = r[1] === "Your", left = r[2] === "left";
      zp[mine ? "you" : "bot"].s[mine ? (left ? 0 : 4) : (left ? 4 : 0)] = z;
    } else if ((r = /^Extra Monster Zone \((left|right)\)$/.exec(L))) {
      zp.emz[r[1]] = z;
    } else {
      return null;
    }
    n++;
  }
  return n ? zp : null;
}

function hintBarHTML(p, withButton, idle, undo, info) {
  if (!practiceMode || !p || msg.result || IS_REPLAY) return "";
  let inner = "";
  if (hintLoading) {
    inner = '<span class="hintwait" role="status">Asking the bot…</span>';
  } else if (hint && hintPrompt === p.id && (hint.options || []).length) {
    inner = '<ol class="hintlist">' + hint.options.map((ho, i) => {
      const pct = Math.round((ho.p || 0) * 100);
      return '<li><span class="hintbadge" aria-hidden="true">' + (i + 1) + '</span> <span class="hintlab">' +
        esc(ho.label || ("Option " + ho.index)) + "</span>" +
        ' <span class="hintconf">' + pct + "%</span><span class=\"sr-only\">, bot's confidence " +
        pct + " percent" + (ho.est != null ? ", win chance " + Math.round(ho.est * 100) + " percent" : "") + "</span>" +
        (ho.est != null ? '<span class="hintest"> · win ' + Math.round(ho.est * 100) + "%</span>" : "") + "</li>";
    }).join("") + "</ol>";
  } else if (info && info.card && !isHidden(info.card)) {
    // A short info panel (phones) has no room for the focused card under this row: its name
    // and type sit here instead (CSS shows this only then), and open the full details.
    const c = info.card;
    const meta = [info.typeStr, statsLine(c)].filter(Boolean).join(" · ");
    inner = '<button class="hintfocus" data-act="detail" data-uid="' + esc(c.uid || "") + '" aria-label="' +
      esc("Card details: " + (c.name || "card")) + '"><span class="hfname">' + esc(c.name || "") + "</span>" +
      (meta ? '<span class="hfmeta">' + esc(meta) + "</span>" : "") + "</button>";
  } else if (withButton && !puzzle) {
    inner = '<span class="hintsub">bot\u2019s top picks</span>';
  } else {
    return "";   // a sheet with no picks yet: nothing to show under its options
  }
  // Idle/battle prompts: Hint + Undo sit in this row under your bar (the bar
  // itself has no room for them next to the phase buttons at phone width).
  const undoBtn = undo && undo.show
    ? '<button class="hintbtn" data-act="undo" aria-label="Undo your last decision"' +
      (undo.enabled ? "" : ' disabled aria-disabled="true"') + ">Undo</button>" : "";
  const hasList = inner.startsWith('<ol');
  return '<div class="hintbar' + (hasList ? " haslist" : "") + '" data-key="hintbar">' +
    (withButton && !puzzle ? '<button class="hintbtn" data-act="hint"' +
      ' aria-label="Ask the bot for a hint"' + (hintLoading || locked || paused || msg.thinking ? " disabled" : "") + ">Hint</button>" : "") +
    undoBtn + inner + "</div>";
}

// The latest move as one readable line for the context bar (actor as a shape/word,
// never colour alone). `kind` = "bot": the opponent's last event; anything else: the last event.
function lastMoveHTML(kind) {
  const list = shownEvents();
  let e = null;
  if (kind === "bot") {
    for (let i = list.length - 1; i >= 0; i--) if (list[i].actor === "bot") { e = list[i]; break; }
  } else {
    e = list[list.length - 1] || null;
  }
  if (!e) return "";
  const who = e.actor === "you" ? ((msg.names && msg.names[0]) || "You")
    : e.actor === "bot" ? oppName() : "";
  return '<div class="ctxlast" data-key="ctxlast" role="status" aria-live="polite">' +
    (who ? '<span class="ctxwho ' + esc(e.actor || "system") + '">' + esc(who) + "</span>" : "") +
    '<span class="ctxt">' + esc(e.text || "") + (e.neg ? negTag(e.neg) : "") + "</span></div>";
}

// A Spell/Trap in hand that offers Set but no activation gets a disabled Activate
// button, so "Set" never looks like the only thing the card can do. Generic only:
// no card names or card-specific rules.
function setOnlySpellTrap(card, opts) {
  if (!card || isMonster(card)) return false;
  const inHand = opts.some((o) => o.where && o.where.zone === "hand");
  const hasSet = opts.some((o) => o.verb === "set" || o.verb === "set_spell");
  const hasAct = opts.some((o) => o.verb === "activate");
  return inHand && hasSet && !hasAct;
}

// The selected card's actions, as a centred modal over the board (user
// 2026-11-12: the in-bar row was clipped under the hand on a phone).
function actionsHTML() {
  const selected = menuUid && !locked ? menuUid : null;
  if (!selected) return "";
  const c = findCard(selected);
  const opts = optionsForUid(selected);
  const name = c ? (isHidden(c) ? "Face-down card" : (c.name || "Card")) : "Selected card";
  const primary = ["activate", "normal", "special", "attack", "direct", "yes", "set", "set_spell", "position"];
  const btns = opts.map((o, i) => {
    return '<button class="ctxbtn' + (primary.includes(o.verb) ? " primary" : "") + '" data-opt="' + o.index + '"' +
      (i === 0 ? " data-autofocus" : "") + ">" + esc(o.label || o.verb) + "</button>";
  });
  const needActivate = setOnlySpellTrap(c, opts);
  if (needActivate) {
    btns.push('<button class="ctxbtn" disabled aria-disabled="true" aria-describedby="menunote">Activate</button>');
  }
  // Doing nothing is legal when the prompt itself offers pass/cancel: its own
  // button says what that means instead of a generic Cancel.
  const own = new Set(opts.map((o) => o.index));
  const pass = ((msg && msg.prompt && msg.prompt.options) || [])
    .filter((o) => (o.verb === "pass" || o.verb === "cancel") && !own.has(o.index));
  const foot = pass.length
    ? pass.map((o) => '<button class="ctxbtn ghost" data-opt="' + o.index + '">' + esc(o.label || o.verb) + "</button>").join("")
    : '<button class="ctxbtn ghost" data-act="closemenu">Cancel</button>';
  const aside = stage.active() !== "menu";
  return (aside ? "" : '<div id="mscrim" data-key="mscrim" data-act="closemenu"></div>') +
    '<div id="menu" data-key="menu" class="' + (aside ? "stage-aside" : "") + '" role="dialog"' +
    (aside ? "" : ' aria-modal="true"') + ' aria-labelledby="menu-title" tabindex="-1"' +
    (aside ? ' inert aria-hidden="true"' : "") + ">" +
    '<div class="mhead2"><h2 id="menu-title" title="' + esc(name) + '">' + esc(name) + "</h2>" +
    '<button class="ctxdetail" data-act="detail" data-uid="' + esc(selected) + '">Details</button></div>' +
    '<div class="mactions" data-key="mactions">' + btns.join("") + "</div>" +
    (needActivate ? '<p class="ctxnote" id="menunote" data-key="menunote">' +
      "Can't activate right now: its activation requirements aren't met.</p>" : "") +
    '<div class="mfoot" data-key="mfoot">' + foot + "</div></div>";
}

// Context bar: the current prompt or latest move (the old floating action menu
// and the unreadable log box are gone).
function contextBarHTML(p, zp) {
  let line = "";
  let last = "";
  if (zp) {
    line = "Tap a highlighted zone";
    last = lastMoveHTML("");
  } else if (p && (p.kind === "idle" || p.kind === "battle")) {
    line = hintFor(p);
    last = lastMoveHTML("bot");
  } else if (p && msg.can_cancel) {
    line = p.text || "";
    last = lastMoveHTML("");
  } else if (p) {
    last = lastMoveHTML("");
  } else {
    last = lastMoveHTML("");
  }
  const canCancel = !!(p && msg.can_cancel && !msg.result);
  const cancelDisabled = locked || paused || !!msg.thinking;
  const cancel = canCancel
    ? '<button class="ctxcancel" data-act="cancel" aria-label="Cancel this action"' +
      (cancelDisabled ? ' disabled aria-disabled="true"' : "") + ">Cancel</button>" : "";
  const head = line || cancel
    ? '<div class="ctxhead" data-key="ctxhead">' +
      (line ? '<div class="ctxline" data-key="ctxline" title="' + esc(line) + '">' + esc(line) + "</div>" : '<div class="ctxline"></div>') +
      cancel + '<button class="ctxlog" data-act="log" aria-label="Open duel log">Log</button></div>'
    : '<div class="ctxhead only" data-key="ctxhead"><button class="ctxlog" data-act="log" aria-label="Open duel log">Log</button></div>';
  return '<div class="ctxbar" data-key="ctxbar">' + head + last + "</div>";
}

function renderGame() {
  syncStage();
  const active = stage.active();
  ovCues.classList.toggle("stage-hidden", stage.lookup());
  const v = msg.view;
  const p = msg.prompt;
  const act = locked || paused ? {} : actionableMap(p);
  const hm = hintMaps(p);
  const undo = {
    show: practiceMode && !msg.result,
    enabled: practiceMode && canUndo && !locked && !paused && !msg.thinking && !msg.result,
  };
  const replayLike = IS_REPLAY || inSolution;
  app.classList.toggle("replay", replayLike);
  if (focusUid && !findCard(focusUid)) focusUid = null;
  const yourTurn = v.turn_player === "you";

  let h = barHTML(v.bot, "bot", {
    thinking: msg.thinking, active: !yourTurn, name: replayName("bot"),
    pauseShow: !msg.room && !IS_REPLAY && !inSolution && !msg.result, paused, noMenu: EMBED,
  });
  if (paused) h += '<div id="pausebanner" data-key="pausebanner" role="status">Paused · tap Resume to continue</div>';
  if (whatIf) h += '<div id="whatifbanner" data-key="whatifbanner">What if · from turn ' + esc(String(whatIf.turn != null ? whatIf.turn : "?")) + "</div>";
  if (puzzle && puzzle.status === "playing") h += '<div id="puzzlebanner" data-key="puzzlebanner">Puzzle · ' + esc(String(puzzle.goal || "Win this turn")) + "</div>";
  if (IS_REPLAY) {
    // Spectator: seat 1's hand face-up under the top header (frames are reveal_all).
    h += '<div id="thand" data-key="thand">' + handHTML(v.bot.hand, {}, MOCK, null) + "</div>";
  }
  if (bannerText) h += '<div id="botbanner" data-key="botbanner">' + esc(bannerText) + "</div>";
  h += '<div id="board" data-key="board">';
  const zp = zonePicks(p, hm.byIndex);   // kept while locked: no flash of the list after a tap
  // The board's zone highlights are a decision surface: only the stage's active
  // decision shows them, so narration never sits over a glowing zone.
  const zpick = active === "zone" ? zp : null;
  h += '<div class="field bot">' + sideHTML(v.bot, {mirror: true, mock: MOCK, actionable: act, focus: focusUid, hint: hm.byUid, zpick: zpick && zpick.bot}) + "</div>";
  h += '<div id="mid"><div id="midstrip" class="zrow">' + midHTML(v, {mock: MOCK, actionable: act, focus: focusUid, names: msg.names || null, hint: hm.byUid, zpick: zpick && zpick.emz}) + "</div>";
  if (winChance && msg.winchance) h += winchanceHTML(msg.winchance);
  const links = cueChain || v.chain || [];
  const chain = chainHTML(links, MOCK, chainNeg, oppName(), cueChain ? cueSolving : 0);
  if (chain) {
    const n = links.length;
    // Older links show a sliver each; long chains tighten it so the newest always fits whole.
    const peek = n <= 1 ? 0 : Math.max(3, Math.min(9, Math.floor(36 / (n - 1))));
    h += '<div id="chain" style="--chain-peek:' + peek + 'px" role="button" tabindex="0" data-pile="chain:all" aria-label="' + esc("Chain, " + n +
      (n === 1 ? " link" : " links") + ". Show all") + '">' + chain + "</div>";
  }
  h += "</div>";
  h += '<div class="field you">' + sideHTML(v.you, {mirror: false, mock: MOCK, actionable: act, focus: focusUid, hint: hm.byUid, zpick: zpick && zpick.you}) + "</div>";
  h += "</div>";
  h += barHTML(v.you, "you", {phaseBtns: phaseBtnsHTML(p, locked || paused, hm.byIndex, false, null), active: yourTurn, name: replayName("you")});

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
    const isSheet = p && p.kind !== "idle" && p.kind !== "battle" && !zp;
    const practiceBar = !menuUid && p && (p.kind === "idle" || p.kind === "battle")
      ? hintBarHTML(p, true, true, undo, info) : "";
    const mainBar = menuUid || active !== "recap" ? contextBarHTML(p, zp) : recapHTML(hintFor(p));
    h += '<div id="info" data-key="info">' + practiceBar + mainBar + "</div>";
    h += '<div id="hand" data-key="hand" class="' + (locked ? "locked" : "") + '">' + handHTML(v.you.hand, act, MOCK, focusUid, hm.byUid) + "</div>";

    const pileShown = !!pileOpen && active === "pile";
    const sheetShown = !pileOpen && p && !msg.result && isSheet && !paused;
    sheet = pileShown ? pileSheetHTML(pileOpen, pileCards(pileOpen), MOCK)
      : (sheetShown
        ? sheetHTML(p, MOCK, sel, expanded, {hintByIndex: hm.byIndex, showHint: practiceMode && !puzzle, hintBusy: hintLoading || locked || !!msg.thinking, hintLoading, showUndo: undo.show, undoEnabled: undo.enabled, canCancel: !!(msg.can_cancel && !msg.result), cancelDisabled: locked || !!msg.thinking})
        : "");
    if (pileShown) h += '<div id="pscrim" data-key="pscrim" data-act="closepile"></div>';
    if (sheet) {
      const kind = pileShown ? "pile" : "sheet";
      const aside = active !== kind;
      const lab = pileShown ? ' role="dialog"' + (aside ? "" : ' aria-modal="true"') + ' tabindex="-1" aria-label="' + esc(pileTitle(pileOpen, msg.names || null)) + '"'
        : replayLike ? "" : ' role="region" aria-label="' + esc((p && p.text) || "Choose") + '"';
      h += '<div id="sheet" data-key="sheet" class="' + (locked && !pileShown ? "locked" : "") + (pileShown ? " pilesheet" : "") +
        (aside ? " stage-aside" : "") + '"' + lab + (aside ? ' inert aria-hidden="true"' : "") + ">" + sheet + "</div>";
    }
  }
  rememberStageFocus();
  const restore = saveScrolls(app, ["#sheet .sbody", "#info .ftext"]);
  morph(app, h);
  restore();
  app.classList.toggle("busy", locked);
  fit();
  markInfoOverflow();

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

  const logh = logHTML(shownEvents(), MOCK, oppName());
  const sideTabs = tabStripHTML(logTab, "s");
  const sideBody = tabPanelOpen("s", logTab) + sidePanelInner(logTab, logh) + "</div>";
  // The reader is never empty: the card you tapped, else the card of the last play.
  const rc = readerCard();
  const rt = rc && rc.code ? cardText(rc.code) : null;
  const rinfo = rc ? {card: rc, text: rt && rt.text, typeStr: rt && rt.type_str} : info;
  const showRecap = recap && active === "recap" && !IS_REPLAY;
  morph(logside,
    '<div class="sidecard" data-key="side">' + (showRecap ? recapHTML("") : rinfo && rinfo.card ? infoHTML(rinfo, null, MOCK, "")
    : '<p class="emptytx">Tap any card to read it here.</p>') + "</div>" +
    sideTabs + sideBody);
  morph(readside, '<p class="rlabel">' + (showRecap ? "This turn" : "Card") + "</p>" + (showRecap ? recapHTML("") : readerHTML(rc, rt)));
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

// The card for the reader: the one you tapped while it is still in the game, else the card of the last play, else
// the first card in your hand.
function readerCard() {
  const c = readCard && (findCard(readCard.uid) || readCard);
  if (c && !isHidden(c)) return c;
  for (let i = events.length - 1; i >= 0; i--) {
    const k = events[i].card;
    if (k && k.code && !isHidden(k)) return findCard(k.uid) || k;
  }
  // Nothing played yet (a duel's first turn): the first card in your hand.
  return ((msg && msg.view && msg.view.you.hand) || []).find((c) => c && c.code && !isHidden(c)) || null;
}

// The card reader column (wide screens): the card large, its name, type line and text.
function readerHTML(c, t) {
  if (!c) return '<p class="emptytx">Cards you tap show here.</p>';
  const meta = [t && t.type_str, statsLine(c)].filter(Boolean).join(" · ");
  return '<div class="rface">' + phHTML(c) + (!MOCK && c.code ? artImg(c.code, "full", 'alt=""') : "") + "</div>" +
    "<h2>" + esc(c.name || "Card") + "</h2>" + (meta ? '<div class="typeline">' + esc(meta) + "</div>" : "") +
    '<div class="rtext">' + kwHTML(esc((t && t.text) || (MOCK ? "Card text loads from the server in a live game."
      : texts.get(c.code) === "failed" ? "The card text did not load: reopen the duel to try again." : "Loading card text…"))) + "</div>";
}

function detailHTML() {
  if (!detail) return "";
  const c = detail;
  const t = cardText(c.code);
  const face = !MOCK && c.code
    ? '<div class="dface">' + phHTML(c) + artImg(c.code, "full", 'alt=""') + "</div>"
    : '<div class="dface">' + phHTML(c) + "</div>";
  const meta = [t && t.type_str, statsLine(c)].filter(Boolean).join(" · ");
  return '<div id="detail" data-key="detail" data-act="closedetail" role="dialog" aria-modal="true" tabindex="-1" aria-label="' +
    esc("Card details: " + (c.name || "hidden card")) + '">' +
    '<button class="dx" data-act="closedetail" aria-label="Close card details">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button>' + face +
    "<h2>" + esc(c.name || "Hidden card") + "</h2>" +
    (meta ? '<div class="typeline">' + esc(meta) + "</div>" : "") +
    '<div class="dtext">' + kwHTML(esc((t && t.text) || (MOCK ? "Card text loads from the server in a live game." : ""))) + "</div>" +
    '<div class="dclose" aria-hidden="true">Tap anywhere to close</div></div>';
}

function renderOverlays(logh) {
  const active = stage.active();
  let lh = "";
  if (logOpen && (active === "log" || active === "review")) {
    const body = tabPanelOpen("d", logTab) + sidePanelInner(logTab, logh) + "</div>";
    lh = '<div id="scrim" data-key="scrim" data-act="closelog"></div>' +
      '<div id="logdrawer" data-key="logdrawer" role="dialog" aria-modal="true" aria-label="This duel" tabindex="-1">' +
      '<div class="dhead">This duel<button class="iconbtn" data-act="closelog" aria-label="Close duel log">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
      tabStripHTML(logTab, "d") + body + "</div>";
  }
  if (logTab === "log") morphBottom(ovLog, lh, "#logdrawer .dbody");
  else morph(ovLog, lh);

  const mh = (menuUid && !locked) ? actionsHTML() : "";
  morph(ovMenu, mh);

  morph(ovGame, (primerOpen && active === "primer" ? primerSheetHTML() : "") +
    (gameMenu && active === "gamemenu" ? gameMenuHTML() : ""));

  morph(ovDetail, detail && active === "detail" ? detailHTML() : "");

  morph(ovKw, kwPop && (active === "detail" || active === "kw") ? kwPopHTML() : "");

  morph(ovSettings, settingsOpen && active === "settings" ? settingsHTML() : "");

  if (!revealLeaving) morph(ovReveal, revealCur && active === "reveal" ? revealHTML(revealCur) : "");

  let oh = "";
  if (IS_REPLAY && msg && msg.title && replayIdx === 0) {
    oh += '<div id="retitle" data-key="retitle"><div class="box"><div class="kicker">decklight replay</div>' +
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
  } else if (msg && msg.room && msg.view && msg.result) {
    oh = roomOverHTML(msg, room && room.ask);   // a room duel: rematch / back to the room / leave; no review
  } else if (msg && msg.view && msg.result && !overDismissed) {
    const missed = review ? (review.missed || 0) : 0;
    const overTuned = isTunedLabel(msg.model_label) ? " · " + tunedSuffix(msg.model_label) : "";
    const reviewLabel = reviewLoading
      ? ("Reviewing… " + (reviewProgress ? reviewProgress.done + "/" + reviewProgress.total : ""))
      : (review ? ("Review game" + overTuned + (missed ? " · " + missed + " missed response" + (missed === 1 ? "" : "s") : ""))
        : "Review game" + overTuned);
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
        (EMBED ? "" : '<div class="row"><button class="bigbtn primary" data-act="rematch"' +
        (starting ? ' disabled aria-busy="true">Starting…' : ">Rematch") + "</button>" + reviewBtn + "</div>") + "</div></div>";
    } else {
      const win = msg.result.winner === "you";
      const dis = starting ? " disabled aria-busy=\"true\"" : "";
      const rematchLabel = starting ? "Starting…" : "Rematch";
      oh = '<div id="gameover" data-key="gameover" role="alertdialog" aria-modal="true" tabindex="-1" aria-label="' + (win ? "Victory" : "Defeat") + '"><div class="box ' + (win ? "win" : "loss") + '">' +
        '<div class="kicker">Duel over · turn ' + msg.view.turn + "</div>" +
        "<h2>" + (win ? "Victory" : "Defeat") + "</h2>" +
        '<p class="reason">' + esc(msg.result.reason === "win" ? "" : msg.result.reason || "") + "</p>" +
        '<div class="score"><span class="you">You<b>' + msg.view.you.lp + "</b></span>" +
        '<span class="bot">decklight<b>' + msg.view.bot.lp + "</b></span></div>" +
        '<div class="row"><button class="bigbtn" data-act="newgame"' + dis + ">New game</button>" +
        '<button class="bigbtn primary" data-act="rematch"' + dis + ">" + rematchLabel + "</button></div>" +
        '<div class="row">' + reviewBtn + "</div></div></div>";
    }
  }
  morph(ovOver, oh);
  let rh = "";
  if (reviewFrame && reviewFrame.entry && active === "review") {
    const e = reviewFrame.entry;
    const opts = (reviewFrame.options || []).map((o) =>
      '<div class="opt' + (o.label === (e.your && e.your.label) ? " you" : "") +
      (o.label === (e.agent && e.agent.label) ? " agent" : "") + '">' +
      '<span class="tx"><span class="lb">' + esc(o.label || "") + "</span></span>" +
      (o.label === (e.your && e.your.label) ? '<span class="tag">You</span>' : "") +
      (o.label === (e.agent && e.agent.label) ? '<span class="tag">decklight’s pick</span>' : "") + "</div>").join("");
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
      "<span>decklight would play: " + esc((e.agent && e.agent.label) || "") + "</span></div>" +
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

// Product (r1 + search) vs r0 on the Forge archetypes (runs/full-r1-20261030-1609/fz_r1): Sky Striker 0.93 pooled
// (n 56, best of 11), Mitsurugi 0.86 (n 56; its gain held with search, Blitzclique's did not). Both lists are trained.
const DEFAULT_DECKS = {you: "mitsurugi-meta2", bot: "sky-striker-meta1"};
const GROUPS = {pool: "Pool", newcards: "New cards", tuned: "Fine-tuned", user: "Pasted", saved: "Yours", prebuilt: "Prebuilt"};

// Forge coverage of a lobby deck (server: deck_coverage.json). The bot side shows it: decks the Forge trained on (or
// that share most of their cards with one) play stronger than the rest.
const FORGE_TAG = {trained: "Trained", related: "Related"};
function forgeTagHTML(d, side) {
  if (side !== "bot" || !d || !d.forge) return "";
  return '<span class="ftag ' + d.forge.level + '">' + FORGE_TAG[d.forge.level] + "</span>";
}
function archName(id) {
  return String(id || "").split("-").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")
    .replace("Light And Darkness", "Light-and-Darkness");
}
function forgeNoteHTML(f) {
  if (!f) {
    return '<p class="prnote fnote">Not trained on this archetype: decklight plays it with its general skill only.</p>';
  }
  const a = esc(archName(f.arch));
  const ci = "+" + f.gain_ci[0].toFixed(0) + " to +" + f.gain_ci[1].toFixed(0);
  const gain = "On " + a + " lists that training added " + f.gain_pp.toFixed(0) + " points of win rate (" + ci +
    ", against the model before it).";
  if (f.level === "trained") {
    return '<p class="prnote fnote"><span class="ftag trained">Trained</span> decklight was trained on this exact list. ' +
      gain + "</p>";
  }
  return '<p class="prnote fnote"><span class="ftag related">Related</span> ' + Math.round(f.overlap * 100) +
    "% of its distinctive cards are in the " + a + " lists decklight was trained on. " + gain +
    " Not measured on this list.</p>";
}

function pickerHTML(decks, side, label) {
  // Opens the deck browser sheet (render/decks.js: search, filters, deck lists).
  const cur = lobbyPick[side];
  const d = decks.find((x) => x.id === cur);
  return '<section class="field-group"><div class="flabel">' + esc(label) + "</div>" +
    '<button class="pickbtn" data-pickopen="' + side + '" aria-haspopup="dialog">' +
    '<span class="dot ' + (side === "you" ? "you" : "bot") + '"></span>' +
    '<span class="nm">' + esc(deckName(decks, cur)) + "</span>" +
    forgeTagHTML(d, side) + (d ? '<span class="g">' + esc(GROUPS[d.group] || d.group) + "</span>" : "") +
    '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></button></section>';
}

function segHTML(label, key, items) {
  return '<section class="field-group"><div class="flabel" id="lb-' + key + '">' + esc(label) + '</div><div class="seg" role="radiogroup" aria-labelledby="lb-' + key + '">' +
    items.map(([k, t]) => '<button role="radio" aria-checked="' + (lobbyPick[key] === k) + '" class="' +
      (lobbyPick[key] === k ? "sel" : "") + (t.length > 10 ? " wide" : "") + '" data-' + key + '="' + k + '">' + t + "</button>").join("") +
    "</div></section>";
}

// decklight's model (lobby): base + per-deck fine-tunes from the server.
// Hidden entirely when the lobby lists no non-base model.
function lobbyModels() {
  return (lobby && Array.isArray(lobby.models) && lobby.models.length) ? lobby.models : [{id: "base", label: "Base model", kind: "base"}];
}
function modelById(id) {
  return lobbyModels().find((m) => m.id === id) || null;
}
function ensureModelPick() {
  const models = lobbyModels();
  const ids = new Set(models.map((m) => m.id));
  const def = (lobby && lobby.model_default && ids.has(lobby.model_default)) ? lobby.model_default : "base";
  if (!lobbyPick.model || !ids.has(lobbyPick.model)) {
    lobbyPick.model = (savedModel && ids.has(savedModel)) ? savedModel : def;
  }
}
function modelTag(m) {
  if (!m || m.kind === "base") return "all decks";
  if (m.date && m.hours != null) return m.date + " · " + m.hours + " h";
  return "custom";
}
function tunedSuffix(label) {
  // "Tuned for Branded" -> "tuned for Branded" for inline suffixes.
  return label ? label.charAt(0).toLowerCase() + label.slice(1) : "tuned model";
}
function isTunedLabel(label) {
  return !!label && label !== "Base model" && label !== "Custom model";
}
function modelRowHTML(m, botName) {
  const sel = lobbyPick.model === m.id;
  const sub = m.kind === "base"
    ? "Plays every deck · the released model"
    : (m.date && m.hours != null ? m.date + " · " + m.hours + " h of training · #" + (m.short || "?") : "Custom model");
  void botName;
  return '<button role="option" aria-selected="' + sel + '" class="deckrow' + (sel ? " sel" : "") + '" data-model="' + esc(m.id) + '">' +
    '<span class="col"><span class="nm">' + esc(m.label) + '</span><span class="sub">' + esc(sub) + "</span></span></button>";
}
function modelPickerHTML() {
  const models = lobbyModels();
  if (!models.some((m) => m.kind && m.kind !== "base")) return "";
  ensureModelPick();
  const decks = (lobby && lobby.decks) || [];
  const botName = deckName(decks, lobbyPick.bot);
  const cur = modelById(lobbyPick.model) || models[0];
  const open = pickerOpen === "model";
  const randomMode = lobbyPick.mode === "random";
  let h = '<section class="field-group"><div class="flabel">decklight\'s model</div>';
  h += '<button class="pickbtn' + (open ? " open" : "") + (randomMode ? " dim" : "") + '" data-pickopen="model" aria-expanded="' + open + '"' +
    (randomMode ? ' aria-disabled="true"' : "") + ">" +
    '<span class="nm">' + esc(cur.label) + "</span>" +
    '<span class="g">' + esc(randomMode ? "not used by Random" : modelTag(cur)) + "</span>" +
    '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></button>';
  if (open && !randomMode) {
    const mine = models.filter((m) => m.kind !== "base" && m.deck_id && m.deck_id === lobbyPick.bot);
    const rest = models.filter((m) => m.kind !== "base" && !(m.deck_id && m.deck_id === lobbyPick.bot));
    h += '<div class="pickpanel"><div class="decklist" role="listbox" aria-label="decklight\'s model">';
    if (mine.length) {
      h += '<div role="group" aria-label="For ' + esc(botName) + '"><div class="grp">For ' + esc(botName) + "</div>" +
        mine.map((m) => modelRowHTML(m, botName)).join("") + "</div>";
    }
    h += '<div role="group" aria-label="General"><div class="grp">General</div>' +
      modelRowHTML(models.find((m) => m.kind === "base") || models[0], botName) + "</div>";
    if (rest.length) {
      h += '<div role="group" aria-label="Tuned for other decks"><div class="grp">Tuned for other decks</div>' +
        rest.map((m) => modelRowHTML(m, botName)).join("") + "</div>";
    }
    h += '</div><div class="modelsub">The bot, hints, win chance and review all use this model.</div></div>';
  }
  h += modelNoteHTML();
  return h + "</section>";
}
// Status / warning / availability line under the model field.
function modelNoteHTML() {
  if (modelNotice) return '<div class="mnote" role="status">' + esc(modelNotice) + "</div>";
  const models = lobbyModels();
  if (!models.some((m) => m.kind && m.kind !== "base")) return "";
  ensureModelPick();
  const decks = (lobby && lobby.decks) || [];
  const cur = modelById(lobbyPick.model);
  if (!cur) return "";
  if (cur.kind !== "base" && cur.deck_id !== lobbyPick.bot) {
    const onName = deckName(decks, lobbyPick.bot);
    return '<div class="mnote warn" role="status"><svg viewBox="0 0 20 20" aria-hidden="true">' +
      '<path d="M10 3 L18 17 H2 Z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>' +
      '<path d="M10 8v4M10 14.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>' +
      esc(cur.label + ". On " + onName + " it is untested and may play worse than the base model.") +
      ' <button class="linkbtn" data-act="usebase">Use base model</button></div>';
  }
  if (cur.kind === "base") {
    const tune = models.filter((m) => m.kind !== "base" && m.deck_id && m.deck_id === lobbyPick.bot)
      .sort((a, b) => String(b.date || "").localeCompare(String(a.date || "")))[0]
      || models.find((m) => m.kind !== "base" && m.deck_id === lobbyPick.bot);
    if (tune) {
      return '<div class="mnote">A model tuned for ' + esc(tune.deck_name || "this deck") + " is available. " +
        '<button class="linkbtn" data-act="usetuned">Use it</button></div>';
    }
  }
  return "";
}

// Puzzles tab (user 2026-11-03: a tab like the other two; "the puzzle search is not good, it is just text and it does
// not show the user the possibilities … use the tags to click in the search to filter just for these, like in a
// pill"). Filter chips per facet (OR inside a facet, AND across facets) with live counts, the decks as chips filtered
// by the search text, and free text over everything.
const PZ_FACETS = [
  ["diff", "Difficulty", [["easy", "Easy"], ["medium", "Medium"], ["hard", "Hard"]], (p) => p.difficulty || "medium"],
  ["prog", "Progress", [["todo", "Unsolved"], ["done", "Solved"]], (p) => (isSolved(p.id) ? "done" : "todo")],
  ["len", "Length", [["s", "1–4 plays"], ["m", "5–8 plays"], ["l", "9+ plays"]],
    (p) => ((p.n_plays || 0) <= 4 ? "s" : (p.n_plays || 0) <= 8 ? "m" : "l")],
];
const PZ_DECKS_SHOWN = 8;
const pzMine = (p) => p.decks[p.seat === 1 ? 1 : 0];
const pzTheirs = (p) => p.decks[p.seat === 1 ? 0 : 1];
function pzText(p) {
  return [pzMine(p), pzTheirs(p), p.goal || "", "turn " + p.turn].join(" ").toLowerCase();
}
// Puzzles that pass every filter except `skip` (a facet key, "deck" or "q"): the base for that facet's counts.
function pzMatch(list, skip) {
  const words = puzzleQ.toLowerCase().split(/\s+/).filter(Boolean);
  return list.filter((p) => {
    for (const [k, , , val] of PZ_FACETS) {
      if (k !== skip && puzzleF[k].size && !puzzleF[k].has(val(p))) return false;
    }
    if (skip !== "deck" && puzzleF.deck.size && !puzzleF.deck.has(pzMine(p)) && !puzzleF.deck.has(pzTheirs(p))) return false;
    if (skip !== "q" && words.length) { const t = pzText(p); if (!words.every((w) => t.includes(w))) return false; }
    return true;
  });
}
function pzActive() {
  return puzzleQ.trim() !== "" || PZ_FACETS.some(([k]) => puzzleF[k].size) || puzzleF.deck.size > 0;
}
function chipHTML(facet, key, label, n) {
  const on = facet === "deck" ? puzzleF.deck.has(key) : puzzleF[facet].has(key);
  return '<button class="chip' + (on ? " on" : "") + '" data-act="pzchip" data-f="' + facet + '" data-k="' + esc(key) +
    '" aria-pressed="' + on + '" aria-label="' + esc(label + ", " + n + (n === 1 ? " puzzle" : " puzzles")) + '"' +
    (!on && !n ? " disabled" : "") + ">" +
    (on ? '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 10.5l3.2 3L15 6.5"/></svg>' : "") +
    "<span>" + esc(label) + '</span><span class="n">' + n + "</span></button>";
}
function puzzlesPanelHTML() {
  const list = Array.isArray(puzzles) ? puzzles : [];
  const shown = pzMatch(list, null);
  let h = '<h2 class="sr-only">Puzzles</h2><p class="tabintro">Find the winning line from a position in a real bot game.</p>';
  h += '<section class="field-group pzfind"><label class="flabel" for="psearch">Search puzzles</label>' +
    '<input id="psearch" class="rinput" type="search" autocomplete="off" spellcheck="false" enterkeyhint="search" ' +
    'placeholder="A deck or a turn, e.g. Branded" aria-controls="pzresults" value="' + esc(puzzleQ) + '"></section>';
  // Facet chips: counts are what that chip would show given every other filter. Difficulty and Progress always show;
  // Length and the deck list fold under "More filters" so the puzzles stay near the top on a phone, except that
  // decks matching the search text always show (typing "snake" offers the Snake decks as pills).
  const q = puzzleQ.trim().toLowerCase();
  const hidden = puzzleF.len.size + puzzleF.deck.size;
  h += '<div class="pzfacets">';
  for (const [k, label, opts, val] of PZ_FACETS) {
    if (k === "len" && !puzzleMore) continue;
    const base = pzMatch(list, k);
    h += '<div class="pzfacet" role="group" aria-labelledby="pzf-' + k + '"><span class="pzflabel" id="pzf-' + k + '">' +
      label + '</span><div class="chips">' +
      opts.map(([key, l]) => chipHTML(k, key, l, base.filter((p) => val(p) === key).length)).join("") + "</div></div>";
  }
  // Decks: every deck in the puzzles, most puzzles first, narrowed by the search text; selected decks always show.
  const dbase = pzMatch(list, "deck");
  const counts = new Map();
  for (const p of dbase) for (const d of new Set([pzMine(p), pzTheirs(p)])) counts.set(d, (counts.get(d) || 0) + 1);
  const all = [...new Set(list.flatMap((p) => [pzMine(p), pzTheirs(p)]))];
  let decks = all.filter((d) => puzzleF.deck.has(d) || ((counts.get(d) || 0) > 0 && (!q || q.split(/\s+/).some((w) => d.toLowerCase().includes(w)))));
  decks.sort((a, b) => (puzzleF.deck.has(b) - puzzleF.deck.has(a)) || ((counts.get(b) || 0) - (counts.get(a) || 0)) || a.localeCompare(b));
  const more = decks.length - PZ_DECKS_SHOWN;
  if (!puzzleAllDecks && more > 0) decks = decks.slice(0, Math.max(PZ_DECKS_SHOWN, puzzleF.deck.size));
  if (all.length && (puzzleMore || q || puzzleF.deck.size)) {
    h += '<div class="pzfacet" role="group" aria-labelledby="pzf-deck"><span class="pzflabel" id="pzf-deck">' +
      (q && !puzzleMore ? "Matching decks" : "Decks") + "</span>" +
      '<div class="chips">' + (decks.length ? decks.map((d) => chipHTML("deck", d, d, counts.get(d) || 0)).join("")
        : '<span class="pznone">No deck matches the search.</span>') +
      (more > 0 ? '<button class="chip more" data-act="pzdecks" aria-expanded="' + puzzleAllDecks + '">' +
        (puzzleAllDecks ? "Fewer decks" : "All " + (decks.length + more) + " decks") + "</button>" : "") +
      "</div></div>";
  }
  h += '<button class="morebtn" data-act="pzmore" aria-expanded="' + puzzleMore + '">' +
    (puzzleMore ? "Fewer filters" : "More filters: length, decks" + (hidden ? " · " + hidden + " on" : "")) +
    '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></button>';
  h += "</div>";
  h += '<div class="pzcount"><p role="status" aria-live="polite">' +
    (puzzles === null ? "Loading puzzles…" : pzActive() ? shown.length + " of " + list.length + " puzzles"
      : list.length + " puzzles · " + list.filter((p) => isSolved(p.id)).length + " solved") + "</p>" +
    (pzActive() ? '<button class="linkbtn" data-act="pzclear">Clear filters</button>' : "") + "</div>";
  h += '<div id="pzresults">';
  if (puzzles !== null && !shown.length) {
    h += '<p class="emptytx">' + (list.length ? "No puzzle matches these filters." : "No puzzles yet.") + "</p>";
  }
  for (const [key, label] of PZ_FACETS[0][2]) {
    const rows = shown.filter((p) => (p.difficulty || "medium") === key);
    if (!rows.length) continue;
    h += '<section class="field-group" aria-labelledby="pz-' + key + '"><h3 class="flabel" id="pz-' + key + '">' +
      esc(label) + ' <span class="g">' + rows.length + "</span></h3>" +
      '<div class="plist" role="list">' + rows.map((p) => {
        return '<div role="listitem"><button class="prow" data-act="puzzlestart" data-id="' + esc(p.id) + '">' +
          '<span class="pt"><span class="ptitle">' + esc(pzMine(p)) + ' <span class="pvs">vs</span> ' + esc(pzTheirs(p)) +
          "</span>" + '<span class="psub">' + esc("Turn " + p.turn + " · " + (p.n_plays || "?") + " plays" +
            (p.goal ? " · " + p.goal : "")) + "</span></span>" +
          (isSolved(p.id) ? '<span class="psolved">Solved</span>' : "") +
          "</button></div>";
      }).join("") + "</div></section>";
  }
  return h + "</div>";
}

// Opponent deck primer (step 5, item 7). One sheet for the lobby and the game
// menu; the header (title + close) stays put while the body scrolls.
function primerCardHTML(code, name) {
  if (!code) return esc(name);
  return '<button class="pname" data-primercode="' + esc(code) + '" data-primername="' +
    esc(name) + '">' + esc(name) + "</button>";
}

function primerSheetHTML() {
  if (!primerOpen) return "";
  let body = primerForge === undefined ? "" : forgeNoteHTML(primerForge);
  if (primerLoading) {
    body = '<p class="emptytx">Loading the primer…</p>';
  } else if (!primer) {
    body = '<p class="emptytx">Not enough bot games with this deck yet.</p>';
    if (canStudy) body += studyButtonHTML();
  } else {
    const p = primer;
    body += '<p class="prsub">' + esc(p.archetype || "Unknown") + " · from " + esc(String(p.games || 0)) +
      " bot games</p>";
    if (p.similar) {
      body += '<p class="prnote">Based on a similar ' + esc(p.similar.archetype || "") + " list (" +
        esc(String(p.similar.match || 0)) + " of " + esc(String(p.similar.of || 0)) +
        " main-deck cards match).</p>";
      body += studyButtonHTML();
    }
    if (p.study && p.study.phase === "games") {
      body += '<p class="prnote">First version: ranked by how often cards are played. ' +
        "The win-chance ranking is still being computed.</p>";
    }
    const scored = (p.key_cards || []).some((c) => c.impact != null);
    if (scored) body += '<p class="prnote">Ranked by how much the bot\'s win-chance estimate moves.</p>';
    if ((p.key_cards || []).length) {
      body += "<h3>Key cards</h3><ul class=\"prlist\">" + p.key_cards.map((c) => {
        const n = Math.round((c.per_game || 0) * 10);
        if (c.impact == null) {
          return "<li>" + primerCardHTML(c.code, c.name) + ' <span class="prmeta">in ' + n + " of 10 games</span></li>";
        }
        const s = (c.impact >= 0 ? "+" : "") + Number(c.impact).toFixed(1);
        return "<li>" + primerCardHTML(c.code, c.name) + " " + esc(s) + " pts a game<br>" +
          ' <span class="prmeta">in ' + n + " of 10 games</span></li>";
      }).join("") + "</ul>";
    }
    if ((p.first_plays || []).length) {
      body += '<h3>How it opens</h3><p class="prnote">Going first, the card it plays first</p>' +
        '<ul class="prlist">' + p.first_plays.map((c) => {
          const n = Math.round((c.share || 0) * 10);
          const meta = n <= 0 ? "first in under 1 of 10 games" : "first in " + n + " of 10 games";
          return "<li>" + primerCardHTML(c.code, c.name) + ' <span class="prmeta">' + meta + "</span></li>";
        }).join("") + "</ul>";
    }
    // "Where it gets stopped" is hidden (user, 2026-10-18): neither the frequency nor the value-head chokepoint
    // ranking predicted anything on held-out games (NOTEBOOK 2026-10-18 primer comparison).
    body += "<h3>Your answers</h3>" +
      '<p class="prnote">Hand traps and negates in your deck, and what they were used on against this deck in bot games.</p>';
    const answers = p.your_answers || [];
    if (!answers.length) {
      body += '<p class="prnote">No hand traps or negates in your deck were seen against this deck.</p>';
    } else {
      body += '<ul class="prlist">' + answers.map((a) => {
        const hits = a.hits || [];
        if (hits.length) {
          return "<li>" + primerCardHTML(a.code, a.name) + " — used on " +
            hits.map((h) => primerCardHTML(h.code, h.name)).join(", ") + "</li>";
        }
        return "<li>" + primerCardHTML(a.code, a.name) + " — not used against this deck in bot games</li>";
      }).join("") + "</ul>";
    }
  }
  const title = primerDeck || "Opponent's deck";
  return '<div id="prscrim" data-key="prscrim" data-act="closeprimer"></div>' +
    '<div id="primersheet" data-key="primersheet" role="dialog" aria-modal="true" tabindex="-1" data-close="closeprimer" aria-label="' +
    esc("Deck primer: " + title) + '">' + '<div class="grab" aria-hidden="true"></div>' +
    '<div class="mhead"><span class="mtitle">' + esc(title) + "</span>" +
    '<button class="iconbtn" data-act="closeprimer" aria-label="Close deck primer">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' +
    '<div class="mbody primer">' + body + "</div></div>";
}

// Deck study on demand: the button (or the live status while it runs) under the
// similar-list note and the not-enough-games text.
function studyForSheet() {
  return studyState && primerHash && studyState.hash === primerHash ? studyState : null;
}

function studyButtonHTML() {
  const st = studyForSheet();
  if (st) {
    if (st.phase === "failed") {
      return '<p class="prnote" role="status">Study failed. Try again.</p>' +
        '<button class="bigbtn" data-act="study">Study this deck</button>';
    }
    if (st.phase === "scoring" || st.phase === "games" || st.phase === "done") {
      const done = st.done || 0, total = st.total || 32;
      const line = st.phase === "scoring"
        ? "Ranking cards… " + done + " of " + total + " games"
        : "Studying… " + done + " of " + total + " games";
      return '<div class="sline" aria-hidden="true"></div>' +
        '<p class="prstudy" role="status" aria-live="polite">' + esc(line) + "</p>";
    }
    return "";
  }
  return '<button class="bigbtn" data-act="study">Study this deck</button>' +
    '<p class="prnote">Plays about 32 bot games with this list on this computer. ' +
    "A first version is ready in about a minute; the full ranking takes about 10 minutes.</p>";
}

function sendPrimerRequest() {
  if (MOCK || !ws) return;
  if (msg && msg.t === "lobby") {
    const el = document.getElementById("ydk");
    const ydk = ((el && el.value) || "").trim();
    ws.send({t: "primer", deck: lobbyPick.bot, my_deck: ydk ? {ydk} : lobbyPick.you});
  } else {
    ws.send({t: "primer"});
  }
}

function sendStudyRequest() {
  studyState = {t: "study_status", phase: "games", done: 0, total: 32, deck: primerDeck, hash: primerHash};
  render();
  if (MOCK || !ws) return;
  if (msg && msg.t === "lobby") ws.send({t: "study", deck: lobbyPick.bot});
  else ws.send({t: "study"});
}

// Loading screen shown at once after Start / Rematch / puzzle start, instead of
// leaving the lobby up with no feedback (the first state can take many seconds:
// the Full bot's setup plus its whole first turn). role=status + aria-live so
// screen readers hear it; motion is one slow hairline, none under reduced motion.
function startingHTML() {
  const decks = (lobby && lobby.decks) || [];
  const youN = deckName(decks, lobbyPick.you);
  const botN = deckName(decks, lobbyPick.bot);
  const modeN = {product: "Full", fast: "Fast", random: "Random"}[lobbyPick.mode] || "Full";
  const sm = modelById(lobbyPick.model);
  const modeTuned = (sm && sm.kind !== "base" && lobbyPick.mode !== "random") ? " · " + tunedSuffix(sm.label) : "";
  const think = msg && msg.t === "state" && msg.thinking && typeof msg.thinking.seconds === "number"
    ? '<div class="sthink">The bot is thinking… ' + Math.round(msg.thinking.seconds) + " s</div>"
    : '<div class="sthink dim">Waiting for the bot…</div>';
  return '<div id="starting" role="status" aria-live="polite">' +
    '<div class="sline" aria-hidden="true"></div>' +
    '<div class="kicker">Setting up the duel</div>' +
    '<div class="sdesc">decklight (' + esc(modeN + modeTuned) + ") · " + esc(botN) + " vs your " + esc(youN) + "</div>" +
    think + "</div>";
}

// Card packs (A5): the active pack, its counts, and the manual update/rollback actions. The update source
// is the server's --pack-url; with no source the panel explains that updates are not configured yet.
function packPanelHTML() {
  const p = (lobby && lobby.packs) || null;
  const url = (lobby && lobby.pack_update_url) || null;
  const state = p && p.state ? p.state : "stock";
  const count = (n, one, many) => n ? n + " " + (n === 1 ? one : many) : "";
  let body;
  if (state === "stock") {
    body = '<p class="packnote">Using the built-in cards.</p>';
  } else {
    const bits = [count(p.cards, "card", "cards"), count(p.scripts, "script", "scripts"),
      count(p.embeddings, "embedding", "embeddings")].filter(Boolean).join(" · ");
    body = '<p class="packid">' + esc(p.id || "card pack") + "</p>" +
      (bits ? '<p class="packnote">' + esc(bits) + "</p>" : "") +
      (p.created ? '<p class="packnote">Installed ' + esc(p.created) + "</p>" : "") +
      (state === "rolled_back" ? '<p class="packnote warn">The newest pack was rejected; using the previous one.</p>' : "") +
      (p.restart_required ? '<p class="packnote warn">Restart the server to use this pack.</p>' : "");
  }
  const update = url
    ? '<button class="pickbtn packrow" data-act="packupdate"' + (packBusy ? ' disabled aria-busy="true"' : "") +
      '><span class="nm">Check for updates</span><span class="g">' + (packBusy ? "Checking…" : "Online") + "</span></button>"
    : '<p class="packnote">No update source is configured yet.</p>';
  const rollback = p && p.previous_id
    ? '<button class="linkbtn packrollback" data-act="packrollback"' + (packBusy ? " disabled" : "") +
      ">Roll back to " + esc(p.previous_id) + "</button>"
    : "";
  return '<section class="field-group packpanel" aria-labelledby="lb-pack">' +
    '<h3 class="flabel" id="lb-pack">Card pack</h3>' +
    '<div role="status" aria-live="polite">' + body + "</div>" + update + rollback +
    '<div class="modelsub">New cards arrive as a versioned pack; images are fetched separately.</div></section>';
}

// Rock-paper-scissors against the bot, shown over the lobby after Start duel
// (user 2026-11-12: the "Rock-paper-scissors" first-player choice was ignored).
// Same markup/words as the multiplayer room's RPS (render/room.js HANDS).
const RPS_NAME = {1: "Scissors", 2: "Rock", 3: "Paper"};
function rpsOutcome(mine, theirs) {
  if (!mine || !theirs || mine === theirs) return "draw";
  return (mine === 2 && theirs === 1) || (mine === 3 && theirs === 2) || (mine === 1 && theirs === 3) ? "you" : "bot";
}
// Uniform 1–3 with crypto; rejection sampling keeps the three hands equal.
function rpsBotHand() {
  if (MOCK && rpsTestHand != null) return rpsTestHand;
  const b = new Uint8Array(1);
  for (;;) {
    crypto.getRandomValues(b);
    if (b[0] < 252) return (b[0] % 3) + 1;
  }
}
function rpsHandHTML(hand, label) {
  return '<span class="rpshand">' + HAND_ICON[hand] + "<span>" +
    label + ": " + esc(RPS_NAME[hand] || "") + "</span></span>";
}
function rpsHTML() {
  if (!rpsGame) return "";
  let body = "";
  if (rpsGame.phase === "pick") {
    body = '<p class="rnote">The winner chooses who goes first.</p>' +
      '<div class="hands" role="group" aria-labelledby="rps-title">' +
      HANDS.map(([v, n]) => '<button class="rbtn quiet" data-act="rpspick" data-hand="' + v + '"' +
        (v === 2 ? " data-autofocus" : "") + '>' + HAND_ICON[v] + "<span>" + n + "</span></button>").join("") + "</div>";
  } else {
    const out = rpsOutcome(rpsGame.mine, rpsGame.theirs);
    let result, actions;
    if (out === "draw") {
      result = "Draw — again";
      actions = '<button class="rbtn" data-act="rpsagain" data-autofocus>Throw again</button>';
    } else if (out === "you") {
      result = "You win";
      actions = '<div class="hands two"><button class="rbtn" data-act="rpsfirst" data-first="me" data-autofocus>Go first</button>' +
        '<button class="rbtn quiet" data-act="rpsfirst" data-first="bot">Go second</button></div>';
    } else {
      result = "Decklight wins";
      actions = '<p class="rnote">Decklight goes first.</p>' +
        '<button class="rbtn" data-act="rpsbotfirst" data-autofocus>Start duel</button>';
    }
    body = '<div class="rpshands">' + rpsHandHTML(rpsGame.mine, "You") + rpsHandHTML(rpsGame.theirs, "Decklight") + "</div>" +
      '<p class="rpsresult" role="status">' + esc(result) + "</p>" + actions;
  }
  return '<div id="rps" data-key="rps" role="dialog" aria-modal="true" aria-labelledby="rps-title" tabindex="-1">' +
    '<div class="rpsbox"><h2 id="rps-title">Rock, paper, scissors</h2>' + body + "</div></div>";
}

function renderLobby() {
  syncStage();
  const active = stage.active();
  const decks = (lobby && lobby.decks) || [];
  // First visit: decks the Forge trained on that the product plays best (user 2026-11-01: "leave decks that were fine
  // tuned and have very good performance"; NOTEBOOK 2026-11-01 "Default lobby decks"). Fallback: the first two.
  const has = (id) => decks.some((x) => x.id === id);
  // Never default to a deck that cannot be played (a saved deck with problems, e.g. Side 18/15).
  const ok = decks.filter((d) => d.legal !== false);
  if (!lobbyPick.you && ok.length) lobbyPick.you = has(DEFAULT_DECKS.you) ? DEFAULT_DECKS.you : ok[0].id;
  if (!lobbyPick.bot && ok.length) {
    lobbyPick.bot = has(DEFAULT_DECKS.bot) ? DEFAULT_DECKS.bot : ok[Math.min(1, ok.length - 1)].id;
  }
  if (room) {
    // In a room and no duel on screen: the waiting room, rock-paper-scissors, or the room's error.
    finishLobby(roomHTML(room, roomUi));
    return;
  }
  if (starting) {
    const h = '<div id="lobby"><div class="lscroll">' + startingHTML() +
      '</div><div id="startwrap"><button id="startbtn" data-act="start" disabled aria-busy="true">Starting…</button></div></div>';
    app.innerHTML = h;
    app.classList.remove("busy");
    logside.innerHTML = '<div class="loghead">Duel log</div><div class="loglist"><p class="emptytx">No duel yet.</p></div>';
    readside.innerHTML = "";
    morph(ovLog, "");
    morph(ovMenu, "");
    morph(ovGame, (primerOpen && active === "primer" ? primerSheetHTML() : "") +
      (deckBrowser && active === "deck" ? deckSheet() : ""));
    morph(ovDetail, detail && active === "detail" ? detailHTML() : "");
    morph(ovReveal, "");
    morph(ovOver, "");
    morph(ovSettings, settingsOpen && active === "settings" ? settingsHTML() : "");
    syncOv();
    return;
  }
  // The deck editor takes the whole screen (its own header and back button).
  if (lobbyTab === "decks" && db.editing()) { finishLobby(db.fullView()); return; }
  // Lobby (user 2026-10-31: "Essentials + Options"): decks, mode and Start first; the rarely changed settings
  // fold into Options; Puzzles is a tab; pasting a .ydk lives in the "Your deck" picker.
  let h = '<div id="lobby"><div class="lscroll">';
  h += '<header class="lhead"><div class="lmark" aria-hidden="true">' +
    '<svg viewBox="70 145 370 255"><title>decklight mark</title>' +
    '<defs><radialGradient id="dl-glow"><stop offset="0" stop-color="#b89a5e" stop-opacity="0.5"/>' +
    '<stop offset="1" stop-color="#b89a5e" stop-opacity="0"/></radialGradient>' +
    '<radialGradient id="dl-cardglow"><stop offset="0" stop-color="#b89a5e" stop-opacity="0.10"/>' +
    '<stop offset="1" stop-color="#b89a5e" stop-opacity="0"/></radialGradient></defs>' +
    '<circle cx="337.2" cy="276.3" r="175.0" fill="url(#dl-cardglow)"/>' +
    '<g transform="rotate(-20 256.0 499.5)"><rect x="191.0" y="167.0" width="130.0" height="190.0" rx="12.5" class="e1" fill="#141310" stroke="#7a756a" stroke-width="4.0"/></g>' +
    '<g transform="rotate(0 256.0 499.5)"><rect x="191.0" y="167.0" width="130.0" height="190.0" rx="12.5" class="e2" fill="#141310" stroke="#958f82" stroke-width="4.0"/></g>' +
    '<g transform="rotate(20 256.0 499.5)"><rect x="191.0" y="167.0" width="130.0" height="190.0" rx="12.5" class="e3" fill="#141310" stroke="#d6cfc0" stroke-width="5.0"/></g>' +
    '<polyline class="pth" points="158.5,315.3 227.4,260.1 357.2,221.4" fill="none" stroke="#b89a5e" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>' +
    '<circle class="nd" cx="158.5" cy="315.3" r="7.5" fill="#141310" stroke="#b89a5e" stroke-width="3.8"/>' +
    '<circle class="nd" cx="227.4" cy="260.1" r="7.5" fill="#141310" stroke="#b89a5e" stroke-width="3.8"/>' +
    '<circle cx="357.2" cy="221.4" r="65.0" fill="url(#dl-glow)"/>' +
    '<circle class="st" cx="357.2" cy="221.4" r="13.8" fill="#b89a5e"/>' +
    "</svg></div>" +
    '<h1>decklight</h1>' +
    '<button class="iconbtn lgear" data-act="opensettings" aria-label="Settings">' +
    // Lucide "settings" (lucide.dev), ISC licence.
    '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>' +
    '<circle cx="12" cy="12" r="3"/></svg></button>' +
    "</header>";
  if (lobbyTab === "people" || lobbyTab === "play" || lobbyTab === "games") {
    const sub = (k, label) => '<button data-act="lobbytab" data-tab="' + k + '"' +
      (lobbyTab === k ? ' class="sel" aria-current="page"' : "") + ">" + label + "</button>";
    h += '<nav class="lsub" aria-label="Play">' + sub("people", "Multiplayer") + sub("play", "Bot") + sub("games", "Games") + "</nav>";
  }
  if (startupPending || startupFailed) {
    h += '<p class="lstatus' + (startupFailed ? " fail" : "") + '" id="startnote" role="' +
      (startupFailed ? "alert" : "status") + '">' +
      esc(startupFailed || "Loading the bot…") + "</p>";
  }
  if (lobbyTab === "ask") {
    h = h.replace('<div id="lobby">', '<div id="lobby" class="askview">');
    h += ask.panelHTML() + "</div>" + ask.footHTML() + ask.sheetHTML() + "</div>";
    finishLobby(h);
    return;
  }
  if (lobbyTab === "games") {
    h += games.panelHTML() + "</div></div>";
    finishLobby(h);
    return;
  }
  if (lobbyTab === "decks") {
    h += '<div data-feature="deckbuilder">' + db.panelHTML() + "</div>";
    if (MOCK) h += '<div class="mocknav">mock ' + (mockIdx + 1) + "/" + MOCKS.length + " · ← → to step</div>";
    h += "</div></div>";
    finishLobby(h);
    return;
  }
  if (lobbyTab === "puzzles") {
    h += puzzlesPanelHTML();
    if (MOCK) h += '<div class="mocknav">mock ' + (mockIdx + 1) + "/" + MOCKS.length + " · ← → to step</div>";
    h += "</div></div>";
    finishLobby(h);
    return;
  }
  const paste = pasteOpen
    ? '<div class="pastebox"><label for="ydk">Deck list (.ydk), used instead of the deck above</label>' +
      '<textarea id="ydk" placeholder="#main&#10;14558128&#10;…" spellcheck="false"></textarea>' +
      '<button class="linkbtn" data-act="closepaste">Use the deck above instead</button></div>'
    : "";
  if (lobbyTab === "people") {
    h += peopleHTML({deck: pickerHTML(decks, "you", "Your deck"), paste, name: duelist, form: roomForm,
      rooms, scanning, scanErr, note: roomUi.note});
    if (MOCK) h += '<div class="mocknav">mock ' + (mockIdx + 1) + "/" + MOCKS.length + " · ← → to step</div>";
    h += "</div></div>";
    finishLobby(h);
    return;
  }
  h += '<h2 class="sr-only">Duel the bot</h2><p class="tabintro">Duel decklight, the bot, on this device.</p>';
  h += pickerHTML(decks, "you", "Your deck");
  h += paste;
  h += pickerHTML(decks, "bot", "decklight plays");
  h += '<div class="aboutrow"><button class="linkbtn" data-act="primer">About this deck</button></div>';
  h += '<section class="field-group"><div class="flabel" id="lb-play">Mode</div>' +
    '<div class="modecards" role="radiogroup" aria-labelledby="lb-play">' +
    [["duel", "Duel", "A normal game. Counts in your stats; no hints or take-backs."],
     ["practice", "Practice", "Hints from the bot, undo and what-if replays. Not counted."]]
      .map(([k, t, d]) => '<button role="radio" aria-checked="' + (lobbyPick.play === k) + '" class="modecard' +
        (lobbyPick.play === k ? " sel" : "") + '" data-play="' + k + '"><span class="mt">' + t +
        '</span><span class="md">' + d + "</span></button>").join("") + "</div></section>";
  const firstLbl = {rps: "Rock-paper-scissors", me: "You go first", bot: "Bot goes first"}[lobbyPick.first] || "";
  const modeLbl = {product: "Full strength", fast: "Fast", random: "Random"}[lobbyPick.mode] || "";
  const curModel = modelById(lobbyPick.model);
  const optSum = [firstLbl, modeLbl, curModel && curModel.kind !== "base" ? curModel.label : ""].filter(Boolean).join(" · ");
  h += '<section class="optsec"><button class="optbtn' + (optsOpen ? " open" : "") + '" data-act="opts" aria-expanded="' +
    optsOpen + '" aria-controls="optbody"><span class="ot"><span class="nm">Options</span><span class="os">' +
    esc(optSum) + '</span></span><svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg></button>';
  if (optsOpen) {
    h += '<div id="optbody">';
    h += segHTML("Going first", "first", [["rps", "Rock-paper-scissors"], ["me", "Me"], ["bot", "Bot"]]);
    h += segHTML("Bot strength", "mode", [["product", "Full"], ["fast", "Fast"], ["random", "Random"]]);
    h += modelPickerHTML();
    h += "</div>";
  }
  h += "</section>";
  if (MOCK) h += '<div class="mocknav">mock ' + (mockIdx + 1) + "/" + MOCKS.length + " · ← → to step</div>";
  const notReady = startupPending || !!startupFailed;
  h += '</div><div id="startwrap"><button id="startbtn" data-act="start"' +
    (notReady ? ' disabled aria-describedby="startnote"' : "") + '>Start duel</button></div></div>';
  finishLobby(h);
}

function bottomBarHTML() {
  const cur = lobbyTab === "people" || lobbyTab === "games" ? "play" : lobbyTab;
  return '<nav class="bbar" aria-label="Sections">' + BAR_TABS.map(([k, label]) => '<button data-act="lobbytab" data-tab="' +
    (k === "play" ? playSub : k) + '"' + (cur === k ? ' class="sel" aria-current="page"' : "") + ">" + (barIcons[k] || "") +
    "<span>" + label + "</span></button>").join("") + "</nav>";
}

// The first screen is up: fade the first-paint mark out (web/index.html #boot); at once for screenshots and replays.
function hideBoot() {
  const b = document.getElementById("boot");
  if (!b || b.classList.contains("out")) return;
  if (MOCK || IS_REPLAY || matchMedia("(prefers-reduced-motion: reduce)").matches) { b.remove(); return; }
  b.classList.add("out");
  setTimeout(() => b.remove(), 400);
}

// Put a lobby screen (Bot, Multiplayer, Puzzles, or a room) on screen and clear the duel layers.
function finishLobby(h) {
  syncStage();
  const active = stage.active();
  hideBoot();
  // Every lobby click re-renders: keep the scroll position, the pasted .ydk and its open state.
  const oldYdk = document.getElementById("ydk");
  const ydkVal = oldYdk ? oldYdk.value : "";
  // The bottom bar (Play · Puzzles · Decks · Ask) closes every tab screen; not rooms, the deck editor or start-up.
  if (!room && !(lobbyTab === "decks" && db.editing()) && h.startsWith('<div id="lobby"') && !h.includes('aria-busy="true"')) {
    h = h.slice(0, -"</div>".length) + bottomBarHTML() + "</div>";
  }
  // Ask: an answer streaming in keeps the thread at its end, unless the user scrolled up to read.
  const sc0 = app.querySelector("#lobby.askview .lscroll");
  const atEnd = !!sc0 && sc0.scrollHeight - sc0.scrollTop - sc0.clientHeight < 80;
  const restoreScroll = saveScrolls(app, [".lscroll"]);
  // Patch the lobby in place when it is already shown (the scroller element survives, so no browser resets it;
  // a scrollTop restore alone did not hold on the user's phone, 2026-10-31); build it fresh otherwise.
  if (app.querySelector("#lobby .lscroll") && !app.querySelector("#startbtn[aria-busy]")) morph(app, h);
  else app.innerHTML = h;
  const newYdk = document.getElementById("ydk");
  if (newYdk && !newYdk.value) newYdk.value = ydkVal;
  restoreScroll();
  const sc1 = app.querySelector("#lobby.askview .lscroll");
  if (sc1 && ask.following() && atEnd) sc1.scrollTop = sc1.scrollHeight;
  app.classList.remove("busy");
  logside.innerHTML = '<div class="loghead">Duel log</div><div class="loglist"><p class="emptytx">No duel yet.</p></div>';
    readside.innerHTML = "";
  morph(ovLog, "");
  morph(ovMenu, "");
  morph(ovGame, (primerOpen && active === "primer" ? primerSheetHTML() : "") +
    (deckBrowser && active === "deck" ? deckSheet() : "") + (active === "dbsheet" ? db.sheetHTML() : ""));
  morph(ovDetail, detail && active === "detail" ? detailHTML() : "");
  morph(ovReveal, "");
  morph(ovOver, rpsGame && active === "rps" ? rpsHTML() : "");
  morph(ovSettings, settingsOpen && active === "settings" ? settingsHTML() : "");
  syncOv();
}

// ---- deck browser sheet (render/decks.js) ----
function deckSheet() {
  if (!deckBrowser) return "";
  const decks = (lobby && lobby.decks) || [];
  return deckSheetHTML(deckBrowser, decks, deckIndex, lobbyPick[deckBrowser.side], {art: cardArt && !MOCK, forgeNote: forgeNoteHTML});
}
function renderDeckSheet() {
  syncStage();
  const active = stage.active();
  morph(ovGame, (primerOpen && active === "primer" ? primerSheetHTML() : "") +
    (deckBrowser && active === "deck" ? deckSheet() : ""));
  syncOv();
  manageFocus();
}
let deckIndexLoading = false;
function loadDeckIndex() {
  // Refetched on every open: the server adds decks pasted this session; the last copy stays on screen meanwhile.
  if (deckIndexLoading) return;
  deckIndexLoading = true;
  if (deckIndex === undefined) deckIndex = null;
  const url = MOCK ? "mock/deck-index.json" : "/deck_index.json" + (TOKEN ? "?t=" + encodeURIComponent(TOKEN) : "");
  fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
    .then((j) => { deckIndex = j; })
    .catch(() => { if (!deckIndex) deckIndex = {decks: {}, names: {}}; })  // names-only browsing still works
    .finally(() => { deckIndexLoading = false; if (deckBrowser) renderDeckSheet(); });
}
// Mock: two saved decks from the deck builder (one not legal), as lobby_decks sends them; pool rows become prebuilt.
function mockSavedDecks(l) {
  if (!l.decks || l.decks.some((d) => d.group === "saved")) return l;
  // meta: the summary the server sends with each saved deck (same shape as a deck_index entry).
  l.decks = [{id: "user:blue-eyes-a1e314", name: "Blue-Eyes", group: "saved", user: true, legal: true, problems: [],
    meta: {"arch":"","origin":"Built in Decks","year":0,"mech":["Synchro"],"counts":[21,13,6],"sig":[62089826,89631139,11443677],"main":[[89631139,3],[38517737,2],[30576089,1],[64202399,1],[17947697,3],[8240199,3],[54332792,2],[97268402,3],[14558127,3],[80326401,3],[17725109,2],[62089826,2],[43219114,1],[71143015,1],[24382602,1],[48800175,3],[38120068,3],[10045474,3]],"extra":[[24361622,1],[42097666,3],[59822133,3],[89604813,3],[40908371,1],[11443677,1],[2129638,1],[56532353,1],[16699558,1]]}},
    {id: "user:branded-d666ba", name: "Branded", group: "saved", user: true, legal: false,
      problems: ["Side 18/15 · 3 too many"], meta: {"arch":"","origin":"Built in Decks","year":0,"mech":["Fusion"],"counts":[30,7,3],"sig":[33760966,68468459,44362883],"main":[[44482554,2],[70871153,2],[1498449,3],[75003700,3],[7375867,1],[84477320,1],[6153210,1],[32548318,3],[5431722,1],[69932023,1],[80208225,1],[73819701,2],[68468459,1],[55273560,1],[95515789,1],[44362883,2],[30271097,1],[33854624,1],[6637331,1],[42141493,3],[14558127,3],[73642296,3],[97268402,2]],"extra":[[33760966,3],[79755671,1],[89851827,2],[87746184,2],[41373230,1],[44146295,1],[78397661,1],[72578374,1],[42097666,1],[24361622,1],[29301450,1]]}}].concat(l.decks.map((d) => Object.assign({}, d,
      d.group === "pool" || d.group === "newcards" ? {group: "prebuilt"} : {})));
  return l;
}
function openDeckBrowser(side, el) {
  deckBrowser = {side, q: "", f: {}, sort: "name", view: null, top: 0};
  opener = describe(el) || opener;
  loadDeckIndex();
  renderDeckSheet();
}
function closeDeckBrowser() {
  deckBrowser = null;
  renderLobby();
  manageFocus();
}

function render() {
  // Static screenshots in replay mode: no entry animations, no transitions.
  document.documentElement.classList.toggle("replay", IS_REPLAY);
  games.paintWatch();
  const before = snapshotRects();
  if (!msg || msg.t === "lobby") {
    renderLobby();
    manageFocus();
    return;
  }
  if (msg.t === "state") {
    hideBoot();
    renderGame();
    if (!IS_REPLAY) flipAnimate(before);
    document.title = msg.prompt && msg.prompt.text ? msg.prompt.text + " · decklight"
      : msg.caption && msg.caption.text ? msg.caption.text + " · decklight" : "decklight";
    manageFocus();
  } else if (msg.t === "error") {
    banner.hidden = false;
    banner.textContent = msg.text || "error";
  }
}

// ---- events (delegated) ----
function cardTap(el, keyboard) {
  const uid = el.getAttribute("data-uid");
  const c = findCard(uid);
  if (el.classList.contains("actionable") && optionsForUid(uid).length && !locked) {
    const opening = menuUid !== uid;
    menuUid = opening ? uid : null;
    if (opening) opener = describe(el) || opener;   // focus returns to this card on close
    if (c && !isHidden(c)) { focusUid = uid; readCard = c; }
  } else if (c && !isHidden(c)) {
    // Not actionable: open the full detail (field monsters included, both sides); the reader on wide screens.
    menuUid = null;
    if (WIDE.matches) readCard = c; else detail = c;
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
      if (WIDE.matches) readCard = c; else detail = c;
      render();
      return true;
    }
    return true; // a card grid cell without a resolvable card: never choose
  }
  const uid = el.getAttribute("data-detailuid");
  if (uid) {
    const c = findCard(uid);
    if (c && !isHidden(c)) {
      if (WIDE.matches) readCard = c; else detail = c;
      render();
    }
    return true;
  }
  return false;
}

app.addEventListener("click", (ev) => {
  const t = ev.target;
  // Self-contained features (deck builder, describe panel) own their markup: their taps go straight to their actions,
  // so the lobby's attribute pickers below (data-mode, data-model, data-first, ...) can never swallow them
  // (user 2026-11-13: tapping "Describe a card" did nothing; its data-mode was read as the Bot tab's mode picker).
  const feat = t.closest("[data-feature]");
  if (feat) {
    const a = t.closest("[data-act]");
    if (a && feat.contains(a)) onAction(a.getAttribute("data-act"), a);
    return;
  }
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
  // Tapping cut-off effect text expands it; anywhere else (or text already shown whole) chooses. A fully shown line
  // that swallowed the tap made Mulcharmy Fuwalos look unclickable (user 2026-11-13).
  if (ef && (ef.scrollHeight > ef.clientHeight + 1 || ef.closest(".opt.open"))) {
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
    if (side === "model" && lobbyPick.mode === "random") return;  // not used by Random
    if (side !== "model") {
      openDeckBrowser(side, pick);
      return;
    }
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
      modelNotice = null;
      renderLobby();
      return;
    }
  }
  {
    const el = t.closest("[data-model]");
    if (el) {
      const m = modelById(el.getAttribute("data-model"));
      if (m && lobbyPick.mode !== "random") {
        lobbyPick.model = m.id;
        saveSetting("ygo-model", m.id);
        modelNotice = null;
        if (m.kind !== "base" && m.deck_id && m.deck_id !== lobbyPick.bot) {
          lobbyPick.bot = m.deck_id;
          const decks = (lobby && lobby.decks) || [];
          modelNotice = "decklight now plays " + deckName(decks, m.deck_id) +
            ", the deck this model was tuned for.";
        }
      }
      pickerOpen = null;
      renderLobby();
      return;
    }
  }
  for (const key of ["first", "mode", "play"]) {
    const el = t.closest("[data-" + key + "]");
    if (el) {
      lobbyPick[key] = el.getAttribute("data-" + key);
      modelNotice = null;
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
    cardTap(c, ev.detail === 0);
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

function openCodeDetail(code, name) {
  if (!code) return;
  detail = {uid: "code:" + code, code, name: name || ("#" + code),
    faceup: true, pos: null, atk: 0, def: 0, level: 0, rank: 0, link: 0,
    counters: 0, overlays: []};
  if (msg && msg.t === "state") {
    render();
    return;
  }
  // The lobby does not rebuild overlays in renderLobby, so paint the detail here
  // (and take the deck/primer sheet off stage: one lookup at a time).
  syncStage();
  const active = stage.active();
  morph(ovGame, (primerOpen && active === "primer" ? primerSheetHTML() : "") +
    (deckBrowser && active === "deck" ? deckSheet() : ""));
  morph(ovDetail, active === "detail" ? detailHTML() : "");
  syncOv();
  manageFocus();
}

function openStatDetail(code) {
  if (!code) return;
  detail = {uid: "stat:" + code, code, name: statName(code) || ("#" + code),
    faceup: true, pos: null, atk: 0, def: 0, level: 0, rank: 0, link: 0,
    counters: 0, overlays: []};
  if (logOpen) {  // the detail replaces the drawer (back closes the detail)
    logOpen = false;
  }
  render();
}

function onOverlayClick(ev) {
  const pc = ev.target.closest("[data-primercode]");
  if (pc) {
    openCodeDetail(parseInt(pc.getAttribute("data-primercode"), 10), pc.getAttribute("data-primername"));
    return;
  }
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
      if (logOpen) {  // the detail replaces the drawer (back closes the detail)
        logOpen = false;
      }
      render();
    }
    return;
  }
  if (onRecapRow(ev.target)) return;
  if (onDeckSheetClick(ev.target)) return;
  const act = ev.target.closest("[data-act]");
  if (act) onAction(act.getAttribute("data-act"), act);
}

// Deck browser sheet: filter chips, sort, and picking a deck (from a row or the list view's button).
function onDeckSheetClick(t) {
  if (!deckBrowser) return false;
  const dfl = t.closest("[data-dfilter]");
  if (dfl) {
    const [k, v] = dfl.getAttribute("data-dfilter").split(":");
    const cur = deckBrowser.f[k] || [];
    deckBrowser.f[k] = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
    renderDeckSheet();
    return true;
  }
  const dsr = t.closest("[data-dsrc]");
  if (dsr) {
    deckBrowser.src = dsr.getAttribute("data-dsrc");
    renderDeckSheet();
    return true;
  }
  const dso = t.closest("[data-dsort]");
  if (dso) {
    deckBrowser.sort = dso.getAttribute("data-dsort");
    renderDeckSheet();
    return true;
  }
  for (const key of ["youdeck", "botdeck"]) {
    const el = t.closest("#decksheet [data-" + key + "]");
    if (el) {
      lobbyPick[key === "youdeck" ? "you" : "bot"] = el.getAttribute("data-" + key);
      if (key === "youdeck") pasteOpen = false;
      modelNotice = null;
      closeDeckBrowser();
      return true;
    }
  }
  return false;
}
document.getElementById("layer").addEventListener("click", onOverlayClick);
logside.addEventListener("click", onOverlayClick);

function sendGame(o) {
  inLobby = false;
  if (MOCK) return;
  if (ws) ws.send(o);
}

// Reset the client and send new_game for the chosen first player ("me" or
// "bot"; the RPS dialog resolves to one of the same two values). Mock mode
// just shows the first board, as before.
function beginDuel(first, fromStart) {
  if (starting) return;
  if (MOCK) { loadMock(1); return; }
  const ydk = fromStart ? ((document.getElementById("ydk") || {}).value || "").trim() : "";
  resetGameUi();
  gameMode = lobbyPick.mode;
  ensureModelPick();
  gameModel = (modelById(lobbyPick.model) || {}).label || "Base model";
  practiceMode = lobbyPick.play === "practice";
  primerOpen = false;
  primer = null;
  primerLoading = false;
  primerDeck = "";
  primerHash = null;
  canStudy = false;
  resetCues();
  starting = true;
  render();
  sendGame({t: "new_game", deck: ydk ? {ydk} : lobbyPick.you, bot_deck: lobbyPick.bot,
    first, mode: lobbyPick.mode, chain: chainMode,
    practice: lobbyPick.play === "practice", model: lobbyPick.model});
}

// Push both client settings (the server keeps the win-chance switch per
// connection, off by default, so reconnects re-send it).
function sendSettings() {
  if (MOCK || !ws) return;
  ws.send({t: "settings", chain: chainMode, winchance: winChance});
}

function goLobby() {
  packBusy = false;
  msg = lobby || {t: "lobby", decks: []};
  detail = null;
  readCard = null;
  menuUid = null;
  rpsGame = null;
  pileOpen = null;
  gameMenu = false;
  settingsOpen = false;
  gameMode = null;
  gameModel = null;
  modelNotice = null;
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
  primerOpen = false;
  primer = null;
  primerLoading = false;
  primerDeck = "";
  primerHash = null;
  canStudy = false;
  studyState = null;
  inSolution = false;
  solutionFrames = null;
  solutionIdx = -1;
  liveEnd = null;
  hidePuzzleResult = false;
  inLobby = true;
  starting = false;
  resetCues();
  render();
}

// Per-game client state, cleared when a new game starts (bot game or a duel in a room).
function resetGameUi() {
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
  settingsOpen = false;
  paused = false;
  pauseQueue = [];
  stage.reset();
}

// ---- rooms (A2b-3c) ----
// Client-side check of a join address; the server parses it again (room.rs parse_address).
const ADDR_RE = /^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?)(?::(\d{1,5}))?(?:#\d{1,10})?$/;
function addressError(a) {
  if (!a) return "Type the room's address.";
  const mm = ADDR_RE.exec(a);
  if (!mm || (mm[2] && (+mm[2] < 1 || +mm[2] > 65535))) return "Use an address like 192.0.2.20 or name:7911.";
  return null;
}
function roomDeck() {
  const ydk = ((document.getElementById("ydk") || {}).value || "").trim();
  roomUi.deckLabel = ydk ? "Pasted deck" : deckName((lobby && lobby.decks) || [], lobbyPick.you);
  return ydk ? {ydk} : lobbyPick.you;
}
function sendRoom(o) {
  if (MOCK || !ws) return false;
  ws.send(o);
  return true;
}
function joinRoom(address) {
  const a = (address || "").trim();
  roomForm.address = a;
  roomForm.err = addressError(a);
  roomUi.note = null;
  if (!roomForm.err && sendRoom({t: "room_join", address: a, password: roomForm.password, deck: roomDeck(), name: duelist})) {
    roomForm.busy = "join";
  }
  renderLobby();
  if (roomForm.err) { const f = document.getElementById("raddr"); if (f) f.focus(); }
}
function scanRooms() {
  if (scanning) return;
  scanErr = null;
  if (sendRoom({t: "room_scan"})) scanning = true;
  else rooms = rooms || [];
}
function leaveRoom() {
  clearTimeout(roomCloseTimer);
  const inDuel = msg && msg.t === "state" && msg.room && !msg.result;
  if (inDuel) sendRoom({t: "surrender"});
  sendRoom({t: "room_leave"});
  room = null;
  roomUi = {closeArm: false, copied: null, note: null, deckLabel: roomUi.deckLabel};
  lobbyTab = "people";
  gameMenu = false;
  closeOv("gamemenu");
  goLobby();
}
function onRoomAction(kind, el) {
  if (kind === "lobbytab") {
    const t = el.getAttribute("data-tab");
    lobbyTab = LOBBY_TABS.includes(t) ? t : "play";
    if (lobbyTab === "people" || lobbyTab === "play" || lobbyTab === "games") playSub = lobbyTab;
    saveSetting("ygo-lobbytab", lobbyTab);
    if (lobbyTab === "ask") ask.enter();
    if (lobbyTab === "games") games.enter();
    if (lobbyTab === "people" && rooms === null) scanRooms();
    if (lobbyTab === "decks") db.enter();
    if (lobbyTab === "puzzles" && puzzles === null) {
      if (MOCK) puzzles = mockPuzzles();
      else if (ws) ws.send({t: "puzzles"});
    }
  } else if (kind === "roomhost") {
    roomUi.note = null;
    if (sendRoom({t: "room_host", deck: roomDeck(), name: duelist, password: ""})) roomForm.busy = "host";
  } else if (kind === "roomjoin") {
    joinRoom(roomForm.address);
    return;
  } else if (kind === "roomjoinaddr") {
    joinRoom(el.getAttribute("data-addr") || "");
    return;
  } else if (kind === "roompw") {
    roomForm.showPw = !roomForm.showPw;
  } else if (kind === "roomscan") {
    scanRooms();
  } else if (kind === "roomleave") {
    const opp = room && (room.seats || []).some((x) => x && !x.you);
    const inDuel = msg && msg.t === "state" && msg.room;
    // Closing a hosted room with a guest in it asks once (P2); leaving from the duel menu surrenders at once.
    if (room && room.role === "host" && opp && !inDuel && !roomUi.closeArm && room.state !== "ended") {
      roomUi.closeArm = true;
      clearTimeout(roomCloseTimer);
      roomCloseTimer = setTimeout(() => { roomUi.closeArm = false; render(); }, 4000);
    } else {
      leaveRoom();
      return;
    }
  } else if (kind === "roomready") {
    const me = room && (room.seats || []).find((x) => x && x.you);
    sendRoom({t: "room_ready", ready: !(me && me.ready)});
  } else if (kind === "roomstart") {
    sendRoom({t: "room_start"});
  } else if (kind === "roomrps") {
    const hand = parseInt(el.getAttribute("data-hand"), 10);
    if (hand >= 1 && hand <= 3 && sendRoom({t: "room_rps", hand}) && room) room = Object.assign({}, room, {ask: null});
  } else if (kind === "roomfirst") {
    if (sendRoom({t: "room_first", first: el.getAttribute("data-tp") === "1"}) && room) room = Object.assign({}, room, {ask: null});
  } else if (kind === "roomrematch") {
    const yes = el.getAttribute("data-yes") === "1";
    if (sendRoom({t: "rematch", yes}) && room) room = Object.assign({}, room, {ask: null});
    if (yes) say("Rematch accepted. Waiting for the other player.");
  } else if (kind === "roomback") {
    if (sendRoom({t: "room_back"}) && room) room = Object.assign({}, room, {state: "connecting", ask: null, error: null});
    goLobby();
    return;
  } else if (kind === "roomcopy") {
    const a = el.getAttribute("data-addr") || "";
    const done = () => { roomUi.copied = a; say("Copied " + a); render(); };
    const fallback = () => {
      // No clipboard API (or refused): select the address so the system copy works.
      const code = el.parentNode && el.parentNode.querySelector("code");
      if (code) { const r = document.createRange(); r.selectNodeContents(code); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }
      say("Address selected. Copy it with your device's copy command.");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(a).then(done, fallback);
    else fallback();
    return;
  }
  render();
}

// A {"t":"room"} frame: the room's whole state (server room.rs room_frame).
function onRoomFrame(m) {
  roomForm.busy = null;
  if (m.state === "closed") {
    room = null;
    roomUi = {closeArm: false, copied: null, note: null, deckLabel: roomUi.deckLabel};
    lobbyTab = "people";
    if (m.error && m.field === "address") roomForm.err = m.error;
    else if (m.error) roomUi.note = m.error;
    if (msg && msg.t === "state") goLobby();
    else render();
    return;
  }
  const prev = room;
  room = m;
  roomForm.err = null;
  if (m.state === "dueling" && (!prev || prev.state !== "dueling")) {
    // A new duel in the room: drop the last board so this duel's first state is applied fresh.
    resetGameUi();
    resetCues();
    recap = null;
    recapAcc = null;
    gameMenu = false;
    practiceMode = false;
    gameMode = null;
    msg = lobby || {t: "lobby", decks: []};
    inLobby = false;
    say(m.ask === "rps" ? "The duel is starting. Rock, paper, scissors." : "The duel is starting.");
  }
  if (m.ask && (!prev || prev.ask !== m.ask)) {
    if (m.ask === "first") say("You won. Choose who goes first.");
    else if (m.ask === "rematch") say("Rematch offered.");
  }
  roomUi.note = m.error && m.state !== "ended" ? m.error : null;
  if (m.state !== "waiting") { roomUi.closeArm = false; clearTimeout(roomCloseTimer); }
  render();
}

function onAction(kind, el) {
  if (kind.startsWith("db:")) { db.act(kind, el); return; }
  if (kind.startsWith("dm:")) { dm.act(kind); return; }
  if (kind.startsWith("ask:")) { ask.act(kind, el); return; }
  if (kind.startsWith("games:")) { games.act(kind, el); return; }
  if (kind === "lobbytab" || kind.startsWith("room")) { onRoomAction(kind, el); return; }
  if (kind === "logall") logAll = !logAll;
  else if (kind === "log") logOpen = true;
  else if (kind === "closelog") { closeState("log"); render(); closeOv("log"); return; }
  else if (kind === "closepile") { closeState("pile"); render(); closeOv("pile"); return; }
  else if (kind === "cancel") {
    if (!msg || msg.result || locked || paused || msg.thinking || !msg.can_cancel) return;
    if (MOCK) {
      msg = Object.assign({}, msg, {can_cancel: false});
      say("Cancelled");
      render();
      return;
    }
    locked = true;
    render();
    if (ws) ws.send({t: "cancel"});
    return;
  }
  else if (kind === "unfocus") { focusUid = null; readCard = null; }
  else if (kind === "detail") {
    detail = findCard(el.getAttribute("data-uid"));
    closeState("menu");
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
  } else if (kind === "pausetoggle") {
    setPaused(!paused);
  } else if (kind === "gamemenu") { gameMenu = true; surrenderArm = false; }
  else if (kind === "closegamemenu") { closeState("gamemenu"); render(); closeOv("gamemenu"); return; }
  else if (kind === "opensettings") {
    // Opening over the duel closes the game menu; the board stays live underneath.
    const gb = gameMenu ? document.querySelector('[data-act="gamemenu"]') : null;
    settingsOpen = true;
    if (gameMenu) {
      gameMenu = false;
      surrenderArm = false;
    }
    render();
    if (gb) opener = describe(gb) || opener;   // focus returns to the game-menu button
    return;
  }
  else if (kind === "closesettings") { closeState("settings"); render(); closeOv("settings"); return; }
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
  } else if (kind === "closedecks") {
    closeDeckBrowser();
    return;
  } else if (kind === "decklist" && deckBrowser) {
    const sc = document.querySelector("#decksheet .dklist");
    deckBrowser.top = sc ? sc.scrollTop : 0;
    deckBrowser.view = el.getAttribute("data-deck");
    renderDeckSheet();
    const b = document.querySelector("#decksheet .dkview");
    if (b) b.scrollTop = 0;
    return;
  } else if (kind === "deckback" && deckBrowser) {
    const id = deckBrowser.view;
    deckBrowser.view = null;
    renderDeckSheet();
    const sc = document.querySelector("#decksheet .dklist");
    if (sc) sc.scrollTop = deckBrowser.top;
    const back = document.querySelector('#decksheet [data-act="decklist"][data-deck="' + CSS.escape(id) + '"]');
    if (back) back.focus({preventScroll: true});
    return;
  } else if (kind === "dclear" && deckBrowser) {
    deckBrowser.f = {};
    renderDeckSheet();
    return;
  } else if (kind === "packupdate") {
    if (!MOCK && ws && !packBusy) {
      packBusy = true;
      ws.send({t: "packs_update"});
      render();
    }
    return;
  } else if (kind === "packrollback") {
    if (!MOCK && ws && !packBusy) {
      packBusy = true;
      ws.send({t: "packs_rollback"});
      render();
    }
    return;
  } else if (kind === "opts") {
    optsOpen = !optsOpen;
    renderLobby();
    return;
  } else if (kind === "pasteydk") {
    pasteOpen = true;
    pickerOpen = null;
    deckBrowser = null;
    renderLobby();
    const ta = document.getElementById("ydk");
    if (ta) ta.focus({preventScroll: true});
    return;
  } else if (kind === "closepaste") {
    pasteOpen = false;
    renderLobby();
    return;
  } else if (kind === "cardart") {
    cardArt = !cardArt;
    cardArtChosen = true;
    saveSetting("ygo-card-art", cardArt ? "1" : "0");
    setCardArt(cardArt);
    sendArtNetwork();
    sendArtToggle();
  } else if (kind === "artpause") {
    if (!MOCK && ws) ws.send({t: "art_sync_pause"});
    return;
  } else if (kind === "artresume") {
    if (!MOCK && ws) ws.send({t: "art_sync_resume"});
    return;
  } else if (kind === "artdata") {
    if (!MOCK && ws) ws.send({t: "art_sync_allow_data"});
    return;
  } else if (kind === "artdelete") {
    if (MOCK) { artDeleteArm = !artDeleteArm; render(); return; }
    if (!ws) return;
    if (!artDeleteArm) { artDeleteArm = true; render(); return; }
    artDeleteArm = false;
    ws.send({t: "art_sync_delete"});
    return;
  } else if (kind === "closerecap") {
    recap = null;
  } else if (kind === "chainset") {
    const v = el.getAttribute("data-v");
    if (v === "auto" || v === "all") {
      chainMode = v;
      try { localStorage.setItem("ygo-chain", v); } catch (e) { /* ignore */ }
      if (!MOCK && ws) sendSettings();
    }
  } else if (kind === "themeset") {
    const v = el.getAttribute("data-v");
    if (v === "dark" || v === "light" || v === "system") {
      themeMode = v;
      saveSetting("ygo-theme", v);
      applyTheme();
    }
  } else if (kind === "winchanceset") {
    winChance = !winChance;
    saveSetting("ygo-winchance", winChance ? "1" : "0");
    if (!MOCK && ws) sendSettings();
  } else if (kind === "start" || kind === "rematch") {
    // No duel before the bot is ready, and a second click sends no second new_game.
    if (starting || startupPending || startupFailed) return;
    if (kind === "start" && lobbyPick.first === "rps") {
      // Play rock-paper-scissors here first; the duel starts when it resolves.
      rpsGame = {phase: "pick", mine: null, theirs: null};
      rpsTestHand = null;
      render();
      return;
    }
    beginDuel(lobbyPick.first, kind === "start");
    return;
  } else if (kind === "rpspick") {
    if (!rpsGame || rpsGame.phase !== "pick") return;
    const mine = parseInt(el.getAttribute("data-hand"), 10);
    if (mine < 1 || mine > 3) return;
    const theirs = rpsBotHand();
    const out = rpsOutcome(mine, theirs);
    rpsGame = {phase: "result", mine, theirs};
    say("Rock, paper, scissors: you threw " + RPS_NAME[mine] + ", decklight threw " + RPS_NAME[theirs] + ". " +
      (out === "draw" ? "A draw." : out === "you" ? "You win." : "Decklight wins."));
  } else if (kind === "rpsagain") {
    if (!rpsGame) return;
    rpsGame = {phase: "pick", mine: null, theirs: null};
  } else if (kind === "rpsfirst") {
    const first = el.getAttribute("data-first") === "bot" ? "bot" : "me";
    rpsGame = null;
    closeOv("rps");
    beginDuel(first, true);
    return;
  } else if (kind === "rpsbotfirst") {
    rpsGame = null;
    closeOv("rps");
    beginDuel("bot", true);
    return;
  } else if (kind === "rpscancel") {
    rpsGame = null;
    closeOv("rps");
  } else if (kind === "closemenu") {
    const uid = menuUid;
    menuUid = null;
    render();
    closeOv("menu");
    const card = uid && app.querySelector('.card[data-uid="' + CSS.escape(uid) + '"]');
    if (card) card.focus({preventScroll: true});
    return;
  } else if (kind === "usebase") {
    lobbyPick.model = "base";
    saveSetting("ygo-model", "base");
    modelNotice = null;
    render();
    return;
  } else if (kind === "usetuned") {
    const models = lobbyModels();
    const tune = models.filter((m) => m.kind !== "base" && m.deck_id && m.deck_id === lobbyPick.bot)[0];
    if (tune) {
      lobbyPick.model = tune.id;
      saveSetting("ygo-model", tune.id);
    }
    modelNotice = null;
    render();
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
    if (!msg || !msg.prompt || hintLoading || paused) return;
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
  } else if (kind === "primer") {
    primerOpen = true;
    primerLoading = !MOCK && !!ws;
    primer = null;
    gameMenu = false;
    opener = describe(el) || opener;  // focus returns here when the sheet closes
    if (msg && msg.t === "lobby") {
      primerDeck = deckName((lobby && lobby.decks) || [], lobbyPick.bot);
      const bd = ((lobby && lobby.decks) || []).find((x) => x.id === lobbyPick.bot);
      primerForge = bd ? (bd.forge || null) : undefined;
      if (MOCK) {
        primerDeck = "Tenpai Dragon";
      } else if (ws) {
        sendPrimerRequest();
      }
    } else {
      primerDeck = "Opponent's deck";
      primerForge = undefined;
      if (!MOCK && ws) ws.send({t: "primer"});
    }
    if (MOCK) {
      primer = mockPrimer();
      primerDeck = "Tenpai Dragon";
      primerHash = "mockhash";
      canStudy = false;
      studyState = null;
      primerLoading = false;
    }
  } else if (kind === "study") {
    sendStudyRequest();
    return;
  } else if (kind === "closeprimer") {
    closeState("primer");
    render();
    closeOv("primer");
    return;
  } else if (kind === "puzzlestart") {
    if (starting) return;   // a second click sends no second puzzle request
    const id = el.getAttribute("data-id");
    if (!id) return;
    if (MOCK) { loadMock(1); puzzle = {id, goal: "Win this turn · opponent has 6,700 LP", n_plays: 4, status: "playing"}; render(); return; }
    starting = true;
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
  } else if (kind === "pzchip") {
    const set = puzzleF[el.getAttribute("data-f")];
    const k = el.getAttribute("data-k");
    if (set) { if (set.has(k)) set.delete(k); else set.add(k); }
    render();
    // Selected decks move to the front: keep focus on the chip that was pressed, wherever it went.
    const same = [...document.querySelectorAll('#lobby [data-act="pzchip"]')]
      .find((b) => b.getAttribute("data-f") === el.getAttribute("data-f") && b.getAttribute("data-k") === k);
    if (same) same.focus();
    return;
  } else if (kind === "pzmore") {
    puzzleMore = !puzzleMore;
    render();
    return;
  } else if (kind === "pzdecks") {
    puzzleAllDecks = !puzzleAllDecks;
    render();
    return;
  } else if (kind === "pzclear") {
    puzzleQ = "";
    for (const k in puzzleF) puzzleF[k].clear();
    render();
    const f = document.getElementById("psearch");
    if (f) { f.value = ""; f.focus(); }
    return;
  } else if (kind === "puzzlesback") {
    hidePuzzleResult = false;
    inSolution = false;
    solutionFrames = null;
    liveEnd = null;
    lobbyTab = "puzzles";
    goLobby();
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
    {id: "p-easy", decks: ["Blue Eyes", "Branded"], seat: 0, turn: 2, difficulty: "easy", goal: "Win this turn · opponent has 6,700 LP", n_plays: 3},
    {id: "p-medium", decks: ["Mitsurugi", "Snake-Eye"], seat: 1, turn: 3, difficulty: "medium", goal: "Win this turn · opponent has 4,200 LP", n_plays: 5},
    {id: "p-hard", decks: ["Branded", "Blue Eyes"], seat: 0, turn: 4, difficulty: "hard", goal: "Win this turn · opponent has 7,900 LP", n_plays: 9},
  ];
}

// Mock primer for screenshots (shape of the server's trimmed reply).
function mockPrimer() {
  return {
    archetype: "Tenpai Dragon",
    games: 48,
    key_cards: [
      {code: 82570174, name: "Sangenpai Bident Dragion", games: 40, per_game: 0.833, impact: 8.4},
      {code: 91810826, name: "Tenpai Dragon Chundra", games: 45, per_game: 0.938, impact: 7.3},
      {code: 39931513, name: "Tenpai Dragon Paidra", games: 43, per_game: 0.896, impact: 4.7},
      {code: 30336082, name: "Sangen Summoning", games: 37, per_game: 0.771, impact: 4.5},
    ],
    first_plays: [
      {code: 30336082, name: "Sangen Summoning", games: 6, share: 0.4},
      {code: 39931513, name: "Tenpai Dragon Paidra", games: 4, share: 0.27},
      {code: 66730191, name: "Sangen Kaimen", games: 3, share: 0.2},
      {code: 73628505, name: "Terraforming", games: 1, share: 0.07},
    ],
    first_n: 15,
    your_answers: [
      {code: 14558127, name: "Ash Blossom & Joyous Spring", hits: [
        {code: 39931513, name: "Tenpai Dragon Paidra", times: 5},
        {code: 30336082, name: "Sangen Summoning", times: 2},
      ]},
      {code: 23434538, name: "Maxx \"C\"", note: "used against other decks", hits: []},
    ],
  };
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

// Long press (user 2026-11-15: "holding longer to select the game to then go to the selection menu like other apps
// do"): an element with `data-hold` runs that action after 500 ms held still, and the click that ends the press is
// swallowed. A right click (desktop) does the same. The tap action and a visible button stay the accessible path.
const HOLD_MS = 500;
let hold = null;   // {el, x, y, timer, fired}
document.addEventListener("pointerdown", (ev) => {
  const el = ev.target.closest && ev.target.closest("[data-hold]");
  if (!el || ev.button > 0) return;
  hold = {el, x: ev.clientX, y: ev.clientY, fired: false, timer: setTimeout(() => {
    hold.fired = true;
    try { if (navigator.vibrate) navigator.vibrate(12); } catch (e) { /* ignore */ }
    onAction(el.getAttribute("data-hold"), el);
  }, HOLD_MS)};
});
document.addEventListener("pointermove", (ev) => {
  if (hold && !hold.fired && Math.hypot(ev.clientX - hold.x, ev.clientY - hold.y) > 10) { clearTimeout(hold.timer); hold = null; }
});
for (const evt of ["pointerup", "pointercancel"]) {
  document.addEventListener(evt, () => {
    if (!hold) return;
    if (!hold.fired) { clearTimeout(hold.timer); hold = null; return; }
    // The click that ends this press comes right after pointerup (or never, when the row was re-rendered under the
    // finger): swallow only that one, never a later tap.
    const h = hold;
    setTimeout(() => { if (hold === h) hold = null; }, 0);
  });
}
document.addEventListener("click", (ev) => {
  if (hold && hold.fired) { hold = null; ev.stopPropagation(); ev.preventDefault(); }
}, true);
document.addEventListener("contextmenu", (ev) => {
  const el = ev.target.closest && ev.target.closest("[data-hold]");
  if (!el) return;
  ev.preventDefault();
  // A touch long press also fires contextmenu: the timer already ran the action.
  if (hold && hold.fired) return;
  if (hold) { clearTimeout(hold.timer); hold = null; }
  onAction(el.getAttribute("data-hold"), el);
});

// Swipe down on a dismissable sheet's header (its grab bar and title) closes it, as the grab bar promises (user
// 2026-11-15: "if I see a menu that had the top bar in it I'm expecting to be able to swipe it down to close it").
// Prompt sheets are not dismissable (the duel waits on them), so only these: the pile, log, deck, primer and game-menu
// sheets, and every feature sheet or dialog that names its close action in `data-close` (Ask's model sheet, ...).
const SWIPE_SHEETS = "#sheet.pilesheet, #logdrawer, #decksheet, [data-close]";
let swipe = null;   // {el, y0, t0, dy} while a header drag is in progress
document.addEventListener("pointerdown", (ev) => {
  const t = ev.target;
  const el = t.closest && t.closest(SWIPE_SHEETS);
  if (!el || t.closest("button, a, input, select, textarea, [data-act]:not(.grab)")) return;
  const top = el.getBoundingClientRect().top;
  if (!t.closest(".grab, .stitle, .dhead, .mhead, h2") && ev.clientY - top > 56) return;   // only the header band starts a swipe
  swipe = {el, y0: ev.clientY, t0: performance.now(), dy: 0};
});
// A touch drag that started a swipe must not also scroll: the browser would take it over and cancel the pointer
// (pointercancel), which is how the swipe silently did nothing on phones where the header is not touch-action: none.
document.addEventListener("touchmove", (ev) => { if (swipe && ev.cancelable) ev.preventDefault(); }, {passive: false});
document.addEventListener("pointermove", (ev) => {
  if (!swipe) return;
  swipe.dy = Math.max(0, ev.clientY - swipe.y0);
  swipe.el.style.transition = "none";
  swipe.el.style.translate = "0 " + swipe.dy + "px";   // `translate`, not `transform`: centred sheets keep translateX(-50%)
});
for (const evt of ["pointerup", "pointercancel"]) {
  document.addEventListener(evt, (ev) => {
    if (!swipe) return;
    const {el, dy, t0} = swipe;
    swipe = null;
    el.style.transition = "";
    el.style.translate = "";
    const fast = dy > 24 && dy / Math.max(1, performance.now() - t0) > 0.5;   // a flick: > 0.5 px/ms
    if (evt === "pointerup" && (dy > 80 || fast)) {
      const c = el.getAttribute("data-close");
      if (c) onAction(c, el); else closeTop();
    }
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
  if (t.matches(".dksearch") && deckBrowser) {
    deckBrowser.q = t.value;
    renderDeckSheet();
  }
  // Room fields: kept in state so lobby re-renders write back what was typed.
  if (t.id === "rname" || t.id === "sname") {
    duelist = t.value.trim() || "Duelist";
    saveSetting("ygo-duelist", duelist);
  } else if (t.id === "raddr") {
    roomForm.address = t.value;
    if (roomForm.err) { roomForm.err = null; renderLobby(); }
  } else if (t.id === "rpw") {
    roomForm.password = t.value;
  } else if (t.id === "gsearch") {
    games.search(t.value);
  } else if (t.id === "psearch") {
    puzzleQ = t.value;
    renderLobby();
  }
});

// ---- keyboard and focus (WCAG 2.1.1, 2.4.3, 2.4.7) ----
// Modal layers, topmost first. Focus moves into the top one when it opens, Tab stays inside,
// Escape closes it, and focus returns to the control that opened it.
const FOCUSABLE = 'button:not([disabled]), [href], input, textarea, select, summary, [tabindex]:not([tabindex="-1"])';
const visible = (el) => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden" && !el.closest(".stage-aside");
function topModal() {
  if (IS_REPLAY) return null;
  for (const sel of ["#rps", "#settings", "#kwpop", "#detail", "[data-close][aria-modal=true]", "#menu", "#gamemenu", "#primersheet", "#decksheet", "#sheet.pilesheet", "#reviewframe", "#logdrawer", "#puzzleresult[role=alertdialog]", "#gameover[role=alertdialog]"]) {
    const el = document.querySelector(sel);
    if (el && !el.closest("[inert]") && visible(el)) return el;
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
  for (const a of ["data-uid", "data-act", "data-pile", "data-opt", "data-pickopen", "data-recap", "data-primercode"]) {
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
  if (stageFocus) {
    const o = stageFocus;
    let el = o.el && o.el.isConnected && visible(o.el) ? o.el : null;
    if (!el && o.sel) el = [...document.querySelectorAll(o.sel)].find(visible) || null;
    if (el && !el.closest(".stage-aside") && visible(el)) {
      stageFocus = null;
      el.focus({preventScroll: true});
      modalTop = top;
      return;
    }
    if (!o.sel) stageFocus = null;
  }
  if (top === modalTop) {
    if (top && !top.contains(document.activeElement)) (focusables(top)[0] || top).focus({preventScroll: true});
    return;
  }
  const prev = modalTop;
  modalTop = top;
  if (top) {
    if (!prev || !prev.isConnected) opener = describe(document.activeElement) || opener;
    const auto = top.querySelector("[data-autofocus]");
    (auto || focusables(top)[0] || top).focus({preventScroll: true});
  } else {
    restoreFocus();
  }
  if (!top && document.activeElement === document.body) {
    // A new prompt sheet after the old focus target went away: start at its first choice.
    const first = document.querySelector("#sheet:not(.pilesheet):not(.stage-aside) [data-opt]");
    if (first && visible(first) && !locked) first.focus({preventScroll: true});
  }
}
function closeTop() {
  if (rpsGame) { rpsGame = null; render(); closeOv("rps"); return true; }
  if (settingsOpen) { closeState("settings"); render(); closeOv("settings"); return true; }
  if (kwPop) { closeKw(); return true; }
  if (menuUid) {
    const uid = menuUid;
    closeState("menu"); render(); closeOv("menu");
    const card = app.querySelector('.card[data-uid="' + CSS.escape(uid) + '"]');
    if (card) card.focus({preventScroll: true});
    return true;
  }
  if (detail) { closeState("detail"); render(); closeOv("detail"); return true; }
  // A feature's sheet or dialog (Escape, Android back): its own close action.
  const fd = [...document.querySelectorAll("[data-close]")].find((el) => !el.closest("[inert]") && visible(el));
  if (fd) { onAction(fd.getAttribute("data-close"), fd); return true; }
  if (reviewFrame) { reviewFrame = null; render(); return true; }
  if (gameMenu) { closeState("gamemenu"); render(); closeOv("gamemenu"); return true; }
  if (primerOpen) { closeState("primer"); render(); closeOv("primer"); return true; }
  if (deckBrowser) {
    if (deckBrowser.view) onAction("deckback", null);
    else closeDeckBrowser();
    return true;
  }
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
  // Enter in the room address or password field joins (there is no <form>: the frame has nowhere to submit).
  if (ev.key === "Enter" && (t.id === "raddr" || t.id === "rpw")) {
    ev.preventDefault();
    joinRoom(roomForm.address);
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
  // Pause vs the bot: Space toggles (not while typing or on another control).
  if (ev.key === " " && !typing && !topModal() && msg && msg.t === "state" && !msg.room &&
      !IS_REPLAY && !inSolution && !msg.result && !ev.ctrlKey && !ev.metaKey && !ev.altKey &&
      !(t.closest && t.closest("button, a, input, textarea, select, [role=button], [contenteditable=true]"))) {
    ev.preventDefault();
    setPaused(!paused);
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
  apply: (m, info) => applyState(m, info),
  mock: MOCK,
  names: () => (msg && msg.names) || null,
  speed: () => (cueSpeed === "2" ? 2 : 1),
  pause: () => cuePause && !IS_REPLAY,
  held: () => paused || stage.lookup(),
  announce: (t) => say(t),
  reduced: () => reducedMotion() && !IS_REPLAY,
  ruleNotes: () => ruleNotesOn,
  onScene: () => { stage.progress(); if (stage.request("cues")) render(); },
  onChain: (links, solving) => { cueChain = links; cueSolving = solving; render(); },
  onIdle: () => { stage.done("cues"); render(); },
  textFor: (code) => {
    const t = cardText(code);
    return t && t.text ? t.text : null;
  },
});

function resetCues() {
  cue.reset();
  cueChain = null;
  cueSolving = 0;
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
  const oppName = (m.names && m.names[1]) || "decklight";
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
    say("Duel over. " + (w === "you" ? "Victory." : w === "bot" || w === "opponent" ? "Defeat."
      : w == null ? (m.result.reason === "disconnect" ? "Connection lost." : "Draw.") : String(w) + " wins."));
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

function applyState(m, info) {
  if (!m) return;
  // A new prompt that opens a sheet replaces the card detail and the
  // info-panel focus: no half-covered panel stays behind the sheet.
  // Idle/battle prompts open no sheet, so they keep the focus.
  const prevPrompt = msg && msg.t === "state" ? msg.prompt : null;
  if (m.prompt && (!prevPrompt || m.prompt.id !== prevPrompt.id) &&
      m.prompt.kind !== "idle" && m.prompt.kind !== "battle") {
    detail = null;
    focusUid = null;
  }
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
  if (m.model_label) gameModel = m.model_label;
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

// One state frame from the server (or a mock test hook). While paused, frames
// wait in pauseQueue: no board change, prompt or cue is applied until Resume.
function handleState(m) {
  if (inLobby) return;   // Back to lobby: ignore states until the next new_game
  // A room duel: the opponent gets their name; "you" stays second person (render/board.js, cues.js).
  if (m.room) m.names = [null, m.room.opponent || "Opponent"];
  if (paused && msg && msg.t === "state") { pauseQueue.push(m); return; }
  if (starting || !msg || msg.t !== "state") {
    // First state of a new game: apply the board at once, skipping that
    // message's cues (they would animate against no board), but still run
    // freshCues + collectRecap so the opponent-turn recap still works.
    // Peek/think states count too: the board appears with the thinking
    // indicator while the bot plays turn 1; later moves animate as now.
    const cues = freshCues(m);
    for (const c of cues) if (c.t === "activate" && c.card && c.card.code) cardText(c.card.code);
    collectRecap(m, cues);
    starting = false;
    applyState(m);
    return;
  }
  receiveState(m);
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
  if (msg.t === "lobby") {
    lobby = mockSavedDecks(msg);
    lobbyPick.model = null;
    modelNotice = null;
    ensureModelPick();
    if (qs.get("model")) {
      const m = modelById(qs.get("model"));
      if (m) { lobbyPick.model = m.id; saveSetting("ygo-model", m.id); }
    }
    if (qs.get("bot")) {
      const decks = (lobby && lobby.decks) || [];
      if (decks.some((d) => d.id === qs.get("bot"))) lobbyPick.bot = qs.get("bot");
    }
  }
  if (msg.t === "state" && qs.get("modellabel")) {
    msg = Object.assign({}, msg, {model_label: qs.get("modellabel")});
    gameModel = msg.model_label;
  }
  // Screenshot hook for the setting-up status: ?starting=1 shows the loading
  // screen right after Start (start button disabled, no second new_game).
  starting = qs.get("starting") === "1";
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
  rpsGame = null;
  rpsTestHand = null;
  paused = false;
  pauseQueue = [];
  stage.reset();
  detail = null;
  gameMenu = false;
  settingsOpen = false;
  primerOpen = false;
  primer = null;
  primerLoading = false;
  primerDeck = "";
  primerHash = null;
  canStudy = false;
  studyState = null;
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
  artSync = (msg.t === "lobby" && msg.art_sync) ? msg.art_sync : null;
  artDeleteArm = false;
  // Screenshot hook: ?art=running|wifi|done shows the sync panel in Settings without a server.
  {
    const av = qs.get("art");
    const base = {t: "art_sync", total: 14792, bytes: 243900000, rate_per_s: 11.2, allow_data: false,
      unavailable: 37, have: 8412};
    if (av === "running") artSync = Object.assign({}, base, {state: "running"});
    else if (av === "wifi") artSync = Object.assign({}, base, {state: "waiting_wifi", have: 1204, bytes: 35600000});
    else if (av === "done") artSync = Object.assign({}, base, {state: "done", have: 14755, bytes: 428100000});
  }
  if (artSync) { cardArt = true; setCardArt(true); }
  focusUid = qs.get("focus") || null;
  pileOpen = msg.openPile || qs.get("pile") || null;
  if (qs.get("menu") && msg.t === "state") {
    // Screenshot hook: open a card's action panel.
    menuUid = qs.get("menu");
  }
  // Mocks 84-87 carry the action panel / RPS state themselves.
  if (msg.openMenu && msg.t === "state") menuUid = msg.openMenu;
  if (msg.rps && msg.t === "lobby") rpsGame = msg.rps;
  if (qs.get("sel") && msg.t === "state") sel = qs.get("sel").split(",").map(Number);
  if (qs.get("picker") && msg.t === "lobby") {
    const side = qs.get("picker");
    if (side === "model") pickerOpen = side;
    else { deckBrowser = {side, q: "", f: {}, sort: "name", view: qs.get("deckview"), top: 0}; loadDeckIndex(); }
  }
  if (qs.get("play") === "practice" && msg.t === "lobby") lobbyPick.play = "practice";
  if (qs.get("opts") === "1" && msg.t === "lobby") optsOpen = true;
  // Screenshot hooks for the deck builder: ?ltab=decks&dbopen=<id>&dbview=add&dbq=<text>&dbsheet=export|import|menu|card
  if (msg.t === "lobby" && lobbyTab === "decks") db.mockInit(qs);
  if (msg.t === "lobby") {
    dm.refresh();
    if (qs.get("dmstate") === "downloading") dm.act("dm:download");
    if (qs.get("dmstate") === "installed") {
      dm.onMessage({t: "describe_status", installed: true, downloading: false, bytes: 1200510818, total: 1200510818, error: null});
    }
    if (qs.get("settings") === "describe") openSettingsAt("set-describe");
  }
  if (qs.get("gamemenu") && msg.t === "state") gameMenu = true;
  // Screenshot hook: ?settings=1 opens the app-wide Settings screen (lobby or over a duel).
  if (qs.get("settings") === "1") settingsOpen = true;
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
    lobbyTab = "puzzles";
  }
  // Primer screenshots: ?primer=1 (a primer) or ?primer=null (not enough games),
  // ?primer=similar (similar-list note + study button), ?primer=study (empty +
  // study button), ?primer=studying (mid-study status), ?primer=partial
  // (first-version note), in the lobby or over a game.
  {
    const pv = qs.get("primer");
    if (pv === "1" || pv === "null" || pv === "similar" || pv === "study" ||
        pv === "studying" || pv === "partial") {
      primerOpen = true;
      primerLoading = false;
      primerDeck = "Tenpai Dragon";
      primerHash = "mockhash";
      canStudy = pv === "null" || pv === "study" || pv === "studying";
      if (pv === "null" || pv === "study" || pv === "studying") {
        primer = null;
      } else {
        primer = mockPrimer();
      }
      if (pv === "similar" && primer) {
        primer.similar = {archetype: "Tenpai Dragon", match: 37, of: 40};
      }
      if (pv === "partial" && primer) {
        primer.study = {phase: "games", games: 21};
      }
      if (pv === "studying") {
        studyState = {t: "study_status", phase: "games", done: 12, total: 32,
          deck: "Tenpai Dragon", hash: "mockhash"};
      }
    }
  }
  if (MOCK && msg.t === "state" && msg.mockPaused) paused = true;
  // Screenshot hook for the app-startup state: ?loadingbot=1 shows the lobby with "Loading the bot…" and Start off.
  startupPending = qs.get("loadingbot") === "1";
  startupFailed = null;
  render();
  // A mock may carry a reveal that arrives after its prompt is up (the same
  // sequence as a live state frame whose log adds a bot activation).
  const bootMsg = msg;
  if (MOCK && bootMsg.t === "state" && bootMsg.reveal) {
    setTimeout(() => {
      if (msg !== bootMsg) return;   // another mock loaded meanwhile
      revealQueue = [];
      revealCur = bootMsg.reveal;
      stage.request("reveal");
      render();
    }, 120);
  }
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
    stage.request("cues");
    render();
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

// ?replay=embed: the app's own player for Watch, exports and the chat's stills (web/lib/player.js). The page that
// holds it hands the frames in with loadFrames(frames, {speed}) and drives showFrame / cuePlan like game_video.py.
async function loadReplay() {
  if (EMBED) {
    window.loadFrames = (frames, o) => {
      o = o || {};
      if (o.speed === "1" || o.speed === "2") cueSpeed = o.speed;
      startReplay(frames);
    };
    // The speed applies from the next scene on (opts.speed reads it): no reload, so a play in progress goes on.
    window.setSpeed = (s) => { if (s === "1" || s === "2") cueSpeed = s; };
    window.playerReady = true;
    return;
  }
  let url = REPLAY_URL;
  if (TOKEN && !url.includes("t=")) url += (url.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN);
  const r = await fetch(url);
  if (!r.ok) {
    msg = {t: "error", text: "replay load failed: " + REPLAY_URL};
    render();
    return;
  }
  const j = await r.json();
  startReplay(Array.isArray(j) ? j : (j.frames || []));
  if (replayFrames.length) await window.showFrame(0);
}

function startReplay(frames) {
  replayFrames = frames;
  replayIdx = -1;
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
}

// The app's gate reports when the bot is ready or has failed. Until the first lobby message arrives over the socket
// the lobby stays up with "Loading the bot…"; a failed start shows the gate's message in its place.
function watchReady() {
  let done = false;
  const tick = () => {
    if (done) return;
    // The first lobby message already cleared the status (e.g. served by the plain server, which has no /__ready).
    if (!startupPending && !startupFailed) { done = true; return; }
    fetch("/__ready", {cache: "no-store"})
      .then((r) => r.json())
      .then((j) => {
        if (j.failed) {
          done = true;
          startupPending = false;
          startupFailed = j.msg || "The bot could not start.";
          // No server will answer the socket: stop the quiet retries (they would turn into a banner later).
          try { if (ws) ws.close(); } catch (e) { /* ignore */ }
          render();
          return;
        }
        // Ready means the server is up; the first lobby message clears the status line.
        if (j.ready) { done = true; return; }
        setTimeout(tick, 400);
      })
      .catch(() => setTimeout(tick, 600));
  };
  tick();
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
      lobby = mockSavedDecks(await (await fetch(MOCKS[0])).json());
    } catch (e) {
      lobby = {t: "lobby", decks: []};
    }
    msg = lobby;
    renderLobby();
    return;
  }
  // The client is served before the bot is ready: show the lobby at once, with a quiet status line and Start off.
  startupPending = true;
  startupFailed = null;
  render();
  watchReady();
  ws = connectWS(TOKEN, {
    onOpen() {
      banner.hidden = true;
      // A request lost with the old socket never gets its answer: unlock Host/Join and the scan.
      roomForm.busy = null;
      scanning = false;
      sendSettings();
      // The sync is idle until the client says which network it is on and whether art is on.
      sendArtNetwork();
      sendArtToggle();
    },
    onDrop() {
      hideBoot();  // the "Reconnecting…" banner must be seen
      banner.hidden = false;
      banner.textContent = "Reconnecting…";
      packBusy = false;
    },
    onTakenOver() {
      hideBoot();
      banner.hidden = false;
      banner.innerHTML = 'Open in another window. <button type="button" id="usehere">Use here</button>';
      packBusy = false;
      const b = document.getElementById("usehere");
      b.addEventListener("click", () => {
        banner.textContent = "Reconnecting…";
        ws.resume();
      });
      b.focus();
    },
    onMessage(m) {
      if (games.onMessage(m)) return;   // before Ask: its ask_data and games| boards
      if (ask.onMessage(m)) return;   // first: its card searches carry an ask: req the deck builder must not take
      if (db.onMessage(m)) return;
      if (dm.onMessage(m)) return;
      if (m.t === "art_sync") {
        artSync = m;
        // Don't re-render a live board once a second: the panel is only visible in the lobby or Settings.
        if (!msg || msg.t === "lobby" || gameMenu || settingsOpen) render();
        return;
      } else if (m.t === "lobby_decks") {
        if (!lobby) return;
        lobby.decks = m.decks || [];
        // A saved or deleted deck may be the picked one: drop it and let renderLobby pick a default.
        const ids = new Set(lobby.decks.map((d) => d.id));
        if (lobbyPick.you && !ids.has(lobbyPick.you)) lobbyPick.you = null;
        if (lobbyPick.bot && !ids.has(lobbyPick.bot)) lobbyPick.bot = null;
        render();
        return;
      } else if (m.t === "lobby") {
        lobby = MOCK ? mockSavedDecks(m) : m;
        dm.refresh();
        startupPending = false;
        startupFailed = null;
        if (m.card_art != null && !cardArtChosen) {
          cardArt = !!m.card_art;
          setCardArt(cardArt);
        }
        ensureModelPick();
        if (!msg || msg.t === "lobby") msg = m;
        if (puzzles === null && !MOCK && ws) ws.send({t: "puzzles"});
        if (lobbyTab === "people" && rooms === null && !room) scanRooms();
        if (lobbyTab === "decks") db.enter();
        if (lobbyTab === "ask") ask.enter();
        if (lobbyTab === "games") games.enter();
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
      } else if (m.t === "primer") {
        primer = m.primer || null;
        primerDeck = m.deck || "Opponent's deck";
        primerHash = m.hash || null;
        canStudy = !!m.can_study;
        primerLoading = false;
        primerOpen = true;
        gameMenu = false;
        if (primer && primer.study && primer.study.phase === "scored" &&
            studyState && studyState.hash === primerHash) studyState = null;
        render();
        return;
      } else if (m.t === "study_status") {
        if (!m.busy) studyState = m;
        render();
        const finished = (m.phase === "games" && (m.done || 0) >= (m.total || 0)) || m.phase === "done";
        if (finished && primerOpen && primerHash && m.hash === primerHash) sendPrimerRequest();
        return;
      } else if (m.t === "packs") {
        packBusy = !!m.busy;
        if (lobby) lobby.packs = m.packs || null;
        if (lobby && typeof m.update_url === "string") lobby.pack_update_url = m.update_url;
        const p = m.packs || {};
        if (m.busy) say("Checking for a card pack update.");
        else say(m.installed ? "Card pack installed. Restart the server to use it."
          : m.rolled_back ? "Rolled back to the previous card pack."
          : p.state === "active" ? "Card pack " + (p.id || "") + " is installed."
          : "Using the built-in cards.");
        render();
        return;
      } else if (m.t === "room") {
        onRoomFrame(m);
        return;
      } else if (m.t === "rooms") {
        scanning = false;
        rooms = Array.isArray(m.rooms) ? m.rooms : [];
        scanErr = m.error || null;
        say(rooms.length ? rooms.length + (rooms.length === 1 ? " room" : " rooms") + " nearby." : "No rooms nearby.");
        render();
        return;
      } else if (m.t === "error" && (room || roomForm.busy)) {
        // A refused room request (or the room's own error): show it in place, keep the screen.
        roomForm.busy = null;
        roomUi.note = m.text || "error";
        if (msg && msg.t === "state") showBanner(roomUi.note);
        else render();
        return;
      } else if (m.t === "puzzle_solution") {
        startSolution(m.frames || []);
        return;
      } else if (m.t === "state") {
        handleState(m);
        return;
      } else {
        if (m.t === "error") { undoPending = false; hintLoading = false; primerLoading = false; starting = false; packBusy = false; }   // the request was refused
        if (m.t === "error" && (!msg || msg.t === "lobby")) {
          // A refused lobby request: say so, but keep the lobby (render() draws no lobby for an error frame).
          banner.hidden = false;
          banner.textContent = m.text || "error";
          renderLobby();
          return;
        }
        msg = m;
      }
      render();
    },
  });
}

// Test hook (mock screenshots only): web_screens.py simulates a late card-text
// arrival by re-rendering the lobby with an open card detail.
if (MOCK) window.render = render;
// Test hook (mock only): load a real puzzle list into the Puzzles tab.
if (MOCK) window.__setPuzzles = (items) => { puzzles = items; render(); };
// Test hooks (mock only): the duel pause, state-frame intake, cue player and sheet hold.
if (MOCK) {
  window.__state = handleState;
  window.__pause = setPaused;
  window.__paused = () => paused;
  window.__pauseLen = () => pauseQueue.length;
  window.__msg = () => msg;
  window.__cue = cue;
  window.__stage = stage;
  window.__stageActive = () => stage.active();
  window.__stageFocus = () => stageFocus;
  // Test hooks for the attention rules: a reveal arriving now, and a recap.
  window.__revealCard = (uid) => {
    const c = findCard(uid);
    if (!c || isHidden(c)) return false;
    revealQueue = [];
    revealCur = {card: c, actor: "bot", kind: "activate", n: -1, text: "Activates " + (c.name || "")};
    stage.request("reveal");
    render();
    return true;
  };
  window.__recap = (r) => { recap = r; render(); };
}

// Wi-Fi <-> mobile-data changes update the sync's permission state (the Network Information API;
// the Android bridge calls __decklightNetChanged from its default-network callback).
{
  const c = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (c && c.addEventListener) c.addEventListener("change", sendArtNetwork);
  window.__decklightNetChanged = sendArtNetwork;
}

boot();
