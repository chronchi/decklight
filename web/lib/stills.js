// Stills of the play screen for the chat's and the Games screen's game blocks (a turn's plays, the board at a
// decision): one hidden replay player (./player.js) draws each board the way a duel shows it, one at a time, into an
// image (user 2026-11-15: replays look like the game, not a different style). Tapping a still opens Watch there.
import {Player} from "./player.js";
import {esc} from "../render/card.js";

export const STILL_W = 360, STILL_H = 640;   // CSS px (9:16, as the exports), drawn at 1.5x for a sharp ~300 px wide picture
const KEEP = 48;                             // stills kept (object URLs); the oldest are dropped first
const STILL_MS = 30000;                      // one still at most: the stills are drawn one at a time, so a hang held all
const cache = new Map();                     // key -> {pending} | {url} | {error}
let player = null;
// Stills waiting to be drawn, one at a time, the newest request first: it is the one on screen. Oldest first, a turn
// playing on a slow phone asked for boards faster than they were drawn and the shown one never came (2026-11-16).
const waiting = [];
let drawing = false;
let changed = () => {};

// The app's render(): called when a still is ready.
export function onStill(fn) { changed = fn; }

// The still of `frame` (a play-screen `state` frame) under `key`: {pending} until it is drawn, then {url} or {error}.
export function still(key, frame) {
  if (cache.has(key)) {
    const e = cache.get(key);
    cache.delete(key);
    cache.set(key, e);   // most recently used last
    return e;
  }
  const e = {pending: true};
  cache.set(key, e);
  if (typeof document === "undefined" || !frame) return e;
  waiting.push([key, frame, e]);
  pump();
  return e;
}

// Whether the still under `key` is drawn (or failed, or was never asked for): a playing turn steps on only then.
export function stillDone(key) {
  const e = cache.get(key);
  return !e || !e.pending;
}

async function pump() {
  if (drawing) return;
  drawing = true;
  while (waiting.length) {
    const [key, frame, e] = waiting.pop();
    if (cache.get(key) !== e) continue;   // dropped from the cache meanwhile
    await draw(frame, e);
  }
  drawing = false;
}

async function draw(frame, e) {
  let timer = null;
  try {
    if (!player) player = new Player(document.body, {width: STILL_W, height: STILL_H, hidden: true, title: "Replay stills"});
    const p = player;
    const job = (async () => {
      await p.load([frame]);
      await p.show(0, {cues: false});
      const c = await p.snap(1.5);
      const blob = await new Promise((res) => c.toBlob(res, "image/png"));
      if (!blob) throw new Error("the picture could not be made");
      return URL.createObjectURL(blob);
    })();
    const late = new Promise((_, rej) => { timer = setTimeout(() => rej(new Error("it took more than " + STILL_MS / 1000 + " s")), STILL_MS); });
    e.url = await Promise.race([job, late]);
  } catch (err) {
    e.error = String((err && err.message) || err);
    // A player that failed or hung is replaced for the next still.
    if (player) { player.destroy(); player = null; }
  } finally {
    clearTimeout(timer);
  }
  delete e.pending;
  while (cache.size > KEEP) {
    const [k, old] = cache.entries().next().value;
    if (old.pending) break;
    if (old.url) URL.revokeObjectURL(old.url);
    cache.delete(k);
  }
  changed();
}

// Forget a still that failed, so the next render tries again (`live:<key>`: the live board).
export function retryStill(key) {
  if (key.startsWith("live:")) { if (lv && "live:" + lv.key === key) lv = null; return; }
  const e = cache.get(key);
  if (e && e.error) cache.delete(key);
}

// ---- the live board ----
// The one turn the user plays or steps through shows the play screen itself, switching frames (near instant), not a
// still per play: each still waited 3.2-4.9 s on the laptop and longer on the phone (2026-11-16: "As it transitions
// from one frame to the next it takes some seconds ... It should be near instant"). One at a time; the other blocks
// keep their stills.
let lv = null;      // {key, screens, want, shown, host, player, busy, err}
let syncing = false;

// The live board's markup for block `key` at frame `i` of `screens` (play-screen `state` frames), from a render: the
// player is put in (or moved on) right after the render lands. o: {label, ns, id, choice} as stillHTML's.
export function liveBoard(key, screens, i, o) {
  if (lv && lv.key !== key) dropLive();
  if (!lv) lv = {key, screens, want: i, shown: -1, host: null, player: null, busy: false, err: null};
  lv.screens = screens;
  lv.want = i;
  if (lv.err) return null;
  if (!syncing) { syncing = true; requestAnimationFrame(() => { syncing = false; syncLive(); }); }
  return '<div class="ab-still ab-live"><div class="ab-lh" data-key="live:' + esc(key) + '" data-keep></div>' +
    '<button class="ab-ltap" data-act="' + o.ns + ':watchat" data-id="' + esc(o.id) + '" data-choice="' + esc(o.choice) + '" aria-label="' + esc(o.label + ". Opens the replay here") + '"></button></div>';
}
// Why the live board could not start, for the block to say so with Try again (null while it works).
export function liveError(key) { return lv && lv.key === key ? lv.err : null; }

function dropLive() {
  if (lv && lv.player) lv.player.destroy();
  lv = null;
}

async function syncLive() {
  const me = lv;
  if (!me || me.busy || typeof document === "undefined") return;
  const host = [...document.querySelectorAll(".ab-lh[data-key]")].find((h) => h.getAttribute("data-key") === "live:" + me.key);
  if (!host) { dropLive(); return; }
  me.busy = true;
  try {
    if (host !== me.host) {
      if (me.player) me.player.destroy();
      me.host = host;
      me.shown = -1;
      me.player = new Player(host, {width: STILL_W, height: STILL_H, title: "Replay board"});
      await me.player.load(me.screens);
    }
    // The player is the stills' fixed size, scaled to the block's width (the same picture as a still).
    me.player.frame.style.transform = "scale(" + host.clientWidth / STILL_W + ")";
    while (lv === me && me.shown !== me.want) {
      const i = me.want;
      await me.player.show(i, {cues: false});
      me.shown = i;
    }
  } catch (err) {
    if (lv === me) { me.err = String((err && err.message) || err); if (me.player) me.player.destroy(); me.player = null; me.host = null; changed(); }
  } finally {
    me.busy = false;
  }
}
if (typeof window !== "undefined") window.addEventListener("resize", () => { if (lv) syncLive(); });
