// Watch: a saved game replayed in the play screen itself (user 2026-11-15: "the replays/everything should be like what
// we have when we play, not a different style"; mock shots scripts/app/watch_mock_shots.py, approved the same day with
// "one play" steps and the opponent's hand shown). The board is ../../lib/player.js (the app in replay mode, fed the
// server's `games:watch` frames, with the duel's own cue animations); this file adds the controls around it: back to
// the start of the turn, one play back, play/pause, one play on, the next turn, the speed, Export and Close, in one bar
// over the board's bottom that hides while the game plays (a tap, a pause or keyboard focus brings it back).
// Its DOM lives outside the app's re-rendered markup (the player's iframe must survive every render).
import {esc} from "../../render/card.js";
import {Player} from "../../lib/player.js";

const KEY = "games:watch:";
const HOLD = {"1": 1100, "2": 550};   // a pause on each play while playing (ms)
const SHOWN_MS = 3000;                 // the bar brought back while playing hides again after this
const ICO = (d, cls) => '<svg class="' + (cls || "w-ico") + '" viewBox="0 0 24 24" aria-hidden="true">' + d + "</svg>";
const I = {
  first: ICO('<path d="M7 6v12M17 6l-7 6 7 6"/>'),
  prev: ICO('<path d="m14.5 6-6 6 6 6"/>'),
  play: ICO('<path d="M8 5.5v13l10-6.5z"/>', "w-ico w-fill"),
  pause: ICO('<path d="M9 6v12M15 6v12"/>'),
  next: ICO('<path d="m9.5 6 6 6-6 6"/>'),
  last: ICO('<path d="M17 6v12M7 6l7 6-7 6"/>'),
  share: ICO('<path d="M12 4v11M8 8l4-4 4 4"/><path d="M6 12v6.5A1.5 1.5 0 0 0 7.5 20h9a1.5 1.5 0 0 0 1.5-1.5V12"/>'),
  close: ICO('<path d="M7 7l10 10M17 7 7 17"/>'),
};
const reduced = () => { try { return matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; } };
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

export function watchView(ctx) {
  // ctx = {send(msg), mock, act(kind, el) (the Games feature's actions), exportFrom({id, title, lastTurn, turn, play, playsByTurn})}
  let n = 0;
  let w = null;   // {id, at, req, data: {pending}|{error}|result, i, playing, shown, hideT, speed, busy, want, root, player, opener}

  function ask() {
    n += 1;
    w.req = n;
    w.data = {pending: true};
    if (!ctx.mock) ctx.send({t: "ask_data", key: KEY + n, what: "watch", args: {record: w.id}});
  }

  // at: {choice} (a still's play, a decision) or {turn}.
  function open(id, at) {
    if (w) close();
    if (typeof document === "undefined") return;
    const root = document.createElement("div");
    root.id = "watch";
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-label", "Replay");
    root.setAttribute("data-close", "games:wclose");
    root.tabIndex = -1;
    root.innerHTML = '<div class="w-stage"></div><div class="w-tap" aria-hidden="true"></div><div class="w-msg"></div><div class="w-bar"></div><div class="w-line" aria-hidden="true"><i></i></div><p class="w-sr" aria-live="polite"></p><div class="w-sheet"></div>';
    document.body.appendChild(root);
    root.addEventListener("click", onClick);
    // Tabbing into the hidden bar shows it, and it stays while a control has keyboard focus.
    root.addEventListener("focusin", (ev) => { if (w && w.playing && !w.shown && keyFocus()) reveal(); });
    w = {id, at: at || {}, i: 0, playing: false, shown: false, hideT: 0, speed: "1", busy: false, want: null, root, player: null,
         opener: document.activeElement, data: null};
    ask();
    paint();
    root.focus();
  }

  function close() {
    if (!w) return false;
    const o = w.opener;
    w.playing = false;
    clearTimeout(w.hideT);
    if (w.player) w.player.destroy();
    w.root.remove();
    w = null;
    if (o && o.isConnected) o.focus({preventScroll: true});
    return true;
  }

  function onReply(mkey, m) {
    if (!w || +mkey !== w.req) return;
    if (m.error) { w.data = {error: m.error}; paint(); return; }
    w.data = m.result;
    start().catch((e) => { if (w) { w.data = {error: String((e && e.message) || e)}; paint(); } });
  }

  async function start() {
    const r = w.data;
    const fs = r.frames || [];
    if (!fs.length) throw new Error("the replay has no plays");
    w.root.setAttribute("aria-label", "Replay: " + r.title);
    let i = 0;
    if (w.at.choice != null) {
      const k = fs.findIndex((f) => f.choice >= +w.at.choice);
      i = k < 0 ? fs.length - 1 : k;
    } else if (w.at.turn != null) {
      const k = fs.findIndex((f) => f.turn === +w.at.turn && f.stop);
      i = k < 0 ? 0 : k;
    }
    w.i = i;
    w.player = new Player(w.root.querySelector(".w-stage"), {title: "Replay board"});
    paint();
    const me = w;
    await me.player.load(fs, {speed: me.speed});
    await me.player.show(i, {cues: false});
    if (w !== me) return;
    w.ready = true;
    paint();
  }

  // ---- moving through the game ----
  const frames = () => (w && w.data && w.data.frames) || [];
  function nextStop(i) { const fs = frames(); for (let k = i + 1; k < fs.length; k++) if (fs[k].stop) return k; return null; }
  function prevStop(i) { const fs = frames(); for (let k = i - 1; k >= 0; k--) if (fs[k].stop) return k; return null; }
  function turnStart(turn) { return frames().findIndex((f) => f.turn === turn && f.stop); }

  // Show frame `target`: the plays in between with their cue animations (`animate`), or the board at once.
  async function go(target, animate) {
    if (!w || !w.ready || target == null) return;
    if (w.busy) { w.want = {target, animate: false}; return; }
    const me = w;
    me.busy = true;
    try {
      if (animate && target > me.i && !reduced()) {
        for (let j = me.i + 1; j <= target; j++) {
          if (me.want || w !== me) break;
          await me.player.show(j, {});
          me.i = j;
          paint();
        }
      } else {
        await me.player.show(target, {cues: false});
        me.i = target;
      }
    } finally {
      me.busy = false;
    }
    if (w !== me) return;
    paint(true);
    if (me.want) { const x = me.want; me.want = null; await go(x.target, x.animate); }
  }

  // The bar while playing: hidden, or brought back for SHOWN_MS (kept while a control has keyboard focus).
  const keyFocus = () => !!w.root.querySelector(".w-bar :focus-visible");
  function reveal() {
    if (!w) return;
    clearTimeout(w.hideT);
    w.shown = true;
    const me = w;
    if (me.playing) me.hideT = setTimeout(function hide() {
      if (w !== me || !me.playing) return;
      if (keyFocus()) { me.hideT = setTimeout(hide, SHOWN_MS); return; }
      me.shown = false;
      paint();
    }, SHOWN_MS);
    paint();
  }

  async function play() {
    const me = w;
    me.playing = true;
    if (keyFocus()) reveal(); else me.shown = false;
    paint();
    if (nextStop(me.i) == null) await go(0, false);
    while (w === me && me.playing) {
      if (typeof document !== "undefined" && document.hidden) { me.playing = false; break; }
      const k = nextStop(me.i);
      if (k == null) { me.playing = false; break; }
      await go(k, true);
      if (w !== me || !me.playing) break;
      await sleep(HOLD[me.speed]);
    }
    if (w === me) paint(true);
  }

  async function setSpeed() {
    w.speed = w.speed === "1" ? "2" : "1";
    paint();
    if (!w.ready) return;
    // Only the speed changes: reloading the frames here reset a play in progress, and Watch waited on it for good.
    await w.player.speed(w.speed);
  }

  function onClick(ev) {
    // The export sheet inside Watch's layer: its games:* actions go to the Games feature (app.js only routes clicks
    // inside the app's own markup).
    const sa = ev.target.closest && ev.target.closest(".w-sheet [data-act]");
    if (sa) { ctx.act(sa.getAttribute("data-act"), sa); return; }
    if (ev.target.closest && ev.target.closest(".w-tap")) { reveal(); return; }
    // A control used while playing keeps the bar up for another SHOWN_MS.
    if (w && w.playing && ev.target.closest && ev.target.closest(".w-bar")) reveal();
    const b = ev.target.closest && ev.target.closest("[data-w]");
    if (!b || !w || b.disabled) return;
    const a = b.getAttribute("data-w");
    const fs = frames();
    if (a === "close") { close(); return; }
    if (a === "retry") { if (w.player) { w.player.destroy(); w.player = null; } w.ready = false; ask(); paint(); return; }
    if (a === "export") { w.playing = false; paint(); ctx.exportFrom(exportInfo()); return; }
    if (a === "speed") { setSpeed(); return; }
    if (a === "play") { if (w.playing) { w.playing = false; paint(); } else play(); return; }
    w.playing = false;
    const f = fs[w.i];
    if (a === "next") go(nextStop(w.i), true);
    else if (a === "prev") go(prevStop(w.i), false);
    else if (a === "first") {
      const s = turnStart(f.turn);
      go(s >= 0 && s < w.i ? s : turnStart(f.turn - 1), false);
    } else if (a === "last") {
      const s = turnStart(f.turn + 1);
      go(s >= 0 ? s : fs.length - 1, false);
    }
  }

  function exportInfo() {
    const r = w.data, fs = frames(), f = fs[w.i];
    const playsByTurn = {};
    for (const x of fs) if (x.stop) playsByTurn[x.turn] = (playsByTurn[x.turn] || 0) + 1;
    const inTurn = fs.filter((x) => x.turn === f.turn && x.stop);
    const k = inTurn.filter((x) => x.choice <= f.choice).length;
    return {id: w.id, title: r.title, lastTurn: r.last_turn, turn: f.turn, playsByTurn,
            play: {k: Math.max(0, k - 1), n: inTurn.length, choice: f.choice, text: (f.caption && f.caption.text) || "Turn " + f.turn + " starts"}};
  }

  // ---- drawing the controls ----
  function paint(announce) {
    if (!w) return;
    const R = w.root;
    const msg = R.querySelector(".w-msg"), bar = R.querySelector(".w-bar");
    const d = w.data;
    // The bar is drawn anew: the control that had focus gets it back (keyboard use lost it on every play).
    const fa = typeof document !== "undefined" && document.activeElement;
    const fk = fa && bar.contains(fa) ? fa.getAttribute("data-w") : null;
    const refocus = () => { const b = fk && bar.querySelector('[data-w="' + fk + '"]'); if (b && !b.disabled) b.focus({preventScroll: true}); else if (fk) R.focus({preventScroll: true}); };
    R.classList.toggle("hidebar", !!(w.ready && w.playing && !w.shown));
    const closeB = '<button class="w-btn w-x" data-w="close" aria-label="Close the replay">' + I.close + "</button>";
    const exportB = '<button class="w-btn w-xp" data-w="export" aria-label="Export"' + (w.ready ? "" : " disabled") + ">" + I.share + "</button>";
    if (d && d.error) {
      msg.innerHTML = '<div class="w-card" role="alert"><b>Could not open the replay</b><p>' + esc(d.error) + '.</p><div class="w-acts">' +
        '<button class="ak-btn" data-w="retry">Try again</button><button class="ak-btn" data-w="close">Close</button></div></div>';
      bar.innerHTML = "";
      return;
    }
    if (!w.ready) {
      msg.innerHTML = '<div class="w-card" role="status">Replaying the game…</div>';
      bar.innerHTML = '<div class="w-row">' + closeB + exportB + "</div>";
      refocus();
      return;
    }
    msg.innerHTML = "";
    const fs = frames(), f = fs[w.i], last = d.last_turn;
    const inTurn = fs.filter((x) => x.turn === f.turn && x.stop);
    const k = Math.max(1, inTurn.filter((x) => x.choice <= f.choice).length);
    const pct = (100 * w.i) / Math.max(1, fs.length - 1);
    const atEnd = nextStop(w.i) == null, atStart = prevStop(w.i) == null;
    const btn = (a, ico, label, off, cls) => '<button class="w-btn' + (cls ? " " + cls : "") + '" data-w="' + a + '" aria-label="' + label + '"' + (off ? " disabled" : "") + ">" + ico + "</button>";
    bar.innerHTML = '<div class="w-meta"><div class="w-prog" aria-hidden="true"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
      '<span class="w-pos">Turn ' + f.turn + " of " + last + " · " + k + "/" + inTurn.length + '</span></div><div class="w-row">' + closeB +
      btn("first", I.first, "Start of the turn", atStart) + btn("prev", I.prev, "Previous play", atStart) +
      btn("play", w.playing ? I.pause : I.play, w.playing ? "Pause" : atEnd ? "Play from the start" : "Play", false, "w-play") +
      btn("next", I.next, "Next play", atEnd) + btn("last", I.last, "Next turn", atEnd) +
      '<button class="w-btn w-sp" data-w="speed" aria-label="Speed ' + (w.speed === "2" ? "double, tap for normal" : "normal, tap for double") + '">' + w.speed + "×</button>" + exportB + "</div>";
    R.querySelector(".w-line > i").style.width = pct.toFixed(1) + "%";
    refocus();
    if (announce && !w.playing) {
      const c = f.caption;
      R.querySelector(".w-sr").textContent = "Turn " + f.turn + ", play " + k + " of " + inTurn.length + (c ? ": " + (c.actor ? c.actor + ", " : "") + (c.text || "") : "");
    }
  }

  if (typeof document !== "undefined") document.addEventListener("visibilitychange", () => { if (document.hidden && w) { w.playing = false; paint(); } });

  return {
    open, close, onReply,
    get isOpen() { return !!w; },
    // The socket came back: a replay asked on the old one is asked again.
    enter() { if (w && w.data && w.data.pending) ask(); },
  };
}
