// Games > Export (mocks docs/ui/mocks/ask/mocks.html export-*; user 2026-11-15: "it can export anything, either whole
// game or parts of it ... yrp, MP4 or gif or anything easily shareable"). One sheet for every entry point: the game's
// header (the whole game), a turn's play sequence, Watch (that turn, or the play on screen) and Select mode (several
// games). The server makes the EDOPro replay and the text (rust/decklight-server/src/ask/export.rs); the video, GIF and
// image are the play screen itself (user 2026-11-15: "the replays/everything should be like what we have when we
// play"): ../../lib/player.js replays the server's `games:watch` frames off screen with the duel's cue animations, and
// each picture is a snapshot of it, encoded with WebCodecs (H.264, ../../lib/mp4.js) or ../../lib/gif.js.
// The files leave through Android's share sheet (ShareBridge.kt) or, in a browser, as a download.
import {esc} from "../../render/card.js";
import {Player} from "../../lib/player.js";
import {muxMp4} from "../../lib/mp4.js";
import {GifWriter} from "../../lib/gif.js";

const KEY = "games:export:";
// [key, label, pause on each play (ms), cue animation length (x the duel's)]
const SPEEDS = [["slow", "Slow", 1800, 1.5], ["normal", "Normal", 1100, 1], ["fast", "Fast", 500, 0.6]];
const TICK_MS = 100;            // the video's frame interval (10 fps): a held board repeats, an animation samples
const CUE_SAMPLES = 4;         // pictures per cue scene at most (each snapshot costs ~0.1 s on the laptop, more on a phone)
const END_HOLD_MS = 2000;       // the last play stays on screen a little longer
const GIF_MAX_PLAYS = 80;       // past this a GIF is tens of MB: pick fewer turns
const W = 360, H = 640;         // the player's size in CSS pixels (9:16); video 2x, image 3x, GIF 1x
const CODECS = ["avc1.42001f", "avc1.4d001f", "avc1.640028"];
const ICO = (d) => '<svg class="ak-ico" viewBox="0 0 24 24" aria-hidden="true">' + d + "</svg>";
export const SHARE = ICO('<path d="M12 4v11M8 8l4-4 4 4"/><path d="M6 12v6.5A1.5 1.5 0 0 0 7.5 20h9a1.5 1.5 0 0 0 1.5-1.5V12"/>');

const FORMATS = {
  mp4: {tag: "MP4", name: "Video", verb: "Make video", ready: "Video ready"},
  gif: {tag: "GIF", name: "Animated GIF", verb: "Make GIF", ready: "GIF ready"},
  png: {tag: "PNG", name: "Image", verb: "Make image", ready: "Image ready"},
  yrp: {tag: "YRP", name: "EDOPro replay (.yrpX)", verb: "Make replay", ready: "Replay ready"},
  txt: {tag: "TXT", name: "Text", verb: "Make text", ready: "Text ready"},
};

function secs(ms) {
  const s = Math.round(ms / 1000);
  return s < 60 ? s + " s" : Math.floor(s / 60) + " min " + (s % 60 ? (s % 60) + " s" : "");
}
function sizeWord(n) {
  return n < 1024 ? n + " B" : n < 1048576 ? Math.round(n / 1024) + " KB" : (n / 1048576).toFixed(1) + " MB";
}

const visible = () => (typeof document === "undefined" || !document.hidden ? Promise.resolve()
  : new Promise((res) => document.addEventListener("visibilitychange", function on() { if (!document.hidden) { document.removeEventListener("visibilitychange", on); res(); } })));

async function pickCodec() {
  if (typeof VideoEncoder === "undefined") return null;
  for (const codec of CODECS) {
    const cfg = {codec, width: W * 2, height: H * 2, bitrate: 2500000, framerate: 1000 / TICK_MS, avc: {format: "avc"}};
    try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) return cfg; } catch (e) { /* next */ }
  }
  return null;
}

// ---- the sheet --------------------------------------------------------------------------------------------------
export function exportSheet(ctx) {
  // ctx = {send(msg), render(), mock}
  let n = 0;
  let x = null;   // the open sheet: {ids, game, turns, turn, play, what, as, from, to, reveal, speed, job}

  const send = (m) => { if (!ctx.mock) ctx.send(m); };
  function ask(args, what) {
    n += 1;
    x.req = n;
    send({t: "ask_data", key: KEY + n, what: what || "export", args});
  }

  // opts: {ids, game (its page, one game) or info (Watch: {lastTurn, playsByTurn}), turn (an open turn), play
  // ({k, n, choice, text} on screen), what}
  function open(opts) {
    const one = opts.ids.length === 1 && !!(opts.game || opts.info);
    x = {
      ids: opts.ids, game: one ? opts.game || null : null, info: one ? opts.info || null : null, turn: opts.turn || null, play: opts.play || null,
      what: one ? opts.what || "game" : "games", as: one ? (opts.what === "moment" ? "png" : "mp4") : "yrp",
      from: opts.turn || 1, to: opts.turn || 1, reveal: true, speed: "normal", job: null,
    };
    const last = lastTurn();
    if (x.what === "game") { x.from = 1; x.to = last; }
    if (!formatOk(x.as)) x.as = Object.keys(FORMATS).find(formatOk) || "txt";
  }
  const lastTurn = () => (x.info ? x.info.lastTurn : x.game && x.game.log && x.game.log.length ? x.game.log[x.game.log.length - 1].turn : 1);
  function range() {
    if (x.what === "game") return [1, lastTurn()];
    if (x.what === "turn" || x.what === "moment") return [x.turn, x.turn];
    return [Math.min(x.from, x.to), Math.max(x.from, x.to)];
  }
  // Plays in the range, from the game page's turn log (an estimate: a frame can hold several logged events).
  function playsIn() {
    const [a, b] = range();
    let k = 0;
    if (x.info) {
      for (const [t, c] of Object.entries(x.info.playsByTurn || {})) if (+t >= a && +t <= b) k += c;
      return k;
    }
    for (const t of (x.game && x.game.log) || []) {
      if (t.turn < a || t.turn > b) continue;
      const ev = t.events || [];
      const m = typeof ev[ev.length - 1] === "string" && /^\.\.\. (\d+) more/.exec(ev[ev.length - 1]);
      k += 1 + (m ? ev.length - 1 + +m[1] : ev.length);
    }
    return k;
  }
  const speed = () => SPEEDS.find((s) => s[0] === x.speed);
  const stepMs = () => speed()[2];
  function formatOk(f) {
    if (x.what === "games") return f === "yrp" || f === "txt";
    if (x.what === "moment") return f !== "mp4" && f !== "gif";
    if (f === "gif") return playsIn() <= GIF_MAX_PLAYS;
    return true;
  }
  function formatDesc(f) {
    const [a, b] = range();
    const plays = playsIn();
    if (f === "mp4") return "The play screen with its animations · at least " + secs(plays * stepMs() + END_HOLD_MS) + " · up to " + plays + " plays";
    if (f === "gif") return formatOk("gif") ? "Loops in any chat, no sound · at least " + secs(plays * stepMs()) : "Too long for a GIF (" + plays + " plays): pick a turn or a few turns";
    if (f === "png") return x.what === "moment" ? "The board at this play, with its caption" : b > a ? "One image a turn: the board after its last play" : "The board after the turn's last play";
    if (f === "yrp") {
      if (x.what === "games") return "One file per game, sent together";
      if (x.what === "moment") return "EDOPro replays always start at turn 1: this one plays up to this play and stops";
      return b >= lastTurn() ? "Open it in EDOPro: Replay, pick the file, Start" : "EDOPro replays always start at turn 1: this one plays turns 1–" + b + " and stops";
    }
    return x.what === "games" ? "All " + x.ids.length + " games’ plays in one text" : "The plays as text, to paste anywhere";
  }

  // ---- render ----
  const row = (f) => {
    const on = x.as === f, ok = formatOk(f), F = FORMATS[f];
    return '<button class="gx-row' + (on ? " sel" : "") + '" role="radio" aria-checked="' + on + '" data-act="games:xpas" data-f="' + f + '"' + (ok ? "" : ' aria-disabled="true" disabled') + ">" +
      '<span class="gx-ai" aria-hidden="true">' + F.tag + '</span><span class="gx-tx"><span class="gx-an">' + F.name + '</span><span class="gx-ad">' + esc(formatDesc(f)) + "</span></span>" +
      '<span class="gx-radio' + (on ? " on" : "") + '" aria-hidden="true"></span></button>';
  };
  const seg = (label, act, cur, opts) => '<div class="gx-seg" role="radiogroup" aria-label="' + label + '">' + opts.map(([k, l]) =>
    '<button role="radio" aria-checked="' + (cur === k) + '"' + (cur === k ? ' class="sel"' : "") + ' data-act="games:' + act + '" data-v="' + k + '">' + l + "</button>").join("") + "</div>";
  const step = (which, v) => '<span class="gx-step"><button data-act="games:xpstep" data-w="' + which + '" data-d="-1" aria-label="' + which + " turn earlier" + '"' + (v <= 1 ? " disabled" : "") + ">‹</button><b>" + v +
    '</b><button data-act="games:xpstep" data-w="' + which + '" data-d="1" aria-label="' + which + " turn later" + '"' + (v >= lastTurn() ? " disabled" : "") + ">›</button></span>";

  function sheet(inner, label) {
    return '<div class="gm-scrim" data-act="games:xpclose"></div><div class="gm-sheet gx" role="dialog" aria-modal="true" aria-label="' + label + '" data-close="games:xpclose"><div class="grab"></div>' + inner + "</div>";
  }

  function jobHTML() {
    const j = x.job, F = FORMATS[x.as];
    if (j.error) {
      const alt = x.as === "mp4" ? '<button class="ak-btn" data-act="games:xpas" data-f="gif">Make a GIF</button><button class="ak-btn" data-act="games:xpas" data-f="yrp">EDOPro replay</button>' : "";
      return sheet('<h2>Export</h2><div class="ab-fail" role="alert"><b>' + esc(j.error.title) + "</b><br>" + esc(j.error.text) + "</div>" +
        '<div class="gm-acts">' + alt + '<button class="ak-btn" data-act="games:xpback">Back</button></div>', "Export");
    }
    if (j.files) {
      const total = j.files.reduce((s, f) => s + f.size, 0);
      const names = j.files.length === 1 ? esc(j.files[0].name) : j.files.length + " files";
      const copy = j.text != null ? '<button class="ak-btn" data-act="games:xpcopy">' + (j.copied ? "Copied" : "Copy text") + "</button>" : "";
      return sheet("<h2>" + F.ready + '</h2><div class="gx-ready"><div class="gx-rn">' + names + '</div><div class="gm-note">' + esc([j.detail, sizeWord(total)].filter(Boolean).join(" · ")) + "</div></div>" +
        (j.shareError ? '<div class="ab-fail" role="alert">' + esc(j.shareError) + "</div>" : "") +
        '<div class="gx-foot"><span class="gm-note">' + (typeof window !== "undefined" && window.DecklightShare ? "Opens Android’s share sheet: send it in a chat, or save it to Files or Drive." : "Saves to your downloads.") + "</span>" + copy +
        '<button class="ak-btn pri" data-act="games:xpshare">' + (j.files.length > 1 ? "Share or save files" : "Share or save file") + "</button></div>" +
        '<button class="ak-link gx-again" data-act="games:xpback">Export something else</button>', F.ready);
    }
    const p = j.total ? Math.round((100 * j.done) / j.total) : 0;
    return sheet("<h2>Making the " + (x.as === "mp4" ? "video" : x.as === "gif" ? "GIF" : x.as === "png" ? "image" : "file") + "</h2>" +
      '<div class="gx-prog" role="progressbar" aria-valuenow="' + (j.done || 0) + '" aria-valuemin="0" aria-valuemax="' + (j.total || 0) + '" aria-label="Plays drawn"><i style="width:' + p + '%"></i></div>' +
      '<p class="gm-note" role="status">' + esc(j.step || "") + "</p>" +
      (x.as === "mp4" || x.as === "gif" || x.as === "png" ? '<p class="gm-note">The app plays the game in the play screen and records it. If you leave the app it pauses here and carries on when you come back.</p>' : "") +
      '<div class="gx-foot"><span></span><button class="ak-btn" data-act="games:xpcancel">Cancel</button></div>', "Making the export");
  }

  function html() {
    if (!x) return "";
    if (x.job) return jobHTML();
    const one = x.what !== "games";
    let h = "<h2>" + (one ? "Export" : "Export " + x.ids.length + " games") + "</h2>";
    if (one) {
      const chips = [["game", "Whole game · " + lastTurn() + " turns"]];
      if (x.turn) chips.push(["turn", "Turn " + x.turn]);
      chips.push(["range", "Turns…"]);
      if (x.play) chips.push(["moment", "One play"]);
      h += '<div class="gx-sec">What</div><div class="gx-chips" role="radiogroup" aria-label="What to export">' + chips.map(([k, l]) =>
        '<button role="radio" aria-checked="' + (x.what === k) + '"' + (x.what === k ? ' class="sel"' : "") + ' data-act="games:xpwhat" data-v="' + k + '">' + l + "</button>").join("") + "</div>";
      if (x.what === "range") h += '<div class="gx-rng">From turn ' + step("from", x.from) + " to turn " + step("to", x.to) + "</div>";
      if (x.what === "moment") h += '<p class="gm-note">Turn ' + esc(x.turn) + " · play " + (x.play.k + 1) + " of " + x.play.n + ": “" + esc(x.play.text) + "”. Move the replay to pick another play.</p>";
    }
    h += '<div class="gx-sec">As</div><div role="radiogroup" aria-label="Format">' + Object.keys(FORMATS).filter((f) => one || formatOk(f)).map(row).join("") + "</div>";
    if (one && x.as !== "yrp" && x.as !== "txt") {
      h += '<div class="gx-opt"><span>Opponent’s hand<span class="gx-ad">' + (x.reveal ? "Shown face up, as in the replay" : "As you saw it in the duel: face down until played") + "</span></span>" +
        seg("Opponent hand", "xpreveal", x.reveal ? "1" : "0", [["0", "Hidden"], ["1", "Shown"]]) + "</div>";
    }
    if (one && (x.as === "mp4" || x.as === "gif")) h += '<div class="gx-opt"><span>Speed</span>' + seg("Speed", "xpspeed", x.speed, SPEEDS.map(([k, l]) => [k, l])) + "</div>";
    if (x.as === "yrp") h += '<p class="gm-note">EDOPro replays hold both decks and every card drawn, like any EDOPro replay.</p>';
    const note = x.as === "mp4" ? "Made on this phone. Keep the app open while it draws (it pauses if you leave)." : x.as === "gif" ? "Made on this phone." : "";
    h += '<div class="gx-foot"><span class="gm-note">' + note + '</span><button class="ak-btn pri" data-act="games:xpmake"' + (formatOk(x.as) ? "" : " disabled") + ">" + FORMATS[x.as].verb + "</button></div>";
    return sheet(h, "Export");
  }

  // ---- making ----
  function fail(title, text) {
    if (x && x.job) { x.job = {error: {title, text}}; ctx.render(); }
  }
  function make() {
    const [a, b] = range();
    const job = {done: 0, total: 0, step: "Replaying the game…"};
    x.job = job;
    if (x.as === "mp4" || x.as === "gif" || x.as === "png") {
      // The play screen's frames (replay.rs `watch`), one game.
      const w = {record: x.ids[0], reveal: x.reveal};
      if (x.what !== "game") { w.from_turn = a; w.to_turn = b; }
      ask(w, "watch");
      return;
    }
    const args = {ids: x.ids, as: x.as};
    if (x.what !== "game" && x.what !== "games") { args.from_turn = a; args.to_turn = b; }
    if (x.what === "moment" && x.as === "yrp") { delete args.from_turn; delete args.to_turn; args.upto_choice = Math.max(1, x.play.choice); }
    if (x.what === "moment" && x.as === "txt") { args.from_turn = args.to_turn = x.turn; }
    ask(args);
  }
  function onReply(mkey, m) {
    if (!x || !x.job || +mkey !== x.req) return;
    if (m.error) { fail("The export stopped", m.error + "."); return; }
    const r = m.result;
    if (r.files) {
      const files = r.files.map((f) => {
        const bin = atob(f.data);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return {name: f.name, mime: f.mime, bytes, size: bytes.length};
      });
      x.job = {files, text: r.text != null ? r.text : null, detail: x.as === "yrp" ? (files.length > 1 ? files.length + " replays" : "EDOPro replay") : ""};
      ctx.render();
      return;
    }
    draw(r).catch((e) => fail("The export stopped", String(e && e.message || e) + ". Try again; if it keeps failing, export the EDOPro replay or the text instead."));
  }

  async function draw(r) {
    const job = x.job;
    const fs = r.frames || [];
    if (!fs.length) throw new Error("no plays to draw in these turns");
    const stops = fs.filter((f) => f.stop).length;
    if (x.as === "gif" && stops > GIF_MAX_PLAYS) throw new Error(stops + " plays is too long for a GIF: pick a turn or a few turns");
    const [a, b] = range();
    const base = r.title + " " + r.day + (x.what === "game" ? "" : x.what === "moment" ? " - turn " + x.turn + " play " + (x.play.k + 1) : a === b ? " - turn " + a : " - turns " + a + "-" + b);
    const clean = base.replace(/[^\p{L}\p{N} .,_()-]/gu, "").replace(/\s+/g, " ").trim() || "game";
    const [, , hold, slow] = speed();
    const tick = (k, total) => { job.done = k; job.total = total; job.step = "Recording play " + k + " of " + total; ctx.render(); };
    const player = new Player(document.body, {width: W, height: H, hidden: true, title: "Export"});
    const live = () => x && x.job === job;
    try {
      await player.load(fs, {speed: "1"});
      if (x.as === "png") {
        // One play, or the board at the end of each turn.
        let pick;
        if (x.what === "moment") {
          const k = fs.findIndex((f) => f.choice >= x.play.choice);
          pick = [k < 0 ? fs.length - 1 : k];
        } else {
          pick = [];
          fs.forEach((f, i) => { if (i === fs.length - 1 || fs[i + 1].turn !== f.turn) pick.push(i); });
        }
        const files = [];
        tick(0, pick.length);
        for (let k = 0; k < pick.length; k++) {
          await visible();
          if (!live()) return;
          await player.show(pick[k], {cues: false});
          const c = await player.snap(3);
          const blob = await new Promise((res) => c.toBlob(res, "image/png"));
          const bytes = new Uint8Array(await blob.arrayBuffer());
          files.push({name: clean + (pick.length > 1 ? " - turn " + fs[pick[k]].turn : "") + ".png", mime: "image/png", bytes, size: bytes.length});
          tick(k + 1, pick.length);
        }
        x.job = {files, detail: files.length > 1 ? files.length + " images · 1080 × 1920" : "1080 × 1920"};
        ctx.render();
        return;
      }
      // Video and GIF: the board at the start, then each frame's cue scenes sampled every TICK_MS and the board after
      // each play held for the speed's pause (the zone and target picks inside a play are not held).
      const scale = x.as === "mp4" ? 2 : 1;
      const out = x.as === "mp4" ? await videoOut(clean) : gifOut(clean);
      if (!out) return;
      let ms = 0;
      const emit = async (dur) => { const c = await player.snap(scale); await out.add(c, dur); ms += dur; };
      tick(0, stops);
      let done = 0;
      await player.show(0, {cues: false});
      await emit(hold);
      for (let i = 1; i < fs.length; i++) {
        await visible();
        if (!live()) { out.cancel(); return; }
        const plan = await player.plan(i);
        for (let k = 0; k < plan.length; k++) {
          const ticks = Math.max(1, Math.round((plan[k] * slow) / TICK_MS));
          const steps = Math.min(ticks, CUE_SAMPLES);
          for (let t = 0; t < steps; t++) {
            await player.show(i, {cue: k, t: t / steps});
            await emit(TICK_MS * (Math.floor((ticks * (t + 1)) / steps) - Math.floor((ticks * t) / steps)));
          }
        }
        if (fs[i].stop || i === fs.length - 1) {
          await player.show(i, {cues: false});
          await emit(i === fs.length - 1 ? hold + END_HOLD_MS : hold);
          if (fs[i].stop) tick(++done, stops);
        }
      }
      const res = await out.done();
      if (!live()) return;
      x.job = {files: [res.file], detail: secs(ms) + " · " + res.size};
      ctx.render();
    } finally {
      player.destroy();
    }
  }

  // GIF: identical boards in a row become one frame with a longer delay.
  function gifOut(clean) {
    const gif = new GifWriter(W, H);
    let last = null, lastMs = 0;
    const flush = () => { if (last) gif.add(last, lastMs); last = null; lastMs = 0; };
    return {
      async add(c, dur) {
        const px = c.getContext("2d").getImageData(0, 0, W, H).data;
        if (last && last.length === px.length && same(last, px)) { lastMs += dur; return; }
        flush();
        last = px; lastMs = dur;
      },
      cancel() {},
      async done() {
        flush();
        const bytes = gif.done();
        return {file: {name: clean + ".gif", mime: "image/gif", bytes, size: bytes.length}, size: W + " × " + H};
      },
    };
  }
  function same(p, q) {
    for (let i = 0; i < p.length; i += 4) if (p[i] !== q[i] || p[i + 1] !== q[i + 1] || p[i + 2] !== q[i + 2]) return false;
    return true;
  }

  // MP4: a steady TICK_MS frame rate (players and chat apps expect one); a held board repeats (a few bytes a frame).
  async function videoOut(clean) {
    const cfg = await pickCodec();
    if (!cfg) {
      fail("This phone cannot make MP4 video", "Its video encoder refused H.264 at " + W * 2 + " × " + H * 2 + " (WebCodecs). Make a GIF of a few turns instead, or export the EDOPro replay and record it there.");
      return null;
    }
    const samples = [];
    let avcC = null, encErr = null, t = 0, frame = 0;
    const enc = new VideoEncoder({
      output: (chunk, md) => {
        if (md && md.decoderConfig && md.decoderConfig.description && !avcC) avcC = new Uint8Array(md.decoderConfig.description.slice ? md.decoderConfig.description.slice(0) : md.decoderConfig.description);
        const data = new Uint8Array(chunk.byteLength);
        chunk.copyTo(data);
        samples.push({data, key: chunk.type === "key", ts: chunk.timestamp});
      },
      error: (e) => { encErr = e; },
    });
    enc.configure(cfg);
    return {
      async add(c, dur) {
        if (encErr) throw encErr;
        for (let k = 0, reps = Math.max(1, Math.round(dur / TICK_MS)); k < reps; k++) {
          const vf = new VideoFrame(c, {timestamp: t * 1000, duration: TICK_MS * 1000});
          enc.encode(vf, {keyFrame: frame % 50 === 0});
          vf.close();
          t += TICK_MS;
          frame++;
          while (enc.encodeQueueSize > 4) await new Promise((res) => setTimeout(res, 10));
        }
      },
      cancel() { try { enc.close(); } catch (e) { /* closed */ } },
      async done() {
        await enc.flush();
        enc.close();
        if (encErr) throw encErr;
        if (!avcC) throw new Error("the video encoder gave no H.264 configuration");
        samples.sort((p, q) => p.ts - q.ts);
        const bytes = muxMp4({width: W * 2, height: H * 2, avcC, samples: samples.map((s) => ({data: s.data, key: s.key, dur: TICK_MS}))});
        return {file: {name: clean + ".mp4", mime: "video/mp4", bytes, size: bytes.length}, size: W * 2 + " × " + H * 2};
      },
    };
  }

  // ---- leaving the app ----
  function share() {
    const j = x.job;
    j.shareError = null;
    const B = typeof window !== "undefined" ? window.DecklightShare : null;
    if (B && typeof B.send === "function") {
      const err = (e) => { if (e) throw new Error(e); };
      try {
        err(B.reset());
        for (const f of j.files) {
          err(B.begin(f.name));
          for (let i = 0; i < f.bytes.length; i += 1 << 18) {
            let s = "";
            const part = f.bytes.subarray(i, i + (1 << 18));
            for (let k = 0; k < part.length; k += 8192) s += String.fromCharCode.apply(null, part.subarray(k, k + 8192));
            err(B.append(f.name, btoa(s)));
          }
        }
        const mimes = [...new Set(j.files.map((f) => f.mime))];
        err(B.send(JSON.stringify(j.files.map((f) => f.name)), mimes.length === 1 ? mimes[0] : "*/*"));
      } catch (e) {
        j.shareError = "Could not open the share sheet: " + (e.message || e) + ". Try again; if it keeps failing, restart the app.";
      }
      return;
    }
    if (B && typeof B.shareYdk === "function") {
      j.shareError = "This version of the app can only share decks: update the app to share exports.";
      return;
    }
    // A browser: one download per file, spaced out (Chrome drops a burst of downloads from one click).
    j.files.forEach((f, i) => setTimeout(() => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([f.bytes], {type: f.mime}));
      a.download = f.name;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    }, i * 600));
  }

  function act(a, el) {
    const v = el && el.getAttribute("data-v");
    if (a === "xpclose") { if (x && x.job && !x.job.files && !x.job.error) x.job = null; x = null; return; }
    if (!x) return;
    if (a === "xpwhat") {
      x.what = v;
      if (v === "range" && x.from === x.to) { x.from = Math.max(1, (x.turn || 1)); x.to = Math.min(lastTurn(), x.from + 2); }
      if (!formatOk(x.as)) x.as = Object.keys(FORMATS).find(formatOk);
    } else if (a === "xpas") { x.as = el.getAttribute("data-f"); x.job = null; if (!formatOk(x.as)) x.as = Object.keys(FORMATS).find(formatOk); }
    else if (a === "xpstep") {
      const w = el.getAttribute("data-w"), d = +el.getAttribute("data-d");
      x[w] = Math.max(1, Math.min(lastTurn(), x[w] + d));
      if (x.from > x.to) { if (w === "from") x.to = x.from; else x.from = x.to; }
      if (!formatOk(x.as)) x.as = Object.keys(FORMATS).find(formatOk);
    } else if (a === "xpreveal") x.reveal = v === "1";
    else if (a === "xpspeed") x.speed = v;
    else if (a === "xpmake") { if (formatOk(x.as)) make(); }
    else if (a === "xpcancel" || a === "xpback") x.job = null;
    else if (a === "xpshare") share();
    else if (a === "xpcopy") {
      const t = x.job && x.job.text;
      if (t != null && navigator.clipboard) navigator.clipboard.writeText(t).then(() => { if (x && x.job) { x.job.copied = true; ctx.render(); } }, () => { x.job.shareError = "Could not copy: share the file instead."; ctx.render(); });
    }
  }

  return {
    open, html, act, onReply,
    get isOpen() { return !!x; },
    close() { if (!x) return false; act("xpclose"); return true; },
    // The socket came back: a reply asked on the old one is lost.
    enter() { if (x && x.job && !x.job.files && !x.job.error && !x.job.total) fail("The export stopped", "the connection to the app's server dropped. Make it again."); },
  };
}
