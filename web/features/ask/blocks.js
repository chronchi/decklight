// Game blocks under an answer (mocks: docs/ui/mocks/ask/mocks.html ask-replay / ask-botview / ask-sequence, user
// 2026-11-15 "Go ahead with everything"): drawn by the app from the replay and bot_view tool results, never by the
// model. A game card (result, life points by turn), a turn's plays step by step, and the bot's view of a decision.
// The boards come from `ask_frames` (the server replays the saved game) as play-screen frames, shown as stills of the
// play screen itself (user 2026-11-15: "the replays/everything should be like what we have when we play").
import {esc, imgUrl, cardArtOn} from "../../render/card.js";
import {still, stillDone, liveBoard, liveError} from "../../lib/stills.js";

const PHASE = {draw: "Draw", standby: "Standby", main1: "Main 1", battle: "Battle", main2: "Main 2", end: "End"};
export const phaseName = (p) => PHASE[p] || (p ? String(p) : "");
const MAX_BLOCKS = 3;

function parse(s) {
  if (s && typeof s === "object") return s;
  try { return JSON.parse(s); } catch (e) { return null; }
}

// The review's grades (server.rs mark_for_delta), so the chat and the review call the same pick the same thing.
export function grade(delta) {
  if (delta >= -3) return "";
  if (delta >= -7) return "INACCURACY";
  if (delta >= -15) return "MISTAKE";
  return "BLUNDER";
}

// What the answer's tool calls can show, in call order: one block per game, turn and decision.
export function blockSpecs(calls) {
  const out = [];
  const seen = new Set();
  const add = (b) => { if (!seen.has(b.key) && out.length < MAX_BLOCKS) { seen.add(b.key); out.push(b); } };
  const turnsAsked = new Map();
  for (const c of calls) {
    if (c.ok === false || !c.done) continue;
    const r = parse(c.result);
    if (!r || typeof r !== "object" || !r.id) continue;
    if (c.name === "replay" && Array.isArray(r.events) && r.turn != null) {
      if (!turnsAsked.has(r.id)) turnsAsked.set(r.id, new Set());
      turnsAsked.get(r.id).add(r.turn);
    }
  }
  for (const c of calls) {
    if (c.ok === false || !c.done) continue;
    const r = parse(c.result);
    if (!r || typeof r !== "object" || !r.id) continue;
    if (c.name === "replay" && Array.isArray(r.log)) add({kind: "game", key: "g:" + r.id, r, hot: turnsAsked.get(r.id) || new Set()});
    else if (c.name === "replay" && Array.isArray(r.events) && r.turn != null) add({kind: "seq", key: "t:" + r.id + ":" + r.turn, id: r.id, turn: r.turn});
    else if (c.name === "replay" && r.decision != null && r.you) add({kind: "board", key: "d:" + r.id + ":" + r.decision, id: r.id, decision: r.decision});
    else if (c.name === "bot_view" && Array.isArray(r.bot_top)) add({kind: "bot", key: "b:" + r.id + ":" + r.decision, id: r.id, r});
  }
  return out;
}

// The ask_frames request a block needs ({record, turn} or {record, decision}), keyed like the cache.
export function framesNeeded(b) {
  if (b.kind === "seq") return {key: b.key, record: b.id, turn: b.turn};
  if (b.kind === "board" || b.kind === "bot") {
    const d = b.kind === "bot" ? b.r.decision : b.decision;
    return {key: "d:" + b.id + ":" + d, record: b.id, decision: d};
  }
  return null;
}

// ---- the small board ------------------------------------------------------------------------------------------
function mc(c, o) {
  if (!c) return "";
  o = o || {};
  const def = c.p === "def" || (c.p === "set" && o.mon);
  if (!c.c) return '<span class="ab-mc ab-back' + (def ? " ab-def" : "") + '" role="img" aria-label="' + (def ? "Face-down monster" : "Set card") + '"></span>';
  const art = cardArtOn() ? ' style="background-image:url(' + esc(imgUrl(c.c, "small")) + ')"' : "";
  const st = c.a != null && !o.hand ? '<span class="ab-st">' + c.a + "</span>" : "";
  return '<span class="ab-mc' + (art ? "" : " ab-noart") + (def ? " ab-def" : "") + (o.hl ? " ab-hl" : "") + (o.gone ? " ab-gone" : "") +
    '" role="img" aria-label="' + esc((c.n || "Card") + (c.a != null ? ", ATK " + c.a : "") + (o.gone ? ", just left the field" : "")) + '"' + art + ">" + st + "</span>";
}
// A card's small art (option thumbnails on the Games screen's decision page).
export const miniCard = (code, name) => mc({c: code, n: name});

// A still of the play screen (../../lib/stills.js): the board as a duel shows it. Tapping it opens Watch at that
// play (`<ns>:watchat` with the game and its choice count).
function stillHTML(key, f, label, ns, id, choice) {
  const e = still(key, f);
  if (e.error) {
    return '<div class="ab-fail" role="alert">Could not draw the board: ' + esc(e.error) + ' <button class="ak-btn" data-act="' + ns + ':still" data-key="' + esc(key) + '">Try again</button></div>';
  }
  if (!e.url) return '<div class="ab-still ab-sw" role="img" aria-label="' + esc(label) + '"><span>Drawing the board…</span></div>';
  return '<button class="ab-still" data-act="' + ns + ':watchat" data-id="' + esc(id) + '" data-choice="' + esc(choice) + '" aria-label="' + esc(label + ". Opens the replay here") +
    '"><img src="' + esc(e.url) + '" alt=""></button>';
}

// ---- blocks ---------------------------------------------------------------------------------------------------
const ICO = (d) => '<svg class="ak-ico" viewBox="0 0 24 24" aria-hidden="true">' + d + "</svg>";
const PLAY = ICO('<path d="M8 5.5v13l10-6.5z"/>'), PAUSE = ICO('<path d="M9 6v12M15 6v12"/>');
const PREV = ICO('<path d="m14.5 6-6 6 6 6"/>'), NEXT = ICO('<path d="m9.5 6 6 6-6 6"/>');

function loadingHTML(fr, key, what, ns) {
  if (fr && fr.error) {
    return '<div class="ab-fail" role="alert">Could not show ' + what + ": " + esc(fr.error) +
      ' <button class="ak-btn" data-act="' + (ns || "ask") + ':frames" data-key="' + esc(key) + '">Try again</button></div>';
  }
  return '<div class="ab-wait" role="status">Replaying the game…</div>';
}

function spark(r, hot, hotLabel) {
  const log = r.log || [];
  const pts = log.map((t) => ({turn: t.turn, y: (t.lp_at_start || [])[0], o: (t.lp_at_start || [])[1]}));
  const fin = r.final_lp || {};
  pts.push({turn: null, y: fin.you, o: fin.opponent});
  const top = Math.max(8000, ...pts.map((p) => Math.max(p.y || 0, p.o || 0)));
  const W = 340, n = pts.length;
  const x = (i) => 6 + i * (W / Math.max(1, n - 1)), yv = (v) => 6 + (1 - Math.max(0, v || 0) / top) * 40;
  const path = (k) => "M" + pts.map((p, i) => x(i).toFixed(1) + " " + yv(p[k]).toFixed(1)).join(" L");
  const every = n > 14 ? Math.ceil(n / 12) : 1;
  const labels = pts.map((p, i) => (p.turn != null && (i % every === 0) ? '<text x="' + (x(i) - 2).toFixed(1) + '" y="54">' + p.turn + "</text>" : "")).join("");
  const dots = pts.map((p, i) => (p.turn != null && hot.has(p.turn) ? '<circle class="ab-hot" cx="' + x(i).toFixed(1) + '" cy="' + yv(p.y).toFixed(1) + '" r="3.5"/>' : "")).join("");
  const label = "Life points by turn: you ended at " + (fin.you != null ? fin.you : "?") + ", the opponent at " + (fin.opponent != null ? fin.opponent : "?") +
    (hot.size ? "; marked: turn " + [...hot].join(", ") : "");
  return '<svg class="ab-spark" viewBox="0 0 352 56" role="img" aria-label="' + esc(label) + '"><line class="ab-g" x1="6" x2="346" y1="46" y2="46"/>' + labels +
    '<path class="ab-o" d="' + path("o") + '"/><path class="ab-y" d="' + path("y") + '"/>' + dots + "</svg>" +
    '<div class="ab-legend"><span><i class="ab-ly"></i>You</span><span><i class="ab-lo"></i>Opponent</span>' +
    (hot.size ? '<span class="ab-lr"><i class="ab-lh"></i>' + (hotLabel || "the turn" + (hot.size > 1 ? "s" : "") + " this answer looked at") + "</span>" : "") + "</div>";
}

export function gameHTML(b, hotLabel) {
  const r = b.r;
  const res = r.result === "you won" ? ['<span class="ab-res ab-w">WON</span>'] : r.result === "you lost" ? ['<span class="ab-res ab-l">LOST</span>'] : [""];
  const went = r.you_went === "first" ? "you went 1st" : r.you_went === "second" ? "you went 2nd" : "";
  const sub = [r.turns != null ? r.turns + " turns" : "", went, r.ended_by ? "by " + r.ended_by : ""].filter(Boolean).join(" · ");
  return '<div class="ab-rb" role="group" aria-label="' + esc("Game " + (r.your_deck || "") + " against " + (r.opponent_deck || "") + ", " + (r.result || "")) + '">' +
    '<div class="ab-rh"><span class="ab-k">Game</span>' + res[0] + '<span class="ab-r">' + esc(sub) + "</span></div>" +
    '<div class="ab-title">' + esc((r.your_deck || "Your deck") + " vs " + (r.opponent_deck || "opponent")) + "</div>" + spark(r, b.hot, hotLabel) + "</div>";
}

// One step of every playing turn (the chat's and the Games screen's timers): `uis` maps a sequence's key to its
// {i, playing}, `frames(key)` gives its frames. The next play comes only once the board on screen is drawn, so a slow
// phone never runs ahead of its pictures (2026-11-16: "Drawing the board…" at play 11 of 17). True while any plays.
// A turn's controls (prev | play/pause | next) on block `key` of `uis` with `n` plays; returns whether it now plays.
// The block becomes the live board (lib/stills.js liveBoard) and every other block of `uis` goes back to its still.
export function seqAct(uis, key, n, d) {
  const u = uis.get(key);
  for (const [k, x] of uis) if (k !== key) x.board = false;
  u.board = true;
  if (d === "prev") { u.i = Math.max(0, u.i - 1); u.playing = false; }
  else if (d === "next") { u.i = Math.min(n - 1, u.i + 1); u.playing = false; }
  else if (u.playing) u.playing = false;
  else { if (u.i + 1 >= n) u.i = 0; u.playing = n > 1; }
  return u.playing;
}

export function stepSeqs(uis, frames) {
  let any = false;
  for (const [k, u] of uis) {
    if (!u.playing) continue;
    const fr = frames(k);
    // Its plays still on their way (a slow phone): it waits for them. Stopping here, a turn opened before its plays
    // arrived never played.
    if (!fr || fr.pending) { any = true; continue; }
    const n = fr.frames ? fr.frames.length : 0;
    if (u.i + 1 < n && (u.board || stillDone(k + ":" + u.i))) u.i += 1;
    if (u.i + 1 >= n) u.playing = false; else any = true;
  }
  return any;
}

export function seqHTML(b, fr, ui, ns) {
  ns = ns || "ask";
  // b.xp: the Games screen's export action for this turn (an icon in the header; the chat has none).
  const xp = b.xp ? '<button class="ak-ibtn gm-xpi" data-act="' + esc(b.xp) + '" aria-label="Export this turn">' + b.xpIcon + "</button>" : "";
  const head = '<div class="ab-rh"><span class="ab-k">Turn ' + esc(b.turn) + "</span>" + (fr && fr.player === "opponent" ? "Opponent's plays" : "Your plays") + xp +
    (fr && fr.decisions ? '<span class="ab-r">decisions ' + esc(fr.decisions) + "</span>" : "") + "</div>";
  if (!fr || !Array.isArray(fr.frames)) return '<div class="ab-rb" role="group" aria-label="Turn ' + esc(b.turn) + '">' + head + loadingHTML(fr, b.key, "this turn", ns) + "</div>";
  const fs = fr.frames;
  if (!fs.length) return '<div class="ab-rb">' + head + '<div class="ab-wait">No plays on this turn.</div></div>';
  const i = Math.min(fs.length - 1, Math.max(0, ui.i || 0));
  const f = fs[i];
  const who = f.who === "you" ? '<span class="ab-who ab-y">YOU</span>' : f.who === "opp" ? '<span class="ab-who ab-o">OPP</span>' : "";
  const dots = fs.map((_, k) => '<i class="' + (k === i ? "ab-on" : k < i ? "ab-done" : "") + '"></i>').join("");
  const k = esc(b.key);
  const label = "Board, turn " + b.turn + ", play " + (i + 1) + " of " + fs.length + ": " + (f.text || "");
  // The turn being played or stepped through is the live board; a board that could not start says so (Try again).
  const lerr = ui.board && liveError(b.key);
  const board = lerr ? '<div class="ab-fail" role="alert">Could not start the board: ' + esc(lerr) + ' <button class="ak-btn" data-act="' + ns + ':still" data-key="live:' + k + '">Try again</button></div>'
    : (ui.board && liveBoard(b.key, fs.map((x) => x.screen), i, {label, ns, id: fr.id || b.id, choice: f.choice})) || stillHTML(b.key + ":" + i, f.screen, label, ns, fr.id || b.id, f.choice);
  return '<div class="ab-rb" role="group" aria-label="' + esc("Turn " + b.turn + ", play " + (i + 1) + " of " + fs.length) + '">' + head + board +
    '<div class="ab-cap"' + (ui.playing ? "" : ' aria-live="polite"') + ">" + who + "<span>" + esc(f.text) + "</span></div>" +
    '<div class="ab-scrub"><button class="ab-sb" data-act="' + ns + ':seq" data-key="' + k + '" data-do="prev" aria-label="Previous play"' + (i === 0 ? " disabled" : "") + ">" + PREV + "</button>" +
    '<button class="ab-pp" data-act="' + ns + ':seq" data-key="' + k + '" data-do="play" aria-label="' + (ui.playing ? "Pause" : "Play the turn") + '">' + (ui.playing ? PAUSE : PLAY) + "</button>" +
    '<button class="ab-sb" data-act="' + ns + ':seq" data-key="' + k + '" data-do="next" aria-label="Next play"' + (i === fs.length - 1 ? " disabled" : "") + ">" + NEXT + "</button>" +
    '<span class="ab-dots" aria-hidden="true">' + dots + '</span><span class="ab-n">' + (i + 1) + " / " + fs.length + "</span></div>" +
    (fr.more ? '<div class="ab-note">' + fr.more + (ns === "ask" ? " more plays on this turn are in the replay tool's log." : " more plays on this turn are not shown.") + "</div>" : "") + "</div>";
}

function boardBlockHTML(b, fr) {
  const head = '<div class="ab-rh"><span class="ab-k">Decision ' + esc(b.decision) + "</span>" +
    (fr && fr.turn != null ? "Turn " + esc(fr.turn) + " · " + esc(phaseName(fr.phase)) : "") + "</div>";
  if (!fr || !fr.screen) return '<div class="ab-rb">' + head + loadingHTML(fr, b.key, "this decision") + "</div>";
  return '<div class="ab-rb" role="group" aria-label="' + esc("Board at decision " + b.decision) + '">' + head +
    stillHTML(b.key, fr.screen, "Board at decision " + b.decision, "ask", fr.id || b.id, fr.choice) +
    (fr.you_picked ? '<div class="ab-cap"><span class="ab-who ab-y">YOU</span><span>' + esc(fr.you_picked) + "</span></div>" : "") + "</div>";
}

function optHTML(o, tags, cls, best) {
  const th = o.code ? '<div class="ab-th">' + mc({c: o.code, n: o.option}) + "</div>" : '<div class="ab-th ab-nocard">' + NEXT + "</div>";
  return '<div class="ab-opt' + (cls ? " " + cls : "") + '">' + th + '<div class="ab-ol">' + esc(o.option) +
    (tags.length ? '<div class="ab-tags">' + tags.map(([t, c]) => '<span class="ab-tag ' + c + '">' + t + "</span>").join("") + "</div>" : "") +
    '</div><div class="ab-nums"><div class="ab-w">' + esc(o.win_chance) + '%</div><div class="ab-p">bot ' + esc(o.bot_prob) + "%</div></div>" +
    '<div class="ab-bars" aria-hidden="true"><i style="width:' + Math.max(0, Math.min(100, +o.bot_prob || 0)) + '%"></i></div></div>';
}

// `o.ns` names the actions (ask | games); `o.acts` replaces the Watch the turn row; `o.key` is the board's frames key.
export function botHTML(b, fr, ui, seqFr, o) {
  o = o || {};
  const ns = o.ns || "ask";
  const r = b.r;
  const top = r.bot_top || [];
  const yours = r.you_picked || null;
  const rows = top.slice();
  // Options are told apart by index (two copies of a card share a label); label only for results without one.
  const same = (o) => !!yours && (o.index != null && yours.index != null ? o.index === yours.index : o.option === yours.option);
  const yi = rows.findIndex(same);
  if (yours && yi < 0) rows.push(yours);
  const best = Math.max(...rows.map((o) => +o.win_chance || 0));
  const head = '<div class="ab-rh"><span class="ab-k">Decision ' + esc(r.decision) + "</span>Turn " + esc(r.turn) + " · " + esc(phaseName(r.phase)) +
    '<span class="ab-r">bot\'s view</span></div>';
  let h = '<div class="ab-rb" role="group" aria-label="' + esc("Bot's view of decision " + r.decision) + '">' + head;
  const bk = o.key || "d:" + r.id + ":" + r.decision;
  h += fr && fr.screen ? stillHTML(bk, fr.screen, "Board at decision " + r.decision, ns, r.id, fr.choice) : loadingHTML(fr, bk, "the board", ns);
  const now = Math.max(0, Math.min(100, +r.win_chance_now || 0));
  h += '<div class="ab-gauge">Win chance before <div class="ab-bar" aria-hidden="true"><i style="width:' + now + '%"></i></div><b>' + esc(r.win_chance_now) + "%</b></div>";
  h += '<div class="ab-opts">' + rows.map((o, j) => {
    const tags = [];
    const mine = same(o);
    if (j === 0) tags.push(["BOT'S PICK", "ab-bot"]);
    if (mine) {
      tags.push(["YOUR PICK", "ab-you"]);
      const d = Math.round((+o.win_chance || 0) - best);
      const g = grade(d);
      if (g) tags.push(["−" + Math.abs(d) + " PTS · " + g, "ab-bad"]);
    }
    return optHTML(o, tags, mine ? "ab-mine" : j === 0 ? "ab-best" : "", best);
  }).join("") + "</div>";
  h += '<p class="ab-help">Win chance looks one step ahead with the bot on both sides: an estimate, not a solved line.</p>';
  const sk = o.seqKey || "t:" + r.id + ":" + r.turn;
  if (o.acts != null) return h + (o.acts ? '<div class="ab-acts">' + o.acts + "</div>" : "") + "</div>";
  h += '<div class="ab-acts"><button class="ak-btn" data-act="' + ns + ':watch" data-key="' + esc(b.key) + '" aria-expanded="' + !!ui.watch + '">' +
    (ui.watch ? "Hide the turn" : "Watch the turn") + "</button></div></div>";
  if (ui.watch) h += seqHTML({kind: "seq", key: sk, id: r.id, turn: r.turn}, seqFr, ui.seqUi || {}, ns);
  return h;
}

// One block; `frames(key)` is the ask_frames cache, `ui(key)` the block's play state.
export function blockHTML(b, frames, ui) {
  if (b.kind === "game") return gameHTML(b);
  if (b.kind === "seq") return seqHTML(b, frames(b.key), ui(b.key));
  if (b.kind === "board") return boardBlockHTML(b, frames(b.key));
  if (b.kind === "bot") {
    const u = ui(b.key);
    const sk = "t:" + b.r.id + ":" + b.r.turn;
    return botHTML(b, frames("d:" + b.r.id + ":" + b.r.decision), {watch: u.watch, seqUi: ui(sk)}, frames(sk));
  }
  return "";
}
