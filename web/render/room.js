// Rooms against people (A2b-3c; spec and approved mockup: docs/ui/mocks/room-ui.md / .html).
// P1 the Multiplayer tab, P2/P3 the waiting room (host / guest), rock-paper-scissors and turn order, room errors.
// Pure HTML builders: app.js owns the state and the data-act handlers.
import {esc} from "./card.js";

export const FAIR_NOTE = "Hints, win chance and undo are off in rooms.";
// EDOPro's hand encoding (1 scissors, 2 rock, 3 paper): the order on screen is rock, paper, scissors.
export const HANDS = [[2, "Rock"], [3, "Paper"], [1, "Scissors"]];
const HAND_NAME = {1: "Scissors", 2: "Rock", 3: "Paper"};
// Hand icons: Phosphor "hand-fist", "hand-palm", "hand-peace", light weight (phosphoricons.com, MIT licence,
// @phosphor-icons/core 2.1.1), copied as published. Chosen by the user 2026-11-12 over hand-drawn stones and hands.
const ICON = (d) => '<svg class="rsvg" viewBox="0 0 256 256" aria-hidden="true" fill="currentColor"><path d="' + d + '"/></svg>';
export const HAND_ICON = {
  2: ICON("M200,82H182V64a30,30,0,0,0-54-18A30,30,0,0,0,74,64v0A30,30,0,0,0,26,88v40a102,102,0,0,0,204,0V112A30,30,0,0,0,200,82ZM152,46a18,18,0,0,1,18,18V82H136a14.71,14.71,0,0,0-2,.16V64A18,18,0,0,1,152,46ZM86,64a18,18,0,0,1,36,0v40a18,18,0,0,1-36,0ZM38,88a18,18,0,0,1,36,0v16a18,18,0,0,1-36,0Zm180,40a90,90,0,0,1-180,0h0a30,30,0,0,0,42-6,30,30,0,0,0,45.12,3.3A30.18,30.18,0,0,0,140,139.51,45.92,45.92,0,0,0,122,176a6,6,0,0,0,12,0,34,34,0,0,1,34-34,6,6,0,0,0,0-12H152a18,18,0,0,1-18-18V96a2,2,0,0,1,2-2h64a18,18,0,0,1,18,18Z"),
  3: ICON("M188,90a25.8,25.8,0,0,0-14,4.11V60a26,26,0,0,0-40.59-21.51A26,26,0,0,0,82,44V54.11A26,26,0,0,0,42,76v76a86,86,0,0,0,172,0V116A26,26,0,0,0,188,90Zm14,62a74,74,0,0,1-148,0V76a14,14,0,0,1,28,0v44a6,6,0,0,0,12,0V44a14,14,0,0,1,28,0v68a6,6,0,0,0,12,0V60a14,14,0,0,1,28,0v70.39A46.07,46.07,0,0,0,122,176a6,6,0,0,0,12,0,34,34,0,0,1,34-34,6,6,0,0,0,6-6V116a14,14,0,0,1,28,0Z"),
  1: ICON("M210.51,31a26,26,0,0,0-47.63,6.27L148,92.82,133.12,37.28A26,26,0,0,0,82.89,50.74L93.1,88.82,81.21,86.53a26,26,0,0,0-21.37,45.85,25.81,25.81,0,0,0-9.31,15.15,26.66,26.66,0,0,0-.53,5.32V160a78,78,0,0,0,78,78h.6c42.68-.32,77.4-35.71,77.4-78.9v-3.34a53.86,53.86,0,0,0-12-33.82l19.07-71.2A25.83,25.83,0,0,0,210.51,31ZM95.88,37a14,14,0,0,1,25.65,3.38L137,98a26,26,0,0,0-17.23,5.71c-.38-.7-.79-1.39-1.24-2.07a25.72,25.72,0,0,0-12.18-9.85L94.48,47.63A13.94,13.94,0,0,1,95.88,37Zm-33.59,72.2A14,14,0,0,1,78.87,98.3l20.74,4A14,14,0,1,1,94,129.71l-12-2.31-.72-.14-.4-.07-7.63-1.47a14,14,0,0,1-10.89-16.51Zm0,40.72A14,14,0,0,1,78.66,139l1,.2L88.8,141a14,14,0,1,1-5.63,27.42l-10-2a14,14,0,0,1-10.9-16.51ZM194,159.1c0,36.62-29.39,66.63-65.5,66.9a66.06,66.06,0,0,1-64.61-50.26,25.69,25.69,0,0,0,6.92,2.46h0l10,2a26.79,26.79,0,0,0,5.25.53,26,26,0,0,0,21.64-40.34l-.42-.6a26.26,26.26,0,0,0,5.81-3.52q.57,1.06,1.23,2.07a25.86,25.86,0,0,0,16.52,11.15l8.34,1.66A34.3,34.3,0,0,0,138,160a33.85,33.85,0,0,0,9.71,23.8,6,6,0,0,0,8.57-8.4,22,22,0,0,1-3.64-25.86,6,6,0,0,0-4.1-8.75l-15.34-3.07a14,14,0,1,1,5.63-27.42l21.4,4.27A42.12,42.12,0,0,1,194,155.76Zm7.51-111.47-17.36,64.79a53.6,53.6,0,0,0-21.57-9.61l-4.59-.92,16.48-61.5a14,14,0,0,1,27,7.24Z"),
};

const RESCAN = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M16 10a6 6 0 1 1-2-4.5M16 3v3h-3"/></svg>';
const ring = (on) => (on ? '<span class="ring" aria-hidden="true"></span>' : "");

// P1. o: {deck (picker HTML), paste (paste box HTML), name, form: {address, password, showPw, err, busy},
//         rooms (null = never scanned), scanning, scanErr, note (a refused room request, or null)}
export function peopleHTML(o) {
  const f = o.form;
  let h = '<h2 class="sr-only">Multiplayer</h2>' +
    '<p class="tabintro">Duel other people: host a room or join one.</p>' + o.deck + o.paste;
  h += '<section class="field-group"><label class="flabel" for="rname">Your name</label>' +
    '<input id="rname" class="rinput" maxlength="20" autocomplete="nickname" spellcheck="false" value="' +
    esc(o.name) + '"></section>';
  if (o.note) h += '<p class="ferr" role="alert">' + esc(o.note) + "</p>";
  h += '<section class="rhost" aria-labelledby="rh-h"><h3 id="rh-h">Host a room</h3>' +
    "<p>Friends join you on this Wi-Fi, your hotspot or Tailscale. No internet needed.</p>" +
    '<button class="rbtn" data-act="roomhost"' + (f.busy ? " disabled" : "") +
    (f.busy === "host" ? ' aria-busy="true">Opening…' : ">Host") + "</button></section>";
  const bad = !!f.err;
  h += '<section class="rsec" aria-labelledby="rj-h"><h3 id="rj-h">Join by address</h3>' +
    '<div class="rfield"><label for="raddr">Address</label>' +
    '<input id="raddr" class="rinput mono" inputmode="url" autocapitalize="off" autocomplete="off" spellcheck="false"' +
    ' placeholder="192.0.2.20:7911" value="' + esc(f.address) + '"' +
    (bad ? ' aria-invalid="true" aria-describedby="raddr-err"' : ' aria-describedby="raddr-hint"') + ">" +
    (bad ? '<p id="raddr-err" class="ferr" role="alert">' + esc(f.err) + "</p>"
      : '<p id="raddr-hint" class="rhint">An address like 192.0.2.20, name:7911 or a Tailscale name.</p>') +
    "</div>" +
    // Show/Hide sits inside the field and Join ends the same row (no lone button on a row of its own).
    '<div class="rfield"><label for="rpw">Password</label><div class="pwrow"><div class="pwbox">' +
    '<input id="rpw" class="rinput" type="' + (f.showPw ? "text" : "password") + '" maxlength="20" autocomplete="off"' +
    ' spellcheck="false" placeholder="optional" value="' + esc(f.password) + '">' +
    '<button class="pwshow" data-act="roompw" aria-pressed="' + !!f.showPw + '" aria-label="Show password">' +
    (f.showPw ? "Hide" : "Show") + "</button></div>" +
    '<button class="rbtn" data-act="roomjoin"' + (f.busy ? " disabled" : "") +
    (f.busy === "join" ? ' aria-busy="true">Joining…' : ">Join") + "</button></div></div></section>";
  let list;
  if (o.scanning && !(o.rooms && o.rooms.length)) list = '<p class="rnote">' + ring(true) + "Looking for rooms…</p>";
  else if (o.scanErr) list = '<p class="rnote">' + esc(o.scanErr) + "</p>";
  else if (o.rooms && o.rooms.length) {
    list = '<div class="nearby" role="list">' + o.rooms.map((r) =>
      '<div role="listitem"><button data-act="roomjoinaddr" data-addr="' + esc(r.address) + '"' + (f.busy ? " disabled" : "") +
      ' aria-label="Join ' + esc(r.name) + ", " + esc(r.rule || "") + ", " + esc(r.address) + '">' +
      '<span class="nm">' + esc(r.name) + '</span><span class="meta">' +
      esc([r.seats ? r.seats[0] + "/" + r.seats[1] : "", r.rule, r.address].filter(Boolean).join(" · ")) +
      '</span><span class="go" aria-hidden="true">Join</span></button></div>').join("") + "</div>";
  } else list = '<p class="rnote">No rooms on this network yet.</p>';
  h += '<section class="rsec" aria-labelledby="rn-h"><div class="rhd"><h3 id="rn-h">Nearby rooms</h3>' +
    '<button class="iconbtn' + (o.scanning ? " spin" : "") + '" data-act="roomscan" aria-label="Scan again"' +
    (o.scanning ? ' aria-busy="true"' : "") + ">" + RESCAN + "</button></div>" +
    '<div aria-live="polite">' + list + "</div></section>";
  return h;
}

function seatsHTML(r, ui) {
  const guest = r.role === "guest";
  const rows = (r.seats || []).map((s, i) => {
    if (!s) {
      return '<li class="open"><span class="who">Open seat</span><span class="deck">Waiting for a player</span>' +
        '<span class="st wait">Empty</span></li>';
    }
    const tags = [s.you ? "you" : "", s.host ? "host" : ""].filter(Boolean).join(" · ");
    const deck = s.you ? (ui.deckLabel || s.deck_name || "Your deck") : "Deck hidden until the duel";
    // The host's Start is its own ready (server room.rs RoomCmd::Start): its seat says so, not "Not ready".
    const st = s.ready ? '<span class="st ok">Ready</span>'
      : s.host ? '<span class="st wait">Starts the duel</span>'
        : '<span class="st wait">' + (s.you ? "Not ready" : "Choosing a deck") + "</span>";
    return '<li data-key="seat' + i + '"><span class="who">' + (s.you ? '<span class="you-dot" aria-hidden="true"></span>' : "") +
      esc(s.name || "Duelist") + (tags ? " <small>" + tags + "</small>" : "") + '</span><span class="deck">' + esc(deck) +
      "</span>" + st + "</li>";
  }).join("");
  const me = (r.seats || []).find((s) => s && s.you);
  const myReady = guest && me && me.ready;
  return '<ul class="seats" aria-label="Seats">' + rows + "</ul>" + (guest && r.state === "waiting"
    ? '<button class="rbtn' + (myReady ? " quiet" : "") + ' wide" data-act="roomready" aria-pressed="' + !!myReady + '">' +
      (myReady ? "Not ready" : "Ready") + "</button>"
    : "");
}

function fairHTML(r) {
  return '<div class="fair"><p>' + FAIR_NOTE + "</p>" + (r.note ? "<p>" + esc(r.note) + "</p>" : "") + "</div>";
}

// The status line: words first, the ring only marks "waiting".
function statusText(r, opp) {
  if (r.state === "connecting") return [r.role === "host" ? "Opening the room…" : "Connecting to " + (r.server || "") + "…", true];
  if (r.state === "starting") return ["Starting the duel…", true];
  if (r.state === "dueling") return [r.ask ? "Duel starting" : "Starting the duel…", !r.ask];
  if (r.state === "ended") return ["The duel is over.", false];
  if (r.role === "host") {
    if (!opp) return ["Waiting for a player…", true];
    return opp.ready ? [opp.name + " is ready. Start when you are.", false] : [opp.name + " joined. Waiting for them to be ready…", true];
  }
  const me = (r.seats || []).find((s) => s && s.you);
  return me && me.ready ? ["Ready. Waiting for the host to start…", true] : ["Joined " + (r.server || "") + ". Tap Ready when your deck is set.", false];
}

function askHTML(r, opp) {
  const oppName = (opp && opp.name) || "Your opponent";
  if (r.ask === "rps") {
    let last = "";
    if (r.hand_result) {
      const [mine, theirs] = r.hand_result;
      last = mine === theirs
        ? '<p class="rnote">Both threw ' + esc(HAND_NAME[mine] || "") + ". Throw again.</p>"
        : '<p class="rnote">You threw ' + esc(HAND_NAME[mine] || "") + ", " + esc(oppName) + " threw " + esc(HAND_NAME[theirs] || "") + ".</p>";
    }
    return '<section class="rask" aria-labelledby="ra-h"><h3 id="ra-h">Rock, paper, scissors</h3>' +
      "<p>The winner chooses who goes first.</p>" + last +
      '<div class="hands" role="group" aria-labelledby="ra-h">' +
      HANDS.map(([v, n]) => '<button class="rbtn quiet" data-act="roomrps" data-hand="' + v + '">' + HAND_ICON[v] + "<span>" + n + "</span></button>").join("") +
      "</div></section>";
  }
  if (r.ask === "first") {
    let won = "";
    if (r.hand_result) {
      const [mine, theirs] = r.hand_result;
      won = "<p>You threw " + esc(HAND_NAME[mine] || "") + ", " + esc(oppName) + " threw " + esc(HAND_NAME[theirs] || "") + ". You choose.</p>";
    }
    return '<section class="rask" aria-labelledby="rf-h"><h3 id="rf-h">Who goes first?</h3>' + won +
      '<div class="hands two"><button class="rbtn" data-act="roomfirst" data-tp="1">Go first</button>' +
      '<button class="rbtn quiet" data-act="roomfirst" data-tp="0">Go second</button></div></section>';
  }
  if (r.state === "dueling" && r.hand_result) {
    const [mine, theirs] = r.hand_result;
    return '<p class="rnote">You threw ' + esc(HAND_NAME[mine] || "") + ", " + esc(oppName) + " threw " +
      esc(HAND_NAME[theirs] || "") + (mine === theirs ? "." : ". Waiting for the turn order…") + "</p>";
  }
  return "";
}

// P2 / P3 and the error card. r: the last room frame; ui: {closeArm, copied, note, deckLabel}
export function roomHTML(r, ui) {
  const host = r.role === "host";
  const opp = (r.seats || []).find((s) => s && !s.you) || null;
  const title = host ? "Your room" : (r.state === "connecting" ? "Joining" : (opp && opp.host ? opp.name + "'s room" : r.name || "Room"));
  const leaveLabel = host ? (ui.closeArm ? "Tap again to close" : "Close") : "Leave";
  let h = '<div id="lobby" class="roomview"><div class="lscroll">';
  h += '<header class="rtop"><h1>' + esc(title) + '</h1><button class="rbtn quiet' + (ui.closeArm ? " confirm" : "") +
    '" data-act="roomleave">' + leaveLabel + "</button></header>";
  if (ui.closeArm && opp) h += '<p class="rnote" role="status">Close the room? ' + esc(opp.name) + " will be disconnected.</p>";
  if (r.error && r.state === "ended") {
    // A guest that got the room's rules had joined: the room closed on them, the join itself worked.
    const head = r.role === "host" || r.rules ? "The room closed" : "Could not join " + (r.server || "the room");
    h += '<div class="err" role="alert"><span class="t">' + esc(head) + "</span><p>" + esc(r.error) + "</p>" +
      '<button class="rbtn quiet" data-act="roomleave">Back</button></div>';
    return h + "</div></div>";
  }
  const [st, waiting] = statusText(r, opp);
  const ask = askHTML(r, opp);
  if (host) {
    h += '<section class="room" aria-label="Room"><div class="top"><span class="title">' + esc(r.name || "Your room") +
      '</span><span class="status" role="status" aria-live="polite">' + ring(waiting) + esc(st) + "</span></div>";
    if ((r.addresses || []).length) {
      h += '<p class="rnote">Join from decklight or EDOPro at:</p>' + r.addresses.map((a, i) =>
        '<div class="addr"><code id="raddr' + i + '">' + esc(a) + '</code><button class="rbtn quiet small" data-act="roomcopy"' +
        ' data-addr="' + esc(a) + '" aria-label="Copy ' + esc(a) + '">' + (ui.copied === a ? "Copied" : "Copy") + "</button></div>").join("");
    } else if (r.state !== "connecting") {
      h += '<p class="rnote">No network found. Turn on Wi-Fi, a hotspot or Tailscale, then open the room again.</p>';
    }
    if (r.rules && r.rules.text) h += '<div class="facts"><span class="rnote">Rules</span><span class="rule">' + esc(r.rules.text) + "</span></div>";
    h += fairHTML(r) + "</section>";
  } else {
    h += '<p class="status" role="status" aria-live="polite">' + ring(waiting) + esc(st) + "</p>";
    if (r.rules && r.rules.text) h += '<div class="facts"><span class="rnote">Rules</span><span class="rule">' + esc(r.rules.text) + "</span></div>";
  }
  if (ui.note) h += '<p class="ferr" role="alert">' + esc(ui.note) + "</p>";
  if (ask) h += ask;
  else if (r.state !== "connecting" && r.state !== "dueling") h += seatsHTML(r, ui);
  if (!host) h += fairHTML(r);
  h += "</div>";
  if (host && (r.state === "waiting" || r.state === "starting")) {
    const can = r.state === "waiting" && opp && opp.ready;
    h += '<div id="startwrap"><button id="startbtn" data-act="roomstart"' + (can ? "" : " disabled") +
      (r.state === "starting" ? ' aria-busy="true">Starting…' : ">Start duel") + "</button>" +
      (can || r.state === "starting" ? "" : '<p class="rhint center">Start unlocks when the other player is ready.</p>') + "</div>";
  } else if (r.state === "ended") {
    h += '<div id="startwrap"><button id="startbtn" data-act="roomback">Back to the room</button></div>';
  }
  return h + "</div>";
}

// The game-over box for a room duel (P6). m: the final state frame; ask: the room's current ask.
export function roomOverHTML(m, ask) {
  const opp = (m.room && m.room.opponent) || "Opponent";
  const w = m.result.winner;
  const win = w === "you";
  const head = win ? "Victory" : w === "opponent" ? "Defeat" : m.result.reason === "disconnect" ? "Duel ended" : "Draw";
  const why = m.result.text ? m.result.text
    : m.result.reason === "surrender" ? (win ? opp + " surrendered" : "You surrendered")
      : m.result.reason === "disconnect" ? (win ? opp + " left the duel" : "Connection lost") : "";
  const rematch = ask === "rematch"
    ? '<p class="reason">' + esc(opp) + " can play again. Rematch?</p>" +
      '<div class="row"><button class="bigbtn primary" data-act="roomrematch" data-yes="1">Rematch</button>' +
      '<button class="bigbtn" data-act="roomrematch" data-yes="0">No thanks</button></div>'
    : "";
  return '<div id="gameover" data-key="gameover" role="alertdialog" aria-modal="true" tabindex="-1" aria-label="' + head + '">' +
    '<div class="box ' + (win ? "win" : "loss") + '">' +
    '<div class="kicker">Duel over · turn ' + m.view.turn + "</div><h2>" + head + "</h2>" +
    '<p class="reason">' + esc(why) + "</p>" +
    '<div class="score"><span class="you">You<b>' + m.view.you.lp + "</b></span>" +
    '<span class="bot">' + esc(opp) + "<b>" + m.view.bot.lp + "</b></span></div>" + rematch +
    '<div class="row"><button class="bigbtn' + (rematch ? "" : " primary") + '" data-act="roomback">Back to the room</button>' +
    '<button class="bigbtn" data-act="roomleave">Leave room</button></div>' +
    '<p class="reason small">Saved on this device.</p></div></div>';
}
