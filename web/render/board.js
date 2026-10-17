// Board HTML builders: player bars, the 7-column zone rows, piles, the EMZ strip.
// Columns (point-symmetric). The two outer columns are narrower pile columns:
//   you monster: [Field] [M1..M5] [GY]      you S/T: [Extra] [S1..S5] [Deck]
//   bot S/T:     [Deck] [S5..S1] [Extra]    bot monster: [GY] [M5..M1] [Field]
import {cardHTML, thumbHTML, esc, isHidden} from "./card.js";

function slotHTML(card, ctx, extraCls) {
  const cls = "slot" + (extraCls ? " " + extraCls : "") + (card ? "" : " empty");
  if (!card) return '<div class="' + cls + '"></div>';
  return '<div class="' + cls + '">' + cardHTML(card, ctx) + "</div>";
}

function topOf(list) {
  for (let i = (list || []).length - 1; i >= 0; i--) if (list[i]) return list[i];
  return null;
}

// Narrow pile cell: face-down back or the top card, with a count pill.
function pileHTML(o) {
  const n = o.count || 0;
  const open = o.openable && n ? ' data-pile="' + o.pile + ":" + o.who + '"' : "";
  const el = open ? "button" : "div";
  let face = "";
  if (n) {
    face = o.top && !isHidden(o.top)
      ? cardHTML(o.top, {mock: o.mock}).replace('class="card', 'aria-hidden="true" class="card')
      : '<div class="card back"></div>';
  }
  return "<" + el + ' class="slot side pile' + (n ? "" : " empty") + '"' + open +
    ' aria-label="' + esc(o.label + " " + n) + '">' + face +
    (n ? '<span class="cnt">' + n + "</span>" : '<span class="ptag">' + esc(o.tag) + "</span>") +
    "</" + el + ">";
}

function gyHTML(side, who, mock) {
  const gy = side.gy || [];
  const bn = (side.banished || []).length;
  const ban = bn
    ? '<button class="banpill" data-pile="banished:' + who + '" aria-label="Banished ' + bn + '"><span class="bp" aria-hidden="true">' +
      '<svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.5"/><path d="M3 9 9 3"/></svg>' + bn + "</span></button>"
    : "";
  return '<div class="gycell">' +
    pileHTML({tag: "GY", label: "Graveyard", count: gy.length, top: topOf(gy), mock, pile: "gy", who, openable: true}) +
    ban + "</div>";
}

function deckHTML(side, who) {
  return pileHTML({tag: "Deck", label: "Deck", count: side.deck || 0, pile: "deck", who});
}

function extraHTML(side, who) {
  const n = side.extra_count != null ? side.extra_count : (side.extra || []).length;
  return pileHTML({tag: "Extra", label: "Extra Deck", count: n, pile: "extra", who, openable: who === "you"});
}

// One side's two rows. `mirror` = bot (point-symmetric).
export function sideHTML(side, opts) {
  opts = opts || {};
  const act = opts.actionable || {};
  const who = opts.mirror ? "bot" : "you";
  const ctx = (c, zone) => ({
    mock: opts.mock,
    actionable: !!(c && c.uid && act[c.uid] != null),
    focus: !!(c && c.uid && c.uid === opts.focus),
    rot: zone === "m" && (c && (c.pos === "def" || c.pos === "set")),
    stat: zone === "m",
    kb: true,
  });
  const m = side.mzone.map((c) => slotHTML(c, ctx(c, "m")));
  const s = side.szone.map((c) => slotHTML(c, ctx(c, "s")));
  const field = slotHTML(side.field, ctx(side.field, "f"), "side fieldz");
  const gy = gyHTML(side, who, opts.mock);
  if (opts.mirror) {
    return '<div class="zrow" data-row="szone">' + deckHTML(side, who) + s.slice().reverse().join("") + extraHTML(side, who) + "</div>" +
      '<div class="zrow" data-row="mzone">' + gy + m.slice().reverse().join("") + field + "</div>";
  }
  return '<div class="zrow" data-row="mzone">' + field + m.join("") + gy + "</div>" +
    '<div class="zrow" data-row="szone">' + extraHTML(side, who) + s.join("") + deckHTML(side, who) + "</div>";
}

const PHASES = [
  ["draw", "Draw"],
  ["standby", "Standby"],
  ["main1", "Main 1"],
  ["battle", "Battle"],
  ["main2", "Main 2"],
  ["end", "End"],
];

// EMZ strip on the same 7-column grid: phase in columns 1-2, EMZ over M2 and M4
// (columns 3 and 5), turn number in columns 6-7. The EMZ is shared: your
// emz[0]/[1] are your left/right, the bot's indices are mirrored.
export function midHTML(view, opts) {
  opts = opts || {};
  const yours = view.turn_player === "you";
  const names = opts.names || null;
  const at = PHASES.findIndex(([k]) => k === view.phase);
  const track = PHASES.map((_, i) => '<i class="' + (i < at ? "done" : i === at ? "cur" : "") + '"></i>').join("");
  const name = at >= 0 ? PHASES[at][1] : view.phase;
  const emzSlot = (mine, theirs) => {
    const c = mine || theirs;
    const ctx = {
      mock: opts.mock, stat: true, kb: true,
      actionable: !!(c && c.uid && (opts.actionable || {})[c.uid] != null),
      focus: !!(c && c.uid && c.uid === opts.focus),
      rot: !!(c && (c.pos === "def" || c.pos === "set")),
    };
    return slotHTML(c, ctx, "emz" + (theirs && !mine ? " bot" : ""));
  };
  const you = view.you.emz || [null, null];
  const bot = view.bot.emz || [null, null];
  return (
    '<div class="phase ' + (yours ? "you" : "bot") + '">' +
    '<span class="who">' + (names ? esc(names[yours ? 0 : 1]) + "’s turn" : (yours ? "Your turn" : "Bot's turn")) + "</span>" +
    '<span class="pname">' + esc(name) + "</span>" +
    '<span class="track" aria-hidden="true">' + track + "</span></div>" +
    emzSlot(you[0], bot[1]) + emzSlot(you[1], bot[0]) +
    '<div class="turnno"><span>Turn</span><b>' + view.turn + "</b></div>"
  );
}

// Phase buttons inside your bar (idle and battle prompts only).
export function phaseBtnsHTML(prompt, locked) {
  if (!prompt || (prompt.kind !== "idle" && prompt.kind !== "battle")) return "";
  const btns = (prompt.options || []).filter((o) => ["to_battle", "to_main2", "end_turn"].includes(o.verb));
  if (!btns.length) return "";
  return '<span class="phasebtns">' + btns.map((o) =>
    '<button class="phbtn' + (o.verb === "end_turn" ? "" : " primary") + '" data-opt="' + o.index + '"' +
    (locked ? " disabled" : "") + ' aria-label="' + esc(o.label) + '">' +
    esc({to_battle: "Battle", to_main2: "Main 2", end_turn: "End turn"}[o.verb] || o.label) + "</button>"
  ).join("") + "</span>";
}

export function barHTML(side, who, opts) {
  opts = opts || {};
  const frac = Math.max(0, Math.min(1, side.lp / 8000));
  const name = opts.name || (who === "you" ? "You" : "ygo-zero");
  let right = "";
  if (who === "bot") {
    if (opts.thinking) {
      right += '<span class="think"><span class="pulse"></span>thinking ' +
        Math.max(0, Math.round(opts.thinking.seconds || 0)) + "s</span>";
    }
    const n = (side.hand || []).length;
    right += '<span class="handcount" aria-label="Bot hand ' + n + '"><i></i><i></i><b>' + n + "</b></span>";
    right += '<button class="iconbtn" data-act="gamemenu" aria-label="Game menu">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 6h12M4 10h12M4 14h12"/></svg></button>';
  } else {
    right += opts.phaseBtns || "";
    right += '<button class="iconbtn" data-act="log" aria-label="Open log">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 5h12M4 10h12M4 15h8"/></svg></button>';
  }
  return (
    '<div class="pbar ' + who + (opts.active ? " active" : "") + '" data-key="bar-' + who + '">' +
    '<div class="lpblock"><span class="pname">' + esc(name) + "</span>" +
    '<span class="lp" data-lp="' + side.lp + '">' + side.lp + "</span>" +
    '<span class="lpbar" aria-hidden="true"><i style="width:' + Math.round(frac * 100) + '%"></i></span></div>' +
    '<div class="right">' + right + "</div></div>"
  );
}

// Small thumbnail used by the chain strip.
// Chain links: the controller is a shape (your number in a circle, the bot's in a square seal)
// as well as a colour; a negated link keeps a slash. Spoken as one line per link.
export function chainHTML(chain, mock, negated) {
  if (!chain || !chain.length) return "";
  return '<span class="clabel" aria-hidden="true">Chain</span>' + chain.map((link, i) => {
    const c = link.card;
    const newest = i === chain.length - 1 ? " new" : "";
    const neg = negated && negated.has(i + 1);
    const bot = link.controller === "bot";
    const said = "Chain " + (i + 1) + ": " + ((c && c.name) || "a card") + ", " + (bot ? "ygo-zero" : "yours") + (neg ? ", negated" : "");
    return '<div class="link ' + (bot ? "bot" : "you") + newest + (neg ? " neg" : "") + '" role="img" aria-label="' + esc(said) + '"' +
      (c && c.uid ? ' data-uid="' + esc(c.uid) + '"' : "") + ' title="' + esc(((c && c.name) || "") + (neg ? " (negated)" : "")) + '">' +
      thumbHTML(c, mock) + '<span class="cn">' + (i + 1) + "</span>" +
      (neg ? '<span class="negx"></span>' : "") + "</div>";
  }).join("");
}
