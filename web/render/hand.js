// Hand, prompt sheet, pile sheet and info panel builders.
import {cardHTML, thumbHTML, esc, isHidden, isMonster} from "./card.js";

export function handHTML(hand, actionable, mock, focus) {
  return (hand || []).map((c) => {
    if (!c) return '<div class="hcard"><div class="card back"></div></div>';
    const act = !!(c.uid && actionable[c.uid] != null);
    return '<div class="hcard' + (act ? " lift" : "") + '" data-key="hand:' + esc(c.uid) + '">' +
      cardHTML(c, {mock, actionable: act, focus: c.uid && c.uid === focus}) + "</div>";
  }).join("");
}

// uid -> option index for every option that points at a card on the board or in hand.
export function actionableMap(prompt) {
  const m = {};
  if (!prompt || !prompt.options) return m;
  if (prompt.kind !== "idle" && prompt.kind !== "battle") return m;
  for (const o of prompt.options) {
    if (o.card && o.card.uid && o.where && o.verb !== "select" && m[o.card.uid] == null) m[o.card.uid] = o.index;
  }
  return m;
}

const PRIMARY = ["activate", "normal", "special", "attack", "direct", "yes", "set", "set_spell", "position"];
const SECONDARY = ["pass", "to_battle", "to_main2", "end_turn", "no"];

// A span, not a button: it lives inside the option's <button>, and nested
// <button> would auto-close the outer one in the HTML parser. It still has a
// 44px hit area and is reachable by the .infobtn selector.
export function infoBtn(optIndex) {
  return '<span class="infobtn" role="button" tabindex="0" data-detailcard="' + optIndex + '" aria-label="Card details">ⓘ</span>';
}

function rowHTML(opt, mock, selected, pickNo, expanded) {
  const cls = ["opt"];
  if (PRIMARY.includes(opt.verb)) cls.push("primary");
  if (SECONDARY.includes(opt.verb)) cls.push("secondary");
  if (selected) cls.push("sel");
  if (expanded) cls.push("open");
  const eff = opt.effect ? '<span class="ef">' + esc(opt.effect) + "</span>" : "";
  const th = opt.card ? thumbHTML(opt.card, mock) : "";
  const info = opt.card && !isHidden(opt.card) ? infoBtn(opt.index) : "";
  return '<button class="' + cls.join(" ") + '" data-opt="' + opt.index + '">' + th +
    '<span class="tx"><span class="lb">' + esc(opt.label || opt.verb) + "</span>" + eff + "</span>" +
    (pickNo ? '<span class="pick">' + pickNo + "</span>" : "") + info + "</button>";
}

function headHTML(prompt, mock, sub) {
  const ask = prompt.card ? thumbHTML(prompt.card, mock, "ask") : "";
  return '<div class="grab"></div><div class="stitle">' + ask + '<div class="tt"><span class="t">' +
    esc(prompt.text || "") + "</span>" + (sub ? '<span class="sub">' + esc(sub) + "</span>" : "") + "</div></div>";
}

// Bottom-sheet body per prompt kind. Returns "" for idle/battle (no sheet: cards glow instead).
export function sheetHTML(prompt, mock, sel, expanded) {
  if (!prompt || prompt.kind === "idle" || prompt.kind === "battle") return "";
  sel = sel || [];
  const opts = prompt.options || [];
  if (prompt.kind === "position") {
    const cards = opts.filter((o) => o.verb !== "pass").map((o) => {
      const def = /def/i.test(o.label || "");
      const base = o.card || prompt.card;
      if (!base) return '<button class="bigbtn" data-opt="' + o.index + '">' + esc(o.label) + "</button>";
      const c = Object.assign({}, base, {pos: def ? "def" : "atk"});
      return '<button class="poscard" data-opt="' + o.index + '"><span class="pv">' +
        '<span class="pcard' + (def ? " rot" : "") + '">' + cardHTML(c, {mock}) + "</span>" + infoBtn(o.index) + "</span>" +
        '<span class="plab">' + (def ? "Defense" : "Attack") + "<b>" + (def ? base.def : base.atk) + "</b></span></button>";
    }).join("");
    return headHTML(prompt, mock) + '<div class="sbody"><div class="posrow">' + cards + "</div></div>" + footHTML(prompt, sel);
  }
  if (prompt.kind === "card") {
    const picks = opts.filter((o) => o.verb === "select" || o.verb === "unselect");
    const done = picks.filter((o) => o.verb === "unselect").map((o) => o.index);
    const sub = prompt.min === prompt.max ? "Pick " + prompt.min : "Pick " + prompt.min + "–" + prompt.max;
    const grid = picks.map((o) => {
      const at = done.indexOf(o.index);
      const info = o.card && !isHidden(o.card) ? infoBtn(o.index) : "";
      return '<button class="cell" data-opt="' + o.index + '">' +
        '<span class="cbox">' + cardHTML(o.card, {mock, selected: at >= 0, pick: at >= 0 ? at + 1 : null}) + info + "</span>" +
        '<span class="cname">' + esc((o.card && o.card.name) || o.label) + "</span></button>";
    }).join("");
    return headHTML(prompt, mock, sub) + '<div class="sbody"><div class="selectgrid">' + grid + "</div></div>" +
      footHTML(prompt, sel);
  }
  let body = "";
  if (prompt.kind === "announce") {
    body = '<div class="announce"><input type="search" placeholder="Name a card…" aria-label="Name a card"></div>';
  }
  body += opts.filter((o) => o.verb !== "finish" && o.verb !== "cancel" && !SECONDARY.includes(o.verb)).map((o) =>
    rowHTML(o, mock, sel.includes(o.index), sel.indexOf(o.index) + 1 || null, expanded === o.index)).join("");
  return headHTML(prompt, mock) + '<div class="sbody">' + body + "</div>" + footHTML(prompt, sel);
}

function footHTML(prompt, sel) {
  const opts = prompt.options || [];
  const rest = opts.filter((o) => SECONDARY.includes(o.verb));
  const fin = opts.find((o) => o.verb === "finish");
  const cancel = opts.find((o) => o.verb === "cancel");
  if (!rest.length && !fin && !cancel) return "";
  let h = '<div class="sfoot">';
  for (const o of rest) h += '<button class="bigbtn" data-opt="' + o.index + '">' + esc(o.label) + "</button>";
  if (cancel) h += '<button class="bigbtn" data-opt="' + cancel.index + '">' + esc(cancel.label || "Cancel") + "</button>";
  if (fin) {
    const n = opts.filter((o) => o.verb === "unselect").length;   // the env offers finish only when it is legal
    h += '<button class="bigbtn primary" data-opt="' + fin.index + '">Confirm' +
      '<span class="cnt">' + n + "/" + (prompt.max || prompt.min || 0) + "</span></button>";
  }
  return h + "</div>";
}

// GY / banished / Extra list; each cell opens the card detail.
export function pileSheetHTML(pile, cards, mock) {
  const cells = (cards || []).slice().reverse().map((c) =>
    '<button class="cell" data-detailuid="' + esc(c ? c.uid : "") + '"><span class="cbox">' +
    (c ? cardHTML(c, {mock}) : '<div class="card back"></div>') +
    (c && !isHidden(c) ? '<span class="infobtn" role="button" tabindex="0" data-detailuid="' + esc(c.uid) + '" aria-label="Card details">ⓘ</span>' : "") +
    "</span>" +
    '<span class="cname">' + esc((c && c.name) || "Face-down") + "</span></button>").join("");
  return '<div class="grab"></div><div class="stitle"><div class="tt"><span class="t">' + esc(pileTitle(pile)) +
    '</span><span class="sub">' + (cards || []).length + " cards · newest first</span></div></div>" +
    '<div class="sbody">' + (cells ? '<div class="selectgrid">' + cells + "</div>" : '<p class="emptytx">Empty</p>') +
    '</div><div class="sfoot"><button class="bigbtn" data-act="closepile">Close</button></div>';
}

export function pileTitle(pile) {
  const [kind, who] = (pile || "").split(":");
  const owner = who === "bot" ? "ygo-zero's " : "Your ";
  return owner + ({gy: "Graveyard", banished: "banished cards", extra: "Extra Deck"}[kind] || kind);
}

export function statsLine(c) {
  if (!c || isHidden(c) || !isMonster(c)) return "";
  const parts = [];
  if (c.link) parts.push("Link " + c.link);
  else if (c.rank) parts.push("Rank " + c.rank);
  else if (c.level) parts.push("Level " + c.level);
  parts.push("ATK " + c.atk);
  if (!c.link) parts.push("DEF " + c.def);
  return parts.join(" · ");
}

// Info panel under your bar: the focused card, else the latest duel events.
// info: {card, text, typeStr} for a focused card; events: newest last.
export function infoHTML(info, events, mock, hint) {
  if (info && info.card) {
    const c = info.card;
    const st = statsLine(c);
    const meta = [info.typeStr, st].filter(Boolean).join(" · ");
    return '<div class="focuscard">' + thumbHTML(c, mock, "big") +
      '<div class="fx"><div class="fname">' + esc(c.name || "Face-down card") + "</div>" +
      (meta ? '<div class="fmeta">' + esc(meta) + "</div>" : "") +
      '<div class="ftext">' + (info.text ? esc(info.text) : '<span class="muted">' +
        (mock ? "Card text loads from the server in a live game." : isHidden(c) ? "" : "Loading card text…") + "</span>") +
      "</div></div>" +
      '<button class="iconbtn sm" data-act="unfocus" aria-label="Close card">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>';
  }
  const evs = (events || []).slice(-8);
  const rows = evs.map((e) => {
    const actor = e.actor === "you" || e.actor === "bot" ? e.actor : "system";
    return '<div class="ev ' + actor + '"' + (e.card && e.card.uid ? ' data-focusuid="' + esc(e.card.uid) + '"' : "") + ">" +
      '<span class="who">' + (actor === "you" ? "You" : actor === "bot" ? "Bot" : "") + "</span>" +
      '<span class="et">' + esc(e.text || "") + "</span></div>";
  }).join("");
  return '<div class="feed" data-act="log"><div class="evs">' + rows + "</div>" +
    (hint ? '<div class="hint">' + esc(hint) + "</div>" : "") + "</div>";
}
