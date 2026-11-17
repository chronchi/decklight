// Ask tab markup (mocks: docs/ui/mocks/ask/mocks.html, approved 2026-11-14). Pure functions of the feature's state.
import {esc} from "../../render/card.js";

const ICO = (d, cls) => '<svg class="ak-ico' + (cls ? " " + cls : "") + '" viewBox="0 0 24 24" aria-hidden="true">' + d + "</svg>";
export const I = {
  list: ICO('<path d="M4 7h16M4 12h16M4 17h10"/>'),
  plus: ICO('<path d="M12 5v14M5 12h14"/>'),
  back: ICO('<path d="m14.5 6-6 6 6 6"/>'),
  next: ICO('<path d="m9.5 6 6 6-6 6"/>'),
  down: ICO('<path d="m6 9.5 6 6 6-6"/>'),
  send: ICO('<path d="M12 19V5M6 11l6-6 6 6"/>'),
  stop: ICO('<rect x="7" y="7" width="10" height="10" rx="1.5"/>'),
  retry: ICO('<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4"/>'),
  copy: ICO('<rect x="8" y="8" width="11" height="11" rx="2"/><path d="M5 15V6a1 1 0 0 1 1-1h9"/>'),
  edit: ICO('<path d="M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4"/>'),
  trash: ICO('<path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/>'),
  warn: ICO('<path d="M12 4 3 19h18zM12 10v4M12 16.5v.5"/>'),
  check: ICO('<path d="m5 12.5 4.5 4.5L19 7"/>'),
  key: ICO('<circle cx="8" cy="15" r="4"/><path d="m11 12 8-8M16 7l2 2"/>'),
};

// The boot mark's geometry (web/assets/boot-mark.svg) cropped to the cards; k0..k2 light up left to right.
export const DMARK = '<svg class="ak-mark" aria-hidden="true" viewBox="70 150 380 250">' +
  [-20, 0, 20].map((a, i) => '<rect class="k' + i + '" x="191" y="167" width="130" height="190" rx="14" transform="rotate(' + a +
    ' 256 499.5)"/>').join("") + '<polyline points="158.5,315.3 227.4,260.1 357.2,221.4"/>' +
  [[158.5, 315.3], [227.4, 260.1], [357.2, 221.4]].map(([x, y], i) => '<circle class="k' + i + '" cx="' + x + '" cy="' + y +
    '" r="' + (i === 2 ? 20 : 15) + '"/>').join("") + "</svg>";

const TOOL_LABEL = {search_cards: "Searched cards", card: "Card", deck: "Opened deck", deck_odds: "Draw odds", my_stats: "Your games", replay: "Replay", bot_view: "Bot's view"};
export const toolLabel = (name) => TOOL_LABEL[name] || name;

function parse(s) {
  if (s && typeof s === "object") return s;
  try { return JSON.parse(s); } catch (e) { return s == null ? null : String(s); }
}

// One short phrase about what a call was for, from its arguments (and its result once known).
export function toolArg(name, args, result) {
  const a = parse(args) || {};
  const r = parse(result);
  const rv = (k) => (r && typeof r === "object" ? r[k] : null);
  if (name === "search_cards") return a.text || "";
  if (name === "card") return rv("name") || (a.code != null ? "#" + a.code : "");
  if (name === "deck") return rv("name") || a.id || "";
  if (name === "deck_odds") return [a.hand ? a.hand + " cards" : "", a.deck_size ? "deck of " + a.deck_size : ""].filter(Boolean).join(", ");
  if (name === "replay") {
    const g = !a.record ? "your games" : a.record === "last" ? "last game" : "game " + String(a.record).slice(0, 8);
    return [g, a.turn ? "turn " + a.turn : "", a.decision ? "decision " + a.decision : ""].filter(Boolean).join(", ");
  }
  if (name === "bot_view") return a.decision ? "decision " + a.decision : "";
  if (name === "my_stats") return [a.mode, a.group_by ? "by " + a.group_by : "", a.opponent_deck].filter(Boolean).join(", ");
  return "";
}

// A tool result as a few readable rows (scalars first; a list shows its size and first names).
function outRows(result) {
  const r = parse(result);
  if (r == null) return '<div class="row"><span>No output</span></div>';
  if (typeof r !== "object") return '<div class="row"><span>' + esc(String(r).slice(0, 400)) + "</span></div>";
  const rows = [];
  for (const [k, v] of Object.entries(r)) {
    if (rows.length >= 10) break;
    let val;
    if (v == null) continue;
    if (Array.isArray(v)) {
      const names = v.slice(0, 3).map((x) => (x && typeof x === "object" ? x.name || x.id || "" : String(x))).filter(Boolean);
      val = v.length + (names.length ? " · " + names.join(", ") + (v.length > 3 ? "…" : "") : "");
    } else if (typeof v === "object") val = JSON.stringify(v).slice(0, 80);
    else val = String(v).length > 120 ? String(v).slice(0, 120) + "…" : String(v);
    rows.push('<div class="row"><span>' + esc(k.replace(/_/g, " ")) + "</span><span>" + esc(val) + "</span></div>");
  }
  return rows.join("") || '<div class="row"><span>Empty</span></div>';
}

// Every {code, name} in tool results (search rows, a card, a deck's cards): the names an answer can link.
export function cardsIn(results) {
  const out = new Map();
  const walk = (v, depth) => {
    if (!v || typeof v !== "object" || depth > 6) return;
    if (Array.isArray(v)) { v.forEach((x) => walk(x, depth + 1)); return; }
    if (typeof v.code === "number" && typeof v.name === "string" && v.name.length >= 4 && !out.has(v.name)) out.set(v.name, v.code);
    for (const x of Object.values(v)) walk(x, depth + 1);
  };
  for (const r of results) walk(parse(r), 0);
  return out;
}

// Card names in an answer become buttons that open the app's card details (user 2026-11-14: "show the card details
// upon clicking the card name in the chat ... like we do everywhere else"). Longest name first, one pass.
// The model marks the cards it names as [[Name]] (system.v2); the server resolves exact names into the answer's
// meta.cards. A marked name it could not match is a dotted link that asks which card it is (option 1, user
// 2026-11-14); while an answer streams, marked names are plain text until the server has looked them up.
function linker(cards, live) {
  cards = cards || new Map();
  const names = [...cards.keys()].sort((a, b) => b.length - a.length);
  const byEsc = new Map(names.map((n) => [esc(n), cards.get(n)]));
  const alts = [...byEsc.keys()].map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp("\\[\\[([^\\]]{1,80})\\]\\]" + (alts.length ? "|" + alts.join("|") : ""), "g");
  const btn = (code, label) => '<button class="ak-card" data-act="ask:card" data-code="' + code + '">' + label + "</button>";
  return (s) => s.replace(re, (m, marked) => {
    if (marked == null) return btn(byEsc.get(m), m);
    const code = byEsc.get(marked.trim());
    if (code) return btn(code, marked);
    return live ? marked : '<button class="ak-card unres" data-act="ask:find" data-name="' + marked + '">' + marked + "</button>";
  });
}
// Answer text without the [[ ]] marks (Copy).
export const plainText = (t) => String(t || "").replace(/\[\[([^\]]{1,80})\]\]/g, "$1");

// Markdown-lite for answers: paragraphs, - / 1. lists, **bold**, `code`, linked card names. Everything is escaped first.
export function mdHTML(text, cards, live) {
  const link = linker(cards, live);
  const inline = (s) => link(esc(s)).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [];
  let list = null;
  const flush = () => { if (list) { out.push("<" + list.tag + ">" + list.items.join("") + "</" + list.tag + ">"); list = null; } };
  let para = [];
  const flushP = () => { if (para.length) { out.push("<p>" + para.map(inline).join("<br>") + "</p>"); para = []; } };
  for (const raw of String(text || "").split("\n")) {
    const line = raw.trimEnd();
    const m = /^\s*(?:([-*•])|(\d+)[.)])\s+(.*)$/.exec(line);
    if (m) {
      flushP();
      const tag = m[1] ? "ul" : "ol";
      if (!list || list.tag !== tag) { flush(); list = {tag, items: []}; }
      list.items.push("<li>" + inline(m[3]) + "</li>");
    } else if (!line.trim()) { flushP(); flush(); }
    else {
      flush();
      const h = /^#{1,4}\s+(.*)$/.exec(line);
      if (h) { flushP(); out.push('<p class="h">' + inline(h[1]) + "</p>"); } else para.push(line);
    }
  }
  flushP(); flush();
  return out.join("");
}

// The tool trail (user 2026-11-14): smaller than the answer; while answering the done calls fold into one line and
// the current call shows on its own; when done, the last three show, a tap shows all, a tap on a call its output.
export function trailHTML(key, calls, s) {
  const {open, outs, running, secs, limit} = s;
  if (!calls.length && !limit) return "";
  const done = calls.filter((c) => c.done);
  const cur = calls.find((c) => !c.done);
  const n = running ? done.length : calls.length;
  let h = '<div class="ak-trail" role="group" aria-label="' + n + " tool call" + (n === 1 ? "" : "s") + '">';
  if (n) {
    h += '<button class="sum" data-act="ask:trail" data-key="' + esc(key) + '" aria-expanded="' + !!open + '">' + n + " tool call" +
      (n === 1 ? "" : "s") + (secs ? " · " + secs + " s" : "") + (open ? I.down : I.next) + "</button>";
  }
  const shown = open ? (running ? done : calls) : running ? [] : calls.slice(-3);
  if (shown.length) {
    h += "<ol>";
    shown.forEach((c, i) => {
      const last = !open && i === shown.length - 1;
      const isOpen = outs.has(c.id);
      h += '<li class="' + (c.ok === false ? "err " : "") + (last ? "last" : "") + '"><button data-act="ask:out" data-id="' + esc(c.id) +
        '" aria-expanded="' + isOpen + '">' + esc(toolLabel(c.name)) + (c.arg ? ' <span class="arg">· ' + esc(c.arg) + "</span>" : "") +
        (open ? (isOpen ? I.down : I.next) : "") + "</button></li>";
      if (open && isOpen) h += '<li class="outli"><div class="out">' + outRows(c.result) + "</div></li>";
    });
    h += "</ol>";
  }
  if (running && cur) {
    h += '<div class="now" role="status"><span class="node"></span>' + esc(toolLabel(cur.name)) +
      (cur.arg ? ' <span class="arg">· ' + esc(cur.arg) + "</span>" : "") + "</div>";
  }
  if (limit) h += '<div class="lim">Used every tool call it may make for one answer; it answers with what it found.</div>';
  return h + "</div>";
}

const ACTIONS = {
  retry: ["retry", "Try again"],
  add_key: ["settings", "Add key"],
  replace_key: ["settings", "Replace key"],
  pick_model: ["model", "Pick a model"],
  switch_provider: ["settings", "Other provider"],
  new_chat: ["new", "New chat"],
  check_settings: ["settings", "Open Settings"],
  cancel: ["cancel", "Stop it"],
};

// An error the user can act on (fail loud with an action, user 2026-11-14).
export function errHTML(e, extra) {
  const a = ACTIONS[e.action];
  let acts = a ? '<button class="ak-btn pri" data-act="ask:do" data-do="' + a[0] + '"' + (e.user ? ' data-user="' + esc(e.user) + '"' : "") +
    ">" + a[1] + "</button>" : "";
  if (extra) acts += extra;
  const stopped = /^Stopped\b/.test(e.message || "");
  return '<div class="ak-err' + (stopped ? " quiet" : "") + '" role="alert"><div class="h">' + (stopped ? "" : I.warn) +
    esc(e.message || "Something failed without a message: try again") + "</div>" + (acts ? '<div class="acts">' + acts + "</div>" : "") + "</div>";
}

export function branchHTML(i, n, act, id) {
  if (n < 2) return "";
  return '<span class="ak-branch"><button data-act="' + act + '" data-id="' + esc(id) + '" data-dir="-1" aria-label="Previous version"' +
    (i <= 0 ? " disabled" : "") + ">" + I.back + "</button>" + (i + 1) + " / " + n + '<button data-act="' + act + '" data-id="' + esc(id) +
    '" data-dir="1" aria-label="Next version"' + (i >= n - 1 ? " disabled" : "") + ">" + I.next + "</button></span>";
}
