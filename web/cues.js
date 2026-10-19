// Pacing cues: plays the server's engine cues (state message `cues`) as short,
// quiet scenes over a held board snapshot, then applies the new snapshot.
//
// Design (docs/web-ux-plan.md §1, night design): no flashes, no white, no glow,
// no shake. Slow fades and slides, one thin accent line per scene. Every scene
// is a pure function of time (`at(ms)`), so it can be played in real time,
// skipped, or frozen at any instant (mock screenshots, game_video.py frames).
//
// Board updates: a state message = one segment {cues, msg}. The segment's cues
// play over the board as it was; cards that land on the field stay as "ghosts"
// and cards that leave are hidden, so the board looks progressive; the real
// snapshot replaces all of that when the segment ends (correct by construction).
import {esc, imgUrl, phHTML} from "./render/card.js";
import {ruleShort} from "./glossary.js";

// Scene lengths at 1x (ms). 2x halves them; "instant" plays nothing.
export const DUR = {
  turn: 1100, phase: 750, draw: 450,
  move: {sent: 550, destroyed: 750, banished: 650, added: 750, returned: 600, material: 550,
    set: 600, summon: 650, spsummon: 650, other: 500},
  summon: 850, actMe: 1000, actOpp: 1400, solve: 650, negate: 1600, target: 650,
  attack: 750, battle: 1150, damage: 800, lp: 450, confirm: 950,
};

const HOW = {
  normal: "Normal Summon", tribute: "Tribute Summon", special: "Special Summon", fusion: "Fusion Summon",
  synchro: "Synchro Summon", xyz: "Xyz Summon", link: "Link Summon", ritual: "Ritual Summon",
  pendulum: "Pendulum Summon", flip: "Flip Summon",
};
const MOVE_LABEL = {destroyed: "Destroyed", banished: "Banished", added: "Added to hand", returned: "Returned",
  set: "Set", sent: "Sent to the GY"};
const EARLY = new Set(["Draw", "Standby", "Main 1"]);
const PHASE_NAME = {Battle: "Battle Phase", "Main 2": "Main Phase 2", End: "End Phase",
  Draw: "Draw Phase", Standby: "Standby Phase", "Main 1": "Main Phase 1"};
const FIELD = new Set(["mzone", "szone", "field", "overlay"]);

// ---------- planning (pure: cues -> steps) ----------

const sameZone = (a, b) => !!a && !!b && a.side === b.side && a.loc === b.loc && (a.seq | 0) === (b.seq | 0);
const sameCard = (a, b) => !!a && !!b && ((a.uid && a.uid === b.uid) || (a.code && a.code === b.code && sameZone(a.zone, b.zone)));
// battle cue sides may be {ref, atk, def} or a ref with atk/def on it
const battleRef = (s) => (s ? s.ref || s.card || s : null);

// Who negated what: chain links by number, the link resolving now and the one
// that resolved last. A negation cue arrives while the negating link resolves
// (Ash: "negate the activation"), or, for effects negated in advance
// (Impermanence), when the negated link itself resolves; the cause is then the
// link just above it, which resolved immediately before.
export function chainTracker(baseChain) {
  const links = new Map();
  (baseChain || []).forEach((l, i) => links.set(i + 1, l.card));
  let solving = 0;
  let last = null;   // {link, card} of the last resolved link
  return {
    feed(c) {
      if (c.t === "activate") links.set(c.link || links.size + 1, c.card);
      else if (c.t === "chain_solving") solving = c.link;
      else if (c.t === "chain_solved") { last = {link: c.link, card: links.get(c.link) || null}; solving = 0; }
      else if (c.t === "chain_end") { links.clear(); solving = 0; last = null; }
      else if (c.t === "chain_negated" || c.t === "chain_disabled") {
        const card = links.get(c.link) || null;
        let by = null;
        if (solving && solving !== c.link) by = links.get(solving) || null;
        else if (last && last.link === c.link + 1) by = last.card;
        return {link: c.link, card, by, disabled: c.t === "chain_disabled"};
      }
      return null;
    },
  };
}

// Steps with dur 0 are model-only (no scene). `chain` is the chain as the
// held board shows it (view.chain), so chain resolution can be drawn.
export function planSteps(cues, baseChain) {
  const steps = [];
  let chain = (baseChain || []).map((l, i) => ({link: i + 1, card: l.card, neg: false}));
  const snap = () => chain.map((l) => Object.assign({}, l));
  const push = (kind, cs, dur, extra) => steps.push(Object.assign({kind, cues: cs, dur}, extra || {}));
  const tracker = chainTracker(baseChain);
  cues = cues || [];
  for (let i = 0; i < cues.length; i++) {
    const c = cues[i];
    const neg = tracker.feed(c);
    switch (c.t) {
      case "turn": {
        let j = i + 1;
        while (j < cues.length && cues[j].t === "phase" && EARLY.has(cues[j].phase)) j++;
        push("banner", cues.slice(i, j), DUR.turn, {turn: c.turn, player: c.player});
        i = j - 1;
        break;
      }
      case "phase": {
        let j = i + 1;
        let last = c.phase;
        while (j < cues.length && cues[j].t === "phase") { last = cues[j].phase; j++; }
        if (EARLY.has(last)) push("model", cues.slice(i, j), 0);
        else push("banner", cues.slice(i, j), DUR.phase, {phase: last});
        i = j - 1;
        break;
      }
      case "draw": {
        let j = i + 1;
        while (j < cues.length && cues[j].t === "draw" && cues[j].player === c.player) j++;
        push("draw", cues.slice(i, j), DUR.draw);
        i = j - 1;
        break;
      }
      case "move": {
        const tl = c.to || {};
        if (sameZone(c.from, c.to)) { push("model", [c], 0); break; }
        let j = i + 1;
        while (j < cues.length && cues[j].t === "move" && cues[j].reason === c.reason &&
               (cues[j].to || {}).loc === tl.loc && (cues[j].to || {}).side === tl.side &&
               !sameZone(cues[j].from, cues[j].to)) j++;
        push("moves", cues.slice(i, j), DUR.move[c.reason] || DUR.move.other, {reason: c.reason});
        i = j - 1;
        break;
      }
      case "summon": {
        let j = i + 1;
        while (j < cues.length && cues[j].t === "summon" && cues[j].how === c.how) j++;
        push("summon", cues.slice(i, j), DUR.summon, {how: c.how});
        i = j - 1;
        break;
      }
      case "activate": {
        const before = snap();
        const link = c.link || chain.length + 1;
        chain = chain.filter((l) => l.link < link);
        chain.push({link, card: c.card, neg: false});
        const opp = c.card && c.card.owner === "opp";
        push("activate", [c], opp ? DUR.actOpp : DUR.actMe, {link, stack: before.filter((l) => l.link < link)});
        break;
      }
      case "chain_solving":
        if (chain.length >= 2) push("solve", [c], DUR.solve, {link: c.link, stack: snap()});
        else push("model", [c], 0);
        break;
      case "chain_solved":
        chain = chain.filter((l) => l.link !== c.link);
        push("model", [c], 0);
        break;
      case "chain_negated":
      case "chain_disabled": {
        const l = chain.find((x) => x.link === c.link);
        if (l) l.neg = true;
        push("negate", [c], DUR.negate, {link: c.link, card: l ? l.card : (neg && neg.card), by: neg && neg.by,
          disabled: c.t === "chain_disabled"});
        break;
      }
      case "chain_end":
        chain = [];
        push("model", [c], 0);
        break;
      case "target":
        push("target", [c], DUR.target);
        break;
      case "attack": {
        const n = cues[i + 1];
        if (n && n.t === "battle" && sameCard(battleRef(n.attacker), c.attacker)) {
          push("attack", [c, n], DUR.battle);
          i++;
        } else push("attack", [c], DUR.attack);
        break;
      }
      case "battle":
        push("attack", [c], DUR.battle);
        break;
      case "damage":
      case "pay":
      case "recover":
        push("lp", [c], DUR.damage);
        break;
      case "lp":
        push("lp", [c], DUR.lp);
        break;
      case "confirm":
        if ((c.cards || []).some((x) => x && x.code)) push("confirm", [c], DUR.confirm);
        else push("model", [c], 0);
        break;
      default:
        push("model", [c], 0);   // shuffle, win, unknown: no scene
    }
  }
  // Dim continuity: consecutive dimmed scenes share one scrim (no pulsing).
  const vis = steps.filter((s) => s.dur > 0);
  vis.forEach((s, k) => {
    s.dim = DIMMED.has(s.kind);
    s.dimPrev = k > 0 && DIMMED.has(vis[k - 1].kind);
    s.dimNext = k + 1 < vis.length && DIMMED.has(vis[k + 1].kind);
    s.samePrev = k > 0 && vis[k - 1].kind === s.kind;
    s.sameNext = k + 1 < vis.length && vis[k + 1].kind === s.kind;
  });
  return steps;
}
const DIMMED = new Set(["activate", "solve", "negate", "confirm"]);

// The glossary term a scene teaches (step 5, item 6), or null. `turnPlayer` is
// the board's turn_player ("you"/"bot") before the scene: a hand trap is the
// opponent activating a monster from the hand during your turn.
export function ruleNoteFor(step, turnPlayer) {
  const c = (step.cues && step.cues[0]) || {};
  if (step.kind === "activate" && (step.link || 1) >= 2) return "chain";
  if (step.kind === "negate") return c.t === "chain_disabled" ? "negate-effect" : "negate-activation";
  if (step.kind === "moves" && step.reason === "banished") return "banish";
  if (step.kind === "attack" && (step.cues || []).some((x) => x.t === "battle")) return "damage-step";
  if (
    step.kind === "activate" &&
    c.card && c.card.owner === "opp" &&
    c.card.zone && c.card.zone.loc === "hand" &&
    turnPlayer === "you"
  ) {
    return "hand-trap";
  }
  return null;
}

export function planDurations(cues, baseChain) {
  return planSteps(cues, baseChain).filter((s) => s.dur > 0).map((s) => s.dur);
}

// ---------- spoken lines (screen readers: the polite live region) ----------

// One sentence per visible scene, e.g. "decklight activates Ash Blossom, chain 2".
export function stepText(step, names) {
  const opp = (names && names[1]) || "decklight";
  const you = names ? names[0] : "You";
  const mine = (r) => r && r.owner === "me";
  const who = (r) => (mine(r) ? you : opp);
  const cname = (r) => (r && r.name) || "a face-down card";
  const list = (rs) => {
    const ns = (rs || []).map((r) => (mine(r) && !names ? "your " : "") + cname(r));
    return ns.length > 3 ? ns.slice(0, 3).join(", ") + " and " + (ns.length - 3) + " more" : ns.join(", ");
  };
  const c = step.cues[0] || {};
  switch (step.kind) {
    case "banner":
      if (step.turn) return "Turn " + step.turn + ", " + (step.player === "me" ? (names ? you + "’s" : "your") : opp + "’s") + " turn";
      return PHASE_NAME[step.phase] || step.phase || "";
    case "draw": {
      const n = step.cues.reduce((a, x) => a + (x.count || 1), 0);
      return (c.player === "me" ? (names ? you + " draws " : "You draw ") : opp + " draws ") + n + (n === 1 ? " card" : " cards");
    }
    case "moves":
      return (MOVE_LABEL[step.reason] || "Moved") + ": " + list(step.cues.map((x) => x.card));
    case "summon":
      return who(c.card) + ", " + (HOW[step.how] || "Summon") + ": " + list(step.cues.map((x) => x.card));
    case "activate":
      return (mine(c.card) && !names ? "You activate " : who(c.card) + " activates ") + cname(c.card) + ", chain " + step.link;
    case "solve":
      return "Chain " + step.link + " resolves";
    case "negate":
      return "Chain " + step.link + (step.disabled ? ", effect negated: " : ", activation negated: ") + cname(step.card) +
        (step.by && step.by.name && (!step.card || step.by.name !== step.card.name) ? ", by " + step.by.name : "");
    case "target":
      return (c.source ? cname(c.source) + " targets " : "Targets ") + list(c.cards);
    case "attack": {
      const a = c.t === "battle" ? battleRef(c.attacker) : c.attacker;
      const tg = c.t === "battle" ? battleRef(c.target) : c.target;
      return cname(a) + (tg ? " attacks " + cname(tg) : " attacks directly");
    }
    case "lp": {
      const p = c.player === "me" ? (names ? you : "you") : opp;
      const tail = typeof c.lp_after === "number" ? ", " + c.lp_after + " LP left" : "";
      if (c.t === "damage") return (c.player === "me" && !names ? "You take " : p + " takes ") + c.amount + " damage" + tail;
      if (c.t === "pay") return (c.player === "me" && !names ? "You pay " : p + " pays ") + c.amount + " LP" + tail;
      if (c.t === "recover") return (c.player === "me" && !names ? "You gain " : p + " gains ") + c.amount + " LP" + tail;
      return typeof c.lp === "number" ? p + ": " + c.lp + " LP" : "";
    }
    case "confirm":
      return "Revealed: " + list(c.cards);
    default:
      return "";
  }
}

// ---------- "what happened" lines (recap) ----------

const nm = (r) => (r && r.name) || "a card";
export function recapItem(c, neg) {
  const opp = (r) => r && r.owner === "opp";
  switch (c.t) {
    case "activate":
      return {text: (opp(c.card) ? "Activated " : "You activated ") + nm(c.card), sub: c.effect || "", card: c.card};
    case "summon":
      return opp(c.card) ? {text: (HOW[c.how] || "Summon").replace(" Summon", " Summoned") + " " + nm(c.card), card: c.card} : null;
    case "move":
      if (c.reason === "set" && opp(c.card)) return {text: c.card && c.card.code ? "Set " + nm(c.card) : "Set a card", card: c.card};
      if (c.reason === "destroyed") return {text: (opp(c.card) ? "" : "Your ") + nm(c.card) + " was destroyed", card: c.card};
      if (c.reason === "banished" && !opp(c.card)) return {text: "Your " + nm(c.card) + " was banished", card: c.card};
      if ((c.reason === "summon" || c.reason === "spsummon") && opp(c.card)) return {text: "Summoned " + nm(c.card), card: c.card};
      return null;
    case "attack":
      return {text: nm(c.attacker) + (c.target ? " attacked " + nm(c.target) : " attacked directly"), card: c.attacker};
    case "damage":
      return c.player === "me" ? {text: "You took " + c.amount + " damage", card: null} : null;
    case "chain_negated":
    case "chain_disabled": {
      const nmd = neg && neg.card && neg.card.name;
      const text = (nmd ? nmd : "Chain " + c.link) + (c.t === "chain_disabled" ? "’s effect was negated" : " was negated");
      return {text, sub: neg && neg.by && neg.by.name ? "by " + neg.by.name : "", card: neg && neg.card, neg: true};
    }
    default:
      return null;
  }
}

// ---------- geometry (DOM lookups on the held board) ----------

const q = (s, root) => (root || document).querySelector(s);
const cssq = (s) => (window.CSS && CSS.escape ? CSS.escape(s) : String(s).replace(/"/g, '\\"'));
const R = (r) => ({x: r.left, y: r.top, w: r.width, h: r.height});
const zkey = (z) => (z ? z.side + ":" + (z.loc === "overlay" ? "mzone" : z.loc === "field" ? "szone" : z.loc) + ":" + (z.loc === "field" ? 5 : z.seq | 0) : "");

function zoneEl(zone) {
  if (!zone) return null;
  const who = zone.side === "opp" ? "bot" : "you";
  const f = q("#app .field." + who);
  const seq = zone.seq | 0;
  let loc = zone.loc;
  if (loc === "overlay") loc = "mzone";
  if (loc === "field" || (loc === "szone" && seq >= 5)) return f && q(".slot.fieldz", f);
  if (loc === "mzone" && seq >= 5) {
    const e = document.querySelectorAll("#midstrip > .slot.emz");
    return e[who === "you" ? seq - 5 : 1 - (seq - 5)] || null;
  }
  if (loc === "mzone" || loc === "szone") {
    const s = f && f.querySelectorAll('[data-row="' + loc + '"] > .slot:not(.side)');
    if (!s || s.length < 5) return null;
    return who === "you" ? s[seq] : s[4 - seq];
  }
  if (!f) return null;
  if (loc === "grave") return q(".gycell .pile", f);
  if (loc === "banished") return q(".banpill", f) || q(".gycell .pile", f);
  if (loc === "deck") return q('.pile[aria-label^="Deck"]', f);
  if (loc === "extra") return q('.pile[aria-label^="Extra"]', f);
  if (loc === "hand") {
    return who === "you" ? q("#hand") : (q("#thand .hcard") ? q("#thand") : q(".pbar.bot .handcount"));
  }
  return null;
}

// The board element of the card at a hand/field zone. Uids are
// "ctrl:0xLOC:seq:code" (view.py _uid); a cue ref's uid names the card where it
// ends up, so a card leaving a zone is found by location + sequence instead.
const LOC_HEX = {hand: "0x2", mzone: "0x4", szone: "0x8", field: "0x8", overlay: "0x4"};
function domCardAt(zone) {
  if (!zone || !LOC_HEX[zone.loc]) return null;
  const who = zone.side === "opp" ? "bot" : "you";
  const seq = zone.loc === "field" ? 5 : zone.seq | 0;
  let box;
  if (zone.loc === "hand") box = who === "you" ? q("#hand") : q("#thand");
  else if ((zone.loc === "mzone" || zone.loc === "overlay") && seq >= 5) box = q("#midstrip");
  else box = q("#app .field." + who);
  if (!box) return null;
  for (const el of box.querySelectorAll(".card[data-uid]")) {
    const p = (el.getAttribute("data-uid") || "").split(":");
    if (p.length >= 3 && p[1] === LOC_HEX[zone.loc] && (p[2] | 0) === seq) return el;
  }
  return null;
}

// Line weight for every cue stroke: scales with the board card (2.1 px at 412 px),
// never thinner than 2 px.
function lineW() {
  return Math.max(2, cardSize().w * 0.036);
}

function cardSize() {
  const s = q("#app .field.you [data-row=mzone] > .slot:not(.side)");
  if (s) { const r = s.getBoundingClientRect(); return {w: r.width, h: r.height}; }
  return {w: 52, h: 76};
}

// A card-sized rect centred on `r`.
function centred(r, w, h) {
  return {x: r.x + r.w / 2 - w / 2, y: r.y + r.h / 2 - h / 2, w, h};
}

function appRect() {
  const a = q("#app");
  return a ? R(a.getBoundingClientRect()) : {x: 0, y: 0, w: window.innerWidth, h: window.innerHeight};
}

// ---------- small helpers ----------

const clamp = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const seg = (t, a, b) => clamp((t - a) / (b - a));
const eOut = (k) => 1 - Math.pow(1 - k, 3);
const eInOut = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const lerp = (a, b, k) => a + (b - a) * k;
const lerpR = (a, b, k) => ({x: lerp(a.x, b.x, k), y: lerp(a.y, b.y, k), w: lerp(a.w, b.w, k), h: lerp(a.h, b.h, k)});
function place(el, r) {
  el.style.left = r.x.toFixed(1) + "px";
  el.style.top = r.y.toFixed(1) + "px";
  el.style.width = r.w.toFixed(1) + "px";
  el.style.height = r.h.toFixed(1) + "px";
}
function div(parent, cls, html) {
  const d = document.createElement("div");
  d.className = cls;
  if (html != null) d.innerHTML = html;
  parent.appendChild(d);
  return d;
}
const SVGNS = "http://www.w3.org/2000/svg";
function svg(parent) {
  const s = document.createElementNS(SVGNS, "svg");
  s.setAttribute("class", "cue-svg");
  s.setAttribute("width", String(window.innerWidth));
  s.setAttribute("height", String(window.innerHeight));
  parent.appendChild(s);
  return s;
}
function sv(parent, tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  parent.appendChild(e);
  return e;
}
// A quiet full-width band behind a row of cards (keeps the board from showing through).
function band(parent, y0, y1) {
  const a = appRect();
  const b = div(parent, "cue-band");
  place(b, {x: a.x, y: y0, w: a.w, h: y1 - y0});
  return b;
}
// Mock states use short uids ("yh1") that name the card before it moves; a server
// ref's uid ("0:0x4:2:code") names where it ends up. Only the former locates a source.
const srcUid = (r) => (r && r.uid && r.uid.indexOf(":") < 0 ? r.uid : null);
const isHiddenRef = (r) => !r || (!r.code && !r.name);
const sideCls = (r) => (r && r.owner === "opp" ? "opp" : "me");

// ⊘ stamp over a card: a circle and its slash, each on a dark casing so the mark
// keeps its contrast over bright art. Returns {set(k)} drawing it to fraction k.
function negStamp(parent, r, lw) {
  const s = svg(parent);
  s.classList.add("negmark");
  const cx = r.x + r.w / 2, cy = r.y + r.h / 2;
  const rad = Math.min(r.w, r.h) * 0.36;
  const w = lw * 1.75;
  const c45 = Math.SQRT1_2 * rad;
  const parts = [];
  for (const cls of ["case", "mark"]) {
    const sw = cls === "case" ? w + lw * 1.4 : w;
    const circ = sv(s, "circle", {cx, cy, r: rad, class: cls});
    const sl = sv(s, "line", {x1: cx - c45, y1: cy + c45, x2: cx + c45, y2: cy - c45, class: cls});
    circ.style.strokeWidth = sw.toFixed(2);
    sl.style.strokeWidth = sw.toFixed(2);
    parts.push({circ, sl});
  }
  const C = 2 * Math.PI * rad, L = 2 * rad;
  for (const p of parts) {
    p.circ.style.strokeDasharray = C.toFixed(1);
    p.sl.style.strokeDasharray = L.toFixed(1);
  }
  return {
    svg: s,
    set(kc, ks) {
      for (const p of parts) {
        p.circ.style.strokeDashoffset = (C * (1 - kc)).toFixed(1);
        p.sl.style.strokeDashoffset = (L * (1 - ks)).toFixed(1);
      }
    },
  };
}
const NEG_FILTER = "brightness(.42) saturate(.12)";

// ---------- the player ----------

// opts: {layer, getView(), apply(msg), mock, names() -> [me, opp], speed() -> 1|2|0,
//        pause() -> bool, textFor(code) -> string|null, onIdle()}
// The settled instant of a scene (fraction of its length): what reduced motion shows.
const HOLD = 0.62;

export function createCuePlayer(opts) {
  const layer = opts.layer;
  layer.setAttribute("aria-hidden", "true");   // visual only: the live region speaks each scene
  layer.innerHTML = '<div class="cue-ghosts"></div><div class="cue-dim"></div><div class="cue-stage"></div>' +
    '<div class="cue-hint"></div><div class="cue-rule"></div>';
  const ghostsEl = layer.querySelector(".cue-ghosts");
  const dimEl = layer.querySelector(".cue-dim");
  const stage = layer.querySelector(".cue-stage");
  const hintEl = layer.querySelector(".cue-hint");
  const ruleEl = layer.querySelector(".cue-rule");
  let hideStyle = document.getElementById("cue-hide");
  if (!hideStyle) {
    hideStyle = document.createElement("style");
    hideStyle.id = "cue-hide";
    document.head.appendChild(hideStyle);
  }

  const st = {
    queue: [], cur: null, steps: null, si: -1, scene: null, t0: 0, raf: 0,
    waiting: false, ghosts: new Map(), hidden: new Set(), lp: {}, frozen: null,
    rules: new Set(),   // glossary ids already taught this game (rule notes)
  };

  const ruleOn = () => (opts.ruleNotes ? !!opts.ruleNotes() : false);
  function setRule(text) {
    ruleEl.textContent = text || "";
    ruleEl.classList.toggle("on", !!text);
  }
  // The rule note for a live step: at most once per game per term (teaches, then
  // gets out of the way). Returns the note text to append to the announcement.
  function noteStep(step) {
    if (!ruleOn()) { setRule(""); return null; }
    const gid = ruleNoteFor(step, (opts.getView() || {}).turn_player);
    if (!gid || st.rules.has(gid)) { setRule(""); return null; }
    st.rules.add(gid);
    const short = ruleShort(gid);
    setRule(short);
    return short;
  }

  // ----- persistent board patches (cleared when a snapshot applies) -----
  function hideUid(uid) {
    if (!uid || st.hidden.has(uid)) return;
    st.hidden.add(uid);
    const u = cssq(uid);
    hideStyle.textContent += '#app .slot:has(> .card[data-uid="' + u + '"]) > *, #app .hcard:has(> .card[data-uid="' + u +
      '"]) { visibility: hidden; }\n';
  }
  function clearPatches() {
    st.hidden.clear();
    hideStyle.textContent = "";
    ghostsEl.innerHTML = "";
    st.ghosts.clear();
    st.lp = {};
    stage.innerHTML = "";
    dimEl.style.opacity = "0";
  }
  function setGhost(zone, el) {
    const k = zkey(zone);
    const old = st.ghosts.get(k);
    if (old && old !== el) old.remove();
    el.classList.add("ghost");
    ghostsEl.appendChild(el);
    st.ghosts.set(k, el);
  }
  // Rect of a card at a zone (ghost, then the board element, then the zone).
  function rectAt(zone, uid, opt) {
    const g = zone && st.ghosts.get(zkey(zone));
    if (g && FIELD.has(zone.loc)) return R(g.getBoundingClientRect());
    const cs = cardSize();
    if (uid) {
      const el = q('#app .card[data-uid="' + cssq(uid) + '"]');
      if (el) {
        const box = el.closest(".slot, .hcard") || el;
        const r = R(box.getBoundingClientRect());
        if (r.w > 0) return r;
      }
    }
    const dc = domCardAt(zone);
    if (dc) {
      const r = R((dc.closest(".slot, .hcard") || dc).getBoundingClientRect());
      if (r.w > 0) return r;
    }
    const z = zoneEl(zone);
    if (z) {
      const r = R(z.getBoundingClientRect());
      if (zone.loc === "hand" || zone.loc === "banished") {
        const w = zone.loc === "hand" ? cs.w * 1.1 : cs.w * 0.8;
        return centred(r, w, w * 86 / 59);
      }
      if (r.w > 0) return r;
    }
    const a = appRect();
    if (opt && opt.fallback === "none") return null;
    return centred({x: a.x, y: a.y + a.h * 0.42, w: a.w, h: 0}, cs.w, cs.h);
  }
  // Take a card off the board for an animation: drop its ghost, hide its element.
  function lift(zone, ref) {
    const g = zone && st.ghosts.get(zkey(zone));
    if (g && FIELD.has(zone.loc)) { g.remove(); st.ghosts.delete(zkey(zone)); return; }
    if (!zone || !(FIELD.has(zone.loc) || zone.loc === "hand")) return;
    const dc = domCardAt(zone);
    if (dc) { hideUid(dc.getAttribute("data-uid")); return; }
    const su = srcUid(ref);
    if (su && q('#app .card[data-uid="' + cssq(su) + '"]')) { hideUid(su); return; }
    const z = zoneEl(zone);
    const el = z && zone.loc !== "hand" ? q(".card[data-uid]", z) : null;
    if (el) hideUid(el.getAttribute("data-uid"));
  }

  function cardNode(ref, big) {
    const d = document.createElement("div");
    const hidden = isHiddenRef(ref);
    d.className = "cue-card" + (hidden ? " back" : "");
    if (!hidden) {
      d.innerHTML = phHTML(ref) + (!opts.mock && ref.code
        ? '<img alt="" src="' + imgUrl(ref.code, big ? "full" : "small") + '" onerror="this.remove()">' : "");
    }
    return d;
  }
  const nameOf = (who) => {
    const n = opts.names ? opts.names() : null;
    if (who === "opp") return (n && n[1]) || "decklight";
    return (n && n[0]) || "You";
  };

  function dimAt(step, t) {
    if (!step.dim) return;
    const a = step.dimPrev ? 1 : eOut(seg(t, 0, 220));
    const b = step.dimNext ? 1 : 1 - seg(t, step.dur - 240, step.dur);
    dimEl.style.opacity = (Math.min(a, b)).toFixed(3);
  }
  // fade in over the first `fi` ms and out over the last `fo` ms (skipped when the neighbour is the same kind)
  function inOut(step, t, fi, fo) {
    const a = step.samePrev ? 1 : seg(t, 0, fi);
    const b = step.sameNext ? 1 : 1 - seg(t, step.dur - fo, step.dur);
    return Math.min(a, b);
  }

  // ---------- scenes ----------
  const SCENES = {
    banner(step) {
      const mid = q("#mid") || q("#app");
      const mr = R(mid.getBoundingClientRect());
      const a = appRect();
      const el = div(stage, "cue-banner");
      let text, cls;
      if (step.turn != null) {
        const who = step.player === "opp" ? "opp" : "me";
        const n = nameOf(who);
        text = "Turn " + step.turn + ' <i>·</i> ' + esc(who === "me" && n === "You" ? "Your turn" : n);
        cls = who;
      } else {
        text = esc(PHASE_NAME[step.phase] || step.phase || "");
        cls = "phase";
      }
      el.classList.add(cls);
      el.innerHTML = '<span class="ln l"></span><span class="tx">' + text + '</span><span class="ln r"></span>';
      const h = Math.max(40, Math.min(mr.h - 8, 64));
      place(el, {x: a.x, y: mr.y + mr.h / 2 - h / 2, w: a.w, h});
      const lines = el.querySelectorAll(".ln");
      const tx = el.querySelector(".tx");
      return {
        at(t) {
          const o = inOut(step, t, 260, 280);
          el.style.opacity = o.toFixed(3);
          const g = eOut(seg(t, 60, 520));
          for (const l of lines) l.style.transform = "scaleX(" + g.toFixed(3) + ")";
          tx.style.opacity = eOut(seg(t, 120, 440)).toFixed(3);
          tx.style.transform = "translateY(" + (4 * (1 - eOut(seg(t, 120, 440)))).toFixed(1) + "px)";
        },
        finish() {},
      };
    },

    draw(step) {
      const items = [];
      step.cues.forEach((c) => {
        const who = c.player === "opp" ? "opp" : "me";
        const n = Math.max(1, Math.min(6, c.count || (c.cards || []).length || 1));
        for (let i = 0; i < n; i++) {
          const ref = (c.cards || [])[i] || null;
          const from = rectAt({side: who, loc: "deck", seq: 0});
          const to = rectAt({side: who, loc: "hand", seq: 0});
          const el = cardNode(ref && ref.code ? ref : null);
          stage.appendChild(el);
          items.push({el, from, to, delay: items.length * 50});
        }
      });
      return {
        at(t) {
          for (const it of items) {
            const k = eOut(seg(t, it.delay, step.dur - 60));
            place(it.el, lerpR(it.from, it.to, k));
            it.el.style.opacity = (Math.min(seg(t, it.delay, it.delay + 90), 1 - seg(t, step.dur - 120, step.dur))).toFixed(3);
          }
        },
        finish() {},
      };
    },

    moves(step) {
      const reason = step.reason;
      const items = step.cues.map((c, i) => {
        const ref = c.card || {};
        const from = rectAt(c.from, srcUid(ref));
        lift(c.from, ref);
        const toField = c.to && FIELD.has(c.to.loc) && c.to.loc !== "overlay";
        const to = rectAt(c.to, toField ? null : undefined);
        const el = cardNode(ref);
        stage.appendChild(el);
        place(el, from);
        return {c, el, from, to, toField, delay: i * 60};
      });
      let label = null;
      const lt = MOVE_LABEL[reason];
      if (lt && items.length) {
        const it0 = items[0];
        const anchor = reason === "added" ? it0.to : it0.from;
        const own = (it0.c.card || {}).owner === "opp" ? "opp" : "me";
        label = div(stage, "cue-label " + own, esc(lt) + (items.length > 1 ? " ×" + items.length : ""));
        const above = anchor.y > 120;
        label.style.left = (anchor.x + anchor.w / 2) + "px";
        label.style.top = (above ? anchor.y - 22 : anchor.y + anchor.h + 6) + "px";
      }
      const destroyed = reason === "destroyed";
      const start = destroyed ? 260 : 0;
      return {
        at(t) {
          for (const it of items) {
            const k = eInOut(seg(t, start + it.delay, step.dur - (it.toField ? 80 : 40)));
            place(it.el, lerpR(it.from, it.to, k));
            if (destroyed) {
              const d = seg(t, 0, 240);
              it.el.style.filter = "brightness(" + (1 - 0.42 * d).toFixed(3) + ") saturate(" + (1 - 0.6 * d).toFixed(3) + ")";
            }
            const fade = it.toField ? 1 : 1 - seg(t, step.dur - 180, step.dur);
            it.el.style.opacity = (Math.min(seg(t, 0, 80) + (it.from ? 1 : 0), fade)).toFixed(3);
          }
          if (label) label.style.opacity = (Math.min(seg(t, 80, 260), 1 - seg(t, step.dur - 200, step.dur))).toFixed(3);
        },
        finish() {
          for (const it of items) {
            if (it.toField) {
              place(it.el, it.to);
              it.el.style.opacity = "1";
              it.el.style.filter = "";
              setGhost(it.c.to, it.el);
            }
          }
        },
      };
    },

    summon(step) {
      const items = step.cues.map((c, i) => {
        const ref = c.card || {};
        const fromZ = c.from || {side: ref.owner || "me", loc: "hand", seq: 0};
        const from = rectAt(fromZ, srcUid(ref));
        lift(fromZ, ref);
        const to = rectAt(ref.zone || c.to);
        const el = cardNode(ref);
        stage.appendChild(el);
        place(el, from);
        return {c, ref, el, from, to, delay: i * 90};
      });
      const it0 = items[0];
      const own = sideCls(it0 && it0.ref);
      const label = div(stage, "cue-label sum " + own, esc(HOW[step.how] || "Summon"));
      if (it0) {
        const z0 = it0.ref.zone;
        const above = z0 ? (z0.side !== "opp" || (z0.loc === "mzone" && z0.seq >= 5)) : true;
        label.style.left = (it0.to.x + it0.to.w / 2) + "px";
        label.style.top = (above ? it0.to.y - 22 : it0.to.y + it0.to.h + 6) + "px";
      }
      return {
        at(t) {
          for (const it of items) {
            const k = eOut(seg(t, it.delay, it.delay + 540));
            place(it.el, lerpR(it.from, it.to, k));
            it.el.style.opacity = seg(t, it.delay, it.delay + 140).toFixed(3);
          }
          label.style.opacity = (Math.min(seg(t, 300, 520), 1 - seg(t, step.dur - 200, step.dur))).toFixed(3);
          label.style.transform = "translate(-50%," + (4 * (1 - eOut(seg(t, 300, 520)))).toFixed(1) + "px)";
        },
        finish() {
          for (const it of items) {
            place(it.el, it.to);
            it.el.style.opacity = "1";
            setGhost(it.ref.zone || it.c.to, it.el);
          }
        },
      };
    },

    activate(step) {
      const c = step.cues[0];
      const ref = c.card || {};
      const who = ref.owner === "opp" ? "opp" : "me";
      const a = appRect();
      const H = window.innerHeight;
      const cw = Math.min(a.w * 0.40, 200, H * 0.27 * 59 / 86);
      const ch = cw * 86 / 59;
      const cy = a.y + Math.min(a.h, H) * 0.33;
      const dest = {x: a.x + a.w / 2 - cw / 2, y: cy - ch / 2, w: cw, h: ch};
      const src = rectAt(ref.zone, ref.uid, {fallback: "none"});
      const from = src || {x: dest.x + cw * 0.2, y: dest.y + 30, w: cw * 0.6, h: ch * 0.6};
      const box = div(stage, "cue-act " + who);
      const card = cardNode(ref, true);
      card.classList.add("big");
      box.appendChild(card);
      place(card, from);
      const txt = c.effect || (ref.code && opts.textFor ? opts.textFor(ref.code) : "") || "";
      const kick = (who === "opp" ? esc(nameOf("opp")) + " activates" : (nameOf("me") === "You" ? "You activate" : esc(nameOf("me")) + " activates")) +
        (step.link > 1 ? ' <i>·</i> <b>Chain ' + step.link + "</b>" : "");
      const text = div(box, "cue-text",
        '<div class="kick">' + kick + "</div><h3>" + esc(ref.name || "") + '</h3><span class="rule"></span>' +
        (txt ? '<p>' + esc(txt) + "</p>" : ""));
      const tw = Math.min(a.w - 40, 380);
      text.style.left = (a.x + a.w / 2 - tw / 2) + "px";
      text.style.top = (dest.y + ch + 14) + "px";
      text.style.width = tw + "px";
      const rule = text.querySelector(".rule");
      // Earlier links of this chain: a quiet row above the card.
      let row = null;
      if (step.stack && step.stack.length) {
        const tw2 = Math.min(30, cw * 0.2), th2 = tw2 * 86 / 59;
        row = div(box, "cue-stackrow");
        for (const l of step.stack) {
          const cell = div(row, "cell" + (l.neg ? " neg" : ""));
          const tn = cardNode(l.card);
          tn.style.position = "relative";
          tn.style.width = tw2 + "px";
          tn.style.height = th2 + "px";
          cell.appendChild(tn);
          div(cell, "n", String(l.link));
        }
        row.style.left = (a.x + a.w / 2) + "px";
        row.style.top = Math.max(a.y + 50, dest.y - th2 - 34) + "px";
      }
      const T = step.dur;
      return {
        at(t) {
          dimAt(step, t);
          const k = eOut(seg(t, 0, 380));
          const out = seg(t, T - 260, T);
          const r = lerpR(from, dest, k);
          const s = 1 - 0.06 * out;
          place(card, {x: r.x + r.w * (1 - s) / 2, y: r.y + r.h * (1 - s) / 2, w: r.w * s, h: r.h * s});
          card.style.opacity = (Math.min(src ? 1 : seg(t, 0, 200), 1 - out)).toFixed(3);
          const tk = eOut(seg(t, 200, 480));
          text.style.opacity = (Math.min(tk, 1 - out)).toFixed(3);
          text.style.transform = "translateY(" + (6 * (1 - tk)).toFixed(1) + "px)";
          rule.style.transform = "scaleX(" + eOut(seg(t, 280, 700)).toFixed(3) + ")";
          if (row) row.style.opacity = (Math.min(seg(t, 120, 380), 1 - out)).toFixed(3);
        },
        finish() {},
      };
    },

    solve(step) {
      const a = appRect();
      const H = window.innerHeight;
      const n = step.stack.length;
      const cw = Math.min(70, (a.w - 48) / Math.max(n, 1) - 14, a.w * 0.18);
      const ch = cw * 86 / 59;
      const gap = 14;
      const total = n * cw + (n - 1) * gap;
      const cy = a.y + Math.min(a.h, H) * 0.40;
      const box = div(stage, "cue-solve");
      const bnd = band(box, cy - ch / 2 - 58, cy + ch / 2 + 38);
      const head = div(box, "kick", "Resolving <b>chain " + step.link + "</b>");
      head.style.left = (a.x + a.w / 2) + "px";
      head.style.top = (cy - ch / 2 - 40) + "px";
      const cells = step.stack.map((l, i) => {
        const el = cardNode(l.card);
        box.appendChild(el);
        const r = {x: a.x + a.w / 2 - total / 2 + i * (cw + gap), y: cy - ch / 2, w: cw, h: ch};
        place(el, r);
        const num = div(box, "num" + (l.link === step.link ? " cur" : ""), String(l.link));
        num.style.left = (r.x + cw / 2) + "px";
        num.style.top = (r.y + ch + 8) + "px";
        let slash = null;
        if (l.neg) {
          el.style.filter = NEG_FILTER;
          slash = negStamp(box, r, lineW()).svg;
        }
        return {l, el, r, num, slash};
      });
      const T = step.dur;
      return {
        at(t) {
          dimAt(step, t);
          const o = inOut(step, t, 200, 200);
          head.style.opacity = o.toFixed(3);
          bnd.style.opacity = o.toFixed(3);
          for (const c of cells) {
            const cur = c.l.link === step.link;
            const lift = cur ? 10 * eOut(seg(t, 80, 380)) : 0;
            place(c.el, {x: c.r.x, y: c.r.y - lift, w: c.r.w, h: c.r.h});
            c.el.style.opacity = o.toFixed(3);
            if (!cur && !c.l.neg) c.el.style.filter = "brightness(.5) saturate(.6)";
            c.num.style.opacity = o.toFixed(3);
            if (c.slash) c.slash.style.opacity = o.toFixed(3);
          }
          void T;
        },
        finish() {},
      };
    },

    negate(step) {
      const a = appRect();
      const H = window.innerHeight;
      const cw = Math.min(a.w * 0.34, 156);
      const ch = cw * 86 / 59;
      const cy = a.y + Math.min(a.h, H) * 0.36;
      const r = {x: a.x + a.w / 2 - cw / 2, y: cy - ch / 2, w: cw, h: ch};
      const box = div(stage, "cue-neg");
      const el = cardNode(step.card || {}, true);
      box.appendChild(el);
      place(el, r);
      const stamp = negStamp(box, r, lineW() * 1.0);
      const what = step.disabled ? "Effect negated" : "Activation negated";
      const by = step.by && step.by.name && (!step.card || step.by.name !== step.card.name || step.by.owner !== step.card.owner)
        ? step.by : null;
      const k = div(box, "kick",
        "Chain " + step.link + ' <i>·</i> <b>' + what + "</b>" +
        (step.card && step.card.name ? '<span class="nm">' + esc(step.card.name) + "</span>" : "") +
        (by ? '<span class="by">by <em>' + esc(by.name) + "</em></span>" : ""));
      k.style.left = (a.x + a.w / 2) + "px";
      k.style.top = (r.y + ch + 16) + "px";
      const T = step.dur;
      return {
        at(t) {
          dimAt(step, t);
          const o = inOut(step, t, 200, 260);
          const d = eInOut(seg(t, 160, 560));
          // the card sinks: darker, greyer, a touch smaller
          const sc = 1 - 0.04 * d;
          place(el, {x: r.x + r.w * (1 - sc) / 2, y: r.y + r.h * (1 - sc) / 2, w: r.w * sc, h: r.h * sc});
          el.style.opacity = o.toFixed(3);
          el.style.filter = "brightness(" + (1 - 0.58 * d).toFixed(3) + ") saturate(" + (1 - 0.88 * d).toFixed(3) + ")";
          stamp.set(eOut(seg(t, 220, 640)), eOut(seg(t, 560, 820)));
          stamp.svg.style.opacity = o.toFixed(3);
          const kk = eOut(seg(t, 420, 700));
          k.style.opacity = Math.min(o, kk).toFixed(3);
          k.style.transform = "translate(-50%," + (5 * (1 - kk)).toFixed(1) + "px)";
          void T;
        },
        finish() {},
      };
    },

    target(step) {
      const c = step.cues[0];
      const s = svg(stage);
      const src = c.source ? rectAt(c.source.zone, c.source.uid, {fallback: "none"}) : null;
      const parts = (c.cards || []).map((ref) => {
        const r = rectAt(ref.zone, ref.uid);
        const lw = lineW();
        const ring = sv(s, "rect", {x: r.x - 2 - lw, y: r.y - 2 - lw, width: r.w + 4 + 2 * lw, height: r.h + 4 + 2 * lw,
          rx: 6, class: "ring"});
        ring.style.strokeWidth = lw.toFixed(2);
        let line = null, len = 0;
        if (src) {
          const x1 = src.x + src.w / 2, y1 = src.y + src.h / 2, x2 = r.x + r.w / 2, y2 = r.y + r.h / 2;
          len = Math.hypot(x2 - x1, y2 - y1);
          line = sv(s, "line", {x1, y1, x2, y2, class: "tline"});
          line.style.strokeWidth = lw.toFixed(2);
          line.style.strokeDasharray = len.toFixed(1);
        }
        return {ring, line, len};
      });
      return {
        at(t) {
          const o = 1 - seg(t, step.dur - 180, step.dur);
          for (const p of parts) {
            if (p.line) p.line.style.strokeDashoffset = (p.len * (1 - eOut(seg(t, 0, 360)))).toFixed(1);
            p.ring.style.opacity = (Math.min(seg(t, 220, 420), o)).toFixed(3);
          }
          s.style.opacity = o.toFixed(3);
        },
        finish() {},
      };
    },

    attack(step) {
      const atk = step.cues.find((c) => c.t === "attack");
      const bat = step.cues.find((c) => c.t === "battle");
      const aref = atk ? atk.attacker : battleRef(bat.attacker);
      const tref = atk ? atk.target : (bat.target ? battleRef(bat.target) : null);
      const who = aref && aref.owner === "opp" ? "opp" : "me";
      const ar = rectAt(aref && aref.zone, aref && aref.uid);
      let tr;
      if (tref) tr = rectAt(tref.zone, tref.uid);
      else {
        const lp = q("#app .pbar." + (who === "opp" ? "you" : "bot") + " .lp");
        tr = lp ? R(lp.getBoundingClientRect()) : centred(appRect(), 10, 10);
      }
      const x1 = ar.x + ar.w / 2, y1 = ar.y + ar.h / 2, x2 = tr.x + tr.w / 2, y2 = tr.y + tr.h / 2;
      const L = Math.hypot(x2 - x1, y2 - y1) || 1;
      const nx = -(y2 - y1) / L, ny = (x2 - x1) / L;
      const bend = Math.min(26, L * 0.08) * (x2 >= x1 ? 1 : -1);
      const pt = (u) => {
        const m = 2 * (1 - u) * u;
        return [(1 - u) * (1 - u) * x1 + m * ((x1 + x2) / 2 + nx * bend) + u * u * x2,
          (1 - u) * (1 - u) * y1 + m * ((y1 + y2) / 2 + ny * bend) + u * u * y2];
      };
      const s = svg(stage);
      const brush = sv(s, "path", {class: "brush " + who});
      const brush2 = sv(s, "path", {class: "brush dry " + who});
      // Tapered brush stroke: polygon along the curve, wide at the attacker, thin at the target.
      const strokeTo = (u1, w0, w1, wobble) => {
        const N = 28;
        const left = [], right = [];
        for (let i = 0; i <= N; i++) {
          const u = (i / N) * u1;
          const [x, y] = pt(u);
          const [xa, ya] = pt(Math.min(1, u + 0.01));
          const dx = xa - x, dy = ya - y, dl = Math.hypot(dx, dy) || 1;
          const w = (lerp(w0, w1, u) * (1 + wobble * Math.sin(u * 23))) / 2;
          left.push((x - dy / dl * w).toFixed(1) + "," + (y + dx / dl * w).toFixed(1));
          right.push((x + dy / dl * w).toFixed(1) + "," + (y - dx / dl * w).toFixed(1));
        }
        return "M" + left.join("L") + "L" + right.reverse().join("L") + "Z";
      };
      // Numbers facing each other at the stroke's midpoint.
      let nums = null;
      if (bat) {
        const A = bat.attacker || {};
        const D = bat.target || null;
        const aVal = A.atk != null ? A.atk : (battleRef(A) || {}).atk;
        let defMode = false;
        if (D) {
          const dref = battleRef(D) || {};
          if (dref.pos) defMode = dref.pos === "def" || dref.pos === "set";
          else {
            const el = dref.uid && q('#app .card[data-uid="' + cssq(dref.uid) + '"]');
            defMode = !!(el && el.classList.contains("rot"));
          }
        }
        const dVal = D ? (defMode ? (D.def != null ? D.def : (battleRef(D) || {}).def) : (D.atk != null ? D.atk : (battleRef(D) || {}).atk)) : null;
        const tcls = who === "opp" ? "me" : "opp";
        const win = dVal == null ? "a" : aVal > dVal ? "a" : aVal < dVal ? "d" : "";
        nums = div(stage, "cue-nums",
          '<span class="v ' + who + (win === "a" ? " hi" : "") + '"><b>' + (aVal == null ? "?" : aVal) + "</b><i>ATK</i></span>" +
          (dVal != null ? '<span class="sep"></span><span class="v ' + tcls + (win === "d" ? " hi" : "") + '"><b>' + dVal +
            "</b><i>" + (defMode ? "DEF" : "ATK") + "</i></span>" : ""));
        const [mx, my] = pt(0.5);
        nums.style.left = mx + "px";
        nums.style.top = my + "px";
      }
      const T = step.dur;
      const drawEnd = 420;
      // Heavier than the target line: ~4x the line weight at the attacker, ~1.6x at the tip.
      const lw = lineW();
      const bw0 = Math.max(lw * 4, cardSize().w * 0.15), bw1 = lw * 1.6;
      return {
        at(t) {
          const o = 1 - seg(t, T - 240, T);
          const u = eOut(seg(t, 0, drawEnd));
          brush.setAttribute("d", u > 0 ? strokeTo(u, bw0, bw1, 0.12) : "");
          brush2.setAttribute("d", u > 0 ? strokeTo(u * 0.92, bw1, lw, 0.4) : "");
          s.style.opacity = o.toFixed(3);
          if (nums) {
            const k = eOut(seg(t, drawEnd - 40, drawEnd + 240));
            nums.style.opacity = Math.min(k, o).toFixed(3);
            nums.style.transform = "translate(-50%,-50%) translateY(" + (5 * (1 - k)).toFixed(1) + "px)";
          }
        },
        finish() {},
      };
    },

    lp(step) {
      const c = step.cues[0];
      const who = c.player === "opp" ? "bot" : "you";
      const lpEl = q("#app .pbar." + who + " .lp");
      const bar = q("#app .pbar." + who + " .lpbar > i");
      const from = lpEl ? parseInt(lpEl.textContent, 10) || 0 : 0;
      const to = c.t === "lp" ? c.lp : (c.lp_after != null ? c.lp_after
        : from + (c.t === "recover" ? 1 : -1) * (c.amount || 0));
      st.lp[who] = to;
      let fl = null;
      if (c.t !== "lp" && lpEl) {
        const r = R(lpEl.getBoundingClientRect());
        fl = div(stage, "cue-float " + (c.t === "recover" ? "up" : "down"),
          (c.t === "recover" ? "+" : "−") + (c.amount != null ? c.amount : Math.abs(to - from)));
        fl.style.left = (r.x + r.w + 10) + "px";
        fl.style.top = (r.y + r.h / 2) + "px";
      }
      const T = step.dur;
      const set = (v) => {
        if (lpEl) lpEl.textContent = String(Math.round(v));
        if (bar) bar.style.width = (Math.max(0, Math.min(1, v / 8000)) * 100).toFixed(1) + "%";
      };
      return {
        at(t) {
          set(lerp(from, to, eOut(seg(t, 120, T * 0.75))));
          if (lpEl) lpEl.classList.toggle("cue-lpdown", to < from && t < T);
          if (fl) {
            const k = seg(t, 0, T);
            fl.style.opacity = (Math.min(seg(t, 0, 160), 1 - seg(t, T * 0.6, T))).toFixed(3);
            fl.style.transform = "translateY(" + (-50 - 18 * eOut(k)).toFixed(1) + "%)";
          }
        },
        finish() {
          set(to);
          if (lpEl) lpEl.classList.remove("cue-lpdown");
        },
      };
    },

    confirm(step) {
      const c = step.cues[0];
      const cards = (c.cards || []).filter((x) => x && x.code).slice(0, 5);
      const a = appRect();
      const H = window.innerHeight;
      const n = cards.length;
      const cw = Math.min(96, (a.w - 48) / n - 12);
      const ch = cw * 86 / 59;
      const cy = a.y + Math.min(a.h, H) * 0.40;
      const total = n * cw + (n - 1) * 12;
      const box = div(stage, "cue-solve");
      const bnd = band(box, cy - ch / 2 - 50, cy + ch / 2 + 22);
      const head = div(box, "kick", "Revealed");
      head.style.left = (a.x + a.w / 2) + "px";
      head.style.top = (cy - ch / 2 - 34) + "px";
      const els = cards.map((ref, i) => {
        const el = cardNode(ref, n <= 2);
        box.appendChild(el);
        place(el, {x: a.x + a.w / 2 - total / 2 + i * (cw + 12), y: cy - ch / 2, w: cw, h: ch});
        return el;
      });
      return {
        at(t) {
          dimAt(step, t);
          const o = inOut(step, t, 220, 240);
          head.style.opacity = o.toFixed(3);
          bnd.style.opacity = o.toFixed(3);
          els.forEach((el, i) => {
            const k = eOut(seg(t, i * 60, i * 60 + 300));
            el.style.opacity = Math.min(k, o).toFixed(3);
            el.style.transform = "translateY(" + (6 * (1 - k)).toFixed(1) + "px)";
          });
        },
        finish() {},
      };
    },
  };

  function mount(step) {
    stage.innerHTML = "";
    layer.style.setProperty("--cue-lw", lineW().toFixed(2) + "px");
    const f = SCENES[step.kind];
    let sc = null;
    try {
      sc = f ? f(step) : null;
    } catch (e) {
      console.warn("cue scene failed", step.kind, e);   // a broken scene never stalls the game
      sc = null;
    }
    return sc || {at() {}, finish() {}};
  }
  function endScene() {
    if (!st.scene) return;
    const step = st.steps[st.si];
    try {
      st.scene.at(step.dur);
      st.scene.finish();
    } catch (e) {
      console.warn("cue finish failed", e);
    }
    st.scene = null;
    stage.innerHTML = "";
    if (!step.dimNext) dimEl.style.opacity = "0";
  }

  // ----- live playback -----
  function visibleCount(steps) { return steps.filter((s) => s.dur > 0).length; }

  function startSegment(seg) {
    st.cur = seg;
    const v = opts.getView();
    st.steps = planSteps(seg.cues, v && v.chain);
    st.si = -1;
    seg.visible = visibleCount(st.steps);
    nextStep();
  }
  function nextStep() {
    endScene();
    cancelAnimationFrame(st.raf);
    for (;;) {
      st.si += 1;
      if (st.si >= st.steps.length) { endSegment(); return; }
      if (st.steps[st.si].dur > 0) break;
    }
    const step = st.steps[st.si];
    st.scene = mount(step);
    st.t0 = performance.now();
    const note = noteStep(step);
    if (opts.announce) opts.announce(stepText(step, opts.names && opts.names()) + (note ? ". " + note : ""));
    // Reduced motion: each scene is shown settled (its frame at HOLD) and only fades in and out.
    const still = opts.reduced && opts.reduced();
    stage.style.opacity = "";
    st.scene.at(still ? step.dur * HOLD : 0);
    const tick = (now) => {
      if (!st.scene) return;
      const sp = opts.speed() || 1;
      const el = (now - st.t0) * sp;
      if (el >= step.dur) { stage.style.opacity = ""; nextStep(); return; }
      if (still) stage.style.opacity = Math.min(1, el / 160, (step.dur - el) / 200).toFixed(3);
      else st.scene.at(el);
      st.raf = requestAnimationFrame(tick);
    };
    st.raf = requestAnimationFrame(tick);
  }
  function endSegment() {
    const seg = st.cur;
    st.cur = null;
    st.steps = null;
    setRule("");
    opts.apply(seg.msg);
    clearPatches();
    const m = seg.msg || {};
    if (opts.pause() && seg.visible > 0 && !m.prompt && !m.result) {
      st.waiting = true;
      hintEl.textContent = "Tap or press Enter to continue";
      if (opts.announce) opts.announce("Paused. Press Enter to continue.");
      hintEl.classList.add("on");
      layer.classList.add("waiting");
      return;
    }
    pump();
  }
  function pump() {
    if (st.cur || st.waiting || st.frozen) return;
    const seg = st.queue.shift();
    if (!seg) { layer.classList.remove("playing"); if (opts.onIdle) opts.onIdle(); return; }
    layer.classList.add("playing");
    startSegment(seg);
  }

  const api = {
    // Queue a state message. Returns false when it applied at once (no cues, idle).
    enqueue(cues, msg) {
      if (!cues || !cues.length) {
        if (st.queue.length) { st.queue[st.queue.length - 1].msg = msg; return true; }
        if (st.cur) { st.cur.msg = msg; return true; }
        opts.apply(msg);
        if (st.waiting && (msg.prompt || msg.result)) api.resume();
        return false;
      }
      st.queue.push({cues, msg});
      pump();
      return true;
    },
    busy() { return !!(st.cur || st.queue.length || st.waiting); },
    playing() { return !!st.cur; },
    waiting() { return st.waiting; },
    resume() {
      st.waiting = false;
      hintEl.classList.remove("on");
      layer.classList.remove("waiting");
      pump();
    },
    // Tap while playing: finish the current scene now; while paused: step.
    tap() {
      if (st.waiting) { api.resume(); return true; }
      if (st.cur && st.scene) { nextStep(); return true; }
      return false;
    },
    // Apply everything queued at once (speed set to instant, skip all).
    flush() {
      cancelAnimationFrame(st.raf);
      st.scene = null;
      const segs = (st.cur ? [st.cur] : []).concat(st.queue);
      st.cur = null;
      st.queue = [];
      st.steps = null;
      st.waiting = false;
      hintEl.classList.remove("on");
      setRule("");
      layer.classList.remove("waiting", "playing");
      clearPatches();
      for (const s of segs) opts.apply(s.msg);
    },
    reset() {
      cancelAnimationFrame(st.raf);
      st.scene = null;
      st.cur = null;
      st.queue = [];
      st.steps = null;
      st.waiting = false;
      st.frozen = null;
      st.rules.clear();
      hintEl.classList.remove("on");
      setRule("");
      layer.classList.remove("waiting", "playing");
      clearPatches();
    },
    lpOverride(who) { return st.lp[who]; },

    // ----- deterministic playback (mocks, replay video) -----
    // Plan durations (1x ms) of the visible scenes.
    plan(cues, baseChain) { return planDurations(cues, baseChain); },
    // Freeze at visible scene k, fraction f in [0,1]. Incremental when moving forward.
    freeze(cues, baseChain, k, f, token) {
      let fz = st.frozen;
      if (!fz || fz.token !== token || k < fz.k) {
        api.reset();
        fz = st.frozen = {token, steps: planSteps(cues, baseChain), si: -1, k: -1, scene: null};
      }
      const vis = [];
      fz.steps.forEach((s, i) => { if (s.dur > 0) vis.push(i); });
      if (k >= vis.length) k = vis.length - 1;
      if (k < 0) return 0;
      while (fz.k < k) {
        if (fz.scene) {
          const s0 = fz.steps[vis[fz.k]];
          fz.scene.at(s0.dur);
          fz.scene.finish();
          fz.scene = null;
          stage.innerHTML = "";
          if (!s0.dimNext) dimEl.style.opacity = "0";
        }
        fz.k += 1;
        fz.scene = mount(fz.steps[vis[fz.k]]);
      }
      const step = fz.steps[vis[k]];
      fz.scene.at(Math.max(0, Math.min(1, f)) * step.dur);
      // Deterministic screenshots: show the current scene's rule note if the
      // switch is on (the once-per-game set only applies to live playback).
      const gid = ruleOn() ? ruleNoteFor(step, (opts.getView() || {}).turn_player) : null;
      setRule(gid ? ruleShort(gid) : "");
      return vis.length;
    },
    // Real-time playback of one cue list over the current board (replay showFrame).
    playOnce(cues, baseChain) {
      api.reset();
      return new Promise((resolve) => {
        st.queue.push({cues, msg: null, once: resolve});
        const apply0 = opts.apply;
        opts.apply = (m) => { if (m) apply0(m); };
        const idle0 = opts.onIdle;
        opts.onIdle = () => { opts.apply = apply0; opts.onIdle = idle0; resolve(); };
        void baseChain;
        pump();
      });
    },
    // Wait for the scene images (bounded: a missing image never stalls a video).
    waitImages(ms) {
      const imgs = [...layer.querySelectorAll("img")].filter((im) => !im.complete);
      if (!imgs.length) return Promise.resolve();
      return Promise.race([
        Promise.all(imgs.map((im) => new Promise((res) => {
          im.addEventListener("load", res, {once: true});
          im.addEventListener("error", res, {once: true});
        }))),
        new Promise((res) => setTimeout(res, ms || 2500)),
      ]);
    },
  };
  return api;
}
