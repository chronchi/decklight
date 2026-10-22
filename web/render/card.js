// Card HTML builders. Pure functions of Card objects; no game logic.
// Without images (mock mode, 404, offline) a card renders as a quiet
// placeholder frame: name band at the top, an empty art window below.
const TOKEN = new URLSearchParams(location.search).get("t") || "";

export function imgUrl(code, size) {
  return "/img/" + code + ".jpg?size=" + (size || "small") + (TOKEN ? "&t=" + encodeURIComponent(TOKEN) : "");
}

// Card art is opt-in (settings sheet, off by default): no /img requests at
// all unless the user switches it on. The face-down card back (code 0, the
// server's SVG) is not card art and is unaffected (hidden cards never call here).
let artOn = false;
export function setCardArt(on) { artOn = !!on; }
export function cardArtOn() { return artOn; }
// The <img> for a card face, or "" when art is off (or no code). `attrs` is
// the call site's own attribute string, kept exactly as it renders today.
export function artImg(code, size, attrs) {
  if (!artOn || !code) return "";
  return '<img ' + (attrs ? attrs + ' ' : '') + 'src="' + imgUrl(code, size || "small") + '" onerror="this.remove()">';
}

export function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function isHidden(card) {
  return !card || card.faceup === false || (!card.code && !card.name);
}

export function isMonster(card) {
  return !!card && (card.level > 0 || card.rank > 0 || card.link > 0 || card.atk > 0 || card.def > 0);
}

// Placeholder face: name band + art window. Used for every imageless card.
export function phHTML(card) {
  return '<div class="ph"><span class="nm">' + esc((card && card.name) || "#" + (card && card.code)) +
    '</span><span class="art"></span></div>';
}

// Face of a face-up card: image over the placeholder (the image hides itself on error).
function faceHTML(card, mock, size) {
  if (mock || !card.code) return phHTML(card);
  return phHTML(card) + artImg(card.code, size || "small", 'class="face" loading="lazy" alt=""');
}

// The battle-relevant stat, shown under the card (never rotated with it).
export function statHTML(card) {
  if (!card || card.faceup === false) return "";
  return statInner(card);
}
function statInner(card) {
  if (card.pos !== "atk" && card.pos !== "def") return "";
  if (typeof card.atk !== "number") return "";
  if (card.link > 0) return '<div class="stat atk" aria-hidden="true">' + card.atk + '<i>L' + card.link + "</i></div>";
  return card.pos === "def"
    ? '<div class="stat def" aria-hidden="true">' + card.def + "</div>"
    : '<div class="stat atk" aria-hidden="true">' + card.atk + "</div>";
}

function badges(card) {
  if (!card) return "";
  const out = [];
  if (card.overlays && card.overlays.length) out.push('<span class="bdg xyz">' + card.overlays.length + "</span>");
  if (card.counters) out.push('<span class="bdg ctr">' + card.counters + "</span>");
  if (!out.length) return "";
  return '<div class="badges" aria-hidden="true">' + out.join("") + "</div>";
}

// Spoken name and state of a card: "Ash Blossom, Defense Position, DEF 1800, can act".
export function cardLabel(card, opts) {
  opts = opts || {};
  if (!card) return "";
  const hidden = isHidden(card);
  const parts = [hidden ? (card.name ? "Set card: " + card.name : "Face-down card") : card.name || "#" + card.code];
  if (!hidden && card.pos === "atk") parts.push("Attack Position");
  if (!hidden && card.pos === "def") parts.push("Defense Position");
  if (opts.stat && !hidden && typeof card.atk === "number" && (card.pos === "atk" || card.pos === "def")) {
    parts.push(card.link > 0 ? "ATK " + card.atk + ", Link " + card.link : card.pos === "def" ? "DEF " + card.def : "ATK " + card.atk);
  }
  if (card.overlays && card.overlays.length) parts.push(card.overlays.length + (card.overlays.length === 1 ? " material" : " materials"));
  if (card.counters) parts.push(card.counters + (card.counters === 1 ? " counter" : " counters"));
  if (opts.selected) parts.push("selected");
  if (opts.actionable) parts.push("can act");
  if (opts.hint) parts.push("hint " + opts.hint.rank + ", bot's confidence " + opts.hint.pct + "%");
  return parts.join(", ");
}

// A card filling its container (slot, hand cell, grid cell).
// opts: {mock, actionable, selected, pick, rot (defense: rotate 90deg), stat, focus, bot,
//        kb (board/hand: the card itself is the control, reachable by Tab, named for screen readers),
//        hint ({rank, pct}: "Ask the bot" badge + spoken hint rank/confidence)}
export function cardHTML(card, opts) {
  opts = opts || {};
  if (!card) return "";
  const hidden = isHidden(card);
  const cls = ["card"];
  if (hidden) cls.push("back");
  if (opts.rot) cls.push("rot");
  if (opts.actionable) cls.push("actionable");
  if (opts.selected) cls.push("selected");
  if (opts.focus) cls.push("focus");
  if (opts.hint) cls.push("hinted");
  const uid = card.uid ? ' data-uid="' + esc(card.uid) + '"' : "";
  const key = card.uid ? ' data-key="' + esc(card.uid + ":" + (card.code || 0)) + '"' : "";
  const inner = hidden ? "" : faceHTML(card, opts.mock);
  const pick = opts.pick ? '<div class="pick" aria-hidden="true">' + opts.pick + "</div>" : "";
  const hb = opts.hint
    ? '<span class="hintbadge" aria-hidden="true">' + opts.hint.rank + " · " + opts.hint.pct + "%</span>"
    : "";
  // Controls only where tapping does something: actionable, or face-up (opens the details).
  const a11y = opts.kb && (opts.actionable || !hidden)
    ? ' role="button" tabindex="0" aria-label="' + esc(cardLabel(card, opts)) + '"' + (opts.actionable ? ' aria-haspopup="menu"' : "")
    : opts.kb ? ' role="img" aria-label="' + esc(cardLabel(card, opts)) + '"' : "";
  return '<div class="' + cls.join(" ") + '"' + uid + key + a11y + ">" + inner + hb + "</div>" +
    (opts.stat ? statHTML(card) : "") + badges(card) + pick;
}

// Small fixed-size thumbnail (prompt rows, log, chain, info panel).
export function thumbHTML(card, mock, cls) {
  cls = "thumb" + (cls ? " " + cls : "");
  // Always beside its name in words: hidden from screen readers.
  if (!card) return '<div class="' + cls + ' none" aria-hidden="true"></div>';
  if (isHidden(card)) return '<div class="' + cls + ' back" aria-hidden="true"></div>';
  return '<div class="' + cls + '" aria-hidden="true">' + faceHTML(card, mock) + "</div>";
}
