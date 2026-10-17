// Card HTML builders. Pure functions of Card objects; no game logic.
// Without images (mock mode, 404, offline) a card renders as a quiet
// placeholder frame: name band at the top, an empty art window below.
const TOKEN = new URLSearchParams(location.search).get("t") || "";

export function imgUrl(code, size) {
  return "/img/" + code + ".jpg?size=" + (size || "small") + (TOKEN ? "&t=" + encodeURIComponent(TOKEN) : "");
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
  return phHTML(card) +
    '<img class="face" loading="lazy" alt="" src="' + imgUrl(card.code, size || "small") + '"' +
    ' onerror="this.remove()">';
}

// The battle-relevant stat, shown under the card (never rotated with it).
export function statHTML(card) {
  if (!card || card.faceup === false) return "";
  if (card.pos !== "atk" && card.pos !== "def") return "";
  if (typeof card.atk !== "number") return "";
  if (card.link > 0) return '<div class="stat atk">' + card.atk + '<i>L' + card.link + "</i></div>";
  return card.pos === "def"
    ? '<div class="stat def">' + card.def + "</div>"
    : '<div class="stat atk">' + card.atk + "</div>";
}

function badges(card) {
  if (!card) return "";
  const out = [];
  if (card.overlays && card.overlays.length) out.push('<span class="bdg xyz">' + card.overlays.length + "</span>");
  if (card.counters) out.push('<span class="bdg ctr">' + card.counters + "</span>");
  if (!out.length) return "";
  return '<div class="badges">' + out.join("") + "</div>";
}

// A card filling its container (slot, hand cell, grid cell).
// opts: {mock, actionable, selected, pick, rot (defense: rotate 90deg), stat, focus, bot}
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
  const uid = card.uid ? ' data-uid="' + esc(card.uid) + '"' : "";
  const inner = hidden ? "" : faceHTML(card, opts.mock);
  const pick = opts.pick ? '<div class="pick">' + opts.pick + "</div>" : "";
  return '<div class="' + cls.join(" ") + '"' + uid + ">" + inner + "</div>" +
    (opts.stat ? statHTML(card) : "") + badges(card) + pick;
}

// Small fixed-size thumbnail (prompt rows, log, chain, info panel).
export function thumbHTML(card, mock, cls) {
  cls = "thumb" + (cls ? " " + cls : "");
  if (!card) return '<div class="' + cls + ' none"></div>';
  if (isHidden(card)) return '<div class="' + cls + ' back"></div>';
  return '<div class="' + cls + '">' + faceHTML(card, mock) + "</div>";
}
