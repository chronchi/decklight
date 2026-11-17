// Card search component (docs/design/deck-builder.md "Search and cards as shared services"): the search bar with its mode
// menu, the filter chips and the result list. Pure builders over a state object; the host feature owns the state,
// sends the request this module builds, and passes in what a result row's main button does (the deck builder adds a
// copy; a future Cards tab opens the card). Server contract: docs/design/api/cards-and-decks.md (`cards_search`).
import {esc, artImg} from "../../render/card.js";

export const SEARCH_MODES = [
  ["names_text", "Names & text", "Card names (typos are fine) and words in the card text. Updates as you type."],
  ["describe", "Describe a card", "Say what the card does, like “a tuner that summons itself and searches”. Runs when you tap Search."],
];
export const KIND_CHIPS = [["monster", "Monster"], ["spell", "Spell"], ["trap", "Trap"], ["extra", "Extra Deck"]];

export function searchState() {
  return {q: "", mode: "names_text", chips: [], rows: null, total: 0, loading: false, menu: false, err: null,
    sent: null, describeReady: false, f: {}};
}

// The full filter sheet (user 2026-11-13: "a filter symbol ... opens a full filter window ... there is way more
// information on the cards"). [state key, heading, [[value, label]], cards_search filter key].
const RACES = ["Warrior", "Spellcaster", "Fairy", "Fiend", "Zombie", "Machine", "Aqua", "Pyro", "Rock", "Winged Beast",
  "Plant", "Insect", "Thunder", "Dragon", "Beast", "Beast-Warrior", "Dinosaur", "Fish", "Sea Serpent", "Reptile", "Psychic",
  "Divine-Beast", "Wyrm", "Cyberse", "Illusion"];
export const FILTER_GROUPS = [
  ["frame", "Monster card", [["normal", "Normal"], ["effect", "Effect"], ["ritual", "Ritual"], ["fusion", "Fusion"],
    ["synchro", "Synchro"], ["xyz", "Xyz"], ["pendulum", "Pendulum"], ["link", "Link"]]],
  ["attribute", "Attribute", ["LIGHT", "DARK", "EARTH", "WATER", "FIRE", "WIND", "DIVINE"].map((a) => [a, a])],
  ["race", "Type", RACES.map((r) => [r === "Divine-Beast" ? "Divine" : r, r])],  // the server names it "Divine"
  ["stype", "Spell and Trap", [["normal", "Normal"], ["quickplay", "Quick-Play"], ["continuous", "Continuous"],
    ["equip", "Equip"], ["field", "Field"], ["ritual", "Ritual"], ["counter", "Counter"]], "st_type"],
];
// [state key, label, cards_search key, placeholder max]
export const FILTER_RANGES = [["lvl", "Level / Rank / Link", "level", 13], ["atk", "ATK", "atk", 5000], ["def", "DEF", "def", 5000]];
export function filterCount(st) {
  const f = st.f || {};
  let n = 0;
  for (const [key] of FILTER_GROUPS) n += (f[key] || []).length;
  for (const [key] of FILTER_RANGES) if (f[key] && (f[key][0] != null || f[key][1] != null)) n += 1;
  return n;
}
export function filterSheetHTML(st, p) {
  const f = st.f || {};
  let h = "";
  for (const [key, label, items] of FILTER_GROUPS) {
    h += '<section class="fsec" aria-labelledby="fh-' + key + '"><h3 class="flabel" id="fh-' + key + '">' + esc(label) + "</h3>" +
      '<div class="fchips">' + items.map(([v, t]) => {
        const on = (f[key] || []).includes(v);
        return '<button class="dkchip' + (on ? " on" : "") + '" aria-pressed="' + on + '" data-act="' + p + 'f" data-fk="' + key +
          '" data-fv="' + esc(v) + '">' + esc(t) + "</button>";
      }).join("") + "</div></section>";
    if (key === "race") {
      for (const [rk, rl, , max] of FILTER_RANGES) {
        const r = f[rk] || [];
        const box = (i, ph) => '<input type="number" inputmode="numeric" min="0" max="' + max + '" placeholder="' + ph +
          '" aria-label="' + esc(rl) + (i ? " at most" : " at least") + '" data-dbinput="f:' + rk + ":" + i + '" value="' +
          (r[i] == null ? "" : r[i]) + '">';
        h += '<section class="fsec frange" aria-labelledby="fh-' + rk + '"><h3 class="flabel" id="fh-' + rk + '">' + esc(rl) +
          '</h3><div class="frow">' + box(0, "0") + '<span aria-hidden="true">to</span>' + box(1, String(max)) + "</div></section>";
      }
    }
  }
  return h;
}

// The cards_search request for the current state (null when there is nothing to ask).
export function searchRequest(st, limit = 60) {
  const q = st.q.trim();
  const kinds = st.chips.filter((c) => c !== "extra");
  const nf = filterCount(st);
  if (!q && !st.chips.length && !nf) return null;
  const filters = {};
  const f = st.f || {};
  for (const [key, , , api] of FILTER_GROUPS) if ((f[key] || []).length) filters[api || key] = f[key];
  for (const [key, , api] of FILTER_RANGES) {
    const r = f[key];
    if (r && (r[0] != null || r[1] != null)) filters[api] = [r[0] == null ? 0 : r[0], r[1] == null ? 99999 : r[1]];
  }
  if (kinds.length) filters.kind = kinds;
  if (st.chips.includes("extra")) filters.extra_only = true;
  return {t: "cards_search", q, mode: st.mode, filters, offset: 0, limit};
}

const ICON_FILTER = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 5h14M6 10h8M8.5 15h3"/></svg>';
function filterBtnHTML(st, p) {
  const n = filterCount(st);
  return '<button class="dkchip csch-fbtn' + (n ? " on" : "") + '" data-act="' + p + 'filters" aria-haspopup="dialog" aria-label="' +
    (n ? "Filters, " + n + " on" : "More filters") + '">' + ICON_FILTER + "<span>Filters</span>" + (n ? "<b>" + n + "</b>" : "") +
    "</button>";
}
const ICON_SEARCH = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5"/><path d="m13 13 4.5 4.5"/></svg>';
const ICON_X = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 6l8 8M14 6l-8 8"/></svg>';
const ICON_CHEV = '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 8l5 5 5-5"/></svg>';

// Local filter over cards the client already has (the deck builder's Deck view): every word of the query must appear
// in the name, type line or text, and the card must fit the kind chips. Null when nothing filters.
export function localMatcher(st) {
  const words = st.q.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const kinds = st.chips.filter((c) => c !== "extra");
  if (!words.length && !st.chips.length) return null;
  return (c) => {
    if (!c) return false;
    if (kinds.length && !kinds.includes(c.kind)) return false;
    if (st.chips.includes("extra") && !c.extra) return false;
    const hay = (c.name + " " + (c.line || "") + " " + (c.text || "")).toLowerCase();
    return words.every((w) => hay.includes(w));
  };
}

// The bottom search bar: mode button · input · (Search in describe mode), with the chip row above it.
// o.local: the bar filters a list the host already shows (no mode menu; placeholder from o.placeholder).
export function searchBarHTML(st, p, o = {}) {
  const mode = SEARCH_MODES.find((m) => m[0] === st.mode) || SEARCH_MODES[0];
  const describe = !o.local && st.mode === "describe";
  const chips = '<div class="csch-chiprow"><div class="csch-chips" role="group" aria-label="Card kind">' + KIND_CHIPS.map(([k, t]) => {
    const on = st.chips.includes(k);
    return '<button class="dkchip' + (on ? " on" : "") + '" aria-pressed="' + on + '" data-act="' + p + 'chip" data-chip="' +
      k + '">' + t + "</button>";
  }).join("") + "</div>" + (o.local ? "" : filterBtnHTML(st, p)) + "</div>";
  const menu = st.menu && !o.local
    ? '<div class="csch-menu" role="menu" aria-label="Search mode">' + SEARCH_MODES.map(([k, t, d]) =>
      '<button role="menuitemradio" aria-checked="' + (k === st.mode) + '" class="csch-mi' + (k === st.mode ? " sel" : "") +
      '" data-act="' + p + 'mode" data-smode="' + k + '"><span class="mt">' + t + '</span><span class="md">' + d +
      "</span></button>").join("") + "</div>"
    : "";
  return '<div class="csch-bar">' + chips + menu +
    '<div class="csch-row">' + (o.local ? "" : '<button class="csch-mode" data-act="' + p +
    'modemenu" aria-haspopup="menu" aria-expanded="' + st.menu + '" aria-label="Search mode: ' + mode[1] + '"><span>' +
    mode[1] + "</span>" + ICON_CHEV + "</button>") +
    '<label class="csch-in">' + ICON_SEARCH + '<input type="search" data-dbinput="q" enterkeyhint="search" autocomplete="off" ' +
        'spellcheck="false" aria-label="' + (o.local ? o.placeholder.replace("…", "") : describe ? "Describe a card"
      : "Search card names and text") + '" placeholder="' + (o.local ? o.placeholder : describe ? "Describe a card…"
      : "Search cards…") + '" value="' + esc(st.q) + '">' + (st.q ? '<button class="csch-clear" data-act="' + p +
    'clear" aria-label="Clear search">' + ICON_X + "</button>" : "") + "</label>" +
    (describe ? (st.loading ? '<button class="csch-go stop" data-act="' + p + 'stop" aria-label="Stop searching">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="5" y="5" width="10" height="10" rx="1.5"/></svg>Stop</button>'
      : '<button class="primary csch-go" data-act="' + p + 'go">Search</button>') : "") + "</div></div>";
}

function rowHTML(r, o) {
  const n = o.countOf ? o.countOf(r.code) : 0;
  const why = r.match && r.match.by === "text" && r.match.snippet
    ? '<span class="csch-why">' + esc(r.match.snippet) + "</span>" : "";
  return '<li class="csch-r"><button class="csch-open" data-act="' + o.p + 'card" data-code="' + r.code + '">' +
    '<span class="csch-th k-' + esc(r.kind) + '">' + artImg(r.code, "small", 'alt="" loading="lazy"') + "</span>" +
    '<span class="csch-tx"><span class="nm">' + esc(r.name) + '</span><span class="ln">' + esc(r.line) +
    (r.extra && !o.add ? " · Extra Deck" : "") + "</span>" + why + "</span></button>" +
    (n && !o.add ? '<span class="csch-n" aria-label="' + n + " in deck" + '">×' + n + "</span>" : "") +
    (o.add ? '<span class="csch-adds">' + addBtn(r, o, r.extra ? "extra" : "main") + addBtn(r, o, "side") + "</span>" : "") +
    "</li>";
}

// "+ Main" / "+ Side" (Extra Deck cards: "+ Extra" / "+ Side"), each showing the copies already in that part.
const PLUS = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 4v12M4 10h12"/></svg>';
function addBtn(r, o, sec) {
  const nm = sec === "main" ? "Main" : sec === "extra" ? "Extra" : "Side";
  const k = o.countIn ? o.countIn(r.code, sec) : 0;
  const full = o.countOf && o.countOf(r.code) >= 3;
  return '<button class="csch-add2' + (k ? " has" : "") + '" data-act="' + o.p + 'add" data-sec="' + sec + '" data-code="' +
    r.code + '" aria-label="Add ' + esc(r.name) + " to the " + nm + " Deck" + (k ? ", " + k + " there now" : "") + '"' +
    (full ? " disabled" : "") + ">" + PLUS + "<span>" + nm + "</span>" + (k ? '<b>' + k + "</b>" : "") + "</button>";
}

// Examples pattern (user 2026-11-13: "give an example ... a core UI design in general to really guide the user"):
// an empty state says what to do, then offers 2-3 real inputs; tapping one fills the field and runs it.
const EXAMPLES = {
  describe: ["a tuner that special summons itself", "negate a monster effect from the hand",
    "banish from the graveyard to draw cards"],
  names_text: ["Ash Blossom", "destroy all monsters", "add 1 Spell from your Deck"],
};
export function examplesHTML(lead, examples, p) {
  return '<div class="exhint"><p class="emptytx">' + esc(lead) + '</p><p class="exlead" id="ex-' + p + '">Try</p>' +
    '<ul class="exlist" aria-labelledby="ex-' + p + '">' + examples.map((q) => '<li><button class="exbtn" data-act="' + p +
    'ex" data-q="' + esc(q) + '">“' + esc(q) + "”</button></li>").join("") + "</ul></div>";
}

// The result list with its status line. o = {p: action prefix, add: bool, countOf(code) -> copies in the deck}.
export function resultsHTML(st, o) {
  if (st.err) return '<p class="emptytx" role="alert">' + esc(st.err) + "</p>";
  if (st.mode === "describe" && !st.describeReady) {
    return '<div class="csch-note"><p><b>Describe a card</b> finds cards from a description, like “a light tuner that ' +
      "special summons itself and searches”.</p><p>It needs a one-time 1.2 GB download and then works offline. " +
      "Searching takes under half a second.</p>" +
      '<button class="bigbtn" data-act="' + o.p + 'getdescribe">Download in Settings</button></div>';
  }
  // Busy: a centred spinner (shown after 300 ms, so fast name searches never flash it). Stop sits in the bar.
  if (st.loading) {
    return '<div class="csch-busy" role="status" aria-live="polite"><span class="csch-spin" aria-hidden="true"></span>' +
      "<span>Searching…</span></div>";
  }
  if (st.rows === null) {
    return st.mode === "describe"
      ? examplesHTML("Describe what the card does, then tap Search.", EXAMPLES.describe, o.p)
      : examplesHTML("Type a card name or words from its text, or pick a kind below.", EXAMPLES.names_text, o.p);
  }
  const status = st.total === 0 ? "No cards match." :
    st.total + (st.total === 1 ? " card" : " cards") + (st.total > st.rows.length ? ", showing " + st.rows.length : "");
  return '<p class="csch-status" role="status">' + status + "</p>" +
    (st.rows.length ? '<ul class="csch-list">' + st.rows.map((r) => rowHTML(r, o)).join("") + "</ul>" : "");
}
