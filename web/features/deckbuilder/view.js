// Deck builder views (docs/design/deck-builder.md): the Decks tab list, the editor and its sheets. Pure builders: state in,
// HTML out. Actions are `db:*` (index.js handles them); inputs carry data-dbinput.
import {esc, artImg} from "../../render/card.js";
import {searchBarHTML, resultsHTML, localMatcher, filterSheetHTML, filterCount} from "../cards/search.js";

const SECTIONS = [["main", "Main Deck"], ["extra", "Extra Deck"], ["side", "Side Deck"]];
const I = {
  back: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12 4 6 10l6 6"/></svg>',
  more: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="4.5" cy="10" r="1.3"/><circle cx="10" cy="10" r="1.3"/><circle cx="15.5" cy="10" r="1.3"/></svg>',
  close: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg>',
  plus: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 4v12M4 10h12"/></svg>',
  minus: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h12"/></svg>',
  chev: '<svg class="dbchev" viewBox="0 0 20 20" aria-hidden="true"><path d="m8 4 6 6-6 6"/></svg>',
};

export function countsText(c) {
  return "Main " + c.main + " · Extra " + c.extra + " · Side " + c.side;
}

// One legality line: "Ready to play" or the first problem (+ how many more), always as text.
export function legalLine(legal) {
  if (!legal) return "";
  if (legal.ok) return '<span class="dbok">Ready to play</span>';
  const p = legal.problems || [];
  const first = typeof p[0] === "string" ? p[0] : (p[0] && p[0].text) || "Not legal";
  return '<span class="dbbad">' + esc(first) + (p.length > 1 ? " · " + (p.length - 1) + " more" : "") + "</span>";
}

function cover(code, kind) {
  return '<span class="dbcover k-' + esc(kind || "monster") + '">' + (code ? artImg(code, "small", 'alt="" loading="lazy"') : "") + "</span>";
}

// The Decks tab (inside the lobby scroller).
export function listHTML(st) {
  let h = '<h2 class="sr-only">Your decks</h2><p class="tabintro">Build decks, then play them against the bot or people.</p>' +
    '<div class="dbtop"><button class="bigbtn" data-act="db:new">New deck</button>' +
    '<button class="bigbtn" data-act="db:import">Import</button></div>';
  if (st.decks === null) return h + '<p class="emptytx" role="status">Loading your decks…</p>';
  if (!st.decks.length) {
    return h + '<p class="emptytx">No decks yet. Start a new one, or import a .ydk file or a copied deck list.</p>';
  }
  return h + '<ul class="dbrows">' + st.decks.map((d) =>
    '<li><button class="dbrow" data-act="db:open" data-id="' + esc(d.id) + '">' + cover(d.cover, st.kindOf(d.cover)) +
    '<span class="dbtx"><span class="nm">' + esc(d.name) + '</span><span class="ln">' + countsText(d.counts) +
    '</span><span class="lg">' + legalLine(d.legal) + "</span></span>" + I.chev + "</button></li>").join("") + "</ul>";
}

// Unique cards of one section in deck order, with their copy counts.
function groups(codes) {
  const seen = new Map();
  for (const c of codes) seen.set(c, (seen.get(c) || 0) + 1);
  return [...seen.entries()];
}

function tileHTML(sec, code, n, st) {
  const c = st.card(code);
  const name = c ? c.name : "Card " + code;
  return '<li class="dbtile"><button class="dbcard k-' + esc(c ? c.kind : "unknown") +
    '" data-act="db:s-card" data-sec="' + sec + '" data-code="' + code + '" aria-haspopup="dialog" aria-label="' +
    esc(name) + ", " + n + (n === 1 ? " copy" : " copies") + '">' + artImg(code, "small", 'alt="" loading="lazy"') +
    '<span class="dbname" aria-hidden="true">' + esc(name) + "</span>" +
    (n > 1 ? '<span class="dbn" aria-hidden="true">×' + n + "</span>" : "") + "</button></li>";
}

// The deck grid. With a query or chips from the shared search bar, only matching cards show (counts stay the deck's).
function deckViewHTML(st) {
  const d = st.edit;
  const match = localMatcher(st.search);
  let h = "", shown = 0, all = 0;
  for (const [sec, title] of SECTIONS) {
    const g0 = groups(d[sec]);
    const g = match ? g0.filter(([code]) => match(st.card(code))) : g0;
    for (const [, n] of g0) all += n;
    for (const [, n] of g) shown += n;
    if (match && !g.length) continue;
    h += '<section class="dbsec" aria-labelledby="dbh-' + sec + '"><h3 id="dbh-' + sec + '">' + title +
      ' <span class="dbcount">' + d[sec].length + "</span></h3>";
    if (!g0.length) {
      h += '<p class="dbempty">' + (sec === "side" ? "Empty. Tap a card, then + Side."
        : sec === "extra" ? "Empty. Fusion, Synchro, Xyz and Link monsters land here." : "Empty. Add cards to start.") + "</p>";
    } else {
      h += '<ul class="dbgrid">' + g.map(([code, n]) => tileHTML(sec, code, n, st)).join("") + "</ul>";
    }
    h += "</section>";
  }
  if (!match) return h;
  const status = shown ? shown + " of " + all + (all === 1 ? " card matches" : " cards match")
    : "No cards in this deck match. Try Add cards to search every card.";
  return '<p class="csch-status" role="status">' + status + "</p>" + h;
}

// The editor: header, counts + legality, the Deck / Add cards switch, the body and (Add cards) the search bar.
export function editorHTML(st) {
  const d = st.edit;
  const c = st.counts(d);
  const total = c.main + c.extra + c.side;
  // Saving takes well under a second, so only a failure is shown (the row no longer shifts while adding).
  const save = d.saveErr ? "Not saved: " + d.saveErr : "";
  let h = '<div id="lobby" class="dbedit" data-feature="deckbuilder"><header class="dbhead">' +
    '<button class="iconbtn" data-act="db:close" aria-label="Back to your decks">' + I.back + "</button>" +
    '<input class="dbtitle" data-dbinput="name" value="' + esc(d.name) + '" aria-label="Deck name" maxlength="60" spellcheck="false">' +
    '<button class="dbexport" data-act="db:export">Export</button>' +
    '<button class="iconbtn" data-act="db:menu" aria-label="More deck actions" aria-haspopup="menu">' + I.more + "</button></header>" +
    '<div class="dbsub"><span class="ct">' + countsText(c) + '</span><span class="lg">' + legalLine(st.legal) + "</span>" +
    '<span class="sv' + (d.saveErr ? " bad" : "") + '" role="status">' + esc(save) + "</span></div>" +
    '<div class="dbseg" role="tablist" aria-label="Editor view">' +
    ['deck', 'add'].map((k) => '<button role="tab" aria-selected="' + (st.view === k) + '"' + (st.view === k ? ' class="sel"' : "") +
      ' data-act="db:view" data-view="' + k + '">' + (k === "deck" ? "Deck (" + total + ")" : "Add cards") + "</button>").join("") +
    "</div>";
  if (st.view === "deck") {
    h += '<div class="lscroll dbbody dbresults">' + deckViewHTML(st) + "</div>" +
      searchBarHTML(st.search, "db:s-", {local: true, placeholder: "Find in this deck…"});
  } else {
    h += '<div class="lscroll dbbody dbresults">' + resultsHTML(st.search, {p: "db:s-", add: true, countOf: st.copies, countIn: st.copiesIn}) +
      "</div>" + searchBarHTML(st.search, "db:s-");
  }
  return h + "</div>";
}

function sheet(key, title, body, foot) {
  return '<div class="dbscrim" data-feature="deckbuilder" data-key="dbscrim" data-act="db:closesheet"></div>' +
    '<div class="dbsheet" data-feature="deckbuilder" data-key="dbsheet-' + key + '" role="dialog" aria-modal="true" tabindex="-1" data-close="db:closesheet" aria-label="' + esc(title) + '">' +
    '<div class="grab" aria-hidden="true"></div><div class="mhead"><span class="mtitle">' + esc(title) + '</span><button class="iconbtn" data-act="db:closesheet" aria-label="Close">' +
    I.close + "</button></div>" + '<div class="mbody">' + body + "</div>" + (foot ? '<div class="dbfoot">' + foot + "</div>" : "") + "</div>";
}

// One row per deck part the card may go in (Main or Extra, and Side): name, − count +. Max 3 copies in all.
function secStepper(st, code, sec) {
  const n = st.copiesIn(code, sec);
  const nm = sec === "main" ? "Main Deck" : sec === "extra" ? "Extra Deck" : "Side Deck";
  const full = st.copies(code) >= 3;
  return '<div class="dbsecrow" role="group" aria-label="' + nm + '"><span class="nm">' + nm + "</span>" +
    '<button class="iconbtn" data-act="db:cminus" data-sec="' + sec + '" data-code="' + code + '" aria-label="Remove one from ' +
    nm + '"' + (n ? "" : " disabled") + ">" + I.minus + "</button>" +
    '<span class="n" aria-live="polite" aria-label="' + n + " in " + nm + '">' + n + "</span>" +
    '<button class="iconbtn" data-act="db:cplus" data-sec="' + sec + '" data-code="' + code + '" aria-label="Add one to ' +
    nm + '"' + (full ? " disabled" : "") + ">" + I.plus + "</button></div>";
}

// The open sheet (export, import, menu, delete confirm, card details), or "".
export function sheetHTML(st) {
  const s = st.sheet;
  if (!s) return "";
  if (s.kind === "export") {
    return sheet("export", "Export " + st.edit.name,
      '<p class="dbhelp">A .ydk deck list works in EDOPro, YGO Omega and most deck sites.</p>' +
      '<textarea class="dbydk" readonly aria-label="Deck list (.ydk)" spellcheck="false">' + esc(s.ydk) + "</textarea>" +
      (s.note ? '<p class="dbnote" role="status">' + esc(s.note) + "</p>" : '<p class="dbnote" role="status"></p>'),
      '<button class="primary" data-act="db:copy">Copy</button>' +
      '<button class="bigbtn" data-act="db:share">' + (st.canShare ? "Share or save file" : "Download .ydk") + "</button>" +
      '<button class="linkbtn" data-act="db:copyydke">Copy as ydke:// link</button>');
  }
  if (s.kind === "import") {
    return sheet("import", "Import a deck",
      '<label class="dblabel" for="dbimpname">Name</label><input id="dbimpname" class="dbfield" data-dbinput="impname" ' +
      'maxlength="60" placeholder="Imported deck" value="' + esc(s.name || "") + '">' +
      '<label class="dblabel" for="dbimptext">Deck list (.ydk text or a ydke:// link)</label>' +
      '<textarea id="dbimptext" class="dbydk" data-dbinput="imptext" spellcheck="false" placeholder="#main&#10;14558128&#10;…">' +
      esc(s.text || "") + "</textarea>" +
      '<input type="file" id="dbfile" accept=".ydk,text/plain" hidden data-dbinput="file">' +
      (s.note ? '<p class="dbnote" role="alert">' + esc(s.note) + "</p>" : ""),
      '<button class="primary" data-act="db:doimport"' + ((s.text || "").trim() ? "" : " disabled") + ">Import</button>" +
      '<button class="bigbtn" data-act="db:pickfile">Open a .ydk file</button>');
  }
  if (s.kind === "menu") {
    return sheet("menu", st.edit.name,
      '<div class="dbmenu" role="menu">' +
      '<button role="menuitem" data-act="db:duplicate">Duplicate</button>' +
      '<button role="menuitem" data-act="db:export">Export</button>' +
      '<button role="menuitem" class="danger" data-act="db:askdelete">Delete…</button></div>');
  }
  if (s.kind === "delete") {
    return sheet("delete", "Delete " + st.edit.name + "?",
      '<p class="dbhelp">The deck leaves your list. Its file is kept in the app’s trash folder, so it can be recovered.</p>',
      '<button class="primary danger" data-act="db:delete">Delete deck</button>' +
      '<button class="bigbtn" data-act="db:closesheet">Keep it</button>');
  }
  if (s.kind === "filters") {
    const n = filterCount(st.search);
    return sheet("filters", "Filters", filterSheetHTML(st.search, "db:s-"),
      '<button class="bigbtn" data-act="db:s-fclear"' + (n ? "" : " disabled") + ">Clear all</button>" +
      '<button class="primary" data-act="db:s-fapply">' + (n ? "Show results · " + n + (n === 1 ? " filter" : " filters") : "Show results") +
      "</button>");
  }
  if (s.kind === "card") {
    const c = st.card(s.code);
    const name = c ? c.name : "Card " + s.code;
    const inMain = st.copies(s.code);
    return sheet("card", name,
      '<div class="dbcd"><span class="dbcdart k-' + esc(c ? c.kind : "unknown") + '">' + artImg(s.code, "small", 'alt=""') + "</span>" +
      '<div><p class="ln">' + esc(c ? c.line : "") + (c && c.extra ? " · Extra Deck" : "") + "</p>" +
      '<p class="dbinmain">' + (inMain ? inMain + (inMain === 1 ? " copy" : " copies") + " in this deck" : "Not in this deck") +
      "</p></div></div>" + '<p class="dbcdtext">' + esc(c && c.text ? c.text : "Loading the card text…") + "</p>",
      secStepper(st, s.code, st.isExtra(s.code) ? "extra" : "main") + secStepper(st, s.code, "side"));
  }
  return "";
}
