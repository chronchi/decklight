// Lobby deck browser (user 2026-10-31: "we have no idea how to properly filter based on decklist ... I have no idea
// about their decklists"; chose a full-screen sheet and the Forge tag as the only strength signal).
// Pure builders: state in, HTML out. The index comes from /deck_index.json (scripts/build_deck_index.py).
import {esc, imgUrl} from "./card.js";

// Filter groups: OR inside a group, AND across groups.
export const DECK_FILTERS = [
  ["forge", "decklight's training", [["trained", "Trained"], ["related", "Related"]], "bot"],
  ["mech", "Summons", [["Fusion", "Fusion"], ["Synchro", "Synchro"], ["Xyz", "Xyz"], ["Link", "Link"],
    ["Ritual", "Ritual"], ["Pendulum", "Pendulum"]]],
  ["era", "Era", [["now", "2025–26"], ["mid", "2015–24"], ["old", "Before 2015"]]],
];

function eraOf(y) {
  if (!y) return "";
  return y >= 2025 ? "now" : y >= 2015 ? "mid" : "old";
}

function facets(d, meta) {
  return {forge: d.forge ? d.forge.level : "", mech: (meta && meta.mech) || [], era: eraOf(meta && meta.year)};
}

// A deck matches the query on its name, archetype, plan, or any card in its list (then `hit` names the card).
function matchQuery(d, meta, names, q) {
  if (!q) return {ok: true};
  if (d.name.toLowerCase().includes(q)) return {ok: true};
  if (meta && ((meta.arch || "").toLowerCase().includes(q) || (meta.about || "").toLowerCase().includes(q))) {
    return {ok: true};
  }
  if (meta && names) {
    for (const [c, n] of [...meta.main, ...meta.extra]) {
      const nm = names[c] || "";
      if (nm.toLowerCase().includes(q)) return {ok: true, hit: nm, n};
    }
  }
  return {ok: false};
}

export function filterDecks(decks, index, st) {
  const q = (st.q || "").trim().toLowerCase();
  const names = index && index.names;
  const out = [];
  for (const d of decks) {
    const meta = index && index.decks[d.id];
    const f = facets(d, meta);
    let ok = true;
    for (const [key] of DECK_FILTERS) {
      const want = st.f[key] || [];
      if (!want.length) continue;
      const have = Array.isArray(f[key]) ? f[key] : [f[key]];
      if (!want.some((w) => have.includes(w))) { ok = false; break; }
    }
    if (!ok) continue;
    const m = matchQuery(d, meta, names, q);
    if (m.ok) out.push({d, meta, hit: m.hit, n: m.n});
  }
  // Fine-tuned models' decks and this session's pasted decks first: they were chosen on purpose.
  const pin = (r) => (r.d.group === "user" || r.d.group === "tuned" ? 0 : 1);
  out.sort((a, b) => pin(a) - pin(b));
  if (st.sort === "new") out.sort((a, b) => pin(a) - pin(b) || ((b.meta && b.meta.year) || 0) - ((a.meta && a.meta.year) || 0));
  return out;
}

function tagHTML(d) {
  if (!d.forge) return "";
  return '<span class="ftag ' + d.forge.level + '">' + (d.forge.level === "trained" ? "Trained" : "Related") + "</span>";
}

function sigHTML(meta, names, art) {
  if (!meta || !meta.sig.length) return "";
  if (art) {
    return '<span class="dksig" aria-hidden="true">' + meta.sig.map((c) =>
      '<img loading="lazy" decoding="async" alt="" src="' + imgUrl(c, "small") + '">').join("") + "</span>";
  }
  return '<span class="dksub">' + esc(meta.sig.map((c) => names[c] || "").filter(Boolean).join(" · ")) + "</span>";
}

function rowHTML(r, side, cur, names, art) {
  const {d, meta} = r;
  // Same-archetype tournament lists share a name: their event tells them apart.
  const plain = (x) => String(x || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  const event = meta && /\d{4}-\d\d-\d\d$/.test(meta.origin) ? meta.origin.split(",")[0] : "";
  const runtime = d.group === "tuned" ? "Fine-tuned model's deck" : d.group === "user" ? "Pasted this session" : "";
  const first = meta && (runtime || (plain(meta.arch) === plain(d.name) && event ? event : meta.arch));
  const line = meta ? [first, meta.year, meta.mech.join(", ")].filter(Boolean).join(" · ")
    : d.group === "tuned" ? "The deck a fine-tuned model was trained for" : d.group === "user" ? "Pasted this session" : "";
  const sel = d.id === cur;
  return '<li class="dkrow' + (sel ? " sel" : "") + '">' +
    '<button class="dkpick" data-' + side + 'deck="' + esc(d.id) + '" aria-current="' + sel + '"><span class="dktext">' +
    '<span class="dktop"><span class="dkname">' + esc(d.name) + "</span>" + (side === "bot" ? tagHTML(d) : "") + "</span>" +
    (line ? '<span class="dksub">' + esc(line) + "</span>" : "") +
    (r.hit ? '<span class="dkhit">Has ' + esc(r.hit) + (r.n > 1 ? " ×" + r.n : "") + "</span>" : art ? "" : sigHTML(meta, names, false)) +
    "</span>" + (art ? sigHTML(meta, names, true) : "") + "</button>" +
    '<button class="dkinfo" data-act="decklist" data-deck="' + esc(d.id) + '" aria-label="' + esc("Deck list: " + d.name) + '">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 5h9M6 10h9M6 15h9M3 5h.01M3 10h.01M3 15h.01"/></svg></button></li>';
}

function cardListHTML(title, list, names, art) {
  if (!list.length) return "";
  const n = list.reduce((s, [, k]) => s + k, 0);
  const sorted = [...list].sort((a, b) => b[1] - a[1] || (names[a[0]] || "").localeCompare(names[b[0]] || ""));
  return "<h3>" + esc(title) + " · " + n + "</h3>" + '<ul class="dkcards' + (art ? " art" : "") + '">' + sorted.map(([c, k]) => {
    const nm = names[c] || "#" + c;
    return '<li><button class="dkcard" data-primercode="' + c + '" data-primername="' + esc(nm) + '">' +
      (art ? '<img loading="lazy" decoding="async" alt="" src="' + imgUrl(c, "small") + '">' : "") +
      '<span class="dkcn">' + esc(nm) + "</span>" + (k > 1 ? '<span class="dkcx">×' + k + "</span>" : "") + "</button></li>";
  }).join("") + "</ul>";
}

function detailHTML(d, meta, names, side, art, forgeNote) {
  let b = "";
  if (d.group === "tuned") {
    b += '<p class="prnote">A fine-tuned model was trained on this list. Pick it under Options → decklight\'s model ' +
      "for the bot to play it at its best.</p>";
  } else if (side === "bot" && d.group !== "user") {
    b += forgeNote(d.forge || null);
  }
  if (meta) {
    const style = meta.style && !meta.arch.toLowerCase().includes(meta.style) ? meta.style : "";
    if (meta.arch) b += '<p class="prsub">' + esc([meta.arch, style].filter(Boolean).join(" · ")) + "</p>";
    if (meta.about) b += '<p class="dkabout">' + esc(meta.about) + "</p>";
    if (d.group !== "tuned") b += '<p class="prnote">' + esc(meta.origin) + "</p>";  // tuned: the note above says it
    const [m, s, t] = meta.counts;
    b += '<p class="prnote">' + m + " monsters · " + s + " spells · " + t + " traps" +
      (meta.mech.length ? " · " + esc(meta.mech.join(", ")) : "") + "</p>";
    b += cardListHTML("Main deck", meta.main, names, art) + cardListHTML("Extra deck", meta.extra, names, art);
  } else {
    b += '<p class="emptytx">No deck list available.</p>';
  }
  return b;
}

// st = {side, q, f: {group: [values]}, sort: "name"|"new", view: deckId|null}
export function deckSheetHTML(st, decks, index, cur, opts) {
  const names = (index && index.names) || {};
  const art = !!opts.art;
  const title = st.side === "you" ? "Your deck" : "decklight's deck";
  let head, body;
  if (st.view) {
    const d = decks.find((x) => x.id === st.view) || {id: st.view, name: st.view};
    const meta = index && index.decks[st.view];
    head = '<button class="iconbtn" data-act="deckback" aria-label="Back to all decks">' +
      '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12 4 6 10l6 6"/></svg></button>' +
      '<span class="mtitle">' + esc(d.name) + "</span>";
    body = '<div class="mbody dkview">' + detailHTML(d, meta, names, st.side, art, opts.forgeNote) + "</div>" +
      '<div class="dkfoot"><button class="primary" data-' + st.side + 'deck="' + esc(d.id) + '">' +
      (d.id === cur ? "Keep this deck" : "Choose this deck") + "</button></div>";
  } else {
    const rows = filterDecks(decks, index, st);
    head = '<span class="mtitle">' + esc(title) + "</span>";
    let chips = "";
    for (const [key, label, items, only] of DECK_FILTERS) {
      if (only && only !== st.side) continue;
      chips += '<div class="dkchips" role="group" aria-label="' + esc(label) + '">' + items.map(([v, t]) => {
        const on = (st.f[key] || []).includes(v);
        return '<button class="dkchip' + (on ? " on" : "") + '" aria-pressed="' + on + '" data-dfilter="' + key + ":" + v + '">' +
          esc(t) + "</button>";
      }).join("") + "</div>";
    }
    const anyF = Object.values(st.f).some((v) => v.length);
    body = '<div class="dktools">' +
      '<input class="dksearch" type="search" placeholder="Search decks or cards" aria-label="Search decks or cards" value="' +
      esc(st.q || "") + '"><div class="dkchiprow">' + chips + "</div>" +
      '<div class="dkmeta"><span role="status">' + rows.length + (rows.length === 1 ? " deck" : " decks") + "</span>" +
      (anyF ? '<button class="linkbtn" data-act="dclear">Clear filters</button>' : "") +
      '<span class="dksort" role="group" aria-label="Sort">' +
      ["name", "new"].map((k) => '<button class="dkchip small' + (st.sort === k ? " on" : "") + '" aria-pressed="' + (st.sort === k) +
        '" data-dsort="' + k + '">' + (k === "name" ? "A–Z" : "Newest") + "</button>").join("") + "</span></div></div>" +
      '<div class="mbody dklist">' +
      (st.side === "you" ? '<button class="deckrow paste" data-act="pasteydk"><span class="nm">Paste a deck list (.ydk)…</span></button>' : "") +
      (index === null ? '<p class="prnote">Loading deck details…</p>' : "") +
      (rows.length ? '<ul class="dkrows">' + rows.map((r) => rowHTML(r, st.side, cur, names, art)).join("") + "</ul>"
        : '<p class="emptytx">No deck matches. Try fewer filters.</p>') + "</div>";
  }
  return '<div id="dkscrim" data-key="dkscrim" data-act="closedecks"></div>' +
    '<div id="decksheet" data-key="decksheet" role="dialog" aria-modal="true" tabindex="-1" aria-label="' + esc(title) + '">' +
    '<div class="mhead">' + head + '<button class="iconbtn" data-act="closedecks" aria-label="Close">' +
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>' + body + "</div>";
}
