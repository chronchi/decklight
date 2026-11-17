// The composer's + (mocks ask-attach, ask-attach-game, ask-attached; approved 2026-11-14): a deck, a game, a card or a
// photo of a card goes with the question. The server reads them for the model (rust/decklight-server/src/ask/attach.rs).
import {esc} from "../../render/card.js";
import {I, DMARK, errHTML} from "./view.js";

const ICO = (d) => '<svg class="ak-ico" viewBox="0 0 24 24" aria-hidden="true">' + d + "</svg>";
export const IA = {
  deck: ICO('<rect x="5" y="4" width="11" height="15" rx="1.5"/><path d="M9 4.5V3.8A1 1 0 0 1 10 3h8a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-2"/>'),
  game: ICO('<path d="M4 17.5 9 12l3.5 3L20 7"/><path d="M15.5 7H20v4.5"/>'),
  card: ICO('<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5"/>'),
  photo: ICO('<path d="M4 8.5A1.5 1.5 0 0 1 5.5 7H8l1.5-2h5L16 7h2.5A1.5 1.5 0 0 1 20 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z"/><circle cx="12" cy="13" r="3.2"/>'),
  pick: ICO('<rect x="4" y="5" width="16" height="14" rx="1.5"/><path d="m4 16 5-5 4 4 2.5-2.5L20 17"/><circle cx="15.5" cy="9.5" r="1.5"/>'),
  x: ICO('<path d="M7 7l10 10M17 7 7 17"/>'),
  chev: ICO('<path d="m9.5 6 6 6-6 6"/>'),
};
export const MAX = 5;
const KIND = {deck: "DECK", game: "GAME", card: "CARD", photo: "PHOTO"};

// A chip: in the composer with an x (`i` = its index), in a sent question without.
export function chipHTML(a, i, thumb) {
  const img = thumb ? '<i style="background-image:url(' + esc(thumb) + ')"></i>' : "";
  const name = a.kind === "photo" ? "" : esc(a.label || "");
  const x = i == null ? "" : '<button class="x" data-act="ask:unattach" data-i="' + i + '" aria-label="Remove ' +
    esc(a.kind === "photo" ? "the photo" : a.label || a.kind) + '">' + IA.x + "</button>";
  return '<span class="ak-chip">' + img + '<b class="k">' + KIND[a.kind] + "</b>" + name + x + "</span>";
}

const row = (act, ico, n, d, off) => '<button class="ak-arow" data-act="' + act + '"' + (off ? " disabled" : "") + '><span class="ai">' + ico +
  '</span><span><span class="an">' + n + '</span><span class="ad">' + d + "</span></span>" + IA.chev + "</button>";
const head = (label, title) => '<div class="ak-scrim" data-act="ask:closesheet"></div><div class="ak-bsheet" role="dialog" aria-modal="true" aria-label="' +
  label + '" data-close="ask:closesheet"><div class="grab"></div><h2>' + title + "</h2>";
const back = '<button class="ak-link ak-aback" data-act="ask:attach">' + I.back + "All attachments</button>";
const wait = (t) => '<div class="ak-think">' + DMARK + '<span class="w">' + t + "</span></div>";
const search = (id, ph, q) => '<label class="ak-search ak-asearch"><span class="sr-only">' + ph + '</span><input id="' + id + '" type="search" placeholder="' +
  ph + '" value="' + esc(q || "") + '" autocomplete="off" spellcheck="false"></label>';

// sheet = {attach: "menu" | "deck" | "game" | "card" | "photo", rows, err, q}; `full` when the question has MAX already.
export function attachSheetHTML(sheet, full, hasPhoto) {
  const s = sheet;
  if (s.attach === "menu") {
    return head("Attach", "Attach to your question") +
      '<p class="ak-help">The model reads what you attach first, so it does not have to search for it.' + (full ? " A question takes up to " + MAX + ": remove one to add another." : "") + "</p>" +
      row("ask:attach-deck", IA.deck, "A deck", "One of your saved decks: the list, counts and legality", full) +
      row("ask:attach-game", IA.game, "A game", "A saved game or a sample: the summary, turns and your decisions", full) +
      row("ask:attach-card", IA.card, "A card", "Search by name or effect", full) +
      row("ask:attach-photo", IA.photo, "A photo of a card", hasPhoto ? "One photo per question: remove it to add another" : "Take or pick a photo, any language: the model finds the card", full || hasPhoto) +
      '<p class="ak-help">Photos need a model that accepts images.</p></div>';
  }
  let h = "";
  if (s.attach === "photo") {
    return head("Attach a photo", "A photo of a card") + back +
      row("ask:photo-take", IA.photo, "Take a photo", "Opens the camera: fill the frame with the card") +
      row("ask:photo-pick", IA.pick, "Choose from your photos", "A screenshot works too") +
      (s.err ? errHTML({message: s.err, action: "none"}) : s.busy ? wait("Reading the photo…") : "") + "</div>";
  }
  if (s.attach === "deck") {
    h = head("Attach a deck", "Attach a deck") + back + '<div class="ak-models">';
    if (s.err) h += errHTML({message: s.err, action: "none"});
    else if (!s.rows) h += wait("Loading your decks…");
    else if (!s.rows.length) h += '<p class="ak-help">No saved decks yet: make one in Decks, then attach it here.</p>';
    for (const d of s.rows || []) {
      const c = d.counts || {};
      h += '<button class="ak-mrow" data-act="ask:attach-pick" data-kind="deck" data-id="' + esc(d.id) + '" data-label="' + esc(d.name || "Deck") +
        '"><span><span class="mn">' + esc(d.name || "Deck") + '</span><span class="ml">' + (c.main || 0) + " main · " + (c.extra || 0) + " extra · " +
        (c.side || 0) + " side" + (d.legal && d.legal !== "legal" ? " · " + esc(d.legal) : "") + "</span></span></button>";
    }
    return h + "</div></div>";
  }
  if (s.attach === "game") {
    h = head("Attach a game", "Attach a game") + back + search("akattq", "Search by deck or opponent", s.q) + '<div class="ak-models">';
    if (s.err) h += errHTML({message: s.err, action: "none"});
    else if (!s.rows) h += wait("Loading your games…");
    else if (!s.rows.length) h += '<p class="ak-help">' + (s.q ? "No game matches “" + esc(s.q) + "”." : "No saved games yet: play one, or attach a sample.") + "</p>";
    for (const g of s.rows || []) {
      const res = g.sample ? "SAMPLE" : g.result === "you won" ? "WON" : g.result === "you lost" ? "LOST" : "—";
      const vs = (g.your_deck || "Your deck") + " vs " + (g.opponent_deck || "opponent");
      const meta = [g.sample ? "Bot vs bot" : g.opponent === "remote" ? "Multiplayer" : "Bot", g.turns ? g.turns + " turns" : "",
        g.you_went === "first" ? "went 1st" : g.you_went === "second" ? "went 2nd" : ""].filter(Boolean).join(" · ");
      h += '<button class="ak-mrow" data-act="ask:attach-pick" data-kind="game" data-id="' + esc(g.id) + '" data-label="' + esc(vs) +
        '"><span class="ak-res">' + res + '</span><span><span class="mn">' + esc(vs) + '</span><span class="ml">' + esc(meta) + "</span></span></button>";
    }
    return h + "</div></div>";
  }
  h = head("Attach a card", "Attach a card") + back + search("akattq", "Search by name or effect", s.q) + '<div class="ak-models">';
  if (s.err) h += errHTML({message: s.err, action: "none"});
  else if (s.q && !s.rows) h += wait("Searching the cards…");
  else if (s.q && !s.rows.length) h += '<p class="ak-help">No card matches “' + esc(s.q) + "”.</p>";
  for (const r of s.rows || []) {
    h += '<button class="ak-mrow" data-act="ask:attach-pick" data-kind="card" data-code="' + r.code + '" data-label="' + esc(r.name) +
      '"><span><span class="mn">' + esc(r.name) + '</span><span class="ml">' + esc(r.line || "") + "</span></span></button>";
  }
  return h + "</div></div>";
}

// A picked photo as the server takes it: a JPEG at most 1024 px on its long side, and a small thumbnail for the chip.
export async function readPhoto(file) {
  if (!file) throw new Error("no photo was picked");
  if (file.type && !file.type.startsWith("image/")) throw new Error("that file is not a photo: pick an image");
  let img;
  try { img = await createImageBitmap(file, {imageOrientation: "from-image"}); }
  catch (e) { throw new Error("this photo could not be opened (" + (e.message || e) + "): pick another one or take a new one"); }
  const draw = (max, q) => {
    const k = Math.min(1, max / Math.max(img.width, img.height));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(img.width * k));
    c.height = Math.max(1, Math.round(img.height * k));
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", q);
  };
  const out = {kind: "photo", data: draw(1024, 0.85), thumb: draw(96, 0.7)};
  if (img.close) img.close();
  if (!out.data.startsWith("data:image/jpeg")) throw new Error("this phone could not convert the photo: pick another one");
  return out;
}
