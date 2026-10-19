// Log drawer rows: actor colour, card thumbnail, plain-words text. Newest last.
import {thumbHTML, esc} from "./card.js";

// A negated activation keeps a mark: a ⊘ shape plus words (not colour alone).
export function negTag(by) {
  return ' <span class="negtag"><i aria-hidden="true"></i>negated' +
    (typeof by === "string" && by.startsWith("by ") ? " " + esc(by) : "") + "</span>";
}

// The actor is a shape (your dot, the bot's square seal) and a spoken name, not colour alone.
export function logHTML(events, mock) {
  if (!events || !events.length) return '<p class="emptytx">Nothing yet.</p>';
  return events.map((e) => {
    const actor = e.actor === "you" || e.actor === "bot" ? e.actor : "system";
    const said = actor === "you" ? "You: " : actor === "bot" ? "decklight: " : "";
    return '<div class="ev ' + actor + '"' + (e.card && e.card.uid ? ' data-detailuid="' + esc(e.card.uid) + '" role="button" tabindex="0"' : "") + ">" +
      '<span class="am" aria-hidden="true"></span>' +
      (e.card ? thumbHTML(e.card, mock) : '<span class="nothumb"></span>') + (said ? '<span class="sr-only">' + said + "</span>" : "") +
      '<span class="et">' + esc(e.text || "") + (e.neg ? negTag(e.neg) : "") + "</span></div>";
  }).join("");
}
