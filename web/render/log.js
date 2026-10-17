// Log drawer rows: actor colour, card thumbnail, plain-words text. Newest last.
import {thumbHTML, esc} from "./card.js";

export function logHTML(events, mock) {
  if (!events || !events.length) return '<p class="emptytx">Nothing yet.</p>';
  return events.map((e) => {
    const actor = e.actor === "you" || e.actor === "bot" ? e.actor : "system";
    return '<div class="ev ' + actor + '"' + (e.card && e.card.uid ? ' data-detailuid="' + esc(e.card.uid) + '"' : "") + ">" +
      (e.card ? thumbHTML(e.card, mock) : '<span class="nothumb"></span>') +
      '<span class="et">' + esc(e.text || "") + "</span></div>";
  }).join("");
}
