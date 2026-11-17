// Mock backend for the deck builder (mock mode screenshots, no server): answers the deck_* / card_get /
// cards_search messages of docs/design/api/cards-and-decks.md from web/mock/deckbuilder.json (scripts/app/build_mock_cards.py).
// Search here is a plain substring match; the real ranking lives in the Rust server.
import {parseYdk, toYdk, toYdke} from "../../lib/ydk.js";
import {FORMATS, validate, counts} from "../../lib/deck.js";

export function mockBackend(reply) {
  let data = null;
  const queue = [];
  fetch("mock/deckbuilder.json").then((r) => r.json()).then((d) => {
    data = d;
    for (const m of queue.splice(0)) answer(m);
  });
  const later = (m) => setTimeout(() => reply(m), 40);
  const byCode = () => new Map(data.cards.map((c) => [c.code, c]));
  function legal(d) {
    const cards = byCode();
    const v = validate(d, FORMATS.standard, {nameOf: (c) => (cards.get(c) || {}).name || String(c)});
    return {ok: v.ok, problems: v.problems.map((p) => p.text)};
  }
  function answer(m) {
    const cards = byCode();
    const deck = (id) => data.decks.find((d) => d.id === id);
    if (m.t === "deck_list") {
      const decks = [...data.decks].sort((a, b) => (a.updated < b.updated ? 1 : -1)).map((d) => ({
        id: d.id, name: d.name, updated: d.updated, cover: d.cover || d.main[0] || 0, counts: counts(d), legal: legal(d)}));
      later({t: "user_decks", req: m.req, decks});
    } else if (m.t === "deck_get") {
      const d = deck(m.id);
      if (!d) { later({t: "error", req: m.req, error: "unknown deck " + m.id}); return; }
      later({t: "user_deck", req: m.req, deck: Object.assign({cover: d.main[0] || 0}, d),
        legal: validate(d, FORMATS.standard, {}), unknown: []});
    } else if (m.t === "deck_save") {
      let d = m.id && deck(m.id);
      if (!d) { d = {id: (m.name || "deck").toLowerCase().replace(/\W+/g, "-") + "-" + Date.now().toString(16).slice(-6)}; data.decks.push(d); }
      Object.assign(d, {name: m.name, main: m.main, extra: m.extra, side: m.side, cover: m.cover || 0,
        updated: new Date().toISOString().slice(0, 19)});
      later({t: "deck_saved", req: m.req, id: d.id, updated: d.updated, legal: legal(d)});
    } else if (m.t === "deck_delete") {
      data.decks = data.decks.filter((d) => d.id !== m.id);
      later({t: "deck_deleted", req: m.req, id: m.id});
    } else if (m.t === "deck_import") {
      const p = parseYdk(m.ydk || "");
      const d = {id: "imported-" + Date.now().toString(16).slice(-6), name: m.name || "Imported deck",
        main: p.main, extra: p.extra, side: p.side, updated: new Date().toISOString().slice(0, 19)};
      data.decks.push(d);
      const unknown = [...p.main, ...p.extra, ...p.side].filter((c) => !cards.has(c));
      later({t: "deck_saved", req: m.req, id: d.id, updated: d.updated, legal: legal(d), unknown: [...new Set(unknown)],
        unknown_lines: p.unknown_lines});
    } else if (m.t === "deck_export") {
      const d = deck(m.id);
      later({t: "deck_ydk", req: m.req, id: m.id, name: d.name, ydk: toYdk(d), ydke: toYdke(d)});
    } else if (m.t === "card_get") {
      const c = cards.get(m.code);
      if (!c) { later({t: "error", req: m.req, error: "unknown card " + m.code}); return; }
      later(Object.assign({t: "card", req: m.req, archetypes: [], alias: 0}, c));
    } else if (m.t === "cards_search") {
      // Describe: any word of 4+ letters, answered after 4 s so the busy state and Stop can be seen.
      const words = m.mode === "describe" ? (m.q || "").toLowerCase().split(/\W+/).filter((w) => w.length > 3) : null;
      const q = (m.q || "").toLowerCase();
      const f = m.filters || {};
      // The mock rows carry `type` (cdb bits) but no attribute/race/level/atk/def fields: those filters are skipped
      // unless a future mock JSON adds the field, so the mock never hides every card for a filter it cannot apply.
      const has = (key) => data.cards.some((c) => c[key] !== undefined);
      const list = (v) => (Array.isArray(v) ? v : [v]);
      const anyOf = (v, wanted) => !wanted || list(wanted).some((w) => w === v);
      const T = {MONSTER: 0x1, SPELL: 0x2, TRAP: 0x4, NORMAL: 0x10, EFFECT: 0x20, FUSION: 0x40, RITUAL: 0x80,
        SYNCHRO: 0x2000, TOKEN: 0x4000, QUICKPLAY: 0x10000, CONTINUOUS: 0x20000, EQUIP: 0x40000, FIELD: 0x80000,
        COUNTER: 0x100000, XYZ: 0x800000, PENDULUM: 0x1000000, LINK: 0x4000000};
      const FRAME_BITS = {normal: T.NORMAL, effect: T.EFFECT, ritual: T.RITUAL, fusion: T.FUSION, synchro: T.SYNCHRO,
        xyz: T.XYZ, pendulum: T.PENDULUM, link: T.LINK, token: T.TOKEN};
      const ST_BITS = {quickplay: T.QUICKPLAY, continuous: T.CONTINUOUS, equip: T.EQUIP, field: T.FIELD,
        ritual: T.RITUAL, counter: T.COUNTER};
      const ST_OTHER = Object.values(ST_BITS).reduce((a, b) => a | b, 0);
      const stMatches = (c) => {
        if (!(c.type & (T.SPELL | T.TRAP))) return false;
        return list(f.st_type).some((k) => (k === "normal" ? !(c.type & ST_OTHER) : (c.type & (ST_BITS[k] || 0)) !== 0));
      };
      const inRange = (v, r) => !Array.isArray(r) || (v >= r[0] && v <= r[1]);
      const passes = (c) => anyOf(c.kind, f.kind) && (!f.extra_only || c.extra) &&
        (!f.frame || (c.type & list(f.frame).reduce((a, k) => a | (FRAME_BITS[k] || 0), 0)) !== 0) &&
        (!f.st_type || stMatches(c)) &&
        (!f.attribute || !has("attribute") || anyOf(c.attribute, f.attribute)) &&
        (!f.race || !has("race") || anyOf(c.race, f.race)) &&
        (!f.level || !has("level") || inRange(c.level, f.level)) &&
        (!f.atk || !has("atk") || (c.kind === "monster" && c.atk >= 0 && inRange(c.atk, f.atk))) &&
        (!f.def || !has("def") || (c.kind === "monster" && !(c.type & T.LINK) && c.def >= 0 && inRange(c.def, f.def)));
      const rows = data.cards.filter((c) => passes(c) &&
        (words ? words.some((w) => c.text.toLowerCase().includes(w))
          : !q || c.name.toLowerCase().includes(q) || c.text.toLowerCase().includes(q)))
        .map((c) => {
          const byName = !q || c.name.toLowerCase().includes(q);
          const sent = byName ? "" : (c.text.split(/(?<=\.)\s+/).find((s) => s.toLowerCase().includes(q)) || "").slice(0, 160);
          return {code: c.code, name: c.name, kind: c.kind, line: c.line, extra: c.extra,
            match: {by: byName ? "name" : "text", snippet: sent}};
        })
        .sort((a, b) => (a.match.by === b.match.by ? a.name.localeCompare(b.name) : a.match.by === "name" ? -1 : 1));
      const out = {t: "cards", req: m.req, q: m.q, total: rows.length, rows: rows.slice(0, m.limit || 40)};
      if (words) setTimeout(() => reply(out), 4000); else later(out);
    }
  }
  return (m) => (data ? answer(m) : queue.push(m));
}
