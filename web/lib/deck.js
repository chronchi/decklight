// Deck model and legality. Pure ES module: no DOM.

export const FORMATS = {
  standard: { name: 'Standard', main: [40, 60], extra: [0, 15], side: [0, 15], copies: 3, banlist: null },
};

const TYPE_MONSTER = 0x1;
const TYPE_FUSION = 0x40;
const TYPE_SYNCHRO = 0x2000;
const TYPE_XYZ = 0x800000;
const TYPE_LINK = 0x4000000;
const EXTRA_BITS = TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK;

/** Fusion, Synchro, Xyz and Link monsters live in the Extra Deck. */
export function isExtraDeckType(typeBits) {
  return (typeBits & TYPE_MONSTER) !== 0 && (typeBits & EXTRA_BITS) !== 0;
}

/** `{main, extra, side}` card counts. */
export function counts(deck) {
  return {
    main: (deck?.main ?? []).length,
    extra: (deck?.extra ?? []).length,
    side: (deck?.side ?? []).length,
  };
}

function resolveFormat(format) {
  if (typeof format === 'string') {
    const found = FORMATS[format];
    if (!found) throw new Error(`unknown format ${format}`);
    return found;
  }
  return format ?? FORMATS.standard;
}

function countProblem(label, n, range) {
  const [min, max] = range;
  if (n < min) return { kind: `${label.toLowerCase()}_count`, text: `${label} ${n}/${min} · ${min - n} more` };
  if (n > max) return { kind: `${label.toLowerCase()}_count`, text: `${label} ${n}/${max} · ${n - max} too many` };
  return null;
}

/**
 * Validate a deck against a format (a key of FORMATS or a format object).
 * `nameOf(code) -> string` and `aliasOf(code) -> code` let alt arts share a copy limit.
 */
export function validate(deck, format, { nameOf = (code) => String(code), aliasOf = (code) => code } = {}) {
  const fmt = resolveFormat(format);
  const n = counts(deck);
  const problems = [];
  for (const [label, key] of [
    ['Main', 'main'],
    ['Extra', 'extra'],
    ['Side', 'side'],
  ]) {
    const p = countProblem(label, n[key], fmt[key]);
    if (p) problems.push(p);
  }

  const groups = new Map();
  for (const key of ['main', 'extra', 'side']) {
    for (const code of deck?.[key] ?? []) {
      const alias = aliasOf(code);
      const group = groups.get(alias);
      if (group) {
        group.count += 1;
        if (code < group.minCode) group.minCode = code;
      } else {
        groups.set(alias, { count: 1, minCode: code });
      }
    }
  }
  const ordered = [...groups.values()].sort((a, b) => a.minCode - b.minCode);
  for (const group of ordered) {
    if (group.count > fmt.copies) {
      problems.push({
        kind: 'copies',
        text: `${fmt.copies} copies max: ${nameOf(group.minCode)} ×${group.count}`,
        codes: [group.minCode],
      });
    }
  }

  const banlist = fmt.banlist;
  if (banlist) {
    for (const code of banlist.forbidden ?? []) {
      if ((groups.get(aliasOf(code))?.count ?? 0) > 0) {
        problems.push({ kind: 'banlist', text: `Forbidden: ${nameOf(code)}`, codes: [code] });
      }
    }
    for (const code of banlist.limited ?? []) {
      const used = groups.get(aliasOf(code))?.count ?? 0;
      if (used > 1) {
        problems.push({ kind: 'banlist', text: `Limited (1): ${nameOf(code)} ×${used}`, codes: [code] });
      }
    }
    for (const code of banlist.semi ?? []) {
      const used = groups.get(aliasOf(code))?.count ?? 0;
      if (used > 2) {
        problems.push({ kind: 'banlist', text: `Semi-limited (2): ${nameOf(code)} ×${used}`, codes: [code] });
      }
    }
  }

  return { ok: problems.length === 0, problems };
}
