// .ydk and ydke:// parsing and serialization. Pure ES module: no DOM, no Node-only globals beyond atob/btoa.

const SECTION_KEYS = ['main', 'extra', 'side'];

function b64decode(text) {
  let s = String(text).replace(/-/g, '+').replace(/_/g, '/');
  while (s.length % 4) s += '=';
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64encode(bytes) {
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

function decodeU32(text) {
  const bytes = b64decode(text);
  const out = [];
  for (let i = 0; i + 3 < bytes.length; i += 4) {
    out.push((bytes[i] | (bytes[i + 1] << 8) | (bytes[i + 2] << 16) | (bytes[i + 3] << 24)) >>> 0);
  }
  return out;
}

function encodeU32(codes) {
  const bytes = new Uint8Array(codes.length * 4);
  codes.forEach((code, i) => {
    const c = code >>> 0;
    bytes[i * 4] = c & 0xff;
    bytes[i * 4 + 1] = (c >>> 8) & 0xff;
    bytes[i * 4 + 2] = (c >>> 16) & 0xff;
    bytes[i * 4 + 3] = (c >>> 24) & 0xff;
  });
  return b64encode(bytes);
}

function emptyDeck() {
  return { main: [], extra: [], side: [], name: null, unknown_lines: [] };
}

function parseYdke(text) {
  const deck = emptyDeck();
  const body = text.slice('ydke://'.length);
  const parts = body.split('!');
  for (let i = 0; i < SECTION_KEYS.length; i++) {
    const part = (parts[i] || '').trim();
    if (!part) continue;
    try {
      deck[SECTION_KEYS[i]] = decodeU32(part);
    } catch {
      deck.unknown_lines.push(part);
    }
  }
  for (const part of parts.slice(SECTION_KEYS.length)) {
    if (part.trim()) deck.unknown_lines.push(part.trim());
  }
  return deck;
}

/** Parse .ydk text (or a `ydke://` payload) into `{main, extra, side, name, unknown_lines}`. */
export function parseYdk(text) {
  const src = String(text ?? '');
  if (src.trim().startsWith('ydke://')) return parseYdke(src.trim());
  const deck = emptyDeck();
  let current = 'main';
  for (const raw of src.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (line === '#main') {
      current = 'main';
      continue;
    }
    if (line === '#extra') {
      current = 'extra';
      continue;
    }
    if (line === '!side') {
      current = 'side';
      continue;
    }
    if (line.startsWith('#') || line.startsWith('!')) {
      const named = /^#\s*name\s+(.+)$/i.exec(line);
      if (named) deck.name = named[1].trim();
      continue;
    }
    if (/^\d+$/.test(line)) {
      const code = Number(line);
      if (Number.isSafeInteger(code) && code > 0 && code <= 0xffffffff) {
        deck[current].push(code);
        continue;
      }
    }
    deck.unknown_lines.push(line);
  }
  return deck;
}

/** Serialize a deck to .ydk text. */
export function toYdk(deck, { createdBy = 'decklight' } = {}) {
  const section = (key) => (deck?.[key] ?? []).map((code) => String(code)).join('\n');
  return (
    `#created by ${createdBy}\n#main\n${section('main')}\n#extra\n${section('extra')}\n!side\n${section('side')}\n`
  );
}

/** Serialize a deck to a `ydke://` URL. */
export function toYdke(deck) {
  return (
    'ydke://' +
    SECTION_KEYS.map((key) => encodeU32(deck?.[key] ?? [])).join('!') +
    '!'
  );
}
