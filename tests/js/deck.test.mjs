import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { FORMATS, counts, isExtraDeckType, validate } from '../../web/lib/deck.js';

const vectors = JSON.parse(
  readFileSync(new URL('../data/deck_legality_cases.json', import.meta.url), 'utf8'),
);

test('formats and extra-deck types', () => {
  assert.deepEqual(FORMATS.standard.main, [40, 60]);
  assert.equal(isExtraDeckType(0x40 | 0x1), true);
  assert.equal(isExtraDeckType(0x2000 | 0x1), true);
  assert.equal(isExtraDeckType(0x800000 | 0x1), true);
  assert.equal(isExtraDeckType(0x4000000 | 0x1), true);
  assert.equal(isExtraDeckType(0x20 | 0x1), false);
  assert.equal(isExtraDeckType(0x2), false);
});

test('counts', () => {
  assert.deepEqual(counts({ main: [1, 1, 2], extra: [3], side: [] }), { main: 3, extra: 1, side: 0 });
});

for (const vector of vectors) {
  test(`legality vector: ${vector.name}`, () => {
    const names = vector.names ?? {};
    const aliases = vector.aliases ?? {};
    const result = validate(vector.deck, vector.format, {
      nameOf: (code) => names[String(code)] ?? String(code),
      aliasOf: (code) => aliases[String(code)] ?? code,
    });
    assert.equal(result.ok, vector.expect.ok);
    assert.deepEqual(
      result.problems.map((p) => p.kind),
      vector.expect.kinds,
    );
    assert.deepEqual(
      result.problems.map((p) => p.text),
      vector.expect.texts,
    );
  });
}
