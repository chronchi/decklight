import test from 'node:test';
import assert from 'node:assert/strict';

import { parseYdk, toYdk, toYdke } from '../../web/lib/ydk.js';

test('parses sections, comments, CRLF and leading zeros', () => {
  const text = '#created by decklight\r\n#main\r\n000123\r\n 456 \r\n# not a code\r\n#extra\r\n789\r\n!side\r\n42\r\n';
  const deck = parseYdk(text);
  assert.deepEqual(deck.main, [123, 456]);
  assert.deepEqual(deck.extra, [789]);
  assert.deepEqual(deck.side, [42]);
  assert.deepEqual(deck.unknown_lines, []);
  assert.equal(deck.name, null);
});

test('keeps lines that are not codes in unknown_lines', () => {
  const deck = parseYdk('#main\n123\nash blossom\n0\n-5\n#extra\n999\n');
  assert.deepEqual(deck.main, [123]);
  assert.deepEqual(deck.extra, [999]);
  assert.deepEqual(deck.unknown_lines, ['ash blossom', '0', '-5']);
});

test('reads an optional name comment', () => {
  assert.equal(parseYdk('#name My Snake-Eye\n#main\n1\n').name, 'My Snake-Eye');
});

test('toYdk round-trips through parseYdk', () => {
  const deck = { main: [1, 2, 3], extra: [100, 101], side: [200], name: null, unknown_lines: [] };
  const text = toYdk(deck);
  assert.equal(text, '#created by decklight\n#main\n1\n2\n3\n#extra\n100\n101\n!side\n200\n');
  assert.deepEqual(parseYdk(text), deck);
});

test('toYdk accepts a custom createdBy and empty sections', () => {
  const text = toYdk({ main: [], extra: [], side: [] }, { createdBy: 'someone' });
  assert.ok(text.startsWith('#created by someone\n#main\n\n#extra\n\n!side\n'));
  assert.deepEqual(parseYdk(text), { main: [], extra: [], side: [], name: null, unknown_lines: [] });
});

test('toYdke round-trips through parseYdk', () => {
  const deck = { main: [1, 0x01020304, 9999999], extra: [100], side: [], name: null, unknown_lines: [] };
  const url = toYdke(deck);
  assert.ok(url.startsWith('ydke://'));
  assert.ok(url.endsWith('!'));
  assert.deepEqual(parseYdk(url), deck);
});

test('an empty deck round-trips through ydke', () => {
  const deck = { main: [], extra: [], side: [], name: null, unknown_lines: [] };
  assert.deepEqual(parseYdk(toYdke(deck)), deck);
});
