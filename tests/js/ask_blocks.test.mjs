import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.location = {search: ''};  // web/render/card.js reads the page's token at import
const {blockSpecs, blockHTML, framesNeeded} = await import('../../web/features/ask/blocks.js');

const call = (name, result, ok = true) => ({name, result: JSON.stringify(result), done: true, ok});
const summary = {id: 'g1', result: 'you lost', your_deck: 'A', opponent_deck: 'B', turns: 40, you_went: 'first',
  final_lp: {you: 0, opponent: 6200}, log: Array.from({length: 40}, (_, i) => ({turn: i + 1, lp_at_start: [8000 - i * 200, 8000]}))};

test('one block per game, turn and decision, in call order, at most three', () => {
  const calls = [call('replay', summary), call('replay', {id: 'g1', turn: 2, events: []}), call('replay', summary),
    call('bot_view', {id: 'g1', decision: 4, turn: 2, bot_top: []}), call('replay', {id: 'g1', turn: 3, events: []}),
    call('replay', {error: 'x'}, false)];
  const bs = blockSpecs(calls);
  assert.deepEqual(bs.map((b) => b.key), ['g:g1', 't:g1:2', 'b:g1:4']);
  assert.ok(bs[0].hot.has(2) && bs[0].hot.has(3), 'the game card marks the turns the answer replayed');
  assert.deepEqual(framesNeeded(bs[1]), {key: 't:g1:2', record: 'g1', turn: 2});
  assert.deepEqual(framesNeeded(bs[2]), {key: 'd:g1:4', record: 'g1', decision: 4});
  assert.equal(blockSpecs([call('replay', {games: []})]).length, 0, 'the game list draws no block');
});

test('the game card draws a 40-turn chart without NaN', () => {
  const h = blockHTML(blockSpecs([call('replay', summary)])[0], () => null, () => ({}));
  assert.ok(h.includes('LOST') && h.includes('A vs B'));
  assert.ok(!h.includes('NaN'));
});

test("the bot's view grades your pick like the review", () => {
  const r = {id: 'g1', decision: 4, turn: 2, phase: 'main1', win_chance_now: 44,
    bot_top: [{option: 'Battle', bot_prob: 52, win_chance: 47}, {option: 'Set', bot_prob: 9, win_chance: 41}],
    you_picked: {option: 'Destroy', bot_prob: 3, win_chance: 29, code: 123}};
  const b = blockSpecs([call('bot_view', r)])[0];
  const h = blockHTML(b, () => ({pending: true}), () => ({}));
  assert.ok(h.includes("BOT'S PICK") && h.includes('YOUR PICK'));
  assert.ok(h.includes('−18 PTS · BLUNDER'), 'an 18-point loss is a blunder (review: < -15)');
  assert.ok(h.includes('Replaying the game'), 'the board waits for its frames');
  const err = blockHTML(b, () => ({error: 'no saved game'}), () => ({}));
  assert.ok(err.includes('Try again') && err.includes('no saved game'), 'a failed board says why and offers Try again');
});

test('the sequence shows one play at a time as a still of the play screen that opens Watch', () => {
  const screen = (t) => ({t: 'state', view: {turn: 2}, caption: {text: t}});
  const fr = {id: 'g1', player: 'you', frames: [{who: '', text: 'Turn 2', choice: 7, screen: screen('')},
    {who: 'you', text: 'Tributes X', card: 5, choice: 9, screen: screen('Tributes X')}]};
  const b = blockSpecs([call('replay', {id: 'g1', turn: 2, events: []})])[0];
  const h = blockHTML(b, () => fr, () => ({i: 1}));
  assert.ok(h.includes('2 / 2') && h.includes('Tributes X'));
  // No document in node: the still waits (the app draws it with lib/player.js).
  assert.ok(h.includes('Drawing the board') && h.includes('Board, turn 2, play 2 of 2: Tributes X'), h);
  assert.ok(!h.includes('ab-mb'), 'the small board is gone');
});

test("two options with the same label: your pick is the one with your index", () => {
  const r = {id: 'g1', decision: 5, turn: 3, phase: 'main1', win_chance_now: 50,
    bot_top: [{option: 'Summon X', index: 0, bot_prob: 60, win_chance: 55}, {option: 'Summon X', index: 1, bot_prob: 30, win_chance: 40}],
    you_picked: {option: 'Summon X', index: 1, bot_prob: 30, win_chance: 40}};
  const h = blockHTML(blockSpecs([call('bot_view', r)])[0], () => ({pending: true}), () => ({}));
  const rows = h.split(/<div class="ab-opt[ "]/).slice(1);
  assert.equal(rows.length, 2, 'the pick is not added twice');
  assert.ok(!rows[0].includes('YOUR PICK') && rows[1].includes('YOUR PICK'));
  assert.ok(rows[1].includes('−15 PTS · MISTAKE'));
});
