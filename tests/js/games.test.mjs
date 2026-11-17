import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.location = {search: ''};  // web/render/card.js reads the page's token at import
globalThis.queueMicrotask = (f) => f();
globalThis.requestAnimationFrame = (f) => f();
const {gamesFeature} = await import('../../web/features/games/index.js');

function feature() {
  const sent = [];
  const asked = [];
  const g = gamesFeature({send: (m) => sent.push(m), render: () => {}, mock: false, play: () => {}, ask: (t) => asked.push(t)});
  return {g, sent, asked};
}
const el = (attrs) => ({getAttribute: (k) => (k in attrs ? attrs[k] : null)});
const total = {games: 3, wins: 2, losses: 1, win_rate: 66.7};
const list = {filter: 'all', more: false, stats: {days: 30, total, first: [{key: 'first', ...total}], deck: [], opponent_deck: []},
  games: [{id: 'g1', when: new Date().toISOString(), your_deck: 'A', opponent_deck: 'B<i>', opponent: 'bot', you_went: 'first',
    result: 'you lost', turns: 7, kind: 'practice', lp: [[8000, 8000], [8000, 6000], [0, 6000]]}],
  samples: [{id: 'sample-1', sample: true, your_deck: 'A', opponent_deck: 'B', you_went: 'first', turns: 4, lp: [[8000, 8000], [0, 8000]]}]};
const page = {id: 'g1', your_deck: 'A', opponent_deck: 'B', result: 'you lost', turns: 2, you_went: 'first', final_lp: {you: 0, opponent: 8000},
  log: [{turn: 1, player: 'you', lp_at_start: [8000, 8000], events: ['You: x']}, {turn: 2, player: 'opponent', lp_at_start: [8000, 8000], events: []}],
  decisions: [
    {decision: 1, turn: 1, phase: 'main1', prompt: 'idle', text: 'Your move', you_picked: 'Summon X', picked_index: 0, options: [{index: 0, label: 'Summon X', code: 0}, {index: 3, label: 'End', code: 0}]},
    {decision: 2, turn: 1, phase: 'main1', prompt: 'place', text: 'Choose a zone', you_picked: 'Zone 1', picked_index: 0, options: []}]};

test('the list asks once, shows stats, rows (escaped, practice tagged) and samples', () => {
  const {g, sent} = feature();
  assert.match(g.panelHTML(), /Loading your games/);
  g.panelHTML();
  assert.equal(sent.filter((m) => m.t === 'ask_data').length, 1, 'one request while pending');
  assert.deepEqual(sent[0], {t: 'ask_data', key: 'games:list:all:30:', what: 'list', args: {filter: 'all', limit: 30, q: ''}});
  assert.ok(g.onMessage({t: 'ask_data', key: sent[0].key, result: list}));
  assert.ok(!g.onMessage({t: 'ask_data', key: 'other', result: {}}), 'not a games key: left for Ask');
  const h = g.panelHTML();
  assert.match(h, /2–1/);
  assert.match(h, /B&lt;i&gt;/);
  assert.match(h, /PRACTICE/);
  assert.match(h, /SAMPLE/);
  g.act('games:samples', el({}));
  assert.doesNotMatch(g.panelHTML(), />SAMPLE</, 'Hide sample games');
});

test('a game page folds zone choices, a check marks decisions, errors offer a retry', () => {
  const {g, sent, asked} = feature();
  g.act('games:open', el({'data-id': 'g1'}));
  g.panelHTML();
  const req = sent.find((m) => m.what === 'game');
  assert.deepEqual(req.args, {record: 'g1'});
  g.onMessage({t: 'ask_data', key: req.key, error: 'no saved game "g1"'});
  assert.match(g.panelHTML(), /Could not load this game: no saved game[\s\S]*games:reloadgame/);
  g.act('games:reloadgame', el({}));
  g.panelHTML();
  g.onMessage({t: 'ask_data', key: req.key, result: structuredClone(page)});
  g.act('games:turn', el({'data-turn': '1'}));
  let h = g.panelHTML();
  assert.ok(sent.some((m) => m.t === 'ask_frames' && m.key === 'games|t:g1:1' && m.turn === 1));
  // The turn plays as soon as it opens and waits for its plays: they arrive (one play), so its timer stops.
  g.onMessage({t: 'ask_frames', key: 'games|t:g1:1', player: 'you', frames: [{who: 'you', text: 'x', screen: {}}]});
  assert.match(h, /D1/);
  assert.doesNotMatch(h, /D2</, 'the zone choice is folded');
  assert.match(h, /Show 1 zone and position choice/);
  g.act('games:check', el({}));
  g.onMessage({t: 'ask_data', key: 'games:check:g1', progress: {done: 1, total: 2}});
  assert.match(g.panelHTML(), /Checking… <small>1 \/ 2<\/small>/);
  g.onMessage({t: 'ask_data', key: 'games:check:g1', result: {decisions: [
    {decision: 1, delta: -20, mark: 'Blunder', view: {}}, {decision: 2, delta: -9, mark: 'Mistake', view: {}}]}});
  h = g.panelHTML();
  assert.match(h, /−20 · BLUNDER/);
  assert.match(h, /D2</, 'a marked zone choice is listed');
  g.act('games:askgame', el({}));
  assert.match(asked[0], /my game g1/);
  assert.equal(g.back(), true);
  assert.match(g.panelHTML(), /Loading your games|gm-chips/);
});

test('search asks for the matches after a pause; Select ticks rows and Delete sends the ticked ids', async () => {
  const {g, sent} = feature();
  g.panelHTML();
  g.onMessage({t: 'ask_data', key: sent[0].key, result: {...list, total: 1}});
  g.search('bra');
  g.search('bran');
  await new Promise((r) => setTimeout(r, 300));
  g.panelHTML();
  const q = sent.filter((m) => m.what === 'list');
  assert.equal(q.length, 2, 'one request for the pause, not one per key');
  assert.deepEqual(q[1].args, {filter: 'all', limit: 30, q: 'bran'});
  g.onMessage({t: 'ask_data', key: q[1].key, result: {...list, total: 1, samples: []}});
  let h = g.panelHTML();
  assert.match(h, /1 of your games matches “bran”/);
  assert.doesNotMatch(h, /gm-stat/, 'no stat card while searching');
  g.act('games:select', el({}));
  h = g.panelHTML();
  assert.match(h, /0 selected/);
  assert.match(h, /data-act="games:delask" aria-label="Delete 0 games" disabled/);
  assert.doesNotMatch(h, /SAMPLE/, 'samples cannot be selected');
  g.act('games:tick', el({'data-id': 'g1'}));
  h = g.panelHTML();
  assert.match(h, /aria-pressed="true"/);
  assert.match(h, /aria-label="Delete 1 game"/);
  g.act('games:delask', el({}));
  assert.match(g.panelHTML(), /role="alertdialog"/);
  assert.ok(g.back(), 'back closes the dialog first');
  assert.doesNotMatch(g.panelHTML(), /alertdialog/);
  g.act('games:delask', el({}));
  g.act('games:delyes', el({}));
  const d = sent[sent.length - 1];
  assert.deepEqual([d.what, d.args], ['delete', {ids: ['g1']}]);
  assert.match(g.panelHTML(), /Deleting…/);
  g.onMessage({t: 'ask_data', key: d.key, result: {deleted: 1}});
  h = g.panelHTML();
  assert.match(h, /Deleted 1 game\./);
  assert.doesNotMatch(h, /selected/, 'Select mode ends');
  assert.equal(sent.filter((m) => m.what === 'list').length, 3, 'the list is asked again');
});

test('a failed delete says so and asks for the list again; Select all asks for every id', () => {
  const {g, sent} = feature();
  g.panelHTML();
  g.onMessage({t: 'ask_data', key: sent[0].key, result: {...list, total: 40}});
  g.act('games:select', el({}));
  assert.match(g.panelHTML(), /Select all 40/);
  g.act('games:selall', el({}));
  const ids = sent[sent.length - 1];
  assert.deepEqual([ids.what, ids.args], ['ids', {filter: 'all', q: ''}]);
  g.onMessage({t: 'ask_data', key: ids.key, result: {ids: Array.from({length: 40}, (_, i) => 'x' + i)}});
  assert.match(g.panelHTML(), /40 selected/);
  g.act('games:delask', el({}));
  g.act('games:delyes', el({}));
  const d = sent[sent.length - 1];
  g.onMessage({t: 'ask_data', key: d.key, error: '3 deleted; 1 not found in your games (the list was out of date): x9'});
  const h = g.panelHTML();
  assert.match(h, /The delete stopped: 3 deleted; 1 not found.*tick what is left/);
  assert.match(h, /0 selected/);
  assert.ok(g.back() && !/selected/.test(g.panelHTML()), 'back leaves Select mode');
});

test('Export: the game header opens the sheet; text and EDOPro replays come from the server, the sheet never sends while closed', () => {
  const {g, sent} = feature();
  g.act('games:open', el({'data-id': 'g1'}));
  g.panelHTML();
  const pk = sent.find((m) => m.what === 'game');
  g.onMessage({t: 'ask_data', key: pk.key, result: page});
  let h = g.panelHTML();
  assert.match(h, /data-act="games:xpgame"/);
  g.act('games:xpgame', el({}));
  h = g.panelHTML();
  assert.match(h, /role="dialog" aria-modal="true" aria-label="Export" data-close="games:xpclose"/, 'a grab-bar sheet that closes like the others');
  assert.match(h, /Whole game · 2 turns/);
  assert.match(h, /Make video/);
  g.act('games:xpas', el({'data-f': 'yrp'}));
  h = g.panelHTML();
  assert.match(h, /Open it in EDOPro/);
  g.act('games:xpmake', el({}));
  const x = sent[sent.length - 1];
  assert.deepEqual([x.what, x.args], ['export', {ids: ['g1'], as: 'yrp'}]);
  assert.match(g.panelHTML(), /Making the file/);
  g.onMessage({t: 'ask_data', key: x.key, result: {files: [{name: 'A vs B 2026-11-15.yrpX', mime: 'application/octet-stream', data: btoa('yrpx!')}]}});
  h = g.panelHTML();
  assert.match(h, /Replay ready/);
  assert.match(h, /A vs B 2026-11-15\.yrpX/);
  assert.match(h, /Share or save file/);
  // A range of turns as text.
  g.act('games:xpback', el({}));
  g.act('games:xpwhat', el({'data-v': 'range'}));
  g.act('games:xpas', el({'data-f': 'txt'}));
  g.act('games:xpmake', el({}));
  const t = sent[sent.length - 1];
  assert.deepEqual(t.args, {ids: ['g1'], as: 'txt', from_turn: 1, to_turn: 2});
  g.onMessage({t: 'ask_data', key: t.key, error: 'turns 1 to 2 have no plays'});
  assert.match(g.panelHTML(), /The export stopped/);
  assert.ok(g.back(), 'back closes the sheet');
  assert.doesNotMatch(g.panelHTML(), /gx-/);
  // A stale reply after closing changes nothing.
  g.onMessage({t: 'ask_data', key: t.key, result: {files: []}});
  assert.doesNotMatch(g.panelHTML(), /gx-/);
});

test('Export from Select mode offers the EDOPro replays and text only', () => {
  const {g, sent} = feature();
  g.panelHTML();
  g.onMessage({t: 'ask_data', key: sent[0].key, result: {...list, total: 1}});
  g.act('games:select', el({}));
  g.act('games:tick', el({'data-id': 'g1'}));
  g.act('games:xpsel', el({}));
  const h = g.panelHTML();
  assert.match(h, /Export 1 games|Export 1 game/);
  assert.match(h, /EDOPro replay/);
  assert.doesNotMatch(h, /data-f="mp4"/);
  g.act('games:xpmake', el({}));
  assert.deepEqual(sent[sent.length - 1].args, {ids: ['g1'], as: 'yrp'});
});
