import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, readdirSync} from 'node:fs';

// Every stylesheet of the app (vendored code excluded).
const WEB = new URL('../../web/', import.meta.url);
const sheets = (dir) => readdirSync(new URL(dir, WEB), {withFileTypes: true}).flatMap((e) =>
  e.isDirectory() ? (e.name === 'vendor' ? [] : sheets(dir + e.name + '/')) : e.name.endsWith('.css') ? [[dir + e.name, readFileSync(new URL(dir + e.name, WEB), 'utf8')]] : []);
const css = sheets('');
const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '');

// A keyframe that moves `transform` replaces the element's own transform while it runs: the game menu, centred with
// translateX(-50%), came in from the bottom right (user 2026-11-16). Opening motion moves `translate` instead.
test('no animated element loses its own transform to its keyframes', () => {
  const moving = new Set();
  for (const [, s] of css) for (const m of strip(s).matchAll(/@keyframes\s+([\w-]+)\s*\{((?:[^{}]*\{[^{}]*\})*)[^{}]*\}/g)) {
    if (/(^|[\s;{])transform\s*:/.test(m[2])) moving.add(m[1]);
  }
  const bad = [];
  for (const [f, s] of css) for (const m of strip(s).replace(/@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g, '').matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const anim = /(?:^|;)\s*animation\s*:\s*([\w-]+)/.exec(m[2]);
    if (anim && moving.has(anim[1]) && /(?:^|;)\s*transform\s*:/.test(m[2])) bad.push(`${f}: ${m[1].trim()} (${anim[1]})`);
  }
  assert.deepEqual(bad, []);
});

// Every bottom sheet opens the same way: it rises into place (`up`), over a scrim that fades in.
test('every bottom sheet rises into place', () => {
  const all = css.map(([, s]) => strip(s)).join('\n');
  for (const sel of ['#sheet {', '#gamemenu, #primersheet {', '.dbsheet {', '#decksheet {', '.gm-sheet {', '.ak-bsheet {']) {
    const i = all.indexOf(sel);
    assert.ok(i >= 0, sel);
    assert.match(all.slice(i, all.indexOf('}', i)), /animation:\s*up\b/, sel);
  }
});
