// The export encoders (web/lib/gif.js, web/lib/mp4.js): a GIF decodes back to its frames' palette indices, and the
// MP4's boxes point at its samples.
import {test} from "node:test";
import assert from "node:assert/strict";
import {encodeGif, lzw, quantize} from "../../web/lib/gif.js";
import {muxMp4} from "../../web/lib/mp4.js";

// A plain GIF LZW decoder (the reference side).
function unlzw(bytes, minSize) {
  const clear = 1 << minSize, end = clear + 1;
  let size = minSize + 1, dict = [], next = end + 1, prev = null, pos = 0;
  const reset = () => { dict = []; for (let i = 0; i < clear; i++) dict[i] = [i]; size = minSize + 1; next = end + 1; prev = null; };
  reset();
  const out = [];
  const read = () => {
    let v = 0;
    for (let i = 0; i < size; i++, pos++) v |= ((bytes[pos >> 3] >> (pos & 7)) & 1) << i;
    return v;
  };
  for (;;) {
    const c = read();
    if (c === clear) { reset(); continue; }
    if (c === end) break;
    let e;
    if (c < next && dict[c]) e = dict[c];
    else if (c === next && prev) e = [...prev, prev[0]];
    else throw new Error("bad code " + c);
    out.push(...e);
    if (prev && next < 4096) { dict[next++] = [...prev, e[0]]; }
    if (next === 1 << size && size < 12) size++;
    prev = e;
  }
  return out;
}

test("GIF LZW round-trips long and repetitive streams (code size growth and table resets)", () => {
  for (const n of [0, 1, 7, 5000, 70000]) {
    const idx = new Uint8Array(n);
    let x = 12345;
    for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; idx[i] = i % 97 < 60 ? (x >> 16) & 255 : idx[i - 1] || 0; }
    assert.deepEqual(unlzw(lzw(idx), 8), [...idx], "n=" + n);
  }
});

test("quantize keeps a few colours exact and caps at 256", () => {
  const px = new Uint8Array([0, 0, 0, 255, 248, 248, 248, 255, 200, 40, 40, 255, 200, 40, 40, 255]);
  const q = quantize(px, 256);
  assert.equal(q.colors, 3);
  const c = (i) => [...q.palette.subarray(q.index[i] * 3, q.index[i] * 3 + 3)];
  assert.deepEqual(c(0), [4, 4, 4]);
  assert.deepEqual(c(2), c(3));
  const many = new Uint8Array(64 * 64 * 4);
  for (let i = 0; i < 64 * 64; i++) { many[i * 4] = i & 255; many[i * 4 + 1] = (i >> 4) & 255; many[i * 4 + 2] = (i * 7) & 255; }
  assert.ok(quantize(many, 256).colors <= 256);
});

test("encodeGif writes a looping GIF89a with one image per frame", () => {
  const f = (v) => ({rgba: new Uint8Array(4 * 6 * 4).fill(v), delay: 1600});
  const g = encodeGif(4, 6, [f(10), f(200)]);
  assert.equal(new TextDecoder().decode(g.subarray(0, 6)), "GIF89a");
  assert.equal(g[6] | (g[7] << 8), 4);
  assert.equal(g[g.length - 1], 0x3b);
  assert.equal([...g].filter((b, i) => b === 0x2c && g[i - 1] === 0 && g[i - 7] === 0xf9).length, 2, "two images");
  const gce = g.indexOf(0xf9);
  assert.equal(g[gce + 3] | (g[gce + 4] << 8), 160, "1600 ms = 160 cs");
});

function boxes(b, start, end, path, out) {
  const dv = new DataView(b.buffer, b.byteOffset);
  for (let o = start; o < end;) {
    const n = dv.getUint32(o), type = new TextDecoder().decode(b.subarray(o + 4, o + 8));
    assert.ok(n >= 8 && o + n <= end, path + type + " size " + n);
    out[path + type] = {o, n};
    const kids = {moov: 8, trak: 8, mdia: 8, minf: 8, stbl: 8, dinf: 8}[type];
    if (kids) boxes(b, o + kids, o + n, path + type + "/", out);
    o += n;
  }
  return out;
}

test("muxMp4 puts moov first and its sample table points into mdat", () => {
  const samples = [{data: new Uint8Array([0, 0, 0, 2, 0x65, 1]), dur: 400, key: true}, {data: new Uint8Array([0, 0, 0, 1, 0x41]), dur: 400, key: false},
    {data: new Uint8Array([0, 0, 0, 1, 0x41]), dur: 1600, key: false}];
  const mp4 = muxMp4({width: 720, height: 1280, avcC: new Uint8Array([1, 0x42, 0, 0x1f, 0xff, 0xe0, 0]), samples});
  const b = boxes(mp4, 0, mp4.length, "", {});
  assert.deepEqual(Object.keys(b).filter((k) => !k.includes("/")), ["ftyp", "moov", "mdat"]);
  const dv = new DataView(mp4.buffer);
  const stco = b["moov/trak/mdia/minf/stbl/stco"];
  assert.equal(dv.getUint32(stco.o + 16), b.mdat.o + 8, "chunk offset = mdat payload");
  assert.equal(b.mdat.n - 8, samples.reduce((s, x) => s + x.data.length, 0));
  const mvhd = b["moov/mvhd"];
  assert.equal(dv.getUint32(mvhd.o + 24), 2400, "duration in ms");
  const stss = b["moov/trak/mdia/minf/stbl/stss"];
  assert.equal(dv.getUint32(stss.o + 12), 1);
  assert.throws(() => muxMp4({width: 2, height: 2, avcC: new Uint8Array(1), samples: [{data: new Uint8Array(1), dur: 1, key: false}]}), /key frame/);
});
