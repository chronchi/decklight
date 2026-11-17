// Animated GIF encoder for Games > Export (web/features/games/export.js): each frame gets its own 256-colour palette
// (median cut over a 15-bit histogram), LZW-coded as GIF89a, looping forever. Pure functions: tests/js/media.test.mjs.

class Bytes {
  constructor(n) { this.a = new Uint8Array(n || 1 << 16); this.n = 0; }
  room(k) {
    if (this.n + k <= this.a.length) return;
    let m = this.a.length * 2;
    while (m < this.n + k) m *= 2;
    const b = new Uint8Array(m);
    b.set(this.a.subarray(0, this.n));
    this.a = b;
  }
  byte(v) { this.room(1); this.a[this.n++] = v; }
  u16(v) { this.byte(v & 255); this.byte((v >> 8) & 255); }
  bytes(b) { this.room(b.length); this.a.set(b, this.n); this.n += b.length; }
  str(s) { for (let i = 0; i < s.length; i++) this.byte(s.charCodeAt(i)); }
  done() { return this.a.slice(0, this.n); }
}

// Median cut: up to `max` colours for RGBA pixels (alpha ignored). Returns the palette (768 bytes, unused entries
// black) and each pixel's palette index.
export function quantize(rgba, max) {
  max = max || 256;
  const hist = new Uint32Array(32768);
  const n = rgba.length >> 2;
  const keys = new Uint16Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const k = ((rgba[p] >> 3) << 10) | ((rgba[p + 1] >> 3) << 5) | (rgba[p + 2] >> 3);
    keys[i] = k;
    hist[k]++;
  }
  const used = [];
  for (let k = 0; k < 32768; k++) if (hist[k]) used.push(k);
  const ch = (k, c) => (c === 0 ? k >> 10 : c === 1 ? (k >> 5) & 31 : k & 31);
  const box = (ks) => {
    const lo = [31, 31, 31], hi = [0, 0, 0];
    let count = 0;
    for (const k of ks) {
      for (let c = 0; c < 3; c++) { const v = ch(k, c); if (v < lo[c]) lo[c] = v; if (v > hi[c]) hi[c] = v; }
      count += hist[k];
    }
    const span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const c = span[0] >= span[1] && span[0] >= span[2] ? 0 : span[1] >= span[2] ? 1 : 2;
    return {ks, count, c, span: span[c]};
  };
  const boxes = [box(used)];
  while (boxes.length < max) {
    let bi = -1, best = 0;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      const score = b.span * b.count;
      if (b.ks.length > 1 && b.span > 0 && score > best) { best = score; bi = i; }
    }
    if (bi < 0) break;
    const b = boxes[bi];
    b.ks.sort((x, y) => ch(x, b.c) - ch(y, b.c));
    let half = b.count / 2, cut = 0;
    for (; cut < b.ks.length - 1 && half > 0; cut++) half -= hist[b.ks[cut]];
    cut = Math.max(1, Math.min(b.ks.length - 1, cut));
    boxes.splice(bi, 1, box(b.ks.slice(0, cut)), box(b.ks.slice(cut)));
  }
  const palette = new Uint8Array(768);
  const lut = new Uint8Array(32768);
  boxes.forEach((b, i) => {
    let r = 0, g = 0, bl = 0, w = 0;
    for (const k of b.ks) {
      const h = hist[k];
      r += ((k >> 10) * 8 + 4) * h; g += (((k >> 5) & 31) * 8 + 4) * h; bl += ((k & 31) * 8 + 4) * h; w += h;
      lut[k] = i;
    }
    if (w) { palette[i * 3] = Math.round(r / w); palette[i * 3 + 1] = Math.round(g / w); palette[i * 3 + 2] = Math.round(bl / w); }
  });
  const index = new Uint8Array(n);
  for (let i = 0; i < n; i++) index[i] = lut[keys[i]];
  return {palette, index, colors: boxes.length};
}

// GIF LZW with 8-bit symbols: the code stream as bytes (before the 255-byte sub-blocks).
export function lzw(index) {
  const CLEAR = 256, END = 257;
  const out = new Bytes(index.length >> 1);
  let acc = 0, bits = 0, size = 9, next = 258;
  const emit = (code) => {
    acc |= code << bits;
    bits += size;
    while (bits >= 8) { out.byte(acc & 255); acc >>>= 8; bits -= 8; }
  };
  let dict = new Map();
  emit(CLEAR);
  if (!index.length) { emit(END); if (bits) out.byte(acc & 255); return out.done(); }
  let prefix = index[0];
  for (let i = 1; i < index.length; i++) {
    const k = index[i];
    const key = prefix * 256 + k;
    const hit = dict.get(key);
    if (hit !== undefined) { prefix = hit; continue; }
    emit(prefix);
    if (next < 4096) {
      dict.set(key, next++);
      if (next > 1 << size && size < 12) size++;
    } else {
      emit(CLEAR);
      dict = new Map();
      next = 258;
      size = 9;
    }
    prefix = k;
  }
  emit(prefix);
  emit(END);
  if (bits) out.byte(acc & 255);
  return out.done();
}

// A GIF89a written a frame at a time (a long export never holds every frame): add(rgba, delayMs) per frame, then
// done() for the bytes. Loops forever; each frame has its own palette.
export class GifWriter {
  constructor(width, height) {
    this.w = width; this.h = height; this.frames = 0;
    this.out = new Bytes();
    this.out.str("GIF89a");
    this.out.u16(width); this.out.u16(height);
    this.out.byte(0); this.out.byte(0); this.out.byte(0);        // no global colour table
    this.out.bytes([0x21, 0xff, 11]); this.out.str("NETSCAPE2.0"); this.out.bytes([3, 1, 0, 0, 0]);  // loop forever
  }
  add(rgba, delay) {
    if (rgba.length !== this.w * this.h * 4) throw new Error("GIF frame is " + rgba.length / 4 + " pixels, not " + this.w * this.h);
    const out = this.out;
    const {palette, index} = quantize(rgba, 256);
    out.bytes([0x21, 0xf9, 4, 0x04]);            // graphic control: disposal "leave in place"
    out.u16(Math.max(2, Math.round((delay || 100) / 10)));
    out.bytes([0, 0]);
    out.byte(0x2c); out.u16(0); out.u16(0); out.u16(this.w); out.u16(this.h);
    out.byte(0x80 | 7);                          // local colour table of 256 entries
    out.bytes(palette);
    out.byte(8);
    const data = lzw(index);
    for (let i = 0; i < data.length; i += 255) {
      const n = Math.min(255, data.length - i);
      out.byte(n);
      out.bytes(data.subarray(i, i + n));
    }
    out.byte(0);
    this.frames++;
  }
  get size() { return this.out.n; }
  done() {
    if (!this.frames) throw new Error("no frames in the GIF");
    this.out.byte(0x3b);
    return this.out.done();
  }
}

// frames: [{rgba (width*height*4), delay (ms)}] of the same size -> GIF89a bytes.
export function encodeGif(width, height, frames) {
  const g = new GifWriter(width, height);
  for (const f of frames) g.add(f.rgba, f.delay);
  return g.done();
}
