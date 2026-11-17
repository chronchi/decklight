// MP4 container for one H.264 track (Games > Export video, web/features/games/export.js): the browser's VideoEncoder
// makes the samples (format "avc", so its decoderConfig.description is the avcC box body); this writes ftyp, moov
// first (plays while it downloads) and one mdat chunk. Pure function: tests/js/media.test.mjs.

const enc = new TextEncoder();

function box(type, ...parts) {
  const body = parts.flat(Infinity);
  let n = 8;
  for (const p of body) n += p.length;
  const out = new Uint8Array(n);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, n);
  out.set(enc.encode(type), 4);
  let o = 8;
  for (const p of body) { out.set(p, o); o += p.length; }
  return out;
}
const u8 = (...v) => new Uint8Array(v);
function u16(v) { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, v); return b; }
function u32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v >>> 0); return b; }
const zeros = (n) => new Uint8Array(n);
const MATRIX = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000].map(u32);
const full = (type, version, flags, ...parts) => box(type, u8(version, (flags >> 16) & 255, (flags >> 8) & 255, flags & 255), ...parts);

// samples: [{data: Uint8Array (AVC length-prefixed NAL units), dur (in timescale units), key: bool}]
export function muxMp4({width, height, avcC, samples, timescale}) {
  timescale = timescale || 1000;
  if (!samples.length) throw new Error("no video frames to write");
  if (!samples[0].key) throw new Error("the first video frame is not a key frame");
  const dur = samples.reduce((s, x) => s + x.dur, 0);
  const stts = [];
  for (const s of samples) {
    const last = stts[stts.length - 1];
    if (last && last[1] === s.dur) last[0]++; else stts.push([1, s.dur]);
  }
  const keys = [];
  samples.forEach((s, i) => { if (s.key) keys.push(i + 1); });
  const moov = (offset) => box("moov",
    full("mvhd", 0, 0, u32(0), u32(0), u32(timescale), u32(dur), u32(0x10000), u16(0x100), zeros(10), MATRIX, zeros(24), u32(2)),
    box("trak",
      full("tkhd", 0, 3, u32(0), u32(0), u32(1), u32(0), u32(dur), zeros(8), u16(0), u16(0), u16(0), u16(0), MATRIX, u32(width << 16), u32(height << 16)),
      box("mdia",
        full("mdhd", 0, 0, u32(0), u32(0), u32(timescale), u32(dur), u16(0x55c4), u16(0)),
        full("hdlr", 0, 0, u32(0), enc.encode("vide"), zeros(12), enc.encode("VideoHandler\0")),
        box("minf",
          full("vmhd", 0, 1, zeros(8)),
          box("dinf", full("dref", 0, 0, u32(1), full("url ", 0, 1))),
          box("stbl",
            full("stsd", 0, 0, u32(1), box("avc1", zeros(6), u16(1), zeros(16), u16(width), u16(height), u32(0x480000), u32(0x480000), u32(0),
              u16(1), zeros(32), u16(0x18), u16(0xffff), box("avcC", avcC))),
            full("stts", 0, 0, u32(stts.length), stts.map(([n, d]) => [u32(n), u32(d)])),
            full("stss", 0, 0, u32(keys.length), keys.map(u32)),
            full("stsc", 0, 0, u32(1), u32(1), u32(samples.length), u32(1)),
            full("stsz", 0, 0, u32(0), u32(samples.length), samples.map((s) => u32(s.data.length))),
            full("stco", 0, 0, u32(1), u32(offset)))))));
  const ftyp = box("ftyp", enc.encode("isom"), u32(0x200), enc.encode("isom"), enc.encode("iso2"), enc.encode("avc1"), enc.encode("mp41"));
  // moov's size does not depend on the offset's value, so measure it once, then write it with the real offset.
  const offset = ftyp.length + moov(0).length + 8;
  const mdat = box("mdat", samples.map((s) => s.data));
  const out = new Uint8Array(offset - 8 + mdat.length);
  out.set(ftyp, 0);
  out.set(moov(offset), ftyp.length);
  out.set(mdat, offset - 8);
  return out;
}
