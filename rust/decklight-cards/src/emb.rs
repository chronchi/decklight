//! The card text embedding npz (`data/cards/emb/qwen3-0.6b-d512.npz`) and the frozen raw table
//! (`decklight.model.onnx_export.raw_table` / `sorted_codes`).
//!
//! The npz is read with the `zip` crate (its members are `.npy` files); the small `.npy` reader below handles the
//! dtypes the file actually uses (`<f2`, `<i8`, `|i1`). Arithmetic mirrors numpy: `card` and `effects` are converted
//! to f32, `effects.sum(1)` runs sequentially in f32 (the f16 inputs make this bit-identical to numpy's pairwise
//! summation on this file; the check binary reports the exact max |Δ|), then divide by `max(n_effects, 1)`.

use std::fmt::Write as _;
use std::io::Read;
use std::path::Path;

pub struct Emb {
    pub codes: Vec<i64>,
    pub card: Vec<f32>,
    pub effects: Vec<f32>,
    pub n_effects: Vec<i8>,
    pub dim: usize,
    pub n_effects_slots: usize,
}

pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub fn parse_npy(buf: &[u8]) -> Result<Npy, String> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = buf[6];
    let (hlen, off) = match major {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10usize),
        2 | 3 => {
            if buf.len() < 12 {
                return Err("truncated .npy header".into());
            }
            (u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize, 12usize)
        }
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = std::str::from_utf8(&buf[off..off + hlen]).map_err(|e| format!("npy header: {e}"))?;
    let descr = field(header, "'descr': '", '\'')?;
    if !header.contains("'fortran_order': False") {
        return Err("fortran order not supported".into());
    }
    let shape_str = field(header, "'shape': (", ')')?;
    let mut shape = Vec::new();
    for part in shape_str.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        shape.push(p.parse::<usize>().map_err(|e| format!("shape {shape_str:?}: {e}"))?);
    }
    Ok(Npy { descr, shape, data: buf[off + hlen..].to_vec() })
}

fn field(s: &str, start: &str, end: char) -> Result<String, String> {
    let i = s.find(start).ok_or_else(|| format!("npy header missing {start:?}"))? + start.len();
    let rest = &s[i..];
    let j = rest.find(end).ok_or_else(|| format!("npy header field {start:?} unterminated"))?;
    Ok(rest[..j].to_string())
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let k = 31 - mant.leading_zeros(); // 0..=9
            let exp32 = (k as i32 - 24 + 127) as u32;
            let frac = (mant - (1 << k)) << (23 - k);
            sign | (exp32 << 23) | frac
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

pub fn f32s(npy: &Npy) -> Result<Vec<f32>, String> {
    match npy.descr.as_str() {
        "<f2" => Ok(npy.data.chunks_exact(2).map(|c| f16_to_f32(le_u16(c))).collect()),
        "<f4" => Ok(npy
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        d => Err(format!("unsupported dtype {d} for f32")),
    }
}

pub fn i64s(npy: &Npy) -> Result<Vec<i64>, String> {
    match npy.descr.as_str() {
        "<i8" | "|i8" => Ok(npy
            .data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        d => Err(format!("unsupported dtype {d} for i64")),
    }
}

pub fn i8s(npy: &Npy) -> Result<Vec<i8>, String> {
    match npy.descr.as_str() {
        "|i1" | "<i1" => Ok(npy.data.iter().map(|&b| b as i8).collect()),
        d => Err(format!("unsupported dtype {d} for i8")),
    }
}

fn zip_member(path: &Path, name: &str) -> Result<Vec<u8>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut m = z.by_name(name).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
    let mut buf = Vec::new();
    m.read_to_end(&mut buf).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
    Ok(buf)
}

impl Emb {
    /// Just the code vocabulary (`codes.npy`): the server's deck legality check needs no embeddings themselves,
    /// and the full table is a ~0.5 s read it would otherwise pay at every start.
    pub fn load_codes(path: &Path) -> Result<Vec<i64>, String> {
        i64s(&parse_npy(&zip_member(path, "codes.npy")?)?)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let codes = i64s(&parse_npy(&zip_member(path, "codes.npy")?)?)?;
        let card_npy = parse_npy(&zip_member(path, "card.npy")?)?;
        let eff_npy = parse_npy(&zip_member(path, "effects.npy")?)?;
        let neff_npy = parse_npy(&zip_member(path, "n_effects.npy")?)?;
        let dim_npy = parse_npy(&zip_member(path, "dim.npy")?)?;
        let dim = i64s(&dim_npy)?[0] as usize;
        let card_shape = card_npy.shape.clone();
        let eff_shape = eff_npy.shape.clone();
        if card_shape.len() != 2 || eff_shape.len() != 3 {
            return Err(format!("unexpected shapes {card_shape:?} {eff_shape:?}"));
        }
        if card_shape[1] != dim || eff_shape[2] != dim || card_shape[0] != codes.len() {
            return Err(format!("shape/dim/codes disagree: {card_shape:?} {eff_shape:?} dim={dim}"));
        }
        let neff = i8s(&neff_npy)?;
        if neff.len() != codes.len() || eff_shape[0] != codes.len() {
            return Err("n_effects/codes/effects row counts disagree".into());
        }
        Ok(Emb {
            codes,
            card: f32s(&card_npy)?,
            effects: f32s(&eff_npy)?,
            n_effects: neff,
            dim,
            n_effects_slots: eff_shape[1],
        })
    }

    /// `sorted_codes`: the code vocabulary, int64 (the npz already is sorted).
    pub fn codes(&self) -> &[i64] {
        &self.codes
    }

    /// `raw_table`: `[card, effects.sum(1) / max(n_effects, 1)]` in float32, rows in the npz's code order.
    pub fn raw_table(&self) -> (Vec<f32>, usize, usize) {
        let n = self.codes.len();
        let d = self.dim;
        let e = self.n_effects_slots;
        let cols = 2 * d;
        let mut out = vec![0f32; n * cols];
        for row in 0..n {
            out[row * cols..row * cols + d].copy_from_slice(&self.card[row * d..row * d + d]);
            let denom = (self.n_effects[row] as i32).max(1) as f32;
            for j in 0..d {
                let mut acc = 0f32;
                for k in 0..e {
                    acc += self.effects[(row * e + k) * d + j];
                }
                out[row * cols + d + j] = acc / denom;
            }
        }
        (out, n, cols)
    }
}

/// A compact JSON summary line for the emb check (no serde dependency here).
pub fn summary_json(codes_equal: bool, max_abs: f64, rows: usize, cols: usize, first_bad: Option<(usize, usize)>) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "{{\"codes_equal\":{},\"max_abs\":{max_abs},\"rows\":{rows},\"cols\":{cols},\"first_bad\":"
    , if codes_equal { "true" } else { "false" });
    match first_bad {
        Some((r, c)) => {
            let _ = write!(s, "[{r},{c}]");
        }
        None => s.push_str("null"),
    }
    s.push('}');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), f32::from_bits(0x3380_0000)); // smallest subnormal
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn half_roundtrip_subnormals() {
        for h in [0x0001u16, 0x0002, 0x03FF, 0x0400, 0x3C00, 0xBC00] {
            assert_eq!(f32_to_f16(f16_to_f32(h)), h);
        }
    }

    #[test]
    fn npy_header_roundtrip() {
        // numpy: np.save on an int64 array [1,2]
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        let h = b"{'descr': '<i8', 'fortran_order': False, 'shape': (2,), }          \n";
        b.extend_from_slice(&(h.len() as u16).to_le_bytes());
        b.extend_from_slice(h);
        b.extend_from_slice(&1i64.to_le_bytes());
        b.extend_from_slice(&2i64.to_le_bytes());
        let n = parse_npy(&b).unwrap();
        assert_eq!((n.descr.as_str(), n.shape.clone()), ("<i8", vec![2]));
        assert_eq!(i64s(&n).unwrap(), vec![1, 2]);
    }
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e);
        return sign | ((m >> 13) as u16);
    }
    let mut h = sign | ((e as u16) << 10) | ((mant >> 13) as u16);
    if mant & 0x1000 != 0 {
        h = h.wrapping_add(1);
    }
    h
}

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_s = if shape.len() == 1 {
        format!("{},", shape[0])
    } else {
        shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_s}), }}");
    let mut header = dict.into_bytes();
    let base = 10 + header.len() + 1;
    let pad = (64 - (base % 64)) % 64;
    header.extend(std::iter::repeat_n(b' ', pad));
    header.push(b'\n');
    let mut out = Vec::with_capacity(10 + header.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out
}

fn npy_f16(shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut out = npy_header("<f2", shape);
    for &v in values {
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

fn npy_i64(shape: &[usize], values: &[i64]) -> Vec<u8> {
    let mut out = npy_header("<i8", shape);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn npy_i8(shape: &[usize], values: &[i8]) -> Vec<u8> {
    let mut out = npy_header("|i1", shape);
    out.extend(values.iter().map(|&v| v as u8));
    out
}

fn npy_str0(s: &str) -> Vec<u8> {
    let n = s.chars().count().max(1);
    let mut out = npy_header(&format!("<U{n}"), &[]);
    let mut chars: Vec<u32> = s.chars().map(|c| c as u32).collect();
    chars.resize(n, 0);
    for c in chars {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out
}

fn zip_write(path: &Path, members: &[(&str, Vec<u8>)]) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut z = zip::ZipWriter::new(f);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, data) in members {
        use std::io::Write as _;
        z.start_file(*name, opts).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
        z.write_all(data).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
    }
    z.finish().map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Write an embedding npz in the schema `build` / `append_card_embeddings` use.
pub fn write_npz(
    path: &Path,
    codes: &[i64],
    card: &[f32],
    effects: &[f32],
    n_effects: &[i8],
    dim: usize,
    slots: usize,
    model: &str,
) -> Result<(), String> {
    let n = codes.len();
    if card.len() != n * dim || effects.len() != n * slots * dim || n_effects.len() != n {
        return Err("embedding arrays disagree with codes".into());
    }
    let members = vec![
        ("codes.npy", npy_i64(&[n], codes)),
        ("card.npy", npy_f16(&[n, dim], card)),
        ("effects.npy", npy_f16(&[n, slots, dim], effects)),
        ("n_effects.npy", npy_i8(&[n], n_effects)),
        ("dim.npy", npy_i64(&[], &[dim as i64])),
        ("model.npy", npy_str0(model)),
    ];
    zip_write(path, &members)
}

/// Merge a pack's delta embeddings over the stock table. Codes present in both are taken from the pack
/// (a changed card text must use its new vector); the result is sorted by code.
pub fn merge_emb(base: &Path, extra: &Path, out: &Path) -> Result<usize, String> {
    let a = Emb::load(base)?;
    let b = Emb::load(extra)?;
    let mut rows: std::collections::BTreeMap<i64, (Vec<f32>, Vec<f32>, i8)> = std::collections::BTreeMap::new();
    let slots = a.n_effects_slots.max(b.n_effects_slots);
    let dim = a.dim.max(b.dim);
    if a.dim != b.dim {
        return Err(format!("embedding dims disagree: {} != {}", a.dim, b.dim));
    }
    let put = |rows: &mut std::collections::BTreeMap<i64, (Vec<f32>, Vec<f32>, i8)>, e: &Emb| {
        for (i, &code) in e.codes.iter().enumerate() {
            let card = e.card[i * e.dim..(i + 1) * e.dim].to_vec();
            let mut eff = vec![0f32; slots * dim];
            let row_slots = e.n_effects_slots;
            for k in 0..row_slots {
                let src = (i * row_slots + k) * e.dim;
                let dst = k * dim;
                eff[dst..dst + dim].copy_from_slice(&e.effects[src..src + e.dim]);
            }
            rows.insert(code, (card, eff, e.n_effects[i]));
        }
    };
    put(&mut rows, &a);
    put(&mut rows, &b);
    let codes: Vec<i64> = rows.keys().copied().collect();
    let mut card = Vec::with_capacity(codes.len() * dim);
    let mut effects = Vec::with_capacity(codes.len() * slots * dim);
    let mut n_eff = Vec::with_capacity(codes.len());
    for (c, e, n) in rows.values() {
        card.extend_from_slice(c);
        effects.extend_from_slice(e);
        n_eff.push(*n);
    }
    let tmp = out.with_extension(format!("tmp-{}", std::process::id()));
    write_npz(&tmp, &codes, &card, &effects, &n_eff, dim, slots, "merged")?;
    std::fs::rename(&tmp, out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(codes.len())
}
