//! Compare the Rust raw text table and codes with the Python/numpy reference.
//!
//!     cargo run -p decklight-cards --bin emb_check -- \
//!         --npz data/cards/emb/qwen3-0.6b-d512.npz \
//!         --raw data/cache/onnx_parity/obs/raw.npy --codes data/cache/onnx_parity/obs/codes.npy
//!
//! Bit-identical means exit 0; otherwise the exact max |Δ| is printed and the exit status is 2.

use std::path::PathBuf;
use std::process::ExitCode;

use decklight_cards::emb::{self, Emb};

fn load_npy(path: &PathBuf) -> Result<emb::Npy, String> {
    let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    emb::parse_npy(&buf).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> ExitCode {
    let mut npz = PathBuf::from("../data/cards/emb/qwen3-0.6b-d512.npz");
    let mut raw_path = PathBuf::from("../data/cache/onnx_parity/obs/raw.npy");
    let mut codes_path = PathBuf::from("../data/cache/onnx_parity/obs/codes.npy");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--npz" => npz = args.next().unwrap_or_default().into(),
            "--raw" => raw_path = args.next().unwrap_or_default().into(),
            "--codes" => codes_path = args.next().unwrap_or_default().into(),
            "-h" | "--help" => {
                println!("emb_check --npz FILE --raw FILE --codes FILE");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }
    let e = match Emb::load(&npz) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("ALERT emb_check: {err}");
            return ExitCode::from(2);
        }
    };
    let ref_codes = match load_npy(&codes_path).and_then(|n| emb::i64s(&n)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ALERT emb_check: {err}");
            return ExitCode::from(2);
        }
    };
    let ref_raw = match load_npy(&raw_path).and_then(|n| emb::f32s(&n)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("ALERT emb_check: {err}");
            return ExitCode::from(2);
        }
    };
    let (table, rows, cols) = e.raw_table();
    let codes_equal = e.codes() == ref_codes.as_slice();
    let mut max_abs = 0f64;
    let mut first_bad: Option<(usize, usize)> = None;
    let mut bit_equal = true;
    let n = table.len().min(ref_raw.len());
    for i in 0..n {
        let a = table[i];
        let b = ref_raw[i];
        if a.to_bits() != b.to_bits() {
            bit_equal = false;
            if first_bad.is_none() {
                first_bad = Some((i / cols.max(1), i % cols.max(1)));
            }
            let d = (a as f64 - b as f64).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    if table.len() != ref_raw.len() {
        bit_equal = false;
        if first_bad.is_none() {
            first_bad = Some((n / cols.max(1), 0));
        }
    }
    println!(
        "{}",
        emb::summary_json(codes_equal, max_abs, rows, cols, first_bad)
    );
    if !codes_equal || !bit_equal {
        eprintln!(
            "ALERT emb_check: {} (codes_equal={codes_equal}, max_abs={max_abs}, first_bad={first_bad:?})",
            if codes_equal { "raw table differs" } else { "codes differ" }
        );
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}
