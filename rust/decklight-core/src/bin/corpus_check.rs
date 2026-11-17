//! Re-parse `scripts/app/parse_corpus.py` output in Rust and compare the canonical JSON strings.
//!
//!     cargo run -p decklight-core --bin corpus_check -- --corpus data/cache/app/corpus.jsonl
//!
//! Prints one JSON summary (`msgs`, `fields`, `locs`, `splits`, `mismatches`, `by_class`, `first`) and exits 2
//! on any mismatch.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::ExitCode;

use decklight_core::{messages as M, query as Q};
use serde_json::Value;

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("odd hex length".into());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let b = s.as_bytes();
    for i in (0..b.len()).step_by(2) {
        let hi = (b[i] as char).to_digit(16).ok_or("bad hex")?;
        let lo = (b[i + 1] as char).to_digit(16).ok_or("bad hex")?;
        out.push((hi * 16 + lo) as u8);
    }
    Ok(out)
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        let _ = write!(s, "{x:02x}");
    }
    s
}

fn canon_of(v: &Value) -> String {
    serde_json::to_string(v).unwrap()
}

fn chain_json(v: &Value) -> String {
    let mut s = String::from("[");
    if let Some(items) = v.as_array() {
        for (i, c) in items.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&canon_of(c));
        }
    }
    s.push(']');
    s
}

fn cards_json(v: &Value) -> String {
    let mut s = String::from("[");
    if let Some(items) = v.as_array() {
        for (i, c) in items.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            if c.is_null() {
                s.push_str("null");
            } else {
                s.push_str(&canon_of(c));
            }
        }
    }
    s.push(']');
    s
}

fn main() -> ExitCode {
    let mut corpus = std::path::PathBuf::from("../data/cache/app/corpus.jsonl");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--corpus" => corpus = args.next().unwrap_or_default().into(),
            "-h" | "--help" => {
                println!("corpus_check --corpus PATH");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }
    let f = match File::open(&corpus) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ALERT corpus_check: cannot open {}: {e}", corpus.display());
            return ExitCode::from(2);
        }
    };
    let (mut msgs, mut fields, mut locs, mut splits) = (0usize, 0usize, 0usize, 0usize);
    let mut mismatches = 0usize;
    let mut by_class: BTreeMap<String, usize> = BTreeMap::new();
    let mut first: Option<String> = None;
    for (lineno, line) in BufReader::new(f).lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("ALERT corpus_check: read error at line {}: {e}", lineno + 1);
                return ExitCode::from(2);
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ALERT corpus_check: bad JSON at line {}: {e}", lineno + 1);
                return ExitCode::from(2);
            }
        };
        let kind = v["k"].as_str().unwrap_or("");
        let mut mismatch_detail = None;
        match kind {
            "msg" => {
                msgs += 1;
                let raw = match unhex(v["raw"].as_str().unwrap_or("")) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("ALERT corpus_check: line {}: {e}", lineno + 1);
                        return ExitCode::from(2);
                    }
                };
                match M::parse_one(&raw) {
                    Ok(msg) => {
                        *by_class.entry(msg.cls.to_string()).or_default() += 1;
                        let got = msg.to_canon_json();
                        let want = canon_of(&v["canon"]);
                        if got != want {
                            mismatch_detail = Some((msg.cls.to_string(), want, got));
                        }
                    }
                    Err(e) => mismatch_detail = Some(("parse".into(), String::new(), e)),
                }
            }
            "field" => {
                fields += 1;
                let raw = unhex(v["raw"].as_str().unwrap_or("")).unwrap_or_default();
                let got = match Q::parse_chain(&raw) {
                    Ok(chain) => {
                        let mut s = String::from("[");
                        for (i, c) in chain.iter().enumerate() {
                            if i > 0 {
                                s.push(',');
                            }
                            s.push_str(&c.to_canon_json());
                        }
                        s.push(']');
                        s
                    }
                    Err(e) => format!("ERR:{e}"),
                };
                let want = chain_json(&v["chain"]);
                if got != want {
                    mismatch_detail = Some(("chain".into(), want, got));
                } else {
                    let got_lp = match Q::lps(&raw) {
                        Ok((a, b)) => format!("[{a},{b}]"),
                        Err(e) => format!("ERR:{e}"),
                    };
                    let want_lp = canon_of(&v["lps"]);
                    if got_lp != want_lp {
                        mismatch_detail = Some(("lps".into(), want_lp, got_lp));
                    }
                }
            }
            "loc" => {
                locs += 1;
                let raw = unhex(v["raw"].as_str().unwrap_or("")).unwrap_or_default();
                let con = v["con"].as_u64().unwrap_or(0) as u32;
                let loc = v["loc"].as_u64().unwrap_or(0) as u32;
                let got = match Q::parse_location(&raw, con, loc) {
                    Ok(cards) => {
                        let mut s = String::from("[");
                        for (i, c) in cards.iter().enumerate() {
                            if i > 0 {
                                s.push(',');
                            }
                            match c {
                                Some(c) => s.push_str(&c.to_canon_json()),
                                None => s.push_str("null"),
                            }
                        }
                        s.push(']');
                        s
                    }
                    Err(e) => format!("ERR:{e}"),
                };
                let want = cards_json(&v["cards"]);
                if got != want {
                    mismatch_detail = Some(("loc".into(), want, got));
                }
            }
            "split" => {
                splits += 1;
                let buf = unhex(v["buf"].as_str().unwrap_or("")).unwrap_or_default();
                let got = match M::split_buffer(&buf) {
                    Ok(parts) => format!("[{}]", parts.iter().map(|p| format!("\"{}\"", hex(p))).collect::<Vec<_>>().join(",")),
                    Err(e) => format!("ERR:{e}"),
                };
                let want = canon_of(&v["out"]);
                if got != want {
                    mismatch_detail = Some(("split".into(), want, got));
                }
            }
            other => {
                eprintln!("ALERT corpus_check: line {}: unknown kind {other:?}", lineno + 1);
                return ExitCode::from(2);
            }
        }
        if let Some((cls, want, got)) = mismatch_detail {
            mismatches += 1;
            if first.is_none() {
                let short = |s: &str| s.chars().take(400).collect::<String>();
                first = Some(format!(
                    "{{\"line\":{},\"kind\":\"{kind}\",\"class\":\"{cls}\",\"python\":\"{}\",\"rust\":\"{}\"}}",
                    lineno + 1,
                    short(&want).replace('\\', "\\\\").replace('"', "\\\""),
                    short(&got).replace('\\', "\\\\").replace('"', "\\\"")
                ));
            }
        }
    }
    let class_json = by_class
        .iter()
        .map(|(k, v)| format!("\"{k}\":{v}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"msgs\":{msgs},\"fields\":{fields},\"locs\":{locs},\"splits\":{splits},\"mismatches\":{mismatches},\
\"by_class\":{{{class_json}}},\"first\":{}}}",
        first.clone().unwrap_or_else(|| "null".into())
    );
    if mismatches > 0 {
        eprintln!("ALERT corpus_check: {mismatches} mismatches (first: {})", first.unwrap_or_default());
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}
