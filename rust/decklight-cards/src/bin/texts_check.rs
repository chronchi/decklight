//! Re-read the card store in Rust and compare `scripts/app/texts_dump.py` line by line.
//!
//!     cargo run -p decklight-cards --bin texts_check -- \
//!         --texts data/cache/app/texts.jsonl --cdb-dir ../third_party/BabelCDB \
//!         --strings ../third_party/edopro-strings.conf
//!
//! Prints a JSON summary and exits 2 on any mismatch.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use decklight_cards::texts::Texts;
use serde_json::Value;

fn esc(s: &str) -> String {
    s.chars().take(300).collect::<String>().replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() -> ExitCode {
    let mut texts_path = PathBuf::from("../data/cache/app/texts.jsonl");
    let mut cdb_dir = PathBuf::from("../third_party/BabelCDB");
    let mut strings = PathBuf::from("../third_party/edopro-strings.conf");
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--texts" => texts_path = args.next().unwrap_or_default().into(),
            "--cdb-dir" => cdb_dir = args.next().unwrap_or_default().into(),
            "--strings" => strings = args.next().unwrap_or_default().into(),
            "--extra-dir" => extra.push(args.next().unwrap_or_default().into()),
            "-h" | "--help" => {
                println!("texts_check --texts PATH --cdb-dir DIR --strings FILE [--extra-dir DIR]...");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }
    if extra.is_empty()
        && let Some(raw) = std::env::var_os("YGO_EXTRA_CARDS")
    {
        // Python's `default_cdbs()` honours the env when no explicit dirs are given.
        extra.extend(std::env::split_paths(&raw).filter(|d| d.is_dir()));
    }
    let store = match Texts::load(&cdb_dir, &extra, &strings) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ALERT texts_check: {e}");
            return ExitCode::from(2);
        }
    };
    let f = match File::open(&texts_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ALERT texts_check: cannot open {}: {e}", texts_path.display());
            return ExitCode::from(2);
        }
    };
    let (mut codes, mut sys_strings, mut mismatches) = (0usize, 0usize, 0usize);
    let mut first: Option<String> = None;
    for (lineno, line) in BufReader::new(f).lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("ALERT texts_check: read error at line {}: {e}", lineno + 1);
                return ExitCode::from(2);
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ALERT texts_check: bad JSON at line {}: {e}", lineno + 1);
                return ExitCode::from(2);
            }
        };
        let bad = if let Some(code) = v["code"].as_u64() {
            codes += 1;
            let code = code as u32;
            let name = store.name(code);
            let text = store.effect_text(code);
            let strs: Vec<String> = (0..16).map(|i| store.desc_text(code as i64, i)).collect();
            let stats = store.stats_of(code).map(|s| [s.0, s.1, s.2, s.3, s.4, s.5 as i32]);
            let want_stats = v["stats"].as_array().map(|a| {
                a.iter().map(|x| x.as_i64().unwrap_or(0) as i32).collect::<Vec<_>>()
            });
            let got_stats = stats.map(|s| s.to_vec());
            let ok = name == v["name"].as_str().unwrap_or("")
                && text == v["text"].as_str().unwrap_or("")
                && v["strs"].as_array().is_some_and(|a| {
                    a.len() == 16 && a.iter().zip(&strs).all(|(x, y)| x.as_str() == Some(y.as_str()))
                })
                && want_stats == got_stats;
            if !ok {
                Some(format!(
                    "{{\"line\":{},\"code\":{},\"name_py\":\"{}\",\"name_rs\":\"{}\",\"text_py\":\"{}\",\"text_rs\":\"{}\"}}",
                    lineno + 1,
                    code,
                    esc(v["name"].as_str().unwrap_or("")),
                    esc(name),
                    esc(v["text"].as_str().unwrap_or("")),
                    esc(&text),
                ))
            } else {
                None
            }
        } else if let Some(n) = v["sys"].as_u64() {
            sys_strings += 1;
            let got = store.desc_text(0, n as i64);
            let want = v["text"].as_str().unwrap_or("");
            if got == want {
                None
            } else {
                Some(format!(
                    "{{\"line\":{},\"sys\":{n},\"text_py\":\"{}\",\"text_rs\":\"{}\"}}",
                    lineno + 1,
                    esc(want),
                    esc(&got)
                ))
            }
        } else {
            eprintln!("ALERT texts_check: line {} has neither code nor sys", lineno + 1);
            return ExitCode::from(2);
        };
        if let Some(detail) = bad {
            mismatches += 1;
            if first.is_none() {
                first = Some(detail);
            }
        }
    }
    println!(
        "{{\"codes\":{codes},\"sys_strings\":{sys_strings},\"mismatches\":{mismatches},\"first\":{}}}",
        first.clone().unwrap_or_else(|| "null".into())
    );
    if mismatches > 0 {
        eprintln!("ALERT texts_check: {mismatches} mismatches (first: {})", first.unwrap_or_default());
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}
