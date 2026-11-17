//! Replays the CPython RNG fixtures produced by `scripts/app/gen_search_fixtures.py`.
//!
//!     cargo run -p decklight-search --bin search_rng_check -- data/cache/app/search_rng_fixtures.jsonl.gz
//!
//! Exits 2 on any difference; prints the counts per kind.

use std::io::{BufRead, BufReader};

use decklight_search::pyrandom::Random;
use flate2::read::GzDecoder;
use serde_json::Value;

fn die(msg: &str) -> ! {
    eprintln!("search_rng_check: {msg}");
    std::process::exit(2);
}

fn make(kind: &str, seed: &str) -> Random {
    match kind {
        "int_seed" => Random::from_i128(seed.parse().unwrap_or_else(|_| die(&format!("bad int seed {seed}")))),
        "str_seed" => Random::from_str(seed),
        _ => die(&format!("unknown kind {kind}")),
    }
}

fn bits(x: f64) -> u64 {
    x.to_bits()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "data/cache/app/search_rng_fixtures.jsonl.gz".into());
    let f = std::fs::File::open(&path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    let mut n_int = 0usize;
    let mut n_str = 0usize;
    let mut checks = 0usize;
    for line in BufReader::new(GzDecoder::new(f)).lines() {
        let line = line.unwrap_or_else(|e| die(&e.to_string()));
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).unwrap_or_else(|e| die(&format!("json: {e}")));
        let kind = v["kind"].as_str().unwrap();
        let seed = v["seed"].as_str().unwrap();
        let mut rng = make(kind, seed);
        if kind == "int_seed" {
            n_int += 1;
        } else {
            n_str += 1;
        }
        let random50 = v["random50"].as_array().unwrap();
        for (i, expected) in random50.iter().enumerate() {
            let got = bits(rng.random());
            checks += 1;
            if got != expected.as_u64().unwrap() {
                die(&format!("{kind} {seed}: random()[{i}] {got:#x} != {:#x}", expected.as_u64().unwrap()));
            }
        }
        for pair in v["getrandbits"].as_array().unwrap() {
            let k = pair[0].as_u64().unwrap() as u32;
            let expected: u128 = pair[1].as_str().unwrap().parse().unwrap();
            let got = rng.getrandbits(k);
            checks += 1;
            if got != expected {
                die(&format!("{kind} {seed}: getrandbits({k}) {got} != {expected}"));
            }
        }
        for pair in v["randbelow"].as_array().unwrap() {
            let n: u128 = pair[0].as_str().unwrap().parse().unwrap();
            let expected: u128 = pair[1].as_str().unwrap().parse().unwrap();
            let got = rng.randbelow(n);
            checks += 1;
            if got != expected {
                die(&format!("{kind} {seed}: randbelow({n}) {got} != {expected}"));
            }
        }
        for (n, expected) in v["shuffle"].as_array().unwrap().iter().enumerate() {
            let mut r = make(kind, seed);
            let mut lst: Vec<u32> = (0..n as u32).collect();
            r.shuffle(&mut lst);
            checks += 1;
            let exp: Vec<u32> = expected.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            if lst != exp {
                die(&format!("{kind} {seed}: shuffle({n}) {lst:?} != {exp:?}"));
            }
        }
        let mut r = make(kind, seed);
        for _ in 0..10 {
            r.random();
        }
        let st = r.getstate();
        let words: Vec<u64> = v["state_words"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        checks += 1;
        if st.words().iter().map(|&x| x as u64).ne(words.iter().copied()) {
            die(&format!("{kind} {seed}: state words differ"));
        }
        let mut r2 = Random::from_i128(0);
        r2.setstate(&st);
        for (i, expected) in v["after_restore"].as_array().unwrap().iter().enumerate() {
            let got = bits(r2.random());
            checks += 1;
            if got != expected.as_u64().unwrap() {
                die(&format!("{kind} {seed}: after_restore[{i}] differs"));
            }
        }
    }
    println!("OK: {checks} checks, {n_int} int seeds, {n_str} str seeds");
}
