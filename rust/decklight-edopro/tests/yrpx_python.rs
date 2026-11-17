//! The Rust parser reads a real LZMA-compressed `.yrpX` written by Python (Multirole layout).

use decklight_edopro::yrp::{core_messages, parse_yrpx};
use serde_json::Value;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn parses_python_compressed_yrpx() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/yrpx.json");
    let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let data = hex(f["hex"].as_str().unwrap());
    let x = parse_yrpx(&data).unwrap();
    assert_eq!(x.timestamp, f["timestamp"].as_u64().unwrap());
    let names: Vec<Vec<String>> = f["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_string()).collect())
        .collect();
    assert_eq!(x.names, names);
    assert_eq!(x.flags, f["flags"].as_str().unwrap().parse::<u64>().unwrap());
    let y = x.yrp.as_ref().unwrap();
    let seed: Vec<u64> = f["seed"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().parse().unwrap()).collect();
    assert_eq!(y.seed.to_vec(), seed);
    assert_eq!((y.lp, y.hand, y.draw), (f["lp"].as_u64().unwrap() as u32, f["hand"].as_u64().unwrap() as u32, f["draw"].as_u64().unwrap() as u32));
    let decks: Vec<Vec<(Vec<u32>, Vec<u32>)>> = f["decks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|team| {
            team.as_array()
                .unwrap()
                .iter()
                .map(|d| {
                    let v = d.as_array().unwrap();
                    let codes = |i: usize| v[i].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                    (codes(0), codes(1))
                })
                .collect()
        })
        .collect();
    assert_eq!(y.decks, decks);
    let responses: Vec<Vec<u8>> =
        f["responses"].as_array().unwrap().iter().map(|r| hex(r.as_str().unwrap())).collect();
    assert_eq!(y.responses, responses);
    let msgs: Vec<Vec<u8>> =
        f["core_messages"].as_array().unwrap().iter().map(|m| hex(m.as_str().unwrap())).collect();
    assert_eq!(core_messages(&x), msgs);
}
