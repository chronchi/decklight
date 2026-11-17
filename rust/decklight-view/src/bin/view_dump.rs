//! Rust side of the A2a view parity harness: the same JSON lines as `scripts/app/view_dump.py`.
//!
//!     view_dump --pairs A:B C:D --games N --seed S --choice-seed C --out FILE
//!               [--cdb-dir DIR] [--scripts-dir DIR] [--extra-dir DIR]... [--game-offset N]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use decklight_cards::texts::Texts;
use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
use decklight_view::{Ctx, DuelStats};
use serde_json::{Value, json};
use ygoenv_sys::{DuelConfig, Env, Obs, TCG_FLAGS};

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn die(msg: &str) -> ! {
    eprintln!("view_dump: {msg}");
    std::process::exit(1);
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn resolve_deck(s: &str) -> PathBuf {
    let p = PathBuf::from(s);
    if p.exists() {
        return p;
    }
    let pool = root().join("decks/pool").join(format!("{s}.ydk"));
    if pool.exists() { pool } else { p }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut games, mut seed, mut cseed, mut offset) = (1u64, 0u64, 0u64, 0u64);
    let (mut d0, mut d1) = (None::<PathBuf>, None::<PathBuf>);
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut cdb = root().join("third_party/BabelCDB");
    let mut scripts = root().join("third_party/CardScripts");
    let strings = root().join("third_party/edopro-strings.conf");
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut out_path = root().join("data/cache/app/view_rs.jsonl");
    let mut i = 0usize;
    while i < argv.len() {
        let a = argv[i].clone();
        i += 1;
        let mut val = || {
            let v = argv.get(i).cloned().unwrap_or_else(|| die(&format!("{a} needs a value")));
            i += 1;
            v
        };
        match a.as_str() {
            "--deck0" => d0 = Some(PathBuf::from(val())),
            "--deck1" => d1 = Some(PathBuf::from(val())),
            "--pairs" => {
                while i < argv.len() && !argv[i].starts_with("--") {
                    let p = argv[i].clone();
                    i += 1;
                    let (x, y) = p.split_once(':').unwrap_or_else(|| die("--pairs entry needs deck0:deck1"));
                    pairs.push((resolve_deck(x), resolve_deck(y)));
                }
            }
            "--games" => games = val().parse().unwrap_or_else(|_| die("--games")),
            "--seed" => seed = val().parse().unwrap_or_else(|_| die("--seed")),
            "--choice-seed" => cseed = val().parse().unwrap_or_else(|_| die("--choice-seed")),
            "--game-offset" => offset = val().parse().unwrap_or_else(|_| die("--game-offset")),
            "--cdb-dir" => cdb = PathBuf::from(val()),
            "--scripts-dir" => scripts = PathBuf::from(val()),
            "--extra-dir" => extra.push(PathBuf::from(val())),
            "--out" => out_path = PathBuf::from(val()),
            _ => die(&format!("unknown argument {a}")),
        }
    }
    if let (Some(a), Some(b)) = (d0, d1) {
        pairs = vec![(a, b)];
    }
    if pairs.is_empty() {
        die("--pairs or --deck0/--deck1 is required");
    }
    let decks: Vec<(decklight_cards::Deck, decklight_cards::Deck)> = pairs
        .iter()
        .map(|(a, b)| {
            let read = |p: &Path| {
                parse_ydk(&std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))))
            };
            (read(a), read(b))
        })
        .collect();
    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&e));
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    let engine = build_engine(&cards, &index);
    if cards.is_empty() || index.is_empty() {
        die("ALERT view_dump: no cards or no scripts loaded");
    }
    let texts = Texts::load(&cdb, &extra, &strings).unwrap_or_else(|e| die(&e));
    let card_types: BTreeMap<u32, u32> = cards.iter().map(|(c, d)| (*c, d.type_)).collect();
    let ctx = Ctx::new(texts, card_types, 8000);

    let mut obs = Obs::zeroed();
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| die(&format!("{}: {e}", parent.display())));
    }
    let file = std::fs::File::create(&out_path).unwrap_or_else(|e| die(&format!("{}: {e}", out_path.display())));
    // Stream every line as it is built: a 100-game run is GBs of JSON (it was held in memory and written
    // unbuffered, 1.4 M write calls per game; NOTEBOOK 2026-11-01).
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    let mut n_lines = 0usize;
    let mut emit = |v: Value| {
        serde_json::to_writer(&mut out, &v).unwrap_or_else(|e| die(&format!("serialize: {e}")));
        out.write_all(b"\n").unwrap_or_else(|e| die(&format!("write: {e}")));
        n_lines += 1;
    };
    for i in 0..games {
        let g = offset + i;
        let (deck0, deck1) = &decks[(g % decks.len() as u64) as usize];
        let s = seed + g;
        let (mut m0, mut m1) = (deck0.main.clone(), deck1.main.clone());
        shuffle_main(&mut m0, s, 0);
        shuffle_main(&mut m1, s, 1);
        let cfg = DuelConfig {
            main0: m0,
            extra0: deck0.extra.clone(),
            main1: m1,
            extra1: deck1.extra.clone(),
            seed: s,
            flags: TCG_FLAGS,
            lp: 8000,
            hand: 5,
            draw: 1,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        };
        let h = (g % 2) as i32;
        let mut env = Env::new(&engine, &cfg, false).unwrap_or_else(|e| die(&e));
        env.set_record_trace(true);
        let mut rng = SplitMix(cseed ^ g);
        let mut stats = DuelStats::new(h, 8000);
        let (mut cue_sent, mut log_sent, mut prompt_id, mut last_seq) = (0usize, 0usize, 0i64, -1i64);
        let mut k = 0usize;
        env.reset(&mut obs).unwrap_or_else(|e| die(&e));
        while env.decision().is_some() {
            let dec = env.decision().unwrap();
            let seq = env.decision_seq();
            if seq != last_seq {
                prompt_id += 1;
                last_seq = seq;
            }
            let view = ctx.build_view(&env, h, false);
            let prompt = if dec.player == h { ctx.build_prompt(&env, h, prompt_id) } else { None };
            let events = ctx.build_events(&env, h, log_sent);
            let (cues, new_len) = decklight_view::build_cues(&env, &ctx, h, cue_sent, false);
            cue_sent = new_len;
            stats.feed_cues(&cues);
            stats.note_view(Some(&view));
            emit(json!({
                "g": g,
                "k": k,
                "h": h,
                "view": view,
                "prompt": prompt,
                "events": events,
                "cues": cues,
                "stats": stats.stats_message(false),
            }));
            log_sent = env.history_log().len();
            let n = env.num_options();
            if n <= 0 {
                die("ALERT view_dump: a pending decision with no options");
            }
            let idx = (rng.next() % n as u64) as i32;
            let chain = dec.msg_type == decklight_core::constants::MSG_SELECT_CHAIN as i32 && dec.player == h;
            let had = chain && dec.options.iter().any(|o| o.kind == 6);
            let used = chain && dec.options.get(idx as usize).is_some_and(|o| o.kind == 6);
            env.step(idx, &mut obs).unwrap_or_else(|e| die(&e));
            if chain {
                stats.note_chain_window(had, used);
            }
            k += 1;
        }
        let view = ctx.build_view(&env, h, false);
        let events = ctx.build_events(&env, h, log_sent);
        let (cues, _) = decklight_view::build_cues(&env, &ctx, h, cue_sent, false);
        stats.feed_cues(&cues);
        stats.note_view(Some(&view));
        emit(json!({
            "g": g,
            "k": "end",
            "h": h,
            "view": view,
            "prompt": Value::Null,
            "events": events,
            "cues": cues,
            "stats": stats.stats_message(false),
        }));
        let view_all = ctx.build_view(&env, h, true);
        let (cues_all, _) = decklight_view::build_cues(&env, &ctx, h, 0, true);
        emit(json!({"g": g, "end": true, "view_all": view_all, "cues_all": cues_all}));
        eprintln!("game {g}: decisions {k}");
    }
    out.flush().unwrap_or_else(|e| die(&format!("{}: {e}", out_path.display())));
    eprintln!("wrote {n_lines} lines to {}", out_path.display());
}
