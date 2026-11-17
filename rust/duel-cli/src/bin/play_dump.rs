//! play_dump --cdb-dir DIR --scripts-dir DIR --deck0 A.ydk --deck1 B.ydk [--pairs A:B ...]
//!             [--games N] [--seed S] [--choice-seed C] [--extra-dir DIR]...
//!
//! Decision-level dump matching scripts/app/play_dump.py (PyEnv, auto_single=false) line for line:
//! one JSON line per decision, then one end line per game. Keys, order and number formatting match
//! Python's json.dumps(obj, separators=(",", ":")) exactly, so the outputs compare as strings.
use std::path::PathBuf;

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
use sha2::{Digest, Sha256};
use ygoenv_sys::{DuelConfig, Env, Obs, TCG_FLAGS};

const CARD_FLAGS: u32 = 0x1 | 0x2 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x10000
    | 0x40000 | 0x80000 | 0x100000 | 0x200000 | 0x400000 | 0x800000;
const LOCS: [u32; 7] = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40];
const HINT_A: u8 = 0xA0;
const HINT_B: u8 = 0xA5;

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

fn sha_hex(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(char::from_digit((x >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((x & 15) as u32, 16).unwrap());
    }
    s
}

fn canonical(mut trace: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(trace.len());
    let mut run: Vec<Vec<u8>> = Vec::new();
    for m in trace.drain(..) {
        if !m.is_empty() && (m[0] == HINT_A || m[0] == HINT_B) {
            run.push(m);
            continue;
        }
        run.sort();
        out.append(&mut run);
        out.push(m);
    }
    run.sort();
    out.append(&mut run);
    out
}

fn trace_hash(trace: &[Vec<u8>]) -> String {
    let canon = canonical(trace.to_vec());
    let mut h = Sha256::new();
    for m in &canon {
        h.update((m.len() as u32).to_le_bytes());
        h.update(m);
    }
    hex(&h.finalize())
}

fn locs_hash(env: &Env) -> String {
    let mut h = Sha256::new();
    for p in [0u8, 1] {
        for loc in LOCS {
            let raw = env.query_location(p, loc, CARD_FLAGS).unwrap_or_else(|e| die(&e));
            h.update((raw.len() as u32).to_le_bytes());
            h.update(&raw);
        }
    }
    hex(&h.finalize())
}

fn hist_hash(env: &Env) -> String {
    let mut h = Sha256::new();
    for row in env.history_log() {
        for x in row {
            h.update(x.to_le_bytes());
        }
    }
    hex(&h.finalize())
}

fn esc(s: &str) -> String {
    // Hex digests only; keep the general case correct.
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn die(msg: &str) -> ! {
    eprintln!("play_dump: {msg}");
    std::process::exit(1);
}

fn resolve_deck(s: &str) -> PathBuf {
    let p = PathBuf::from(s);
    if p.exists() {
        return p;
    }
    let mut pool = PathBuf::from("decks/pool");
    pool.push(format!("{s}.ydk"));
    if pool.exists() {
        return pool;
    }
    // Bare filename without .ydk in the pool dir.
    let mut pool2 = PathBuf::from("decks/pool");
    pool2.push(s);
    if pool2.exists() {
        return pool2;
    }
    die(&format!("deck not found: {s}"));
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut cdb, mut scripts, mut d0, mut d1) = (None, None, None, None);
    let mut pairs: Vec<String> = Vec::new();
    let (mut games, mut seed, mut cseed, mut offset) = (1u64, 0u64, 0u64, 0u64);
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut need = || {
            i += 1;
            if i >= argv.len() {
                die(&format!("{a} needs a value"));
            }
            argv[i].clone()
        };
        match a {
            "--cdb-dir" => cdb = Some(PathBuf::from(need())),
            "--scripts-dir" => scripts = Some(PathBuf::from(need())),
            "--deck0" => d0 = Some(need()),
            "--deck1" => d1 = Some(need()),
            "--pairs" => {
                // One or more A:B values follow until the next flag.
                let mut any = false;
                while i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    i += 1;
                    pairs.push(argv[i].clone());
                    any = true;
                }
                if !any {
                    die("--pairs needs a value");
                }
            }
            "--games" => games = need().parse().unwrap_or_else(|_| die("--games")),
            "--seed" => seed = need().parse().unwrap_or_else(|_| die("--seed")),
            "--choice-seed" => cseed = need().parse().unwrap_or_else(|_| die("--choice-seed")),
            "--game-offset" => offset = need().parse().unwrap_or_else(|_| die("--game-offset")),
            "--extra-dir" => extra.push(PathBuf::from(need())),
            s => die(&format!("unknown argument {s}")),
        }
        i += 1;
    }
    // The bare-arg fallback above misroutes: if --pairs was given, bare pair args after other flags
    // land in `pairs`, but so would stray positionals. Require explicit pairs or deck0/deck1.
    let pair_list: Vec<(PathBuf, PathBuf)> = if !pairs.is_empty() {
        pairs
            .iter()
            .map(|it| {
                let (a, b) = it.split_once(':').unwrap_or_else(|| die(&format!("--pairs entry {it:?} needs A:B")));
                (resolve_deck(a), resolve_deck(b))
            })
            .collect()
    } else if d0.is_some() && d1.is_some() {
        vec![(resolve_deck(&d0.unwrap()), resolve_deck(&d1.unwrap()))]
    } else {
        die("--deck0/--deck1 or --pairs is required");
    };

    let cdb = cdb.unwrap_or_else(|| PathBuf::from("third_party/BabelCDB"));
    let scripts = scripts.unwrap_or_else(|| PathBuf::from("third_party/CardScripts"));
    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&e));
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    let engine = build_engine(&cards, &index);
    if cards.is_empty() || index.is_empty() {
        die("ALERT play_dump: no cards or no scripts loaded");
    }

    let read = |p: &PathBuf| parse_ydk(&std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))));
    let decks: Vec<_> = pair_list.iter().map(|(a, b)| (read(a), read(b))).collect();

    let mut obs = Obs::zeroed();
    for i in 0..games {
        let g = offset + i;
        let (deck0, deck1) = &decks[(g as usize) % decks.len()];
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
        let mut env = Env::new(&engine, &cfg, false).unwrap_or_else(|e| die(&e));
        env.set_record_trace(true);
        let mut rng = SplitMix(cseed ^ g);
        let mut pending = env.reset(&mut obs).unwrap_or_else(|e| die(&e));
        let mut k: u64 = 0;
        while pending {
            let d = env.decision().expect("pending decision");
            let st = env.duel_state().unwrap_or_else(|e| die(&e));
            let trace: Vec<Vec<u8>> = (0..env.trace_len()).map(|i| env.trace_msg(i)).collect();
            let sel: Vec<String> = d
                .selected
                .iter()
                .map(|c| format!("[{},{},{},{},{}]", c.code, c.controller, c.location, c.sequence, c.position))
                .collect();
            let opts: Vec<String> = d
                .options
                .iter()
                .map(|o| {
                    format!(
                        "[{},{},{},{},{},{},{},{},{},{}]",
                        o.kind, o.has_card, o.code, o.controller, o.location, o.sequence, o.position, o.opt_code,
                        o.desc, o.arg
                    )
                })
                .collect();
            let n = d.options.len();
            if n == 0 {
                die(&format!("ALERT play_dump: game {g} decision {k}: no legal option"));
            }
            let idx = (rng.next() % n as u64) as i32;
            println!(
                "{{\"g\":{g},\"k\":{k},\"player\":{},\"msg_type\":{},\"step\":{},\"selected\":[{}],\"options\":[{}],\"pending\":{},\"turn\":{},\"turn_player\":{},\"first_player\":{},\"phase\":{},\"select_hint\":{},\"trace_len\":{},\"trace\":{},\"field\":{},\"locs\":{},\"hist\":{},\"choice\":{idx}}}",
                d.player,
                d.msg_type,
                d.step,
                sel.join(","),
                opts.join(","),
                esc(&sha_hex(&env.pending_msg())),
                st.turn,
                st.turn_player,
                st.first_player,
                st.phase,
                st.select_hint,
                trace.len(),
                esc(&trace_hash(&trace)),
                esc(&sha_hex(&env.query_field().unwrap_or_else(|e| die(&e)))),
                esc(&locs_hash(&env)),
                esc(&hist_hash(&env)),
            );
            pending = env.step(idx, &mut obs).unwrap_or_else(|e| die(&e));
            k += 1;
        }
        let st = env.duel_state().unwrap_or_else(|e| die(&e));
        let trace: Vec<Vec<u8>> = (0..env.trace_len()).map(|i| env.trace_msg(i)).collect();
        let ch = env.choices();
        let chs: Vec<String> = ch.iter().map(|c| c.to_string()).collect();
        println!(
            "{{\"g\":{g},\"end\":true,\"winner\":{},\"win_reason\":{},\"choices\":[{}],\"decisions\":{},\"trace_len\":{},\"trace\":{}}}",
            st.winner,
            st.win_reason,
            chs.join(","),
            env.decisions(),
            trace.len(),
            esc(&trace_hash(&trace)),
        );
        for e in env.errors().iter().take(3) {
            eprintln!("lua error (game {g}): {e}");
        }
        if !env.errors().is_empty() {
            die(&format!("ALERT play_dump: {} Lua/script errors", env.errors().len()));
        }
    }
}
