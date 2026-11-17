//! duel-cli --cdb-dir DIR --scripts-dir DIR --deck0 A.ydk --deck1 B.ydk [--games N] [--seed S] [--choice-seed C]
//!          [--extra-dir DIR]... [--max-decisions N]
//!
//! Game g uses duel seed S+g (decks shuffled by it, as DuelConfig.main_of) and picks each option with a splitmix64
//! stream seeded by C^g: index = next() % num_options. The hash is FNV-1a 64 over, for every observation, the
//! fields in `to_dict` order (cards, card_mask, global, chain, chain_mask, actions, action_mask, history,
//! history_mask, hist_long, hist_long_mask, card_known, known_code, priv_code; little-endian, masks as bytes),
//! then the chosen index as 4 LE bytes. scripts/app/duel_hash.py computes the same over Python's CppEnv.
use std::path::PathBuf;
use std::time::Instant;

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
use ygoenv_sys::{DuelConfig, Env, Obs, TCG_FLAGS};

struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn i32s(&mut self, v: &[i32]) {
        for x in v {
            self.bytes(&x.to_le_bytes());
        }
    }
    fn obs(&mut self, o: &Obs) {
        self.i32s(&o.cards);
        self.bytes(&o.card_mask);
        self.i32s(&o.global);
        self.i32s(&o.chain);
        self.bytes(&o.chain_mask);
        self.i32s(&o.actions);
        self.bytes(&o.action_mask);
        self.i32s(&o.history);
        self.bytes(&o.history_mask);
        self.i32s(&o.hist_long);
        self.bytes(&o.hist_long_mask);
        self.bytes(&o.card_known);
        self.i32s(&o.known_code);
        self.i32s(&o.priv_code);
    }
}

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

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut cdb, mut scripts, mut d0, mut d1) = (None, None, None, None);
    let (mut games, mut seed, mut cseed, mut maxd) = (1u64, 0u64, 0u64, 3000i32);
    let mut extra: Vec<PathBuf> = Vec::new();
    while let Some(a) = args.next() {
        let mut v = || args.next().unwrap_or_else(|| die(&format!("{a} needs a value")));
        match a.as_str() {
            "--cdb-dir" => cdb = Some(PathBuf::from(v())),
            "--scripts-dir" => scripts = Some(PathBuf::from(v())),
            "--deck0" => d0 = Some(PathBuf::from(v())),
            "--deck1" => d1 = Some(PathBuf::from(v())),
            "--games" => games = v().parse().unwrap_or_else(|_| die("--games")),
            "--seed" => seed = v().parse().unwrap_or_else(|_| die("--seed")),
            "--choice-seed" => cseed = v().parse().unwrap_or_else(|_| die("--choice-seed")),
            "--max-decisions" => maxd = v().parse().unwrap_or_else(|_| die("--max-decisions")),
            "--extra-dir" => extra.push(PathBuf::from(v())),
            _ => die(&format!("unknown argument {a}")),
        }
    }
    let (cdb, scripts) = (cdb.unwrap_or_else(|| die("--cdb-dir")), scripts.unwrap_or_else(|| die("--scripts-dir")));
    let read = |p: &PathBuf| parse_ydk(&std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))));
    let (deck0, deck1) = (read(&d0.unwrap_or_else(|| die("--deck0"))), read(&d1.unwrap_or_else(|| die("--deck1"))));

    let t0 = Instant::now();
    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&e));
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    let engine = build_engine(&cards, &index);
    eprintln!("engine: {} cards, {} scripts, {:.2}s", engine.num_cards(), index.len(), t0.elapsed().as_secs_f64());
    if cards.is_empty() || index.is_empty() {
        die("ALERT duel-cli: no cards or no scripts loaded");
    }

    let mut obs = Obs::zeroed();
    let mut bad = 0;
    for g in 0..games {
        let s = seed + g;
        let (mut m0, mut m1) = (deck0.main.clone(), deck1.main.clone());
        shuffle_main(&mut m0, s, 0);
        shuffle_main(&mut m1, s, 1);
        let cfg = DuelConfig {
            main0: m0, extra0: deck0.extra.clone(), main1: m1, extra1: deck1.extra.clone(), seed: s,
            flags: TCG_FLAGS, lp: 8000, hand: 5, draw: 1,
            seed_state: None, team_order: [0, 1], edopro_add: false,
        };
        let t = Instant::now();
        let mut env = Env::new(&engine, &cfg, true).unwrap_or_else(|e| die(&e));
        let mut rng = SplitMix(cseed ^ g);
        let mut h = Fnv::new();
        let mut steps = 0;
        let mut pending = env.reset(&mut obs).unwrap_or_else(|e| die(&e));
        while pending && steps < maxd {
            h.obs(&obs);
            let n = env.num_options();
            if n <= 0 {
                die("ALERT duel-cli: a pending decision with no options");
            }
            let idx = (rng.next() % n as u64) as i32;
            h.bytes(&idx.to_le_bytes());
            pending = env.step(idx, &mut obs).unwrap_or_else(|e| die(&e));
            steps += 1;
        }
        let errs = env.errors();
        bad += errs.len();
        println!(
            "{{\"game\":{g},\"seed\":{s},\"decisions\":{steps},\"done\":{},\"winner\":{},\"hash\":\"{:016x}\",\"lua_errors\":{},\"ms\":{}}}",
            env.done(), env.winner(), h.0, errs.len(), t.elapsed().as_millis()
        );
        for e in errs.iter().take(3) {
            eprintln!("lua error (game {g}): {e}");
        }
    }
    if bad > 0 {
        eprintln!("ALERT duel-cli: {bad} Lua/script errors");
        std::process::exit(2);
    }
}

fn die(msg: &str) -> ! {
    eprintln!("duel-cli: {msg}");
    std::process::exit(1);
}
