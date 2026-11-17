//! raw_dump: N raw-vs-raw games (fixed seeds), one gzip JSON line per decision with the obs hash, the 64 logits
//! and the chosen index. `scripts/app/raw_check.py` replays the same games in Python and compares torch argmax.

use std::io::Write;
use std::path::{Path, PathBuf};

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
use decklight_server::raw::RawPolicy;
use serde_json::json;
use ygoenv_sys::{DuelConfig, Env, Obs, TCG_FLAGS};

fn die(msg: &str) -> ! {
    eprintln!("raw_dump: {msg}");
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
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut games, mut seed) = (1u64, 0u64);
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut cdb = root().join("third_party/BabelCDB");
    let mut scripts = root().join("third_party/CardScripts");
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut onnx = root().join("data/cache/onnx/r1");
    let mut out_path = root().join("data/cache/app/raw_rs.jsonl.gz");
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
            "--pairs" => {
                while i < argv.len() && !argv[i].starts_with("--") {
                    let p = argv[i].clone();
                    i += 1;
                    let (x, y) = p.split_once(':').unwrap_or_else(|| die("--pairs entry needs a:b"));
                    pairs.push((resolve_deck(x), resolve_deck(y)));
                }
            }
            "--games" => games = val().parse().unwrap_or_else(|_| die("--games")),
            "--seed" => seed = val().parse().unwrap_or_else(|_| die("--seed")),
            "--cdb-dir" => cdb = PathBuf::from(val()),
            "--scripts-dir" => scripts = PathBuf::from(val()),
            "--extra-dir" => extra.push(PathBuf::from(val())),
            "--onnx" => onnx = PathBuf::from(val()),
            "--out" => out_path = PathBuf::from(val()),
            _ => die(&format!("unknown argument {a}")),
        }
    }
    if pairs.is_empty() {
        die("--pairs is required");
    }
    let decks: Vec<_> = pairs
        .iter()
        .map(|(a, b)| {
            let read = |p: &Path| {
                parse_ydk(&std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))))
            };
            (read(a), read(b))
        })
        .collect();
    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&e.to_string()));
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    let engine = build_engine(&cards, &index);
    if cards.is_empty() || index.is_empty() {
        die("ALERT raw_dump: no cards or no scripts loaded");
    }
    let mut policy = RawPolicy::load(&onnx, &root()).unwrap_or_else(|e| die(&e.to_string()));
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
    }
    let file = std::fs::File::create(&out_path).unwrap_or_else(|e| die(&format!("{}: {e}", out_path.display())));
    let mut enc = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut obs = Obs::zeroed();
    let mut decisions = 0usize;
    for g in 0..games {
        let (d0, d1) = &decks[(g % decks.len() as u64) as usize];
        let s = seed + g;
        let (mut m0, mut m1) = (d0.main.clone(), d1.main.clone());
        shuffle_main(&mut m0, s, 0);
        shuffle_main(&mut m1, s, 1);
        let cfg = DuelConfig {
            main0: m0,
            extra0: d0.extra.clone(),
            main1: m1,
            extra1: d1.extra.clone(),
            seed: s,
            flags: TCG_FLAGS,
            lp: 8000,
            hand: 5,
            draw: 1,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        };
        let mut env = Env::new(&engine, &cfg, false).unwrap_or_else(|e| die(&e.to_string()));
        let mut pending = env.reset(&mut obs).unwrap_or_else(|e| die(&e.to_string()));
        let mut k = 0usize;
        let mut choices: Vec<i32> = Vec::new();
        while pending {
            let dec = env.decision().unwrap();
            let logits = policy.logits(&obs).unwrap_or_else(|e| die(&e.to_string()));
            let n = env.num_options() as usize;
            let mut best = f32::NEG_INFINITY;
            let mut arg = 0i32;
            for j in 0..n {
                if obs.action_mask[j] != 0 && logits[j] > best {
                    best = logits[j];
                    arg = j as i32;
                }
            }
            let mut h = Fnv::new();
            h.obs(&obs);
            let line = json!({
                "g": g, "k": k, "seed": s, "player": dec.player, "msg_type": dec.msg_type,
                "n": n, "choice": arg, "obs": format!("{:016x}", h.0),
                "logits": logits.iter().take(64).collect::<Vec<_>>(),
            });
            serde_json::to_writer(&mut enc, &line).unwrap_or_else(|e| die(&e.to_string()));
            enc.write_all(b"\n").unwrap_or_else(|e| die(&e.to_string()));
            choices.push(arg);
            pending = env.step(arg, &mut obs).unwrap_or_else(|e| die(&e.to_string()));
            k += 1;
            decisions += 1;
        }
        let line = json!({
            "g": g, "end": true, "winner": env.winner(), "decisions": k,
            "responses": env.num_responses(), "choices": choices,
        });
        serde_json::to_writer(&mut enc, &line).unwrap_or_else(|e| die(&e.to_string()));
        enc.write_all(b"\n").unwrap_or_else(|e| die(&e.to_string()));
        eprintln!("game {g}: {k} decisions");
    }
    let mut f = enc.finish().unwrap_or_else(|e| die(&e.to_string()));
    f.flush().unwrap_or_else(|e| die(&e.to_string()));
    eprintln!("wrote {decisions} decisions to {}", out_path.display());
}