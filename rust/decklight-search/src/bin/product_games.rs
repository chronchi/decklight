//! Product games in Rust: S = the product (honest k4 s4 h1, det-skip, ho) at parity seats, R = a splitmix64
//! scripted human fed as one-hot logits (unguarded), exactly as `ProductDriver` plays. The Python oracle is
//! `scripts/app/product_games_py.py`; compare with `scripts/app/product_games_compare.py`.
//!
//! `--io-hash-out` records a hash of every ORT policy/value input batch and output (gate 1).

use std::io::Write;
use std::path::{Path, PathBuf};

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
use decklight_onnx::OnnxNets;
use decklight_search::play::{search_game, PlayParams, PlayerNets};
use decklight_search::window::{Nets, Prof, WindowSearch};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::json;
use ygoenv_sys::{DuelConfig, Obs, TCG_FLAGS};

struct SplitMix(u64);

impl SplitMix {
    fn new(s: u64) -> SplitMix {
        SplitMix(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

struct ProductNets {
    onnx: OnnxNets,
    human: SplitMix,
    io: Option<(std::io::BufWriter<std::fs::File>, usize, u64)>,
}

impl ProductNets {
    fn drain_io(&mut self) {
        let recs = self.onnx.take_io_log();
        let Some((w, idx, g)) = self.io.as_mut() else { return };
        for rec in recs {
            let line = json!({
                "kind": "io", "g": *g, "idx": *idx, "call": rec.kind, "batch": rec.batch,
                "input": rec.input_hex(), "output": rec.output_hex(), "shape": rec.output_shape,
            });
            writeln!(w, "{line}").unwrap();
            *idx += 1;
        }
        w.flush().ok();
    }
}

impl Nets for ProductNets {
    fn forward(&mut self, obs: &[&Obs]) -> Result<Vec<Vec<f32>>, String> {
        let out = self.onnx.policy_logits(obs);
        self.drain_io();
        out
    }
    fn forward_threads(&mut self, obs: &[&Obs], threads: usize) -> Result<Vec<Vec<f32>>, String> {
        let out = self.onnx.policy_logits_threads(obs, threads);
        self.drain_io();
        out
    }
    fn value(&mut self, obs: &[&Obs]) -> Result<Vec<f32>, String> {
        let out = self.onnx.values(obs);
        self.drain_io();
        out
    }
}

impl PlayerNets for ProductNets {
    fn human(&mut self, obs: &Obs) -> Result<i32, String> {
        let n = obs.action_mask.iter().filter(|&&x| x != 0).count();
        if n == 0 {
            return Err("human: no legal option".into());
        }
        Ok((self.human.next() % n as u64) as i32)
    }
}

fn die(msg: &str) -> ! {
    eprintln!("product_games: {msg}");
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

fn bits(x: f64) -> u64 {
    x.to_bits()
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut games, mut seed, mut cseed, mut offset) = (1u64, 0u64, 0u64, 0u64);
    let mut pairs: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    let mut cdb = root().join("third_party/BabelCDB");
    let mut scripts = root().join("third_party/CardScripts");
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut onnx = root().join("data/cache/onnx/r1");
    let mut out_path = root().join("data/cache/app/product_rs.jsonl.gz");
    let mut io_path: Option<PathBuf> = None;
    let mut replay_threads = 1usize;
    let mut search_threads = 1usize;
    let mut prof_on = false;
    let mut ort_threads = 1usize;
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
                    pairs.push((p.clone(), resolve_deck(x), resolve_deck(y)));
                }
            }
            "--games" => games = val().parse().unwrap_or_else(|_| die("--games")),
            "--seed" => seed = val().parse().unwrap_or_else(|_| die("--seed")),
            "--choice-seed" => cseed = val().parse().unwrap_or_else(|_| die("--choice-seed")),
            "--game-offset" => offset = val().parse().unwrap_or_else(|_| die("--game-offset")),
            "--cdb-dir" => cdb = PathBuf::from(val()),
            "--scripts-dir" => scripts = PathBuf::from(val()),
            "--extra-dir" => extra.push(PathBuf::from(val())),
            "--onnx" => onnx = PathBuf::from(val()),
            "--out" => out_path = PathBuf::from(val()),
            "--io-hash-out" => io_path = Some(PathBuf::from(val())),
            "--prof" => prof_on = true,
            "--ort-threads" => ort_threads = val().parse().unwrap_or_else(|_| die("--ort-threads")),
            "--replay-threads" => replay_threads = val().parse().unwrap_or_else(|_| die("--replay-threads")),
            "--search-threads" => search_threads = val().parse().unwrap_or_else(|_| die("--search-threads")),
            _ => die(&format!("unknown argument {a}")),
        }
    }
    if pairs.is_empty() {
        die("--pairs is required");
    }
    let decks: Vec<_> = pairs
        .iter()
        .map(|(_, a, b)| {
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
        die("ALERT product_games: no cards or no scripts loaded");
    }
    let mut onnx_nets = OnnxNets::load_with_threads(&onnx, &root(), ort_threads).unwrap_or_else(|e| die(&e.to_string()));
    if io_path.is_some() {
        onnx_nets.enable_io_log();
    }

    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let file = std::fs::File::create(&out_path).unwrap_or_else(|e| die(&format!("{}: {e}", out_path.display())));
    let mut w = GzEncoder::new(std::io::BufWriter::new(file), Compression::default());
    let mut io_writer: Option<std::io::BufWriter<std::fs::File>> = match &io_path {
        Some(path) => {
            if let Some(p) = path.parent() { std::fs::create_dir_all(p).ok(); }
            let f = std::fs::File::create(path).unwrap_or_else(|e| die(&format!("{}: {e}", path.display())));
            Some(std::io::BufWriter::new(f))
        }
        None => None,
    };

    let mut n_searched = 0usize;
    let mut n_overrides = 0usize;
    let mut n_branches = 0usize;
    let mut n_capped = 0usize;
    let mut n_det_skip = 0usize;
    let mut walls_py: Vec<f64> = Vec::new();
    let mut prof = Prof::default();
    let (mut det_s, mut lat_search_s, mut lat_raw_s) = (0.0f64, 0.0f64, 0.0f64);
    for gi in 0..games {
        let g = offset + gi;
        let (name, _, _) = &pairs[(g as usize) % pairs.len()];
        let (deck0, deck1) = &decks[(g as usize) % decks.len()];
        let s = seed + g;
        let mut m0 = deck0.main.clone();
        let mut m1 = deck1.main.clone();
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
        let s_seat = (g % 2) as i32;
        let mut search = WindowSearch::new(&engine, 4, 1, 4, 1.0, 300, s, None, true);
        if replay_threads < 1 {
            die("--replay-threads must be >= 1");
        }
        if search_threads < 1 {
            die("--search-threads must be >= 1");
        }
        search.replay_threads = replay_threads;
        search.search_threads = search_threads;
        if prof_on {
            search.prof = Some(Prof::default());
        }
        let mut nets = ProductNets {
            onnx: onnx_nets,
            human: SplitMix::new(cseed ^ g),
            io: io_writer.map(|w| (w, 0usize, g)),
        };
        let mut params = PlayParams::product(s_seat, s);
        params.search_threads = search_threads;
        let t0 = std::time::Instant::now();
        let res = search_game(&mut search, &cfg, &mut nets, &params).unwrap_or_else(|e| die(&e.to_string()));
        let game_wall = t0.elapsed().as_secs_f64();
        if let Some((w, _, _)) = nets.io.as_mut() {
            w.flush().ok();
        }
        io_writer = nets.io.take().map(|(w, _, _)| w);
        onnx_nets = nets.onnx;
        n_searched += res.n_searched;
        n_overrides += res.n_override;
        n_branches += res.n_branches;
        n_capped += res.n_capped;
        n_det_skip += res.det_skipped;
        walls_py.extend(res.lat_search.iter().copied());
        det_s += res.det_s - res.det_skip_s; // searched moves only, like lat_search
        lat_search_s += res.lat_search.iter().sum::<f64>();
        lat_raw_s += res.lat_raw.iter().sum::<f64>();
        if let Some(p) = &search.prof {
            prof.replay_s += p.replay_s;
            prof.step_s += p.step_s;
            prof.steps += p.steps;
            prof.fwd_s += p.fwd_s;
            prof.fwd_calls += p.fwd_calls;
            prof.fwd_obs += p.fwd_obs;
            prof.value_s += p.value_s;
            prof.value_calls += p.value_calls;
            prof.value_obs += p.value_obs;
            prof.rollout_s += p.rollout_s;
        }

        let n_prefix = params.prefix.len();
        let (mut si, mut ri) = (0usize, 0usize);
        for d in &res.decisions {
            let wall = if d.k < n_prefix {
                None
            } else if d.searched {
                let v = res.lat_search.get(si).copied();
                si += 1;
                v
            } else {
                let v = res.lat_raw.get(ri).copied();
                ri += 1;
                v
            };
            w.write_all(
                format!(
                    "{}\n",
                    json!({
                        "kind": "decision", "g": g, "k": d.k, "actor": d.actor, "turn": d.turn,
                        "n": d.n_options, "chosen": d.chosen, "searched": d.searched,
                        "wall_bits": wall.map(bits),
                    })
                )
                .as_bytes(),
            )
            .unwrap();
        }
        for e in &res.log {
            let samples: Vec<Vec<u64>> = e
                .samples
                .iter()
                .map(|col| col.iter().map(|&x| bits(decklight_search::play::round4(x))).collect())
                .collect();
            let scores_all: serde_json::Map<String, serde_json::Value> = e
                .scores_all
                .iter()
                .map(|(k, col)| {
                    (
                        k.clone(),
                        json!(col.iter().map(|&x| bits(x)).collect::<Vec<u64>>()),
                    )
                })
                .collect();
            let ho = e.ho.as_ref().map(|h| {
                json!({
                    "status": h.status, "gap_bits": bits(h.gap),
                    "se_bits": h.se.map(bits), "n": h.n, "challenger": h.challenger, "rounds": h.rounds,
                })
            });
            w.write_all(
                format!(
                    "{}\n",
                    json!({
                        "kind": "log", "g": g, "k": e.k, "seat": e.seat, "kind2": e.kind, "turn": e.turn,
                        "cands": e.cands,
                        "scores_bits": e.scores.iter().map(|&x| bits(x)).collect::<Vec<u64>>(),
                        "scores_all_bits": scores_all,
                        "chosen": e.chosen, "played": e.played, "best": e.best,
                        "samples_bits": samples,
                        "ho": ho,
                    })
                )
                .as_bytes(),
            )
            .unwrap();
        }
        w.write_all(
            format!(
                "{}\n",
                json!({
                    "kind": "game", "g": g, "pair": name, "s_seat": s_seat,
                    "seed": s, "choice_seed": cseed, "ws_seed": s, "game_seed": s,
                    "winner": res.winner, "truncated": res.truncated, "turns": res.turns,
                    "choices": res.choices, "actors": res.actors, "decision_turns": res.decision_turns,
                    "n_searched": res.n_searched, "n_override": res.n_override,
                    "n_branches": res.n_branches, "n_capped": res.n_capped,
                    "det_proposals": res.det_proposals, "det_accepted": res.det_accepted,
                    "det_fallbacks": res.det_fallbacks, "det_skipped": res.det_skipped,
                    "search_cut": res.search_cut,
                    "lat_raw_bits": res.lat_raw.iter().map(|&x| bits(x)).collect::<Vec<u64>>(),
                    "lat_search_bits": res.lat_search.iter().map(|&x| bits(x)).collect::<Vec<u64>>(),
                    "lat_turn_bits": res.lat_turn.iter().map(|&x| bits(x)).collect::<Vec<u64>>(),
                    "game_wall_bits": bits(game_wall),
                })
            )
            .as_bytes(),
        )
        .unwrap();
        eprintln!(
            "product_games: g={g} pair={name} S=seat{s_seat} winner={:?} decisions={} searched={} overrides={} det_skip={} wall={game_wall:.1}s",
            res.winner, res.choices.len(), res.n_searched, res.n_override, res.det_skipped
        );
    }
    w.flush().unwrap();
    let mut inner = w.finish().unwrap();
    inner.flush().ok();
    if let Some(mut fw) = io_writer {
        fw.flush().ok();
        eprintln!("product_games: ORT I/O hashes -> {}", io_path.as_ref().map_or_else(|| "<none>".into(), |p| p.display().to_string()));
    }
    let mut sorted = walls_py.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if sorted.is_empty() { 0.0 } else { sorted.iter().sum::<f64>() / sorted.len() as f64 };
    let pct = |q: f64| -> f64 {
        if sorted.is_empty() {
            0.0
        } else {
            sorted[((sorted.len() as f64 - 1.0) * q).round() as usize]
        }
    };
    eprintln!(
        "product_games: games={games} searched={n_searched} overrides={n_overrides} branches={n_branches} \
         capped={n_capped} det_skipped={n_det_skip} s/search mean={mean:.3} p50={:.3} p90={:.3} max={:.3} -> {}",
        pct(0.5),
        pct(0.9),
        sorted.last().copied().unwrap_or(0.0),
        out_path.display()
    );
    if prof_on {
        // Buckets over the searched decisions: lat_search = det + rollouts (replay, step, fwd, value, rest) + outer.
        let rest = prof.rollout_s - prof.replay_s - prof.step_s - prof.fwd_s - prof.value_s;
        let outer = lat_search_s - det_s - prof.rollout_s;
        let pc = |x: f64| if lat_search_s > 0.0 { 100.0 * x / lat_search_s } else { 0.0 };
        eprintln!(
            "{}",
            json!({
                "prof": {
                    "searched": n_searched, "lat_search_s": lat_search_s, "lat_raw_s": lat_raw_s,
                    "det": [det_s, pc(det_s)], "replay": [prof.replay_s, pc(prof.replay_s)],
                    "step": [prof.step_s, pc(prof.step_s)], "fwd": [prof.fwd_s, pc(prof.fwd_s)],
                    "value": [prof.value_s, pc(prof.value_s)], "rollout_rest": [rest, pc(rest)],
                    "outer": [outer, pc(outer)], "steps": prof.steps, "fwd_calls": prof.fwd_calls,
                    "fwd_obs": prof.fwd_obs, "value_calls": prof.value_calls, "value_obs": prof.value_obs,
                    "replay_threads": replay_threads, "ort_threads": ort_threads,
                    "search_threads": search_threads,
                }
            })
        );
    }
}
