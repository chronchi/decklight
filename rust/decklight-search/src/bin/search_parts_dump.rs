//! Parity dump for the A3a search building blocks (the Rust side of `scripts/app/search_parts_dump.py`).
//!
//! Plays seeded random duels over `decks/pool` pairs (`auto_single=True`, the `play_dump.py` splitmix64 choice
//! stream), and per decision emits the masked logits, loop-guard state, argmax, softmax confidence, semantic keys,
//! merged candidates and synthetic honest-rule decisions. Every 25th decision from turn 2 on also builds a
//! `Determinizer`, samples four worlds and records its proposals, counters and RNG state.
//!
//! Synthetic logits and sample lists come from SplitMix64 streams seeded by (choice-seed, game, decision); both
//! implementations generate exactly the same f32/f64 values. Floats that matter are emitted as their bit patterns.

use std::io::Write;
use std::path::PathBuf;

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
use decklight_search::determinize::{
    hash_mains, key_rank_of_choice, Determinizer, SeenTracker, WorldConfig,
};
use decklight_search::honest::{
    decide, decide_pair, decisive_label, merge_candidates, paired_gap, semantic_key, HonestRule,
};
use decklight_search::loopguard::LoopGuard;
use decklight_search::pyrandom::Random;
use decklight_search::select::{argmax_select, masked, softmax_conf};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::{Value, json};
use ygoenv_sys::{DuelConfig, Env, Obs, TCG_FLAGS};

const INV53: f64 = 1.0 / 9_007_199_254_740_992.0;

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
    fn unit53(&mut self) -> f64 {
        (self.next() >> 11) as f64 * INV53
    }
    fn unit11(&mut self) -> f64 {
        self.unit53() * 2.0 - 1.0
    }
}

fn synth_logits(cseed: u64, g: u64, k: u64, mask: &[u8; 64]) -> [f32; 64] {
    let mut sm = SplitMix::new(cseed ^ g ^ k.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let mut out = [0f32; 64];
    for x in out.iter_mut() {
        *x = (sm.unit53() * 16.0 - 8.0) as f32;
    }
    if k % 7 == 0 {
        for i in 0..64 {
            if mask[i] != 0 {
                out[i] = 0.0;
            }
        }
    }
    out
}

fn synth_samples(cseed: u64, g: u64, k: u64) -> Vec<Vec<f64>> {
    let mut sm = SplitMix::new(cseed ^ (g << 1) ^ (k << 17) ^ 0x51ED_270B);
    let nc = 2 + (sm.next() % 2) as usize;
    let mut out = Vec::with_capacity(nc);
    for _ in 0..nc {
        let n = (sm.next() % 6) as usize;
        out.push((0..n).map(|_| sm.unit11()).collect());
    }
    out
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(char::from_digit((x >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((x & 15) as u32, 16).unwrap());
    }
    s
}

fn decision_json(d: &decklight_search::honest::Decision) -> Value {
    json!({
        "chosen": d.chosen_j,
        "challenger": d.challenger_j,
        "gap_bits": d.gap.to_bits(),
        "se_bits": d.se.to_bits(),
        "n": d.n,
        "status": d.status,
    })
}

fn die(msg: &str) -> ! {
    eprintln!("search_parts_dump: {msg}");
    std::process::exit(1);
}

fn resolve_deck(s: &str) -> PathBuf {
    let p = PathBuf::from(s);
    if p.exists() {
        return p;
    }
    let pool = PathBuf::from("decks/pool").join(format!("{s}.ydk"));
    if pool.exists() {
        return pool;
    }
    let pool2 = PathBuf::from("decks/pool").join(s);
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
    let mut out = PathBuf::from("data/cache/app/search_parts_rs.jsonl.gz");
    let mut det_every = 25u64;
    let mut det_tries = 32usize;
    let mut det_samples = 4usize;
    let mut det_min_turn = 2i32;
    let mut search_seed = 7u64;
    let mut maxd = 3000u64;
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
            "--out" => out = PathBuf::from(need()),
            "--det-every" => det_every = need().parse().unwrap_or_else(|_| die("--det-every")),
            "--det-tries" => det_tries = need().parse().unwrap_or_else(|_| die("--det-tries")),
            "--det-samples" => det_samples = need().parse().unwrap_or_else(|_| die("--det-samples")),
            "--det-min-turn" => det_min_turn = need().parse().unwrap_or_else(|_| die("--det-min-turn")),
            "--search-seed" => search_seed = need().parse().unwrap_or_else(|_| die("--search-seed")),
            "--max-decisions" => maxd = need().parse().unwrap_or_else(|_| die("--max-decisions")),
            s => die(&format!("unknown argument {s}")),
        }
        i += 1;
    }
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
    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&format!("{e}")));
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    let engine = build_engine(&cards, &index);
    if cards.is_empty() || index.is_empty() {
        die("ALERT search_parts_dump: no cards or no scripts loaded");
    }
    let read = |p: &PathBuf| parse_ydk(&std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))));
    let decks: Vec<_> = pair_list.iter().map(|(a, b)| (read(a), read(b))).collect();

    std::fs::create_dir_all(out.parent().unwrap_or_else(|| std::path::Path::new("."))).ok();
    let file = std::fs::File::create(&out).unwrap_or_else(|e| die(&format!("{}: {e}", out.display())));
    let mut w = GzEncoder::new(std::io::BufWriter::new(file), Compression::default());

    let mut obs = Obs::zeroed();
    let mut n_decisions = 0u64;
    let mut n_ties = 0u64;
    let mut n_det = 0u64;
    let mut n_det_proposals = 0u64;
    let mut n_det_accepted = 0u64;
    for gi in 0..games {
        let g = offset + gi;
        let (deck0, deck1) = &decks[(g as usize) % decks.len()];
        let s = seed + g;
        let (mut m0, mut m1) = (deck0.main.clone(), deck1.main.clone());
        shuffle_main(&mut m0, s, 0);
        shuffle_main(&mut m1, s, 1);
        let cfg = DuelConfig {
            main0: m0.clone(),
            extra0: deck0.extra.clone(),
            main1: m1.clone(),
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
        let world_cfg = WorldConfig {
            main0: m0.clone(),
            extra0: deck0.extra.clone(),
            main1: m1.clone(),
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
        let mut env = Env::new(&engine, &cfg, true).unwrap_or_else(|e| die(&format!("{e}")));
        let mut choice_rng = SplitMix::new(cseed ^ g);
        let mut trackers = [SeenTracker::new(0), SeenTracker::new(1)];
        let mut guard = LoopGuard::new();
        let mut prefix: Vec<i32> = Vec::new();
        let mut prefix_keys: Vec<([i32; 8], i32)> = Vec::new();
        let mut pending = env.reset(&mut obs).unwrap_or_else(|e| die(&format!("{e}")));
        for t in trackers.iter_mut() {
            t.update(&env);
        }
        let mut k = 0u64;
        while pending && k < maxd {
            let player = env.player() as u8;
            let turn = obs.global[2];
            let mask: [u8; 64] = obs.action_mask;
            let logits = synth_logits(cseed, g, k, &mask);
            n_decisions += 1;
            if k % 7 == 0 {
                n_ties += 1;
            }
            let mbool: Vec<bool> = mask.iter().map(|&x| x != 0).collect();
            let z = masked(&logits, &mbool).unwrap_or_else(|e| die(&format!("{e}")));
            let (z_g, key) = guard.mask(0, &obs, &z);
            let am = argmax_select(&z_g).unwrap_or_else(|| die("no options"));
            let conf = softmax_conf(&z_g);
            let legal: Vec<usize> = (0..64).filter(|&j| mask[j] != 0).collect();
            let keys: Vec<Vec<i64>> = legal.iter().map(|&j| semantic_key(&obs, j, None).as_ints()).collect();
            let merged = merge_candidates(&legal, &obs, None);
            let samples = synth_samples(cseed, g, k);
            let (m, se, n) = paired_gap(&samples[0], &samples[1]);
            let rule = HonestRule::default();
            let dec = decide(&samples, 0, &rule);
            let dec_pair = decide_pair(&samples, 0, 1, &rule);
            let label = decisive_label(&samples, &rule);
            w.write_all(
                format!(
                    "{}\n",
                    json!({
                        "kind": "parts",
                        "g": g, "k": k, "player": player, "turn": turn,
                        "msg": obs.global[21],
                        "z": z.iter().map(|v| v.to_bits()).collect::<Vec<u64>>(),
                        "zg": z_g.iter().map(|v| v.to_bits()).collect::<Vec<u64>>(),
                        "key": hex(&key),
                        "argmax": am,
                        "conf_bits": conf.to_bits(),
                        "sem_keys": keys,
                        "merged": merged,
                        "pair": {"m_bits": m.to_bits(), "se_bits": se.to_bits(), "n": n},
                        "decide": decision_json(&dec),
                        "decide_pair": decision_json(&dec_pair),
                        "label": decision_json(&label),
                    })
                )
                .as_bytes(),
            )
            .unwrap_or_else(|e| die(&format!("{e}")));

            if turn >= det_min_turn && det_every > 0 && k % det_every == 0 {
                let rng = Random::from_str(&format!("{search_seed}:{s}:{k}:det"));
                let det = Determinizer::new(
                    &engine,
                    world_cfg.clone(),
                    true,
                    player,
                    &prefix,
                    &trackers[player as usize].seen,
                    &obs,
                    rng,
                    det_tries,
                    Some(&prefix_keys),
                );
                let mut det = match det {
                    Ok(d) => d,
                    Err(e) => die(&format!("determinizer game {g} k {k}: {e}")),
                };
                let hidden0 = det.hidden_slots(0);
                let hidden1 = det.hidden_slots(1);
                let results = det.sample_many(det_samples);
                n_det += 1;
                n_det_proposals += det.proposals as u64;
                n_det_accepted += det.accepted as u64;
                let res_json: Vec<Value> = results
                    .iter()
                    .map(|r| match r {
                        None => Value::Null,
                        Some(det_res) => json!({
                            "mains": hex(&hash_mains(&det_res.world.main0, &det_res.world.main1)),
                            "resolved": det_res.resolved,
                        }),
                    })
                    .collect();
                let proposals_json: Vec<Value> = det
                    .proposals_log
                    .iter()
                    .map(|p| match p {
                        None => Value::Null,
                        Some(h) => Value::String(hex(h)),
                    })
                    .collect();
                w.write_all(
                    format!(
                        "{}\n",
                        json!({
                            "kind": "det",
                            "g": g, "k": k, "player": player, "turn": turn,
                            "base": [det.base[0], det.base[1]],
                            "hidden": [hidden0, hidden1],
                            "proposals": proposals_json,
                            "results": res_json,
                            "counts": {
                                "proposals": det.proposals,
                                "accepted": det.accepted,
                                "fallbacks": det.fallbacks,
                            },
                            "reasons": {
                                "no_key": det.reasons.no_key,
                                "replay_error": det.reasons.replay_error,
                                "obs_mismatch": det.reasons.obs_mismatch,
                            },
                            "rng_after": hex(&det.rng_state_hash()),
                        })
                    )
                    .as_bytes(),
                )
                .unwrap_or_else(|e| die(&format!("{e}")));
            }

            let n_opt = env.num_options();
            if n_opt <= 0 {
                die(&format!("ALERT search_parts_dump: game {g} decision {k}: no legal option"));
            }
            let choice = (choice_rng.next() % n_opt as u64) as i32;
            guard.note(0, key, choice);
            prefix_keys.push(key_rank_of_choice(&obs, choice as usize));
            prefix.push(choice);
            pending = env.step(choice, &mut obs).unwrap_or_else(|e| die(&format!("{e}")));
            for t in trackers.iter_mut() {
                t.update(&env);
            }
            k += 1;
        }
        w.write_all(
            format!(
                "{}\n",
                json!({
                    "kind": "end", "g": g, "decisions": k,
                    "winner": env.winner(), "choices": env.choices(),
                })
            )
            .as_bytes(),
        )
        .unwrap_or_else(|e| die(&format!("{e}")));
        if !env.errors().is_empty() {
            die(&format!("ALERT search_parts_dump: {} Lua/script errors in game {g}", env.errors().len()));
        }
    }
    w.flush().unwrap_or_else(|e| die(&format!("{e}")));
    let mut inner = w.finish().unwrap_or_else(|e| die(&format!("{e}")));
    inner.flush().ok();
    eprintln!(
        "search_parts_dump: games={games} decisions={n_decisions} ties={n_ties} det_runs={n_det} \
         proposals={n_det_proposals} accepted={n_det_accepted} -> {}",
        out.display()
    );
}
