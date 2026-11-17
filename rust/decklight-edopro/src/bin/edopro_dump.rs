//! A2b-1 parity dump, matching `scripts/app/edopro_dump.py` line for line.
//!
//! edopro_dump --cdb-dir DIR --scripts-dir DIR --pairs A:B ... --games N --seed S --choice-seed C
//!             [--game-offset O] [--extra-dir DIR]... --out FILE.jsonl.gz

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main, CardData};
use decklight_core::constants as C;
use decklight_edopro::filter::{self, Refresh};
use decklight_edopro::inverse;
use decklight_edopro::record::{check_record, seed_words, sha256_hex};
use decklight_edopro::yrp::{self, Yrp};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::{Value, json};
use ygoenv_sys::{Decision, DuelConfig, Env, Obs, TCG_FLAGS};

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
    eprintln!("ALERT edopro_dump: {msg}");
    std::process::exit(2)
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(char::from_digit((x >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((x & 15) as u32, 16).unwrap());
    }
    s
}

fn hx(b: Option<&[u8]>) -> Value {
    b.map(|x| json!(hex(x))).unwrap_or(Value::Null)
}

fn canonical(mut trace: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(trace.len());
    let mut run: Vec<Vec<u8>> = Vec::new();
    for m in trace.drain(..) {
        let hint = m.first().is_some_and(|&t| t == C::MSG_CARD_HINT as u8 || t == C::MSG_PLAYER_HINT as u8);
        if hint {
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

fn trace_of(env: &Env<'_>) -> Vec<Vec<u8>> {
    (0..env.trace_len()).map(|i| env.trace_msg(i)).collect()
}

fn refresh_json(r: Refresh) -> Value {
    match r {
        Refresh::Loc { player, location } => json!(["loc", player, location]),
        Refresh::Single { player, location, seq } => json!(["single", player, location, seq]),
    }
}

fn emit_trace(
    rows: &mut Vec<Value>,
    env: &Env<'_>,
    trace: &[Vec<u8>],
    start: usize,
    pools: &mut [BTreeSet<u32>; 2],
    cards: &BTreeMap<u32, CardData>,
) -> usize {
    let mut n_pkts = 0usize;
    for (i, raw) in trace.iter().enumerate().skip(start) {
        // DecisionOnlyView pool: our deck + every card code on the wire for our seat.
        if raw.first().is_some_and(|&t| t != C::MSG_WIN as u8 && t != C::MSG_RETRY as u8) {
            for viewer in [0u8, 1] {
                if let Ok(Some(seen)) = filter::filtered_message(raw, viewer) {
                    for j in 0..seen.len().saturating_sub(3) {
                        let code = u32::from_le_bytes(seen[j..j + 4].try_into().unwrap());
                        if cards.contains_key(&code) {
                            pools[viewer as usize].insert(code);
                        }
                    }
                }
            }
        }
        let before = filter::before_refreshes(raw).unwrap_or_else(|e| die(&e));
        let after = filter::after_refreshes(raw).unwrap_or_else(|e| die(&e));
        let mut before_pkts = Vec::new();
        for &r in &before {
            let mut pair = Vec::new();
            for viewer in [0u8, 1] {
                let p = filter::refresh_packet(env, r, viewer).unwrap_or_else(|e| die(&e));
                pair.push(hx(p.as_deref()));
            }
            before_pkts.push(Value::Array(pair));
        }
        let mut after_pkts = Vec::new();
        for &r in &after {
            let mut pair = Vec::new();
            for viewer in [0u8, 1] {
                let p = filter::refresh_packet(env, r, viewer).unwrap_or_else(|e| die(&e));
                pair.push(hx(p.as_deref()));
            }
            after_pkts.push(Value::Array(pair));
        }
        n_pkts += before.len() + after.len();
        let f0 = filter::filtered_message(raw, 0).unwrap_or_else(|e| die(&e));
        let f1 = filter::filtered_message(raw, 1).unwrap_or_else(|e| die(&e));
        rows.push(json!({
            "k": "msg", "g": -1, "i": i, "raw": hex(raw),
            "f0": hx(f0.as_deref()), "f1": hx(f1.as_deref()),
            "wait": filter::waiting_player(raw),
            "before": before.iter().map(|&r| refresh_json(r)).collect::<Vec<_>>(),
            "after": after.iter().map(|&r| refresh_json(r)).collect::<Vec<_>>(),
            "before_pkts": before_pkts, "after_pkts": after_pkts,
        }));
    }
    n_pkts
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
    die(&format!("deck not found: {s}"));
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut cdb, mut scripts, mut d0, mut d1) = (None, None, None, None);
    let mut pairs: Vec<String> = Vec::new();
    let (mut games, mut seed, mut cseed, mut offset) = (1u64, 0u64, 0u64, 0u64);
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].clone();
        let mut need = || {
            i += 1;
            if i >= argv.len() {
                die(&format!("{a} needs a value"));
            }
            argv[i].clone()
        };
        match a.as_str() {
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
            "--out" => out = Some(PathBuf::from(need())),
            s => die(&format!("unknown argument {s}")),
        }
        i += 1;
    }
    let out = out.unwrap_or_else(|| die("--out is required"));
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
        die("no cards or no scripts loaded");
    }
    let read = |p: &PathBuf| parse_ydk(&std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))));
    let decks: Vec<_> = pair_list.iter().map(|(a, b)| (read(a), read(b))).collect();
    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    if let Some(p) = out.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let file = std::fs::File::create(&out).unwrap_or_else(|e| die(&format!("{}: {e}", out.display())));
    let mut w = GzEncoder::new(std::io::BufWriter::new(file), Compression::new(6));

    let mut totals = (0usize, 0usize, 0usize, 0usize); // msgs, refreshes, decisions, games
    let mut obs = Obs::zeroed();
    for gi in 0..games {
        let g = offset + gi;
        let (deck0, deck1) = &decks[(g as usize) % decks.len()];
        let s = seed + g;
        let order: [u8; 2] = if g % 2 == 0 { [0, 1] } else { [1, 0] };
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
            team_order: order,
            edopro_add: false,
        };
        let mut env = Env::new(&engine, &cfg, false).unwrap_or_else(|e| die(&e));
        env.set_record_trace(true);
        let mut rng = SplitMix(cseed ^ g);
        let mut pending = env.reset(&mut obs).unwrap_or_else(|e| die(&e));
        let mut rows: Vec<Value> = Vec::new();
        let mut pools = [
            deck0.main.iter().chain(deck0.extra.iter()).copied().collect::<BTreeSet<u32>>(),
            deck1.main.iter().chain(deck1.extra.iter()).copied().collect::<BTreeSet<u32>>(),
        ];
        let mut trace = canonical(trace_of(&env));
        let mut n_pkts = emit_trace(&mut rows, &env, &trace, 0, &mut pools, &cards);
        let mut d = 0usize;
        while pending {
            let dec_start = Decision::from_env(&env)
                .unwrap_or_else(|e| die(&e))
                .unwrap_or_else(|| die("no pending decision"));
            let raw = env.pending_msg();
            let player = dec_start.player();
            let pool: Vec<u32> = pools[player as usize].iter().copied().collect();
            let nres0 = env.num_responses();
            let mut seq: Vec<usize> = Vec::new();
            let mut resp: Option<Vec<u8>> = None;
            while resp.is_none() {
                let n = env.num_options();
                if n <= 0 {
                    die(&format!("game {g} decision {d}: no legal option"));
                }
                let idx = (rng.next() % n as u64) as i32;
                seq.push(idx as usize);
                pending = env.step(idx, &mut obs).unwrap_or_else(|e| die(&e));
                if env.num_responses() > nres0 {
                    resp = Some(env.response(nres0));
                }
                if !pending && resp.is_none() {
                    die(&format!("game {g} decision {d}: ended without a response"));
                }
            }
            let resp = resp.unwrap();
            let inv = inverse::choices_for_response(&dec_start, &resp, false)
                .unwrap_or_else(|| die(&format!("game {g} decision {d}: response not expressible")));
            rows.push(json!({
                "k": "inv", "g": g, "d": d, "msg": hex(&raw), "resp": hex(&resp),
                "seq": inv,
            }));
            let mut stand = Decision::new(&engine, &raw, &pool).unwrap_or_else(|e| die(&e));
            let opts: Vec<Value> = stand
                .options()
                .iter()
                .map(|o| {
                    json!([o.kind, o.has_card, o.code, o.controller, o.location, o.sequence, o.position, o.opt_code, o.desc, o.arg])
                })
                .collect();
            // ANNOUNCE_CARD: the client's pool (deck + codes seen on the wire) differs from the env's known codes
            // by design, so option indices differ; the client must still offer the declared code.
            let fresh = |sq: &[usize]| -> Option<Vec<u8>> {
                let mut dd = Decision::new(&engine, &raw, &pool).unwrap_or_else(|e| die(&e));
                let mut out = None;
                for &i in sq {
                    match dd.choose(i as i32) {
                        Ok(Some(r)) => out = Some(r),
                        Ok(None) => {}
                        Err(e) => die(&e),
                    }
                }
                out
            };
            let (stand_seq, stand_inv): (Vec<usize>, Vec<usize>) = if raw[0] as u32 == C::MSG_ANNOUNCE_CARD {
                let j = (0..opts.len())
                    .find(|&j| fresh(&[j]).as_deref() == Some(resp.as_slice()))
                    .unwrap_or_else(|| die(&format!("game {g} decision {d}: declared code not in the client pool")));
                (vec![j], vec![j])
            } else {
                (seq.iter().map(|&i| i as usize).collect(), inv.iter().map(|&i| i as usize).collect())
            };
            let mut stand_resp: Option<Vec<u8>> = None;
            for &i in &stand_seq {
                match stand.choose(i as i32) {
                    Ok(Some(r)) => stand_resp = Some(r),
                    Ok(None) => {}
                    Err(e) => die(&e),
                }
            }
            if stand_resp.as_deref() != Some(resp.as_slice()) {
                die(&format!("game {g} decision {d}: standalone response differs"));
            }
            let mut check = Decision::new(&engine, &raw, &pool).unwrap_or_else(|e| die(&e));
            let mut check_resp: Option<Vec<u8>> = None;
            for &i in &stand_inv {
                match check.choose(i as i32) {
                    Ok(Some(r)) => check_resp = Some(r),
                    Ok(None) => {}
                    Err(e) => die(&e),
                }
            }
            if check_resp.as_deref() != Some(resp.as_slice()) {
                die(&format!("game {g} decision {d}: inverse sequence response differs"));
            }
            rows.push(json!({
                "k": "dec", "g": g, "d": d, "msg": hex(&raw), "player": player,
                "opts": opts, "seq": seq, "stand_seq": stand_seq,
                "resp": stand_resp.as_deref().map(hex).map(Value::String).unwrap_or(Value::Null),
            }));
            d += 1;
            trace = canonical(trace_of(&env));
            let seen = rows.iter().filter(|r| r["k"] == "msg").count();
            n_pkts += emit_trace(&mut rows, &env, &trace, seen, &mut pools, &cards);
        }
        trace = canonical(trace_of(&env));
        let seen = rows.iter().filter(|r| r["k"] == "msg").count();
        n_pkts += emit_trace(&mut rows, &env, &trace, seen, &mut pools, &cards);
        for r in rows.iter_mut() {
            r["g"] = json!(g);
        }

        let responses: Vec<Vec<u8>> = (0..env.num_responses()).map(|i| env.response(i)).collect();
        let y = Yrp {
            seed: seed_words(s),
            names: vec![vec!["alice".to_string()], vec!["bob".to_string()]],
            lp: cfg.lp,
            hand: cfg.hand,
            draw: cfg.draw,
            flags: cfg.flags,
            decks: vec![
                vec![(yrp::edopro_main(&m0), deck0.extra.clone())],
                vec![(yrp::edopro_main(&m1), deck1.extra.clone())],
            ],
            extra_cards: Vec::new(),
            responses,
        };
        let data = yrp::write_yrpx(&y, &trace, 1_790_000_000, false).unwrap_or_else(|e| die(&e));
        let x = yrp::parse_yrpx(&data).unwrap_or_else(|e| die(&e));
        let rid = format!("00000000-0000-7000-8000-{g:012x}");
        let (rec, problems) =
            yrp::to_record(&x, &engine, &root, Some(0), "decklight-edopro-dump/0", None, Some(rid));
        let check = rec.as_ref().map(|r| check_record(&engine, r)).unwrap_or_default();
        let game = json!({
            "k": "game", "g": g, "order": order, "yrpx": sha256_hex(&data),
            "problems": problems,
            "add_order": rec.as_ref().map(|r| r.get("add_order").cloned().unwrap_or(json!([0, 1]))),
            "winner": rec.as_ref().map(|r| r["result"]["winner"].clone()),
            "turns": rec.as_ref().map(|r| r["result"]["turns"].clone()),
            "seats": rec.as_ref().map(|r| r["seats"].clone()),
            "rseats": rec.as_ref().map(|r| r["responses"].as_array().unwrap().iter().map(|e| e["seat"].clone()).collect::<Vec<_>>()),
            "rec": rec,
            "check": check,
            "decisions": d,
        });
        rows.push(game);
        for r in &rows {
            writeln!(w, "{}", serde_json::to_string(r).unwrap()).unwrap_or_else(|e| die(&e.to_string()));
        }
        let msgs = trace.len();
        totals.0 += msgs;
        totals.1 += n_pkts;
        totals.2 += d;
        totals.3 += 1;
    }
    w.finish().unwrap_or_else(|e| die(&e.to_string()));
    println!(
        "{}",
        json!({"totals": {"msgs": totals.0, "refreshes": totals.1, "decisions": totals.2, "games": totals.3},
               "out": out.to_string_lossy()})
    );
}
