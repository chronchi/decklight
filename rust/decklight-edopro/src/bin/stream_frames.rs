//! A2b-3a stream dumps: the host packets for both seats, the Rust `StreamState` log after every
//! packet, and (for gate 3) the web frames from the stream source next to the host `Env` frames.
//!
//!     stream_frames --pairs A:B ... --games N --seed S --choice-seed C [--game-offset O]
//!                   [--cdb-dir DIR] [--scripts-dir DIR] [--extra-dir DIR]...
//!                   --packets FILE.gz --log FILE.gz --frames FILE.gz
//!
//! `--packets` holds exactly what Python `stream_dump.py` must feed to `StreamState`; `--log` is
//! this binary's state log (one record per packet, the same JSON shape); `--frames` has both
//! `src:"env"` and `src:"stream"` frames for gate 3.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
use decklight_core::constants as C;
use decklight_core::query::ChainLink;
use decklight_edopro::filter::{self, Refresh};
use decklight_edopro::protocol;
use decklight_view::stream::{history_from_message, HistoryRow, StreamState};
use decklight_view::{Ctx, StreamSource, ViewSource};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::{Value, json};
use ygoenv_sys::{DuelConfig, Env, Obs, TCG_FLAGS};

// Optional cue-build timing (`STREAM_BENCH_CUES=1`): the old whole-trace-copy path vs
// the trace_len/trace_msg path, per source, in nanoseconds.
static BENCH: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static BENCH_N: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

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
    eprintln!("ALERT stream_frames: {msg}");
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

fn resolve_deck(s: &str) -> PathBuf {
    let p = PathBuf::from(s);
    if p.exists() {
        return p;
    }
    let pool = PathBuf::from("decks/pool").join(format!("{s}.ydk"));
    if pool.exists() { pool } else { die(&format!("deck not found: {s}")) }
}

fn card_json(c: &decklight_core::query::CardState) -> Value {
    json!({
        "controller": c.controller,
        "location": c.location,
        "sequence": c.sequence,
        "code": c.code,
        "position": c.position,
        "type": c.type_,
        "level": c.level,
        "rank": c.rank,
        "attribute": c.attribute,
        "race": c.race,
        "attack": c.attack,
        "defense": c.defense,
        "base_attack": c.base_attack,
        "base_defense": c.base_defense,
        "owner": c.owner,
        "status": c.status,
        "public": c.public,
        "lscale": c.lscale,
        "rscale": c.rscale,
        "link": c.link,
        "link_marker": c.link_marker,
        "overlay": c.overlay,
        "overlay_index": c.overlay_index,
    })
}

fn chain_json(c: &ChainLink) -> Value {
    json!({
        "code": c.code,
        "controller": c.controller,
        "location": c.location,
        "sequence": c.sequence,
        "position": c.position,
        "triggering_controller": c.triggering_controller,
        "triggering_location": c.triggering_location,
        "triggering_sequence": c.triggering_sequence,
        "description": c.description,
    })
}

fn history_json(h: &HistoryRow) -> Value {
    json!({"row": h.row, "actor": h.actor, "turn": h.turn, "phase": h.phase, "public": h.public})
}

fn state_json(st: &StreamState, hist: &[HistoryRow]) -> Value {
    let mut loc = Vec::new();
    for p in 0..2i32 {
        for location in [
            C::LOCATION_DECK,
            C::LOCATION_HAND,
            C::LOCATION_MZONE,
            C::LOCATION_SZONE,
            C::LOCATION_GRAVE,
            C::LOCATION_REMOVED,
            C::LOCATION_EXTRA,
        ] {
            let cards: Vec<Value> = StreamState::query_location(st, p as u32, location)
                .iter()
                .map(|c| c.as_ref().map(card_json).unwrap_or(Value::Null))
                .collect();
            loc.push(json!([p, location, cards]));
        }
    }
    let ((lp0, lp1), chain) = StreamState::field_state(st);
    json!({
        "k": "state",
        "turn": st.turn,
        "phase": st.phase,
        "first_player": st.first_player,
        "turn_player": st.turn_player,
        "field": {"lp": [lp0, lp1], "chain": chain.iter().map(chain_json).collect::<Vec<_>>()},
        "loc": loc,
        "memory": {"known": [st.memory.known[0], st.memory.known[1]], "codes": [st.memory.codes[0], st.memory.codes[1]]},
        "hist": hist.iter().map(history_json).collect::<Vec<_>>(),
    })
}

fn refresh_loc(player: u8, location: u32) -> Refresh {
    Refresh::Loc { player, location }
}

struct Writers {
    packets: GzEncoder<Box<dyn Write>>,
    log: GzEncoder<Box<dyn Write>>,
    frames: GzEncoder<Box<dyn Write>>,
}

impl Writers {
    fn new(packets: &PathBuf, log: &PathBuf, frames: &PathBuf) -> Writers {
        let open = |p: &PathBuf| -> Box<dyn Write> {
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let f = std::fs::File::create(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
            Box::new(std::io::BufWriter::with_capacity(1 << 20, f))
        };
        Writers {
            packets: GzEncoder::new(open(packets), Compression::new(6)),
            log: GzEncoder::new(open(log), Compression::new(6)),
            frames: GzEncoder::new(open(frames), Compression::new(6)),
        }
    }

    fn line(enc: &mut GzEncoder<Box<dyn Write>>, v: &Value) {
        let mut buf = serde_json::to_vec(v).unwrap_or_else(|e| die(&format!("serialize: {e}")));
        buf.push(b'\n');
        enc.write_all(&buf).unwrap_or_else(|e| die(&format!("write: {e}")));
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut cdb, mut scripts) = (None, None);
    let mut pairs_s: Vec<String> = Vec::new();
    let (mut games, mut seed, mut cseed, mut offset) = (1u64, 0u64, 0u64, 0u64);
    let mut extra: Vec<PathBuf> = Vec::new();
    let (mut packets, mut log, mut frames) = (None, None, None);
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
            "--pairs" => {
                while i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    i += 1;
                    pairs_s.push(argv[i].clone());
                }
            }
            "--games" => games = need().parse().unwrap_or_else(|_| die("--games")),
            "--seed" => seed = need().parse().unwrap_or_else(|_| die("--seed")),
            "--choice-seed" => cseed = need().parse().unwrap_or_else(|_| die("--choice-seed")),
            "--game-offset" => offset = need().parse().unwrap_or_else(|_| die("--game-offset")),
            "--extra-dir" => extra.push(PathBuf::from(need())),
            "--packets" => packets = Some(PathBuf::from(need())),
            "--log" => log = Some(PathBuf::from(need())),
            "--frames" => frames = Some(PathBuf::from(need())),
            s => die(&format!("unknown argument {s}")),
        }
        i += 1;
    }
    if pairs_s.is_empty() {
        die("--pairs is required");
    }
    let pair_list: Vec<(PathBuf, PathBuf)> = pairs_s
        .iter()
        .map(|it| {
            let (a, b) = it.split_once(':').unwrap_or_else(|| die(&format!("--pairs entry {it:?} needs A:B")));
            (resolve_deck(a), resolve_deck(b))
        })
        .collect();
    let cdb = cdb.unwrap_or_else(|| PathBuf::from("third_party/BabelCDB"));
    let scripts = scripts.unwrap_or_else(|| PathBuf::from("third_party/CardScripts"));
    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&e));
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    let engine = build_engine(&cards, &index);
    if cards.is_empty() || index.is_empty() {
        die("no cards or no scripts loaded");
    }
    let texts = decklight_cards::texts::Texts::load(&cdb, &extra, &PathBuf::from("third_party/edopro-strings.conf"))
        .unwrap_or_else(|e| die(&e));
    let card_types: BTreeMap<u32, u32> = cards.iter().map(|(c, d)| (*c, d.type_)).collect();
    let ctx = Ctx::new(texts, card_types, 8000);
    let read = |p: &PathBuf| parse_ydk(&std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))));
    let decks: Vec<_> = pair_list.iter().map(|(a, b)| (read(a), read(b))).collect();

    let (pp, pl, pf) = match (packets, log, frames) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => die("--packets, --log and --frames are required"),
    };
    let mut w = Writers::new(&pp, &pl, &pf);
    let skip_frames = std::env::var_os("STREAM_SKIP_FRAMES").is_some();
    let skip_log = std::env::var_os("STREAM_SKIP_LOG").is_some();
    let mut obs = Obs::zeroed();
    let mut n_games = 0usize;
    let mut frame_no = 0u64;
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
        let mut env = Env::new(&engine, &cfg, false).unwrap_or_else(|e| die(&e));
        env.set_record_trace(true);
        let pending = env.reset(&mut obs).unwrap_or_else(|e| die(&e));

        let mut states = [
            StreamState::new(&m0, &deck0.extra, &cards, 0),
            StreamState::new(&m1, &deck1.extra, &cards, 1),
        ];
        let mut sources = [
            StreamSource::new(&m0, &deck0.extra, &cards, 0, &engine),
            StreamSource::new(&m1, &deck1.extra, &cards, 1, &engine),
        ];
        let mut pkt_i = [0usize; 2];
        let mut log_sent = [0usize; 2];
        let mut cue_sent = [0usize; 2];
        let mut stream_log_sent = [0usize; 2];
        let mut stream_cue_sent = [0usize; 2];

        // Deck header for Python + the same START/extra packets client_parity feeds.
        for seat in 0..2u8 {
            Writers::line(
                &mut w.packets,
                &json!({"k": "stream", "g": g, "s": seat, "main": if seat == 0 { m0.clone() } else { m1.clone() },
                        "extra": if seat == 0 { deck0.extra.clone() } else { deck1.extra.clone() }}),
            );
        }
        let feed = |seat: u8, payload: Vec<u8>,
                        states: &mut [StreamState; 2],
                        sources: &mut [StreamSource; 2],
                        pkt_i: &mut [usize; 2],
                        w: &mut Writers| {
            let si = seat as usize;
            states[si].feed(&payload).unwrap_or_else(|e| die(&format!("state seat {seat}: {e}")));
            let hist = history_from_message(&states[si], &payload);
            sources[si].feed(&payload).unwrap_or_else(|e| die(&format!("source seat {seat}: {e}")));
            Writers::line(
                &mut w.packets,
                &json!({"k": "pkt", "g": g, "s": seat, "i": pkt_i[si], "msg": hex(&payload)}),
            );
            if !skip_log {
                let mut st = state_json(&states[si], &hist);
                st["g"] = json!(g);
                st["s"] = json!(seat);
                st["i"] = json!(pkt_i[si]);
                Writers::line(&mut w.log, &st);
            }
            pkt_i[si] += 1;
        };

        let lp = cfg.lp;
        for seat in 0..2u8 {
            let start = protocol::build_start(
                lp,
                m0.len() as u32,
                deck0.extra.len() as u32,
                m1.len() as u32,
                deck1.extra.len() as u32,
                seat,
            );
            feed(seat, start, &mut states, &mut sources, &mut pkt_i, &mut w);
            for p in 0..2u8 {
                if let Some(pkt) =
                    filter::refresh_packet(&env, refresh_loc(p, C::LOCATION_EXTRA), seat).unwrap_or_else(|e| die(&e))
                {
                    feed(seat, pkt, &mut states, &mut sources, &mut pkt_i, &mut w);
                }
            }
        }

        let mut sent = 0usize;
        let pump = |env: &Env<'_>, sent: &mut usize,
                        states: &mut [StreamState; 2],
                        sources: &mut [StreamSource; 2],
                        pkt_i: &mut [usize; 2],
                        w: &mut Writers| {
            while *sent < env.trace_len() {
                let raw = env.trace_msg(*sent);
                *sent += 1;
                let before = filter::before_refreshes(&raw).unwrap_or_else(|e| die(&e));
                let after = filter::after_refreshes(&raw).unwrap_or_else(|e| die(&e));
                for seat in 0..2u8 {
                    for r in &before {
                        if let Some(pkt) = filter::refresh_packet(env, *r, seat).unwrap_or_else(|e| die(&e)) {
                            feed(seat, pkt, states, sources, pkt_i, w);
                        }
                    }
                    if let Some(msg) = filter::filtered_message(&raw, seat).unwrap_or_else(|e| die(&e)) {
                        feed(seat, msg, states, sources, pkt_i, w);
                    }
                    for r in &after {
                        if let Some(pkt) = filter::refresh_packet(env, *r, seat).unwrap_or_else(|e| die(&e)) {
                            feed(seat, pkt, states, sources, pkt_i, w);
                        }
                    }
                }
            }
        };

        let emit_frames = |g: u64, d: i64, seats: &[u8],
                               env: &Env<'_>,
                               sources: &mut [StreamSource; 2],
                               log_sent: &mut [usize; 2],
                               cue_sent: &mut [usize; 2],
                               stream_log_sent: &mut [usize; 2],
                               stream_cue_sent: &mut [usize; 2],
                               frame_no: &mut u64,
                               w: &mut Writers| {
            if skip_frames {
                return;
            }
            for &seat in seats {
                let si = seat as usize;
                let f = *frame_no;
                *frame_no += 1;
                let pi_env = env.trace_len();
                let view = ctx.build_view(env, seat as i32, false);
                let prompt = if env.decision().as_ref().is_some_and(|x| x.player == seat as i32) {
                    ctx.build_prompt(env, seat as i32, d)
                } else {
                    None
                };
                let events = ctx.build_events(env, seat as i32, log_sent[si]);
                let bench = std::env::var_os("STREAM_BENCH_CUES").is_some();
                let t0 = Instant::now();
                let (cues, clen) = decklight_view::build_cues(env, &ctx, seat as i32, cue_sent[si], false);
                if bench {
                    BENCH[0].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    BENCH_N[0].fetch_add(1, Ordering::Relaxed);
                    let t1 = Instant::now();
                    let trace = env.trace();
                    let _ = decklight_view::cues_for_trace(&ctx, &trace, seat as i32, cue_sent[si], false, ctx.lp0);
                    BENCH[1].fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                log_sent[si] = env.history_log().len();
                cue_sent[si] = clen;
                Writers::line(
                    &mut w.frames,
                    &json!({"k": "frame", "g": g, "s": seat, "d": d, "src": "env", "f": f, "pi": pi_env,
                            "view": view, "prompt": prompt, "events": events, "cues": cues}),
                );

                let pi_stream = sources[si].trace_msgs().len();
                let src: &dyn ViewSource = &sources[si];
                let view = ctx.build_view(src, seat as i32, false);
                let prompt = ctx.build_prompt(src, seat as i32, d);
                let events = ctx.build_events(src, seat as i32, stream_log_sent[si]);
                let t2 = Instant::now();
                let (cues, clen) = decklight_view::build_cues(src, &ctx, seat as i32, stream_cue_sent[si], false);
                if bench {
                    BENCH[2].fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    BENCH_N[1].fetch_add(1, Ordering::Relaxed);
                    let t3 = Instant::now();
                    let trace = src.trace();
                    let _ = decklight_view::cues_for_trace(&ctx, &trace, seat as i32, stream_cue_sent[si], false, ctx.lp0);
                    BENCH[3].fetch_add(t3.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                stream_log_sent[si] = sources[si].history_rows().len();
                stream_cue_sent[si] = clen;
                Writers::line(
                    &mut w.frames,
                    &json!({"k": "frame", "g": g, "s": seat, "d": d, "src": "stream", "f": f, "pi": pi_stream,
                            "view": view, "prompt": prompt, "events": events, "cues": cues}),
                );
            }
        };

        if pending {
            let mut rng = SplitMix(cseed ^ g);
            loop {
                pump(&env, &mut sent, &mut states, &mut sources, &mut pkt_i, &mut w);
                let Some(dec) = env.decision() else { break };
                let n = env.num_options();
                if n <= 0 {
                    die(&format!("game {g}: a pending decision with no options"));
                }
                let idx = (rng.next() % n as u64) as i32;
                let player = dec.player;
                emit_frames(g, dec.step as i64, &[player as u8], &env, &mut sources, &mut log_sent, &mut cue_sent,
                            &mut stream_log_sent, &mut stream_cue_sent, &mut frame_no, &mut w);
                if (0..2).contains(&player) {
                    let si = player as usize;
                    if sources[si].needs_answer() {
                        if (idx as usize) >= sources[si].n_opts() {
                            die(&format!("game {g} seat {player}: host option {idx} outside the stream options"));
                        }
                        sources[si].take(idx as usize).unwrap_or_else(|e| die(&e));
                    }
                }
                let more = env.step(idx, &mut obs).unwrap_or_else(|e| die(&e));
                if !more {
                    break;
                }
            }
        }
        pump(&env, &mut sent, &mut states, &mut sources, &mut pkt_i, &mut w);
        emit_frames(g, -1, &[0, 1], &env, &mut sources, &mut log_sent, &mut cue_sent,
                    &mut stream_log_sent, &mut stream_cue_sent, &mut frame_no, &mut w);
        n_games += 1;
    }
    for enc in [&mut w.packets, &mut w.log, &mut w.frames] {
        enc.try_finish().unwrap_or_else(|e| die(&e.to_string()));
    }
    let mut report = json!({"games": n_games, "packets": pp.to_string_lossy(), "log": pl.to_string_lossy(), "frames": pf.to_string_lossy()});
    let en = BENCH_N[0].load(Ordering::Relaxed).max(1) as f64;
    let sn = BENCH_N[1].load(Ordering::Relaxed).max(1) as f64;
    if std::env::var_os("STREAM_BENCH_CUES").is_some() {
        report["ms_per_frame"] = json!({
            "env_before": BENCH[1].load(Ordering::Relaxed) as f64 / en / 1e6,
            "env_after": BENCH[0].load(Ordering::Relaxed) as f64 / en / 1e6,
            "stream_before": BENCH[3].load(Ordering::Relaxed) as f64 / sn / 1e6,
            "stream_after": BENCH[2].load(Ordering::Relaxed) as f64 / sn / 1e6,
        });
    }
    println!("{report}");
}
