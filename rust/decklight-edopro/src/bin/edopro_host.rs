//! EDOPro-compatible LAN host (Rust): the bot holds one seat and an EDOPro-family client the other, or two
//! clients join. Mirrors `scripts/edopro_host.py`'s flags where they still make sense.
//!
//! edopro-host --deck decks/pool/snake-eye.ydk --port 7911 [--opponent stream|random] [--once]
//!             [--remote 1|2] [--seed S] [--no-replay] [--packet-log FILE.jsonl.gz]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk};
use decklight_edopro::host::{AgentKind, DEFAULT_ROOM_NOTES, DuelHost, HostConfig};

fn die(msg: &str) -> ! {
    eprintln!("ALERT edopro_host: {msg}");
    std::process::exit(2)
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
    die(&format!("deck not found: {s}"))
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut deck: Option<String> = None;
    let mut cdb = PathBuf::from("third_party/BabelCDB");
    let mut scripts = PathBuf::from("third_party/CardScripts");
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut emb: Option<PathBuf> = None;
    let mut cfg = HostConfig {
        record_root: PathBuf::from("runs/edopro"),
        ..HostConfig::default()
    };
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
            "--deck" => deck = Some(need()),
            "--port" => cfg.port = need().parse().unwrap_or_else(|_| die("--port")),
            "--bind" => cfg.bind = need(),
            "--name" => cfg.name = need(),
            "--opponent" => {
                cfg.agent = match need().as_str() {
                    "stream" | "streambot" => AgentKind::Stream,
                    "random" => AgentKind::Random,
                    x => die(&format!("unknown opponent {x:?}")),
                }
            }
            "--agent-seed" => cfg.agent_seed = need().parse().unwrap_or_else(|_| die("--agent-seed")),
            "--seed" => cfg.seed = Some(need().parse().unwrap_or_else(|_| die("--seed"))),
            "--bot-second" => cfg.bot_second = true,
            "--once" => cfg.once = true,
            "--max-duels" => cfg.max_duels = need().parse().unwrap_or_else(|_| die("--max-duels")),
            "--remote" => {
                cfg.remote_seats = need().parse().unwrap_or_else(|_| die("--remote"));
                if !(1..=2).contains(&cfg.remote_seats) {
                    die("--remote must be 1 or 2");
                }
            }
            "--no-replay" => cfg.send_replay = false,
            "--replay" => cfg.send_replay = true,
            "--no-compress-replay" => cfg.compress_replay = false,
            "--packet-log" => cfg.packet_log = Some(PathBuf::from(need())),
            "--record-root" => cfg.record_root = PathBuf::from(need()),
            "--no-discovery" => cfg.discovery = false,
            "--cdb-dir" => cdb = PathBuf::from(need()),
            "--scripts-dir" => scripts = PathBuf::from(need()),
            "--extra-dir" => extra.push(PathBuf::from(need())),
            "--emb" => emb = Some(PathBuf::from(need())),
            "--room-notes" => {
                let _ = need(); // accepted for symmetry; the host does not advertise notes in discovery
            }
            "--help" | "-h" => {
                println!("{DEFAULT_ROOM_NOTES}");
                return;
            }
            s => die(&format!("unknown argument {s}")),
        }
        i += 1;
    }

    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&e));
    if cards.is_empty() {
        die("no cards loaded");
    }
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    if index.is_empty() {
        die("no scripts loaded");
    }
    let engine = build_engine(&cards, &index);

    let emb_codes: Option<BTreeSet<u32>> = {
        let p = emb.unwrap_or_else(|| PathBuf::from("data/cards/emb/qwen3-0.6b-d512.npz"));
        if p.exists() {
            match decklight_cards::emb::Emb::load(&p) {
                Ok(e) => Some(e.codes().iter().map(|&c| c as u32).collect()),
                Err(e) => {
                    eprintln!("ALERT edopro_host: cannot read {p:?}: {e}; deck embedding check disabled");
                    None
                }
            }
        } else {
            eprintln!("ALERT edopro_host: {p:?} missing; deck embedding check disabled");
            None
        }
    };

    let bot_deck = if cfg.remote_seats >= 2 {
        None
    } else {
        let Some(d) = &deck else { die("one remote seat needs --deck for the bot") };
        let path = resolve_deck(d);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| die(&format!("{}: {e}", path.display())));
        let d = parse_ydk(&text);
        let missing: Vec<u32> = d.main.iter().chain(&d.extra).copied().filter(|c| !cards.contains_key(c)).collect();
        if !missing.is_empty() {
            die(&format!("bot deck has {} codes missing from the card DB, e.g. {:?}", missing.len(), &missing[..missing.len().min(3)]));
        }
        Some(d)
    };
    let bot_name = deck
        .as_deref()
        .and_then(|d| Path::new(d).file_stem().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "bot".to_string());

    let mut host = DuelHost::new(&engine, &cards, emb_codes, cfg, bot_deck, bot_name);
    match host.serve() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("ALERT edopro_host: {e}");
            std::process::exit(1);
        }
    }
}
