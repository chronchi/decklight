//! EDOPro network client (Rust): join a room (or create one) and play one seat as a bot.
//!
//! edopro-join --deck decks/pool/snake-eye.ydk --host 127.0.0.1 --port 7911 [--roomid N]
//!             [--create --rule 3 --room-notes TEXT] [--agent stream|random] [--agent-seed S]
//!             [--packet-log FILE.jsonl.gz] [--list-rooms http://host:7922] [--list-lan]

use std::path::PathBuf;
use std::time::Duration;

use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk};
use decklight_edopro::agent::{RandomAgent, StreamBot};
use decklight_edopro::client::{ClientConfig, ClientSession};
use decklight_edopro::discovery;
use decklight_edopro::host::{AgentKind, DEFAULT_ROOM_NOTES};
use decklight_edopro::protocol::{HostInfo, ProtocolConfig};

fn die(msg: &str) -> ! {
    eprintln!("ALERT edopro_client: {msg}");
    std::process::exit(2)
}

fn parse_int_auto(s: &str) -> i64 {
    let t = s.trim();
    if let Some(r) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i64::from_str_radix(r, 16)
    } else {
        t.parse()
    }
    .unwrap_or_else(|_| die(&format!("not an integer: {s:?}")))
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
    let mut cfg = ClientConfig::default();
    let mut deck: Option<String> = None;
    let mut cdb = PathBuf::from("third_party/BabelCDB");
    let mut scripts = PathBuf::from("third_party/CardScripts");
    let mut extra: Vec<PathBuf> = Vec::new();
    let mut agent = AgentKind::Stream;
    let mut agent_seed: u64 = 0;
    let mut create = false;
    let mut rule: u8 = 3;
    let mut start_lp: u32 = 8000;
    let mut list_rooms: Option<String> = None;
    let mut list_lan = false;
    let mut windbot_version: Option<u32> = None;
    let mut pro_version: Option<u16> = None;
    let mut client_version: Option<(u8, u8, u8, u8)> = None;

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
            "--name" => cfg.name = need(),
            "--deck" | "--deckfile" => deck = Some(need()),
            "--host" => cfg.host = need(),
            "--port" => cfg.port = need().parse().unwrap_or_else(|_| die("--port")),
            "--password" | "--hostinfo" => cfg.password = need(),
            "--roomid" | "--room-id" => cfg.gameid = need().parse().unwrap_or_else(|_| die("--roomid")),
            "--room-notes" => cfg.room_notes = need(),
            "--hand" => cfg.hand = need().parse().unwrap_or_else(|_| die("--hand")),
            "--timeout" => cfg.timeout = Duration::from_secs_f64(need().parse().unwrap_or_else(|_| die("--timeout"))),
            "--record-dir" => cfg.record_dir = PathBuf::from(need()),
            "--packet-log" => cfg.packet_log = Some(PathBuf::from(need())),
            "--agent" => {
                agent = match need().as_str() {
                    "stream" | "streambot" => AgentKind::Stream,
                    "random" => AgentKind::Random,
                    x => die(&format!("unknown agent {x:?}")),
                }
            }
            "--agent-seed" | "--seed" => agent_seed = need().parse().unwrap_or_else(|_| die("--agent-seed")),
            "--create" => create = true,
            "--rule" => rule = need().parse().unwrap_or_else(|_| die("--rule")),
            "--start-lp" => start_lp = need().parse().unwrap_or_else(|_| die("--start-lp")),
            "--version" => windbot_version = Some(need().parse().unwrap_or_else(|_| die("--version"))),
            "--pro-version" => pro_version = Some(parse_int_auto(&need()) as u16),
            "--client-version" => {
                client_version = Some(ProtocolConfig::parse_version(&need()).unwrap_or_else(|e| die(&e)))
            }
            "--cdb-dir" => cdb = PathBuf::from(need()),
            "--scripts-dir" => scripts = PathBuf::from(need()),
            "--extra-dir" => extra.push(PathBuf::from(need())),
            "--list-rooms" => list_rooms = Some(need()),
            "--list-lan" => list_lan = true,
            "--help" | "-h" => {
                println!("{DEFAULT_ROOM_NOTES}");
                return;
            }
            s => die(&format!("unknown argument {s}")),
        }
        i += 1;
    }

    if let Some(url) = list_rooms {
        let (host, port) = discovery::parse_http_target(&url).unwrap_or_else(|e| die(&e));
        match discovery::fetch_room_list(&host, port) {
            Ok(rooms) => {
                for r in rooms {
                    println!("{}", serde_json::to_string(&r.raw).unwrap_or_default());
                }
            }
            Err(e) => die(&e),
        }
        return;
    }
    if list_lan {
        match discovery::discover_lan(Duration::from_secs(2)) {
            Ok(hosts) => {
                for (ip, h) in hosts {
                    println!(
                        "{}",
                        serde_json::json!({"address": ip.to_string(), "name": h.name, "port": h.port,
                                           "version": h.info.version})
                    );
                }
            }
            Err(e) => die(&e),
        }
        return;
    }

    let Some(dref) = deck else { die("--deck is required") };
    let dpath = resolve_deck(&dref);
    let text = std::fs::read_to_string(&dpath).unwrap_or_else(|e| die(&format!("{}: {e}", dpath.display())));
    let d = parse_ydk(&text);

    let cards = load_cards(&default_cdbs(&cdb, &extra)).unwrap_or_else(|e| die(&e));
    if cards.is_empty() {
        die("no cards loaded");
    }
    let missing: Vec<u32> = d.main.iter().chain(&d.extra).copied().filter(|c| !cards.contains_key(c)).collect();
    if !missing.is_empty() {
        die(&format!("deck has {} codes missing from the card DB, e.g. {:?}", missing.len(), &missing[..missing.len().min(3)]));
    }
    let index = index_scripts(&scripts, &extra).unwrap_or_else(|e| die(&e));
    if index.is_empty() {
        die("no scripts loaded");
    }
    let engine = build_engine(&cards, &index);

    if let Some(v) = windbot_version {
        cfg.proto = ProtocolConfig::from_windbot_version(v);
    }
    if let Some(v) = pro_version {
        cfg.proto.pro_version = v;
    }
    if let Some(v) = client_version {
        cfg.proto.client_version = v;
    }
    if create {
        if cfg.room_notes.is_empty() {
            cfg.room_notes = DEFAULT_ROOM_NOTES.to_string();
        }
        // The room runs our engine's rules (TCG/MR5 flags); 0 would be a duel without Master Rule 5.
        let flags = ygoenv_sys::TCG_FLAGS;
        cfg.create = Some(HostInfo {
            rule,
            start_lp,
            duel_flag_low: (flags & 0xFFFF_FFFF) as u32,
            duel_flag_high: (flags >> 32) as u32,
            ..HostInfo::default()
        });
    }

    let agent: Box<dyn decklight_edopro::agent::Agent> = match agent {
        AgentKind::Stream => Box::new(StreamBot::new(agent_seed)),
        AgentKind::Random => Box::new(RandomAgent::new(agent_seed ^ 0xA5A5)),
    };
    let mut session = ClientSession::new(
        &engine,
        &cards,
        cfg,
        d.main,
        d.extra,
        d.side,
        agent,
    );
    let code = session.run();
    let w = session.record.winner.map(|(a, b)| format!("{a}:{b}")).unwrap_or_else(|| "-".into());
    println!(
        "client exit {code} (winner {w}, {} responses, {} retries)",
        session.record.n_responses, session.record.retries
    );
    std::process::exit(code);
}
