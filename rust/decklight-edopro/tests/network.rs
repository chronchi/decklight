//! Gate 5: the host session state machine, client lobby errors and malformed peers (never a panic).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use decklight_cards::{CardData, Deck, build_engine, default_cdbs, index_scripts, load_cards, parse_ydk};
use decklight_core::constants as C;
use decklight_core::messages as M;
use decklight_edopro::agent::{Agent, StreamBot};
use decklight_edopro::client::{BAD_ARGS, ClientConfig, ClientSession, TIMEOUT};
use decklight_edopro::host::{
    AgentKind, DuelHost, HostConfig, MAX_FRAME_PAYLOAD, replay_frames,
};
use decklight_edopro::host::{LocalPeer, Seat};
use decklight_edopro::protocol as P;
use ygoenv_sys::Engine;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn engine_and_cards() -> (Engine, BTreeMap<u32, CardData>) {
    let root = repo_root();
    let cards = load_cards(&default_cdbs(&root.join("third_party/BabelCDB"), &[])).unwrap();
    let scripts = index_scripts(&root.join("third_party/CardScripts"), &[]).unwrap();
    let engine = build_engine(&cards, &scripts);
    (engine, cards)
}

fn snake_eye() -> Deck {
    parse_ydk(&std::fs::read_to_string(repo_root().join("decks/pool/snake-eye.ydk")).unwrap())
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("decklight-net-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn recv(peer: &LocalPeer, limit: Duration) -> Result<P::Frame, String> {
    peer.out.recv_timeout(limit).map_err(|e| format!("peer recv: {e}"))
}

fn recv_until(peer: &LocalPeer, mut pred: impl FnMut(&P::Frame) -> bool) -> Result<P::Frame, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(f) = recv(peer, Duration::from_millis(100)) {
            if pred(&f) {
                return Ok(f);
            }
        }
        if Instant::now() > deadline {
            return Err("timed out waiting for a host packet".into());
        }
    }
}

fn is_select(payload: &[u8]) -> bool {
    payload.first().is_some_and(|&t| M::is_select_type(t as u32))
}

/// Drive the human side through the lobby and RPS, letting the human win RPS and go first.
fn human_lobby(peer: &LocalPeer, deck: &Deck, bot_seed: u64) -> Result<(), String> {
    peer.send(P::CTOS_PLAYER_INFO, &P::pack_player_info("human")).map_err(|e| e.to_string())?;
    peer.send(
        P::CTOS_JOIN_GAME,
        &P::pack_join_game(P::PRO_VERSION, 0, P::client_version_bytes(P::CLIENT_VERSION.0, P::CLIENT_VERSION.1, P::CLIENT_VERSION.2, P::CLIENT_VERSION.3), ""),
    )
    .map_err(|e| e.to_string())?;
    recv_until(peer, |f| f.proto == P::STOC_JOIN_GAME)?;
    peer.send(
        P::CTOS_UPDATE_DECK,
        &P::pack_update_deck(&deck.main, &deck.extra, &deck.side),
    )
    .map_err(|e| e.to_string())?;
    recv_until(peer, |f| f.proto == P::STOC_TYPE_CHANGE)?;
    peer.send(P::CTOS_HS_READY, &[]).map_err(|e| e.to_string())?;
    let mut bot = StreamBot::new(bot_seed);
    let bot_throw = bot.rps();
    let win_throw = (1..=3u8).find(|&a| P::rps_beats(a, bot_throw)).unwrap_or(1);
    recv_until(peer, |f| f.proto == P::STOC_SELECT_HAND)?;
    peer.send(P::CTOS_HAND_RESULT, &[win_throw]).map_err(|e| e.to_string())?;
    recv_until(peer, |f| f.proto == P::STOC_SELECT_TP)?;
    peer.send(P::CTOS_TP_RESULT, &[1]).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_record(record_root: &Path) -> serde_json::Value {
    let mut files: Vec<PathBuf> = std::fs::read_dir(record_root)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".match.json.gz"))
        .collect();
    files.sort();
    let f = std::fs::File::open(files.last().expect("no record written")).unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(f).read_to_string(&mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn host<'a>(engine: &'a Engine, cards: &'a BTreeMap<u32, CardData>, agent_seed: u64, root: PathBuf) -> DuelHost<'a> {
    let cfg = HostConfig {
        send_replay: false,
        discovery: false,
        record_root: root,
        agent: AgentKind::Stream,
        agent_seed,
        accept_timeout: Duration::from_secs(5),
        read_timeout: Duration::from_millis(1),
        duel_timeout: Duration::from_secs(30),
        ..HostConfig::default()
    };
    DuelHost::new(engine, cards, None, cfg, None, "bot".into())
}

#[test]
fn host_wrong_response_retries_then_surrender_wins_for_the_other_seat() {
    let (engine, cards) = engine_and_cards();
    let deck = snake_eye();
    let root = temp_dir("surrender");
    let mut h = host(&engine, &cards, 7, root.clone());
    let (local, peer) = Seat::local(0, "human".into());
    let bot = Seat::bot(1, deck.clone(), "bot".into(), Box::new(StreamBot::new(7)));
    let d2 = deck.clone();
    let peer_thread = std::thread::spawn(move || -> Result<(), String> {
        human_lobby(&peer, &d2, 7)?;
        let f = recv_until(&peer, |f| f.proto == P::STOC_GAME_MSG && is_select(&f.payload))?;
        assert!(is_select(&f.payload));
        peer.send(P::CTOS_RESPONSE, &[0xFF, 0xFF]).map_err(|e| e.to_string())?;
        let retry = recv_until(&peer, |f| f.proto == P::STOC_GAME_MSG)?;
        assert_eq!(retry.payload.first().copied(), Some(C::MSG_RETRY as u8));
        peer.send(P::CTOS_SURRENDER, &[]).map_err(|e| e.to_string())?;
        let win = recv_until(&peer, |f| f.proto == P::STOC_GAME_MSG)?;
        assert_eq!(win.payload.first().copied(), Some(C::MSG_WIN as u8));
        assert_eq!(win.payload.get(1).copied(), Some(1)); // the bot held engine seat 1
        recv_until(&peer, |f| f.proto == P::STOC_DUEL_END)?;
        Ok(())
    });
    let path = h.run_duel_with_seats(vec![local, bot]).expect("host failed");
    peer_thread.join().unwrap().expect("peer failed");
    assert!(path.is_some());
    let rec = read_record(&root);
    assert_eq!(rec["result"]["surrendered"], true);
    assert_eq!(rec["result"]["winner"], 1);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn host_disconnect_mid_duel_ends_the_duel_and_records_it() {
    let (engine, cards) = engine_and_cards();
    let deck = snake_eye();
    let root = temp_dir("disconnect");
    let mut h = host(&engine, &cards, 9, root.clone());
    let (local, peer) = Seat::local(0, "human".into());
    let bot = Seat::bot(1, deck.clone(), "bot".into(), Box::new(StreamBot::new(9)));
    let d2 = deck.clone();
    let peer_thread = std::thread::spawn(move || -> Result<(), String> {
        human_lobby(&peer, &d2, 9)?;
        recv_until(&peer, |f| f.proto == P::STOC_GAME_MSG && is_select(&f.payload))?;
        // Drop the app side: the host sees the local channel close.
        drop(peer);
        Ok(())
    });
    let path = h.run_duel_with_seats(vec![local, bot]).expect("host failed");
    peer_thread.join().unwrap().expect("peer failed");
    assert!(path.is_some());
    let rec = read_record(&root);
    assert_eq!(rec["result"]["disconnected"], true);
    assert_eq!(rec["result"]["winner"], 1);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn oversize_replay_is_a_chat_error_then_the_empty_open_prompt() {
    let small = vec![0u8; MAX_FRAME_PAYLOAD];
    let frames = replay_frames(&small);
    assert_eq!(frames[0].0, P::STOC_NEW_REPLAY);
    assert_eq!(frames[0].1.len(), MAX_FRAME_PAYLOAD);
    assert_eq!(frames[1].0, P::STOC_REPLAY);
    assert!(frames[1].1.is_empty());
    let big = vec![0u8; MAX_FRAME_PAYLOAD + 1];
    let frames = replay_frames(&big);
    assert_eq!(frames[0].0, P::STOC_CHAT);
    assert_eq!(frames[0].1.len(), 554); // Chat2: type + isTeam + clientName[20] + msg[256]
    assert_eq!(frames[0].1[0], 3); // PTYPE_SYSTEM_ERROR
    assert_eq!(frames[1].0, P::STOC_REPLAY);
}

// ---------------------------------------------------------------- client lobby errors

fn fake_server(reply: impl FnOnce(&mut TcpStream, &mut P::FrameReader) + Send + 'static) -> (u16, std::thread::JoinHandle<()>) {
    let l = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = l.local_addr().unwrap().port();
    let h = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let _ = s.set_nodelay(true);
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut r = P::FrameReader::new();
        let mut buf = [0u8; 4096];
        // Wait for CTOS_JOIN_GAME (the client sends PLAYER_INFO first).
        loop {
            match s.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    for f in r.feed(&buf[..n]).unwrap_or_default() {
                        if f.proto == P::CTOS_JOIN_GAME || f.proto == P::CTOS_CREATE_GAME {
                            reply(&mut s, &mut r);
                            return;
                        }
                    }
                }
            }
        }
    });
    (port, h)
}

fn client(port: u16, root: &Path, timeout_s: u64) -> ClientSession<'static> {
    // The lobby-error tests never build a decision, so an empty engine/card map is enough; leaking them keeps
    // the borrow simple and the process is a test.
    let engine: &'static Engine = Box::leak(Box::new(Engine::new()));
    let cards: &'static BTreeMap<u32, CardData> = Box::leak(Box::new(BTreeMap::new()));
    let cfg = ClientConfig {
        port,
        record_dir: root.to_path_buf(),
        timeout: Duration::from_secs(timeout_s),
        ..ClientConfig::default()
    };
    ClientSession::new(engine, cards, cfg, vec![], vec![], vec![], Box::new(StreamBot::new(1)))
}

fn one_error(payload: Vec<u8>, tag: &str) -> (i32, Vec<String>) {
    let (port, server) = fake_server(move |s, _r| {
        s.write_all(&P::pack_frame(P::STOC_ERROR_MSG, &payload)).unwrap();
    });
    let root = temp_dir(tag);
    let mut s = client(port, &root, 5);
    let code = s.run();
    server.join().unwrap();
    let alerts = s.record.alerts.clone();
    let _ = std::fs::remove_dir_all(&root);
    (code, alerts)
}

#[test]
fn client_reports_wrong_password_version_and_deck_errors() {
    let (code, alerts) = one_error(P::pack_join_error(P::JERR_PASSWORD), "joinerr");
    assert_eq!(code, BAD_ARGS);
    assert!(alerts.iter().any(|a| a.contains("wrong password")), "{alerts:?}");

    let mut ver = vec![P::ERR_VERERROR, 0, 0, 0];
    ver.extend_from_slice(&[P::CLIENT_VERSION.0, P::CLIENT_VERSION.1, P::CLIENT_VERSION.2, P::CLIENT_VERSION.3]);
    let (code, alerts) = one_error(ver, "vererr");
    assert_eq!(code, BAD_ARGS);
    assert!(alerts.iter().any(|a| a.contains("version error")), "{alerts:?}");

    let (code, alerts) = one_error(P::pack_deck_error(P::DERR_MAINCOUNT, 39, 40, 60, 0), "deckerr");
    assert_eq!(code, BAD_ARGS);
    assert!(alerts.iter().any(|a| a.contains("main count")), "{alerts:?}");
}

#[test]
fn client_rejects_a_malformed_join_game_without_panicking() {
    let (port, server) = fake_server(|s, _r| {
        s.write_all(&P::pack_frame(P::STOC_JOIN_GAME, &[0u8; 10])).unwrap();
    });
    let root = temp_dir("badjoin");
    let mut s = client(port, &root, 5);
    let code = s.run();
    server.join().unwrap();
    assert_eq!(code, BAD_ARGS);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn client_ends_on_a_malformed_select_from_the_server() {
    let (port, server) = fake_server(|s, r| {
        // Complete the lobby, start the duel, then send a truncated select message.
        s.write_all(&P::pack_frame(
            P::STOC_HS_PLAYER_ENTER,
            &P::pack_hs_player_enter("srv", 0),
        ))
        .unwrap();
        s.write_all(&P::pack_frame(P::STOC_JOIN_GAME, &P::HostInfo::default().pack())).unwrap();
        s.write_all(&P::pack_frame(P::STOC_TYPE_CHANGE, &[0x10])).unwrap();
        // Read until READY.
        let mut buf = [0u8; 4096];
        'ready: loop {
            match s.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    for f in r.feed(&buf[..n]).unwrap_or_default() {
                        if f.proto == P::CTOS_HS_READY {
                            break 'ready;
                        }
                    }
                }
            }
        }
        s.write_all(&P::pack_frame(P::STOC_DUEL_START, &[])).unwrap();
        let start = P::build_start(8000, 40, 0, 40, 0, 0);
        s.write_all(&P::pack_frame(P::STOC_GAME_MSG, &start)).unwrap();
        // MSG_SELECT_YESNO needs 1 + 1 + 8 bytes; send three.
        s.write_all(&P::pack_frame(P::STOC_GAME_MSG, &[C::MSG_SELECT_YESNO as u8, 0, 0]))
            .unwrap();
        // Keep the connection open so the client reads the frame before the close.
        std::thread::sleep(Duration::from_millis(300));
    });
    let root = temp_dir("badselect");
    let mut s = client(port, &root, 3);
    let code = s.run();
    server.join().unwrap();
    assert_eq!(code, TIMEOUT);
    assert!(s.record.alerts.iter().any(|a| a.contains("cannot build a decision")), "{:?}", s.record.alerts);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn malformed_frame_bytes_never_panic_the_reader() {
    let mut r = P::FrameReader::new();
    let _ = r.feed(&[0, 0]);
    let _ = r.feed(&[0xFF, 0xFF, 1]);
    for n in 0..64 {
        let junk: Vec<u8> = (0..n).map(|i| (i * 37) as u8).collect();
        let mut r = P::FrameReader::new();
        let _ = r.feed(&junk);
    }
}

#[test]
fn streambot_agent_trait_is_usable() {
    let mut a = StreamBot::new(3);
    assert!(a.act(5) < 5);
    assert!((1..=3).contains(&a.rps()));
}
