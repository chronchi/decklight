//! EDOPro-compatible LAN host, generalized to two seats.
//!
//! A port of `src/decklight/edopro/host.py` (`DuelHost`, `Session`) where each seat is one of:
//! * `Remote`: a TCP EDOPro client (real EDOPro, our Python client, our Rust client);
//! * `Bot`: an in-process agent (`StreamBot` required by the parity gates);
//! * `Local`: an in-process human (the app, A2b-3): the same STOC packets a remote gets go to a channel and
//!   CTOS packets come back through one, so a local and a remote human take the same code path.
//!
//! The authoritative duel is the Rust `Env`. The host filters per seat (`filter`), refreshes the board,
//! sends `MSG_WAITING` while a bot thinks, decodes responses with `inverse`, rejects wrong responses with
//! `MSG_RETRY`, and handles surrender/disconnect. With one remote and one bot it behaves byte for byte like
//! `DuelHost`. At the end it sends the `.yrpX` to every seat (Multirole `SendReplay`: `STOC_NEW_REPLAY` with the
//! replay bytes, or a chat error when it exceeds the u16 payload limit, then `STOC_REPLAY` empty), before
//! `STOC_DUEL_END`. `--no-replay` turns that off for the byte-parity gate against the Python host.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use decklight_cards::{CardData, Deck};
use decklight_core::constants as C;
use serde_json::json;
use ygoenv_sys::{DuelConfig, Engine, Env, Obs, TCG_FLAGS};

use crate::agent::{Agent, RandomAgent};
use crate::filter::{self, Refresh};
use crate::inverse;
use crate::log::PacketLog;
use crate::protocol::{
    self, Frame, FrameReader, PLAYERCHANGE_NOTREADY, PLAYERCHANGE_READY, STOC_DUEL_END,
    STOC_DUEL_START, STOC_ERROR_MSG, STOC_GAME_MSG, STOC_HAND_RESULT, STOC_HS_PLAYER_CHANGE,
    STOC_HS_PLAYER_ENTER, STOC_JOIN_GAME, STOC_NEW_REPLAY, STOC_REPLAY, STOC_SELECT_HAND, STOC_SELECT_TP,
    STOC_TYPE_CHANGE, STOC_CHAT, CTOS_HAND_RESULT, CTOS_HS_NOTREADY, CTOS_HS_READY, CTOS_HS_START,
    CTOS_JOIN_GAME, CTOS_LEAVE_GAME, CTOS_PLAYER_INFO, CTOS_RESPONSE, CTOS_SURRENDER,
    CTOS_TP_RESULT, CTOS_UPDATE_DECK, DERR_INVALIDSIZE, DERR_MAINCOUNT, DERR_UNKNOWNCARD,
    pack_deck_error, pack_frame, parse_update_deck, decode_name, pack_hand_result, pack_hs_player_enter,
    build_start, rps_beats,
};
use crate::record;
use crate::yrp::{self, Yrp};

/// The u16 frame length includes the protocol byte, so the payload of one frame is at most 65534 bytes.
pub const MAX_FRAME_PAYLOAD: usize = 65534;

/// The two frames Multirole's `SendReplay` sends: `STOC_NEW_REPLAY` with the replay, or a chat error when it does
/// not fit the u16 frame length, then `STOC_REPLAY` (empty, the "open replay" prompt).
pub fn replay_frames(data: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let first = if data.len() > MAX_FRAME_PAYLOAD {
        (STOC_CHAT, chat_error("Replay too big to send"))
    } else {
        (STOC_NEW_REPLAY, data.to_vec())
    };
    vec![first, (STOC_REPLAY, Vec::new())]
}

/// Multirole's `STOCMsg::Chat2` system-error packet (type 3, empty client name, UTF-16 message field).
pub fn chat_error(text: &str) -> Vec<u8> {
    const UNITS: usize = 256;
    let mut o = Vec::with_capacity(2 + 40 + UNITS * 2);
    o.push(3); // PTYPE_SYSTEM_ERROR
    o.push(0); // isTeam
    o.extend_from_slice(&[0u8; 40]); // clientName[20]
    let mut msg = Vec::new();
    for u in text.encode_utf16().take(UNITS) {
        msg.extend_from_slice(&u.to_le_bytes());
    }
    msg.resize(UNITS * 2, 0);
    o.extend_from_slice(&msg);
    o
}

pub type SharedLog = Arc<Mutex<PacketLog>>;

#[derive(Debug)]
pub enum NetError {
    Disconnected,
    Fatal(String),
}

impl From<std::io::Error> for NetError {
    fn from(e: std::io::Error) -> NetError {
        match e.kind() {
            std::io::ErrorKind::WouldBlock
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::Interrupted => NetError::Disconnected,
            _ => NetError::Disconnected,
        }
    }
}

/// Why a lobby/duel wait ended.
#[derive(Debug)]
pub enum End {
    Surrender(usize),
    Disconnect(usize),
    Timeout,
    Fatal(String),
}

impl From<End> for String {
    fn from(e: End) -> String {
        format!("{e:?}")
    }
}

impl From<NetError> for End {
    fn from(e: NetError) -> End {
        match e {
            NetError::Disconnected => End::Disconnect(0),
            NetError::Fatal(m) => End::Fatal(m),
        }
    }
}

struct RemoteIo {
    stream: TcpStream,
    reader: FrameReader,
    log: SharedLog,
}

impl RemoteIo {
    fn send(&mut self, proto: u8, payload: &[u8]) -> Result<(), NetError> {
        let bytes = pack_frame(proto, payload);
        self.stream.write_all(&bytes).map_err(|_| NetError::Disconnected)?;
        self.log.lock().unwrap().log("s", proto, payload);
        Ok(())
    }

    fn poll(&mut self) -> Result<Vec<Frame>, NetError> {
        let mut buf = [0u8; 65536];
        match self.stream.read(&mut buf) {
            Ok(0) => Err(NetError::Disconnected),
            Ok(n) => {
                let frames = self.reader.feed(&buf[..n]).map_err(NetError::Fatal)?;
                {
                    let mut log = self.log.lock().unwrap();
                    for f in &frames {
                        log.log("r", f.proto, &f.payload);
                    }
                }
                Ok(frames)
            }
            Err(e) => match e.kind() {
                std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::Interrupted => Ok(Vec::new()),
                _ => Err(NetError::Disconnected),
            },
        }
    }
}

struct LocalIo {
    out: std::sync::mpsc::Sender<Frame>,
    inp: std::sync::mpsc::Receiver<Frame>,
}

impl LocalIo {
    fn send(&mut self, proto: u8, payload: &[u8]) -> Result<(), NetError> {
        self.out
            .send(Frame { proto, payload: payload.to_vec() })
            .map_err(|_| NetError::Disconnected)
    }

    fn poll(&mut self) -> Result<Vec<Frame>, NetError> {
        let mut out = Vec::new();
        loop {
            match self.inp.try_recv() {
                Ok(f) => out.push(f),
                Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(out),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return Err(NetError::Disconnected),
            }
        }
    }
}

enum SeatKind {
    Remote(RemoteIo),
    Local(LocalIo),
    Bot { agent: Box<dyn Agent> },
}

/// One duel seat: lobby position, deck, readiness and its transport (or agent).
pub struct Seat {
    pub pos: u8,
    pub name: String,
    pub deck: Option<Deck>,
    pub main: Vec<u32>,
    pub extra: Vec<u32>,
    pub side: Vec<u32>,
    pub has_deck: bool,
    pub ready: bool,
    /// Sent `CTOS_JOIN_GAME` and was let in: other seats' lobby changes are relayed to it.
    pub joined: bool,
    pub engine_seat: u8,
    /// `remote`, `local` or `bot` (the match record's seat description).
    pub kind_name: &'static str,
    kind: SeatKind,
}

/// The app side of a `Local` seat: STOC frames arrive on `out`, CTOS frames are sent on `inp`.
pub struct LocalPeer {
    pub out: std::sync::mpsc::Receiver<Frame>,
    pub inp: std::sync::mpsc::Sender<Frame>,
}

impl LocalPeer {
    pub fn send(&self, proto: u8, payload: &[u8]) -> Result<(), String> {
        self.inp.send(Frame { proto, payload: payload.to_vec() }).map_err(|_| "local seat closed".to_string())
    }
}

impl Seat {
    pub fn remote(stream: TcpStream, pos: u8, name: String, log: SharedLog) -> Seat {
        Seat {
            pos,
            name,
            deck: None,
            main: Vec::new(),
            extra: Vec::new(),
            side: Vec::new(),
            has_deck: false,
            ready: false,
            joined: false,
            engine_seat: pos,
            kind_name: "remote",
            kind: SeatKind::Remote(RemoteIo { stream, reader: FrameReader::new(), log }),
        }
    }

    pub fn local(pos: u8, name: String) -> (Seat, LocalPeer) {
        let (to_app, out) = std::sync::mpsc::channel();
        let (inp, from_app) = std::sync::mpsc::channel();
        let seat = Seat {
            pos,
            name,
            deck: None,
            main: Vec::new(),
            extra: Vec::new(),
            side: Vec::new(),
            has_deck: false,
            ready: false,
            joined: false,
            engine_seat: pos,
            kind_name: "local",
            kind: SeatKind::Local(LocalIo { out: to_app, inp: from_app }),
        };
        (seat, LocalPeer { out, inp })
    }

    pub fn bot(pos: u8, deck: Deck, name: String, agent: Box<dyn Agent>) -> Seat {
        Seat {
            pos,
            name,
            main: deck.main.clone(),
            extra: deck.extra.clone(),
            side: deck.side.clone(),
            deck: Some(deck),
            has_deck: true,
            ready: true,
            joined: false,
            engine_seat: pos,
            kind_name: "bot",
            kind: SeatKind::Bot { agent },
        }
    }

    pub fn is_bot(&self) -> bool {
        matches!(self.kind, SeatKind::Bot { .. })
    }

    pub fn is_net(&self) -> bool {
        !self.is_bot()
    }

    pub fn send(&mut self, proto: u8, payload: &[u8]) -> Result<(), End> {
        let r = match &mut self.kind {
            SeatKind::Remote(io) => io.send(proto, payload),
            SeatKind::Local(io) => io.send(proto, payload),
            SeatKind::Bot { .. } => Ok(()),
        };
        r.map_err(|e| match e {
            NetError::Disconnected => End::Disconnect(self.pos as usize),
            NetError::Fatal(m) => End::Fatal(m),
        })
    }

    fn poll(&mut self) -> Result<Vec<Frame>, End> {
        let r = match &mut self.kind {
            SeatKind::Remote(io) => io.poll(),
            SeatKind::Local(io) => io.poll(),
            SeatKind::Bot { .. } => Ok(Vec::new()),
        };
        r.map_err(|e| match e {
            NetError::Disconnected => End::Disconnect(self.pos as usize),
            NetError::Fatal(m) => End::Fatal(m),
        })
    }

    fn agent_mut(&mut self) -> Option<&mut Box<dyn Agent>> {
        match &mut self.kind {
            SeatKind::Bot { agent } => Some(agent),
            _ => None,
        }
    }
}

/// Agent choice for a bot seat (a fresh stream per duel; gates run one duel per host process).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentKind {
    Stream,
    Random,
}

#[derive(Clone, Debug)]
pub struct HostConfig {
    pub bind: String,
    pub port: u16,
    pub name: String,
    pub once: bool,
    pub max_duels: usize,
    pub remote_seats: usize,
    pub seed: Option<u64>,
    pub agent: AgentKind,
    pub agent_seed: u64,
    pub bot_second: bool,
    pub record_root: PathBuf,
    pub send_replay: bool,
    pub compress_replay: bool,
    pub packet_log: Option<PathBuf>,
    pub discovery: bool,
    /// Room password (empty = none). A wrong one gets Multirole's join error and ends that lobby.
    pub password: String,
    pub accept_timeout: Duration,
    pub read_timeout: Duration,
    pub duel_timeout: Duration,
}

impl Default for HostConfig {
    fn default() -> Self {
        HostConfig {
            bind: "127.0.0.1".into(),
            port: 7911,
            name: "ygo-zero".into(),
            once: false,
            max_duels: 0,
            remote_seats: 1,
            seed: None,
            agent: AgentKind::Stream,
            agent_seed: 0,
            bot_second: false,
            record_root: PathBuf::from("runs/edopro"),
            send_replay: true,
            compress_replay: true,
            packet_log: None,
            discovery: true,
            password: String::new(),
            accept_timeout: Duration::from_secs(120),
            read_timeout: Duration::from_millis(1),
            duel_timeout: Duration::from_secs(120),
        }
    }
}

pub struct DuelHost<'e> {
    engine: &'e Engine,
    cards: &'e BTreeMap<u32, CardData>,
    emb_codes: Option<BTreeSet<u32>>,
    pub config: HostConfig,
    bot_deck: Option<Deck>,
    bot_name: String,
    rng: RandomAgent,
    log: SharedLog,
}

impl<'e> DuelHost<'e> {
    pub fn new(
        engine: &'e Engine,
        cards: &'e BTreeMap<u32, CardData>,
        emb_codes: Option<BTreeSet<u32>>,
        config: HostConfig,
        bot_deck: Option<Deck>,
        bot_name: String,
    ) -> DuelHost<'e> {
        let log = match &config.packet_log {
            Some(p) => match PacketLog::open(p) {
                Ok(l) => Arc::new(Mutex::new(l)),
                Err(e) => {
                    eprintln!("ALERT edopro_host: cannot open packet log: {e}");
                    Arc::new(Mutex::new(PacketLog::disabled()))
                }
            },
            None => Arc::new(Mutex::new(PacketLog::disabled())),
        };
        DuelHost {
            engine,
            cards,
            emb_codes,
            config,
            bot_deck,
            bot_name,
            rng: RandomAgent::from_time(),
            log,
        }
    }

    fn make_agent(&self) -> Box<dyn Agent> {
        match self.config.agent {
            AgentKind::Stream => Box::new(crate::agent::StreamBot::new(self.config.agent_seed)),
            AgentKind::Random => Box::new(RandomAgent::new(self.config.agent_seed ^ 0xA5A5)),
        }
    }


    fn deal_seed(&mut self) -> u64 {
        self.config.seed.unwrap_or_else(|| self.rng.act(u64::MAX as usize) as u64 >> 1)
    }

    /// Bind, serve `remote_seats` connections (plus a bot seat when `bot_deck` is set) per duel.
    pub fn serve(&mut self) -> Result<(), String> {
        let listener = TcpListener::bind((self.config.bind.as_str(), self.config.port))
            .map_err(|e| format!("bind {}:{}: {e}", self.config.bind, self.config.port))?;
        self.serve_on(listener, Arc::new(std::sync::atomic::AtomicBool::new(false)))
    }

    /// `serve` on an already bound listener until `stop` is set (checked while accepting; the app's room host,
    /// which picks a free port itself and closes the room on demand).
    pub fn serve_on(&mut self, listener: TcpListener, stop: Arc<std::sync::atomic::AtomicBool>) -> Result<(), String> {
        let discovery = if self.config.discovery {
            Some(crate::discovery::spawn_discovery(self.config.port, self.config.name.clone(), stop.clone()))
        } else {
            None
        };
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        println!(
            "ygo-zero host on {}:{} (room '{}'); {} remote seat(s), {}",
            if self.config.bind.is_empty() { "0.0.0.0" } else { &self.config.bind },
            self.config.port,
            self.config.name,
            self.config.remote_seats,
            if self.bot_deck.is_some() { "one bot seat" } else { "no bot seat" },
        );
        let mut duels = 0usize;
        let result = loop {
            let mut seats = Vec::new();
            let deadline = Instant::now() + self.config.accept_timeout;
            while seats.len() < self.config.remote_seats {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                match listener.accept() {
                    Ok((s, addr)) => {
                        println!("connection from {addr} (seat {})", seats.len());
                        let _ = s.set_nodelay(true);
                        s.set_read_timeout(Some(self.config.read_timeout)).ok();
                        seats.push(Seat::remote(s, seats.len() as u8, "human".into(), self.log.clone()));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() >= deadline {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_e) => break,
                }
            }
            if stop.load(Ordering::Relaxed) {
                break Ok(());
            }
            if seats.len() < self.config.remote_seats {
                let _ = listener.set_nonblocking(true);
                if self.config.once {
                    break Err(format!("timed out accepting {} remote seat(s)", self.config.remote_seats));
                }
                continue;
            }
            let _ = listener.set_nonblocking(false);
            if let Some(deck) = &self.bot_deck {
                seats.push(Seat::bot(seats.len() as u8, deck.clone(), self.bot_name.clone(), self.make_agent()));
            }
            match self.run_duel_with_seats(seats) {
                Ok(Some(_)) => {
                    duels += 1;
                    if self.config.once || (self.config.max_duels > 0 && duels >= self.config.max_duels) {
                        break Ok(());
                    }
                }
                Ok(None) => { /* no duel (left in the lobby); keep serving */ }
                Err(e) => {
                    eprintln!("ALERT edopro_host: {e}");
                    if self.config.once {
                        break Err(e);
                    }
                }
            }
        };
        stop.store(true, Ordering::Relaxed);
        if let Some(h) = discovery {
            let _ = h.join();
        }
        let _ = self.log.lock().unwrap().finish();
        result
    }

    /// Run one duel with already-built seats (the test/Local entry point). `None` = no duel was played.
    pub fn run_duel_with_seats(&mut self, mut seats: Vec<Seat>) -> Result<Option<PathBuf>, String> {
        self.lobby(&mut seats)?;
        self.play(&mut seats)
    }

    // --- lobby ---

    fn all_ready(&self, seats: &[Seat]) -> bool {
        seats.iter().all(|s| s.ready)
    }

    fn lobby(&mut self, seats: &mut [Seat]) -> Result<(), String> {
        let deadline = Instant::now() + self.config.accept_timeout;
        loop {
            if self.all_ready(seats) {
                return Ok(());
            }
            let mut batch: Vec<(usize, Frame)> = Vec::new();
            for i in 0..seats.len() {
                if !seats[i].is_net() {
                    continue;
                }
                match seats[i].poll() {
                    Ok(frames) => batch.extend(frames.into_iter().map(|f| (i, f))),
                    Err(End::Disconnect(_)) => {
                        return Err(format!("seat {} ({}) left in the lobby", i, seats[i].name))
                    }
                    Err(End::Fatal(e)) => return Err(e),
                    Err(e) => return Err(format!("seat {i} lobby: {e:?}")),
                }
            }
            for (i, f) in batch {
                if let Err(e) = self.on_lobby(seats, i, f.proto, &f.payload) {
                    return Err(e);
                }
                if self.all_ready(seats) {
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err("timed out waiting for the lobby".into());
            }
            if seats.iter().filter(|s| s.is_net()).all(|s| s.ready) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn on_lobby(&self, seats: &mut [Seat], i: usize, proto: u8, payload: &[u8]) -> Result<(), String> {
        match proto {
            CTOS_PLAYER_INFO => {
                let n = payload.len().min(40);
                let name = decode_name(&payload[..n]);
                seats[i].name = if name.is_empty() { "human".into() } else { name };
            }
            CTOS_JOIN_GAME => {
                if payload.len() < 50 {
                    return Ok(());
                }
                let Ok((_ver, _gameid, version2)) = protocol::unpack_join_game(payload) else {
                    return Ok(());
                };
                if !self.config.password.is_empty() && decode_name(&payload[8..48]) != self.config.password {
                    seats[i].send(STOC_ERROR_MSG, &protocol::pack_join_error(1))?;
                    return Err(format!("seat {i} sent a wrong room password"));
                }
                let info = protocol::host_info(version2);
                seats[i].send(STOC_HS_PLAYER_ENTER, &pack_hs_player_enter(&seats[i].name, seats[i].pos))?;
                seats[i].send(STOC_JOIN_GAME, &info.pack())?;
                let host_bit = if seats[i].pos == 0 { 0x10 } else { 0x00 };
                seats[i].send(STOC_TYPE_CHANGE, &[host_bit | seats[i].pos])?;
                // Multirole tells everyone in the room who is there: the newcomer about the others (and whether
                // they are ready), the others about the newcomer. Only other joined remote/local seats, so a
                // one-human-plus-bot duel sends exactly the packets it always did (the Python byte-parity gate).
                seats[i].joined = true;
                for j in 0..seats.len() {
                    if j == i || !seats[j].is_net() || !seats[j].joined {
                        continue;
                    }
                    let (name_i, pos_i) = (seats[i].name.clone(), seats[i].pos);
                    seats[j].send(STOC_HS_PLAYER_ENTER, &pack_hs_player_enter(&name_i, pos_i))?;
                    let (name_j, pos_j, ready_j) = (seats[j].name.clone(), seats[j].pos, seats[j].ready);
                    seats[i].send(STOC_HS_PLAYER_ENTER, &pack_hs_player_enter(&name_j, pos_j))?;
                    if ready_j {
                        seats[i].send(STOC_HS_PLAYER_CHANGE, &[(pos_j << 4) | PLAYERCHANGE_READY])?;
                    }
                }
            }
            CTOS_UPDATE_DECK => {
                let Ok((codes, side)) = parse_update_deck(payload) else {
                    seats[i].send(
                        STOC_ERROR_MSG,
                        &pack_deck_error(DERR_INVALIDSIZE, 0, 0, 0, 0),
                    )?;
                    return Ok(());
                };
                match split_deck(&codes, self.cards) {
                    Ok((main, extra)) => {
                        seats[i].main = main;
                        seats[i].extra = extra;
                        seats[i].side = side;
                        seats[i].has_deck = true;
                    }
                    Err(code) => {
                        seats[i].send(
                            STOC_ERROR_MSG,
                            &pack_deck_error(DERR_UNKNOWNCARD, 0, 0, 0, code),
                        )?;
                    }
                }
            }
            CTOS_HS_READY => {
                self.player_ready(seats, i)?;
            }
            CTOS_HS_NOTREADY => {
                seats[i].ready = false;
                let change = [(seats[i].pos << 4) | PLAYERCHANGE_NOTREADY];
                seats[i].send(STOC_HS_PLAYER_CHANGE, &change)?;
                Self::tell_others(seats, i, &change)?;
            }
            CTOS_HS_START => {
                // Started by the lobby loop once every seat is ready (a joiner does not send this).
            }
            CTOS_LEAVE_GAME => {
                return Err(format!("seat {i} left the lobby"));
            }
            _ => {}
        }
        Ok(())
    }

    fn player_ready(&self, seats: &mut [Seat], i: usize) -> Result<(), String> {
        let err = if !seats[i].has_deck {
            Some((DERR_MAINCOUNT, 0, 40, 60, 0))
        } else {
            match check_deck(&seats[i].main, &seats[i].extra, self.cards, self.emb_codes.as_ref()) {
                Some(e) => Some(e),
                None => None,
            }
        };
        if let Some((t, current, minimum, maximum, code)) = err {
            seats[i].send(
                STOC_HS_PLAYER_CHANGE,
                &[(seats[i].pos << 4) | PLAYERCHANGE_NOTREADY],
            )?;
            seats[i].send(STOC_ERROR_MSG, &pack_deck_error(t, current, minimum, maximum, code))?;
            return Ok(());
        }
        seats[i].ready = true;
        let change = [(seats[i].pos << 4) | PLAYERCHANGE_READY];
        seats[i].send(STOC_HS_PLAYER_CHANGE, &change)?;
        Self::tell_others(seats, i, &change)?;
        Ok(())
    }

    /// A lobby `STOC_HS_PLAYER_CHANGE` about seat `i`, relayed to the other joined remote/local seats.
    fn tell_others(seats: &mut [Seat], i: usize, change: &[u8]) -> Result<(), End> {
        for j in 0..seats.len() {
            if j != i && seats[j].is_net() && seats[j].joined {
                seats[j].send(STOC_HS_PLAYER_CHANGE, change)?;
            }
        }
        Ok(())
    }

    // --- RPS / TP ---

    fn wait_any<F>(
        &self,
        seats: &mut [Seat],
        mut want: F,
        deadline: Instant,
    ) -> Result<(usize, u8, Vec<u8>), End>
    where
        F: FnMut(u8) -> bool,
    {
        loop {
            for i in 0..seats.len() {
                if !seats[i].is_net() {
                    continue;
                }
                let frames = seats[i].poll()?;
                for f in frames {
                    if f.proto == CTOS_SURRENDER {
                        return Err(End::Surrender(i));
                    }
                    if f.proto == CTOS_LEAVE_GAME {
                        return Err(End::Disconnect(i));
                    }
                    if want(f.proto) {
                        return Ok((i, f.proto, f.payload));
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err(End::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn rps_tp(&mut self, seats: &mut [Seat]) -> Result<usize, End> {
        let deadline = Instant::now() + self.config.duel_timeout;
        let n = seats.len();
        loop {
            let mut throws = vec![0u8; n];
            // Bots throw from their stream first (DuelHost.bot_rps() before prompting the human).
            for i in 0..n {
                if let Some(a) = seats[i].agent_mut() {
                    throws[i] = a.rps();
                }
            }
            for i in 0..n {
                if seats[i].is_net() {
                    seats[i].send(STOC_SELECT_HAND, &[])?;
                }
            }
            let mut got = vec![false; n];
            let mut remaining = (0..n).filter(|&i| seats[i].is_net()).count();
            while remaining > 0 {
                let (i, proto, payload) = self.wait_any(seats, |p| p == CTOS_HAND_RESULT, deadline)?;
                if proto == CTOS_HAND_RESULT && seats[i].is_net() && !got[i] {
                    throws[i] = payload.first().copied().unwrap_or(0);
                    got[i] = true;
                    remaining -= 1;
                }
            }
            if throws.iter().any(|&t| !(1..=3).contains(&t)) {
                continue; // invalid throw: re-prompt like generic_duel
            }
            for i in 0..n {
                if seats[i].is_net() {
                    let other = throws[1 - i];
                    seats[i].send(STOC_HAND_RESULT, &pack_hand_result(throws[i], other))?;
                }
            }
            if throws[0] == throws[1] {
                continue;
            }
            let winner = if rps_beats(throws[0], throws[1]) { 0 } else { 1 };
            let winner_first = if seats[winner].is_bot() {
                !self.config.bot_second
            } else {
                seats[winner].send(STOC_SELECT_TP, &[])?;
                let (_i, _p, payload) = self.wait_any(seats, |p| p == CTOS_TP_RESULT, deadline)?;
                payload.first().copied().unwrap_or(0) == 1
            };
            return Ok(if winner_first { winner } else { 1 - winner });
        }
    }

    // --- duel ---

    fn send_refresh(&self, seats: &mut [Seat], env: &Env<'_>, r: Refresh) -> Result<(), End> {
        for i in 0..seats.len() {
            if !seats[i].is_net() {
                continue;
            }
            let viewer = seats[i].engine_seat;
            if let Some(p) = filter::refresh_packet(env, r, viewer).map_err(End::Fatal)? {
                seats[i].send(STOC_GAME_MSG, &p)?;
            }
        }
        Ok(())
    }

    fn send_msg(&self, seats: &mut [Seat], payload: &[u8], only_bot_acting: bool) -> Result<(), End> {
        for s in seats.iter_mut() {
            if s.is_net() && !(only_bot_acting && s.is_bot()) {
                s.send(STOC_GAME_MSG, payload)?;
            }
        }
        Ok(())
    }

    fn pump(&self, seats: &mut [Seat], env: &Env<'_>, sent: &mut usize) -> Result<(), End> {
        let n = env.trace_len();
        for i in *sent..n {
            let raw = env.trace_msg(i);
            for r in filter::before_refreshes(&raw).map_err(End::Fatal)? {
                self.send_refresh(seats, env, r)?;
            }
            for si in 0..seats.len() {
                if !seats[si].is_net() {
                    continue;
                }
                let es = seats[si].engine_seat;
                if let Some(msg) = filter::filtered_message(&raw, es).map_err(End::Fatal)? {
                    seats[si].send(STOC_GAME_MSG, &msg)?;
                }
            }
            for r in filter::after_refreshes(&raw).map_err(End::Fatal)? {
                self.send_refresh(seats, env, r)?;
            }
        }
        *sent = n;
        Ok(())
    }

    fn step_env(
        env: &mut Env<'_>,
        obs: &mut Obs,
        index: i32,
        responses: &mut Vec<record::ResponseRec>,
        cur_opts: &mut Vec<i32>,
    ) -> Result<bool, String> {
        let player = env.player();
        let before = env.num_responses();
        let pending = env.step(index, obs)?;
        cur_opts.push(index);
        let after = env.num_responses();
        for k in before..after {
            responses.push(record::ResponseRec {
                seat: player,
                r: env.response(k),
                opt: std::mem::take(cur_opts),
                ms: 0,
            });
        }
        Ok(pending)
    }

    fn play(&mut self, seats: &mut [Seat]) -> Result<Option<PathBuf>, String> {
        if seats.len() != 2 {
            return Err(format!("a duel needs exactly two seats, got {}", seats.len()));
        }
        for s in seats.iter_mut() {
            if s.is_net() {
                s.send(STOC_DUEL_START, &[])?;
            }
        }
        let first = match self.rps_tp(seats) {
            Ok(p) => p,
            Err(End::Surrender(i)) => {
                println!("seat {} surrendered before the duel", seats[i].name);
                return Ok(None);
            }
            Err(End::Disconnect(i)) => {
                println!("seat {} disconnected before the duel", seats[i].name);
                return Ok(None);
            }
            Err(End::Timeout) => return Err("timed out in RPS/TP".into()),
            Err(End::Fatal(e)) => return Err(e),
        };
        let second = 1 - first;
        seats[first].engine_seat = 0;
        seats[second].engine_seat = 1;

        let seed = self.deal_seed();
        println!("deal seed {seed} (first seat {first})");
        let mut main0 = seats[first].main.clone();
        let mut main1 = seats[second].main.clone();
        decklight_cards::shuffle_main(&mut main0, seed, 0);
        decklight_cards::shuffle_main(&mut main1, seed, 1);
        let extra0 = seats[first].extra.clone();
        let extra1 = seats[second].extra.clone();
        let cfg = DuelConfig {
            main0: main0.clone(),
            extra0: extra0.clone(),
            main1: main1.clone(),
            extra1: extra1.clone(),
            seed,
            flags: TCG_FLAGS,
            lp: 8000,
            hand: 5,
            draw: 1,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        };
        let mut env = Env::new(self.engine, &cfg, false).map_err(|e| format!("could not start the duel: {e}"))?;
        env.set_record_trace(true);
        let mut obs = Obs::zeroed();
        let mut pending = env.reset(&mut obs)?;

        let start0 = build_start(cfg.lp, main0.len() as u32, extra0.len() as u32, main1.len() as u32, extra1.len() as u32, 0);
        let start1 = build_start(cfg.lp, main0.len() as u32, extra0.len() as u32, main1.len() as u32, extra1.len() as u32, 1);
        for i in 0..seats.len() {
            if !seats[i].is_net() {
                continue;
            }
            let msg = if seats[i].engine_seat == 0 { &start0 } else { &start1 };
            seats[i].send(STOC_GAME_MSG, msg)?;
        }
        for seat in 0..2u8 {
            for i in 0..seats.len() {
                if !seats[i].is_net() {
                    continue;
                }
                let r = Refresh::Loc { player: seat, location: C::LOCATION_EXTRA };
                if let Some(p) = filter::refresh_packet(&env, r, seats[i].engine_seat).map_err(|e| e.to_string())? {
                    seats[i].send(STOC_GAME_MSG, &p)?;
                }
            }
        }

        let mut sent_trace = 0usize;
        let mut responses: Vec<record::ResponseRec> = Vec::new();
        let mut cur_opts: Vec<i32> = Vec::new();
        let mut surrendered = false;
        let mut disconnected = false;
        let mut disconnect_winner: Option<i32> = None;
        let mut replay_win: Option<Vec<u8>> = None;
        let deadline = Instant::now() + self.config.duel_timeout;
        let mut fatal: Option<String> = None;

        'duel: while pending {
            if let Err(e) = self.pump(seats, &env, &mut sent_trace) {
                match e {
                    End::Disconnect(i) => {
                        disconnect_winner = Some(seats[1 - i].engine_seat as i32);
                        disconnected = true;
                        break 'duel;
                    }
                    End::Surrender(i) => {
                        surrendered = true;
                        replay_win = Some(vec![C::MSG_WIN as u8, seats[1 - i].engine_seat, 0]);
                        break 'duel;
                    }
                    End::Fatal(m) => {
                        fatal = Some(m);
                        break 'duel;
                    }
                    End::Timeout => {
                        fatal = Some("timed out".into());
                        break 'duel;
                    }
                }
            }
            if env.done() {
                break;
            }
            let acting_engine = env.player() as u8;
            let acting = match seats.iter().position(|s| s.engine_seat == acting_engine) {
                Some(i) => i,
                None => {
                    fatal = Some(format!("no seat for engine player {acting_engine}"));
                    break;
                }
            };
            if seats[acting].is_bot() {
                let waiting = [C::MSG_WAITING as u8];
                if let Err(e) = self.send_msg(seats, &waiting, false) {
                    match e {
                        End::Disconnect(i) => {
                            disconnect_winner = Some(seats[1 - i].engine_seat as i32);
                            disconnected = true;
                        }
                        End::Surrender(i) => {
                            surrendered = true;
                            replay_win = Some(vec![C::MSG_WIN as u8, seats[1 - i].engine_seat, 0]);
                        }
                        End::Fatal(m) => fatal = Some(m),
                        End::Timeout => fatal = Some("timed out".into()),
                    }
                    break;
                }
                let n = env.num_options() as usize;
                let idx = {
                    let a = seats[acting].agent_mut().expect("bot seat without agent");
                    a.act(n)
                };
                match Self::step_env(&mut env, &mut obs, idx as i32, &mut responses, &mut cur_opts) {
                    Ok(p) => pending = p,
                    Err(e) => {
                        fatal = Some(e);
                        break;
                    }
                }
            } else {
                loop {
                    let (i, _proto, payload) = match self.wait_any(
                        seats,
                        |p| p == CTOS_RESPONSE,
                        deadline,
                    ) {
                        Ok(x) => x,
                        Err(End::Surrender(i)) => {
                            surrendered = true;
                            replay_win = Some(vec![C::MSG_WIN as u8, seats[1 - i].engine_seat, 0]);
                            break 'duel;
                        }
                        Err(End::Disconnect(i)) => {
                            disconnect_winner = Some(seats[1 - i].engine_seat as i32);
                            disconnected = true;
                            break 'duel;
                        }
                        Err(End::Timeout) => {
                            // A silent peer is treated like a disconnect of the acting seat.
                            disconnect_winner = Some(seats[1 - acting].engine_seat as i32);
                            disconnected = true;
                            break 'duel;
                        }
                        Err(End::Fatal(m)) => {
                            fatal = Some(m);
                            break 'duel;
                        }
                    };
                    if i != acting {
                        continue; // a response from the other seat: ignore
                    }
                    let dec = match ygoenv_sys::Decision::from_env(&env) {
                        Ok(Some(d)) => d,
                        Ok(None) => {
                            fatal = Some("duel is awaiting a response but sent no select message".into());
                            break 'duel;
                        }
                        Err(e) => {
                            fatal = Some(e);
                            break 'duel;
                        }
                    };
                    match inverse::choices_for_response(&dec, &payload, false) {
                        Some(seq) => {
                            for idx in seq {
                                match Self::step_env(
                                    &mut env,
                                    &mut obs,
                                    idx as i32,
                                    &mut responses,
                                    &mut cur_opts,
                                ) {
                                    Ok(p) => pending = p,
                                    Err(e) => {
                                        fatal = Some(e);
                                        break 'duel;
                                    }
                                }
                            }
                            break;
                        }
                        None => {
                            let retry = [C::MSG_RETRY as u8];
                            if let Err(e) = seats[acting].send(STOC_GAME_MSG, &retry) {
                                if let End::Disconnect(i) = e {
                                    disconnect_winner = Some(seats[1 - i].engine_seat as i32);
                                    disconnected = true;
                                } else if let End::Surrender(i) = e {
                                    surrendered = true;
                                    replay_win =
                                        Some(vec![C::MSG_WIN as u8, seats[1 - i].engine_seat, 0]);
                                } else {
                                    fatal = Some(format!("{e:?}"));
                                }
                                break 'duel;
                            }
                        }
                    }
                }
            }
        }

        // Final messages (MSG_WIN) after the last step.
        let _ = self.pump(seats, &env, &mut sent_trace);
        if surrendered {
            if let Some(m) = &replay_win {
                for i in 0..seats.len() {
                    if seats[i].is_net() {
                        let _ = seats[i].send(STOC_GAME_MSG, m);
                    }
                }
            }
        }

        let state = env.duel_state().unwrap_or(ygoenv_sys::DuelState {
            turn: 0,
            turn_player: 0,
            first_player: 0,
            phase: 0,
            winner: 0,
            win_reason: 0,
            select_hint: 0,
        });
        let winner = if surrendered {
            replay_win.as_ref().and_then(|m| m.get(1)).map(|w| *w as i32)
        } else if disconnected {
            disconnect_winner
        } else if state.winner == 0 || state.winner == 1 {
            Some(state.winner)
        } else {
            None
        };
        let mut turns = state.turn;
        if turns < 0 {
            turns = 0;
        }

        // Replay: Multirole sends it to every seat at duel end. A disconnect has no peer left to send to.
        if self.config.send_replay && !disconnected {
            let mut core_msgs: Vec<Vec<u8>> = (0..env.trace_len()).map(|i| env.trace_msg(i)).collect();
            if let Some(m) = &replay_win {
                core_msgs.push(m.clone());
            }
            let all_responses: Vec<Vec<u8>> = (0..env.num_responses()).map(|i| env.response(i)).collect();
            // Multirole stores each duelist under its *core* team (`nci.team = GetSwappedTeam(t)`), so the
            // replay's deck 0 is the deck the core added as team 0 (our `cfg.deck0`, the seat that went first).
            // The card-add order is not stored; `to_record` tries both (ADR 0010). Main decks are written in
            // EDOPro's add order (every card at sequence 0), which rebuilds our deck (`yrp::edopro_main`).
            let names = vec![
                vec![seats[first].name.clone()],
                vec![seats[second].name.clone()],
            ];
            let y = Yrp {
                seed: record::seed_words(seed),
                names,
                lp: cfg.lp,
                hand: cfg.hand,
                draw: cfg.draw,
                flags: cfg.flags,
                decks: vec![
                    vec![(yrp::edopro_main(&main0), extra0.clone())],
                    vec![(yrp::edopro_main(&main1), extra1.clone())],
                ],
                extra_cards: Vec::new(),
                responses: all_responses,
            };
            let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
            match yrp::write_yrpx(&y, &core_msgs, ts, self.config.compress_replay) {
                Ok(data) => {
                    let frames = replay_frames(&data);
                    let mut send_err = None;
                    for i in 0..seats.len() {
                        if !seats[i].is_net() {
                            continue;
                        }
                        for (proto, payload) in &frames {
                            if let Err(e) = seats[i].send(*proto, payload) {
                                send_err = Some(e);
                                break;
                            }
                        }
                        if send_err.is_some() {
                            break;
                        }
                    }
                    if let Some(e) = send_err {
                        println!("replay delivery ended: {e:?}");
                    }
                }
                Err(e) => eprintln!("ALERT edopro_host: cannot build the replay: {e}"),
            }
        }

        if !disconnected {
            for i in 0..seats.len() {
                if !seats[i].is_net() {
                    continue;
                }
                if let Err(e) = seats[i].send(STOC_DUEL_END, &[]) {
                    println!("duel end delivery ended: {e:?}");
                }
            }
        }

        let path = self.write_record(
            &cfg,
            seed,
            first,
            seats,
            &responses,
            winner,
            state.win_reason,
            turns,
            surrendered,
            disconnected,
            Some(0),
        )?;
        if let Some(e) = fatal {
            eprintln!("ALERT edopro_host: duel stopped: {e}");
        }
        Ok(Some(path))
    }

    #[allow(clippy::too_many_arguments)]
    fn write_record(
        &self,
        cfg: &DuelConfig,
        seed: u64,
        first: usize,
        seats: &[Seat],
        responses: &[record::ResponseRec],
        winner: Option<i32>,
        win_reason: i32,
        turns: i32,
        surrendered: bool,
        disconnected: bool,
        first_player: Option<i32>,
    ) -> Result<PathBuf, String> {
        let mut seat_desc: [serde_json::Value; 2] = [json!({}), json!({})];
        for s in seats {
            let v = match s.kind_name {
                "remote" => json!({"type": "remote", "client": "unknown"}),
                "local" => json!({"type": "human"}),
                _ => json!({"type": "bot"}),
            };
            if (s.engine_seat as usize) < 2 {
                seat_desc[s.engine_seat as usize] = v;
            }
        }
        let second = 1 - first;
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut renv = record::RecordEnv::new(&root, &self.config.record_root);
        renv.app = format!("decklight-edopro-host/{}", env!("CARGO_PKG_VERSION"));
        let g = record::RecordGame {
            seed,
            first: Some(0),
            main0: cfg.main0.clone(),
            extra0: cfg.extra0.clone(),
            main1: cfg.main1.clone(),
            extra1: cfg.extra1.clone(),
            side0: seats[first].side.clone(),
            side1: seats[second].side.clone(),
            seats: seat_desc,
            mode: "lan".into(),
            responses: responses.to_vec(),
            winner,
            win_reason,
            turns,
            surrendered,
            disconnected,
            first_player,
            ..Default::default()
        };
        let path = renv.write(&g)?;
        println!(
            "record -> {} (winner {:?}, {} turns)",
            path.display(),
            winner,
            turns
        );
        Ok(path)
    }
}

/// UpdateDeck mainc block -> (main, extra) by card type (rituals stay main). `Err(code)` = unknown card.
pub fn split_deck(codes: &[u32], cards: &BTreeMap<u32, CardData>) -> Result<(Vec<u32>, Vec<u32>), u32> {
    let extra_types = C::TYPE_FUSION | C::TYPE_SYNCHRO | C::TYPE_XYZ;
    let mut main = Vec::new();
    let mut extra = Vec::new();
    for &c in codes {
        let Some(card) = cards.get(&c) else { return Err(c) };
        let t = card.type_;
        if t & extra_types != 0 || (t & C::TYPE_LINK != 0 && t & C::TYPE_MONSTER != 0) {
            extra.push(c);
        } else {
            main.push(c);
        }
    }
    Ok((main, extra))
}

/// DeckError args `(type, current, minimum, maximum, code)` or `None` when legal.
pub fn check_deck(
    main: &[u32],
    extra: &[u32],
    cards: &BTreeMap<u32, CardData>,
    emb_codes: Option<&BTreeSet<u32>>,
) -> Option<(u32, u32, u32, u32, u32)> {
    if !(40..=60).contains(&main.len()) {
        return Some((DERR_MAINCOUNT, main.len() as u32, 40, 60, 0));
    }
    if extra.len() > 15 {
        return Some((7, extra.len() as u32, 0, 15, 0)); // DERR_EXTRACOUNT
    }
    for &c in main.iter().chain(extra) {
        if !cards.contains_key(&c) || emb_codes.is_some_and(|e| !e.contains(&c)) {
            return Some((DERR_UNKNOWNCARD, 0, 0, 0, c));
        }
    }
    None
}

/// The room note the app's host sets by default (ADR 0010 §5).
pub const DEFAULT_ROOM_NOTES: &str = "games may be shared anonymously for research";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_notes_match_the_adr_line() {
        assert_eq!(DEFAULT_ROOM_NOTES, "games may be shared anonymously for research");
    }

    #[test]
    fn check_deck_counts() {
        let cards: BTreeMap<u32, CardData> = (1..=100)
            .map(|c| {
                (
                    c,
                    CardData {
                        code: c,
                        alias: 0,
                        setcodes: vec![],
                        type_: C::TYPE_MONSTER | C::TYPE_NORMAL,
                        level: 4,
                        attribute: 1,
                        race: 1,
                        attack: 1000,
                        defense: 1000,
                        lscale: 0,
                        rscale: 0,
                        link_marker: 0,
                    },
                )
            })
            .collect();
        let main: Vec<u32> = (1..=40).collect();
        assert!(check_deck(&main, &[], &cards, None).is_none());
        assert_eq!(check_deck(&main[..39], &[], &cards, None).unwrap().0, DERR_MAINCOUNT);
        assert_eq!(check_deck(&main, &[999], &cards, None).unwrap().0, DERR_UNKNOWNCARD);
    }

    #[test]
    fn split_deck_by_type() {
        let mut cards = BTreeMap::new();
        let mk = |code: u32, t: u32| CardData {
            code,
            alias: 0,
            setcodes: vec![],
            type_: t,
            level: 4,
            attribute: 1,
            race: 1,
            attack: 0,
            defense: 0,
            lscale: 0,
            rscale: 0,
            link_marker: 0,
        };
        cards.insert(1, mk(1, C::TYPE_MONSTER));
        cards.insert(2, mk(2, C::TYPE_FUSION | C::TYPE_MONSTER));
        cards.insert(3, mk(3, C::TYPE_LINK | C::TYPE_MONSTER));
        let (main, extra) = split_deck(&[1, 2, 3], &cards).unwrap();
        assert_eq!(main, vec![1]);
        assert_eq!(extra, vec![2, 3]);
        assert_eq!(split_deck(&[9], &cards), Err(9));
    }

    #[test]
    fn replay_limit_is_the_u16_frame_size() {
        assert_eq!(MAX_FRAME_PAYLOAD, 65534);
        fn pick(n: usize) -> bool {
            n > MAX_FRAME_PAYLOAD
        }
        assert!(!pick(MAX_FRAME_PAYLOAD));
        assert!(pick(MAX_FRAME_PAYLOAD + 1));
    }
}
