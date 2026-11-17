//! A2b-3c rooms: a person against a person over the EDOPro protocol (spec `docs/ui/mocks/room-ui.md`, ADR 0010 §5).
//!
//! * **Host**: the app runs a `DuelHost` (two remote seats, LAN discovery on) on a free port from 7911, then joins
//!   it itself over 127.0.0.1 like any other client. EDOPro desktop and other decklight apps join the same port.
//! * **Join**: a `RoomSession` connects to `host:port` (our host, EDOPro's LAN host or a Multirole room).
//!
//! Either way the app is an ordinary EDOPro client: a `RoomSession` thread speaks CTOS/STOC on the socket, keeps
//! the lobby state for `room` frames, and during the duel folds every `STOC_GAME_MSG` into a `StreamSource`, from
//! which the usual `state` frames are built (view, prompt, log, cues), as the bot duel builds them from its `Env`.
//! Singleton decisions and quiet chain windows are answered for the user, as in a bot duel with chain "auto".
//! Learning features (hints, win chance, undo, what-if) are off in rooms.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use decklight_core::constants as C;
use decklight_core::messages::{self as M, Body as MsgBody};
use decklight_edopro::host::{DuelHost, HostConfig};
use decklight_edopro::protocol::{
    self, CTOS_HAND_RESULT, CTOS_HS_NOTREADY, CTOS_HS_READY, CTOS_JOIN_GAME, CTOS_LEAVE_GAME, CTOS_PLAYER_INFO,
    CTOS_REMATCH_RESPONSE, CTOS_RESPONSE, CTOS_SURRENDER, CTOS_TIME_CONFIRM, CTOS_TP_RESULT, CTOS_UPDATE_DECK,
    ERR_DECKERROR, ERR_JOINERROR, ERR_VERERROR, FrameReader, HostInfo, ProtocolConfig, STOC_CHANGE_SIDE,
    STOC_DUEL_END, STOC_DUEL_START, STOC_ERROR_MSG, STOC_GAME_MSG, STOC_HAND_RESULT, STOC_HS_PLAYER_CHANGE,
    STOC_HS_PLAYER_ENTER, STOC_JOIN_GAME, STOC_NEW_REPLAY, STOC_REMATCH, STOC_SELECT_HAND, STOC_SELECT_TP,
    STOC_TIME_LIMIT, STOC_TYPE_CHANGE, pack_frame, pack_hand_choice, pack_join_game, pack_player_info,
    pack_response, pack_time_confirm, pack_tp_choice, pack_update_deck,
};
use decklight_view::{DuelStats, StreamSource, ViewSource};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::decks::DeckSpec;
use crate::server::{App, Outgoing};

pub const DEFAULT_PORT: u16 = 7911;
pub const NOTE: &str = "Games may be shared anonymously for research";
pub const MAX_NAME: usize = 20;

/// Where a room's frames go: the current WebSocket (swapped on reconnect, `None` while no page is open).
pub type RoomOut = Arc<Mutex<Option<mpsc::UnboundedSender<Outgoing>>>>;

/// Commands from the socket loop to a room's thread.
pub enum RoomCmd {
    Ready(bool),
    Start,
    Rps(u8),
    First(bool),
    Choose { prompt: i64, index: i64 },
    Surrender,
    Rematch(bool),
    Resync,
    Leave,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Host,
    Guest,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Host => "host",
            Role::Guest => "guest",
        }
    }
}

/// What a room session needs: where to connect, as whom, with which deck.
#[derive(Clone, Debug)]
pub struct RoomSpec {
    pub role: Role,
    pub host: String,
    pub port: u16,
    pub gameid: u32,
    pub password: String,
    pub name: String,
    pub deck: DeckSpec,
    /// The room's display name ("Kai's room" when hosting; the address when joining until the host's name is
    /// known).
    pub room_name: String,
    /// Host role: every address the room can be reached on (`ip:port`).
    pub addresses: Vec<String>,
}

/// The duelist name: trimmed, at most 20 characters (the EDOPro name field), "Duelist" when empty.
pub fn clean_name(name: Option<&str>) -> String {
    let n: String = name.unwrap_or("").trim().chars().filter(|c| !c.is_control()).take(MAX_NAME).collect();
    if n.is_empty() { "Duelist".into() } else { n }
}

/// `host`, `host:port`, `[v6]:port`, each optionally followed by `#<room id>` (a Multirole room).
pub fn parse_address(text: &str) -> Result<(String, u16, u32), String> {
    let bad = || "Use an address like 192.0.2.20 or name:7911".to_string();
    let text = text.trim();
    let (addr, gameid) = match text.split_once('#') {
        Some((a, id)) => (a, id.trim().parse::<u32>().map_err(|_| bad())?),
        None => (text, 0),
    };
    let addr = addr.trim();
    if addr.is_empty() || addr.contains(char::is_whitespace) || addr.contains('/') {
        return Err(bad());
    }
    let (host, port) = if let Some(rest) = addr.strip_prefix('[') {
        let (h, tail) = rest.split_once(']').ok_or_else(bad)?;
        let port = match tail.strip_prefix(':') {
            Some(p) => p.parse::<u16>().map_err(|_| bad())?,
            None if tail.is_empty() => DEFAULT_PORT,
            None => return Err(bad()),
        };
        (h.to_string(), port)
    } else if addr.matches(':').count() == 1 {
        let (h, p) = addr.split_once(':').unwrap();
        (h.to_string(), p.parse::<u16>().map_err(|_| bad())?)
    } else if addr.contains(':') {
        (addr.to_string(), DEFAULT_PORT) // a bare IPv6 address
    } else {
        (addr.to_string(), DEFAULT_PORT)
    };
    if host.is_empty() || port == 0 {
        return Err(bad());
    }
    Ok((host, port, gameid))
}

// ------------------------------------------------------------------ addresses + scan

/// Every IPv4 address of this device except loopback (Wi-Fi, hotspot, Tailscale, ethernet), Tailscale last.
pub fn local_addresses() -> Vec<Ipv4Addr> {
    let mut out: Vec<Ipv4Addr> = Vec::new();
    unsafe {
        let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(&mut head) == 0 {
            let mut p = head;
            while !p.is_null() {
                let a = (*p).ifa_addr;
                let name = std::ffi::CStr::from_ptr((*p).ifa_name).to_string_lossy();
                // Container bridges are not a way in for a friend.
                let virt = ["docker", "br-", "veth", "virbr", "lxc"].iter().any(|v| name.starts_with(v));
                if !virt && !a.is_null() && (*a).sa_family as i32 == libc::AF_INET {
                    let sin = &*(a as *const libc::sockaddr_in);
                    let ip = Ipv4Addr::from(u32::from_be(sin.sin_addr.s_addr));
                    if !ip.is_loopback() && !ip.is_unspecified() && !out.contains(&ip) {
                        out.push(ip);
                    }
                }
                p = (*p).ifa_next;
            }
            libc::freeifaddrs(head);
        }
    }
    out.sort_by_key(|ip| (is_tailscale(*ip), *ip));
    out
}

pub fn is_tailscale(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    o[0] == 100 && (64..128).contains(&o[1])
}

/// LAN discovery (EDOPro's broadcast), as `rooms` entries.
pub fn scan(timeout: Duration) -> Result<Vec<Value>, String> {
    let found = decklight_edopro::discovery::discover_lan(timeout)?;
    let mut seen = BTreeSet::new();
    let mut rooms = Vec::new();
    for (ip, pkt) in found {
        let address = match ip {
            IpAddr::V4(v4) => format!("{v4}:{}", pkt.port),
            IpAddr::V6(v6) => format!("[{v6}]:{}", pkt.port),
        };
        if !seen.insert(address.clone()) {
            continue;
        }
        rooms.push(json!({
            "name": if pkt.name.is_empty() { address.clone() } else { pkt.name.clone() },
            "address": address,
            "seats": Value::Null,
            "rule": rules_text(&pkt.info),
            "client": Value::Null,
        }));
    }
    Ok(rooms)
}

fn rules_text(info: &HostInfo) -> String {
    let region = match info.rule {
        0 => "OCG",
        1 => "TCG",
        _ => "OCG/TCG",
    };
    format!("{region} · {} LP · Master Rule {}", info.start_lp, info.duel_rule.max(1))
}

fn rules_json(info: &HostInfo) -> Value {
    json!({
        "text": rules_text(info),
        "start_lp": info.start_lp,
        "start_hand": info.start_hand,
        "draw_count": info.draw_count,
        "duel_rule": info.duel_rule,
        "time_limit": info.time_limit,
        "lflist": info.lflist,
    })
}

// ------------------------------------------------------------------ hosting

/// The app's own `DuelHost`, serving until `close`.
pub struct HostedRoom {
    pub port: u16,
    pub name: String,
    pub password: String,
    pub addresses: Vec<String>,
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl HostedRoom {
    pub fn start(app: &Arc<App>, room_name: &str, password: &str) -> Result<HostedRoom, String> {
        let mut bound = None;
        for port in DEFAULT_PORT..DEFAULT_PORT + 10 {
            if let Ok(l) = TcpListener::bind(("0.0.0.0", port)) {
                bound = Some((l, port));
                break;
            }
        }
        let (listener, port) = bound.ok_or("could not open a port for the room (7911-7920 are all taken)")?;
        let cfg = HostConfig {
            bind: "0.0.0.0".into(),
            port,
            name: room_name.to_string(),
            once: false,
            max_duels: 0,
            remote_seats: 2,
            record_root: app.records.root.join("rooms"),
            discovery: true,
            password: password.to_string(),
            // Gates only: the host's packet log (`DECKLIGHT_ROOM_PACKET_LOG=FILE.jsonl.gz`).
            packet_log: std::env::var_os("DECKLIGHT_ROOM_PACKET_LOG").map(PathBuf::from),
            // A room waits for people: the lobby lives until closed, a duel until it ends.
            accept_timeout: Duration::from_secs(12 * 3600),
            duel_timeout: Duration::from_secs(6 * 3600),
            ..HostConfig::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let app2 = app.clone();
        let join = std::thread::Builder::new()
            .name("room-host".into())
            .spawn(move || {
                let mut host = DuelHost::new(app2.engine, &app2.cards, None, cfg, None, String::new());
                if let Err(e) = host.serve_on(listener, stop2) {
                    eprintln!("ALERT room host: {e}");
                }
            })
            .map_err(|e| format!("could not start the room: {e}"))?;
        let addresses = local_addresses().iter().map(|ip| format!("{ip}:{port}")).collect();
        Ok(HostedRoom {
            port,
            name: room_name.to_string(),
            password: password.to_string(),
            addresses,
            stop,
            join: Some(join),
        })
    }

    /// Stop accepting and join the host thread (a running lobby/duel ends when our own seat leaves it first).
    pub fn close(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for HostedRoom {
    fn drop(&mut self) {
        self.close();
    }
}

// ------------------------------------------------------------------ session

/// The socket side of a running room: its thread owns the connection and sends frames through `RoomOut`.
pub struct RoomSession {
    pub spec: RoomSpec,
    cmd_tx: Sender<RoomCmd>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl RoomSession {
    pub fn start(app: Arc<App>, spec: RoomSpec, out: RoomOut) -> Result<RoomSession, String> {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let spec2 = spec.clone();
        let join = std::thread::Builder::new()
            .name("room".into())
            .spawn(move || {
                let mut s = Session::new(&app, spec2, out, cmd_rx);
                s.run();
            })
            .map_err(|e| format!("could not start the room: {e}"))?;
        Ok(RoomSession { spec, cmd_tx, join: Some(join) })
    }

    pub fn send(&self, cmd: RoomCmd) -> Result<(), String> {
        self.cmd_tx.send(cmd).map_err(|_| "the room has closed".to_string())
    }

    pub fn finished(&self) -> bool {
        self.join.as_ref().is_none_or(|j| j.is_finished())
    }

    /// Leave (surrender when in a duel) and join the thread.
    pub fn leave(&mut self) {
        let _ = self.cmd_tx.send(RoomCmd::Leave);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for RoomSession {
    fn drop(&mut self) {
        self.leave();
    }
}

#[derive(Clone, Debug, Default)]
struct SeatInfo {
    name: String,
    ready: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Connecting,
    Waiting,
    Starting,
    Dueling,
    Ended,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Phase::Connecting => "connecting",
            Phase::Waiting => "waiting",
            Phase::Starting => "starting",
            Phase::Dueling => "dueling",
            Phase::Ended => "ended",
        }
    }
}

/// Why the session loop stops.
enum Stop {
    Left,
    Closed(String),
}

struct Session<'a> {
    app: &'a App,
    spec: RoomSpec,
    out: RoomOut,
    cmd_rx: Receiver<RoomCmd>,
    sock: Option<TcpStream>,
    reader: FrameReader,
    phase: Phase,
    seats: [Option<SeatInfo>; 2],
    my_pos: Option<u8>,
    rules: Value,
    error: Option<String>,
    ask: Option<&'static str>,
    hand_result: Option<(u8, u8)>,
    deck_sent: bool,
    ready_sent: bool,
    duel: Option<Duel<'a>>,
    result: Option<Value>,
    replay: Option<Vec<u8>>,
    duels: u64,
    connected_at: Instant,
}

/// One duel's stream view and the bookkeeping `state` frames need.
struct Duel<'a> {
    src: StreamSource<'a>,
    seat: i32,
    stats: DuelStats,
    prompt_id: i64,
    decision_seq: i64,
    last_seq: i64,
    cue_sent: usize,
    log_sent: usize,
    auto_passed: BTreeSet<usize>,
    dirty: bool,
    wait_since: Option<Instant>,
    last_tick: Instant,
}

impl<'a> Session<'a> {
    fn new(app: &'a App, spec: RoomSpec, out: RoomOut, cmd_rx: Receiver<RoomCmd>) -> Session<'a> {
        Session {
            app,
            spec,
            out,
            cmd_rx,
            sock: None,
            reader: FrameReader::new(),
            phase: Phase::Connecting,
            seats: [None, None],
            my_pos: None,
            rules: Value::Null,
            error: None,
            ask: None,
            hand_result: None,
            deck_sent: false,
            ready_sent: false,
            duel: None,
            result: None,
            replay: None,
            duels: 0,
            connected_at: Instant::now(),
        }
    }

    fn emit(&self, v: Value) {
        if let Some(tx) = self.out.lock().unwrap().as_ref() {
            let _ = tx.send(Outgoing::Frame(v));
        }
    }

    fn room_frame(&self) -> Value {
        let seats: Vec<Value> = (0..2u8)
            .map(|pos| match &self.seats[pos as usize] {
                Some(s) => {
                    let you = Some(pos) == self.my_pos;
                    json!({
                        "pos": pos,
                        "name": s.name,
                        "client": Value::Null,
                        "deck_name": if you { json!(self.spec.deck.name) } else { Value::Null },
                        "ready": s.ready,
                        "host": pos == 0,
                        "you": you,
                    })
                }
                None => Value::Null,
            })
            .collect();
        json!({
            "t": "room",
            "role": self.spec.role.as_str(),
            "state": self.phase.as_str(),
            "name": self.spec.room_name,
            "server": format!("{}:{}", self.spec.host, self.spec.port),
            "addresses": self.spec.addresses,
            "password": !self.spec.password.is_empty(),
            "rules": self.rules,
            "note": NOTE,
            "seats": seats,
            "ask": self.ask,
            "hand_result": self.hand_result.map(|(a, b)| json!([a, b])).unwrap_or(Value::Null),
            "error": self.error,
        })
    }

    fn send_room(&self) {
        self.emit(self.room_frame());
    }

    fn send(&mut self, proto: u8, payload: &[u8]) -> Result<(), String> {
        let Some(sock) = self.sock.as_mut() else { return Err("not connected".into()) };
        sock.write_all(&pack_frame(proto, payload)).map_err(|e| format!("send failed: {e}"))
    }

    fn connect(&mut self) -> Result<(), String> {
        let target = format!("{}:{}", self.spec.host, self.spec.port);
        let addrs: Vec<_> = (self.spec.host.as_str(), self.spec.port)
            .to_socket_addrs()
            .map_err(|_| format!("Could not reach {target}"))?
            .collect();
        let mut last = None;
        for a in addrs {
            match TcpStream::connect_timeout(&a, Duration::from_secs(5)) {
                Ok(s) => {
                    let _ = s.set_nodelay(true);
                    s.set_read_timeout(Some(Duration::from_millis(50))).map_err(|e| e.to_string())?;
                    self.sock = Some(s);
                    self.reader = FrameReader::new();
                    self.connected_at = Instant::now();
                    self.reset_lobby();
                    return self.join_room();
                }
                Err(e) => last = Some(e),
            }
        }
        let _ = last;
        Err(format!("Could not reach {target}"))
    }

    fn join_room(&mut self) -> Result<(), String> {
        let proto = ProtocolConfig::default();
        let v = proto.client_version;
        self.send(CTOS_PLAYER_INFO, &pack_player_info(&self.spec.name))?;
        self.send(
            CTOS_JOIN_GAME,
            &pack_join_game(
                proto.pro_version,
                self.spec.gameid,
                protocol::client_version_bytes(v.0, v.1, v.2, v.3),
                &self.spec.password,
            ),
        )
    }

    fn reset_lobby(&mut self) {
        self.seats = [None, None];
        self.my_pos = None;
        self.deck_sent = false;
        self.ready_sent = false;
        self.ask = None;
        self.hand_result = None;
        if self.spec.role == Role::Host {
            // Our host runs its lobby once both seats are connected; until a guest comes, show our own seat.
            self.my_pos = Some(0);
            self.seats[0] = Some(SeatInfo { name: self.spec.name.clone(), ready: false });
        }
    }

    fn close_sock(&mut self) {
        if let Some(s) = self.sock.take() {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
    }

    fn run(&mut self) {
        self.send_room();
        if let Err(e) = self.connect() {
            self.phase = Phase::Ended;
            self.error = Some(e);
            self.send_room();
            self.idle_after_end();
            return;
        }
        self.phase = Phase::Waiting;
        self.send_room();
        match self.pump() {
            Stop::Left => {}
            Stop::Closed(why) => {
                self.close_sock();
                self.end_with(why);
                self.idle_after_end();
            }
        }
        self.close_sock();
    }

    /// After the room is over, keep answering `resync` (the result stays on screen) until the user leaves.
    fn idle_after_end(&mut self) {
        loop {
            match self.cmd_rx.recv() {
                Ok(RoomCmd::Leave) | Err(_) => return,
                Ok(RoomCmd::Resync) => {
                    self.send_room();
                    self.send_state(true, None);
                }
                Ok(_) => self.emit(json!({"t": "error", "text": "the room has closed"})),
            }
        }
    }

    fn end_with(&mut self, why: String) {
        if self.phase == Phase::Dueling && self.result.is_none() {
            // A dropped socket cannot tell which side lost its network, so no winner is claimed (review a2b3c: both
            // peers could see Victory for the same drop). A real leave arrives as the host's MSG_WIN instead.
            let opp = self.opponent_name();
            self.result =
                Some(json!({"winner": Value::Null, "reason": "disconnect", "text": format!("The connection to {opp} was lost")}));
            self.send_state(false, None);
        }
        self.phase = Phase::Ended;
        if !why.is_empty() {
            self.error = Some(why);
        }
        self.ask = None;
        self.send_room();
    }

    fn opponent_name(&self) -> String {
        (0..2u8)
            .filter(|p| Some(*p) != self.my_pos)
            .find_map(|p| self.seats[p as usize].as_ref().map(|s| s.name.clone()))
            .unwrap_or_else(|| "Your opponent".into())
    }

    /// The connection loop: commands, socket frames, the opponent-thinking ticker.
    fn pump(&mut self) -> Stop {
        let mut buf = vec![0u8; 65536];
        loop {
            loop {
                match self.cmd_rx.try_recv() {
                    Ok(cmd) => {
                        if let Some(stop) = self.on_cmd(cmd) {
                            return stop;
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.leave_now();
                        return Stop::Left;
                    }
                }
            }
            let read = match self.sock.as_mut() {
                Some(s) => s.read(&mut buf),
                None => return Stop::Closed(String::new()),
            };
            match read {
                Ok(0) => {
                    if let Some(stop) = self.on_closed() {
                        return stop;
                    }
                    continue;
                }
                Ok(n) => {
                    let frames = match self.reader.feed(&buf[..n]) {
                        Ok(f) => f,
                        Err(e) => return Stop::Closed(format!("The room sent a broken packet ({e})")),
                    };
                    for f in frames {
                        if let Some(stop) = self.on_frame(f.proto, &f.payload) {
                            return stop;
                        }
                    }
                    if let Some(stop) = self.settle() {
                        return stop;
                    }
                }
                Err(e) => match e.kind() {
                    std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted => {}
                    _ => {
                        if let Some(stop) = self.on_closed() {
                            return stop;
                        }
                        continue;
                    }
                },
            }
            if self.spec.role == Role::Guest
                && self.my_pos.is_none()
                && self.connected_at.elapsed() > Duration::from_secs(10)
            {
                // A full room's host does not accept until its duel ends: the connection just sits there.
                return Stop::Closed("That room is full or has closed".into());
            }
            self.tick();
        }
    }

    /// The server closed the connection. Our own host drops every seat when a lobby fails (someone left, a wrong
    /// password): the hosting app then rejoins its room. Anything else ends the room.
    fn on_closed(&mut self) -> Option<Stop> {
        self.close_sock();
        if self.phase == Phase::Ended {
            return Some(Stop::Closed(String::new()));
        }
        if self.spec.role == Role::Host && matches!(self.phase, Phase::Waiting | Phase::Starting) {
            self.reset_lobby();
            self.phase = Phase::Waiting;
            std::thread::sleep(Duration::from_millis(200));
            if let Err(e) = self.connect() {
                return Some(Stop::Closed(e));
            }
            self.send_room();
            return None;
        }
        if self.phase == Phase::Dueling {
            return Some(Stop::Closed("Connection lost".into()));
        }
        let opp = self.opponent_name();
        Some(Stop::Closed(match self.spec.role {
            Role::Guest if self.my_pos.is_some() => format!("{opp} closed the room"),
            _ => format!("Could not reach {}:{}", self.spec.host, self.spec.port),
        }))
    }

    fn leave_now(&mut self) {
        if self.phase == Phase::Dueling {
            let _ = self.send(CTOS_SURRENDER, &[]);
        } else {
            let _ = self.send(CTOS_LEAVE_GAME, &[]);
        }
        self.close_sock();
    }

    fn on_cmd(&mut self, cmd: RoomCmd) -> Option<Stop> {
        let res = match cmd {
            RoomCmd::Leave => {
                self.leave_now();
                return Some(Stop::Left);
            }
            RoomCmd::Resync => {
                self.send_room();
                if self.duel.is_some() {
                    self.send_state(true, None);
                }
                Ok(())
            }
            RoomCmd::Ready(on) => {
                if self.spec.role == Role::Host {
                    Err("the host starts the duel with room_start".to_string())
                } else if !self.deck_sent || self.phase != Phase::Waiting {
                    Err("not in a room lobby".to_string())
                } else {
                    self.error = None;
                    self.send(if on { CTOS_HS_READY } else { CTOS_HS_NOTREADY }, &[])
                }
            }
            RoomCmd::Start => {
                let other_ready = (0..2u8)
                    .filter(|p| Some(*p) != self.my_pos)
                    .any(|p| self.seats[p as usize].as_ref().is_some_and(|s| s.ready));
                if self.spec.role != Role::Host {
                    Err("only the host starts the duel".to_string())
                } else if !self.deck_sent || self.phase != Phase::Waiting || !other_ready {
                    Err("wait until the other player is ready".to_string())
                } else {
                    // Our host starts the duel when every seat is ready: the host's Start is its own ready.
                    self.ready_sent = true;
                    self.error = None;
                    self.phase = Phase::Starting;
                    self.send_room();
                    self.send(CTOS_HS_READY, &[])
                }
            }
            RoomCmd::Rps(hand) => {
                if self.ask != Some("rps") {
                    Err("no rock-paper-scissors to answer".to_string())
                } else {
                    match pack_hand_choice(hand) {
                        Ok(p) => {
                            self.ask = None;
                            let r = self.send(CTOS_HAND_RESULT, &p);
                            self.send_room();
                            r
                        }
                        Err(_) => Err("hand must be 1 (scissors), 2 (rock) or 3 (paper)".to_string()),
                    }
                }
            }
            RoomCmd::First(first) => {
                if self.ask != Some("first") {
                    Err("no turn order to choose".to_string())
                } else {
                    self.ask = None;
                    let r = self.send(CTOS_TP_RESULT, &pack_tp_choice(first));
                    self.send_room();
                    r
                }
            }
            RoomCmd::Rematch(yes) => {
                if self.ask != Some("rematch") {
                    Err("no rematch was offered".to_string())
                } else {
                    self.ask = None;
                    let r = self.send(CTOS_REMATCH_RESPONSE, &[yes as u8]);
                    self.send_room();
                    r
                }
            }
            RoomCmd::Surrender => {
                if self.phase != Phase::Dueling {
                    Err("no duel to surrender".to_string())
                } else {
                    self.send(CTOS_SURRENDER, &[])
                }
            }
            RoomCmd::Choose { prompt, index } => self.choose(prompt, index),
        };
        if let Err(e) = res {
            if e.starts_with("send failed") {
                return self.on_closed();
            }
            self.emit(json!({"t": "error", "text": e}));
        }
        None
    }

    fn choose(&mut self, prompt: i64, index: i64) -> Result<(), String> {
        let Some(d) = self.duel.as_mut() else { return Err("no active duel".into()) };
        if !d.src.needs_answer() {
            return Err("it is not your turn to choose".into());
        }
        let cur = d.current_prompt_id();
        if prompt >= 0 && prompt != cur {
            return Ok(()); // a stale click (the bot duel ignores these too)
        }
        let n = d.src.n_opts() as i64;
        if index < 0 || index >= n {
            return Err(format!("index must be an option number (0..{n})"));
        }
        let resp = d.src.take(index as usize)?;
        d.decision_seq += 1;
        d.dirty = true;
        if let Some(resp) = resp {
            self.send(CTOS_RESPONSE, &pack_response(&resp))?;
        }
        if let Some(stop) = self.settle() {
            if let Stop::Closed(why) = stop {
                return Err(why);
            }
        }
        Ok(())
    }

    /// Answer what needs no person (one option, a quiet chain window), then send a state frame if anything changed.
    fn settle(&mut self) -> Option<Stop> {
        let mut sends: Vec<Vec<u8>> = Vec::new();
        if let Some(d) = self.duel.as_mut() {
            loop {
                if !d.src.needs_answer() {
                    break;
                }
                let index = if d.src.n_opts() <= 1 {
                    Some(0usize)
                } else {
                    d.auto_pass_index().map(|i| {
                        d.auto_passed.insert(d.src.history_rows().len());
                        i
                    })
                };
                let Some(index) = index else { break };
                match d.src.take(index) {
                    Ok(Some(resp)) => sends.push(resp),
                    Ok(None) => {}
                    Err(e) => return Some(Stop::Closed(format!("Could not answer the duel ({e})"))),
                }
                d.decision_seq += 1;
                d.dirty = true;
            }
        }
        for resp in sends {
            if self.send(CTOS_RESPONSE, &pack_response(&resp)).is_err() {
                return self.on_closed();
            }
        }
        if self.duel.as_ref().is_some_and(|d| d.dirty) {
            self.send_state(false, None);
        }
        None
    }

    /// Once a second while the opponent chooses: the last state with `thinking` (the bot duel's ticker frame).
    fn tick(&mut self) {
        if self.phase != Phase::Dueling || self.result.is_some() {
            return;
        }
        let Some(d) = self.duel.as_mut() else { return };
        if d.src.needs_answer() || d.src.done() {
            d.wait_since = None;
            return;
        }
        let since = *d.wait_since.get_or_insert_with(Instant::now);
        if d.last_tick.elapsed() < Duration::from_secs(1) || since.elapsed() < Duration::from_secs(1) {
            return;
        }
        d.last_tick = Instant::now();
        let secs = since.elapsed().as_secs_f64();
        let who = self.opponent_name();
        self.send_state(true, Some(json!({"seconds": (secs * 10.0).round() / 10.0, "who": who})));
    }

    fn send_state(&mut self, peek: bool, thinking: Option<Value>) {
        let opp = self.opponent_name();
        let app = self.app;
        let result = self.result.clone();
        let Some(d) = self.duel.as_mut() else { return };
        let mut frame = d.state_message(app, peek, result);
        frame["thinking"] = thinking.unwrap_or(Value::Null);
        frame["room"] = json!({"opponent": opp, "role": self.spec.role.as_str()});
        if !peek {
            d.dirty = false;
            d.wait_since = None;
        }
        self.emit(frame);
    }

    fn on_frame(&mut self, proto: u8, payload: &[u8]) -> Option<Stop> {
        let r: Result<(), String> = match proto {
            STOC_ERROR_MSG => return self.on_error(payload),
            STOC_JOIN_GAME => {
                if let Ok(info) = protocol::parse_join_game(payload) {
                    self.rules = rules_json(&info);
                }
                let d = &self.spec.deck;
                let r = self.send(CTOS_UPDATE_DECK, &pack_update_deck(&d.main, &d.extra, &d.side));
                self.deck_sent = true;
                self.send_room();
                r
            }
            STOC_TYPE_CHANGE => {
                if let Ok((pos, _is_host)) = protocol::parse_type_change(payload)
                    && pos < 2
                {
                    self.my_pos = Some(pos);
                    let name = self.spec.name.clone();
                    let s = self.seats[pos as usize].get_or_insert_with(SeatInfo::default);
                    s.name = name;
                }
                self.send_room();
                Ok(())
            }
            STOC_HS_PLAYER_ENTER => {
                if let Ok((name, pos)) = protocol::parse_hs_player_enter(payload)
                    && pos < 2
                {
                    self.seats[pos as usize] = Some(SeatInfo { name, ready: false });
                    if self.spec.role == Role::Guest && pos == 0 && Some(pos) != self.my_pos {
                        let host = self.seats[0].as_ref().map(|s| s.name.clone()).unwrap_or_default();
                        self.spec.room_name = format!("{host}'s room");
                    }
                }
                self.send_room();
                Ok(())
            }
            STOC_HS_PLAYER_CHANGE => {
                if let Ok((pos, state)) = protocol::parse_hs_player_change(payload)
                    && pos < 2
                {
                    match state.as_str() {
                        "ready" | "not ready" => {
                            if let Some(s) = self.seats[pos as usize].as_mut() {
                                s.ready = state == "ready";
                            }
                        }
                        "leave" | "observe" => self.seats[pos as usize] = None,
                        _ => {}
                    }
                }
                self.send_room();
                Ok(())
            }
            STOC_SELECT_HAND => {
                // EDOPro hosts send DUEL_START first, then rock-paper-scissors: the phase is already "dueling".
                self.ask = Some("rps");
                self.send_room();
                Ok(())
            }
            STOC_HAND_RESULT => {
                self.hand_result = protocol::parse_hand_result(payload).ok();
                self.send_room();
                Ok(())
            }
            STOC_SELECT_TP => {
                self.ask = Some("first");
                self.send_room();
                Ok(())
            }
            STOC_DUEL_START => {
                self.phase = Phase::Dueling;
                self.hand_result = None; // a rematch must not show the previous duel's throws
                self.result = None;
                self.replay = None;
                self.ask = None;
                self.error = None;
                self.send_room();
                Ok(())
            }
            STOC_TIME_LIMIT => match protocol::parse_time_limit(payload) {
                Ok((player, _)) if self.duel.as_ref().is_some_and(|d| d.seat == player as i32) => {
                    self.send(CTOS_TIME_CONFIRM, &pack_time_confirm())
                }
                _ => Ok(()),
            },
            STOC_GAME_MSG => self.on_game_msg(payload),
            STOC_NEW_REPLAY => {
                self.replay = Some(payload.to_vec());
                Ok(())
            }
            STOC_DUEL_END => {
                self.on_duel_end();
                Ok(())
            }
            STOC_REMATCH => {
                self.ask = Some("rematch");
                self.send_room();
                Ok(())
            }
            STOC_CHANGE_SIDE => {
                let d = &self.spec.deck;
                self.send(CTOS_UPDATE_DECK, &pack_update_deck(&d.main, &d.extra, &d.side))
            }
            _ => Ok(()),
        };
        match r {
            Ok(()) => None,
            Err(e) if e.starts_with("send failed") => self.on_closed(),
            Err(e) => Some(Stop::Closed(e)),
        }
    }

    fn on_error(&mut self, payload: &[u8]) -> Option<Stop> {
        let etype = payload.first().copied().unwrap_or(0);
        let text = friendly_error(payload);
        if etype == ERR_DECKERROR {
            // The host refused our ready: stay in the lobby with the reason shown.
            self.error = Some(text);
            self.send_room();
            return None;
        }
        Some(Stop::Closed(text))
    }

    fn on_game_msg(&mut self, payload: &[u8]) -> Result<(), String> {
        let Some(&t) = payload.first() else { return Ok(()) };
        let t = t as u32;
        if t == C::MSG_START {
            let seat = payload.get(1).map(|b| (b & 0xF) as u32).unwrap_or(self.my_pos.unwrap_or(0) as u32);
            let d = &self.spec.deck;
            let src = StreamSource::new(&d.main, &d.extra, &self.app.cards, seat, self.app.engine);
            let lp0 = if payload.len() >= 6 {
                u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as i64
            } else {
                8000
            };
            self.duel = Some(Duel::new(src, seat as i32, lp0));
            self.duels += 1;
            self.result = None;
        }
        let Some(d) = self.duel.as_mut() else { return Ok(()) };
        if t == C::MSG_RETRY {
            eprintln!("ALERT room: MSG_RETRY: our response disagrees with the server");
            return Err("The room rejected our move (the app and the room disagree about the duel)".into());
        }
        d.src.feed(payload)?;
        if d.src.is_select_for_me(payload) {
            d.decision_seq += 1;
        }
        d.dirty = true;
        Ok(())
    }

    fn on_duel_end(&mut self) {
        if let Some(d) = self.duel.as_ref()
            && self.result.is_none()
        {
            // MSG_WIN's winner byte: 0/1 a seat, 2 a draw (review a2b3c: a draw was shown as Defeat on both sides).
            let you = d.src.winner().filter(|w| (0..=1).contains(w)).map(|w| w == d.seat);
            // MSG_WIN's reason byte: 0 is a surrender (EDOPro `WIN_REASON_SURRENDER`).
            let surrendered = d
                .src
                .trace_msgs()
                .iter()
                .rev()
                .find(|m| m.first().map(|&b| b as u32) == Some(C::MSG_WIN))
                .is_some_and(|m| m.get(2) == Some(&0));
            let reason = if surrendered { "surrender" } else { "win" };
            self.result = Some(match you {
                Some(true) => json!({"winner": "you", "reason": reason}),
                Some(false) => json!({"winner": "opponent", "reason": reason}),
                None => json!({"winner": Value::Null, "reason": "draw"}),
            });
        }
        if let Some(data) = self.replay.take() {
            let dir: PathBuf = self.app.records.root.join("rooms");
            let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
            if std::fs::create_dir_all(&dir).is_ok() {
                let path = dir.join(format!("{stamp}-{}.yrpx", self.spec.role.as_str()));
                if let Err(e) = std::fs::write(&path, &data) {
                    eprintln!("ALERT room: cannot save the replay: {e}");
                }
            }
        }
        if self.duel.is_some() {
            self.send_state(false, None);
        }
        self.phase = Phase::Ended;
        self.send_room();
    }
}

impl<'a> Duel<'a> {
    fn new(src: StreamSource<'a>, seat: i32, lp0: i64) -> Duel<'a> {
        Duel {
            src,
            seat,
            stats: DuelStats::new(seat, lp0),
            prompt_id: 0,
            decision_seq: 0,
            last_seq: -1,
            cue_sent: 0,
            log_sent: 0,
            auto_passed: BTreeSet::new(),
            dirty: true,
            wait_since: None,
            last_tick: Instant::now(),
        }
    }

    fn current_prompt_id(&mut self) -> i64 {
        if self.decision_seq != self.last_seq {
            self.last_seq = self.decision_seq;
            self.prompt_id += 1;
        }
        self.prompt_id
    }

    /// The bot duel's chain "auto" rule (`Game::auto_pass_index`), read from the stream.
    fn auto_pass_index(&self) -> Option<usize> {
        let raw = self.src.pending_msg();
        let msg = M::parse_one(&raw).ok()?;
        let (forced, spe_count, hint_timing, hint_timing_other) = match msg.body {
            MsgBody::SelectChain { forced, spe_count, hint_timing, hint_timing_other, .. } => {
                (forced, spe_count, hint_timing, hint_timing_other)
            }
            _ => return None,
        };
        let chain_empty = self.src.field_state().1.is_empty();
        if !crate::server::quiet_chain_window(forced, spe_count, hint_timing, hint_timing_other, chain_empty) {
            return None;
        }
        let history = self.src.history_rows();
        let start = history.iter().rposition(|e| e[10] == self.seat).map(|i| i + 1).unwrap_or(0);
        if history[start..].iter().any(|e| !matches!(e[0], 8 | 9 | 10 | 11 | 13)) {
            return None;
        }
        let dec = self.src.decision()?;
        dec.options.iter().position(|o| o.kind == 11)
    }

    fn state_message(&mut self, app: &App, peek: bool, result: Option<Value>) -> Value {
        let view = app.ctx.build_view(&self.src, self.seat, false);
        let mut prompt = Value::Null;
        if self.src.needs_answer() && result.is_none() {
            let pid = self.current_prompt_id();
            if let Some(p) = app.ctx.build_prompt(&self.src, self.seat, pid) {
                prompt = p;
            }
        }
        let mut log = app.ctx.build_events(&self.src, self.seat, if peek { 0 } else { self.log_sent });
        for e in log.iter_mut() {
            let n = e.get("n").and_then(Value::as_u64).unwrap_or(0) as usize;
            if self.auto_passed.contains(&n) {
                e["actor"] = json!("system");
                e["text"] = json!("(auto-passed)");
                e["card"] = Value::Null;
                e["kind"] = json!("auto_pass");
            }
        }
        let cues: Vec<Value> = if peek {
            Vec::new()
        } else {
            let (c, next) = decklight_view::build_cues(&self.src, &app.ctx, self.seat, self.cue_sent, false);
            self.cue_sent = next;
            c
        };
        self.stats.feed_cues(&cues);
        self.stats.note_view(Some(&view));
        if !peek {
            self.log_sent = self.src.history_rows().len();
        }
        json!({
            "t": "state",
            "view": view,
            "prompt": prompt,
            "log": log,
            "cues": cues,
            "stats": self.stats.stats_message(false),
            "winchance": Value::Null,
            "thinking": Value::Null,
            "result": result,
            "practice": false,
            "can_undo": false,
            "what_if": Value::Null,
            "puzzle": Value::Null,
            "model_label": "",
        })
    }
}

/// `STOC_ERROR_MSG` as the plain sentence the room screen shows (spec P3).
pub fn friendly_error(payload: &[u8]) -> String {
    let etype = payload.first().copied().unwrap_or(0);
    let u32_at = |i: usize| -> u32 {
        payload.get(i..i + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(0)
    };
    match etype {
        ERR_JOINERROR => match u32_at(4) {
            1 => "Wrong password".into(),
            2 => "The room refused to let you join".into(),
            _ => "That room is full or has closed".into(),
        },
        ERR_DECKERROR => {
            let text = protocol::parse_error_msg(payload);
            format!("Your deck was refused: {}", text.trim_start_matches("deck error: "))
        }
        ERR_VERERROR | protocol::ERR_VERERROR2 => {
            let room = payload.get(4..8).map(|v| format!("{}.{}", v[0], v[1])).unwrap_or_else(|| "?".into());
            let v = ProtocolConfig::default().client_version;
            format!(
                "Your app and the room use different EDOPro versions (room {room}, app {}.{}). Update the card pack.",
                v.0, v.1
            )
        }
        _ => protocol::parse_error_msg(payload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_parse() {
        assert_eq!(parse_address("192.0.2.20").unwrap(), ("192.0.2.20".into(), 7911, 0));
        assert_eq!(parse_address(" kai-phone:7912 ").unwrap(), ("kai-phone".into(), 7912, 0));
        assert_eq!(parse_address("[fe80::1]:7911").unwrap(), ("fe80::1".into(), 7911, 0));
        assert_eq!(parse_address("fe80::1").unwrap(), ("fe80::1".into(), 7911, 0));
        assert_eq!(parse_address("mr.example:7922#4242").unwrap(), ("mr.example".into(), 7922, 4242));
        for bad in ["", "a b", "host:", "host:99999", "host:x", "http://x", "h#x", "[::1"] {
            assert!(parse_address(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn names_are_cleaned() {
        assert_eq!(clean_name(None), "Duelist");
        assert_eq!(clean_name(Some("   ")), "Duelist");
        assert_eq!(clean_name(Some(" Kai\n")), "Kai");
        assert_eq!(clean_name(Some(&"x".repeat(40))).chars().count(), MAX_NAME);
    }

    #[test]
    fn join_errors_are_sentences() {
        assert_eq!(friendly_error(&protocol::pack_join_error(1)), "Wrong password");
        assert_eq!(friendly_error(&protocol::pack_join_error(0)), "That room is full or has closed");
        let deck = protocol::pack_deck_error(protocol::DERR_UNKNOWNCARD, 0, 0, 0, 1234);
        assert!(friendly_error(&deck).starts_with("Your deck was refused: "), "{}", friendly_error(&deck));
        let ver = [ERR_VERERROR, 0, 0, 0, 41, 1, 11, 0];
        assert!(friendly_error(&ver).contains("room 41.1"), "{}", friendly_error(&ver));
    }

    #[test]
    fn tailscale_range() {
        assert!(is_tailscale(Ipv4Addr::new(100, 101, 2, 3)));
        assert!(!is_tailscale(Ipv4Addr::new(100, 1, 2, 3)));
        assert!(!is_tailscale(Ipv4Addr::new(192, 168, 1, 2)));
    }
}
