//! EDOPro network client: join (or create) a room and play one seat from the received stream.
//!
//! A port of `src/decklight/edopro/client.py` (`ClientSession`, `DecisionOnlyView`). There is no local duel: every
//! select message the server sends becomes an A2b-1 `Decision` built from those raw bytes, the agent picks an
//! option, and the completed core response goes back as `CTOS_RESPONSE`. The client keeps the `.yrpX` the server
//! sends at the end (`STOC_NEW_REPLAY`) and imports it with A2b-1 `to_record` (`source: "yrpx"`).
//!
//! Exit codes mirror `scripts/edopro_join.py`: 0 duel finished, 2 bad args/server-side error, 3 view/server
//! disagreement (`MSG_RETRY`), 4 timeout or disconnect before the duel ended.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use decklight_cards::CardData;
use decklight_core::constants as C;
use decklight_core::messages as M;
use serde_json::{Value, json};
use ygoenv_sys::{Decision, Engine};

use crate::agent::Agent;
use crate::log::PacketLog;
use crate::protocol::{
    self, CTOS_CREATE_GAME, CTOS_HAND_RESULT, CTOS_HS_READY, CTOS_HS_START, CTOS_JOIN_GAME,
    CTOS_PLAYER_INFO, CTOS_REMATCH_RESPONSE, CTOS_RESPONSE, CTOS_TIME_CONFIRM, CTOS_TP_RESULT,
    CTOS_UPDATE_DECK, Frame, FrameReader, HostInfo, ProtocolConfig, STOC_CHANGE_SIDE, STOC_CHAT,
    STOC_CREATE_GAME, STOC_DUEL_END, STOC_DUEL_START, STOC_ERROR_MSG, STOC_GAME_MSG, STOC_HAND_RESULT,
    STOC_HS_PLAYER_CHANGE, STOC_HS_PLAYER_ENTER, STOC_HS_WATCH_CHANGE, STOC_JOIN_GAME, STOC_NEW_REPLAY,
    STOC_REMATCH, STOC_REPLAY, STOC_SELECT_HAND, STOC_SELECT_TP, STOC_TIME_LIMIT, STOC_TP_RESULT,
    STOC_TYPE_CHANGE, STOC_WAITING_REMATCH, pack_create_game, pack_frame, pack_hand_choice,
    pack_join_game, pack_player_info, pack_response, pack_time_confirm, pack_tp_choice,
    pack_update_deck, parse_create_game, parse_error_msg, parse_hand_result, parse_hs_player_change,
    parse_hs_player_enter, parse_join_game, parse_time_limit, parse_type_change,
};
use crate::record;
use crate::yrp;

pub const OK: i32 = 0;
pub const BAD_ARGS: i32 = 2;
pub const RETRY: i32 = 3;
pub const TIMEOUT: i32 = 4;

pub const DEFAULT_RECORD_DIR: &str = "data/cache/edopro_client";

#[derive(Clone, Debug, Default)]
pub struct DuelRecord {
    pub server: String,
    pub seat: Option<u8>,
    pub rps: Option<u8>,
    pub going_first: Option<bool>,
    pub n_responses: usize,
    pub retries: usize,
    pub winner: Option<(u8, u8)>,
    pub duel_end: bool,
    pub packets: BTreeMap<String, u64>,
    /// `"chain:<code>"` / `"summon:<code>"` counts, both players.
    pub events: BTreeMap<String, u64>,
    pub alerts: Vec<String>,
    pub exit_code: i32,
    pub server_version: Option<[u8; 4]>,
    pub room_id: Option<u32>,
    pub hand_result: Option<(u8, u8)>,
}

impl DuelRecord {
    fn count(&mut self, key: String) {
        *self.packets.entry(key).or_insert(0) += 1;
    }

    pub fn to_json(&self) -> Value {
        let winner = self.winner.map(|(a, b)| json!([a, b])).unwrap_or(Value::Null);
        let hand = self.hand_result.map(|(a, b)| json!([a, b])).unwrap_or(Value::Null);
        let mut v = json!({
            "server": self.server,
            "seat": self.seat,
            "rps": self.rps,
            "going_first": self.going_first,
            "n_responses": self.n_responses,
            "retries": self.retries,
            "winner": winner,
            "duel_end": self.duel_end,
            "packets": self.packets,
            "events": self.events,
            "alerts": self.alerts,
            "exit_code": self.exit_code,
        });
        if let Some(ver) = self.server_version {
            v["server_version"] = json!(ver.to_vec());
        }
        if let Some(id) = self.room_id {
            v["room_id"] = json!(id);
        }
        if self.hand_result.is_some() {
            v["hand_result"] = hand;
        }
        v
    }
}

/// `DecisionOnlyView`: answers selects from the raw stream; no board.
struct View<'e> {
    #[allow(dead_code)]
    seat: u8,
    engine: &'e Engine,
    cards: &'e BTreeMap<u32, CardData>,
    pool: BTreeSet<u32>,
    dec: Option<Decision>,
    auto: VecDeque<Vec<u8>>,
    winner: Option<(u8, u8)>,
}

impl<'e> View<'e> {
    fn new(seat: u8, deck: &[u32], engine: &'e Engine, cards: &'e BTreeMap<u32, CardData>) -> View<'e> {
        View {
            seat,
            engine,
            cards,
            pool: deck.iter().copied().collect(),
            dec: None,
            auto: VecDeque::new(),
            winner: None,
        }
    }

    fn feed(&mut self, payload: &[u8]) -> Result<(), String> {
        if payload.is_empty() {
            return Ok(());
        }
        let t = payload[0] as u32;
        if t == C::MSG_WIN {
            self.winner = if payload.len() >= 3 { Some((payload[1], payload[2])) } else { None };
            return Ok(());
        }
        if t == C::MSG_RETRY {
            return Ok(());
        }
        // The client pool: our deck plus every card code seen on the wire (`DecisionOnlyView`).
        let end = payload.len().saturating_sub(3);
        for i in 0..end {
            let code = u32::from_le_bytes([payload[i], payload[i + 1], payload[i + 2], payload[i + 3]]);
            if self.cards.contains_key(&code) {
                self.pool.insert(code);
            }
        }
        if !M::is_select_type(t) {
            return Ok(());
        }
        let pool: Vec<u32> = self.pool.iter().copied().collect();
        let mut dec = Decision::new(self.engine, payload, &pool)?;
        if dec.options().len() <= 1 {
            // Singleton: answered internally without involving the agent or the record.
            for _ in 0..128 {
                match dec.choose(0) {
                    Ok(Some(resp)) => {
                        self.auto.push_back(resp);
                        return Ok(());
                    }
                    Ok(None) => {}
                    Err(e) => return Err(e),
                }
            }
            return Err(format!("singleton decision never completes: message {t:#x}"));
        }
        if self.dec.is_some() {
            return Err("new select arrived while one is still pending".into());
        }
        self.dec = Some(dec);
        Ok(())
    }

    fn needs_answer(&self) -> bool {
        self.dec.is_some()
    }

    fn n_opts(&self) -> usize {
        self.dec.as_ref().map(|d| d.options().len()).unwrap_or(0)
    }

    fn take(&mut self, index: usize) -> Result<Option<Vec<u8>>, String> {
        let Some(dec) = self.dec.as_mut() else { return Ok(None) };
        let resp = dec.choose(index as i32)?;
        if resp.is_some() {
            self.dec = None;
        }
        Ok(resp)
    }

    fn take_auto(&mut self) -> Option<Vec<u8>> {
        self.auto.pop_front()
    }
}

#[derive(Clone, Debug)]
pub struct ClientConfig {
    pub host: String,
    pub port: u16,
    pub name: String,
    pub password: String,
    pub gameid: u32,
    /// Forced RPS throw (WindBot `Hand=`); 0 = the agent decides.
    pub hand: u8,
    pub timeout: Duration,
    pub record_dir: PathBuf,
    pub create: Option<HostInfo>,
    pub room_notes: String,
    pub packet_log: Option<PathBuf>,
    pub proto: ProtocolConfig,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            host: "127.0.0.1".into(),
            port: 7911,
            name: "decklight".into(),
            password: String::new(),
            gameid: 0,
            hand: 0,
            timeout: Duration::from_secs(30),
            record_dir: PathBuf::from(DEFAULT_RECORD_DIR),
            create: None,
            room_notes: String::new(),
            packet_log: None,
            proto: ProtocolConfig::default(),
        }
    }
}

pub struct ClientSession<'e> {
    engine: &'e Engine,
    cards: &'e BTreeMap<u32, CardData>,
    cfg: ClientConfig,
    main: Vec<u32>,
    extra: Vec<u32>,
    side: Vec<u32>,
    agent: Box<dyn Agent>,
    sock: Option<TcpStream>,
    reader: FrameReader,
    log: PacketLog,
    pub record: DuelRecord,
    ties: usize,
    lobby_seat: Option<u8>,
    replay: Option<Vec<u8>>,
    view: Option<View<'e>>,
}

impl<'e> ClientSession<'e> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &'e Engine,
        cards: &'e BTreeMap<u32, CardData>,
        cfg: ClientConfig,
        main: Vec<u32>,
        extra: Vec<u32>,
        side: Vec<u32>,
        agent: Box<dyn Agent>,
    ) -> ClientSession<'e> {
        let log = match &cfg.packet_log {
            Some(p) => match PacketLog::open(p) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("ALERT edopro_client: cannot open packet log: {e}");
                    PacketLog::disabled()
                }
            },
            None => PacketLog::disabled(),
        };
        ClientSession {
            engine,
            cards,
            cfg: cfg.clone(),
            main,
            extra,
            side,
            agent,
            sock: None,
            reader: FrameReader::new(),
            log,
            record: DuelRecord { server: format!("{}:{}", cfg.host, cfg.port), exit_code: TIMEOUT, ..Default::default() },
            ties: 0,
            lobby_seat: None,
            replay: None,
            view: None,
        }
    }

    fn alert(&mut self, text: &str) {
        let line = format!("ALERT edopro_client: {text}");
        eprintln!("{line}");
        self.record.alerts.push(line);
    }

    fn connect(&mut self) -> Result<(), String> {
        let deadline = Instant::now() + self.cfg.timeout.min(Duration::from_secs(10));
        loop {
            match TcpStream::connect((self.cfg.host.as_str(), self.cfg.port)) {
                Ok(s) => {
                    let _ = s.set_nodelay(true);
                    s.set_read_timeout(Some(self.cfg.timeout)).map_err(|e| e.to_string())?;
                    self.sock = Some(s);
                    return Ok(());
                }
                Err(e) => {
                    if Instant::now() >= deadline {
                        return Err(e.to_string());
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    fn send(&mut self, proto: u8, payload: &[u8]) -> Result<(), String> {
        let Some(sock) = self.sock.as_mut() else { return Err("not connected".into()) };
        sock.write_all(&pack_frame(proto, payload)).map_err(|e| e.to_string())?;
        self.log.log("s", proto, payload);
        self.record.count(format!("C{proto:#x}"));
        Ok(())
    }

    /// Next frames: `Ok(None)` on timeout, `Ok(Some(vec![]))` on a clean close.
    fn wait(&mut self) -> Result<Option<Vec<Frame>>, String> {
        let Some(sock) = self.sock.as_mut() else { return Ok(Some(Vec::new())) };
        let mut buf = [0u8; 65536];
        match sock.read(&mut buf) {
            Ok(0) => Ok(Some(Vec::new())),
            Ok(n) => {
                let frames = self.reader.feed(&buf[..n])?;
                for f in &frames {
                    self.log.log("r", f.proto, &f.payload);
                }
                Ok(Some(frames))
            }
            Err(e) => match e.kind() {
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => Ok(None),
                _ => Ok(Some(Vec::new())),
            },
        }
    }

    fn close(&mut self) {
        if let Some(s) = self.sock.take() {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
    }

    pub fn run(&mut self) -> i32 {
        if let Err(e) = self.connect() {
            let text = format!("connect to {}:{} failed: {e}", self.cfg.host, self.cfg.port);
            self.alert(&text);
            return self.finish(TIMEOUT);
        }
        if let Err(e) = self.send(CTOS_PLAYER_INFO, &pack_player_info(&self.cfg.name)) {
            self.alert(&format!("send player info failed: {e}"));
            return self.finish(TIMEOUT);
        }
        let join = match &self.cfg.create {
            Some(info) => {
                self.send(CTOS_CREATE_GAME, &pack_create_game(info, &self.cfg.name, &self.cfg.password, &self.cfg.room_notes))
            }
            None => self.send(
                CTOS_JOIN_GAME,
                &pack_join_game(
                    self.cfg.proto.pro_version,
                    self.cfg.gameid,
                    protocol::client_version_bytes(
                        self.cfg.proto.client_version.0,
                        self.cfg.proto.client_version.1,
                        self.cfg.proto.client_version.2,
                        self.cfg.proto.client_version.3,
                    ),
                    &self.cfg.password,
                ),
            ),
        };
        if let Err(e) = join {
            self.alert(&format!("send join failed: {e}"));
            return self.finish(TIMEOUT);
        }
        let code = match self.lobby() {
            Ok(pending) => self.duel_result(pending),
            Err(code) => code,
        };
        self.finish(code)
    }

    fn finish(&mut self, code: i32) -> i32 {
        self.record.exit_code = code;
        self.close();
        let stamp = now_stamp();
        let safe: String = self
            .cfg
            .name
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .take(32)
            .collect();
        let base = format!("{stamp}-{safe}-{}", self.cfg.port);
        if let Some(data) = self.replay.take() {
            let _ = std::fs::create_dir_all(&self.cfg.record_dir);
            let rp = self.cfg.record_dir.join(format!("{base}.yrpx"));
            if let Err(e) = std::fs::write(&rp, &data) {
                self.alert(&format!("cannot write replay: {e}"));
            } else {
                match yrp::parse_yrpx(&data) {
                    Ok(x) => {
                        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                        let rid = record::uuid7((now_secs()) * 1000);
                        let (rec, problems) = yrp::to_record(
                            &x,
                            self.engine,
                            &root,
                            None,
                            "decklight-edopro-client",
                            None,
                            Some(rid),
                        );
                        match rec {
                            Some(rec) => {
                                let mp = self.cfg.record_dir.join(format!("{base}.match.json.gz"));
                                if let Err(e) = record::save_gz(&rec, &mp) {
                                    self.alert(&format!("cannot write imported record: {e}"));
                                }
                            }
                            None => self.alert(&format!("cannot import the replay: {}", problems.join("; "))),
                        }
                    }
                    Err(e) => self.alert(&format!("cannot parse the replay: {e}")),
                }
            }
        }
        if let Err(e) = std::fs::create_dir_all(&self.cfg.record_dir) {
            eprintln!("ALERT edopro_client: cannot write record: {e}");
            return code;
        }
        let path = self.cfg.record_dir.join(format!("{base}.json"));
        let text = serde_json::to_string_pretty(&self.record.to_json()).unwrap_or_else(|_| "{}".into());
        if let Err(e) = std::fs::write(&path, format!("{text}\n")) {
            eprintln!("ALERT edopro_client: cannot write record: {e}");
        }
        let _ = self.log.finish();
        code
    }

    // --- lobby ---

    fn maybe_start(&mut self, is_host: bool, ready_sent: bool, others_ready: &BTreeSet<u8>, start_sent: &mut bool) -> Result<(), String> {
        if self.cfg.create.is_some() && is_host && ready_sent && !others_ready.is_empty() && !*start_sent {
            self.send(CTOS_HS_START, &[])?;
            *start_sent = true;
        }
        Ok(())
    }

    /// Lobby until `STOC_DUEL_START`: `Ok(frames after DUEL_START)` = go to the duel, `Err(code)` = exit.
    fn lobby(&mut self) -> Result<Vec<Frame>, i32> {
        let mut deck_sent = false;
        let mut pending_ready = false;
        let mut seat: Option<u8> = None;
        let mut is_host = false;
        let mut ready_sent = false;
        let mut others_ready: BTreeSet<u8> = BTreeSet::new();
        let mut start_sent = false;
        loop {
            let frames = match self.wait() {
                Ok(Some(frames)) => frames,
                Ok(None) => {
                    self.alert("lobby timed out waiting for the server");
                    return Err(TIMEOUT);
                }
                Err(e) => {
                    self.alert(&format!("lobby read failed: {e}"));
                    return Err(TIMEOUT);
                }
            };
            if frames.is_empty() {
                self.alert("server closed the connection in the lobby");
                return Err(TIMEOUT);
            }
            for (idx, f) in frames.iter().enumerate() {
                self.record.count(format!("{:#x}", f.proto));
                match f.proto {
                    STOC_ERROR_MSG => {
                        let text = parse_error_msg(&f.payload);
                        self.alert(&format!("server rejected us: {text}"));
                        return Err(BAD_ARGS);
                    }
                    STOC_JOIN_GAME => {
                        let info = match parse_join_game(&f.payload) {
                            Ok(i) => i,
                            Err(e) => {
                                self.alert(&format!("bad STOC_JOIN_GAME: {e}"));
                                return Err(BAD_ARGS);
                            }
                        };
                        self.record.server_version = Some(info.version);
                        if let Err(e) = self.send(
                            CTOS_UPDATE_DECK,
                            &pack_update_deck(&self.main, &self.extra, &self.side),
                        ) {
                            self.alert(&e);
                            return Err(TIMEOUT);
                        }
                        deck_sent = true;
                        if pending_ready {
                            if let Err(e) = self.send(CTOS_HS_READY, &[]) {
                                self.alert(&e);
                                return Err(TIMEOUT);
                            }
                            pending_ready = false;
                            ready_sent = true;
                            if let Err(e) = self.maybe_start(is_host, ready_sent, &others_ready, &mut start_sent) {
                                self.alert(&e);
                                return Err(TIMEOUT);
                            }
                        }
                    }
                    STOC_TYPE_CHANGE => {
                        match parse_type_change(&f.payload) {
                            Ok((s, h)) => {
                                seat = Some(s);
                                is_host = h;
                                self.record.seat = Some(s);
                            }
                            Err(e) => {
                                self.alert(&format!("bad STOC_TYPE_CHANGE: {e}"));
                                return Err(BAD_ARGS);
                            }
                        }
                        if deck_sent {
                            if let Err(e) = self.send(CTOS_HS_READY, &[]) {
                                self.alert(&e);
                                return Err(TIMEOUT);
                            }
                            ready_sent = true;
                            if let Err(e) = self.maybe_start(is_host, ready_sent, &others_ready, &mut start_sent) {
                                self.alert(&e);
                                return Err(TIMEOUT);
                            }
                        } else {
                            pending_ready = true;
                        }
                    }
                    STOC_CREATE_GAME => match parse_create_game(&f.payload) {
                        Ok(id) => {
                            self.record.room_id = Some(id);
                        }
                        Err(e) => {
                            self.alert(&format!("bad STOC_CREATE_GAME: {e}"));
                            return Err(BAD_ARGS);
                        }
                    },
                    STOC_HS_PLAYER_ENTER => {
                        let _ = parse_hs_player_enter(&f.payload);
                    }
                    STOC_HS_PLAYER_CHANGE => {
                        let Ok((pos, state)) = parse_hs_player_change(&f.payload) else { continue };
                        if Some(pos) != seat {
                            if state == "ready" {
                                others_ready.insert(pos);
                            } else {
                                others_ready.remove(&pos);
                            }
                            if let Err(e) = self.maybe_start(is_host, ready_sent, &others_ready, &mut start_sent) {
                                self.alert(&e);
                                return Err(TIMEOUT);
                            }
                        }
                    }
                    STOC_SELECT_HAND => {
                        let mut rps = if self.cfg.hand != 0 { self.cfg.hand } else { self.agent.rps() };
                        if !(1..=3).contains(&rps) {
                            self.alert(&format!("illegal RPS throw {rps}, sending 1"));
                            rps = 1;
                        }
                        self.record.rps = Some(rps);
                        if let Err(e) = self.send(CTOS_HAND_RESULT, &[rps]) {
                            self.alert(&e);
                            return Err(TIMEOUT);
                        }
                    }
                    STOC_HAND_RESULT => {
                        if self.note_hand_result(&f.payload) {
                            self.alert("50 consecutive RPS ties (both sides use the same seed?)");
                            return Err(RETRY);
                        }
                    }
                    STOC_SELECT_TP => {
                        let first = self.agent.go_first();
                        self.record.going_first = Some(first);
                        if let Err(e) = self.send(CTOS_TP_RESULT, &pack_tp_choice(first)) {
                            self.alert(&e);
                            return Err(TIMEOUT);
                        }
                    }
                    STOC_TP_RESULT => {}
                    STOC_DUEL_START => {
                        self.record.seat = seat;
                        self.lobby_seat = seat;
                        // Hand the frames after DUEL_START to the duel loop.
                        return Ok(frames[idx + 1..].to_vec());
                    }
                    STOC_DUEL_END => {
                        self.record.duel_end = true;
                        return Err(OK);
                    }
                    STOC_REPLAY | STOC_CHAT | STOC_HS_WATCH_CHANGE => {}
                    _ => {}
                }
            }
        }
    }

    fn note_hand_result(&mut self, payload: &[u8]) -> bool {
        let Ok((ours, theirs)) = parse_hand_result(payload) else { return false };
        self.record.hand_result = Some((ours, theirs));
        if ours == theirs {
            self.ties += 1;
            return self.ties >= 50;
        }
        self.ties = 0;
        false
    }

    // --- duel ---

    fn duel_result(&mut self, pending: Vec<Frame>) -> i32 {
        let mut queue: VecDeque<Frame> = pending.into();
        loop {
            let frames: Vec<Frame> = if let Some(f) = queue.pop_front() {
                vec![f]
            } else {
                match self.wait() {
                    Ok(None) => {
                        self.alert("timed out waiting for the server mid-duel");
                        return TIMEOUT;
                    }
                    Ok(Some(frames)) => {
                        if frames.is_empty() {
                            self.alert("server closed the connection mid-duel");
                            return if self.record.duel_end { OK } else { TIMEOUT };
                        }
                        frames
                    }
                    Err(e) => {
                        self.alert(&format!("duel read failed: {e}"));
                        return TIMEOUT;
                    }
                }
            };
            for f in frames {
                self.record.count(format!("{:#x}", f.proto));
                match f.proto {
                    STOC_ERROR_MSG => {
                        let text = parse_error_msg(&f.payload);
                        self.alert(&format!("server error mid-duel: {text}"));
                        return BAD_ARGS;
                    }
                    STOC_NEW_REPLAY => {
                        self.replay = Some(f.payload);
                    }
                    STOC_GAME_MSG => {
                        if f.payload.is_empty() {
                            continue;
                        }
                        let t = f.payload[0] as u32;
                        if t == C::MSG_START {
                            let seat = if f.payload.len() >= 2 { f.payload[1] & 0xF } else { self.record.seat.unwrap_or(0) };
                            self.record.seat = Some(seat);
                            let full: Vec<u32> = self.main.iter().chain(&self.extra).chain(&self.side).copied().collect();
                            self.view = Some(View::new(seat, &full, self.engine, self.cards));
                            continue;
                        }
                        if t == C::MSG_RETRY {
                            self.record.retries += 1;
                            self.alert("MSG_RETRY: our response disagrees with the server");
                            return RETRY;
                        }
                        if self.view.is_none() {
                            continue;
                        }
                        self.note_event(t, &f.payload);
                        if let Err(e) = self.view.as_mut().unwrap().feed(&f.payload) {
                            self.alert(&format!("cannot build a decision: {e}"));
                            return TIMEOUT;
                        }
                        loop {
                            let auto = self.view.as_mut().unwrap().take_auto();
                            if let Some(auto) = auto {
                                if let Err(e) = self.send(CTOS_RESPONSE, &pack_response(&auto)) {
                                    self.alert(&e);
                                    return TIMEOUT;
                                }
                                self.record.n_responses += 1;
                                continue;
                            }
                            if !self.view.as_ref().unwrap().needs_answer() {
                                break;
                            }
                            let n = self.view.as_ref().unwrap().n_opts();
                            let index = self.agent.act(n);
                            let resp = match self.view.as_mut().unwrap().take(index) {
                                Ok(r) => r,
                                Err(e) => {
                                    self.alert(&format!("decision choose failed: {e}"));
                                    return TIMEOUT;
                                }
                            };
                            if let Some(resp) = resp {
                                if let Err(e) = self.send(CTOS_RESPONSE, &pack_response(&resp)) {
                                    self.alert(&e);
                                    return TIMEOUT;
                                }
                                self.record.n_responses += 1;
                            }
                        }
                        if let Some(w) = self.view.as_ref().unwrap().winner {
                            self.record.winner = Some(w);
                        }
                    }
                    STOC_REMATCH => {
                        if let Err(e) = self.send(CTOS_REMATCH_RESPONSE, &[0]) {
                            self.alert(&e);
                            return TIMEOUT;
                        }
                    }
                    STOC_SELECT_HAND => {
                        let mut rps = if self.cfg.hand != 0 { self.cfg.hand } else { self.agent.rps() };
                        if !(1..=3).contains(&rps) {
                            rps = 1;
                        }
                        self.record.rps = Some(rps);
                        if let Err(e) = self.send(CTOS_HAND_RESULT, &pack_hand_choice(rps).unwrap_or_default()) {
                            self.alert(&e);
                            return TIMEOUT;
                        }
                    }
                    STOC_SELECT_TP => {
                        let first = self.agent.go_first();
                        self.record.going_first = Some(first);
                        if let Err(e) = self.send(CTOS_TP_RESULT, &pack_tp_choice(first)) {
                            self.alert(&e);
                            return TIMEOUT;
                        }
                    }
                    STOC_HAND_RESULT => {
                        if self.note_hand_result(&f.payload) {
                            self.alert("50 consecutive RPS ties (both sides use the same seed?)");
                            return RETRY;
                        }
                    }
                    STOC_TIME_LIMIT => match parse_time_limit(&f.payload) {
                        Ok((player, _secs)) => {
                            if self.record.seat == Some(player) {
                                if let Err(e) = self.send(CTOS_TIME_CONFIRM, &pack_time_confirm()) {
                                    self.alert(&e);
                                    return TIMEOUT;
                                }
                            }
                        }
                        Err(_) => continue,
                    },
                    STOC_CHANGE_SIDE => {
                        if let Err(e) =
                            self.send(CTOS_UPDATE_DECK, &pack_update_deck(&self.main, &self.extra, &self.side))
                        {
                            self.alert(&e);
                            return TIMEOUT;
                        }
                    }
                    STOC_DUEL_END => {
                        self.record.duel_end = true;
                        if let Some(w) = self.view.as_ref().and_then(|v| v.winner) {
                            self.record.winner = Some(w);
                        }
                        return OK;
                    }
                    STOC_TP_RESULT
                    | STOC_REPLAY
                    | STOC_CHAT
                    | STOC_HS_PLAYER_ENTER
                    | STOC_HS_PLAYER_CHANGE
                    | STOC_HS_WATCH_CHANGE
                    | STOC_CREATE_GAME
                    | STOC_JOIN_GAME
                    | STOC_TYPE_CHANGE
                    | STOC_DUEL_START
                    | STOC_WAITING_REMATCH => {}
                    _ => {}
                }
            }
        }
    }

    fn note_event(&mut self, t: u32, payload: &[u8]) {
        let kind = match t {
            C::MSG_CHAINING => "chain",
            C::MSG_SUMMONING | C::MSG_SPSUMMONING | C::MSG_FLIPSUMMONING => "summon",
            _ => return,
        };
        if payload.len() < 5 {
            return;
        }
        let code = u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]);
        *self.record.events.entry(format!("{kind}:{code}")).or_insert(0) += 1;
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn now_stamp() -> String {
    // Unix seconds + pid: unique per process and monotone enough to sort; the Python client's local-time stamp
    // is not part of any wire or record contract.
    format!("{}-{}", now_secs(), std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_json_shape() {
        let mut r = DuelRecord::default();
        r.seat = Some(1);
        r.winner = Some((0, 5));
        r.count("C0x1".into());
        let v = r.to_json();
        assert_eq!(v["winner"], json!([0, 5]));
        assert_eq!(v["packets"]["C0x1"], 1);
    }
}
