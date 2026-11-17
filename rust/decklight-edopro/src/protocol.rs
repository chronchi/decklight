//! EDOPro LAN protocol (`gframe/network.h`, `netserver.cpp`): packet ids, structs, framing.
//!
//! Byte-for-byte port of `src/decklight/edopro/protocol.py`. Framing (netserver.cpp `ServerEchoRead`):
//! u16 LE length (proto byte + payload), then u8 proto, then payload. One TCP read can hold a split or
//! several merged frames.

use std::fmt::Write as _;

pub const NETWORK_SERVER_ID: u32 = 0x7428;
pub const NETWORK_CLIENT_ID: u32 = 0xDEF6;
pub const SERVER_HANDSHAKE: u32 = 4_043_399_681;
pub const PRO_VERSION: u16 = 0x1354;
pub const CLIENT_VERSION: (u8, u8, u8, u8) = (41, 0, 11, 0);

pub const DISCOVERY_PORT: u16 = 7920;
pub const DISCOVERY_REPLY_PORT: u16 = 7921;

pub const CTOS_RESPONSE: u8 = 0x1;
pub const CTOS_UPDATE_DECK: u8 = 0x2;
pub const CTOS_HAND_RESULT: u8 = 0x3;
pub const CTOS_TP_RESULT: u8 = 0x4;
pub const CTOS_PLAYER_INFO: u8 = 0x10;
pub const CTOS_CREATE_GAME: u8 = 0x11;
pub const CTOS_JOIN_GAME: u8 = 0x12;
pub const CTOS_LEAVE_GAME: u8 = 0x13;
pub const CTOS_SURRENDER: u8 = 0x14;
pub const CTOS_TIME_CONFIRM: u8 = 0x15;
pub const CTOS_CHAT: u8 = 0x16;
pub const CTOS_REMATCH_RESPONSE: u8 = 0xF0;
pub const CTOS_HS_TODUELIST: u8 = 0x20;
pub const CTOS_HS_TOOBSERVER: u8 = 0x21;
pub const CTOS_HS_READY: u8 = 0x22;
pub const CTOS_HS_NOTREADY: u8 = 0x23;
pub const CTOS_HS_KICK: u8 = 0x24;
pub const CTOS_HS_START: u8 = 0x25;

pub const STOC_GAME_MSG: u8 = 0x1;
pub const STOC_ERROR_MSG: u8 = 0x2;
pub const STOC_SELECT_HAND: u8 = 0x3;
pub const STOC_SELECT_TP: u8 = 0x4;
pub const STOC_HAND_RESULT: u8 = 0x5;
pub const STOC_TP_RESULT: u8 = 0x6;
pub const STOC_CHANGE_SIDE: u8 = 0x7;
pub const STOC_WAITING_SIDE: u8 = 0x8;
pub const STOC_CREATE_GAME: u8 = 0x11;
pub const STOC_JOIN_GAME: u8 = 0x12;
pub const STOC_TYPE_CHANGE: u8 = 0x13;
pub const STOC_LEAVE_GAME: u8 = 0x14;
pub const STOC_DUEL_START: u8 = 0x15;
pub const STOC_DUEL_END: u8 = 0x16;
pub const STOC_NEW_REPLAY: u8 = 0x30;
pub const STOC_REMATCH: u8 = 0xF1;
pub const STOC_WAITING_REMATCH: u8 = 0xF2;
pub const STOC_REPLAY: u8 = 0x17;
pub const STOC_TIME_LIMIT: u8 = 0x18;
pub const STOC_CHAT: u8 = 0x19;
pub const STOC_HS_PLAYER_ENTER: u8 = 0x20;
pub const STOC_HS_PLAYER_CHANGE: u8 = 0x21;
pub const STOC_HS_WATCH_CHANGE: u8 = 0x22;

pub const MODE_SINGLE: u8 = 0x0;

pub const PLAYERCHANGE_READY: u8 = 0x9;
pub const PLAYERCHANGE_NOTREADY: u8 = 0xA;

pub const ERR_JOINERROR: u8 = 0x1;
pub const ERR_DECKERROR: u8 = 0x2;
pub const ERR_SIDEERROR: u8 = 0x3;
pub const ERR_VERERROR: u8 = 0x4;
pub const ERR_VERERROR2: u8 = 0x5;

pub const JERR_UNABLE: u32 = 0;
pub const JERR_PASSWORD: u32 = 1;
pub const JERR_REFUSED: u32 = 2;
const JERR_NAMES: [&str; 3] = ["unable to join", "wrong password", "join refused"];

pub const DERR_NONE: u32 = 0;
pub const DERR_LFLIST: u32 = 1;
pub const DERR_OCGONLY: u32 = 2;
pub const DERR_TCGONLY: u32 = 3;
pub const DERR_UNKNOWNCARD: u32 = 4;
pub const DERR_CARDCOUNT: u32 = 5;
pub const DERR_MAINCOUNT: u32 = 6;
pub const DERR_EXTRACOUNT: u32 = 7;
pub const DERR_SIDECOUNT: u32 = 8;
pub const DERR_FORBTYPE: u32 = 9;
pub const DERR_UNOFFICIALCARD: u32 = 10;
pub const DERR_INVALIDSIZE: u32 = 11;
pub const DERR_TOOMANYLEGENDS: u32 = 12;
pub const DERR_TOOMANYSKILLS: u32 = 13;
const DERR_NAMES: [&str; 14] = [
    "no error",
    "forbidden-list",
    "ocg-only",
    "tcg-only",
    "unknown card",
    "card count",
    "main count",
    "extra count",
    "side count",
    "forbidden type",
    "unofficial card",
    "invalid size",
    "too many legends",
    "too many skills",
];

pub const HS_OBSERVE: u8 = 0x8;
pub const HS_READY: u8 = 0x9;
pub const HS_NOTREADY: u8 = 0xA;
pub const HS_LEAVE: u8 = 0xB;

fn hs_name(state: u8) -> String {
    match state {
        HS_OBSERVE => "observe".into(),
        HS_READY => "ready".into(),
        HS_NOTREADY => "not ready".into(),
        HS_LEAVE => "leave".into(),
        s => format!("state {s}"),
    }
}

/// Wire versions for one end of the EDOPro LAN protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolConfig {
    /// The u16 `CTOS_JOIN_GAME.version` slot.
    pub pro_version: u16,
    /// The 4-byte `ClientVersion` (EDOPro major, EDOPro minor, core major, core minor).
    pub client_version: (u8, u8, u8, u8),
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        ProtocolConfig { pro_version: PRO_VERSION, client_version: CLIENT_VERSION }
    }
}

impl ProtocolConfig {
    /// `'41.0.11.0'` -> `(41, 0, 11, 0)`.
    pub fn parse_version(s: &str) -> Result<(u8, u8, u8, u8), String> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return Err(format!("client version needs 4 dotted parts, got {s:?}"));
        }
        let mut v = [0u8; 4];
        for (i, p) in parts.iter().enumerate() {
            let x: i64 = p.parse().map_err(|_| format!("client version needs integers, got {s:?}"))?;
            if !(0..=255).contains(&x) {
                return Err(format!("client version parts must be 0-255, got {s:?}"));
            }
            v[i] = x as u8;
        }
        Ok((v[0], v[1], v[2], v[3]))
    }

    /// `'0x1354'` (or `'4948'`) + `'41.0.11.0'`.
    pub fn from_strings(pro: &str, client: &str) -> Result<ProtocolConfig, String> {
        let pro_version = parse_int_auto(pro)?;
        if !(0..=0xFFFF).contains(&pro_version) {
            return Err(format!("pro version out of u16 range: {pro:?}"));
        }
        Ok(ProtocolConfig { pro_version: pro_version as u16, client_version: Self::parse_version(client)? })
    }

    /// WindBot-style packed int: major | minor<<8 | core_major<<16 | core_minor<<24.
    pub fn version_int(&self) -> u32 {
        let (a, b, c, d) = self.client_version;
        a as u32 | (b as u32) << 8 | (c as u32) << 16 | (d as u32) << 24
    }

    /// WindBot's `Version=` int -> the wire bytes WindBot sends.
    pub fn from_windbot_version(v: u32) -> ProtocolConfig {
        let b = v.to_le_bytes();
        ProtocolConfig { pro_version: (v & 0xFFFF) as u16, client_version: (b[0], b[1], b[2], b[3]) }
    }
}

fn parse_int_auto(s: &str) -> Result<i64, String> {
    let t = s.trim();
    let (sign, rest) = if let Some(r) = t.strip_prefix('-') { (-1i64, r) } else { (1i64, t) };
    let v = if let Some(r) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        i64::from_str_radix(r, 16)
    } else {
        rest.parse::<i64>()
    };
    v.map(|x| sign * x).map_err(|_| format!("not an integer: {s:?}"))
}

/// UTF-16LE name field, NUL-padded (`BufferIO::EncodeUTF16` semantics).
pub fn encode_name(s: &str, units: usize) -> Vec<u8> {
    // Python: `s[:units].encode("utf-16-le")` then NUL-pad to units*2 (a non-BMP char can make the
    // encoded field longer than units*2; Python then pads nothing, and we mirror that exactly).
    let mut out = Vec::with_capacity(units * 2);
    for c in s.chars().take(units) {
        let mut buf = [0u16; 2];
        for u in c.encode_utf16(&mut buf) {
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
    let want = units * 2;
    if out.len() < want {
        out.resize(want, 0);
    }
    out
}

pub fn decode_name(b: &[u8]) -> String {
    let mut units = Vec::with_capacity(b.len() / 2);
    for c in b.chunks_exact(2) {
        units.push(u16::from_le_bytes([c[0], c[1]]));
    }
    let mut s = String::from_utf16_lossy(&units);
    if let Some(i) = s.find('\0') {
        s.truncate(i);
    }
    s
}

pub fn client_version_bytes(major: u8, minor: u8, core_major: u8, core_minor: u8) -> [u8; 4] {
    [major, minor, core_major, core_minor]
}

fn need(b: &[u8], pos: usize, n: usize) -> Result<(), String> {
    if pos.checked_add(n).is_none_or(|e| e > b.len()) {
        return Err(format!("truncated at {pos}: need {n}, have {}", b.len().saturating_sub(pos)));
    }
    Ok(())
}

fn u8_at(b: &[u8], pos: usize) -> Result<u8, String> {
    need(b, pos, 1)?;
    Ok(b[pos])
}

fn u16_at(b: &[u8], pos: usize) -> Result<u16, String> {
    need(b, pos, 2)?;
    Ok(u16::from_le_bytes(b[pos..pos + 2].try_into().unwrap()))
}

fn u32_at(b: &[u8], pos: usize) -> Result<u32, String> {
    need(b, pos, 4)?;
    Ok(u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap()))
}

fn i32_at(b: &[u8], pos: usize) -> Result<i32, String> {
    need(b, pos, 4)?;
    Ok(i32::from_le_bytes(b[pos..pos + 4].try_into().unwrap()))
}

/// `HostInfo`, the 68-byte lobby struct (natural C++ alignment, not packed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostInfo {
    pub lflist: u32,
    pub rule: u8,
    pub mode: u8,
    pub duel_rule: u8,
    pub no_check_deck_content: u8,
    pub no_shuffle_deck: u8,
    pub start_lp: u32,
    pub start_hand: u8,
    pub draw_count: u8,
    pub time_limit: u16,
    pub duel_flag_high: u32,
    pub handshake: u32,
    pub version: [u8; 4],
    pub team1: i32,
    pub team2: i32,
    pub best_of: i32,
    pub duel_flag_low: u32,
    pub forbiddentypes: u32,
    pub extra_rules: u16,
    pub main_min: u16,
    pub main_max: u16,
    pub extra_min: u16,
    pub extra_max: u16,
    pub side_min: u16,
    pub side_max: u16,
}

impl Default for HostInfo {
    fn default() -> Self {
        HostInfo {
            lflist: 0,
            rule: 0,
            mode: MODE_SINGLE,
            duel_rule: 5,
            no_check_deck_content: 0,
            no_shuffle_deck: 0,
            start_lp: 8000,
            start_hand: 5,
            draw_count: 1,
            time_limit: 0,
            duel_flag_high: 0,
            handshake: SERVER_HANDSHAKE,
            version: client_version_bytes(CLIENT_VERSION.0, CLIENT_VERSION.1, CLIENT_VERSION.2, CLIENT_VERSION.3),
            team1: 1,
            team2: 1,
            best_of: 1,
            duel_flag_low: 0,
            forbiddentypes: 0,
            extra_rules: 0,
            main_min: 40,
            main_max: 60,
            extra_min: 0,
            extra_max: 15,
            side_min: 0,
            side_max: 15,
        }
    }
}

pub const HOSTINFO_SIZE: usize = 68;

impl HostInfo {
    pub fn pack(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(HOSTINFO_SIZE);
        o.extend_from_slice(&self.lflist.to_le_bytes());
        o.extend_from_slice(&[
            self.rule,
            self.mode,
            self.duel_rule,
            self.no_check_deck_content,
            self.no_shuffle_deck,
        ]);
        o.extend_from_slice(&[0, 0, 0]); // 3x pad
        o.extend_from_slice(&self.start_lp.to_le_bytes());
        o.extend_from_slice(&[self.start_hand, self.draw_count]);
        o.extend_from_slice(&self.time_limit.to_le_bytes());
        o.extend_from_slice(&self.duel_flag_high.to_le_bytes());
        o.extend_from_slice(&self.handshake.to_le_bytes());
        o.extend_from_slice(&self.version);
        o.extend_from_slice(&self.team1.to_le_bytes());
        o.extend_from_slice(&self.team2.to_le_bytes());
        o.extend_from_slice(&self.best_of.to_le_bytes());
        o.extend_from_slice(&self.duel_flag_low.to_le_bytes());
        o.extend_from_slice(&self.forbiddentypes.to_le_bytes());
        o.extend_from_slice(&self.extra_rules.to_le_bytes());
        for v in [self.main_min, self.main_max, self.extra_min, self.extra_max, self.side_min, self.side_max] {
            o.extend_from_slice(&v.to_le_bytes());
        }
        o.extend_from_slice(&[0, 0]); // 2x pad
        debug_assert_eq!(o.len(), HOSTINFO_SIZE);
        o
    }

    pub fn unpack(b: &[u8]) -> Result<HostInfo, String> {
        need(b, 0, HOSTINFO_SIZE)?;
        let version: [u8; 4] = b[28..32].try_into().unwrap();
        Ok(HostInfo {
            lflist: u32_at(b, 0)?,
            rule: u8_at(b, 4)?,
            mode: u8_at(b, 5)?,
            duel_rule: u8_at(b, 6)?,
            no_check_deck_content: u8_at(b, 7)?,
            no_shuffle_deck: u8_at(b, 8)?,
            start_lp: u32_at(b, 12)?,
            start_hand: u8_at(b, 16)?,
            draw_count: u8_at(b, 17)?,
            time_limit: u16_at(b, 18)?,
            duel_flag_high: u32_at(b, 20)?,
            handshake: u32_at(b, 24)?,
            version,
            team1: i32_at(b, 32)?,
            team2: i32_at(b, 36)?,
            best_of: i32_at(b, 40)?,
            duel_flag_low: u32_at(b, 44)?,
            forbiddentypes: u32_at(b, 48)?,
            extra_rules: u16_at(b, 52)?,
            main_min: u16_at(b, 54)?,
            main_max: u16_at(b, 56)?,
            extra_min: u16_at(b, 58)?,
            extra_max: u16_at(b, 60)?,
            side_min: u16_at(b, 62)?,
            side_max: u16_at(b, 64)?,
        })
    }
}

/// `DuelHost.host_info(version2)`: the host announces its rule set with the client's own version2 echoed, so any
/// EDOPro-family client matches. The duel flags are the TCG/MR5 flags our engine runs with.
pub fn host_info(version: [u8; 4]) -> HostInfo {
    let flags = ygoenv_sys::TCG_FLAGS;
    HostInfo {
        version,
        duel_flag_low: (flags & 0xFFFF_FFFF) as u32,
        duel_flag_high: (flags >> 32) as u32,
        ..HostInfo::default()
    }
}

/// `CTOS_JOIN_GAME`: u16 version, 2 pad, u32 gameid, u16 pass[20], ClientVersion (52 bytes).
pub fn pack_join_game(version: u16, gameid: u32, version2: [u8; 4], password: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(52);
    o.extend_from_slice(&version.to_le_bytes());
    o.extend_from_slice(&[0, 0]);
    o.extend_from_slice(&gameid.to_le_bytes());
    o.extend_from_slice(&encode_name(password, 20));
    o.extend_from_slice(&version2);
    debug_assert_eq!(o.len(), 52);
    o
}

/// `CTOS_JOIN_GAME` payload -> `(version, gameid, version2)`.
pub fn unpack_join_game(payload: &[u8]) -> Result<(u16, u32, [u8; 4]), String> {
    need(payload, 0, 52)?;
    let version = u16_at(payload, 0)?;
    let gameid = u32_at(payload, 4)?;
    let v2: [u8; 4] = payload[48..52].try_into().unwrap();
    Ok((version, gameid, v2))
}

/// `CTOS_CREATE_GAME`: HostInfo, u16 name[20], u16 pass[20], char notes[200] (348 bytes).
pub fn pack_create_game(info: &HostInfo, name: &str, password: &str, notes: &str) -> Vec<u8> {
    let mut raw = notes.as_bytes().to_vec();
    raw.truncate(199);
    let mut o = info.pack();
    o.extend_from_slice(&encode_name(name, 20));
    o.extend_from_slice(&encode_name(password, 20));
    o.extend_from_slice(&raw);
    o.resize(o.len() + (200 - raw.len()), 0);
    debug_assert_eq!(o.len(), 348);
    o
}

/// `STOC_CREATE_GAME` payload -> the new room id.
pub fn parse_create_game(payload: &[u8]) -> Result<u32, String> {
    if payload.len() < 4 {
        return Err(format!("create-game packet too short: {}", payload.len()));
    }
    u32_at(payload, 0)
}

/// `STOC_HS_PLAYER_ENTER`: u16 name[20], u8 pos, 1 pad (42 bytes).
pub fn pack_hs_player_enter(name: &str, pos: u8) -> Vec<u8> {
    let mut o = encode_name(name, 20);
    o.extend_from_slice(&[pos, 0]);
    o
}

/// `CTOS_PLAYER_INFO` payload: u16 name[20].
pub fn pack_player_info(name: &str) -> Vec<u8> {
    encode_name(name, 20)
}

/// `CTOS_UPDATE_DECK` payload: u32 mainc (= main + extra), u32 sidec, then codes.
pub fn pack_update_deck(main: &[u32], extra: &[u32], side: &[u32]) -> Vec<u8> {
    let codes: Vec<u32> = main.iter().chain(extra).chain(side).copied().collect();
    let mut o = Vec::with_capacity(8 + 4 * codes.len());
    o.extend_from_slice(&((main.len() + extra.len()) as u32).to_le_bytes());
    o.extend_from_slice(&(side.len() as u32).to_le_bytes());
    for c in codes {
        o.extend_from_slice(&c.to_le_bytes());
    }
    o
}

/// `CTOS_HAND_RESULT` payload: the RPS throw (1 scissors / 2 rock / 3 paper).
pub fn pack_hand_choice(res: u8) -> Result<Vec<u8>, String> {
    if !(1..=3).contains(&res) {
        return Err(format!("RPS throw must be 1/2/3, got {res:?}"));
    }
    Ok(vec![res])
}

/// `CTOS_TP_RESULT` payload: 1 = go first, 0 = go second.
pub fn pack_tp_choice(first: bool) -> Vec<u8> {
    vec![u8::from(first)]
}

/// `CTOS_RESPONSE` payload: the raw core response (already encoded).
pub fn pack_response(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

pub fn pack_time_confirm() -> Vec<u8> {
    Vec::new()
}

/// `STOC_HAND_RESULT` payload -> `(ours, theirs)`.
pub fn parse_hand_result(payload: &[u8]) -> Result<(u8, u8), String> {
    if payload.len() < 2 {
        return Err(format!("hand result too short: {}", payload.len()));
    }
    Ok((payload[0], payload[1]))
}

/// `STOC_TP_RESULT` payload -> true when we go first.
pub fn parse_tp_result(payload: &[u8]) -> Result<bool, String> {
    if payload.is_empty() {
        return Err("tp result is empty".into());
    }
    Ok(payload[0] == 1)
}

pub fn pack_hand_result(res1: u8, res2: u8) -> Vec<u8> {
    vec![res1, res2]
}

/// `STOC_JOIN_GAME` payload -> HostInfo.
pub fn parse_join_game(payload: &[u8]) -> Result<HostInfo, String> {
    if payload.len() < HOSTINFO_SIZE {
        return Err(format!("join-game packet too short: {}", payload.len()));
    }
    HostInfo::unpack(payload)
}

/// `STOC_TYPE_CHANGE` payload -> `(seat, is_host)`.
pub fn parse_type_change(payload: &[u8]) -> Result<(u8, bool), String> {
    let t = *payload.first().ok_or("type-change packet is empty")?;
    Ok((t & 0xF, ((t >> 4) & 0xF) != 0))
}

/// `STOC_HS_PLAYER_ENTER` payload -> `(name, seat)`.
pub fn parse_hs_player_enter(payload: &[u8]) -> Result<(String, u8), String> {
    if payload.len() < 41 {
        return Err(format!("player-enter packet too short: {}", payload.len()));
    }
    Ok((decode_name(&payload[..40]), payload[40]))
}

/// `STOC_HS_PLAYER_CHANGE` payload -> `(seat, state name)`.
pub fn parse_hs_player_change(payload: &[u8]) -> Result<(u8, String), String> {
    let status = *payload.first().ok_or("player-change packet is empty")?;
    Ok(((status >> 4) & 0xF, hs_name(status & 0xF)))
}

/// `STOC_TIME_LIMIT` payload -> `(player, seconds left)`.
pub fn parse_time_limit(payload: &[u8]) -> Result<(u8, u16), String> {
    if payload.len() < 4 {
        return Err(format!("time-limit packet too short: {}", payload.len()));
    }
    Ok((payload[0], u16_at(payload, 2)?))
}

/// `STOC_ERROR_MSG` payload -> readable text.
pub fn parse_error_msg(payload: &[u8]) -> String {
    let Some(&etype) = payload.first() else {
        return "empty error message".into();
    };
    match etype {
        ERR_JOINERROR => {
            let code = if payload.len() >= 8 { u32_at(payload, 4).unwrap_or(0) } else { 0 };
            let name = JERR_NAMES.get(code as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("code {code}"));
            format!("join error: {name}")
        }
        ERR_DECKERROR => {
            if payload.len() < 24 {
                return "deck error: (truncated)".into();
            }
            let t = u32_at(payload, 4).unwrap_or(0);
            let current = u32_at(payload, 8).unwrap_or(0);
            let minimum = u32_at(payload, 12).unwrap_or(0);
            let maximum = u32_at(payload, 16).unwrap_or(0);
            let code = u32_at(payload, 20).unwrap_or(0);
            let name = DERR_NAMES.get(t as usize).map(|s| s.to_string()).unwrap_or_else(|| format!("type {t}"));
            let mut msg = format!("deck error: {name} (current {current}, allowed {minimum}-{maximum})");
            if t == DERR_UNKNOWNCARD {
                let _ = write!(msg, " card {code}");
            }
            msg
        }
        ERR_SIDEERROR => {
            let code = if payload.len() >= 8 { u32_at(payload, 4).unwrap_or(0) } else { 0 };
            format!("side-deck error: code {code}")
        }
        ERR_VERERROR | ERR_VERERROR2 => {
            let ver = if payload.len() >= 8 { &payload[4..8] } else { &[][..] };
            let want =
                if ver.is_empty() { "?".to_string() } else { ver.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(".") };
            format!("version error: server wants client version {want}")
        }
        t => format!("unknown error type {t}"),
    }
}

/// `STOC_ERROR_MSG` payload for a DECKERROR (`network.h` DeckError layout).
pub fn pack_deck_error(derr_type: u32, current: u32, minimum: u32, maximum: u32, code: u32) -> Vec<u8> {
    let mut o = Vec::with_capacity(24);
    o.push(ERR_DECKERROR);
    o.extend_from_slice(&[0, 0, 0]);
    for v in [derr_type, current, minimum, maximum, code] {
        o.extend_from_slice(&v.to_le_bytes());
    }
    o
}

pub fn pack_join_error(code: u32) -> Vec<u8> {
    let mut o = Vec::with_capacity(8);
    o.push(ERR_JOINERROR);
    o.extend_from_slice(&[0, 0, 0]);
    o.extend_from_slice(&code.to_le_bytes());
    o
}

/// `CTOS_UPDATE_DECK` payload -> `(main_plus_extra codes, side codes)`.
pub fn parse_update_deck(payload: &[u8]) -> Result<(Vec<u32>, Vec<u32>), String> {
    if payload.len() < 8 {
        return Err(format!("deck packet too short: {}", payload.len()));
    }
    let mainc = u32_at(payload, 0)? as usize;
    let sidec = u32_at(payload, 4)? as usize;
    if 8 + 4 * (mainc + sidec) > payload.len() {
        return Err(format!(
            "deck packet claims {mainc}+{sidec} cards but holds {} bytes",
            payload.len()
        ));
    }
    let mut codes = Vec::with_capacity(mainc + sidec);
    for i in 0..(mainc + sidec) {
        codes.push(u32_at(payload, 8 + 4 * i)?);
    }
    let side = codes.split_off(mainc);
    Ok((codes, side))
}

/// `MSG_START` for one recipient (`generic_duel.cpp` TPResult): seat = recipient's engine seat.
pub fn build_start(lp: u32, deck0: u32, extra0: u32, deck1: u32, extra1: u32, seat: u8) -> Vec<u8> {
    let mut o = Vec::with_capacity(20);
    o.extend_from_slice(&[4, seat]);
    o.extend_from_slice(&lp.to_le_bytes());
    o.extend_from_slice(&lp.to_le_bytes());
    for v in [deck0, extra0, deck1, extra1] {
        o.extend_from_slice(&(v as u16).to_le_bytes());
    }
    o
}

pub fn pack_frame(proto: u8, payload: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(3 + payload.len());
    o.extend_from_slice(&((payload.len() + 1) as u16).to_le_bytes());
    o.push(proto);
    o.extend_from_slice(payload);
    o
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub proto: u8,
    pub payload: Vec<u8>,
}

/// Accumulates TCP reads into frames; handles split and merged reads. Never panics on malformed input.
#[derive(Default)]
pub struct FrameReader {
    buf: Vec<u8>,
}

impl FrameReader {
    pub fn new() -> FrameReader {
        FrameReader::default()
    }

    pub fn feed(&mut self, data: &[u8]) -> Result<Vec<Frame>, String> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();
        let mut pos = 0usize;
        while self.buf.len() - pos >= 2 {
            let n = u16::from_le_bytes([self.buf[pos], self.buf[pos + 1]]) as usize;
            if n == 0 {
                return Err("zero-length frame (no protocol byte)".into());
            }
            if self.buf.len() - pos < 2 + n {
                break;
            }
            let frame = &self.buf[pos + 2..pos + 2 + n];
            out.push(Frame { proto: frame[0], payload: frame[1..].to_vec() });
            pos += 2 + n;
        }
        self.buf.drain(..pos);
        Ok(out)
    }

    /// No partial frame may remain at the end of a stream.
    pub fn finish(&self) -> Result<(), String> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(format!("truncated frame: {} buffered bytes", self.buf.len()))
        }
    }

    pub fn buffered(&self) -> usize {
        self.buf.len()
    }
}

/// Whether throw `a` beats `b`: 1 scissors, 2 rock, 3 paper (EDOPro `windbot_panel.cpp`; `generic_duel.cpp`
/// HandResult lets the opposing player choose turn order on (1, 2), (2, 3), (3, 1)).
pub fn rps_beats(a: u8, b: u8) -> bool {
    matches!((a, b), (2, 1) | (3, 2) | (1, 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostinfo_round_trip() {
        let h = HostInfo { rule: 4, start_lp: 4000, main_min: 40, side_max: 15, ..HostInfo::default() };
        let b = h.pack();
        assert_eq!(b.len(), HOSTINFO_SIZE);
        assert_eq!(HostInfo::unpack(&b).unwrap(), h);
    }

    #[test]
    fn frame_reader_splits_and_merges() {
        let a = pack_frame(STOC_GAME_MSG, &[1, 2, 3]);
        let b = pack_frame(STOC_CHAT, b"hi");
        let mut all = a.clone();
        all.extend_from_slice(&b);
        let mut r = FrameReader::new();
        let mut got = r.feed(&all[..2]).unwrap();
        assert!(got.is_empty());
        got.extend(r.feed(&all[2..]).unwrap());
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], Frame { proto: STOC_GAME_MSG, payload: vec![1, 2, 3] });
        assert_eq!(got[1], Frame { proto: STOC_CHAT, payload: b"hi".to_vec() });
        r.finish().unwrap();
    }

    #[test]
    fn frame_reader_rejects_zero_length() {
        let mut r = FrameReader::new();
        assert!(r.feed(&[0, 0]).is_err());
    }

    #[test]
    fn malformed_structs_are_errors_not_panics() {
        for n in 0..64 {
            let junk = vec![0xFFu8; n];
            let _ = HostInfo::unpack(&junk);
            let _ = unpack_join_game(&junk);
            let _ = parse_join_game(&junk);
            let _ = parse_hs_player_enter(&junk);
            let _ = parse_time_limit(&junk);
            let _ = parse_update_deck(&junk);
            let _ = parse_error_msg(&junk);
        }
    }

    #[test]
    fn versions() {
        assert_eq!(ProtocolConfig::parse_version("41.0.11.0").unwrap(), (41, 0, 11, 0));
        assert!(ProtocolConfig::parse_version("1.2.3").is_err());
        assert_eq!(ProtocolConfig::from_windbot_version(40 | 1 << 8 | 10 << 16).version_int(), 40 | 1 << 8 | 10 << 16);
    }
}
