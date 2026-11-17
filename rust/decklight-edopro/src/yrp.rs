//! EDOPro/Multirole replays (`.yrpX`, sent to every client as `STOC_NEW_REPLAY`) -> match record v1.
//!
//! Port of `src/decklight/play/yrp.py` (Multirole `src/Multirole/YGOPro/Replay.cpp`, f258e00). The replay
//! does not record in which order the two teams' cards were added to the core; `to_record` replays both
//! orders and keeps the one whose core messages equal the recorded ones (ADR 0010).

use std::io::{Read, Write};
use std::path::Path;

use decklight_core::constants as C;
use lzma_rust2::{EncodeMode, LzmaOptions, LzmaReader, LzmaWriter, MfType};
use serde_json::{Value, json};
use ygoenv_sys::Engine;

use crate::record::{self, FORMAT};

pub const YRP1: u32 = 0x3170_7279;
pub const YRPX: u32 = 0x5870_7279;
pub const F_COMPRESSED: u32 = 0x1;
pub const F_TAG: u32 = 0x2;
pub const F_SINGLE: u32 = 0x8;
pub const F_EXTENDED: u32 = 0x200;
pub const OLD_REPLAY_FORMAT: u8 = 231;
pub const HDR_SIZE: usize = 72;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Yrp {
    pub seed: [u64; 4],
    pub names: Vec<Vec<String>>,
    pub lp: u32,
    pub hand: u32,
    pub draw: u32,
    pub flags: u64,
    /// `[team][duelist] -> (main in core order, extra)`.
    pub decks: Vec<Vec<(Vec<u32>, Vec<u32>)>>,
    pub extra_cards: Vec<u32>,
    pub responses: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct YrpX {
    pub timestamp: u64,
    pub names: Vec<Vec<String>>,
    pub flags: u64,
    /// Type byte + data, in order.
    pub messages: Vec<Vec<u8>>,
    pub yrp: Option<Yrp>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, p: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p.checked_add(n).is_none_or(|e| e > self.b.len()) {
            return Err(format!("replay truncated at {}: need {n}, have {}", self.p, self.b.len() - self.p));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn names(&mut self) -> Result<Vec<Vec<String>>, String> {
        let mut out = Vec::new();
        for _ in 0..2 {
            let n = self.u32()?;
            if n > 3 {
                return Err(format!("bad duelist count {n}"));
            }
            let mut team = Vec::new();
            for _ in 0..n {
                team.push(decode_utf16_name(self.take(40)?));
            }
            out.push(team);
        }
        Ok(out)
    }

    fn codes(&mut self) -> Result<Vec<u32>, String> {
        let n = self.u32()?;
        if n > 1000 {
            return Err(format!("bad card count {n}"));
        }
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            out.push(self.u32()?);
        }
        Ok(out)
    }

    fn left(&self) -> usize {
        self.b.len() - self.p
    }
}

fn decode_utf16_name(b: &[u8]) -> String {
    let units: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut s = String::from_utf16_lossy(&units);
    if let Some(i) = s.find('\0') {
        s.truncate(i);
    }
    s
}

fn decompress(body: &[u8], size: u64, props: &[u8; 8]) -> Result<Vec<u8>, String> {
    let lc = (props[0] % 9) as u32;
    let lp = ((props[0] / 9) % 5) as u32;
    let pb = (props[0] / 45) as u32;
    let dict_size = u32::from_le_bytes(props[1..5].try_into().unwrap()).max(4096);
    let mut r = LzmaReader::new(body, size, lc, lp, pb, dict_size, None).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(size as usize);
    r.read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

fn compress(body: &[u8]) -> Result<(Vec<u8>, [u8; 5]), String> {
    let mut opts = LzmaOptions::with_preset(1);
    opts.dict_size = 1 << 20;
    opts.lc = 3;
    opts.lp = 0;
    opts.pb = 2;
    opts.mode = EncodeMode::Normal;
    opts.nice_len = 128;
    opts.mf = MfType::Bt4;
    opts.depth_limit = 0;
    let mut w = LzmaWriter::new_no_header(Vec::new(), &opts, true).map_err(|e| e.to_string())?;
    w.write_all(body).map_err(|e| e.to_string())?;
    let data = w.finish().map_err(|e| e.to_string())?;
    let mut props = [0u8; 5];
    props[0] = ((opts.pb * 5 + opts.lp) * 9 + opts.lc) as u8;
    props[1..].copy_from_slice(&opts.dict_size.to_le_bytes());
    Ok((data, props))
}

fn header(rtype: u32, flags: u32, ts: u64, size: u64, props: &[u8], seed: [u64; 4]) -> Vec<u8> {
    let mut o = Vec::with_capacity(HDR_SIZE);
    for v in [rtype, 0, flags, ts as u32, size as u32, 0] {
        o.extend_from_slice(&v.to_le_bytes());
    }
    let mut p = [0u8; 8];
    let n = props.len().min(8);
    p[..n].copy_from_slice(&props[..n]);
    o.extend_from_slice(&p);
    o.extend_from_slice(&1u64.to_le_bytes());
    for w in seed {
        o.extend_from_slice(&w.to_le_bytes());
    }
    debug_assert_eq!(o.len(), HDR_SIZE);
    o
}

fn parse_body(data: &[u8], want: u32) -> Result<(u32, u64, Vec<u8>, [u64; 4]), String> {
    if data.len() < HDR_SIZE {
        return Err("replay shorter than its header".into());
    }
    let h0 = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let flags = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let ts = u32::from_le_bytes(data[12..16].try_into().unwrap()) as u64;
    let size = u32::from_le_bytes(data[16..20].try_into().unwrap()) as u64;
    let props: [u8; 8] = data[24..32].try_into().unwrap();
    let mut seed = [0u64; 4];
    for (i, w) in seed.iter_mut().enumerate() {
        *w = u64::from_le_bytes(data[40 + 8 * i..48 + 8 * i].try_into().unwrap());
    }
    if h0 != want {
        return Err(format!("replay type {h0:#x}, expected {want:#x}"));
    }
    if flags & F_EXTENDED == 0 {
        return Err("old replay header (no 256-bit seed): not supported".into());
    }
    let raw = &data[HDR_SIZE..];
    let body = if flags & F_COMPRESSED != 0 { decompress(raw, size, &props)? } else { raw.to_vec() };
    if body.len() as u64 != size {
        return Err(format!("replay body is {} bytes, header says {size}", body.len()));
    }
    Ok((flags, ts, body, seed))
}

pub fn parse_yrp(data: &[u8]) -> Result<Yrp, String> {
    let (flags, _ts, body, _seed) = parse_body(data, YRP1)?;
    if flags & F_TAG != 0 {
        return Err("tag duels are not supported (match record v1 is 1v1)".into());
    }
    let mut r = Reader::new(&body);
    let names = r.names()?;
    let lp = r.u32()?;
    let hand = r.u32()?;
    let draw = r.u32()?;
    let dflags = r.u64()?;
    let mut decks = Vec::new();
    for team in &names {
        let mut t = Vec::new();
        for _ in team {
            t.push((r.codes()?, r.codes()?));
        }
        decks.push(t);
    }
    let extra = r.codes()?;
    let mut responses = Vec::new();
    while r.left() > 0 {
        let n = r.u8()? as usize;
        responses.push(r.take(n)?.to_vec());
    }
    Ok(Yrp { seed: _seed, names, lp, hand, draw, flags: dflags, decks, extra_cards: extra, responses })
}

pub fn parse_yrpx(data: &[u8]) -> Result<YrpX, String> {
    let (flags, timestamp, body, _seed) = parse_body(data, YRPX)?;
    let mut r = Reader::new(&body);
    let names = r.names()?;
    let bflags = r.u64()?;
    let mut out = YrpX { timestamp, names, flags: bflags, messages: Vec::new(), yrp: None };
    while r.left() > 0 {
        let t = r.u8()?;
        let n = r.u32()? as usize;
        let payload = r.take(n)?;
        let mut msg = Vec::with_capacity(1 + n);
        msg.push(t);
        msg.extend_from_slice(payload);
        if t == OLD_REPLAY_FORMAT {
            out.yrp = Some(parse_yrp(&msg[1..])?);
        } else {
            out.messages.push(msg);
        }
    }
    if out.yrp.is_none() {
        return Err("no embedded .yrp (message 231): cannot replay".into());
    }
    let _ = flags;
    Ok(out)
}

pub fn core_messages(x: &YrpX) -> Vec<Vec<u8>> {
    let server = [C::MSG_START as u8, C::MSG_UPDATE_DATA as u8, C::MSG_UPDATE_CARD as u8, OLD_REPLAY_FORMAT];
    x.messages.iter().filter(|m| !server.contains(&m[0])).cloned().collect()
}

/// The messages Multirole's `Replay::RecordMsg` keeps: no select/announce messages, no player-specific hints
/// (types 1, 2, 3, 5). Idempotent, so it also applies to filtered replays (`yrp.recorded` in Python).
pub fn recorded(msgs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    const UNRECORDED: [u32; 21] = [
        C::MSG_SELECT_BATTLECMD, C::MSG_SELECT_IDLECMD, C::MSG_SELECT_EFFECTYN, C::MSG_SELECT_YESNO,
        C::MSG_SELECT_OPTION, C::MSG_SELECT_CHAIN, C::MSG_SELECT_PLACE, C::MSG_SELECT_DISFIELD,
        C::MSG_SELECT_POSITION, C::MSG_SELECT_COUNTER, C::MSG_SELECT_SUM, C::MSG_SORT_CARD, C::MSG_SORT_CHAIN,
        C::MSG_ROCK_PAPER_SCISSORS, C::MSG_ANNOUNCE_RACE, C::MSG_ANNOUNCE_ATTRIB, C::MSG_ANNOUNCE_CARD,
        C::MSG_ANNOUNCE_NUMBER, C::MSG_SELECT_CARD, C::MSG_SELECT_TRIBUTE, C::MSG_SELECT_UNSELECT_CARD,
    ];
    msgs.iter()
        .filter(|m| {
            let Some(&t) = m.first() else { return false };
            !UNRECORDED.contains(&(t as u32))
                && !(t as u32 == C::MSG_HINT && m.len() > 1 && matches!(m[1], 1 | 2 | 3 | 5))
        })
        .cloned()
        .collect()
}

/// Our add order <-> EDOPro's (every card at sequence 0), giving the same deck: our index 1 goes to the bottom
/// (core field::add_card: sequence 1 = deck bottom). Its own inverse (`yrp.edopro_main` in Python).
pub fn edopro_main(main: &[u32]) -> Vec<u32> {
    let mut m = main.to_vec();
    if m.len() > 1 {
        m.swap(0, 1);
    }
    m
}

/// The trace with each run of consecutive CARD_HINT/PLAYER_HINT messages sorted (hint order follows
/// heap addresses; NOTEBOOK 2026-10-01). Used for replay comparison: hints are UI-only.
pub fn canonical_trace(mut trace: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
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

fn names_bytes(names: &[Vec<String>]) -> Vec<u8> {
    let mut out = Vec::new();
    for team in names {
        out.extend_from_slice(&(team.len() as u32).to_le_bytes());
        for n in team {
            // Python: n.encode("utf-16-le")[:40].ljust(40, b"\0") (byte truncation, not code points).
            let mut b = Vec::new();
            for u in n.encode_utf16() {
                b.extend_from_slice(&u.to_le_bytes());
            }
            b.truncate(40);
            b.resize(40, 0);
            out.extend_from_slice(&b);
        }
    }
    out
}

fn codes_bytes(codes: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 4 * codes.len());
    out.extend_from_slice(&(codes.len() as u32).to_le_bytes());
    for c in codes {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out
}

/// Serialize like Multirole's `Replay::Serialize` (MSG_START first, `recorded` core messages, the yrp appended as
/// message 231). `y.decks` are in EDOPro's add order (`edopro_main` of ours).
pub fn write_yrpx(y: &Yrp, core_msgs: &[Vec<u8>], ts: u64, compress_body: bool) -> Result<Vec<u8>, String> {
    let mut flags = 0x10u32 | 0x100 | 0x20 | F_EXTENDED;
    let mut yb = names_bytes(&y.names);
    yb.extend_from_slice(&y.lp.to_le_bytes());
    yb.extend_from_slice(&y.hand.to_le_bytes());
    yb.extend_from_slice(&y.draw.to_le_bytes());
    yb.extend_from_slice(&y.flags.to_le_bytes());
    for team in &y.decks {
        for (main, extra) in team {
            yb.extend_from_slice(&codes_bytes(main));
            yb.extend_from_slice(&codes_bytes(extra));
        }
    }
    yb.extend_from_slice(&codes_bytes(&y.extra_cards));
    for r in &y.responses {
        yb.push(r.len() as u8);
        yb.extend_from_slice(r);
    }
    let mut yrp = vec![OLD_REPLAY_FORMAT];
    yrp.extend_from_slice(&header(YRP1, flags, 0, yb.len() as u64, &[], y.seed));
    yrp.extend_from_slice(&yb);

    let mut start = vec![C::MSG_START as u8, 0];
    start.extend_from_slice(&y.lp.to_le_bytes());
    start.extend_from_slice(&y.lp.to_le_bytes());
    for v in [0u16, 0, 0, 0] {
        start.extend_from_slice(&v.to_le_bytes());
    }

    let mut body = names_bytes(&y.names);
    body.extend_from_slice(&y.flags.to_le_bytes());
    for m in std::iter::once(&start).chain(recorded(core_msgs).iter()).chain(std::iter::once(&yrp)) {
        body.push(m[0]);
        body.extend_from_slice(&((m.len() - 1) as u32).to_le_bytes());
        body.extend_from_slice(&m[1..]);
    }

    let (data, props) = if compress_body {
        let (data, props) = compress(&body)?;
        flags |= F_COMPRESSED;
        (data, props.to_vec())
    } else {
        (body.clone(), Vec::new())
    };
    let mut out = header(YRPX, flags, ts, body.len() as u64, &props, [0, 0, 0, 0]);
    out.extend_from_slice(&data);
    Ok(out)
}

/// A saved `full` record as an EDOPro replay (Games > Export, mocks export-*): the record's responses, cut to the first
/// `upto` when set (EDOPro replays always start at turn 1, so a part of a game is a prefix), written like Multirole
/// does. The file is then read back the way EDOPro reads it (`to_record`: rebuild the duel from the file alone) and
/// must reproduce the same responses, so a file that comes out of here plays in an EDOPro with the same core and
/// scripts. `names` are the two teams' names in core order (≤ 19 characters show in full).
pub fn record_to_yrpx(
    engine: &Engine,
    root: &Path,
    rec: &Value,
    names: [String; 2],
    upto: Option<usize>,
    ts: u64,
) -> Result<Vec<u8>, String> {
    let cfg = record::config_of(rec)?;
    if cfg.team_order != [0, 1] {
        return Err("this game came from a room replay whose card order EDOPro would not rebuild; \
                    export it as text or video instead"
            .into());
    }
    let mut responses: Vec<Vec<u8>> = rec["responses"]
        .as_array()
        .ok_or("the saved game has no responses")?
        .iter()
        .map(|e| hex_decode(e["r"].as_str().unwrap_or("")))
        .collect::<Result<_, _>>()?;
    if let Some(n) = upto {
        if n == 0 || n > responses.len() {
            return Err(format!("cannot cut the replay at response {n} of {}", responses.len()));
        }
        responses.truncate(n);
    }
    let out = record::replay_responses(engine, rec, &responses)?;
    let seed = cfg.seed_state.ok_or("the saved game has no engine seed")?;
    let main = |m: &Vec<u32>| if cfg.edopro_add { m.clone() } else { edopro_main(m) };
    let y = Yrp {
        seed,
        names: names.iter().map(|n| vec![n.chars().take(19).collect()]).collect(),
        lp: cfg.lp,
        hand: cfg.hand,
        draw: cfg.draw,
        flags: cfg.flags,
        decks: vec![vec![(main(&cfg.main0), cfg.extra0.clone())], vec![(main(&cfg.main1), cfg.extra1.clone())]],
        extra_cards: Vec::new(),
        responses: responses.clone(),
    };
    let data = write_yrpx(&y, &out.trace, ts, true)?;
    let back = parse_yrpx(&data).map_err(|e| format!("the replay file does not read back: {e}"))?;
    let (rebuilt, problems) = to_record(&back, engine, root, None, "export-check", None, None);
    let rebuilt = rebuilt.ok_or_else(|| format!("the replay file does not rebuild the game: {}", problems.join("; ")))?;
    // EDOPro adds every card at sequence 0, team 0 first: the file must rebuild that way, not by another order.
    if rebuilt["deck_add"] != "edopro" || rebuilt.get("add_order").is_some() {
        return Err("the replay file rebuilds the game only with another card order than EDOPro's".into());
    }
    let n = rebuilt["responses"].as_array().map_or(0, Vec::len);
    if n != responses.len() {
        return Err(format!("the replay file rebuilds {n} of {} responses", responses.len()));
    }
    Ok(data)
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err(format!("bad response hex {s:?}"));
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| format!("bad response hex {s:?}"))).collect()
}

fn iso(ts: u64) -> String {
    // datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    let days = (ts / 86_400) as i64;
    let secs = (ts % 86_400) as u32;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", secs / 3600, (secs / 60) % 60, secs % 60)
}

/// Howard Hinnant's `civil_from_days` (days since 1970-01-01 -> Y/M/D).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// A `full` record (source "yrpx") and the problems found; None + problems when no card-add way and order replays
/// (EDOPro servers add at sequence 0, our hosts with the list index: `yrp.to_record` in Python).
pub fn to_record(
    x: &YrpX,
    engine: &Engine,
    root: &Path,
    viewer: Option<i32>,
    app: &str,
    seats: Option<Vec<Value>>,
    record_id: Option<String>,
) -> (Option<Value>, Vec<String>) {
    let Some(y) = &x.yrp else {
        return (None, vec!["no embedded .yrp (message 231): cannot replay".into()]);
    };
    if y.decks.iter().any(|t| t.len() != 1) {
        return (None, vec![format!("not a 1v1 duel: {:?} duelists", y.decks.iter().map(Vec::len).collect::<Vec<_>>())]);
    }
    if !y.extra_cards.is_empty() {
        return (None, vec![format!("extra-rule cards are not supported yet: {:?}", y.extra_cards)]);
    }
    let want = canonical_trace(recorded(&core_messages(x)));
    let mut problems = Vec::new();
    for (add, order) in [("edopro", [0u8, 1u8]), ("edopro", [1, 0]), ("ours", [0, 1]), ("ours", [1, 0])] {
        let decks: Vec<Value> = y
            .decks
            .iter()
            .map(|t| {
                let (file_main, extra) = &t[0];
                let main = if add == "edopro" { file_main.clone() } else { edopro_main(file_main) };
                json!({
                    "main": main,
                    "extra": extra,
                    "hash": record::deck_hash(&main, extra),
                })
            })
            .collect();
        let mut rec = json!({
            "format": FORMAT,
            "id": record_id.clone().unwrap_or_else(|| record::uuid7(x.timestamp * 1000)),
            "created": iso(x.timestamp),
            "app": app,
            "engine": record::engine_info(root),
            "cards": {"pack": "server", "sha256": "0".repeat(64), "server": true},
            "rules": {
                "duel_flags": y.flags.to_string(),
                "start_lp": y.lp,
                "start_hand": y.hand,
                "draw_count": y.draw,
            },
            "kind": "full",
            "engine_seed": y.seed.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
            "decks": decks,
            "seats": seats.clone().unwrap_or_else(|| {
                vec![
                    json!({"type": if viewer == Some(0) {"human"} else {"remote"}, "client": "unknown"}),
                    json!({"type": if viewer == Some(1) {"human"} else {"remote"}, "client": "unknown"}),
                ]
            }),
            "mode": "online",
            "responses": y.responses.iter().map(|r| json!({"seat": -1, "r": hex(r), "opt": null})).collect::<Vec<_>>(),
            "result": {"winner": null, "turns": 0},
            "source": "yrpx",
        });
        if order != [0, 1] {
            rec["add_order"] = json!(order);
        }
        if add == "edopro" {
            rec["deck_add"] = json!("edopro");
        }
        let responses: Vec<Vec<u8>> = y.responses.clone();
        let out = match record::replay_responses(engine, &rec, &responses) {
            Ok(o) => o,
            Err(e) => {
                problems.push(format!("{add} add, order ({}, {}): replay failed: {e}", order[0], order[1]));
                continue;
            }
        };
        let got = &canonical_trace(recorded(&out.trace));
        let n = got.len().min(want.len());
        let differs = got[..n] != want[..n] || got.len() + 1 < want.len();
        if differs {
            let k = (0..n).find(|&i| got[i] != want[i]).unwrap_or(n);
            problems.push(format!(
                "{add} add, order ({}, {}): recorded messages differ at {k} of {}",
                order[0],
                order[1],
                want.len()
            ));
            continue;
        }
        let mut responses_json = rec["responses"].as_array_mut().unwrap().clone();
        for (e, s) in responses_json.iter_mut().zip(out.actors.iter()) {
            e["seat"] = json!(s);
        }
        rec["responses"] = Value::Array(responses_json);
        rec["result"] = result(&out, x);
        return (Some(rec), Vec::new());
    }
    (None, problems)
}

fn result(out: &record::ReplayOut, x: &YrpX) -> Value {
    let mut winner = out.winner;
    let mut surrendered = false;
    if winner.is_none() {
        if let Some(m) = x.messages.iter().filter(|m| m[0] == C::MSG_WIN as u8).next_back() {
            if m.len() > 1 && (m[1] == 0 || m[1] == 1) {
                winner = Some(m[1] as i32);
                surrendered = true;
            }
        }
    }
    json!({
        "winner": winner,
        "reason": out.win_reason,
        "turns": out.turn,
        "surrendered": surrendered,
        "disconnected": false,
    })
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Yrp {
        Yrp {
            seed: [1, 2, 3, 4],
            names: vec![vec!["a".into()], vec!["b".into()]],
            lp: 8000,
            hand: 5,
            draw: 1,
            flags: 0x2E800,
            decks: vec![vec![(vec![10, 11, 12], vec![20])], vec![(vec![13, 14], vec![21])]],
            extra_cards: vec![],
            responses: vec![vec![1, 0, 0, 0], vec![2, 0, 0, 0, 1, 1]],
        }
    }

    #[test]
    fn yrpx_round_trip_both_compressions() {
        for compress in [false, true] {
            let y = sample();
            let data = write_yrpx(&y, &[vec![0x5A, 1, 2, 3]], 1234, compress).unwrap();
            let x = parse_yrpx(&data).unwrap();
            assert_eq!(x.timestamp, 1234);
            assert_eq!(x.yrp.as_ref().unwrap().decks, y.decks);
            assert_eq!(x.yrp.as_ref().unwrap().responses, y.responses);
            assert_eq!(x.yrp.as_ref().unwrap().seed, y.seed);
        }
    }

    #[test]
    fn truncated_replays_are_errors_not_panics() {
        let y = sample();
        let data = write_yrpx(&y, &[vec![0x5A, 1]], 1, true).unwrap();
        for n in 0..data.len() {
            let _ = parse_yrpx(&data[..n]);
        }
        for n in [0usize, 1, 40, 71, 72, 100] {
            let _ = parse_yrp(&vec![0xFFu8; n]);
        }
    }
}
