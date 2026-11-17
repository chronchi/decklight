//! Engine-message cues for the web client (port of `src/decklight/web/cues.py`, behaviour for behaviour).

use std::collections::BTreeMap;

use decklight_core::constants as C;
use serde_json::{Value, json};

use crate::{Ctx, uid};

const PILES: [u32; 5] = [
    C::LOCATION_DECK,
    C::LOCATION_HAND,
    C::LOCATION_GRAVE,
    C::LOCATION_REMOVED,
    C::LOCATION_EXTRA,
];

const MATERIAL_REASONS: u32 = C::REASON_FUSION
    | C::REASON_SYNCHRO
    | C::REASON_XYZ
    | C::REASON_LINK
    | C::REASON_RITUAL
    | C::REASON_MATERIAL
    | C::REASON_RELEASE;

/// `MSG_FOO` for a known message type, else `MSG_<hex>`.
pub fn message_name(t: u32) -> String {
    let n = match t {
        0x1 => "MSG_RETRY",
        0x2 => "MSG_HINT",
        0x3 => "MSG_WAITING",
        0x4 => "MSG_START",
        0x5 => "MSG_WIN",
        0x6 => "MSG_UPDATE_DATA",
        0x7 => "MSG_UPDATE_CARD",
        0x8 => "MSG_REQUEST_DECK",
        0xA => "MSG_SELECT_BATTLECMD",
        0xB => "MSG_SELECT_IDLECMD",
        0xC => "MSG_SELECT_EFFECTYN",
        0xD => "MSG_SELECT_YESNO",
        0xE => "MSG_SELECT_OPTION",
        0xF => "MSG_SELECT_CARD",
        0x10 => "MSG_SELECT_CHAIN",
        0x12 => "MSG_SELECT_PLACE",
        0x13 => "MSG_SELECT_POSITION",
        0x14 => "MSG_SELECT_TRIBUTE",
        0x15 => "MSG_SORT_CHAIN",
        0x16 => "MSG_SELECT_COUNTER",
        0x17 => "MSG_SELECT_SUM",
        0x18 => "MSG_SELECT_DISFIELD",
        0x19 => "MSG_SORT_CARD",
        0x1A => "MSG_SELECT_UNSELECT_CARD",
        0x1E => "MSG_CONFIRM_DECKTOP",
        0x1F => "MSG_CONFIRM_CARDS",
        0x20 => "MSG_SHUFFLE_DECK",
        0x21 => "MSG_SHUFFLE_HAND",
        0x22 => "MSG_REFRESH_DECK",
        0x23 => "MSG_SWAP_GRAVE_DECK",
        0x24 => "MSG_SHUFFLE_SET_CARD",
        0x25 => "MSG_REVERSE_DECK",
        0x26 => "MSG_DECK_TOP",
        0x27 => "MSG_SHUFFLE_EXTRA",
        0x28 => "MSG_NEW_TURN",
        0x29 => "MSG_NEW_PHASE",
        0x2A => "MSG_CONFIRM_EXTRATOP",
        0x32 => "MSG_MOVE",
        0x35 => "MSG_POS_CHANGE",
        0x36 => "MSG_SET",
        0x37 => "MSG_SWAP",
        0x38 => "MSG_FIELD_DISABLED",
        0x3C => "MSG_SUMMONING",
        0x3D => "MSG_SUMMONED",
        0x3E => "MSG_SPSUMMONING",
        0x3F => "MSG_SPSUMMONED",
        0x40 => "MSG_FLIPSUMMONING",
        0x41 => "MSG_FLIPSUMMONED",
        0x46 => "MSG_CHAINING",
        0x47 => "MSG_CHAINED",
        0x48 => "MSG_CHAIN_SOLVING",
        0x49 => "MSG_CHAIN_SOLVED",
        0x4A => "MSG_CHAIN_END",
        0x4B => "MSG_CHAIN_NEGATED",
        0x4C => "MSG_CHAIN_DISABLED",
        0x50 => "MSG_CARD_SELECTED",
        0x51 => "MSG_RANDOM_SELECTED",
        0x53 => "MSG_BECOME_TARGET",
        0x5A => "MSG_DRAW",
        0x5B => "MSG_DAMAGE",
        0x5C => "MSG_RECOVER",
        0x5D => "MSG_EQUIP",
        0x5E => "MSG_LPUPDATE",
        0x5F => "MSG_UNEQUIP",
        0x60 => "MSG_CARD_TARGET",
        0x61 => "MSG_CANCEL_TARGET",
        0x64 => "MSG_PAY_LPCOST",
        0x65 => "MSG_ADD_COUNTER",
        0x66 => "MSG_REMOVE_COUNTER",
        0x6E => "MSG_ATTACK",
        0x6F => "MSG_BATTLE",
        0x70 => "MSG_ATTACK_DISABLED",
        0x71 => "MSG_DAMAGE_STEP_START",
        0x72 => "MSG_DAMAGE_STEP_END",
        0x78 => "MSG_MISSED_EFFECT",
        0x79 => "MSG_BE_CHAIN_TARGET",
        0x7A => "MSG_CREATE_RELATION",
        0x7B => "MSG_RELEASE_RELATION",
        0x82 => "MSG_TOSS_COIN",
        0x83 => "MSG_TOSS_DICE",
        0x84 => "MSG_ROCK_PAPER_SCISSORS",
        0x85 => "MSG_HAND_RES",
        0x8C => "MSG_ANNOUNCE_RACE",
        0x8D => "MSG_ANNOUNCE_ATTRIB",
        0x8E => "MSG_ANNOUNCE_CARD",
        0x8F => "MSG_ANNOUNCE_NUMBER",
        0xA0 => "MSG_CARD_HINT",
        0xA1 => "MSG_TAG_SWAP",
        0xA2 => "MSG_RELOAD_FIELD",
        0xA3 => "MSG_AI_NAME",
        0xA4 => "MSG_SHOW_HINT",
        0xA5 => "MSG_PLAYER_HINT",
        0xAA => "MSG_MATCH_KILL",
        0xB4 => "MSG_CUSTOM_MSG",
        0xBE => "MSG_REMOVE_CARDS",
        _ => return format!("MSG_{t:#x}"),
    };
    n.to_string()
}

fn short_phase(phase: Option<&str>) -> &'static str {
    match phase.unwrap_or("main1") {
        "draw" => "Draw",
        "standby" => "Standby",
        "main1" => "Main 1",
        "battle" => "Battle",
        "main2" => "Main 2",
        "end" => "End",
        _ => "Main 1",
    }
}

fn zone_name(loc: u32) -> &'static str {
    match loc {
        C::LOCATION_HAND => "hand",
        C::LOCATION_DECK => "deck",
        C::LOCATION_EXTRA => "extra",
        C::LOCATION_MZONE => "mzone",
        C::LOCATION_SZONE => "szone",
        C::LOCATION_GRAVE => "grave",
        C::LOCATION_REMOVED => "banished",
        C::LOCATION_OVERLAY => "overlay",
        _ => "unknown",
    }
}

fn side(ctrl: u32, viewer: i32) -> &'static str {
    if ctrl as i32 == viewer { "me" } else { "opp" }
}

fn zone(ctrl: u32, loc: u32, seq: i64, viewer: i32) -> Value {
    let mut name = zone_name(loc);
    if loc == C::LOCATION_SZONE && seq == 5 {
        name = "field";
    }
    json!({"side": side(ctrl, viewer), "loc": name, "seq": seq})
}

fn end_visible(ctrl: u32, loc: u32, pos: u32, viewer: i32) -> bool {
    if loc == C::LOCATION_GRAVE || loc == C::LOCATION_OVERLAY {
        return true;
    }
    if loc == 0 {
        return false;
    }
    let faceup = pos & C::POS_FACEUP != 0;
    let owner = ctrl as i32 == viewer;
    if matches!(
        loc,
        C::LOCATION_MZONE | C::LOCATION_SZONE | C::LOCATION_REMOVED | C::LOCATION_DECK | C::LOCATION_EXTRA
    ) {
        return faceup || owner;
    }
    owner
}

fn move_reason(reason: u32, prev_loc: u32, cur_loc: u32, cur_pos: u32) -> &'static str {
    if reason & C::REASON_DESTROY != 0 {
        return "destroyed";
    }
    if reason & MATERIAL_REASONS != 0 {
        return "material";
    }
    if reason & C::REASON_SUMMON != 0 {
        return "summon";
    }
    if reason & C::REASON_SPSUMMON != 0 {
        return "spsummon";
    }
    if matches!(cur_loc, C::LOCATION_MZONE | C::LOCATION_SZONE) && cur_pos & C::POS_FACEDOWN != 0 {
        return "set";
    }
    if cur_loc == C::LOCATION_REMOVED {
        return "banished";
    }
    if reason & C::REASON_RETURN != 0 {
        return "returned";
    }
    if matches!(cur_loc, C::LOCATION_DECK | C::LOCATION_EXTRA) && prev_loc != cur_loc {
        return "returned";
    }
    if cur_loc == C::LOCATION_HAND {
        return if prev_loc == C::LOCATION_DECK { "added" } else { "returned" };
    }
    if cur_loc == C::LOCATION_GRAVE {
        return "sent";
    }
    "other"
}

fn int32(v: u32) -> i64 {
    if v >= 1 << 31 { v as i64 - (1 << 32) } else { v as i64 }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, p: 1 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err(format!("truncated at {}: need {n}", self.p));
        }
        let out = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn loc_info(&mut self) -> Result<(u32, u32, u32, u32), String> {
        Ok((self.u8()? as u32, self.u8()? as u32, self.u32()?, self.u32()?))
    }
}

struct CueState<'a> {
    ctx: &'a Ctx,
    viewer: i32,
    reveal_all: bool,
    turn: i64,
    lp: [i64; 2],
    chain_card: BTreeMap<u32, Value>,
    last_link: Option<u32>,
    pending: Vec<(u32, u32)>,
    last_release: BTreeMap<u32, i64>,
    /// Core-message counter (host refresh packets excluded) and its value per trace index:
    /// the distance heuristics only make sense over the messages the engine logged.
    msg_no: i64,
    msg_at: Vec<i64>,
    slot_code: BTreeMap<(u32, u32, i64), u32>,
    cues: Vec<Value>,
    skipped: BTreeMap<String, usize>,
}

impl CueState<'_> {
    fn skip(&mut self, name: &str) {
        *self.skipped.entry(name.to_string()).or_insert(0) += 1;
    }

    fn forget(&mut self, ctrl: u32, loc: u32, seq: i64) {
        if !PILES.contains(&loc) {
            self.slot_code.remove(&(ctrl, loc, seq));
        }
    }

    fn card_ref(
        &self, code: u32, ctrl: u32, loc: u32, seq: i64, pos: u32, force_show: bool,
    ) -> Value {
        let show = self.reveal_all || force_show || end_visible(ctrl, loc, pos, self.viewer);
        let code = if show { code } else { 0 };
        let name = if code == 0 { String::new() } else { self.ctx.name_or(code as i64) };
        json!({
            "code": code,
            "name": name,
            "owner": side(ctrl, self.viewer),
            "zone": zone(ctrl, loc, seq, self.viewer),
            "uid": uid(ctrl as i64, loc as i64, seq, code as i64),
        })
    }

    fn sp_how(&self, code: u32) -> &'static str {
        let t = self.ctx.card_type(code);
        if t & C::TYPE_LINK != 0 {
            "link"
        } else if t & C::TYPE_XYZ != 0 {
            "xyz"
        } else if t & C::TYPE_SYNCHRO != 0 {
            "synchro"
        } else if t & C::TYPE_FUSION != 0 {
            "fusion"
        } else if t & C::TYPE_RITUAL != 0 {
            "ritual"
        } else if t & C::TYPE_PENDULUM != 0 {
            "pendulum"
        } else {
            "special"
        }
    }

    fn handle(&mut self, n: usize, raw: &[u8], emit: bool) -> Result<(), String> {
        let t = raw[0] as u32;
        let mut r = Reader::new(raw);
        match t {
            C::MSG_NEW_TURN => {
                let p = r.u8()? as u32;
                self.turn += 1;
                if emit {
                    self.cues.push(json!({"t": "turn", "n": n, "turn": self.turn, "player": side(p & 1, self.viewer)}));
                }
            }
            C::MSG_NEW_PHASE => {
                let phase = r.u16()? as u32;
                let short = short_phase(crate::phase_name(phase as i32));
                if emit {
                    self.cues.push(json!({"t": "phase", "n": n, "phase": short}));
                }
            }
            C::MSG_DRAW => {
                let p = r.u8()? as u32;
                let count = r.u32()?;
                let mut refs = Vec::new();
                for _ in 0..count {
                    let code = r.u32()?;
                    let pos = r.u32()?;
                    refs.push(self.card_ref(code, p, C::LOCATION_HAND, -1, pos, false));
                }
                if emit {
                    self.cues.push(json!({"t": "draw", "n": n, "player": side(p, self.viewer), "count": count, "cards": refs}));
                }
            }
            C::MSG_MOVE => {
                let code = r.u32()?;
                let (pc, pl, ps, pp) = r.loc_info()?;
                let (cc, cl, cs, cp) = r.loc_info()?;
                let reason = r.u32()?;
                if reason & C::REASON_RELEASE != 0 {
                    self.last_release.insert(pc, self.msg_no);
                }
                if pc == cc && pl == cl && PILES.contains(&pl) {
                    self.skip(&message_name(t));
                    return Ok(());
                }
                if reason & (C::REASON_SUMMON | C::REASON_SPSUMMON | C::REASON_FLIP) != 0 {
                    let hit = self
                        .pending
                        .iter()
                        .position(|(sc, ss)| *ss == cc && (*sc == code || *sc == 0 || code == 0));
                    if let Some(i) = hit {
                        self.pending.remove(i);
                        self.forget(pc, pl, ps as i64);
                        self.slot_code.insert((cc, cl, cs as i64), code);
                        return Ok(());
                    }
                }
                self.forget(pc, pl, ps as i64);
                if code != 0 {
                    self.slot_code.insert((cc, cl, cs as i64), code);
                } else {
                    self.slot_code.remove(&(cc, cl, cs as i64));
                }
                if !emit {
                    return Ok(());
                }
                let show = if cl == 0 {
                    self.reveal_all || end_visible(pc, pl, pp, self.viewer)
                } else {
                    self.reveal_all || end_visible(cc, cl, cp, self.viewer)
                };
                let card = self.card_ref(code, cc, cl, cs as i64, cp, show);
                self.cues.push(json!({
                    "t": "move",
                    "n": n,
                    "card": card,
                    "from": zone(pc, pl, ps as i64, self.viewer),
                    "to": zone(cc, cl, cs as i64, self.viewer),
                    "reason": move_reason(reason, pl, cl, cp),
                }));
            }
            C::MSG_SUMMONING | C::MSG_SPSUMMONING | C::MSG_FLIPSUMMONING => {
                let code = r.u32()?;
                let (fc, fl, fs, _fp) = r.loc_info()?;
                self.pending.push((code, fc));
                if code != 0 {
                    self.slot_code.insert((fc, fl, fs as i64), code);
                }
                let how = if t == C::MSG_FLIPSUMMONING {
                    "flip"
                } else if t == C::MSG_SUMMONING {
                    if self.msg_no - self.last_release.get(&fc).copied().unwrap_or(-(1_000_000_000)) <= 8 {
                        "tribute"
                    } else {
                        "normal"
                    }
                } else {
                    self.sp_how(code)
                };
                let mut dest: Option<(u32, i64)> = None;
                let mut from_zone = zone(fc, fl, fs as i64, self.viewer);
                if code != 0 && emit {
                    for i in (0..self.cues.len()).rev() {
                        let cue_n = self.cues[i]["n"].as_i64().unwrap_or(0).max(0) as usize;
                        let too_old = self.msg_no - self.msg_at.get(cue_n).copied().unwrap_or(0) > 4;
                        if too_old {
                            break;
                        }
                        let matches = self.cues[i]["t"] == "move"
                            && self.cues[i]["card"]["code"].as_u64().unwrap_or(0) as u32 == code
                            && self.cues[i]["card"]["owner"].as_str().unwrap_or("") == side(fc, self.viewer)
                            && self.cues[i]["to"]["loc"].as_str().unwrap_or("") == "mzone";
                        if matches {
                            let to = &self.cues[i]["to"];
                            let dside = if to["side"].as_str().unwrap_or("") == "me" {
                                self.viewer
                            } else {
                                1 - self.viewer
                            };
                            dest = Some((dside as u32, to["seq"].as_i64().unwrap_or(-1)));
                            from_zone = self.cues[i]["from"].clone();
                            self.cues.remove(i);
                            break;
                        }
                    }
                }
                let card = match dest {
                    Some((ctrl, seq)) => self.card_ref(
                        code, ctrl, C::LOCATION_MZONE, seq, C::POS_FACEUP_ATTACK, true,
                    ),
                    None => self.card_ref(
                        code, fc, C::LOCATION_MZONE, -1, C::POS_FACEUP_ATTACK, code != 0,
                    ),
                };
                if emit {
                    self.cues.push(json!({"t": "summon", "n": n, "card": card, "how": how, "from": from_zone}));
                }
            }
            C::MSG_CHAINING => {
                let code = r.u32()?;
                let (hc, hl, hs, hp) = r.loc_info()?;
                let _ = (r.u8()?, r.u8()?, r.u32()?);
                let desc = r.u64()?;
                let link = r.u32()?;
                let effect = {
                    let s = self.ctx.desc_text((desc >> 20) as i64, (desc & 0xFFFFF) as i64);
                    if s.is_empty() { Value::Null } else { Value::String(s) }
                };
                let rf = self.card_ref(code, hc, hl, hs as i64, hp, true);
                self.chain_card.insert(link, rf.clone());
                self.last_link = Some(link);
                if code != 0 {
                    self.slot_code.insert((hc, hl, hs as i64), code);
                }
                if emit {
                    self.cues.push(json!({"t": "activate", "n": n, "card": rf, "link": link, "effect": effect}));
                }
            }
            C::MSG_CHAIN_SOLVING | C::MSG_CHAIN_SOLVED | C::MSG_CHAIN_NEGATED | C::MSG_CHAIN_DISABLED => {
                let link = r.u8()? as u32;
                if emit {
                    let name = match t {
                        C::MSG_CHAIN_SOLVING => "chain_solving",
                        C::MSG_CHAIN_SOLVED => "chain_solved",
                        C::MSG_CHAIN_NEGATED => "chain_negated",
                        _ => "chain_disabled",
                    };
                    self.cues.push(json!({"t": name, "n": n, "link": link}));
                }
            }
            C::MSG_CHAIN_END => {
                self.chain_card.clear();
                self.last_link = None;
                if emit {
                    self.cues.push(json!({"t": "chain_end", "n": n}));
                }
            }
            C::MSG_BECOME_TARGET => {
                let count = r.u32()?;
                let mut targets = Vec::new();
                for _ in 0..count {
                    targets.push(r.loc_info()?);
                }
                let source = self
                    .last_link
                    .and_then(|l| self.chain_card.get(&l).cloned())
                    .unwrap_or(Value::Null);
                if emit {
                    let cards: Vec<Value> = targets
                        .iter()
                        .map(|(tc, tl, ts, tp)| {
                            let code = self.slot_code.get(&(*tc, *tl, *ts as i64)).copied().unwrap_or(0);
                            self.card_ref(code, *tc, *tl, *ts as i64, *tp, false)
                        })
                        .collect();
                    self.cues.push(json!({"t": "target", "n": n, "source": source, "cards": cards}));
                }
            }
            C::MSG_ATTACK => {
                let (ac, al, aseq, ap) = r.loc_info()?;
                let (tc, tl, ts, tp) = r.loc_info()?;
                if !emit {
                    return Ok(());
                }
                let code = self.slot_code.get(&(ac, al, aseq as i64)).copied().unwrap_or(0);
                let attacker = self.card_ref(code, ac, al, aseq as i64, ap, false);
                let target = if tl == 0 {
                    Value::Null
                } else {
                    let code = self.slot_code.get(&(tc, tl, ts as i64)).copied().unwrap_or(0);
                    self.card_ref(code, tc, tl, ts as i64, tp, false)
                };
                self.cues.push(json!({"t": "attack", "n": n, "attacker": attacker, "target": target}));
            }
            C::MSG_BATTLE => {
                let (ac, al, aseq, ap) = r.loc_info()?;
                let aatk = int32(r.u32()?);
                let adef = int32(r.u32()?);
                let _ = r.u8()?;
                let (tc, tl, ts, tp) = r.loc_info()?;
                let datk = int32(r.u32()?);
                let ddef = int32(r.u32()?);
                let _ = r.u8()?;
                if !emit {
                    return Ok(());
                }
                let code = self.slot_code.get(&(ac, al, aseq as i64)).copied().unwrap_or(0);
                let mut attacker = self.card_ref(code, ac, al, aseq as i64, ap, false);
                set_num(&mut attacker, "atk", aatk);
                set_num(&mut attacker, "def", adef);
                let defender = if tl == 0 {
                    Value::Null
                } else {
                    let code = self.slot_code.get(&(tc, tl, ts as i64)).copied().unwrap_or(0);
                    let mut d = self.card_ref(code, tc, tl, ts as i64, tp, false);
                    set_num(&mut d, "atk", datk);
                    set_num(&mut d, "def", ddef);
                    d
                };
                self.cues.push(json!({"t": "battle", "n": n, "attacker": attacker, "target": defender}));
            }
            C::MSG_DAMAGE => {
                let p = r.u8()? as usize;
                let amount = r.u32()? as i64;
                self.lp[p] = 0.max(self.lp[p] - amount);
                if emit {
                    self.cues.push(json!({
                        "t": "damage",
                        "n": n,
                        "player": side(p as u32, self.viewer),
                        "amount": amount,
                        "lp_after": self.lp[p],
                    }));
                }
            }
            C::MSG_RECOVER => {
                let p = r.u8()? as usize;
                let amount = r.u32()? as i64;
                self.lp[p] += amount;
                if emit {
                    self.cues.push(json!({
                        "t": "recover",
                        "n": n,
                        "player": side(p as u32, self.viewer),
                        "amount": amount,
                        "lp_after": self.lp[p],
                    }));
                }
            }
            C::MSG_LPUPDATE => {
                let p = r.u8()? as usize;
                let val = r.u32()?;
                self.lp[p] = int32(val);
                if emit {
                    self.cues.push(json!({"t": "lp", "n": n, "player": side(p as u32, self.viewer), "lp": self.lp[p]}));
                }
            }
            C::MSG_PAY_LPCOST => {
                let p = r.u8()? as usize;
                let cost = r.u32()? as i64;
                self.lp[p] = 0.max(self.lp[p] - cost);
                if emit {
                    self.cues.push(json!({
                        "t": "pay",
                        "n": n,
                        "player": side(p as u32, self.viewer),
                        "amount": cost,
                        "lp_after": self.lp[p],
                    }));
                }
            }
            C::MSG_CONFIRM_CARDS => {
                let p = r.u8()? as u32;
                let count = r.u32()?;
                let mut cards = Vec::new();
                for _ in 0..count {
                    let code = r.u32()?;
                    let cc = r.u8()? as u32;
                    let cl = r.u8()? as u32;
                    let cs = r.u32()?;
                    if code != 0 {
                        self.slot_code.insert((cc, cl, cs as i64), code);
                    }
                    if emit {
                        cards.push(self.card_ref(code, cc, cl, cs as i64, C::POS_FACEUP, true));
                    }
                }
                if emit {
                    self.cues.push(json!({"t": "confirm", "n": n, "player": side(p, self.viewer), "cards": cards}));
                }
            }
            C::MSG_SHUFFLE_HAND | C::MSG_SHUFFLE_DECK => {
                let p = r.u8()? as u32;
                if emit {
                    self.cues.push(json!({
                        "t": "shuffle",
                        "n": n,
                        "player": side(p, self.viewer),
                        "loc": if t == C::MSG_SHUFFLE_HAND { "hand" } else { "deck" },
                    }));
                }
            }
            C::MSG_WIN => {
                let w = r.u8()?;
                let reason = r.u8()?;
                if emit {
                    self.cues.push(json!({
                        "t": "win",
                        "n": n,
                        "winner": if w == 2 { "draw" } else { side(w as u32, self.viewer) },
                        "reason": reason,
                    }));
                }
            }
            _ => self.skip(&message_name(t)),
        }
        Ok(())
    }
}

fn set_num(v: &mut Value, key: &str, n: i64) {
    if let Some(o) = v.as_object_mut() {
        o.insert(key.to_string(), json!(n));
    }
}

/// Cues for `trace[since:]`, the new trace length, and skipped-message counts.
pub fn cues_for_trace(
    ctx: &Ctx,
    trace: &[Vec<u8>],
    viewer: i32,
    since: usize,
    reveal_all: bool,
    lp0: i64,
) -> (Vec<Value>, usize, BTreeMap<String, usize>) {
    cues_from(ctx, trace.len(), |n| trace[n].clone(), viewer, since, reveal_all, lp0)
}

/// `cues_for_trace` over an arbitrary trace accessor (no whole-trace copy).
fn cues_from<F: Fn(usize) -> Vec<u8>>(
    ctx: &Ctx,
    len: usize,
    get: F,
    viewer: i32,
    since: usize,
    reveal_all: bool,
    lp0: i64,
) -> (Vec<Value>, usize, BTreeMap<String, usize>) {
    let mut st = CueState {
        ctx,
        viewer,
        reveal_all,
        turn: 0,
        lp: [lp0, lp0],
        chain_card: BTreeMap::new(),
        last_link: None,
        pending: Vec::new(),
        last_release: BTreeMap::new(),
        msg_no: 0,
        msg_at: Vec::with_capacity(len),
        slot_code: BTreeMap::new(),
        cues: Vec::new(),
        skipped: BTreeMap::new(),
    };
    for n in 0..len {
        let raw = get(n);
        let emit = n >= since;
        if raw.is_empty() {
            st.skip("MSG_empty");
            continue;
        }
        let t = raw[0] as u32;
        if t != C::MSG_UPDATE_DATA && t != C::MSG_UPDATE_CARD {
            st.msg_no += 1;
        }
        st.msg_at.push(st.msg_no);
        if st.handle(n, &raw, emit).is_err() {
            st.skip(&message_name(t));
        }
    }
    (st.cues, len, st.skipped)
}

/// Cues for the source's trace `[since:]` and the new trace length.
pub fn build_cues(
    source: &dyn crate::ViewSource, ctx: &Ctx, viewer: i32, since: usize, reveal_all: bool,
) -> (Vec<Value>, usize) {
    let (cues, len, _) =
        cues_from(ctx, source.trace_len(), |n| source.trace_msg(n), viewer, since, reveal_all, ctx.lp0);
    (cues, len)
}
