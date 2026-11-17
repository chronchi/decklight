//! Per-player filtering of core messages and board refresh packets (1v1, no observers).
//!
//! A port of `generic_duel.cpp` `Sending` / `BeforeParsing` / `AfterParsing` for a single human viewer
//! (engine seat 0 or 1), byte-for-byte with `src/decklight/edopro/filter.py`.

use decklight_core::constants as C;
use decklight_core::messages as M;
use decklight_core::query::{self, CardState};
use ygoenv_sys::Env;

fn need(b: &[u8], pos: usize, n: usize) -> Result<(), String> {
    if pos.checked_add(n).is_none_or(|e| e > b.len()) {
        return Err(format!("message truncated at {pos}: need {n}, have {}", b.len().saturating_sub(pos)));
    }
    Ok(())
}

fn u8_at(b: &[u8], pos: usize) -> Result<u8, String> {
    need(b, pos, 1)?;
    Ok(b[pos])
}

fn u32_at(b: &[u8], pos: usize) -> Result<u32, String> {
    need(b, pos, 4)?;
    Ok(u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap()))
}

fn put_u32(b: &mut [u8], pos: usize, v: u32) -> Result<(), String> {
    need(b, pos, 4)?;
    b[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
    Ok(())
}

/// `loc_info` at `pos` -> `(controller, location, sequence, position)`.
fn loc(b: &[u8], pos: usize) -> Result<(u8, u8, u32, u32), String> {
    let con = u8_at(b, pos)?;
    let location = u8_at(b, pos + 1)?;
    let seq = u32_at(b, pos + 2)?;
    let position = u32_at(b, pos + 6)?;
    Ok((con, location, seq, position))
}

/// The copy of core message `raw` for engine seat `viewer` (Ok(None) = send nothing).
pub fn filtered_message(raw: &[u8], viewer: u8) -> Result<Option<Vec<u8>>, String> {
    let t = *raw.first().ok_or("empty core message")? as u32;
    if t == C::MSG_HINT {
        let htype = u8_at(raw, 1)?;
        let player = u8_at(raw, 2)?;
        if matches!(htype, 1 | 2 | 3 | 5 | 200) {
            return Ok((viewer == player).then(|| raw.to_vec()));
        }
        if matches!(htype, 4 | 6 | 7 | 8 | 9 | 11) {
            return Ok((viewer != player).then(|| raw.to_vec()));
        }
        if matches!(htype, 10 | 201 | 202 | 203) {
            return Ok(Some(raw.to_vec()));
        }
        return Ok(None);
    }
    if t == C::MSG_SELECT_IDLECMD || t == C::MSG_SELECT_BATTLECMD || t == C::MSG_WIN {
        if t == C::MSG_WIN {
            return Ok(Some(raw.to_vec()));
        }
        let player = u8_at(raw, 1)?;
        return Ok((viewer == player).then(|| raw.to_vec()));
    }
    if M::is_select_type(t)
        && t != C::MSG_SELECT_CARD
        && t != C::MSG_SELECT_TRIBUTE
        && t != C::MSG_SELECT_UNSELECT_CARD
        && t != C::MSG_SELECT_PLACE
        && t != C::MSG_SELECT_DISFIELD
    {
        let player = u8_at(raw, 1)?;
        return Ok((viewer == player).then(|| raw.to_vec()));
    }
    if t == C::MSG_SELECT_PLACE || t == C::MSG_SELECT_DISFIELD {
        let player = u8_at(raw, 1)?;
        return Ok((viewer == player).then(|| raw.to_vec()));
    }
    if t == C::MSG_SELECT_CARD {
        let player = u8_at(raw, 1)?;
        if viewer != player {
            return Ok(None);
        }
        let count = u32_at(raw, 11)? as usize;
        let mut out = raw.to_vec();
        let mut pos = 15usize;
        for _ in 0..count {
            if u8_at(&out, pos + 4)? != player {
                put_u32(&mut out, pos, 0)?;
            }
            pos += 14;
        }
        return Ok(Some(out));
    }
    if t == C::MSG_SELECT_TRIBUTE {
        let player = u8_at(raw, 1)?;
        if viewer != player {
            return Ok(None);
        }
        let count = u32_at(raw, 11)? as usize;
        let mut out = raw.to_vec();
        let mut pos = 15usize;
        for _ in 0..count {
            if u8_at(&out, pos + 4)? != player {
                put_u32(&mut out, pos, 0)?;
            }
            pos += 11;
        }
        return Ok(Some(out));
    }
    if t == C::MSG_SELECT_UNSELECT_CARD {
        let player = u8_at(raw, 1)?;
        if viewer != player {
            return Ok(None);
        }
        let mut out = raw.to_vec();
        let mut pos = 12usize;
        for _ in 0..2 {
            let count = u32_at(&out, pos)? as usize;
            pos += 4;
            for _ in 0..count {
                if u8_at(&out, pos + 4)? != player {
                    put_u32(&mut out, pos, 0)?;
                }
                pos += 14;
            }
        }
        return Ok(Some(out));
    }
    if t == C::MSG_CONFIRM_CARDS {
        let player = u8_at(raw, 1)?;
        let count = u32_at(raw, 2)? as usize;
        if count == 0 {
            return Ok(Some(raw.to_vec()));
        }
        let location = u8_at(raw, 11)? as u32;
        if location == C::LOCATION_DECK || location == C::LOCATION_EXTRA {
            return Ok((viewer == player).then(|| raw.to_vec()));
        }
        return Ok(Some(raw.to_vec()));
    }
    if t == C::MSG_SHUFFLE_HAND || t == C::MSG_SHUFFLE_EXTRA {
        let player = u8_at(raw, 1)?;
        if viewer == player {
            return Ok(Some(raw.to_vec()));
        }
        let count = u32_at(raw, 2)? as usize;
        let mut out = raw.to_vec();
        for i in 0..count {
            put_u32(&mut out, 6 + 4 * i, 0)?;
        }
        return Ok(Some(out));
    }
    if t == C::MSG_MOVE {
        let _code = u32_at(raw, 1)?;
        let _prev = loc(raw, 5)?;
        let (con, location, _seq, position) = loc(raw, 15)?;
        if viewer == con {
            return Ok(Some(raw.to_vec()));
        }
        let location = location as u32;
        if location & (C::LOCATION_GRAVE + C::LOCATION_OVERLAY) == 0
            && (location & (C::LOCATION_DECK + C::LOCATION_HAND) != 0 || (position & C::POS_FACEDOWN) != 0)
        {
            let mut out = raw.to_vec();
            put_u32(&mut out, 1, 0)?;
            return Ok(Some(out));
        }
        return Ok(Some(raw.to_vec()));
    }
    if t == C::MSG_SET {
        let mut out = raw.to_vec();
        put_u32(&mut out, 1, 0)?;
        return Ok(Some(out));
    }
    if t == C::MSG_SPSUMMONING {
        let (con, _, _seq, position) = loc(raw, 5)?;
        if position & C::POS_FACEDOWN == 0 {
            return Ok(Some(raw.to_vec()));
        }
        if viewer == con {
            return Ok(Some(raw.to_vec()));
        }
        let mut out = raw.to_vec();
        put_u32(&mut out, 1, 0)?;
        return Ok(Some(out));
    }
    if t == C::MSG_DRAW {
        let player = u8_at(raw, 1)?;
        if viewer == player {
            return Ok(Some(raw.to_vec()));
        }
        let count = u32_at(raw, 2)? as usize;
        let mut out = raw.to_vec();
        let mut pos = 6usize;
        for _ in 0..count {
            let p = u32_at(&out, pos + 4)?;
            if p & C::POS_FACEUP == 0 {
                put_u32(&mut out, pos, 0)?;
            }
            pos += 8;
        }
        return Ok(Some(out));
    }
    Ok(Some(raw.to_vec()))
}

/// The engine seat that must answer `raw`, if it awaits a response.
pub fn waiting_player(raw: &[u8]) -> Option<u8> {
    let t = *raw.first()? as u32;
    if M::is_select_type(t) {
        raw.get(1).copied()
    } else {
        None
    }
}

/// One board query the host runs around a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refresh {
    Loc { player: u8, location: u32 },
    Single { player: u8, location: u32, seq: u32 },
}

fn both(location: u32) -> Vec<Refresh> {
    vec![Refresh::Loc { player: 0, location }, Refresh::Loc { player: 1, location }]
}

/// Queries EDOPro runs before sending the message (`BeforeParsing`).
pub fn before_refreshes(raw: &[u8]) -> Result<Vec<Refresh>, String> {
    let t = *raw.first().ok_or("empty core message")? as u32;
    let mut out = Vec::new();
    if t == C::MSG_SELECT_IDLECMD || t == C::MSG_SELECT_BATTLECMD {
        out.extend(both(C::LOCATION_MZONE));
        out.extend(both(C::LOCATION_SZONE));
        out.extend(both(C::LOCATION_HAND));
    } else if t == C::MSG_SELECT_CHAIN || t == C::MSG_NEW_TURN {
        out.extend(both(C::LOCATION_MZONE));
        out.extend(both(C::LOCATION_SZONE));
    } else if t == C::MSG_FLIPSUMMONING {
        let (con, location, seq, _pos) = loc(raw, 5)?;
        out.push(Refresh::Single { player: con, location: location as u32, seq });
    }
    Ok(out)
}

/// Queries EDOPro runs after sending the message (`AfterParsing`).
pub fn after_refreshes(raw: &[u8]) -> Result<Vec<Refresh>, String> {
    let t = *raw.first().ok_or("empty core message")? as u32;
    let mut out = Vec::new();
    if t == C::MSG_SHUFFLE_HAND || t == C::MSG_DRAW {
        out.push(Refresh::Loc { player: u8_at(raw, 1)?, location: C::LOCATION_HAND });
    } else if t == C::MSG_SHUFFLE_EXTRA {
        out.push(Refresh::Loc { player: u8_at(raw, 1)?, location: C::LOCATION_EXTRA });
    } else if t == C::MSG_SWAP_GRAVE_DECK {
        out.push(Refresh::Loc { player: u8_at(raw, 1)?, location: C::LOCATION_GRAVE });
    } else if t == C::MSG_REVERSE_DECK {
    } else if t == C::MSG_SHUFFLE_SET_CARD {
        let location = u8_at(raw, 2)? as u32;
        out.push(Refresh::Loc { player: 0, location });
        out.push(Refresh::Loc { player: 1, location });
    } else if t == C::MSG_NEW_TURN {
    } else if matches!(
        t,
        C::MSG_DAMAGE_STEP_START
            | C::MSG_DAMAGE_STEP_END
            | C::MSG_SUMMONED
            | C::MSG_SPSUMMONED
            | C::MSG_FLIPSUMMONED
            | C::MSG_NEW_PHASE
            | C::MSG_CHAINED
            | C::MSG_CHAIN_SOLVED
            | C::MSG_CHAIN_END
    ) {
        out.extend(both(C::LOCATION_MZONE));
        if t != C::MSG_DAMAGE_STEP_START && t != C::MSG_DAMAGE_STEP_END {
            out.extend(both(C::LOCATION_SZONE));
        }
        if matches!(t, C::MSG_NEW_PHASE | C::MSG_CHAINED | C::MSG_CHAIN_END) {
            out.extend(both(C::LOCATION_HAND));
        }
    } else if t == C::MSG_MOVE {
        let (pcon, ploc, _pseq, _ppos) = loc(raw, 5)?;
        let (ccon, cloc, cseq, _cpos) = loc(raw, 15)?;
        if cloc != 0 && cloc & 0x80 == 0 && (cloc != ploc || pcon != ccon) {
            out.push(Refresh::Single { player: ccon, location: cloc as u32, seq: cseq });
        }
    } else if t == C::MSG_POS_CHANGE {
        let (cc, cl, cs, pp, cp) = (
            u8_at(raw, 5)?,
            u8_at(raw, 6)?,
            u8_at(raw, 7)?,
            u8_at(raw, 8)?,
            u8_at(raw, 9)?,
        );
        if (pp & C::POS_FACEDOWN as u8) != 0 && (cp & C::POS_FACEUP as u8) != 0 {
            out.push(Refresh::Single { player: cc, location: cl as u32, seq: cs as u32 });
        }
    } else if t == C::MSG_SWAP {
        let (pcon, ploc, pseq, _p) = loc(raw, 5)?;
        let (ccon, cloc, cseq, _c) = loc(raw, 19)?;
        out.push(Refresh::Single { player: pcon, location: ploc as u32, seq: pseq });
        out.push(Refresh::Single { player: ccon, location: cloc as u32, seq: cseq });
    }
    Ok(out)
}

/// One board-refresh packet for `viewer`, queried now (None when it names an empty slot).
pub fn refresh_packet(env: &Env<'_>, refresh: Refresh, viewer: u8) -> Result<Option<Vec<u8>>, String> {
    match refresh {
        Refresh::Loc { player, location } => {
            let cards = query_location(env, player, location)?;
            Ok(Some(encode_update_data(player, location, &cards, viewer != player)))
        }
        Refresh::Single { player, location, seq } => {
            let cards = query_location(env, player, location)?;
            let Some(card) = cards.get(seq as usize) else {
                return Ok(None);
            };
            let Some(card) = card else {
                return Ok(None);
            };
            Ok(Some(encode_update_card(player, location, seq, Some(card), viewer != player)))
        }
    }
}

fn query_location(env: &Env<'_>, player: u8, location: u32) -> Result<Vec<Option<CardState>>, String> {
    let raw = env.query_location(player, location, query::CARD_FLAGS)?;
    query::parse_location(&raw, player as u32, location)
}

// --- MSG_UPDATE_DATA / MSG_UPDATE_CARD encoders (CoreUtils::Query::GenerateBuffer) ---

#[derive(Clone, Copy)]
enum Kind {
    U32,
    U64,
    I32,
    U8,
    Codes,
    Link,
}

/// `(flag, kind)`, ascending flag order (core_utils.cpp `GetFlagSize`).
const FIELDS: [(u32, Kind); 18] = [
    (C::QUERY_CODE, Kind::U32),
    (C::QUERY_POSITION, Kind::U32),
    (C::QUERY_TYPE, Kind::U32),
    (C::QUERY_LEVEL, Kind::U32),
    (C::QUERY_RANK, Kind::U32),
    (C::QUERY_ATTRIBUTE, Kind::U32),
    (C::QUERY_RACE, Kind::U64),
    (C::QUERY_ATTACK, Kind::I32),
    (C::QUERY_DEFENSE, Kind::I32),
    (C::QUERY_BASE_ATTACK, Kind::I32),
    (C::QUERY_BASE_DEFENSE, Kind::I32),
    (C::QUERY_OVERLAY_CARD, Kind::Codes),
    (C::QUERY_OWNER, Kind::U8),
    (C::QUERY_STATUS, Kind::U32),
    (C::QUERY_IS_PUBLIC, Kind::U8),
    (C::QUERY_LSCALE, Kind::U32),
    (C::QUERY_RSCALE, Kind::U32),
    (C::QUERY_LINK, Kind::Link),
];

fn is_private(flag: u32) -> bool {
    matches!(
        flag,
        C::QUERY_CODE
            | C::QUERY_TYPE
            | C::QUERY_LEVEL
            | C::QUERY_RANK
            | C::QUERY_ATTRIBUTE
            | C::QUERY_RACE
            | C::QUERY_ATTACK
            | C::QUERY_DEFENSE
            | C::QUERY_BASE_ATTACK
            | C::QUERY_BASE_DEFENSE
            | C::QUERY_STATUS
            | C::QUERY_LSCALE
            | C::QUERY_RSCALE
            | C::QUERY_LINK
    )
}

fn field_value(flag: u32, kind: Kind, card: &CardState) -> Vec<u8> {
    match kind {
        Kind::U8 => {
            let v = if flag == C::QUERY_OWNER { card.owner } else { u8::from(card.public) };
            vec![v]
        }
        Kind::U64 => card.race.to_le_bytes().to_vec(),
        Kind::I32 => {
            let v = match flag {
                C::QUERY_ATTACK => card.attack,
                C::QUERY_DEFENSE => card.defense,
                C::QUERY_BASE_ATTACK => card.base_attack,
                C::QUERY_BASE_DEFENSE => card.base_defense,
                _ => 0,
            };
            v.to_le_bytes().to_vec()
        }
        Kind::Codes => {
            let mut o = Vec::with_capacity(4 + 4 * card.overlay.len());
            o.extend_from_slice(&(card.overlay.len() as u32).to_le_bytes());
            for c in &card.overlay {
                o.extend_from_slice(&c.to_le_bytes());
            }
            o
        }
        Kind::Link => {
            let mut o = Vec::with_capacity(8);
            o.extend_from_slice(&card.link.to_le_bytes());
            o.extend_from_slice(&card.link_marker.to_le_bytes());
            o
        }
        Kind::U32 => {
            let v = match flag {
                C::QUERY_CODE => card.code,
                C::QUERY_POSITION => card.position,
                C::QUERY_TYPE => card.type_,
                C::QUERY_LEVEL => card.level,
                C::QUERY_RANK => card.rank,
                C::QUERY_ATTRIBUTE => card.attribute,
                C::QUERY_STATUS => card.status,
                C::QUERY_LSCALE => card.lscale,
                C::QUERY_RSCALE => card.rscale,
                _ => 0,
            };
            v.to_le_bytes().to_vec()
        }
    }
}

/// One `QueryLocation` slot (empty slot = u16 0; record ends with `QUERY_END`).
pub fn encode_record(card: Option<&CardState>, public_only: bool) -> Vec<u8> {
    let Some(card) = card else {
        return vec![0, 0];
    };
    let public = (card.position & C::POS_FACEUP) != 0 || card.public;
    let mut out = Vec::new();
    for (flag, kind) in FIELDS {
        if public_only && !public && is_private(flag) {
            continue;
        }
        let val = field_value(flag, kind, card);
        out.extend_from_slice(&((val.len() + 4) as u16).to_le_bytes());
        out.extend_from_slice(&flag.to_le_bytes());
        out.extend_from_slice(&val);
    }
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&C::QUERY_END.to_le_bytes());
    out
}

pub fn encode_update_data(player: u8, location: u32, cards: &[Option<CardState>], public_only: bool) -> Vec<u8> {
    let mut recs = Vec::new();
    for c in cards {
        recs.extend_from_slice(&encode_record(c.as_ref(), public_only));
    }
    let mut out = Vec::with_capacity(7 + recs.len());
    out.extend_from_slice(&[C::MSG_UPDATE_DATA as u8, player, location as u8]);
    out.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    out.extend_from_slice(&recs);
    out
}

pub fn encode_update_card(
    player: u8,
    location: u32,
    seq: u32,
    card: Option<&CardState>,
    public_only: bool,
) -> Vec<u8> {
    let mut out = vec![C::MSG_UPDATE_CARD as u8, player, location as u8, (seq & 0xFF) as u8];
    out.extend_from_slice(&encode_record(card, public_only));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_record_is_two_zero_bytes() {
        assert_eq!(encode_record(None, false), vec![0, 0]);
    }

    #[test]
    fn malformed_messages_are_errors_not_panics() {
        for n in 0..40 {
            let junk = vec![0xFFu8; n];
            for t in [
                C::MSG_HINT,
                C::MSG_SELECT_CARD,
                C::MSG_SELECT_TRIBUTE,
                C::MSG_SELECT_UNSELECT_CARD,
                C::MSG_CONFIRM_CARDS,
                C::MSG_MOVE,
                C::MSG_DRAW,
                C::MSG_POS_CHANGE,
                C::MSG_SWAP,
            ] {
                let mut m = junk.clone();
                if !m.is_empty() {
                    m[0] = t as u8;
                }
                let _ = filtered_message(&m, 0);
                let _ = before_refreshes(&m);
                let _ = after_refreshes(&m);
            }
        }
    }
}
