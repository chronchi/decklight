//! Read the board through `OCG_DuelQueryLocation` / `OCG_DuelQueryField` (port of
//! `src/decklight/engine/query.py`, behaviour for behaviour).

use std::fmt::Write as _;

use crate::constants as C;

pub const CARD_FLAGS: u32 = C::QUERY_CODE
    | C::QUERY_POSITION
    | C::QUERY_TYPE
    | C::QUERY_LEVEL
    | C::QUERY_RANK
    | C::QUERY_ATTRIBUTE
    | C::QUERY_RACE
    | C::QUERY_ATTACK
    | C::QUERY_DEFENSE
    | C::QUERY_BASE_ATTACK
    | C::QUERY_BASE_DEFENSE
    | C::QUERY_OVERLAY_CARD
    | C::QUERY_OWNER
    | C::QUERY_STATUS
    | C::QUERY_IS_PUBLIC
    | C::QUERY_LSCALE
    | C::QUERY_RSCALE
    | C::QUERY_LINK;

/// Locations queried, in the order the core lists them per player.
pub const QUERY_LOCATIONS: [u32; 7] = [
    C::LOCATION_DECK,
    C::LOCATION_HAND,
    C::LOCATION_MZONE,
    C::LOCATION_SZONE,
    C::LOCATION_GRAVE,
    C::LOCATION_REMOVED,
    C::LOCATION_EXTRA,
];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CardState {
    pub controller: u32,
    pub location: u32,
    pub sequence: u32,
    pub code: u32,
    pub position: u32,
    pub type_: u32,
    pub level: u32,
    pub rank: u32,
    pub attribute: u32,
    pub race: u64,
    pub attack: i32,
    pub defense: i32,
    pub base_attack: i32,
    pub base_defense: i32,
    pub owner: u8,
    pub status: u32,
    pub public: bool,
    pub lscale: u32,
    pub rscale: u32,
    pub link: u32,
    pub link_marker: u32,
    pub overlay: Vec<u32>,
    pub overlay_index: u32,
}

impl CardState {
    /// Canonical JSON, byte-identical to Python's `json.dumps(canon_card(card), separators=(",", ":"))`.
    pub fn to_canon_json(&self) -> String {
        let mut o = String::new();
        o.push_str("{\"controller\":");
        let _ = write!(o, "{}", self.controller);
        o.push_str(",\"location\":");
        let _ = write!(o, "{}", self.location);
        o.push_str(",\"sequence\":");
        let _ = write!(o, "{}", self.sequence);
        o.push_str(",\"code\":");
        let _ = write!(o, "{}", self.code);
        o.push_str(",\"position\":");
        let _ = write!(o, "{}", self.position);
        o.push_str(",\"type\":");
        let _ = write!(o, "{}", self.type_);
        o.push_str(",\"level\":");
        let _ = write!(o, "{}", self.level);
        o.push_str(",\"rank\":");
        let _ = write!(o, "{}", self.rank);
        o.push_str(",\"attribute\":");
        let _ = write!(o, "{}", self.attribute);
        o.push_str(",\"race\":");
        let _ = write!(o, "{}", self.race);
        o.push_str(",\"attack\":");
        let _ = write!(o, "{}", self.attack);
        o.push_str(",\"defense\":");
        let _ = write!(o, "{}", self.defense);
        o.push_str(",\"base_attack\":");
        let _ = write!(o, "{}", self.base_attack);
        o.push_str(",\"base_defense\":");
        let _ = write!(o, "{}", self.base_defense);
        o.push_str(",\"owner\":");
        let _ = write!(o, "{}", self.owner);
        o.push_str(",\"status\":");
        let _ = write!(o, "{}", self.status);
        o.push_str(",\"public\":");
        o.push_str(if self.public { "true" } else { "false" });
        o.push_str(",\"lscale\":");
        let _ = write!(o, "{}", self.lscale);
        o.push_str(",\"rscale\":");
        let _ = write!(o, "{}", self.rscale);
        o.push_str(",\"link\":");
        let _ = write!(o, "{}", self.link);
        o.push_str(",\"link_marker\":");
        let _ = write!(o, "{}", self.link_marker);
        o.push_str(",\"overlay\":[");
        for (i, v) in self.overlay.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            let _ = write!(o, "{v}");
        }
        o.push_str("],\"overlay_index\":");
        let _ = write!(o, "{}", self.overlay_index);
        o.push('}');
        o
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChainLink {
    pub code: u32,
    pub controller: u8,
    pub location: u8,
    pub sequence: u32,
    pub position: u32,
    pub triggering_controller: u8,
    pub triggering_location: u8,
    pub triggering_sequence: u32,
    pub description: u64,
}

impl ChainLink {
    /// Canonical JSON, byte-identical to Python's `json.dumps(canon_chain(link), separators=(",", ":"))`.
    pub fn to_canon_json(&self) -> String {
        format!(
            "{{\"code\":{},\"controller\":{},\"location\":{},\"sequence\":{},\"position\":{},\
\"triggering_controller\":{},\"triggering_location\":{},\"triggering_sequence\":{},\"description\":{}}}",
            self.code,
            self.controller,
            self.location,
            self.sequence,
            self.position,
            self.triggering_controller,
            self.triggering_location,
            self.triggering_sequence,
            self.description
        )
    }
}

fn need(buf: &[u8], pos: usize, n: usize) -> Result<(), String> {
    if pos + n > buf.len() {
        return Err(format!("query buffer truncated at {pos}: need {n}, have {}", buf.len().saturating_sub(pos)));
    }
    Ok(())
}

fn u16_at(buf: &[u8], pos: usize) -> Result<u16, String> {
    need(buf, pos, 2)?;
    Ok(u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()))
}

fn u32_at(buf: &[u8], pos: usize) -> Result<u32, String> {
    need(buf, pos, 4)?;
    Ok(u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()))
}

fn u64_at(buf: &[u8], pos: usize) -> Result<u64, String> {
    need(buf, pos, 8)?;
    Ok(u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()))
}

/// Fill `card` from one `get_infos` record starting at `pos`; returns the position after `QUERY_END`.
pub fn parse_card(buf: &[u8], mut pos: usize, card: &mut CardState) -> Result<usize, String> {
    loop {
        let size = u16_at(buf, pos)?;
        let flag = u32_at(buf, pos + 2)?;
        let body = pos + 6;
        pos += 2 + size as usize;
        if flag == C::QUERY_END {
            return Ok(pos);
        }
        if flag == C::QUERY_RACE {
            card.race = u64_at(buf, body)?;
        } else if flag == C::QUERY_OWNER || flag == C::QUERY_IS_PUBLIC || flag == C::QUERY_IS_HIDDEN {
            need(buf, body, 1)?;
            let v = buf[body];
            if flag == C::QUERY_OWNER {
                card.owner = v;
            } else if flag == C::QUERY_IS_PUBLIC {
                card.public = v != 0;
            }
        } else if flag == C::QUERY_LINK {
            card.link = u32_at(buf, body)?;
            card.link_marker = u32_at(buf, body + 4)?;
        } else if flag == C::QUERY_OVERLAY_CARD {
            let n = u32_at(buf, body)? as usize;
            let mut overlay = Vec::with_capacity(n.min(buf.len().saturating_sub(body + 4) / 4));
            for i in 0..n {
                overlay.push(u32_at(buf, body + 4 + 4 * i)?);
            }
            card.overlay = overlay;
        } else if size == 8 {
            let v = u32_at(buf, body)?;
            match flag {
                C::QUERY_CODE => card.code = v,
                C::QUERY_POSITION => card.position = v,
                C::QUERY_TYPE => card.type_ = v,
                C::QUERY_LEVEL => card.level = v,
                C::QUERY_RANK => card.rank = v,
                C::QUERY_ATTRIBUTE => card.attribute = v,
                C::QUERY_ATTACK => card.attack = v as i32,
                C::QUERY_DEFENSE => card.defense = v as i32,
                C::QUERY_BASE_ATTACK => card.base_attack = v as i32,
                C::QUERY_BASE_DEFENSE => card.base_defense = v as i32,
                C::QUERY_STATUS => card.status = v,
                C::QUERY_LSCALE => card.lscale = v,
                C::QUERY_RSCALE => card.rscale = v,
                _ => {}
            }
        }
    }
}

/// A QueryLocation buffer: u32 total size, then per slot either a record or an empty i16 0.
pub fn parse_location(buf: &[u8], controller: u32, location: u32) -> Result<Vec<Option<CardState>>, String> {
    let mut out = Vec::new();
    let mut pos = 4usize;
    let end = buf.len();
    let mut seq = 0u32;
    while pos < end {
        let size = u16_at(buf, pos)?;
        if size == 0 {
            out.push(None);
            pos += 2;
        } else {
            let mut card = CardState { controller, location, sequence: seq, ..Default::default() };
            pos = parse_card(buf, pos, &mut card)?;
            out.push(Some(card));
        }
        seq += 1;
    }
    Ok(out)
}

/// The chain section at the end of a QueryField buffer.
pub fn parse_chain(buf: &[u8]) -> Result<Vec<ChainLink>, String> {
    let mut pos = 4usize; // duel options
    for _ in 0..2 {
        pos += 4; // lp
        for _ in 0..(7 + 8) {
            need(buf, pos, 1)?;
            let occupied = buf[pos];
            pos += 1 + if occupied != 0 { 5 } else { 0 };
        }
        pos += 6 * 4;
    }
    let n = u32_at(buf, pos)? as usize;
    pos += 4;
    let mut chain = Vec::with_capacity(n.min(buf.len().saturating_sub(pos) / 31));
    for _ in 0..n {
        let code = u32_at(buf, pos)?;
        need(buf, pos + 4, 2)?;
        let controller = buf[pos + 4];
        let location = buf[pos + 5];
        let sequence = u32_at(buf, pos + 6)?;
        let position = u32_at(buf, pos + 10)?;
        need(buf, pos + 14, 2)?;
        let triggering_controller = buf[pos + 14];
        let triggering_location = buf[pos + 15];
        let triggering_sequence = u32_at(buf, pos + 16)?;
        let description = u64_at(buf, pos + 20)?;
        pos += 4 + 1 + 1 + 4 + 4 + 1 + 1 + 4 + 8;
        chain.push(ChainLink {
            code,
            controller,
            location,
            sequence,
            position,
            triggering_controller,
            triggering_location,
            triggering_sequence,
            description,
        });
    }
    Ok(chain)
}

/// Both players' LP from a QueryField buffer.
pub fn lps(buf: &[u8]) -> Result<(u32, u32), String> {
    let lp0 = u32_at(buf, 4)?;
    let mut pos = 8usize;
    for _ in 0..(7 + 8) {
        need(buf, pos, 1)?;
        pos += 1 + if buf[pos] != 0 { 5 } else { 0 };
    }
    pos += 6 * 4;
    Ok((lp0, u32_at(buf, pos)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16b(v: u16) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn u32b(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn u64b(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn record(flag: u32, payload: &[u8]) -> Vec<u8> {
        [u16b((4 + payload.len()) as u16), u32b(flag), payload.to_vec()].concat()
    }
    fn end() -> Vec<u8> {
        [u16b(4), u32b(C::QUERY_END)].concat()
    }

    #[test]
    fn card_signed_values_overlay_and_race() {
        let mut buf = record(C::QUERY_CODE, &u32b(1234));
        buf.extend(record(C::QUERY_ATTACK, &u32b(0xFFFF_FFFF)));
        buf.extend(record(C::QUERY_DEFENSE, &u32b(0x8000_0000)));
        buf.extend(record(C::QUERY_RACE, &u64b(0x1_0000_0021)));
        buf.extend(record(C::QUERY_IS_PUBLIC, &[1]));
        buf.extend(record(C::QUERY_OVERLAY_CARD, &[u32b(2), u32b(10), u32b(11)].concat()));
        buf.extend(end());
        let mut c = CardState::default();
        let p = parse_card(&buf, 0, &mut c).unwrap();
        assert_eq!(p, buf.len());
        assert_eq!((c.code, c.attack, c.defense, c.race, c.public), (1234, -1, i32::MIN, 0x1_0000_0021, true));
        assert_eq!(c.overlay, vec![10, 11]);
        assert!(c.to_canon_json().starts_with("{\"controller\":0,\"location\":0,\"sequence\":0,\"code\":1234,"));
    }

    #[test]
    fn location_advances_seq_for_empty_slots() {
        let mut buf = u32b(0);
        buf.extend(u16b(0)); // empty slot 0
        let mut card = record(C::QUERY_CODE, &u32b(99));
        card.extend(end());
        buf.extend(card);
        buf.extend(u16b(0)); // empty slot 2
        let out = parse_location(&buf, 3, C::LOCATION_MZONE).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out[0].is_none());
        assert_eq!(out[1].as_ref().unwrap().sequence, 1);
        assert_eq!(out[1].as_ref().unwrap().controller, 3);
        assert_eq!(out[1].as_ref().unwrap().code, 99);
        assert!(out[2].is_none());
    }

    fn field_buffer(lps: (u32, u32), chain: &[(u32, u8, u8, u32, u32, u8, u8, u32, u64)]) -> Vec<u8> {
        let mut b = u32b(0);
        for lp in [lps.0, lps.1] {
            b.extend(u32b(lp));
            b.extend([0u8; 15]);
            b.extend([0u8; 24]);
        }
        b.extend(u32b(chain.len() as u32));
        for &(code, con, loc, seq, pos, tcon, tloc, tseq, desc) in chain {
            b.extend(u32b(code));
            b.extend([con, loc]);
            b.extend(u32b(seq));
            b.extend(u32b(pos));
            b.extend([tcon, tloc]);
            b.extend(u32b(tseq));
            b.extend(u64b(desc));
        }
        b
    }

    #[test]
    fn field_lps_and_chain_layout() {
        let buf = field_buffer((7000, 6400), &[(42, 1, 4, 3, 5, 0, 8, 1, 77)]);
        assert_eq!(lps(&buf).unwrap(), (7000, 6400));
        let chain = parse_chain(&buf).unwrap();
        assert_eq!(
            chain[0].to_canon_json(),
            "{\"code\":42,\"controller\":1,\"location\":4,\"sequence\":3,\"position\":5,\
\"triggering_controller\":0,\"triggering_location\":8,\"triggering_sequence\":1,\"description\":77}"
        );
    }

    #[test]
    fn field_skips_occupied_zones() {
        let mut b = u32b(0);
        for lp in [1234u32, 4321] {
            b.extend(u32b(lp));
            b.extend([1u8, 0, 0, 0, 0, 0]); // one occupied slot: byte + 5
            b.extend([0u8; 14]);
            b.extend([0u8; 24]);
        }
        b.extend(u32b(0));
        assert_eq!(lps(&b).unwrap(), (1234, 4321));
        assert!(parse_chain(&b).unwrap().is_empty());
    }
}
