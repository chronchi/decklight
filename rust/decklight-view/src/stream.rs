//! The client's seat rebuilt from the filtered EDOPro message stream (port of
//! `src/decklight/play/stream.py`, behaviour for behaviour).
//!
//! `StreamState` maintains the board from the exact packet stream a host sends one seat (START,
//! filtered core messages, before/after `MSG_UPDATE_DATA` / `MSG_UPDATE_CARD` refreshes) plus our
//! deck list and card DB. `StreamSource` adds the pending `ygo_decision` built from the received
//! select message (as `decklight_edopro::client` does) and implements the crate's `ViewSource`.
//!
//! Seats are absolute engine seats everywhere: START's seat byte is the viewer's absolute seat and
//! every controller/location reference is untranslated.
//!
//! Documented stream limits (same as Python): the opponent's hand order after a shuffle (their
//! `MSG_SHUFFLE_HAND` codes are zeroed) and identities the stream never carried are never known.
//! Gate 3 also measures three consequences of the packet set: `GRAVE` is only refreshed on
//! `MSG_SWAP_GRAVE_DECK` (so its order/derived stats can lag the engine), the deck count is kept
//! from `START`/`DRAW`/`MOVE` (it can differ by a card the host moved without a naming message),
//! and a chain link's live position/sequence is whatever the last naming message said until the
//! next refresh. These only affect displayed state/history, never the offered options.

use std::collections::{BTreeMap, BTreeSet};

use decklight_cards::CardData;
use decklight_core::constants as C;
use decklight_core::messages as M;
use decklight_core::query::{self as Q, CardState, ChainLink};
use ygoenv_sys::{Decision, DecisionState, Engine};

use crate::ViewSource;

const SLOTS: [(u32, usize); 2] = [(C::LOCATION_MZONE, 7), (C::LOCATION_SZONE, 8)];
const LISTS: [u32; 4] = [C::LOCATION_HAND, C::LOCATION_GRAVE, C::LOCATION_REMOVED, C::LOCATION_EXTRA];

/// Status bits a client can observe (`obs.STATUS_OBS_MASK`).
pub const STATUS_OBS_MASK: u32 = C::STATUS_DISABLED
    | C::STATUS_SET_TURN
    | C::STATUS_SUMMON_TURN
    | C::STATUS_FLIP_SUMMON_TURN
    | C::STATUS_SPSUMMON_TURN
    | C::STATUS_FORM_CHANGED
    | C::STATUS_ATTACK_CANCELED
    | C::STATUS_CHAINING
    | C::STATUS_FORBIDDEN;

fn rd_u8(b: &[u8], p: usize) -> Result<u8, String> {
    b.get(p).copied().ok_or_else(|| format!("truncated at {p}: need 1, have {}", b.len().saturating_sub(p)))
}

fn rd_u32(b: &[u8], p: usize) -> Result<u32, String> {
    let s = b
        .get(p..p + 4)
        .ok_or_else(|| format!("truncated at {p}: need 4, have {}", b.len().saturating_sub(p)))?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn rd_u64(b: &[u8], p: usize) -> Result<u64, String> {
    let s = b
        .get(p..p + 8)
        .ok_or_else(|| format!("truncated at {p}: need 8, have {}", b.len().saturating_sub(p)))?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}

/// `(controller, location, sequence, position)` at `off`.
fn loc_info(b: &[u8], off: usize) -> Result<(u8, u8, u32, u32), String> {
    Ok((rd_u8(b, off)?, rd_u8(b, off + 1)?, rd_u32(b, off + 2)?, rd_u32(b, off + 6)?))
}

/// The last-writer-wins card DB layer `deck_card_state` reads.
fn deck_card_state(code: u32, cards: &BTreeMap<u32, CardData>, controller: u32, sequence: i64) -> CardState {
    let d = match cards.get(&code) {
        Some(d) => d,
        None => {
            return CardState {
                controller,
                location: C::LOCATION_DECK,
                sequence: sequence.max(0) as u32,
                code,
                ..Default::default()
            };
        }
    };
    let monster = d.type_ & C::TYPE_MONSTER != 0;
    let lv = d.level & 0xFFFF_FFFF;
    let mut out = CardState {
        controller,
        location: C::LOCATION_DECK,
        sequence: sequence.max(0) as u32,
        code,
        position: 0,
        type_: d.type_,
        level: if (d.type_ & (C::TYPE_XYZ | C::TYPE_LINK)) != 0 || !monster { 0 } else { lv },
        rank: if d.type_ & C::TYPE_XYZ != 0 { lv } else { 0 },
        attribute: if monster { d.attribute } else { 0 },
        race: if monster { d.race } else { 0 },
        attack: if monster { d.attack } else { 0 },
        defense: if monster { d.defense } else { 0 },
        base_attack: if monster { d.attack } else { 0 },
        base_defense: if monster { d.defense } else { 0 },
        owner: controller as u8,
        status: 0,
        public: false,
        lscale: d.lscale,
        rscale: d.rscale,
        link: if d.type_ & C::TYPE_LINK != 0 { lv } else { 0 },
        link_marker: d.link_marker,
        overlay: Vec::new(),
        overlay_index: 0,
    };
    out.overlay.clear();
    out
}

/// `obs.loc_id`.
pub fn loc_id(location: u32) -> i32 {
    if location & C::LOCATION_OVERLAY != 0 {
        return 8;
    }
    match location {
        C::LOCATION_DECK => 1,
        C::LOCATION_HAND => 2,
        C::LOCATION_MZONE => 3,
        C::LOCATION_SZONE => 4,
        C::LOCATION_GRAVE => 5,
        C::LOCATION_REMOVED => 6,
        C::LOCATION_EXTRA => 7,
        _ => 0,
    }
}

// ------------------------------------------------------------------ hand memory

fn source_public(from_loc: u32, from_pos: u32) -> bool {
    if from_loc & C::LOCATION_OVERLAY != 0 {
        return true;
    }
    if from_loc == C::LOCATION_GRAVE {
        return true;
    }
    if from_loc == C::LOCATION_REMOVED || from_loc == C::LOCATION_EXTRA {
        return from_pos & C::POS_FACEUP != 0;
    }
    if from_loc == C::LOCATION_MZONE || from_loc == C::LOCATION_SZONE {
        return from_pos & C::POS_FACEUP != 0;
    }
    false
}

/// Card-knowledge memory (`engine.knowledge.HandMemory`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HandMemory {
    pub known: [Vec<bool>; 2],
    pub codes: [Vec<u32>; 2],
}

impl HandMemory {
    /// Fold one raw core message in; `false` when a shuffle-length mismatch rebuilt a list.
    pub fn note(&mut self, raw: &[u8]) -> bool {
        let Some(&t) = raw.first() else { return true };
        match t as u32 {
            C::MSG_MOVE => {
                self.mv(raw);
                true
            }
            C::MSG_DRAW => {
                self.draw(raw);
                true
            }
            C::MSG_CONFIRM_CARDS => {
                self.confirm(raw);
                true
            }
            C::MSG_SHUFFLE_HAND => self.shuffle(raw),
            _ => true,
        }
    }

    /// Resync guard: rebuild any list whose length differs from the queried hand size.
    pub fn guard(&mut self, sizes: [usize; 2]) -> bool {
        let mut ok = true;
        for p in 0..2 {
            if self.known[p].len() != sizes[p] {
                self.known[p] = vec![false; sizes[p]];
                self.codes[p] = vec![0; sizes[p]];
                ok = false;
            }
        }
        ok
    }

    fn mv(&mut self, raw: &[u8]) {
        if raw.len() != 29 {
            return;
        }
        let (Ok(code), Ok(from), Ok(to)) = (rd_u32(raw, 1), loc_info(raw, 5), loc_info(raw, 15)) else {
            return;
        };
        let (fc, fl, fs, fp) = from;
        let (tc, tl, ts, tp) = to;
        let from_hand = fl == C::LOCATION_HAND as u8 && fc < 2;
        let to_hand = tl == C::LOCATION_HAND as u8 && tc < 2;
        if !from_hand && !to_hand {
            return;
        }
        let mut old = false;
        if from_hand && (fs as usize) < self.known[fc as usize].len() {
            old = self.known[fc as usize][fs as usize];
            self.known[fc as usize].remove(fs as usize);
            self.codes[fc as usize].remove(fs as usize);
        }
        if to_hand {
            let known = if from_hand && fc == tc {
                old
            } else if from_hand {
                true
            } else {
                tp & C::POS_FACEUP != 0 || source_public(fl as u32, fp)
            };
            let i = (ts as usize).min(self.known[tc as usize].len());
            self.known[tc as usize].insert(i, known);
            self.codes[tc as usize].insert(i, code);
        }
    }

    fn draw(&mut self, raw: &[u8]) {
        if raw.len() < 6 {
            return;
        }
        let (Ok(player), Ok(drawn)) = (rd_u8(raw, 1), rd_u32(raw, 2)) else { return };
        if player > 1 || raw.len() != 6 + 8 * drawn as usize {
            return;
        }
        for i in 0..drawn as usize {
            let (Ok(code), Ok(pos)) = (rd_u32(raw, 6 + 8 * i), rd_u32(raw, 6 + 8 * i + 4)) else {
                return;
            };
            self.known[player as usize].push(pos & C::POS_FACEUP != 0);
            self.codes[player as usize].push(code);
        }
    }

    fn confirm(&mut self, raw: &[u8]) {
        if raw.len() < 6 {
            return;
        }
        let (Ok(viewer), Ok(count)) = (rd_u8(raw, 1), rd_u32(raw, 2)) else { return };
        if viewer > 1 || raw.len() != 6 + 10 * count as usize {
            return;
        }
        for i in 0..count as usize {
            let off = 6 + 10 * i;
            let (Ok(con), Ok(loc), Ok(seq)) = (rd_u8(raw, off + 4), rd_u8(raw, off + 5), rd_u32(raw, off + 6))
            else {
                return;
            };
            if loc == C::LOCATION_HAND as u8
                && con < 2
                && viewer != con
                && (seq as usize) < self.known[con as usize].len()
            {
                self.known[con as usize][seq as usize] = true;
            }
        }
    }

    fn shuffle(&mut self, raw: &[u8]) -> bool {
        if raw.len() < 6 {
            return true;
        }
        let (Ok(player), Ok(count)) = (rd_u8(raw, 1), rd_u32(raw, 2)) else { return true };
        if player > 1 || !(raw.len() == 6 + 4 * count as usize || raw.len() == 6 + 5 * count as usize) {
            return true;
        }
        let mut codes = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let Ok(c) = rd_u32(raw, 6 + 4 * i) else { return true };
            codes.push(c & 0x7FFF_FFFF);
        }
        let known = &mut self.known[player as usize];
        let cur = &mut self.codes[player as usize];
        if codes.len() != known.len() || cur.len() != known.len() {
            self.known[player as usize] = vec![false; codes.len()];
            self.codes[player as usize] = codes;
            return false;
        }
        let mut keep: BTreeMap<u32, usize> = BTreeMap::new();
        for (k, code) in known.iter().zip(cur.iter()) {
            if *k {
                *keep.entry(*code).or_insert(0) += 1;
            }
        }
        let out: Vec<bool> = codes
            .iter()
            .map(|c| {
                let n = keep.get(c).copied().unwrap_or(0);
                if n > 0 {
                    keep.insert(*c, n - 1);
                    true
                } else {
                    false
                }
            })
            .collect();
        self.known[player as usize] = out;
        self.codes[player as usize] = codes;
        true
    }
}

// ------------------------------------------------------------------ state

/// A history entry rebuilt from a public message (`StreamState`'s row is the 10-int option row).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryRow {
    pub row: [i32; 10],
    pub actor: i32,
    pub turn: i32,
    pub phase: i32,
    pub public: bool,
}

impl HistoryRow {
    /// The 14-int env `history_log` row.
    pub fn to_log_row(&self) -> [i32; 14] {
        let mut out = [0i32; 14];
        out[..10].copy_from_slice(&self.row);
        out[10] = self.actor;
        out[11] = self.turn;
        out[12] = self.phase;
        out[13] = i32::from(self.public);
        out
    }
}

/// Fill a card known only by code with its card-DB stats (identity fields untouched).
fn fill_stats(cards_db: &BTreeMap<u32, CardData>, card: &mut CardState, force: bool) {
    if card.code == 0 || (card.type_ != 0 && !force) {
        return;
    }
    let d = deck_card_state(card.code, cards_db, card.controller, card.sequence as i64);
    card.type_ = d.type_;
    card.level = d.level;
    card.rank = d.rank;
    card.attribute = d.attribute;
    card.race = d.race;
    card.attack = d.attack;
    card.defense = d.defense;
    card.base_attack = d.base_attack;
    card.base_defense = d.base_defense;
    card.lscale = d.lscale;
    card.rscale = d.rscale;
    card.link = d.link;
    card.link_marker = d.link_marker;
}

fn count_sub(a: &BTreeMap<u32, i64>, b: &BTreeMap<u32, i64>) -> BTreeMap<u32, i64> {
    let mut out = BTreeMap::new();
    for (&code, &n) in a {
        let m = n - b.get(&code).copied().unwrap_or(0);
        if m > 0 {
            out.insert(code, m);
        }
    }
    out
}

/// Insertion-ordered counter (Python's `Counter` iteration order; the loose deck statuses are
/// assigned in that order, so a BTreeMap would swap equal-code cards between sequences).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderedCounter {
    items: Vec<((u32, u32), i64)>,
}

impl OrderedCounter {
    fn add(&mut self, key: (u32, u32), n: i64) {
        if let Some(i) = self.items.iter().position(|(k, _)| *k == key) {
            self.items[i].1 += n;
            if self.items[i].1 <= 0 {
                self.items.remove(i);
            }
        } else if n > 0 {
            self.items.push((key, n));
        }
    }

    fn get_mut(&mut self, key: (u32, u32)) -> Option<&mut i64> {
        self.items.iter_mut().find(|(k, _)| *k == key).map(|(_, n)| n)
    }

    fn remove(&mut self, key: (u32, u32)) -> Option<i64> {
        let i = self.items.iter().position(|(k, _)| *k == key)?;
        Some(self.items.remove(i).1)
    }

    fn keys(&self) -> Vec<(u32, u32)> {
        self.items.iter().map(|(k, _)| *k).collect()
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

/// A `BoardSource` maintained from the filtered message stream (absolute seats).
#[derive(Clone, Debug)]
pub struct StreamState<'a> {
    pub seat: u32,
    pub cards_db: &'a BTreeMap<u32, CardData>,
    pub deck_list: BTreeMap<u32, i64>,
    pub extra_list: Vec<u32>,
    pub slots: BTreeMap<(u32, u32), Vec<Option<CardState>>>,
    pub lists: BTreeMap<(u32, u32), Vec<CardState>>,
    pub deck_count: [i32; 2],
    pub deck_seqs: BTreeMap<i64, u32>,
    pub deck_stat: BTreeMap<i64, u32>,
    pub deck_loose: OrderedCounter,
    pub turn: i32,
    pub phase: i32,
    pub first_player: Option<i32>,
    pub turn_player: i32,
    pub lp: [i64; 2],
    pub chain: Vec<ChainLink>,
    pub memory: HandMemory,
}

impl<'a> StreamState<'a> {
    pub fn new(deck_main: &[u32], deck_extra: &[u32], cards: &'a BTreeMap<u32, CardData>, seat: u32) -> Self {
        let mut deck_list: BTreeMap<u32, i64> = BTreeMap::new();
        for &c in deck_main {
            *deck_list.entry(c).or_insert(0) += 1;
        }
        let mut slots = BTreeMap::new();
        for p in 0..2u32 {
            for (loc, n) in SLOTS {
                slots.insert((p, loc), vec![None; n]);
            }
        }
        let mut lists = BTreeMap::new();
        for p in 0..2u32 {
            for loc in LISTS {
                lists.insert((p, loc), Vec::new());
            }
        }
        StreamState {
            seat,
            cards_db: cards,
            deck_list,
            extra_list: deck_extra.to_vec(),
            slots,
            lists,
            deck_count: [0, 0],
            deck_seqs: BTreeMap::new(),
            deck_stat: BTreeMap::new(),
            deck_loose: OrderedCounter::default(),
            turn: 0,
            phase: 0,
            first_player: None,
            turn_player: 0,
            lp: [8000, 8000],
            chain: Vec::new(),
            memory: HandMemory::default(),
        }
    }

    // --- BoardSource surface ---

    pub fn query_location(&self, player: u32, location: u32) -> Vec<Option<CardState>> {
        if let Some(cards) = self.slots.get(&(player, location)) {
            return cards.clone();
        }
        if let Some(cards) = self.lists.get(&(player, location)) {
            return cards.iter().cloned().map(Some).collect();
        }
        if location == C::LOCATION_DECK {
            if player == self.seat {
                return self.own_deck().into_iter().map(Some).collect();
            }
            return (0..self.deck_count[player as usize])
                .map(|i| {
                    Some(CardState { controller: player, location, sequence: i as u32, ..Default::default() })
                })
                .collect();
        }
        Vec::new()
    }

    pub fn field_state(&self) -> ((i64, i64), Vec<ChainLink>) {
        ((self.lp[0], self.lp[1]), self.chain.clone())
    }

    fn own_deck(&self) -> Vec<CardState> {
        let mut elsewhere: BTreeMap<u32, i64> = BTreeMap::new();
        for (&(p, _), cards) in &self.slots {
            if p == self.seat {
                for c in cards.iter().flatten() {
                    if c.code != 0 {
                        *elsewhere.entry(c.code).or_insert(0) += 1;
                        for o in &c.overlay {
                            *elsewhere.entry(*o).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        for (&(p, _), cards) in &self.lists {
            if p == self.seat {
                for c in cards {
                    if c.code != 0 {
                        *elsewhere.entry(c.code).or_insert(0) += 1;
                        for o in &c.overlay {
                            *elsewhere.entry(*o).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        let mut unknown_own: i64 = 0; // our cards held by the opponent whose code we never saw
        for (&(p, _), cards) in &self.lists {
            if p != self.seat {
                for c in cards {
                    if c.owner as u32 == self.seat {
                        if c.code != 0 {
                            *elsewhere.entry(c.code).or_insert(0) += 1;
                        } else {
                            unknown_own += 1;
                        }
                    }
                }
            }
        }
        for (&(p, _), cards) in &self.slots {
            if p != self.seat {
                for c in cards.iter().flatten() {
                    if c.owner as u32 == self.seat {
                        if c.code != 0 {
                            *elsewhere.entry(c.code).or_insert(0) += 1;
                        } else {
                            unknown_own += 1;
                        }
                    }
                }
            }
        }
        let mut rest = count_sub(&self.deck_list, &elsewhere);
        let size: i64 = (rest.values().sum::<i64>() - unknown_own).max(0);
        let mut known: BTreeMap<i64, u32> = BTreeMap::new();
        let mut stat: BTreeMap<i64, u32> = BTreeMap::new();
        for (&seq, &code) in &self.deck_seqs {
            if seq < size && rest.get(&code).copied().unwrap_or(0) > 0 {
                known.insert(seq, code);
                *rest.get_mut(&code).unwrap() -= 1;
                if let Some(&s) = self.deck_stat.get(&seq) {
                    stat.insert(seq, s);
                }
            }
        }
        let free: Vec<i64> = (0..size).filter(|s| !known.contains_key(s)).collect();
        let mut fill: Vec<u32> = Vec::new();
        for (&code, &n) in &rest {
            for _ in 0..n {
                fill.push(code);
            }
        }
        for (seq, code) in free.into_iter().zip(fill) {
            known.insert(seq, code);
        }
        let mut loose = self.deck_loose.clone();
        let mut ordered: Vec<(i64, u32)> = known.into_iter().collect();
        ordered.sort_by_key(|(seq, code)| (*code, *seq));
        let mut out = Vec::new();
        for (seq, code) in ordered {
            let mut c = deck_card_state(code, self.cards_db, self.seat, seq);
            if let Some(&s) = stat.get(&seq) {
                c.status = s;
            } else {
                for key in loose.keys() {
                    if key.0 != code {
                        continue;
                    }
                    if let Some(n) = loose.get_mut(key) {
                        if *n > 0 {
                            c.status = key.1;
                            *n -= 1;
                            if *n <= 0 {
                                loose.remove(key);
                            }
                            break;
                        }
                    }
                }
            }
            out.push(c);
        }
        out
    }

    pub fn learn_deck_seqs(&mut self, refs: &[M::CardRef]) {
        for r in refs {
            if r.controller as u32 == self.seat && r.location == C::LOCATION_DECK && r.code != 0 {
                self.deck_seqs.insert(r.sequence as i64, r.code);
            }
        }
    }

    // --- packet feed ---

    /// Fold one STOC_GAME_MSG payload in (SELECTs for our seat carry no state here).
    pub fn feed(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.is_empty() {
            return Ok(());
        }
        self.memory.note(raw);
        let t = raw[0] as u32;
        match t {
            C::MSG_START => self.start(raw),
            C::MSG_UPDATE_DATA => self.update_data(raw),
            C::MSG_UPDATE_CARD => self.update_card(raw),
            C::MSG_MOVE => self.mv(raw),
            C::MSG_DRAW => self.draw(raw),
            C::MSG_NEW_TURN => {
                self.turn += 1;
                self.turn_player = (rd_u8(raw, 1)? & 1) as i32;
                if self.first_player.is_none() {
                    self.first_player = Some(self.turn_player);
                }
                self.deck_stat.clear();
                self.deck_loose.clear();
                Ok(())
            }
            C::MSG_NEW_PHASE => {
                let b = raw.get(1..3).ok_or("MSG_NEW_PHASE truncated")?;
                self.phase = u16::from_le_bytes(b.try_into().unwrap()) as i32;
                Ok(())
            }
            C::MSG_DAMAGE | C::MSG_PAY_LPCOST => {
                let player = rd_u8(raw, 1)? as usize;
                let amount = rd_u32(raw, 2)? as i64;
                if player > 1 {
                    return Err(format!("LP message for player {player}"));
                }
                self.lp[player] -= amount;
                Ok(())
            }
            C::MSG_RECOVER => {
                let player = rd_u8(raw, 1)? as usize;
                let amount = rd_u32(raw, 2)? as i64;
                if player > 1 {
                    return Err(format!("LP message for player {player}"));
                }
                self.lp[player] += amount;
                Ok(())
            }
            C::MSG_LPUPDATE => {
                let player = rd_u8(raw, 1)? as usize;
                let lp = rd_u32(raw, 2)? as i64;
                if player > 1 {
                    return Err(format!("LP message for player {player}"));
                }
                self.lp[player] = lp;
                Ok(())
            }
            C::MSG_CHAINING => {
                let link = self.chain_link(raw)?;
                let (con, loc, seq) = (link.controller as u32, link.location as u32, link.sequence);
                self.chain.push(link.clone());
                self.set_status(con, loc, seq, C::STATUS_CHAINING, true);
                let card = self.entry_at(con, loc, seq).cloned();
                if let Some(c) = card {
                    if loc != C::LOCATION_HAND || c.code == link.code {
                        self.set_public(con, loc, seq, true);
                    }
                }
                Ok(())
            }
            C::MSG_CHAIN_SOLVING => {
                for cards in self.slots.values_mut() {
                    for c in cards.iter_mut().flatten() {
                        c.status &= !C::STATUS_CHAINING;
                    }
                }
                for cards in self.lists.values_mut() {
                    for c in cards {
                        c.status &= !C::STATUS_CHAINING;
                    }
                }
                for s in self.deck_stat.values_mut() {
                    *s &= !C::STATUS_CHAINING;
                }
                let mut old = std::mem::take(&mut self.deck_loose);
                for key in old.keys() {
                    let n = old.remove(key).unwrap_or(0);
                    let (code, status) = key;
                    if status & C::STATUS_CHAINING != 0 {
                        let rest = status & !C::STATUS_CHAINING;
                        if rest != 0 {
                            self.deck_loose.add((code, rest), n);
                        }
                    } else {
                        self.deck_loose.add((code, status), n);
                    }
                }
                // The resolving handler may be a monster placed as a Continuous Trap
                // (EFFECT_CHANGE_TYPE): the core reverts it when the activation resolves,
                // with no message. The engine view already shows its card-DB stats.
                if let Some(ch) = self.chain.last().cloned() {
                    self.revert_trap_monster(ch.controller as u32, ch.location as u32, ch.sequence);
                }
                Ok(())
            }
            C::MSG_CHAIN_SOLVED => {
                if let Some(solved) = self.chain.pop() {
                    let (con, loc, seq) = (solved.controller as u32, solved.location as u32, solved.sequence);
                    self.set_public(con, loc, seq, false);
                    self.revert_trap_monster(con, loc, seq);
                }
                Ok(())
            }
            C::MSG_CHAIN_END => {
                for ch in std::mem::take(&mut self.chain) {
                    self.set_public(ch.controller as u32, ch.location as u32, ch.sequence, false);
                }
                Ok(())
            }
            C::MSG_POS_CHANGE => self.pos_change(raw),
            C::MSG_SWAP => self.swap(raw),
            C::MSG_REMOVE_CARDS => self.remove_cards(raw),
            C::MSG_SWAP_GRAVE_DECK => {
                let player = rd_u8(raw, 1)? as usize;
                if player > 1 {
                    return Err(format!("MSG_SWAP_GRAVE_DECK for player {player}"));
                }
                let grave = self.lists.get(&(player as u32, C::LOCATION_GRAVE)).cloned().unwrap_or_default();
                self.deck_count[player] = grave.len() as i32;
                if player as u32 == self.seat {
                    for c in &grave {
                        if c.code != 0 && c.status & STATUS_OBS_MASK != 0 {
                            self.deck_loose.add((c.code, c.status & STATUS_OBS_MASK), 1);
                        }
                    }
                    self.deck_seqs.clear();
                    self.deck_stat.clear();
                }
                self.lists.insert((player as u32, C::LOCATION_GRAVE), Vec::new());
                Ok(())
            }
            C::MSG_SHUFFLE_DECK => {
                let player = rd_u8(raw, 1)?;
                if player as u32 == self.seat {
                    self.demote_deck();
                }
                Ok(())
            }
            C::MSG_REVERSE_DECK => {
                self.demote_deck();
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn start(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.len() < 18 {
            return Err(format!("MSG_START truncated: {} bytes", raw.len()));
        }
        let seat = raw[1];
        let lp0 = rd_u32(raw, 2)? as i64;
        let lp1 = rd_u32(raw, 6)? as i64;
        let n0 = u16::from_le_bytes(raw[10..12].try_into().unwrap()) as i32;
        let n1 = u16::from_le_bytes(raw[14..16].try_into().unwrap()) as i32;
        if seat as u32 != self.seat {
            return Err(format!("START for seat {seat}, viewer {}", self.seat));
        }
        self.lp = [lp0, lp1];
        self.deck_count = [n0, n1];
        Ok(())
    }

    fn update_data(&mut self, raw: &[u8]) -> Result<(), String> {
        let player = rd_u8(raw, 1)? as u32;
        let loc = rd_u8(raw, 2)? as u32;
        let cards = Q::parse_location(&raw[3.min(raw.len())..], player, loc)?;
        if let Some(slot) = self.slots.get_mut(&(player, loc)) {
            *slot = cards;
        } else if self.lists.contains_key(&(player, loc)) {
            self.lists.insert((player, loc), cards.into_iter().flatten().collect());
            self.resync_chain_list(player, loc);
        }
        Ok(())
    }

    /// A refreshed list is authoritative (a shuffle reorders it): chain handlers are
    /// live card pointers, so move links with that code to their new index.
    fn resync_chain_list(&mut self, player: u32, loc: u32) {
        let cards = self.lists.get(&(player, loc)).cloned().unwrap_or_default();
        let mut used = BTreeSet::new();
        for i in 0..self.chain.len() {
            let (code, con, l) = {
                let ch = &self.chain[i];
                (ch.code, ch.controller as u32, ch.location as u32)
            };
            if (con, l) != (player, loc) || code == 0 {
                continue;
            }
            let cur = self.chain[i].sequence as usize;
            if cur < cards.len() && !used.contains(&cur) && cards[cur].code == code {
                // Duplicate codes: keep the link's live index when it still fits.
                used.insert(cur);
                let mut ch = self.chain[i].clone();
                ch.position = cards[cur].position;
                self.chain[i] = ch;
                continue;
            }
            for (j, c) in cards.iter().enumerate() {
                if used.contains(&j) || c.code != code {
                    continue;
                }
                used.insert(j);
                let mut ch = self.chain[i].clone();
                ch.sequence = j as u32;
                ch.position = c.position;
                self.chain[i] = ch;
                break;
            }
        }
    }

    fn update_card(&mut self, raw: &[u8]) -> Result<(), String> {
        let player = rd_u8(raw, 1)? as u32;
        let loc = rd_u8(raw, 2)? as u32;
        let seq = rd_u8(raw, 3)? as u32;
        let mut card = CardState { controller: player, location: loc, sequence: seq, ..Default::default() };
        Q::parse_card(&raw[4.min(raw.len())..], 0, &mut card)?;
        if let Some(slot) = self.slots.get_mut(&(player, loc)) {
            while slot.len() <= seq as usize {
                slot.push(None);
            }
            slot[seq as usize] = Some(card);
        } else if let Some(list) = self.lists.get_mut(&(player, loc)) {
            while list.len() <= seq as usize {
                list.push(CardState { controller: player, location: loc, sequence: list.len() as u32, ..Default::default() });
            }
            list[seq as usize] = card;
        }
        Ok(())
    }

    fn take_from(&mut self, con: u32, loc: u32, seq: u32) -> Option<CardState> {
        let key = (con, loc & !C::LOCATION_OVERLAY);
        if let Some(slot) = self.slots.get_mut(&key) {
            if (seq as usize) < slot.len() {
                return slot[seq as usize].take();
            }
            return None;
        }
        if let Some(list) = self.lists.get_mut(&(con, loc)) {
            if (seq as usize) < list.len() {
                let card = list.remove(seq as usize);
                for (i, c) in list.iter_mut().enumerate() {
                    c.sequence = i as u32;
                }
                // Chain handlers are live card pointers: later cards in the same list
                // shift down with it, and the core reports their current index.
                self.shift_chain(con, loc, seq + 1, -1, None);
                return Some(card);
            }
            return None;
        }
        if loc == C::LOCATION_DECK && con < 2 {
            self.deck_count[con as usize] = (self.deck_count[con as usize] - 1).max(0);
        }
        None
    }

    fn put_to(&mut self, card: Option<CardState>, code: u32, con: u32, loc: u32, seq: u32, pos: u32) {
        if loc == C::LOCATION_DECK {
            if con < 2 {
                self.deck_count[con as usize] += 1;
            }
            return;
        }
        let old = card.as_ref().map(|c| (c.location, c.sequence));
        let mut card = card.unwrap_or(CardState {
            controller: con,
            location: loc & !C::LOCATION_OVERLAY,
            sequence: seq,
            ..Default::default()
        });
        card.controller = con;
        card.location = loc;
        card.sequence = seq;
        card.position = pos;
        if code != 0 {
            card.code = code;
        }
        // Known by code only (a place/move from hand or deck names no stats): fill the
        // card DB row; a later refresh replaces it with the engine's live values.
        fill_stats(self.cards_db, &mut card, false);
        let key = (con, loc);
        if let Some(slot) = self.slots.get_mut(&key) {
            while slot.len() <= seq as usize {
                slot.push(None);
            }
            slot[seq as usize] = Some(card);
            return;
        }
        if let Some(list) = self.lists.get_mut(&key) {
            let i = (seq as usize).min(list.len());
            let skip = old.and_then(|(l, s)| (l == loc).then_some(s));
            list.insert(i, card);
            for (i, c) in list.iter_mut().enumerate() {
                c.sequence = i as u32;
            }
            self.shift_chain(con, loc, i as u32, 1, skip);
        }
    }

    fn shift_chain(&mut self, con: u32, loc: u32, at: u32, delta: i32, skip_seq: Option<u32>) {
        for ch in self.chain.iter_mut() {
            if (ch.controller as u32, ch.location as u32) != (con, loc) || ch.sequence < at {
                continue;
            }
            if skip_seq == Some(ch.sequence) {
                continue;
            }
            ch.sequence = (ch.sequence as i64 + delta as i64).max(0) as u32;
        }
    }

    fn mv(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.len() != 29 {
            return Err(format!("MSG_MOVE is {} bytes, expected 29", raw.len()));
        }
        let code = rd_u32(raw, 1)?;
        let (fc, fl, fs, fp) = loc_info(raw, 5)?;
        let (tc, tl, ts, tp) = loc_info(raw, 15)?;
        let (fc, tc) = (fc as u32, tc as u32);
        let (fl, tl) = (fl as u32, tl as u32);
        // Resolve the chain handler *before* the move: a list shift can move a different
        // link into (fc, fl, fs), which must not be mistaken for the moved card.
        let hit = self.chain_hit(fc, fl, fs, code);
        if fl & C::LOCATION_OVERLAY != 0 {
            self.detach_overlay(fc, fl, fs, fp, code);
            if tl & C::LOCATION_OVERLAY != 0 {
                self.attach_overlay(tc, tl, ts, tp, code);
                self.track_chain_at(hit, code, tc, tl, ts, tp);
                return Ok(());
            }
            if tc < 2 {
                self.put_to(None, code, tc, tl, ts, tp);
                self.track_chain_at(hit, code, tc, tl, ts, tp);
            }
            return Ok(());
        }
        if tl & C::LOCATION_OVERLAY != 0 {
            // Attached as xyz material: the holder is `tc` in `tl & ~OVERLAY` at `ts`,
            // and `tp` is the material index (core `card::get_info_location`).
            if fl == C::LOCATION_DECK && fc == self.seat {
                self.deck_remove(fs);
            }
            self.take_from(fc, fl, fs);
            self.attach_overlay(tc, tl, ts, tp, code);
            self.track_chain_at(hit, code, tc, tl, ts, tp);
            return Ok(());
        }
        if fc > 1 || tc > 1 {
            return Ok(());
        }
        if fl == C::LOCATION_DECK && fc == self.seat {
            self.deck_remove(fs);
        }
        let card = self.take_from(fc, fl, fs);
        if tl == C::LOCATION_DECK && tc == self.seat {
            let status = card.as_ref().map(|c| c.status).unwrap_or(0);
            self.deck_insert(ts, code, status);
        }
        self.put_to(card, code, tc, tl, ts, tp);
        self.track_chain_at(hit, code, tc, tl, ts, tp);
        Ok(())
    }

    fn demote_deck(&mut self) {
        for (&seq, &code) in &self.deck_seqs {
            if let Some(&status) = self.deck_stat.get(&seq) {
                self.deck_loose.add((code, status), 1);
            }
        }
        self.deck_seqs.clear();
        self.deck_stat.clear();
    }

    fn deck_remove(&mut self, seq: u32) {
        let seq = seq as i64;
        self.deck_seqs.remove(&seq);
        self.deck_stat.remove(&seq);
        self.deck_seqs = self.deck_seqs.iter().map(|(&s, &c)| (if s > seq { s - 1 } else { s }, c)).collect();
        self.deck_stat = self.deck_stat.iter().map(|(&s, &c)| (if s > seq { s - 1 } else { s }, c)).collect();
    }

    fn deck_insert(&mut self, seq: u32, code: u32, status: u32) {
        let seq = seq as i64;
        self.deck_seqs = self.deck_seqs.iter().map(|(&s, &c)| (if s >= seq { s + 1 } else { s }, c)).collect();
        self.deck_stat = self.deck_stat.iter().map(|(&s, &c)| (if s >= seq { s + 1 } else { s }, c)).collect();
        if code != 0 {
            self.deck_seqs.insert(seq, code);
            if status & STATUS_OBS_MASK != 0 {
                self.deck_stat.insert(seq, status & STATUS_OBS_MASK);
            }
        }
    }

    fn chain_hit(&self, con: u32, loc: u32, seq: u32, code: u32) -> Option<usize> {
        self.chain.iter().position(|ch| {
            (ch.controller as u32, ch.location as u32, ch.sequence) == (con, loc, seq)
                && !(code != 0 && ch.code != 0 && code != ch.code)
        })
    }

    fn track_chain_at(&mut self, index: Option<usize>, code: u32, tc: u32, tl: u32, ts: u32, tp: u32) {
        let Some(i) = index else { return };
        if let Some(ch) = self.chain.get_mut(i) {
            *ch = ChainLink {
                code: if ch.code != 0 { ch.code } else { code },
                controller: tc as u8,
                location: tl as u8,
                sequence: ts,
                position: tp,
                triggering_controller: ch.triggering_controller,
                triggering_location: ch.triggering_location,
                triggering_sequence: ch.triggering_sequence,
                description: ch.description,
            };
        }
    }

    fn entry_at(&self, con: u32, loc: u32, seq: u32) -> Option<&CardState> {
        if let Some(slot) = self.slots.get(&(con, loc)) {
            return slot.get(seq as usize).and_then(|c| c.as_ref());
        }
        if let Some(list) = self.lists.get(&(con, loc)) {
            return list.get(seq as usize);
        }
        None
    }

    pub fn set_status(&mut self, con: u32, loc: u32, seq: u32, bit: u32, on: bool) {
        if let Some(card) = self.entry_at_mut(con, loc, seq) {
            card.status = if on { card.status | bit } else { card.status & !bit };
        }
    }

    fn entry_at_mut(&mut self, con: u32, loc: u32, seq: u32) -> Option<&mut CardState> {
        if let Some(slot) = self.slots.get_mut(&(con, loc)) {
            return slot.get_mut(seq as usize).and_then(|c| c.as_mut());
        }
        if let Some(list) = self.lists.get_mut(&(con, loc)) {
            return list.get_mut(seq as usize);
        }
        None
    }

    pub fn set_public(&mut self, con: u32, loc: u32, seq: u32, on: bool) {
        if let Some(card) = self.entry_at_mut(con, loc, seq) {
            card.public = on;
        }
    }

    fn set_type(&mut self, con: u32, loc: u32, seq: u32, type_: u32) {
        if let Some(card) = self.entry_at_mut(con, loc, seq) {
            card.type_ = type_;
        }
    }

    /// A monster placed as a Continuous Trap reverts to its card-DB type (and stats)
    /// when its activation resolves; no message or refresh names it.
    fn revert_trap_monster(&mut self, con: u32, loc: u32, seq: u32) {
        let Some(card) = self.entry_at(con, loc, seq).cloned() else { return };
        if card.location != C::LOCATION_SZONE || card.position & C::POS_FACEUP == 0 {
            return;
        }
        let Some(d) = self.cards_db.get(&card.code) else { return };
        if card.type_ == d.type_ {
            return;
        }
        self.set_type(con, loc, seq, d.type_);
        let db = self.cards_db;
        if let Some(card) = self.entry_at_mut(con, loc, seq) {
            fill_stats(db, card, true);
        }
    }

    pub fn summon_revert(&mut self, r: &M::CardRef, db: &BTreeMap<u32, CardData>) {
        if let Some(card) = self.entry_at_mut(r.controller as u32, r.location, r.sequence) {
            if card.code != 0
                && let Some(d) = db.get(&card.code)
                && card.type_ != d.type_
            {
                card.type_ = d.type_;
                fill_stats(db, card, true);
            }
        }
    }

    fn attach_overlay(&mut self, con: u32, loc: u32, seq: u32, index: u32, code: u32) {
        if let Some(holder) = self.entry_at_mut(con, loc & !C::LOCATION_OVERLAY, seq) {
            while holder.overlay.len() <= index as usize {
                holder.overlay.push(0);
            }
            holder.overlay[index as usize] = code;
        }
    }

    fn detach_overlay(&mut self, con: u32, loc: u32, seq: u32, index: u32, code: u32) {
        if let Some(holder) = self.entry_at_mut(con, loc & !C::LOCATION_OVERLAY, seq) {
            let i = index as usize;
            if i < holder.overlay.len() && holder.overlay[i] == code {
                holder.overlay.remove(i);
            } else if let Some(j) = holder.overlay.iter().position(|x| *x == code) {
                holder.overlay.remove(j);
            }
        }
    }

    fn draw(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.len() < 6 {
            return Err(format!("MSG_DRAW truncated: {} bytes", raw.len()));
        }
        let player = rd_u8(raw, 1)?;
        let drawn = rd_u32(raw, 2)?;
        if player > 1 || raw.len() != 6 + 8 * drawn as usize {
            return Err(format!("MSG_DRAW malformed: {} bytes, {drawn} drawn", raw.len()));
        }
        let p = player as usize;
        self.deck_count[p] = (self.deck_count[p] - drawn as i32).max(0);
        if player as u32 == self.seat {
            let size = self.deck_count[p] as i64;
            self.deck_seqs.retain(|&s, _| s < size);
            self.deck_stat.retain(|&s, _| s < size);
        }
        let lst = self.lists.entry((player as u32, C::LOCATION_HAND)).or_default();
        for i in 0..drawn as usize {
            let code = rd_u32(raw, 6 + 8 * i)?;
            let pos = rd_u32(raw, 6 + 8 * i + 4)?;
            lst.push(CardState {
                controller: player as u32,
                location: C::LOCATION_HAND,
                sequence: lst.len() as u32,
                code,
                position: pos,
                ..Default::default()
            });
        }
        Ok(())
    }

    fn chain_link(&self, raw: &[u8]) -> Result<ChainLink, String> {
        if raw.len() < 29 {
            return Err(format!("MSG_CHAINING truncated: {} bytes", raw.len()));
        }
        let code = rd_u32(raw, 1)?;
        let (con, loc, seq, pos) = loc_info(raw, 5)?;
        let tcon = rd_u8(raw, 15)?;
        let tloc = rd_u8(raw, 16)?;
        let tseq = rd_u32(raw, 17)?;
        let desc = rd_u64(raw, 21)?;
        Ok(ChainLink {
            code,
            controller: con,
            location: loc,
            sequence: seq,
            position: pos,
            triggering_controller: tcon,
            triggering_location: tloc,
            triggering_sequence: tseq,
            description: desc,
        })
    }

    fn pos_change(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.len() < 10 {
            return Err(format!("MSG_POS_CHANGE truncated: {} bytes", raw.len()));
        }
        let code = rd_u32(raw, 1)?;
        let (cc, cl, cs, _pp, cp) = (raw[5] as u32, raw[6] as u32, raw[7] as u32, raw[8], raw[9] as u32);
        let db = self.cards_db;
        if let Some(card) = self.entry_at_mut(cc, cl, cs) {
            card.position = cp;
            if code != 0 {
                card.code = code;
            }
            fill_stats(db, card, false);
        }
        for ch in self.chain.iter_mut() {
            if (ch.controller as u32, ch.location as u32, ch.sequence) == (cc, cl, cs) {
                if code != 0 {
                    ch.code = code;
                }
                ch.position = cp;
            }
        }
        Ok(())
    }

    fn swap(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.len() != 29 {
            return Err(format!("MSG_SWAP is {} bytes, expected 29", raw.len()));
        }
        let code1 = rd_u32(raw, 1)?;
        let (c1, l1, s1, p1) = loc_info(raw, 5)?;
        let code2 = rd_u32(raw, 15)?;
        let (c2, l2, s2, p2) = loc_info(raw, 19)?;
        let a = self.take_from(c1 as u32, l1 as u32, s1);
        let b = self.take_from(c2 as u32, l2 as u32, s2);
        self.put_to(a, code1, c2 as u32, l2 as u32, s2, p2);
        self.put_to(b, code2, c1 as u32, l1 as u32, s1, p1);
        Ok(())
    }

    fn remove_cards(&mut self, raw: &[u8]) -> Result<(), String> {
        if raw.len() < 6 {
            return Err(format!("MSG_REMOVE_CARDS truncated: {} bytes", raw.len()));
        }
        let n = rd_u32(raw, 1)?;
        for i in 0..n as usize {
            let off = 5 + 10 * i;
            if off + 10 > raw.len() {
                return Err(format!("MSG_REMOVE_CARDS truncated at entry {i}"));
            }
            let (con, loc, seq, _pos) = loc_info(raw, off)?;
            self.take_from(con as u32, loc as u32, seq);
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ history

fn entry_row(
    kind: i32, ref_: &M::CardRef, code: u32, desc: u64, arg: i32, actor: i32,
) -> [i32; 10] {
    [
        kind,
        -1,
        code as i32,
        ((desc >> 20) & 0xFFFF_FFFF) as i32,
        (desc & 0xF_FFFF) as i32,
        arg,
        // msg_type is filled by the caller through `history_entry`
        0,
        loc_id(ref_.location),
        (ref_.controller as i32) ^ actor,
        0,
    ]
}

fn hist(
    kind: i32, ref_: &M::CardRef, code: u32, desc: u64, arg: i32, actor: i32, public: bool,
    msg_type: u32, turn: i32, phase: i32,
) -> HistoryRow {
    let mut row = entry_row(kind, ref_, code, desc, arg, actor);
    row[6] = msg_type as i32;
    HistoryRow { row, actor, turn, phase, public }
}

fn place_row(ref_: &M::CardRef, actor: i32, turn: i32, phase: i32) -> HistoryRow {
    let zone = if ref_.location == C::LOCATION_MZONE { ref_.sequence } else { ref_.sequence + 8 };
    let mut row = entry_row(19, ref_, 0, 0, zone as i32, actor);
    row[6] = C::MSG_SELECT_PLACE as i32;
    HistoryRow { row, actor, turn, phase, public: false }
}

/// Opponent history entries rebuilt from a public message ([] when unmappable).
pub fn history_from_message(state: &StreamState, raw: &[u8]) -> Vec<HistoryRow> {
    let seat = state.seat as i32;
    let turn = state.turn;
    let phase = state.phase;
    let Some(&tb) = raw.first() else { return Vec::new() };
    let t = tb as u32;
    let mut out = Vec::new();
    match t {
        C::MSG_SUMMONING | C::MSG_SPSUMMONING | C::MSG_FLIPSUMMONING if raw.len() >= 15 => {
            let Ok(code) = rd_u32(raw, 1) else { return out };
            let Ok((con, loc, seq, pos)) = loc_info(raw, 5) else { return out };
            if con as i32 != 1 - seat {
                return out;
            }
            let ref_ = M::CardRef { code, controller: con, location: loc as u32, sequence: seq, position: pos };
            if t == C::MSG_FLIPSUMMONING {
                out.push(hist(3, &ref_, code, 0, 0, con as i32, true, C::MSG_SELECT_IDLECMD, turn, phase));
            } else if state.chain.is_empty() {
                let kind = if t == C::MSG_SUMMONING { 1 } else { 2 };
                out.push(hist(kind, &ref_, code, 0, 0, con as i32, true, C::MSG_SELECT_IDLECMD, turn, phase));
                out.push(place_row(&ref_, con as i32, turn, phase));
            }
        }
        C::MSG_SET if raw.len() >= 15 => {
            let Ok((con, loc, seq, pos)) = loc_info(raw, 5) else { return out };
            if con as i32 != 1 - seat {
                return out;
            }
            let kind = if loc as u32 == C::LOCATION_MZONE { 4 } else { 5 };
            let ref_ = M::CardRef { code: 0, controller: con, location: loc as u32, sequence: seq, position: pos };
            out.push(hist(kind, &ref_, 0, 0, 0, con as i32, false, C::MSG_SELECT_IDLECMD, turn, phase));
            out.push(place_row(&ref_, con as i32, turn, phase));
        }
        C::MSG_CHAINING if raw.len() >= 29 => {
            let Ok(code) = rd_u32(raw, 1) else { return out };
            let Ok((con, loc, seq, pos)) = loc_info(raw, 5) else { return out };
            let Ok(tcon) = rd_u8(raw, 15) else { return out };
            let Ok(desc) = rd_u64(raw, 21) else { return out };
            if tcon as i32 == 1 - seat {
                let public = pos & C::POS_FACEUP != 0;
                let ref_ =
                    M::CardRef { code, controller: con, location: loc as u32, sequence: seq, position: pos };
                out.push(hist(6, &ref_, code, desc, 0, tcon as i32, public, t, turn, phase));
            }
        }
        C::MSG_ATTACK if raw.len() >= 21 => {
            let Ok((con, loc, seq, pos)) = loc_info(raw, 1) else { return out };
            let Ok((tcon, tloc, tseq, _tpos)) = loc_info(raw, 11) else { return out };
            if con as i32 == 1 - seat {
                let mut code = 0u32;
                if let Some(cards) = state.slots.get(&(con as u32, C::LOCATION_MZONE)) {
                    for c in cards.iter().flatten() {
                        if c.sequence == seq {
                            code = c.code;
                            break;
                        }
                    }
                }
                let direct = tloc == 0 && tcon == 0 && tseq == 0;
                let ref_ =
                    M::CardRef { code, controller: con, location: loc as u32, sequence: seq, position: pos };
                out.push(hist(7, &ref_, code, 0, i32::from(direct), con as i32, true, C::MSG_SELECT_BATTLECMD, turn, phase));
            }
        }
        C::MSG_POS_CHANGE if raw.len() >= 10 => {
            let Ok(code) = rd_u32(raw, 1) else { return out };
            let (cc, cl, cs, _pp, cp) = (raw[5], raw[6], raw[7], raw[8], raw[9]);
            if cc as i32 == 1 - seat && state.chain.is_empty() {
                let public = cp as u32 & C::POS_FACEUP != 0;
                let ref_ = M::CardRef {
                    code,
                    controller: cc,
                    location: cl as u32,
                    sequence: cs as u32,
                    position: cp as u32,
                };
                out.push(hist(3, &ref_, code, 0, 0, cc as i32, public, C::MSG_SELECT_IDLECMD, turn, phase));
            }
        }
        C::MSG_NEW_PHASE if raw.len() >= 3 => {
            let phase_now = u16::from_le_bytes(raw[1..3].try_into().unwrap()) as u32;
            if state.turn_player != seat {
                let kind_msg = match phase_now {
                    C::PHASE_BATTLE_START => Some((8, C::MSG_SELECT_IDLECMD)),
                    C::PHASE_MAIN2 => Some((9, C::MSG_SELECT_BATTLECMD)),
                    C::PHASE_END => Some((10, C::MSG_SELECT_BATTLECMD)),
                    _ => None,
                };
                if let Some((kind, msg_type)) = kind_msg {
                    let actor = state.turn_player;
                    let ref_ = M::CardRef { code: 0, controller: actor as u8, location: 0, sequence: 0, position: 0 };
                    out.push(hist(kind, &ref_, 0, 0, 0, actor, true, msg_type, turn, phase));
                }
            }
        }
        _ => {}
    }
    out
}

// ------------------------------------------------------------------ stream source

/// The stream plus the pending `ygo_decision` built from the received select message.
pub struct StreamSource<'e> {
    pub state: StreamState<'e>,
    engine: &'e Engine,
    cards: &'e BTreeMap<u32, CardData>,
    pending: Option<Decision>,
    pending_state: Option<DecisionState>,
    pending_raw: Vec<u8>,
    trace: Vec<Vec<u8>>,
    history: Vec<HistoryRow>,
    winner: Option<i32>,
    pub retries: usize,
    clear_deck_after: bool,
}

impl<'e> StreamSource<'e> {
    pub fn new(
        deck_main: &[u32], deck_extra: &[u32], cards: &'e BTreeMap<u32, CardData>, seat: u32,
        engine: &'e Engine,
    ) -> StreamSource<'e> {
        StreamSource {
            state: StreamState::new(deck_main, deck_extra, cards, seat),
            engine,
            cards,
            pending: None,
            pending_state: None,
            pending_raw: Vec::new(),
            trace: Vec::new(),
            history: Vec::new(),
            winner: None,
            retries: 0,
            clear_deck_after: false,
        }
    }

    pub fn seat(&self) -> u32 {
        self.state.seat
    }

    pub fn is_select_for_me(&self, payload: &[u8]) -> bool {
        payload.len() >= 2
            && M::is_select_type(payload[0] as u32)
            && payload[1] as u32 == self.state.seat
    }

    /// Fold one STOC_GAME_MSG payload in (select messages for our seat become pending decisions).
    pub fn feed(&mut self, payload: &[u8]) -> Result<(), String> {
        if payload.is_empty() {
            return Ok(());
        }
        let t = payload[0] as u32;
        if t == C::MSG_WIN {
            self.winner = payload.get(1).copied().map(|v| v as i32);
            self.trace.push(payload.to_vec());
            return Ok(());
        }
        if t == C::MSG_RETRY {
            self.retries += 1;
            self.trace.push(payload.to_vec());
            return Ok(());
        }
        if M::is_select_type(t) && payload.get(1).copied().map(u32::from) == Some(self.state.seat) {
            self.learn_select(payload)?;
            let pool = self.known_pool();
            let dec = Decision::new(self.engine, payload, &pool)?;
            self.pending_state = Some(DecisionState {
                msg_type: dec.msg_type(),
                player: dec.player(),
                step: dec.step(),
                selected: Vec::new(),
                options: { let mut o = dec.options(); o.truncate(ygoenv_sys::MAX_OPTIONS); o },
            });
            self.pending_raw = payload.to_vec();
            self.pending = Some(dec);
            self.trace.push(payload.to_vec());
            return Ok(());
        }
        self.state.feed(payload)?;
        self.history.extend(history_from_message(&self.state, payload));
        self.trace.push(payload.to_vec());
        Ok(())
    }

    /// ANNOUNCE_CARD pool, as `PyEnv.known_codes`: our deck list, every card this seat can
    /// see, and codes in its visible history (own rows or public ones).
    fn known_pool(&self) -> Vec<u32> {
        let mut codes: BTreeSet<u32> = self.state.deck_list.keys().copied().collect();
        codes.extend(self.state.extra_list.iter().copied());
        codes.extend(crate::Board::new(self, self.state.seat as i32).visible_codes());
        for h in &self.history {
            if h.actor == self.state.seat as i32 || h.public {
                let c = h.row[2];
                if c > 0 {
                    codes.insert(c as u32);
                }
            }
        }
        codes.remove(&0);
        codes.into_iter().collect()
    }

    fn learn_select(&mut self, payload: &[u8]) -> Result<(), String> {
        let msg = match M::parse_one(payload) {
            Ok(m) => m,
            Err(e) => return Err(format!("cannot parse select: {e}")),
        };
        let mut refs: Vec<M::CardRef> = Vec::new();
        match &msg.body {
            M::Body::SortCards { cards, .. } => {
                self.state.learn_deck_seqs(cards);
                if cards.iter().any(|r| {
                    r.controller as u32 == self.state.seat && r.location == C::LOCATION_DECK
                }) {
                    self.clear_deck_after = true;
                }
                return Ok(());
            }
            M::Body::SelectCard { cards, .. } => refs = cards.clone(),
            M::Body::SelectTribute { cards, .. } => refs = cards.iter().map(|(c, _)| *c).collect(),
            M::Body::SelectSum { must, cards, .. } => {
                refs = must.iter().chain(cards.iter()).map(|(c, _)| *c).collect()
            }
            M::Body::SelectUnselectCard { selectable, unselectable, .. } => {
                refs = selectable.iter().chain(unselectable.iter()).copied().collect()
            }
            _ => {}
        }
        self.state.learn_deck_seqs(&refs);
        Ok(())
    }

    /// Answer the pending decision with `index` (the stream's side effects, as `StreamView.take`).
    pub fn take(&mut self, index: usize) -> Result<Option<Vec<u8>>, String> {
        let Some(opts) = self.pending.as_ref().map(|d| d.options()) else { return Ok(None) };
        let opt = opts.get(index).cloned().ok_or_else(|| format!("option {index} of {}", opts.len()))?;
        // Our own picks are known to the client and belong in its history exactly as the
        // host records them for the same decision (mirrors `StreamView.take` in Python).
        if let Some(ps) = &self.pending_state {
            let me = self.state.seat as i32;
            let entry = crate::Board::new(self, me).history_entry(
                &opt,
                me,
                ps.msg_type,
                ps.step,
                self.state.turn,
                self.state.phase,
            );
            self.history.push(entry);
        }
        let dec = self.pending.as_mut().unwrap();
        if opt.has_card != 0 {
            let r = M::CardRef {
                code: opt.code,
                controller: opt.controller as u8,
                location: opt.location,
                sequence: opt.sequence,
                position: opt.position,
            };
            if opt.kind == 1 || opt.kind == 2 {
                self.state.summon_revert(&r, self.cards);
            }
            if opt.kind == 6 {
                self.state.set_status(r.controller as u32, r.location, r.sequence, C::STATUS_CHAINING, true);
            }
        }
        let resp = dec.choose(index as i32)?;
        if resp.is_some() {
            if self.clear_deck_after {
                self.state.demote_deck();
                self.clear_deck_after = false;
            }
            self.pending = None;
            self.pending_state = None;
            self.pending_raw.clear();
        } else {
            self.pending_state = Some(DecisionState {
                msg_type: dec.msg_type(),
                player: dec.player(),
                step: dec.step(),
                selected: Vec::new(),
                options: { let mut o = dec.options(); o.truncate(ygoenv_sys::MAX_OPTIONS); o },
            });
        }
        Ok(resp)
    }

    pub fn needs_answer(&self) -> bool {
        self.pending.is_some()
    }

    pub fn n_opts(&self) -> usize {
        self.pending_state.as_ref().map(|d| d.options.len()).unwrap_or(0)
    }

    pub fn history_rows(&self) -> Vec<[i32; 14]> {
        self.history.iter().map(|h| h.to_log_row()).collect()
    }

    pub fn trace_msgs(&self) -> &[Vec<u8>] {
        &self.trace
    }

    pub fn winner(&self) -> Option<i32> {
        self.winner
    }
}

impl ViewSource for StreamSource<'_> {
    fn duel_view(&self) -> crate::DuelView {
        crate::DuelView {
            turn: self.state.turn,
            phase: self.state.phase,
            turn_player: self.state.turn_player,
            first_player: self.state.first_player.unwrap_or(0),
            winner: self.winner.unwrap_or(0),
        }
    }

    fn query_location(&self, player: i32, location: u32) -> Vec<Option<CardState>> {
        if !(0..2).contains(&player) {
            return Vec::new();
        }
        StreamState::query_location(&self.state, player as u32, location)
    }

    fn field_state(&self) -> ((i64, i64), Vec<ChainLink>) {
        StreamState::field_state(&self.state)
    }

    fn trace(&self) -> Vec<Vec<u8>> {
        self.trace.clone()
    }

    fn trace_len(&self) -> usize {
        self.trace.len()
    }

    fn trace_msg(&self, i: usize) -> Vec<u8> {
        self.trace.get(i).cloned().unwrap_or_default()
    }

    fn history(&self) -> Vec<[i32; 14]> {
        self.history_rows()
    }

    fn decision(&self) -> Option<DecisionState> {
        self.pending_state.clone()
    }

    fn pending_msg(&self) -> Vec<u8> {
        self.pending_raw.clone()
    }

    fn done(&self) -> bool {
        self.winner.is_some()
    }

    fn player(&self) -> i32 {
        self.pending_state.as_ref().map(|d| d.player).unwrap_or(self.state.turn_player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> StreamState<'static> {
        static CARDS: std::sync::OnceLock<BTreeMap<u32, CardData>> = std::sync::OnceLock::new();
        let cards = CARDS.get_or_init(BTreeMap::new);
        StreamState::new(&[1, 2, 3], &[], cards, 0)
    }

    #[test]
    fn malformed_packets_are_errors_not_panics() {
        for n in 0..40usize {
            let mut raw = vec![0xFFu8; n];
            for t in [
                C::MSG_START,
                C::MSG_MOVE,
                C::MSG_DRAW,
                C::MSG_NEW_TURN,
                C::MSG_NEW_PHASE,
                C::MSG_CHAINING,
                C::MSG_POS_CHANGE,
                C::MSG_SWAP,
                C::MSG_REMOVE_CARDS,
                C::MSG_SWAP_GRAVE_DECK,
                C::MSG_SHUFFLE_DECK,
                C::MSG_UPDATE_DATA,
                C::MSG_UPDATE_CARD,
            ] {
                if raw.is_empty() {
                    raw.push(t as u8);
                } else {
                    raw[0] = t as u8;
                }
                let _ = state().feed(&raw);
            }
        }
    }

    #[test]
    fn start_and_draw_keep_counts_and_hand() {
        let mut st = state();
        let mut start = vec![C::MSG_START as u8, 0];
        start.extend(8000u32.to_le_bytes());
        start.extend(8000u32.to_le_bytes());
        for v in [3u16, 0, 3, 0] {
            start.extend(v.to_le_bytes());
        }
        st.feed(&start).unwrap();
        assert_eq!(st.deck_count, [3, 3]);
        let mut draw = vec![C::MSG_DRAW as u8, 0];
        draw.extend(2u32.to_le_bytes());
        for code in [10u32, 11] {
            draw.extend(code.to_le_bytes());
            draw.extend(0u32.to_le_bytes());
        }
        st.feed(&draw).unwrap();
        assert_eq!(st.deck_count[0], 1);
        let hand = st.query_location(0, C::LOCATION_HAND);
        assert_eq!(hand.len(), 2);
        assert_eq!(hand[1].as_ref().unwrap().code, 11);
    }

    #[test]
    fn history_rows_are_14_ints_with_actor_and_public() {
        let mut st = state();
        st.turn = 3;
        st.phase = C::PHASE_MAIN1 as i32;
        let mut raw = vec![C::MSG_CHAINING as u8];
        raw.extend(99u32.to_le_bytes());
        raw.extend([1u8, C::LOCATION_MZONE as u8]);
        raw.extend(2u32.to_le_bytes());
        raw.extend(C::POS_FACEUP_ATTACK.to_le_bytes());
        raw.extend([1u8, 0u8]);
        raw.extend(0u32.to_le_bytes());
        raw.extend(7u64.to_le_bytes());
        raw.extend(1u32.to_le_bytes());
        let rows = history_from_message(&st, &raw);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].row[0], 6);
        assert_eq!(rows[0].actor, 1);
        assert_eq!(rows[0].row[2], 99);
        let log = rows[0].to_log_row();
        assert_eq!(log[10], 1);
        assert_eq!(log[13], 1);
    }
}
