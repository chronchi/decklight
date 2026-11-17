//! Web state frame over the Rust env: `build_view`, `build_prompt`, `build_events`, `build_cues` and `DuelStats`
//! (behaviour-for-behaviour ports of `src/decklight/web/view.py`, `cues.py`, `stats.py`).

pub mod cues;
pub mod stats;
pub mod stream;

use std::collections::HashMap;

use decklight_cards::texts::{Stats, Texts};
use decklight_core::constants as C;
use decklight_core::messages::{self as M, Body};
use decklight_core::query::{self as Q, CardState, ChainLink};
use serde_json::{Value, json};
use ygoenv_sys::{CardRef as EnvRef, DecisionState, DuelState, Env};

pub use cues::{build_cues, cues_for_trace};
pub use stats::DuelStats;
pub use stream::{HistoryRow, StreamSource, StreamState, history_from_message};

pub const MAX_CARDS: usize = 160;

/// The duel timing/result fields the builders read (the subset of `DuelState` they use).
#[derive(Clone, Copy, Debug, Default)]
pub struct DuelView {
    pub turn: i32,
    pub phase: i32,
    pub turn_player: i32,
    pub first_player: i32,
    pub winner: i32,
}

/// Everything `build_view`, `build_prompt`, `build_events`, `build_cues` and `DuelStats` read.
///
/// Implemented by the local Rust `Env` (bot duels, unchanged behaviour) and by the stream source
/// (a network/local client's seat). `field_state` returns LP already sign-extended, exactly as the
/// old `build_view` did after reading `query.lps`.
pub trait ViewSource {
    fn duel_view(&self) -> DuelView;
    fn query_location(&self, player: i32, location: u32) -> Vec<Option<CardState>>;
    fn field_state(&self) -> ((i64, i64), Vec<ChainLink>);
    fn trace(&self) -> Vec<Vec<u8>>;
    /// Number of messages in the trace (no copy).
    fn trace_len(&self) -> usize {
        self.trace().len()
    }
    /// One message of the trace (copies just that message).
    fn trace_msg(&self, i: usize) -> Vec<u8> {
        self.trace().get(i).cloned().unwrap_or_default()
    }
    fn history(&self) -> Vec<[i32; 14]>;
    fn decision(&self) -> Option<DecisionState>;
    fn pending_msg(&self) -> Vec<u8>;
    fn done(&self) -> bool;
    fn player(&self) -> i32;
}

fn signed(v: u32) -> i64 {
    if v >= 1 << 31 { v as i64 - (1i64 << 32) } else { v as i64 }
}

impl ViewSource for Env<'_> {
    fn duel_view(&self) -> DuelView {
        let d = self.duel_state().unwrap_or(DuelState {
            turn: 0,
            turn_player: 0,
            first_player: 0,
            phase: 0,
            winner: 0,
            win_reason: 0,
            select_hint: 0,
        });
        DuelView {
            turn: d.turn,
            phase: d.phase,
            turn_player: d.turn_player,
            first_player: d.first_player,
            winner: d.winner,
        }
    }

    fn query_location(&self, player: i32, location: u32) -> Vec<Option<CardState>> {
        if !(0..2).contains(&player) {
            return Vec::new();
        }
        self.query_location(player as u8, location, Q::CARD_FLAGS)
            .ok()
            .and_then(|b| Q::parse_location(&b, player as u32, location).ok())
            .unwrap_or_default()
    }

    fn field_state(&self) -> ((i64, i64), Vec<ChainLink>) {
        let field = self.query_field().unwrap_or_default();
        let lp = Q::lps(&field).unwrap_or((0, 0));
        let chain = Q::parse_chain(&field).unwrap_or_default();
        ((signed(lp.0), signed(lp.1)), chain)
    }

    fn trace(&self) -> Vec<Vec<u8>> {
        (0..Env::trace_len(self)).map(|i| Env::trace_msg(self, i)).collect()
    }

    fn trace_len(&self) -> usize {
        Env::trace_len(self)
    }

    fn trace_msg(&self, i: usize) -> Vec<u8> {
        Env::trace_msg(self, i)
    }

    fn history(&self) -> Vec<[i32; 14]> {
        self.history_log()
    }

    fn decision(&self) -> Option<DecisionState> {
        Env::decision(self)
    }

    fn pending_msg(&self) -> Vec<u8> {
        Env::pending_msg(self)
    }

    fn done(&self) -> bool {
        Env::done(self)
    }

    fn player(&self) -> i32 {
        Env::player(self)
    }
}

pub struct Ctx {
    pub texts: Texts,
    pub card_types: std::collections::BTreeMap<u32, u32>,
    pub lp0: i64,
}

impl Ctx {
    pub fn new(texts: Texts, card_types: std::collections::BTreeMap<u32, u32>, lp0: i64) -> Ctx {
        Ctx { texts, card_types, lp0 }
    }

    pub(crate) fn name_or(&self, code: i64) -> String {
        if (0..=u32::MAX as i64).contains(&code)
            && let Some(n) = self.texts.names().get(&(code as u32))
        {
            return n.clone();
        }
        format!("#{code}")
    }

    pub(crate) fn stats_of(&self, code: i64) -> Option<Stats> {
        if !(0..=u32::MAX as i64).contains(&code) {
            return None;
        }
        self.texts.stats_of(code as u32).copied()
    }

    pub(crate) fn card_type(&self, code: u32) -> u32 {
        if let Some(t) = self.card_types.get(&code) {
            return *t;
        }
        self.texts.stats_of(code).map(|s| s.5).unwrap_or(0)
    }

    pub(crate) fn desc_text(&self, hi: i64, lo: i64) -> String {
        self.texts.desc_text(hi, lo)
    }

    // ------------------------------------------------------------------ view

    pub fn build_view(&self, source: &dyn ViewSource, human_seat: i32, reveal_all: bool) -> Value {
        let duel = source.duel_view();
        let mut q: HashMap<(i32, u32), Vec<Option<CardState>>> = HashMap::new();
        for p in 0..2i32 {
            for loc in [
                C::LOCATION_HAND,
                C::LOCATION_MZONE,
                C::LOCATION_SZONE,
                C::LOCATION_GRAVE,
                C::LOCATION_REMOVED,
                C::LOCATION_EXTRA,
                C::LOCATION_DECK,
            ] {
                q.insert((p, loc), source.query_location(p, loc));
            }
        }
        let (lp, chain) = source.field_state();
        let me = human_seat;

        let empty = Vec::new();
        let get = |p: i32, loc: u32| q.get(&(p, loc)).unwrap_or(&empty);

        let side = |p: i32| -> Value {
            let mine = reveal_all || p == me;
            let mut hand = Vec::new();
            for c in get(p, C::LOCATION_HAND) {
                if let Some(c) = c {
                    hand.push(if mine || public(c) {
                        self.state_card(c, true)
                    } else {
                        Value::Null
                    });
                }
            }
            let mz = get(p, C::LOCATION_MZONE);
            let mzone: Vec<Value> = pad_slots(mz, 0, 5).iter().map(|c| self.slot(c.as_ref(), mine)).collect();
            let emz: Vec<Value> = pad_slots(mz, 5, 7).iter().map(|c| self.slot(c.as_ref(), mine)).collect();
            let sz = get(p, C::LOCATION_SZONE);
            let szone: Vec<Value> = pad_slots(sz, 0, 5).iter().map(|c| self.slot(c.as_ref(), mine)).collect();
            let fcard = if sz.len() > 5 { self.slot(sz[5].as_ref(), mine) } else { Value::Null };
            let gy: Vec<Value> = get(p, C::LOCATION_GRAVE)
                .iter()
                .flatten()
                .map(|c| self.state_card(c, true))
                .collect();
            let mut ban = Vec::new();
            for c in get(p, C::LOCATION_REMOVED) {
                if let Some(c) = c {
                    ban.push(if mine || public(c) { self.state_card(c, true) } else { Value::Null });
                }
            }
            let extra_all: Vec<&CardState> = get(p, C::LOCATION_EXTRA).iter().flatten().collect();
            let extra: Vec<Value> = if mine {
                extra_all.iter().map(|c| self.state_card(c, true)).collect()
            } else {
                extra_all
                    .iter()
                    .filter(|c| c.position & C::POS_FACEUP != 0)
                    .map(|c| self.state_card(c, true))
                    .collect()
            };
            let deck_n = get(p, C::LOCATION_DECK).iter().flatten().count();
            let lpv = if p == 0 { lp.0 } else { lp.1 };
            json!({
                "lp": lpv.max(0),
                "deck": deck_n,
                "extra_count": extra_all.len(),
                "hand": hand,
                "mzone": mzone,
                "emz": emz,
                "szone": szone,
                "field": fcard,
                "gy": gy,
                "banished": ban,
                "extra": extra,
            })
        };

        let mut chain_out = Vec::new();
        for ch in &chain {
            let mut found: Option<&CardState> = None;
            for p in 0..2i32 {
                for loc in [
                    C::LOCATION_MZONE,
                    C::LOCATION_SZONE,
                    C::LOCATION_HAND,
                    C::LOCATION_GRAVE,
                    C::LOCATION_REMOVED,
                ] {
                    for c in get(p, loc) {
                        if let Some(c) = c
                            && c.controller == ch.controller as u32
                            && c.location == ch.location as u32
                            && c.sequence == ch.sequence
                            && c.code == ch.code
                        {
                            found = Some(c);
                        }
                    }
                }
            }
            let card = match found {
                Some(c) => self.state_card(c, true),
                None => self.named_card(
                    ch.code as i64,
                    uid(ch.controller as i64, ch.location as i64, ch.sequence as i64, ch.code as i64),
                ),
            };
            chain_out.push(json!({
                "card": card,
                "controller": if ch.controller as i32 == me { "you" } else { "bot" },
            }));
        }
        let phase = phase_name(duel.phase).unwrap_or("main1");
        json!({
            "turn": duel.turn,
            "phase": phase,
            "turn_player": if duel.turn_player == me { "you" } else { "bot" },
            "you": side(me),
            "bot": side(1 - me),
            "chain": chain_out,
        })
    }

    fn slot(&self, c: Option<&CardState>, mine: bool) -> Value {
        match c {
            None => Value::Null,
            Some(c) => self.state_card(c, mine || public(c)),
        }
    }

    /// `web.replay._fallback_label`: the label of a choice at a position with no multi-option prompt.
    pub fn fallback_label(&self, source: &dyn ViewSource, seat: i32, index: usize) -> String {
        let Some(dec) = source.decision() else { return format!("Option {index}") };
        let Some(opt) = dec.options.get(index) else { return format!("Option {index}") };
        let board = Board::new(source, seat);
        let card = self.ref_card(seat, opt.has_card != 0, &env_ref(opt), Some(&board));
        let where_ = where_ref(opt.has_card != 0, &env_ref(opt), seat);
        let effect = self.effect(opt);
        let s = label(opt, card.as_ref(), where_.as_ref(), effect.as_deref(), seat, dec.player);
        if s.is_empty() { kind_lower(kind_of(opt.kind)).to_uppercase() } else { s }
    }

    // ---------------------------------------------------------------- prompt

    pub fn build_prompt(&self, source: &dyn ViewSource, human_seat: i32, prompt_id: i64) -> Option<Value> {
        if source.done() || source.player() != human_seat {
            return None;
        }
        let dec = source.decision()?;
        let opts = &dec.options;
        if opts.len() <= 1 {
            return None;
        }
        let view = self.build_view(source, human_seat, false);
        let board = Board::new(source, human_seat);
        let raw = source.pending_msg();
        let msg = M::parse_one(&raw).ok()?;
        let kind = prompt_kind(&msg.body);
        let mut options = Vec::new();
        for (i, opt) in opts.iter().enumerate() {
            let card = self.ref_card(human_seat, opt.has_card != 0, &env_ref(opt), Some(&board));
            let where_ = where_ref(opt.has_card != 0, &env_ref(opt), human_seat);
            let effect = self.effect(opt);
            let label = label(opt, card.as_ref(), where_.as_ref(), effect.as_deref(), human_seat, dec.player);
            options.push(json!({
                "index": i,
                "verb": verb(opt),
                "label": label,
                "card": card,
                "where": where_,
                "effect": effect,
            }));
        }
        let (min_n, max_n) = msg_min_max(&msg.body);
        Some(json!({
            "id": prompt_id,
            "kind": kind,
            "text": self.prompt_text(source, human_seat, kind, &view, &msg.body),
            "card": self.prompt_card(source, human_seat, kind, &view, &msg.body),
            "options": options,
            "min": min_n,
            "max": max_n,
        }))
    }

    fn prompt_text(&self, source: &dyn ViewSource, human_seat: i32, kind: &str, view: &Value, body: &Body) -> String {
        match kind {
            "idle" => format!("Your move — {}", phase_label(view["phase"].as_str().unwrap_or("main1"))),
            "battle" => "Battle Phase — choose your attack".to_string(),
            "chain" => {
                let chain = view["chain"].as_array().cloned().unwrap_or_default();
                let nm = chain
                    .last()
                    .and_then(|c| c.get("card"))
                    .and_then(|c| c.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if nm.is_empty() {
                    "Respond to the chain?".to_string()
                } else {
                    format!("Chain {nm}'s effect?")
                }
            }
            "effectyn" => {
                let ref_ = message_card_ref(body);
                let board = Board::new(source, human_seat);
                let card = ref_.and_then(|r| self.ref_card_m(human_seat, r, &board));
                let nm = card
                    .as_ref()
                    .and_then(|c| c.get("name"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("this card")
                    .to_string();
                let (hi, lo) = description(body).map(|d| ((d >> 20) as i64, (d & 0xFFFFF) as i64)).unwrap_or((0, 0));
                let loc = loc_word(ref_);
                if hi <= 0 && (lo == 0 || lo == 221) {
                    if loc.is_empty() {
                        format!("Use {nm}'s effect?")
                    } else {
                        format!("Use {nm}'s effect in the {loc}?")
                    }
                } else {
                    let d = fill(&self.desc_text(hi, lo), &nm, &loc);
                    if d.is_empty() {
                        format!("Use {nm}'s effect?")
                    } else {
                        format!("Use {nm}'s effect? (\"{d}\")")
                    }
                }
            }
            "yesno" => {
                let d = description(body)
                    .map(|v| fill(&self.desc_text((v >> 20) as i64, (v & 0xFFFFF) as i64), "", ""))
                    .unwrap_or_default();
                if d.is_empty() { "Answer yes or no".to_string() } else { d }
            }
            "option" => "Choose one of the effect's options".to_string(),
            "card" => {
                let hint = self.select_hint(source, human_seat);
                let mut base = if hint.is_empty() { "Select the card(s)".to_string() } else { hint };
                let (mn, mx) = msg_min_max(body);
                if mx > mn {
                    base += &format!(" (pick {mn}–{mx})");
                } else if mn == 1 && mx == 1 {
                    base += " (pick 1)";
                }
                base
            }
            "place" => {
                let nm = self.place_card_name(source, human_seat);
                if nm.is_empty() { "Choose a zone".to_string() } else { format!("Choose a zone for {nm}") }
            }
            "position" => "Choose a battle position".to_string(),
            "announce" => {
                if matches!(body, Body::AnnounceCard { .. }) {
                    "Declare a card".to_string()
                } else {
                    "Make your declaration".to_string()
                }
            }
            "sort" => "Order the cards".to_string(),
            _ => "Choose".to_string(),
        }
    }

    fn prompt_card(&self, source: &dyn ViewSource, human_seat: i32, kind: &str, view: &Value, body: &Body) -> Option<Value> {
        if kind == "effectyn" {
            let board = Board::new(source, human_seat);
            return message_card_ref(body).and_then(|r| self.ref_card_m(human_seat, r, &board));
        }
        if let Body::SelectPosition { code, .. } = body {
            if *code != 0 {
                return Some(self.named_card(*code as i64, format!("pos:{code}")));
            }
        }
        if kind == "place" {
            return None;
        }
        view["chain"]
            .as_array()
            .and_then(|a| a.last())
            .and_then(|c| c.get("card"))
            .cloned()
    }

    fn effect(&self, opt: &ygoenv_sys::Opt) -> Option<String> {
        let k = opt.kind;
        if !matches!(k, 6 | 14 | 12 | 13) || opt.desc == 0 {
            return None;
        }
        let hi = (opt.desc >> 20) as i64;
        let lo = (opt.desc & 0xFFFFF) as i64;
        if hi <= 0 && (lo == 0 || lo == 221) && matches!(k, 12 | 13) {
            return None;
        }
        let nm = if opt.has_card != 0 { self.name_or(opt.code as i64) } else { String::new() };
        let loc = if opt.has_card != 0 { loc_word_loc(opt.location) } else { String::new() };
        let s = fill(&self.desc_text(hi, lo), &nm, &loc);
        if s.is_empty() { None } else { Some(s) }
    }

    fn ref_card_m(&self, human_seat: i32, r: &M::CardRef, board: &Board) -> Option<Value> {
        let env_ref = EnvRef {
            code: r.code,
            controller: r.controller as u32,
            location: r.location,
            sequence: r.sequence,
            position: r.position,
        };
        self.ref_card(human_seat, true, &env_ref, Some(board))
    }

    fn ref_card(&self, human: i32, has_card: bool, r: &EnvRef, board: Option<&Board>) -> Option<Value> {
        if !has_card {
            return None;
        }
        let board = board;
        if let Some(board) = board
            && let Some((row_card, visible)) = board.row_of_env(r)
        {
            let mut d = self.state_card(row_card, *visible);
            if d["code"].as_u64().unwrap_or(0) != 0
                && r.location != C::LOCATION_MZONE
                && r.location != C::LOCATION_SZONE
                && let Some(o) = d.as_object_mut()
            {
                o.insert("faceup".into(), Value::Bool(true));
            }
            return Some(d);
        }
        if r.controller as i32 == human || r.location == C::LOCATION_DECK || r.location == C::LOCATION_EXTRA {
            return Some(self.named_card(
                r.code as i64,
                uid(r.controller as i64, r.location as i64, r.sequence as i64, r.code as i64),
            ));
        }
        let mut d = blank_card(uid(r.controller as i64, r.location as i64, r.sequence as i64, 0));
        if let Some(o) = d.as_object_mut() {
            o.insert("pos".into(), Value::String("set".into()));
        }
        Some(d)
    }

    fn select_hint(&self, source: &dyn ViewSource, human_seat: i32) -> String {
        for i in (0..source.trace_len()).rev() {
            let raw = source.trace_msg(i);
            if raw.len() < 11 || raw[0] as u32 != C::MSG_HINT || raw[1] as u32 != C::HINT_SELECTMSG
                || raw[2] as i32 != human_seat
            {
                continue;
            }
            let data = u64::from_le_bytes(raw[3..11].try_into().unwrap());
            return fill(&self.desc_text((data >> 20) as i64, (data & 0xFFFFF) as i64), "", "");
        }
        String::new()
    }

    fn place_card_name(&self, source: &dyn ViewSource, human_seat: i32) -> String {
        for i in (0..source.trace_len()).rev() {
            let raw = source.trace_msg(i);
            if raw.len() < 11 || raw[0] as u32 != C::MSG_HINT || raw[1] as u32 != C::HINT_SELECTMSG
                || raw[2] as i32 != human_seat
            {
                continue;
            }
            let data = u64::from_le_bytes(raw[3..11].try_into().unwrap());
            return self
                .texts
                .names()
                .get(&(data as u32))
                .cloned()
                .unwrap_or_default();
        }
        String::new()
    }

    // ---------------------------------------------------------------- events

    pub fn build_events(&self, source: &dyn ViewSource, human_seat: i32, since: usize) -> Vec<Value> {
        let history = source.history();
        let mut out = Vec::new();
        for (idx, e) in history.iter().enumerate().skip(since) {
            out.push(self.event_from_row(idx + 1, e, human_seat));
        }
        out
    }

    pub fn event_from_row(&self, n: usize, e: &[i32; 14], human_seat: i32) -> Value {
        let kind = kind_of(e[0]);
        let code = e[2] as i64;
        let actor = e[10];
        let is_public = e[13] != 0;
        let me = actor == human_seat;
        let who = if me { "you" } else { "bot" };
        let (text, card) = if !me && !is_public {
            (hidden_say(kind).to_string(), Value::Null)
        } else {
            let nm = if code != 0 { self.name_or(code) } else { String::new() };
            let text = if kind == Some(KIND_PLACE) {
                if nm.is_empty() { "chooses a zone".to_string() } else { format!("places {nm}") }
            } else if nm.is_empty() && kind.is_some_and(needs_object) {
                hidden_say(kind).to_string()
            } else {
                let say = kind.and_then(say).unwrap_or("plays");
                format!("{say} {nm}").trim().to_string()
            };
            let card = if code != 0 { self.named_card(code, format!("e{n}")) } else { Value::Null };
            (text, card)
        };
        json!({
            "n": n,
            "actor": who,
            "text": capitalize(&text),
            "card": card,
            "kind": kind.map(|k| kind_lower(Some(k))),
        })
    }

    // ---------------------------------------------------------------- cards

    pub(crate) fn state_card(&self, c: &CardState, visible: bool) -> Value {
        let code = if visible { c.code as i64 } else { 0 };
        let uid = uid(c.controller as i64, c.location as i64, c.sequence as i64, code);
        if !visible {
            let mut d = blank_card(uid);
            if let Some(o) = d.as_object_mut() {
                o.insert("pos".into(), pos_of(c.location, c.position));
            }
            return d;
        }
        let overlays: Vec<Value> = c
            .overlay
            .iter()
            .enumerate()
            .map(|(i, code)| {
                self.full_card(c.controller as i64, C::LOCATION_OVERLAY as i64, i as i64, *code as i64, true, Value::Null,
                               0, 0, 0, 0, 0, Vec::new())
            })
            .collect();
        let faceup = c.location == C::LOCATION_HAND || c.position & C::POS_FACEUP != 0;
        self.full_card(
            c.controller as i64,
            c.location as i64,
            c.sequence as i64,
            c.code as i64,
            faceup,
            pos_of(c.location, c.position),
            c.attack as i64,
            c.defense as i64,
            c.level as i64,
            c.rank as i64,
            c.link as i64,
            overlays,
        )
    }

    pub(crate) fn named_card(&self, code: i64, uid: String) -> Value {
        match self.stats_of(code) {
            None => self.full_card(0, 0, 0, code, true, Value::Null, 0, 0, 0, 0, 0, Vec::new()),
            Some((atk, dfn, level, rank, link, _t)) => {
                let mut d = self.full_card(
                    0, 0, 0, code, true, Value::Null, atk as i64, dfn as i64, level as i64, rank as i64, link as i64,
                    Vec::new(),
                );
                if let Some(o) = d.as_object_mut() {
                    o.insert("uid".into(), Value::String(uid));
                }
                d
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn full_card(
        &self,
        controller: i64,
        location: i64,
        sequence: i64,
        code: i64,
        faceup: bool,
        pos: Value,
        atk: i64,
        dfn: i64,
        level: i64,
        rank: i64,
        link: i64,
        overlays: Vec<Value>,
    ) -> Value {
        json!({
            "uid": uid(controller, location, sequence, code),
            "code": code,
            "name": self.name_or(code),
            "faceup": faceup,
            "pos": pos,
            "atk": atk,
            "def": dfn,
            "level": level,
            "rank": rank,
            "link": link,
            "counters": 0,
            "overlays": overlays,
        })
    }
}

pub(crate) fn env_ref(opt: &ygoenv_sys::Opt) -> EnvRef {
    EnvRef {
        code: opt.code,
        controller: opt.controller,
        location: opt.location,
        sequence: opt.sequence,
        position: opt.position,
    }
}

pub(crate) fn uid(controller: i64, location: i64, sequence: i64, code: i64) -> String {
    format!("{controller}:{location:#x}:{sequence}:{code}")
}

pub(crate) fn blank_card(uid: String) -> Value {
    json!({
        "uid": uid,
        "code": 0,
        "name": Value::Null,
        "faceup": false,
        "pos": Value::Null,
        "atk": 0,
        "def": 0,
        "level": 0,
        "rank": 0,
        "link": 0,
        "counters": 0,
        "overlays": Vec::<Value>::new(),
    })
}

pub(crate) fn pos_of(location: u32, position: u32) -> Value {
    let fd = position & C::POS_FACEDOWN != 0;
    if location == C::LOCATION_MZONE {
        return Value::String(if fd {
            "set".into()
        } else if position & C::POS_FACEUP_ATTACK != 0 {
            "atk".into()
        } else {
            "def".into()
        });
    }
    if location == C::LOCATION_SZONE && fd {
        return Value::String("set".into());
    }
    Value::Null
}

pub(crate) fn public(c: &CardState) -> bool {
    if c.location == C::LOCATION_GRAVE || c.location == C::LOCATION_OVERLAY {
        return true;
    }
    c.public || (c.position & C::POS_FACEUP != 0 && c.location != C::LOCATION_HAND)
}

pub fn phase_name(phase: i32) -> Option<&'static str> {
    Some(match phase as u32 {
        C::PHASE_DRAW => "draw",
        C::PHASE_STANDBY => "standby",
        C::PHASE_MAIN1 => "main1",
        C::PHASE_BATTLE_START | C::PHASE_BATTLE_STEP | C::PHASE_DAMAGE | C::PHASE_DAMAGE_CAL | C::PHASE_BATTLE => {
            "battle"
        }
        C::PHASE_MAIN2 => "main2",
        C::PHASE_END => "end",
        _ => return None,
    })
}

fn phase_label(phase: &str) -> &str {
    match phase {
        "draw" => "Draw Phase",
        "standby" => "Standby Phase",
        "main1" => "Main Phase 1",
        "battle" => "Battle Phase",
        "main2" => "Main Phase 2",
        "end" => "End Phase",
        other => other,
    }
}

fn description(body: &Body) -> Option<u64> {
    match body {
        Body::SelectEffectYn { description, .. } | Body::SelectYesNo { description, .. } => Some(*description),
        _ => None,
    }
}

fn message_card_ref(body: &Body) -> Option<&M::CardRef> {
    match body {
        Body::SelectEffectYn { card, .. } => Some(card),
        _ => None,
    }
}

fn msg_min_max(body: &Body) -> (i64, i64) {
    match body {
        Body::SelectCard { min, max, .. }
        | Body::SelectUnselectCard { min, max, .. }
        | Body::SelectTribute { min, max, .. }
        | Body::SelectSum { min, max, .. } => (*min as i64, *max as i64),
        _ => (0, 0),
    }
}

fn prompt_kind(body: &Body) -> &'static str {
    match body {
        Body::SelectIdleCmd { .. } => "idle",
        Body::SelectBattleCmd { .. } => "battle",
        Body::SelectChain { .. } => "chain",
        Body::SelectEffectYn { .. } => "effectyn",
        Body::SelectYesNo { .. } => "yesno",
        Body::SelectOption { .. } => "option",
        Body::SelectCard { .. } | Body::SelectTribute { .. } | Body::SelectSum { .. }
        | Body::SelectUnselectCard { .. } => "card",
        Body::SelectPlace { .. } => "place",
        Body::SelectPosition { .. } => "position",
        Body::AnnounceNumber { .. } | Body::AnnounceRace { .. } | Body::AnnounceAttrib { .. }
        | Body::AnnounceCard { .. } | Body::RockPaperScissors { .. } => "announce",
        Body::SortCards { .. } => "sort",
        Body::SelectCounter { .. } => "counter",
        _ => "card",
    }
}

pub(crate) fn loc_word(r: Option<&M::CardRef>) -> String {
    r.map(|r| loc_word_loc(r.location)).unwrap_or_default()
}

pub(crate) fn loc_word_loc(location: u32) -> String {
    let s = match location {
        C::LOCATION_HAND => "hand",
        C::LOCATION_GRAVE => "GY",
        C::LOCATION_REMOVED => "banishment",
        C::LOCATION_EXTRA => "Extra Deck",
        C::LOCATION_DECK => "Deck",
        C::LOCATION_MZONE => "Monster Zone",
        C::LOCATION_SZONE => "Spell & Trap Zone",
        C::LOCATION_OVERLAY => "Xyz Materials",
        _ => "",
    };
    s.to_string()
}

/// `view._fill`: fill EDOPro printf placeholders, never leaking a raw `%`.
pub(crate) fn fill(text: &str, name: &str, loc: &str) -> String {
    if text.is_empty() || !text.contains('%') {
        return text.to_string();
    }
    let args = [name, loc];
    let mut ai = 0usize;
    let mut out = String::with_capacity(text.len());
    let b: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == '%' {
            let mut j = i + 1;
            if j < b.len() && b[j] == 'l' {
                j += 1;
            }
            if j < b.len() {
                let c = b[j];
                if matches!(c, 's' | 'd' | 'u' | 'x' | 'X') {
                    if ai < args.len() {
                        out.push_str(args[ai]);
                        ai += 1;
                    }
                    i = j + 1;
                    continue;
                }
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out.replace("\"\"", "").replace("[]", "").replace("  ", " ").trim().to_string()
}

fn zone_label(arg: i64, human_seat: i32, selecting_player: i32) -> String {
    let (rel, loc, seq) = zone_bit(arg);
    let mine = (selecting_player ^ rel) == human_seat;
    let who = if mine { "Your" } else { "Opponent's" };
    if loc == C::LOCATION_MZONE as i64 {
        if seq >= 5 {
            let side = if (seq == 5) == mine { "left" } else { "right" };
            return format!("Extra Monster Zone ({side})");
        }
        let col = if mine { seq + 1 } else { 5 - seq };
        return format!("{who} Monster Zone {col}");
    }
    if seq <= 4 {
        let col = if mine { seq + 1 } else { 5 - seq };
        return format!("{who} Spell & Trap Zone {col}");
    }
    if seq == 5 {
        return format!("{who} Field Zone");
    }
    let side = if (seq == 6) == mine { "left" } else { "right" };
    format!("{who} Pendulum Zone ({side})")
}

pub(crate) fn zone_bit(zone: i64) -> (i32, i64, i64) {
    let rel = (zone >> 4) as i32;
    let z = zone & 15;
    if z >= 8 {
        (rel, C::LOCATION_SZONE as i64, z - 8)
    } else {
        (rel, C::LOCATION_MZONE as i64, z)
    }
}

fn verb(opt: &ygoenv_sys::Opt) -> &'static str {
    match opt.kind {
        7 => {
            if opt.arg != 0 {
                "direct"
            } else {
                "attack"
            }
        }
        _ => match opt.kind {
            1 => "normal",
            2 => "special",
            3 => "position",
            4 => "set",
            5 => "set_spell",
            6 => "activate",
            8 => "to_battle",
            9 => "to_main2",
            10 => "end_turn",
            11 => "pass",
            12 => "yes",
            13 => "no",
            14 => "option",
            15 => "select",
            16 => "unselect",
            17 => "finish",
            18 => "cancel",
            19 => "place",
            20 => "position",
            21 | 22 | 23 | 24 => "announce",
            25 => "sort",
            26 => "select",
            27 => "select",
            _ => "option",
        },
    }
}

fn where_ref(has_card: bool, r: &EnvRef, human_seat: i32) -> Option<Value> {
    if !has_card {
        return None;
    }
    let zone = if r.location == C::LOCATION_MZONE {
        if r.sequence < 5 { "mzone" } else { "emz" }
    } else if r.location == C::LOCATION_SZONE {
        if r.sequence == 5 { "field" } else { "szone" }
    } else {
        match r.location {
            C::LOCATION_HAND => "hand",
            C::LOCATION_GRAVE => "gy",
            C::LOCATION_REMOVED => "banished",
            C::LOCATION_EXTRA => "extra",
            C::LOCATION_DECK => "deck",
            C::LOCATION_OVERLAY => "extra",
            _ => "deck",
        }
    };
    Some(json!({
        "side": if r.controller as i32 == human_seat { "you" } else { "bot" },
        "zone": zone,
        "seq": r.sequence,
    }))
}

fn label(
    opt: &ygoenv_sys::Opt,
    card: Option<&Value>,
    where_: Option<&Value>,
    effect: Option<&str>,
    human_seat: i32,
    selecting_player: i32,
) -> String {
    let mut nm = card
        .and_then(|c| c.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if nm.is_empty() && card.is_some() {
        nm = "a face-down card".to_string();
    }
    let k = opt.kind;
    match k {
        11 => "Pass".into(),
        12 => {
            if nm.is_empty() { "Yes".into() } else { format!("Yes — use {nm}") }
        }
        13 => {
            if nm.is_empty() { "No".into() } else { format!("No — decline {nm}") }
        }
        17 => "Done".into(),
        18 => "Cancel".into(),
        8 => "Go to Battle Phase".into(),
        9 => "Go to Main Phase 2".into(),
        10 => "End the turn".into(),
        7 => {
            if opt.arg != 0 { format!("Direct attack with {nm}") } else { format!("Attack with {nm}") }
        }
        1 => format!("Normal Summon {nm}"),
        2 => format!("Special Summon {nm}"),
        4 => format!("Set {nm} (monster)"),
        5 => format!("Set {nm}"),
        6 => {
            let s = format!("Activate {nm}");
            match effect {
                Some(e) if !e.is_empty() => format!("{s} (\"{e}\")"),
                _ => s,
            }
        }
        3 => format!("Change {nm}'s position"),
        20 => match opt.arg {
            1 => "Attack Position".into(),
            2 => "Face-down Attack".into(),
            4 => "Defense Position".into(),
            8 => "Face-down Defense".into(),
            _ => "Choose a position".into(),
        },
        21 => format!("Declare {}", opt.arg),
        22 | 23 => format!("Declare ({:#x})", opt.arg),
        24 => {
            if nm.is_empty() { "Declare a card".into() } else { nm }
        }
        25 => format!("Put {nm} next"),
        14 => match effect {
            Some(e) if !e.is_empty() => e.to_string(),
            _ => format!("Option {}", opt.arg + 1),
        },
        15 => {
            let loc = where_
                .and_then(|w| w.get("zone"))
                .and_then(Value::as_str)
                .filter(|z| *z != "hand" && *z != "deck")
                .map(|z| format!(" ({z})"))
                .unwrap_or_default();
            if nm.is_empty() { "Select a face-down card".into() } else { format!("{nm}{loc}") }
        }
        16 => {
            if nm.is_empty() { "Unselect".into() } else { format!("Unselect {nm}") }
        }
        19 => zone_label(opt.arg, human_seat, selecting_player),
        26 => {
            if nm.is_empty() { "Remove a counter".into() } else { format!("Remove from {nm}") }
        }
        _ => {
            let base = kind_lower(Some(k));
            format!("{base} {nm}").trim().to_string()
        }
    }
}

pub(crate) const KIND_PLACE: i32 = 19;

pub(crate) fn kind_of(k: i32) -> Option<i32> {
    (1..=27).contains(&k).then_some(k)
}

pub(crate) fn kind_lower(k: Option<i32>) -> &'static str {
    match k {
        Some(1) => "summon",
        Some(2) => "spsummon",
        Some(3) => "repos",
        Some(4) => "mset",
        Some(5) => "sset",
        Some(6) => "activate",
        Some(7) => "attack",
        Some(8) => "to_battle",
        Some(9) => "to_main2",
        Some(10) => "to_end",
        Some(11) => "pass",
        Some(12) => "yes",
        Some(13) => "no",
        Some(14) => "option",
        Some(15) => "select",
        Some(16) => "unselect",
        Some(17) => "finish",
        Some(18) => "cancel",
        Some(19) => "place",
        Some(20) => "position",
        Some(21) => "number",
        Some(22) => "attribute",
        Some(23) => "race",
        Some(24) => "declare",
        Some(25) => "sort",
        Some(26) => "counter",
        Some(27) => "rps",
        _ => "",
    }
}

fn say(k: i32) -> Option<&'static str> {
    Some(match k {
        1 => "Normal Summons",
        2 => "Special Summons",
        3 => "changes the position of",
        4 => "sets a monster",
        5 => "sets a spell/trap",
        6 => "activates",
        7 => "attacks with",
        8 => "enters the Battle Phase",
        9 => "goes to Main Phase 2",
        10 => "ends the turn",
        12 => "uses the effect of",
        13 => "declines the effect of",
        14 => "chooses",
        15 => "selects",
        16 => "unselects",
        17 => "finishes the selection",
        18 => "cancels",
        19 => "places in",
        20 => "chooses position for",
        21 => "declares the number",
        22 => "declares the attribute",
        23 => "declares the type",
        24 => "declares the card",
        25 => "orders",
        26 => "removes counters from",
        27 => "plays rock-paper-scissors",
        11 => "passes",
        _ => return None,
    })
}

fn hidden_say(kind: Option<i32>) -> &'static str {
    match kind {
        Some(6) => "activates a face-down card",
        Some(15) => "selects a card",
        Some(16) => "selects a card",
        Some(1) => "summons a monster",
        Some(2) => "special summons a monster",
        Some(4) => "sets a monster",
        Some(5) => "sets a spell/trap",
        Some(3) => "changes a monster's position",
        Some(20) => "chooses a position",
        Some(12) => "uses a hidden effect",
        Some(13) => "declines a hidden effect",
        Some(14) => "chooses an effect",
        Some(7) => "attacks with a hidden monster",
        Some(19) => "chooses a zone",
        _ => "makes a hidden move",
    }
}

fn needs_object(k: i32) -> bool {
    matches!(k, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 12 | 13 | 14 | 15 | 16 | 20 | 25 | 26)
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// The zone layout view: exactly `hi - lo` slots, empty ones `None` (Python's list slice + pad).
pub(crate) fn pad_slots(cards: &[Option<CardState>], lo: usize, hi: usize) -> Vec<Option<CardState>> {
    let mut out: Vec<Option<CardState>> =
        cards.get(lo..hi.min(cards.len())).map(|s| s.to_vec()).unwrap_or_default();
    out.resize(hi - lo, None);
    out
}

// ---------------------------------------------------------------- board index

pub(crate) struct Board {
    index: HashMap<(u32, u32, u32, u32), (CardState, bool)>,
}

impl Board {
    fn new(source: &dyn ViewSource, me: i32) -> Board {
        let mut q: HashMap<(i32, u32), Vec<Option<CardState>>> = HashMap::new();
        for p in 0..2i32 {
            for loc in Q::QUERY_LOCATIONS {
                q.insert((p, loc), source.query_location(p, loc));
            }
        }
        let empty = Vec::new();
        let get = |p: i32, loc: u32| q.get(&(p, loc)).unwrap_or(&empty);
        let mut rows: Vec<(CardState, bool)> = Vec::new();
        for p in [me, 1 - me] {
            let mine = p == me;
            for loc in [
                C::LOCATION_HAND,
                C::LOCATION_MZONE,
                C::LOCATION_SZONE,
                C::LOCATION_GRAVE,
                C::LOCATION_REMOVED,
                C::LOCATION_EXTRA,
            ] {
                for c in get(p, loc) {
                    let Some(c) = c else { continue };
                    if loc == C::LOCATION_EXTRA && !mine && c.position & C::POS_FACEUP == 0 {
                        continue;
                    }
                    rows.push((c.clone(), mine || public(c)));
                }
            }
            for c in get(p, C::LOCATION_MZONE) {
                let Some(c) = c else { continue };
                for (i, code) in c.overlay.iter().enumerate() {
                    let mut overlay = CardState {
                        controller: p as u32,
                        location: C::LOCATION_OVERLAY,
                        sequence: c.sequence,
                        code: *code,
                        public: true,
                        overlay_index: i as u32,
                        ..Default::default()
                    };
                    overlay.overlay.clear();
                    rows.push((overlay, true));
                }
            }
        }
        let mut deck: Vec<CardState> = get(me, C::LOCATION_DECK).iter().flatten().cloned().collect();
        deck.sort_by_key(|c| (c.code, c.sequence));
        rows.extend(deck.into_iter().map(|c| (c, true)));
        rows.truncate(MAX_CARDS);
        let mut index = HashMap::new();
        for (c, visible) in rows {
            index.insert((c.controller, c.location, c.sequence, c.overlay_index), (c, visible));
        }
        Board { index }
    }

    fn row_of_env(&self, r: &EnvRef) -> Option<&(CardState, bool)> {
        let key = if r.location & C::LOCATION_OVERLAY != 0 {
            (r.controller, C::LOCATION_OVERLAY, r.sequence, r.position)
        } else {
            (r.controller, r.location, r.sequence, 0)
        };
        self.index.get(&key)
    }

    /// Codes this viewer can see (`Board.visible_codes`).
    pub(crate) fn visible_codes(&self) -> std::collections::BTreeSet<u32> {
        self.index.values().filter(|(_, visible)| *visible).map(|(c, _)| c.code).filter(|c| *c != 0).collect()
    }

    fn row_of_opt(&self, opt: &ygoenv_sys::Opt) -> Option<&(CardState, bool)> {
        if opt.has_card == 0 {
            return None;
        }
        let key = if opt.location & C::LOCATION_OVERLAY != 0 {
            (opt.controller, C::LOCATION_OVERLAY, opt.sequence, opt.position)
        } else {
            (opt.controller, opt.location, opt.sequence, 0)
        };
        self.index.get(&key)
    }

    /// `Board.option_code`: the code the viewer may see for an option's card.
    pub(crate) fn option_code(&self, opt: &ygoenv_sys::Opt, me: i32) -> i32 {
        // `opt_code` is the option's semantic code (Python `Option.code`); `opt.code` is
        // the option card's code, filled only when `has_card`.
        if opt.has_card == 0 {
            return opt.opt_code as i32;
        }
        if let Some((card, visible)) = self.row_of_opt(opt) {
            if !*visible {
                return 0;
            }
            if opt.opt_code != 0 {
                return opt.opt_code as i32;
            }
            return card.code as i32;
        }
        let own_or_revealed = opt.controller as i32 == me
            || opt.location == C::LOCATION_DECK
            || opt.location == C::LOCATION_EXTRA;
        if own_or_revealed { opt.opt_code as i32 } else { 0 }
    }

    fn card_is_public(c: &CardState) -> bool {
        if c.location == C::LOCATION_GRAVE || c.location == C::LOCATION_OVERLAY {
            return true;
        }
        c.public || (c.position & C::POS_FACEUP != 0 && c.location != C::LOCATION_HAND)
    }

    /// `history_entry(option)` from the viewer's board: the option row, actor and public flag.
    pub(crate) fn history_entry(
        &self, opt: &ygoenv_sys::Opt, me: i32, msg_type: i32, step: i32, turn: i32, phase: i32,
    ) -> crate::stream::HistoryRow {
        let has_card = opt.has_card != 0;
        let mut row = [0i32; 10];
        row[0] = opt.kind;
        row[1] = -1;
        row[2] = self.option_code(opt, me);
        row[3] = ((opt.desc >> 20) & 0xFFFF_FFFF) as i32;
        row[4] = (opt.desc & 0xF_FFFF) as i32;
        row[5] = opt.arg as u64 as u32 as i32;
        row[6] = msg_type;
        row[7] = if has_card { crate::stream::loc_id(opt.location) } else { 0 };
        row[8] = if has_card { (opt.controller as i32) ^ me } else { -1 };
        row[9] = step;
        const REVEALING: [i32; 5] = [6, 1, 2, 3, 7];
        let public = if REVEALING.contains(&opt.kind) {
            true
        } else if !has_card {
            !(opt.kind == 20 && (opt.arg as u32 & C::POS_FACEDOWN) != 0)
        } else if opt.controller as i32 != me {
            true
        } else {
            self.row_of_opt(opt).map(|(c, _)| Self::card_is_public(c)).unwrap_or(false)
        };
        crate::stream::HistoryRow { row, actor: me, turn, phase, public }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::DuelStats;

    fn ctx() -> Ctx {
        Ctx::new(Texts::default(), std::collections::BTreeMap::new(), 8000)
    }

    /// A minimal in-memory `ViewSource`: the builders must read only through the trait.
    struct FakeSource {
        locs: HashMap<(i32, u32), Vec<Option<CardState>>>,
        trace: Vec<Vec<u8>>,
        history: Vec<[i32; 14]>,
    }

    impl ViewSource for FakeSource {
        fn duel_view(&self) -> DuelView {
            DuelView { turn: 2, phase: C::PHASE_MAIN1 as i32, turn_player: 0, first_player: 0, winner: 0 }
        }
        fn query_location(&self, player: i32, location: u32) -> Vec<Option<CardState>> {
            self.locs.get(&(player, location)).cloned().unwrap_or_default()
        }
        fn field_state(&self) -> ((i64, i64), Vec<ChainLink>) {
            ((6400, 8000), Vec::new())
        }
        fn trace(&self) -> Vec<Vec<u8>> {
            self.trace.clone()
        }
        fn history(&self) -> Vec<[i32; 14]> {
            self.history.clone()
        }
        fn decision(&self) -> Option<DecisionState> {
            None
        }
        fn pending_msg(&self) -> Vec<u8> {
            Vec::new()
        }
        fn done(&self) -> bool {
            false
        }
        fn player(&self) -> i32 {
            -1
        }
    }

    #[test]
    fn builders_read_the_view_source() {
        let mut locs = HashMap::new();
        locs.insert(
            (0, C::LOCATION_HAND),
            vec![Some(CardState {
                controller: 0,
                location: C::LOCATION_HAND,
                sequence: 0,
                code: 7,
                position: C::POS_FACEUP,
                ..Default::default()
            })],
        );
        let mut row = [0i32; 14];
        row[0] = 1;
        row[2] = 7;
        row[10] = 0;
        row[13] = 1;
        let src = FakeSource { locs, trace: Vec::new(), history: vec![row] };
        let view = ctx().build_view(&src, 0, false);
        assert_eq!(view["turn"], 2);
        assert_eq!(view["you"]["lp"], 6400);
        assert_eq!(view["you"]["hand"][0]["code"], 7);
        let events = ctx().build_events(&src, 0, 0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["kind"], "summon");
    }

    fn loc_info(out: &mut Vec<u8>, ctrl: u8, loc: u32, seq: u32, pos: u32) {
        out.push(ctrl);
        out.push(loc as u8);
        out.extend(seq.to_le_bytes());
        out.extend(pos.to_le_bytes());
    }

    fn draw(p: u8, cards: &[(u32, u32)]) -> Vec<u8> {
        let mut raw = vec![C::MSG_DRAW as u8, p];
        raw.extend((cards.len() as u32).to_le_bytes());
        for (code, pos) in cards {
            raw.extend(code.to_le_bytes());
            raw.extend(pos.to_le_bytes());
        }
        raw
    }

    fn move_msg(code: u32, from: (u8, u32, u32, u32), to: (u8, u32, u32, u32), reason: u32) -> Vec<u8> {
        let mut raw = vec![C::MSG_MOVE as u8];
        raw.extend(code.to_le_bytes());
        loc_info(&mut raw, from.0, from.1, from.2, from.3);
        loc_info(&mut raw, to.0, to.1, to.2, to.3);
        raw.extend(reason.to_le_bytes());
        raw
    }

    fn chaining(code: u32, ctrl: u8, loc: u32, seq: u32, pos: u32, desc: u64, link: u32) -> Vec<u8> {
        let mut raw = vec![C::MSG_CHAINING as u8];
        raw.extend(code.to_le_bytes());
        loc_info(&mut raw, ctrl, loc, seq, pos);
        raw.extend([0u8, 0u8]);
        raw.extend(0u32.to_le_bytes());
        raw.extend(desc.to_le_bytes());
        raw.extend(link.to_le_bytes());
        raw
    }

    #[test]
    fn state_card_keeps_overlays_and_link_fields() {
        let c = CardState {
            controller: 0,
            location: C::LOCATION_MZONE,
            sequence: 2,
            code: 42,
            position: C::POS_FACEUP_ATTACK,
            attack: 1500,
            defense: 1200,
            level: 4,
            link: 2,
            overlay: vec![7, 8],
            ..Default::default()
        };
        let v = ctx().state_card(&c, true);
        assert_eq!(v["code"], 42);
        assert_eq!(v["link"], 2);
        assert_eq!(v["faceup"], true);
        assert_eq!(v["overlays"].as_array().unwrap().len(), 2);
        assert_eq!(v["overlays"][0]["uid"], "0:0x80:0:7");
        assert_eq!(v["overlays"][0]["code"], 7);
    }

    #[test]
    fn hidden_opponent_cards_never_leak() {
        let ctx = ctx();
        let trace = vec![draw(1, &[(123, 0)]), draw(0, &[(123, 0)])];
        let (cues, len, _) = cues_for_trace(&ctx, &trace, 0, 0, false, 8000);
        assert_eq!(len, 2);
        assert_eq!(cues[0]["cards"][0]["code"], 0);
        assert_eq!(cues[0]["cards"][0]["name"], "");
        assert_eq!(cues[1]["cards"][0]["code"], 123);
        assert_eq!(cues[1]["cards"][0]["name"], "#123");
    }

    #[test]
    fn face_down_set_and_empty_zone_slots() {
        let ctx = ctx();
        let raw = move_msg(
            55,
            (1, C::LOCATION_HAND, 0, 0),
            (1, C::LOCATION_SZONE, 3, C::POS_FACEDOWN_DEFENSE),
            0,
        );
        let (cues, _, _) = cues_for_trace(&ctx, &[raw], 0, 0, false, 8000);
        assert_eq!(cues[0]["t"], "move");
        assert_eq!(cues[0]["card"]["code"], 0);
        assert_eq!(cues[0]["reason"], "set");
        assert_eq!(cues[0]["to"]["loc"], "szone");
        assert_eq!(cues[0]["to"]["seq"], 3);
        let hidden = CardState {
            controller: 1,
            location: C::LOCATION_SZONE,
            sequence: 3,
            code: 55,
            position: C::POS_FACEDOWN_DEFENSE,
            ..Default::default()
        };
        let v = ctx.state_card(&hidden, false);
        assert_eq!(v["code"], 0);
        assert_eq!(v["name"], Value::Null);
        assert_eq!(v["pos"], "set");
    }

    #[test]
    fn place_event_is_public_only_when_announced() {
        let ctx = ctx();
        let mut row = [0i32; 14];
        row[0] = 19;
        row[2] = 12345;
        row[10] = 0;
        row[13] = 1;
        let e = ctx.event_from_row(1, &row, 0);
        assert_eq!(e["kind"], "place");
        assert_eq!(e["text"], "Places #12345");
        row[10] = 1;
        row[13] = 0;
        let e = ctx.event_from_row(2, &row, 0);
        assert_eq!(e["text"], "Chooses a zone");
        assert_eq!(e["card"], Value::Null);
    }

    #[test]
    fn chain_cues_and_stats_negation() {
        let ctx = ctx();
        let trace = vec![
            chaining(100, 1, C::LOCATION_HAND, 0, 0, 5, 1),
            vec![C::MSG_CHAIN_SOLVING as u8, 1],
            vec![C::MSG_CHAIN_NEGATED as u8, 1],
            vec![C::MSG_CHAIN_END as u8],
        ];
        let (cues, _, _) = cues_for_trace(&ctx, &trace, 0, 0, false, 8000);
        let types: Vec<&str> = cues.iter().map(|c| c["t"].as_str().unwrap()).collect();
        assert_eq!(types, ["activate", "chain_solving", "chain_negated", "chain_end"]);
        assert_eq!(cues[0]["card"]["code"], 100);
        assert_eq!(cues[0]["link"], 1);
        let mut st = DuelStats::new(0, 8000);
        st.feed_cues(&cues);
        let msg = st.stats_message(false);
        assert_eq!(msg["neg_were_bot"], 1);
        assert_eq!(msg["neg_bot"], 1);
        assert_eq!(msg["moments"][0]["kind"], "negation");
        assert_eq!(msg["moments"][0]["text"], "Bot's #100 was negated");
    }

    #[test]
    fn chain_solved_records_the_activation_as_used() {
        let ctx = ctx();
        let trace = vec![
            chaining(100, 0, C::LOCATION_HAND, 0, 0, 0, 1),
            vec![C::MSG_CHAIN_SOLVING as u8, 1],
            vec![C::MSG_CHAIN_SOLVED as u8, 1],
            vec![C::MSG_CHAIN_END as u8],
        ];
        let (cues, _, _) = cues_for_trace(&ctx, &trace, 0, 0, false, 8000);
        let mut st = DuelStats::new(0, 8000);
        st.feed_cues(&cues);
        let msg = st.stats_message(false);
        assert_eq!(msg["you_used"][0]["code"], 100);
        assert_eq!(msg["you_used"][0]["count"], 1);
        st.note_chain_window(true, true);
        assert_eq!(st.stats_message(false)["windows_used"], 1);
    }

    #[test]
    fn fill_zone_bits_and_labels() {
        assert_eq!(fill("a  b", "", ""), "a  b");
        assert_eq!(fill("a  %ls b", "x", ""), "a x b");
        assert_eq!(fill("x %ls y", "N", ""), "x N y");
        assert_eq!(fill("no placeholders", "N", "hand"), "no placeholders");
        assert_eq!(zone_bit(0), (0, C::LOCATION_MZONE as i64, 0));
        assert_eq!(zone_bit(8), (0, C::LOCATION_SZONE as i64, 0));
        assert_eq!(zone_bit(16 + 5), (1, C::LOCATION_MZONE as i64, 5));
        assert_eq!(zone_label(0, 0, 0), "Your Monster Zone 1");
        assert_eq!(zone_label(16, 0, 0), "Opponent's Monster Zone 5");
        assert_eq!(zone_label(8, 0, 0), "Your Spell & Trap Zone 1");
        assert_eq!(zone_label(13, 0, 0), "Your Field Zone");
        assert_eq!(zone_label(14, 0, 0), "Your Pendulum Zone (left)");
    }

    #[test]
    fn empty_zones_pad_to_the_full_layout() {
        assert_eq!(pad_slots(&[], 0, 5), vec![None, None, None, None, None]);
        let c = CardState { code: 9, ..Default::default() };
        let cards = vec![None, Some(c.clone()), None];
        let out = pad_slots(&cards, 0, 5);
        assert_eq!(out.len(), 5);
        assert!(out[0].is_none());
        assert_eq!(out[1].as_ref().unwrap().code, 9);
        assert!(out[3].is_none() && out[4].is_none());
        assert_eq!(pad_slots(&cards, 5, 7), vec![None, None]);
    }
}
