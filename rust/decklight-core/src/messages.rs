//! Parse the core's message buffer (port of `src/decklight/engine/messages.py`, behaviour for behaviour).
//!
//! `OCG_DuelGetMessage` returns `[u32 size][size bytes]...`; each payload is one message whose first byte is the
//! MSG_* type. Every message that awaits a response (the SELECT_*/ANNOUNCE_*/SORT_* family) is parsed in full; of the
//! rest, only the few that drive the loop (WIN, NEW_TURN, NEW_PHASE, HINT) are decoded. Everything else stays a plain
//! `Message(type, raw)`.

use std::fmt::Write as _;

use crate::constants as C;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardRef {
    pub code: u32,
    pub controller: u8,
    pub location: u32,
    pub sequence: u32,
    pub position: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChainOption {
    pub card: CardRef,
    pub description: u64,
    pub client_mode: u8,
}

/// The decoded body of one message. `Plain` is Python's bare `Message(type, raw)`.
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Win { player: u8, reason: u8 },
    NewTurn { player: u8 },
    NewPhase { phase: u16 },
    Hint { hint_type: u8, player: u8, data: u64 },
    SelectIdleCmd {
        player: u8,
        summonable: Vec<CardRef>,
        spsummonable: Vec<CardRef>,
        repositionable: Vec<CardRef>,
        msetable: Vec<CardRef>,
        ssetable: Vec<CardRef>,
        activatable: Vec<ChainOption>,
        can_battle_phase: bool,
        can_end_phase: bool,
        can_shuffle: bool,
    },
    SelectBattleCmd {
        player: u8,
        activatable: Vec<ChainOption>,
        attackable: Vec<(CardRef, bool)>,
        can_main2: bool,
        can_end_phase: bool,
    },
    SelectEffectYn { player: u8, card: CardRef, description: u64 },
    SelectYesNo { player: u8, description: u64 },
    SelectOption { player: u8, options: Vec<u64> },
    SelectCard { player: u8, cancelable: bool, min: u32, max: u32, cards: Vec<CardRef> },
    SelectUnselectCard {
        player: u8,
        finishable: bool,
        cancelable: bool,
        min: u32,
        max: u32,
        selectable: Vec<CardRef>,
        unselectable: Vec<CardRef>,
    },
    SelectChain {
        player: u8,
        spe_count: u8,
        forced: bool,
        hint_timing: u32,
        hint_timing_other: u32,
        chains: Vec<ChainOption>,
    },
    SelectPlace { player: u8, count: u8, flag: u32 },
    SelectPosition { player: u8, code: u32, positions: u8 },
    SelectTribute { player: u8, cancelable: bool, min: u32, max: u32, cards: Vec<(CardRef, u8)> },
    SelectCounter { player: u8, counter_type: u16, count: u16, cards: Vec<(CardRef, u16)> },
    SelectSum {
        player: u8,
        at_least: bool,
        acc: u32,
        min: u32,
        max: u32,
        must: Vec<(CardRef, u32)>,
        cards: Vec<(CardRef, u32)>,
    },
    SortCards { player: u8, cards: Vec<CardRef> },
    AnnounceRace { player: u8, count: u8, available: u64 },
    AnnounceAttrib { player: u8, count: u8, available: u32 },
    AnnounceCard { player: u8, opcodes: Vec<u64> },
    AnnounceNumber { player: u8, options: Vec<u64> },
    RockPaperScissors { player: u8 },
    Plain,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub cls: &'static str,
    pub type_: u32,
    pub raw: Vec<u8>,
    pub body: Body,
}

impl Message {
    pub fn player(&self) -> Option<u8> {
        match &self.body {
            Body::Win { player, .. }
            | Body::NewTurn { player }
            | Body::Hint { player, .. }
            | Body::SelectIdleCmd { player, .. }
            | Body::SelectBattleCmd { player, .. }
            | Body::SelectEffectYn { player, .. }
            | Body::SelectYesNo { player, .. }
            | Body::SelectOption { player, .. }
            | Body::SelectCard { player, .. }
            | Body::SelectUnselectCard { player, .. }
            | Body::SelectChain { player, .. }
            | Body::SelectPlace { player, .. }
            | Body::SelectPosition { player, .. }
            | Body::SelectTribute { player, .. }
            | Body::SelectCounter { player, .. }
            | Body::SelectSum { player, .. }
            | Body::SortCards { player, .. }
            | Body::AnnounceRace { player, .. }
            | Body::AnnounceAttrib { player, .. }
            | Body::AnnounceCard { player, .. }
            | Body::AnnounceNumber { player, .. }
            | Body::RockPaperScissors { player } => Some(*player),
            Body::NewPhase { .. } | Body::Plain => None,
        }
    }

    /// Canonical JSON, byte-identical to Python's `json.dumps(canon_msg(msg), separators=(",", ":"))`.
    pub fn to_canon_json(&self) -> String {
        let mut o = String::new();
        o.push_str("{\"cls\":");
        jstr(&mut o, self.cls);
        o.push_str(",\"type\":");
        let _ = write!(o, "{}", self.type_);
        match &self.body {
            Body::Win { player, reason } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "reason");
                num(&mut o, *reason);
            }
            Body::NewTurn { player } => {
                key(&mut o, "player");
                num(&mut o, *player);
            }
            Body::NewPhase { phase } => {
                key(&mut o, "phase");
                num(&mut o, *phase);
            }
            Body::Hint { hint_type, player, data } => {
                key(&mut o, "hint_type");
                num(&mut o, *hint_type);
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "data");
                num(&mut o, *data);
            }
            Body::SelectIdleCmd {
                player,
                summonable,
                spsummonable,
                repositionable,
                msetable,
                ssetable,
                activatable,
                can_battle_phase,
                can_end_phase,
                can_shuffle,
            } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "summonable");
                cardrefs(&mut o, summonable);
                key(&mut o, "spsummonable");
                cardrefs(&mut o, spsummonable);
                key(&mut o, "repositionable");
                cardrefs(&mut o, repositionable);
                key(&mut o, "msetable");
                cardrefs(&mut o, msetable);
                key(&mut o, "ssetable");
                cardrefs(&mut o, ssetable);
                key(&mut o, "activatable");
                chainopts(&mut o, activatable);
                key(&mut o, "can_battle_phase");
                boolean(&mut o, *can_battle_phase);
                key(&mut o, "can_end_phase");
                boolean(&mut o, *can_end_phase);
                key(&mut o, "can_shuffle");
                boolean(&mut o, *can_shuffle);
            }
            Body::SelectBattleCmd { player, activatable, attackable, can_main2, can_end_phase } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "activatable");
                chainopts(&mut o, activatable);
                key(&mut o, "attackable");
                o.push('[');
                for (i, (card, direct)) in attackable.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    o.push('[');
                    cardref(&mut o, card);
                    o.push(',');
                    boolean(&mut o, *direct);
                    o.push(']');
                }
                o.push(']');
                key(&mut o, "can_main2");
                boolean(&mut o, *can_main2);
                key(&mut o, "can_end_phase");
                boolean(&mut o, *can_end_phase);
            }
            Body::SelectEffectYn { player, card, description } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "card");
                cardref(&mut o, card);
                key(&mut o, "description");
                num(&mut o, *description);
            }
            Body::SelectYesNo { player, description } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "description");
                num(&mut o, *description);
            }
            Body::SelectOption { player, options } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "options");
                u64s(&mut o, options);
            }
            Body::SelectCard { player, cancelable, min, max, cards } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "cancelable");
                boolean(&mut o, *cancelable);
                key(&mut o, "min");
                num(&mut o, *min);
                key(&mut o, "max");
                num(&mut o, *max);
                key(&mut o, "cards");
                cardrefs(&mut o, cards);
            }
            Body::SelectUnselectCard {
                player,
                finishable,
                cancelable,
                min,
                max,
                selectable,
                unselectable,
            } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "finishable");
                boolean(&mut o, *finishable);
                key(&mut o, "cancelable");
                boolean(&mut o, *cancelable);
                key(&mut o, "min");
                num(&mut o, *min);
                key(&mut o, "max");
                num(&mut o, *max);
                key(&mut o, "selectable");
                cardrefs(&mut o, selectable);
                key(&mut o, "unselectable");
                cardrefs(&mut o, unselectable);
            }
            Body::SelectChain {
                player,
                spe_count,
                forced,
                hint_timing,
                hint_timing_other,
                chains,
            } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "spe_count");
                num(&mut o, *spe_count);
                key(&mut o, "forced");
                boolean(&mut o, *forced);
                key(&mut o, "hint_timing");
                num(&mut o, *hint_timing);
                key(&mut o, "hint_timing_other");
                num(&mut o, *hint_timing_other);
                key(&mut o, "chains");
                chainopts(&mut o, chains);
            }
            Body::SelectPlace { player, count, flag } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "count");
                num(&mut o, *count);
                key(&mut o, "flag");
                num(&mut o, *flag);
            }
            Body::SelectPosition { player, code, positions } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "code");
                num(&mut o, *code);
                key(&mut o, "positions");
                num(&mut o, *positions);
            }
            Body::SelectTribute { player, cancelable, min, max, cards } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "cancelable");
                boolean(&mut o, *cancelable);
                key(&mut o, "min");
                num(&mut o, *min);
                key(&mut o, "max");
                num(&mut o, *max);
                key(&mut o, "cards");
                cardrefs_with_num(&mut o, cards);
            }
            Body::SelectCounter { player, counter_type, count, cards } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "counter_type");
                num(&mut o, *counter_type);
                key(&mut o, "count");
                num(&mut o, *count);
                key(&mut o, "cards");
                cardrefs_with_num(&mut o, cards);
            }
            Body::SelectSum { player, at_least, acc, min, max, must, cards } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "at_least");
                boolean(&mut o, *at_least);
                key(&mut o, "acc");
                num(&mut o, *acc);
                key(&mut o, "min");
                num(&mut o, *min);
                key(&mut o, "max");
                num(&mut o, *max);
                key(&mut o, "must");
                cardrefs_with_num(&mut o, must);
                key(&mut o, "cards");
                cardrefs_with_num(&mut o, cards);
            }
            Body::SortCards { player, cards } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "cards");
                cardrefs(&mut o, cards);
            }
            Body::AnnounceRace { player, count, available } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "count");
                num(&mut o, *count);
                key(&mut o, "available");
                num(&mut o, *available);
            }
            Body::AnnounceAttrib { player, count, available } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "count");
                num(&mut o, *count);
                key(&mut o, "available");
                num(&mut o, *available);
            }
            Body::AnnounceCard { player, opcodes } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "opcodes");
                u64s(&mut o, opcodes);
            }
            Body::AnnounceNumber { player, options } => {
                key(&mut o, "player");
                num(&mut o, *player);
                key(&mut o, "options");
                u64s(&mut o, options);
            }
            Body::RockPaperScissors { player } => {
                key(&mut o, "player");
                num(&mut o, *player);
            }
            Body::Plain => {}
        }
        o.push('}');
        o
    }
}

fn key(o: &mut String, k: &str) {
    o.push(',');
    jstr(o, k);
    o.push(':');
}

fn num<T: std::fmt::Display>(o: &mut String, v: T) {
    let _ = write!(o, "{v}");
}

fn boolean(o: &mut String, v: bool) {
    o.push_str(if v { "true" } else { "false" });
}

fn jstr(o: &mut String, s: &str) {
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
}

fn cardref(o: &mut String, c: &CardRef) {
    let _ = write!(o, "[{},{},{},{},{}]", c.code, c.controller, c.location, c.sequence, c.position);
}

fn cardrefs(o: &mut String, cs: &[CardRef]) {
    o.push('[');
    for (i, c) in cs.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        cardref(o, c);
    }
    o.push(']');
}

fn cardrefs_with_num<T: std::fmt::Display>(o: &mut String, cs: &[(CardRef, T)]) {
    o.push('[');
    for (i, (c, n)) in cs.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push('[');
        cardref(o, c);
        o.push(',');
        num(o, n);
        o.push(']');
    }
    o.push(']');
}

fn chainopt(o: &mut String, ch: &ChainOption) {
    o.push('[');
    cardref(o, &ch.card);
    let _ = write!(o, ",{},{}]", ch.description, ch.client_mode);
}

fn chainopts(o: &mut String, cs: &[ChainOption]) {
    o.push('[');
    for (i, c) in cs.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        chainopt(o, c);
    }
    o.push(']');
}

fn u64s(o: &mut String, vs: &[u64]) {
    o.push('[');
    for (i, v) in vs.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        num(o, *v);
    }
    o.push(']');
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Result<Self, String> {
        if b.is_empty() {
            return Err("empty message".into());
        }
        Ok(Self { b, p: 1 })
    }

    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.p)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.remaining() < n {
            return Err(format!("message truncated at {}: need {n}, have {}", self.p, self.remaining()));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
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

    fn boolean(&mut self) -> Result<bool, String> {
        Ok(self.u8()? != 0)
    }

    fn card_info(&mut self) -> Result<CardRef, String> {
        Ok(CardRef {
            code: self.u32()?,
            controller: self.u8()?,
            location: self.u8()? as u32,
            sequence: self.u32()?,
            position: self.u32()?,
        })
    }

    fn card_seq32(&mut self) -> Result<CardRef, String> {
        Ok(CardRef {
            code: self.u32()?,
            controller: self.u8()?,
            location: self.u8()? as u32,
            sequence: self.u32()?,
            position: 0,
        })
    }

    fn card_seq8(&mut self) -> Result<CardRef, String> {
        Ok(CardRef {
            code: self.u32()?,
            controller: self.u8()?,
            location: self.u8()? as u32,
            sequence: self.u8()? as u32,
            position: 0,
        })
    }

    fn chain_short(&mut self) -> Result<ChainOption, String> {
        Ok(ChainOption { card: self.card_seq32()?, description: self.u64()?, client_mode: self.u8()? })
    }

    fn chain_full(&mut self) -> Result<ChainOption, String> {
        Ok(ChainOption { card: self.card_info()?, description: self.u64()?, client_mode: self.u8()? })
    }

    fn list<T>(&mut self, mut f: impl FnMut(&mut Reader<'a>) -> Result<T, String>) -> Result<Vec<T>, String> {
        let n = self.u32()? as usize;
        if n > self.remaining() {
            return Err(format!("list count {n} exceeds {} remaining bytes", self.remaining()));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f(self)?);
        }
        Ok(out)
    }
}

fn parse(t: u32, raw: &[u8]) -> Result<Message, String> {
    let mut r = Reader::new(raw)?;
    let (cls, body) = match t {
        C::MSG_WIN => {
            let player = r.u8()?;
            let reason = r.u8()?;
            ("Win", Body::Win { player, reason })
        }
        C::MSG_NEW_TURN => ("NewTurn", Body::NewTurn { player: r.u8()? }),
        C::MSG_NEW_PHASE => ("NewPhase", Body::NewPhase { phase: r.u16()? }),
        C::MSG_HINT => {
            let hint_type = r.u8()?;
            let player = r.u8()?;
            let data = r.u64()?;
            ("Hint", Body::Hint { hint_type, player, data })
        }
        C::MSG_SELECT_IDLECMD => {
            let player = r.u8()?;
            let summonable = r.list(|r| r.card_seq32())?;
            let spsummonable = r.list(|r| r.card_seq32())?;
            let repositionable = r.list(|r| r.card_seq8())?;
            let msetable = r.list(|r| r.card_seq32())?;
            let ssetable = r.list(|r| r.card_seq32())?;
            let activatable = r.list(|r| r.chain_short())?;
            let can_battle_phase = r.boolean()?;
            let can_end_phase = r.boolean()?;
            let can_shuffle = r.boolean()?;
            (
                "SelectIdleCmd",
                Body::SelectIdleCmd {
                    player,
                    summonable,
                    spsummonable,
                    repositionable,
                    msetable,
                    ssetable,
                    activatable,
                    can_battle_phase,
                    can_end_phase,
                    can_shuffle,
                },
            )
        }
        C::MSG_SELECT_BATTLECMD => {
            let player = r.u8()?;
            let activatable = r.list(|r| r.chain_short())?;
            let attackable = r.list(|r| Ok((r.card_seq8()?, r.boolean()?)))?;
            let can_main2 = r.boolean()?;
            let can_end_phase = r.boolean()?;
            ("SelectBattleCmd", Body::SelectBattleCmd { player, activatable, attackable, can_main2, can_end_phase })
        }
        C::MSG_SELECT_EFFECTYN => {
            let player = r.u8()?;
            let card = r.card_info()?;
            let description = r.u64()?;
            ("SelectEffectYn", Body::SelectEffectYn { player, card, description })
        }
        C::MSG_SELECT_YESNO => {
            let player = r.u8()?;
            let description = r.u64()?;
            ("SelectYesNo", Body::SelectYesNo { player, description })
        }
        C::MSG_SELECT_OPTION => {
            let player = r.u8()?;
            let n = r.u8()? as usize;
            let mut options = Vec::with_capacity(n);
            for _ in 0..n {
                options.push(r.u64()?);
            }
            ("SelectOption", Body::SelectOption { player, options })
        }
        C::MSG_SELECT_CARD => {
            let player = r.u8()?;
            let cancelable = r.boolean()?;
            let min = r.u32()?;
            let max = r.u32()?;
            let cards = r.list(|r| r.card_info())?;
            ("SelectCard", Body::SelectCard { player, cancelable, min, max, cards })
        }
        C::MSG_SELECT_UNSELECT_CARD => {
            let player = r.u8()?;
            let finishable = r.boolean()?;
            let cancelable = r.boolean()?;
            let min = r.u32()?;
            let max = r.u32()?;
            let selectable = r.list(|r| r.card_info())?;
            let unselectable = r.list(|r| r.card_info())?;
            (
                "SelectUnselectCard",
                Body::SelectUnselectCard { player, finishable, cancelable, min, max, selectable, unselectable },
            )
        }
        C::MSG_SELECT_CHAIN => {
            let player = r.u8()?;
            let spe_count = r.u8()?;
            let forced = r.boolean()?;
            let hint_timing = r.u32()?;
            let hint_timing_other = r.u32()?;
            let chains = r.list(|r| r.chain_full())?;
            (
                "SelectChain",
                Body::SelectChain { player, spe_count, forced, hint_timing, hint_timing_other, chains },
            )
        }
        C::MSG_SELECT_PLACE | C::MSG_SELECT_DISFIELD => {
            let player = r.u8()?;
            let count = r.u8()?;
            let flag = r.u32()?;
            ("SelectPlace", Body::SelectPlace { player, count, flag })
        }
        C::MSG_SELECT_POSITION => {
            let player = r.u8()?;
            let code = r.u32()?;
            let positions = r.u8()?;
            ("SelectPosition", Body::SelectPosition { player, code, positions })
        }
        C::MSG_SELECT_TRIBUTE => {
            let player = r.u8()?;
            let cancelable = r.boolean()?;
            let min = r.u32()?;
            let max = r.u32()?;
            let cards = r.list(|r| Ok((r.card_seq32()?, r.u8()?)))?;
            ("SelectTribute", Body::SelectTribute { player, cancelable, min, max, cards })
        }
        C::MSG_SELECT_COUNTER => {
            let player = r.u8()?;
            let counter_type = r.u16()?;
            let count = r.u16()?;
            let cards = r.list(|r| Ok((r.card_seq8()?, r.u16()?)))?;
            ("SelectCounter", Body::SelectCounter { player, counter_type, count, cards })
        }
        C::MSG_SELECT_SUM => {
            let player = r.u8()?;
            let at_least = r.boolean()?;
            let acc = r.u32()?;
            let min = r.u32()?;
            let max = r.u32()?;
            let must = r.list(|r| Ok((r.card_info()?, r.u32()?)))?;
            let cards = r.list(|r| Ok((r.card_info()?, r.u32()?)))?;
            ("SelectSum", Body::SelectSum { player, at_least, acc, min, max, must, cards })
        }
        C::MSG_SORT_CARD | C::MSG_SORT_CHAIN => {
            // SortCards reads CardRef(u32, u8, u32, u32): location is a u32 there, unlike card_info's u8.
            let player = r.u8()?;
            let cards = r.list(|r| {
                Ok(CardRef {
                    code: r.u32()?,
                    controller: r.u8()?,
                    location: r.u32()?,
                    sequence: r.u32()?,
                    position: 0,
                })
            })?;
            ("SortCards", Body::SortCards { player, cards })
        }
        C::MSG_ANNOUNCE_RACE => {
            let player = r.u8()?;
            let count = r.u8()?;
            let available = r.u64()?;
            ("AnnounceRace", Body::AnnounceRace { player, count, available })
        }
        C::MSG_ANNOUNCE_ATTRIB => {
            let player = r.u8()?;
            let count = r.u8()?;
            let available = r.u32()?;
            ("AnnounceAttrib", Body::AnnounceAttrib { player, count, available })
        }
        C::MSG_ANNOUNCE_CARD => {
            let player = r.u8()?;
            let n = r.u8()? as usize;
            let mut opcodes = Vec::with_capacity(n);
            for _ in 0..n {
                opcodes.push(r.u64()?);
            }
            ("AnnounceCard", Body::AnnounceCard { player, opcodes })
        }
        C::MSG_ANNOUNCE_NUMBER => {
            let player = r.u8()?;
            let n = r.u8()? as usize;
            let mut options = Vec::with_capacity(n);
            for _ in 0..n {
                options.push(r.u64()?);
            }
            ("AnnounceNumber", Body::AnnounceNumber { player, options })
        }
        C::MSG_ROCK_PAPER_SCISSORS => ("RockPaperScissors", Body::RockPaperScissors { player: r.u8()? }),
        _ => ("Message", Body::Plain),
    };
    Ok(Message { cls, type_: t, raw: raw.to_vec(), body })
}

pub fn parse_one(raw: &[u8]) -> Result<Message, String> {
    let t = *raw.first().ok_or("empty message")? as u32;
    parse(t, raw)
}

/// The message types that await a response (`messages.SELECT_TYPES`).
pub fn is_select_type(t: u32) -> bool {
    matches!(
        t,
        C::MSG_SELECT_BATTLECMD
            | C::MSG_SELECT_IDLECMD
            | C::MSG_SELECT_EFFECTYN
            | C::MSG_SELECT_YESNO
            | C::MSG_SELECT_OPTION
            | C::MSG_SELECT_CARD
            | C::MSG_SELECT_CHAIN
            | C::MSG_SELECT_PLACE
            | C::MSG_SELECT_POSITION
            | C::MSG_SELECT_TRIBUTE
            | C::MSG_SORT_CHAIN
            | C::MSG_SELECT_COUNTER
            | C::MSG_SELECT_SUM
            | C::MSG_SELECT_DISFIELD
            | C::MSG_SORT_CARD
            | C::MSG_SELECT_UNSELECT_CARD
            | C::MSG_ROCK_PAPER_SCISSORS
            | C::MSG_ANNOUNCE_RACE
            | C::MSG_ANNOUNCE_ATTRIB
            | C::MSG_ANNOUNCE_CARD
            | C::MSG_ANNOUNCE_NUMBER
    )
}

pub const SELECT_TYPES: [u32; 21] = [
    C::MSG_SELECT_BATTLECMD,
    C::MSG_SELECT_IDLECMD,
    C::MSG_SELECT_EFFECTYN,
    C::MSG_SELECT_YESNO,
    C::MSG_SELECT_OPTION,
    C::MSG_SELECT_CARD,
    C::MSG_SELECT_CHAIN,
    C::MSG_SELECT_PLACE,
    C::MSG_SELECT_POSITION,
    C::MSG_SELECT_TRIBUTE,
    C::MSG_SORT_CHAIN,
    C::MSG_SELECT_COUNTER,
    C::MSG_SELECT_SUM,
    C::MSG_SELECT_DISFIELD,
    C::MSG_SORT_CARD,
    C::MSG_SELECT_UNSELECT_CARD,
    C::MSG_ROCK_PAPER_SCISSORS,
    C::MSG_ANNOUNCE_RACE,
    C::MSG_ANNOUNCE_ATTRIB,
    C::MSG_ANNOUNCE_CARD,
    C::MSG_ANNOUNCE_NUMBER,
];

/// Split an `OCG_DuelGetMessage` buffer into raw messages.
pub fn split_buffer(buf: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= buf.len() {
        let size = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + size > buf.len() {
            return Err(format!(
                "truncated message buffer at {pos}: need {size}, have {}",
                buf.len() - pos
            ));
        }
        if size > 0 {
            out.push(buf[pos..pos + size].to_vec());
        }
        pos += size;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32b(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn u64b(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn card_info(code: u32, con: u8, loc: u8, seq: u32, pos: u32) -> Vec<u8> {
        [u32b(code), vec![con, loc], u32b(seq), u32b(pos)].concat()
    }

    #[test]
    fn split_skips_empty_and_rejects_truncated() {
        let mut buf = [u32b(2), vec![0x28, 1], u32b(0), u32b(3), vec![0x29, 1, 0]].concat();
        assert_eq!(split_buffer(&buf).unwrap(), vec![vec![0x28, 1], vec![0x29, 1, 0]]);
        buf = [u32b(5), vec![1, 2]].concat();
        assert!(split_buffer(&buf).is_err());
    }

    #[test]
    fn sort_cards_keeps_u32_location() {
        let raw = [vec![C::MSG_SORT_CARD as u8, 1], u32b(1), u32b(123), vec![0], u32b(0x100), u32b(2)].concat();
        let m = parse_one(&raw).unwrap();
        assert_eq!(m.cls, "SortCards");
        assert_eq!(
            m.to_canon_json(),
            "{\"cls\":\"SortCards\",\"type\":25,\"player\":1,\"cards\":[[123,0,256,2,0]]}"
        );
    }

    #[test]
    fn unknown_type_is_plain() {
        let m = parse_one(&[0x99, 1, 2, 3]).unwrap();
        assert_eq!(m.cls, "Message");
        assert_eq!(m.to_canon_json(), "{\"cls\":\"Message\",\"type\":153}");
        assert!(!is_select_type(0x99));
    }

    #[test]
    fn bool_from_u8_and_nested_sum_lists() {
        let card = card_info(7, 0, 4, 1, 5);
        let raw = [
            vec![C::MSG_SELECT_CARD as u8, 0],
            vec![2],
            u32b(1),
            u32b(2),
            u32b(0),
        ]
        .concat();
        let m = parse_one(&raw).unwrap();
        assert_eq!(
            m.to_canon_json(),
            "{\"cls\":\"SelectCard\",\"type\":15,\"player\":0,\"cancelable\":true,\"min\":1,\"max\":2,\"cards\":[]}"
        );
        let raw = [
            vec![C::MSG_SELECT_SUM as u8, 0, 0],
            u32b(100),
            u32b(1),
            u32b(2),
            u32b(1),
            card.clone(),
            u32b(9),
            u32b(1),
            card,
            u32b(3),
        ]
        .concat();
        let m = parse_one(&raw).unwrap();
        assert_eq!(
            m.to_canon_json(),
            "{\"cls\":\"SelectSum\",\"type\":23,\"player\":0,\"at_least\":false,\"acc\":100,\"min\":1,\"max\":2,\
\"must\":[[[7,0,4,1,5],9]],\"cards\":[[[7,0,4,1,5],3]]}"
        );
    }

    #[test]
    fn announce_number_u64_options() {
        let raw = [vec![C::MSG_ANNOUNCE_NUMBER as u8, 1, 2], u64b(0xFFFF_FFFF_FFFF_FFFF), u64b(5)].concat();
        let m = parse_one(&raw).unwrap();
        assert_eq!(
            m.to_canon_json(),
            "{\"cls\":\"AnnounceNumber\",\"type\":143,\"player\":1,\"options\":[18446744073709551615,5]}"
        );
    }

    #[test]
    fn chain_full_layout_and_short_default_position() {
        // SELECT_CHAIN: player, spe_count, forced, hint_timing, hint_timing_other, chains
        let mut raw = vec![C::MSG_SELECT_CHAIN as u8, 1, 3, 1];
        raw.extend(u32b(0x11));
        raw.extend(u32b(0x22));
        raw.extend(u32b(1));
        raw.extend(card_info(42, 1, 4, 3, 5));
        raw.extend(u64b(77));
        raw.push(2);
        let m = parse_one(&raw).unwrap();
        assert_eq!(
            m.to_canon_json(),
            "{\"cls\":\"SelectChain\",\"type\":16,\"player\":1,\"spe_count\":3,\"forced\":true,\
\"hint_timing\":17,\"hint_timing_other\":34,\"chains\":[[[42,1,4,3,5],77,2]]}"
        );
    }

    #[test]
    fn idlecmd_reads_reposition_short_and_bools() {
        let mut raw = vec![C::MSG_SELECT_IDLECMD as u8, 0];
        raw.extend(u32b(1));
        raw.extend([u32b(1), vec![0, 2], u32b(0)].concat()); // summonable: card_seq32
        raw.extend(u32b(0)); // spsummonable
        raw.extend(u32b(1));
        raw.extend([u32b(2), vec![0, 4, 7]].concat()); // repositionable: seq u8
        raw.extend(u32b(0)); // msetable
        raw.extend(u32b(0)); // ssetable
        raw.extend(u32b(1)); // activatable
        raw.extend([u32b(9), vec![0, 4], u32b(0), u64b(4), vec![1]].concat());
        raw.extend([1, 0, 2]);
        let m = parse_one(&raw).unwrap();
        assert_eq!(
            m.to_canon_json(),
            "{\"cls\":\"SelectIdleCmd\",\"type\":11,\"player\":0,\"summonable\":[[1,0,2,0,0]],\
\"spsummonable\":[],\"repositionable\":[[2,0,4,7,0]],\"msetable\":[],\"ssetable\":[],\
\"activatable\":[[[9,0,4,0,0],4,1]],\"can_battle_phase\":true,\"can_end_phase\":false,\"can_shuffle\":true}"
        );
    }
}
