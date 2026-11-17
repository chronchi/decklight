//! Rust side of `cpp/ygoenv/capi.h`: the raw C ABI, a `#[repr(C)]` mirror of `ygo::Obs` (checked against the C++
//! layout at runtime by [`check_layout`]), and thin owning wrappers [`Engine`] and [`Env`].
use std::ffi::{CString, c_char, c_int};
use std::marker::PhantomData;

pub const OBS_VERSION: u32 = 1;
pub const MAX_CARDS: usize = 160;
pub const CARD_COLS: usize = 21;
pub const MAX_CHAIN: usize = 16;
pub const CHAIN_COLS: usize = 5;
pub const MAX_OPTIONS: usize = 64;
pub const ACTION_COLS: usize = 10;
pub const HISTORY: usize = 32;
pub const HISTORY_COLS: usize = 13;
pub const HISTORY_MAX: usize = 512;
pub const GLOBAL_COLS: usize = 24;

/// Mirror of `ygo::Obs` (cpp/ygoenv/ygoenv.h), field for field.
#[repr(C)]
#[derive(Clone)]
pub struct Obs {
    pub cards: [i32; MAX_CARDS * CARD_COLS],
    pub card_mask: [u8; MAX_CARDS],
    pub card_known: [u8; MAX_CARDS],
    pub known_code: [i32; MAX_CARDS],
    pub priv_code: [i32; MAX_CARDS],
    pub global: [i32; GLOBAL_COLS],
    pub chain: [i32; MAX_CHAIN * CHAIN_COLS],
    pub chain_mask: [u8; MAX_CHAIN],
    pub actions: [i32; MAX_OPTIONS * ACTION_COLS],
    pub action_mask: [u8; MAX_OPTIONS],
    pub history: [i32; HISTORY * HISTORY_COLS],
    pub history_mask: [u8; HISTORY],
    pub hist_long: [i32; HISTORY_MAX * HISTORY_COLS],
    pub hist_long_mask: [u8; HISTORY_MAX],
    /// The observable history window (own decisions + the opponent's public events), read by observable nets.
    pub obs_history: [i32; HISTORY * HISTORY_COLS],
    pub obs_history_mask: [u8; HISTORY],
}

impl Obs {
    pub fn zeroed() -> Box<Obs> {
        // SAFETY: every field is an integer array; all-zero is a valid value (C++ value-initialises the same way).
        unsafe { Box::new(std::mem::zeroed()) }
    }
}

#[repr(C)]
pub struct RawEngine {
    _p: [u8; 0],
}
#[repr(C)]
pub struct RawEnv {
    _p: [u8; 0],
}
#[repr(C)]
pub struct RawDecision {
    _p: [u8; 0],
}

unsafe extern "C" {
    pub fn ygo_obs_version() -> u32;
    pub fn ygo_obs_size() -> usize;
    pub fn ygo_obs_offset(field: u32) -> usize;
    pub fn ygo_engine_new() -> *mut RawEngine;
    pub fn ygo_engine_free(e: *mut RawEngine);
    pub fn ygo_engine_set_precompile(e: *mut RawEngine, on: c_int);
    pub fn ygo_engine_add_card(
        e: *mut RawEngine, code: u32, alias: u32, setcodes: *const u16, n_setcodes: usize, type_: u32, level: u32,
        attribute: u32, race: u64, attack: i32, defense: i32, lscale: u32, rscale: u32, link_marker: u32,
    );
    pub fn ygo_engine_set_script(e: *mut RawEngine, name: *const c_char, path: *const c_char);
    pub fn ygo_engine_set_script_range(
        e: *mut RawEngine, name: *const c_char, path: *const c_char, offset: u64, len: u32,
    );
    pub fn ygo_engine_script(e: *mut RawEngine, name: *const c_char, buf: *mut u8, cap: usize) -> usize;
    pub fn ygo_engine_finalize(e: *mut RawEngine);
    pub fn ygo_engine_num_cards(e: *const RawEngine) -> usize;
    pub fn ygo_engine_card_type(e: *const RawEngine, code: u32, type_: *mut u32) -> c_int;
    pub fn ygo_env_new(
        e: *mut RawEngine, main0: *const u32, n_main0: usize, extra0: *const u32, n_extra0: usize, main1: *const u32,
        n_main1: usize, extra1: *const u32, n_extra1: usize, seed: u64, flags: u64, lp: u32, hand: u32, draw: u32,
        seed_state: *const u64, n_seed_state: usize, team_order: *const u8, n_team_order: usize,
        edopro_add: c_int, auto_single: c_int, err: *mut c_char, err_len: usize,
    ) -> *mut RawEnv;
    pub fn ygo_env_free(env: *mut RawEnv);
    pub fn ygo_env_reset(env: *mut RawEnv, obs: *mut Obs, err: *mut c_char, err_len: usize) -> c_int;
    pub fn ygo_env_step(env: *mut RawEnv, index: i32, obs: *mut Obs, err: *mut c_char, err_len: usize) -> c_int;
    pub fn ygo_env_num_options(env: *const RawEnv) -> i32;
    pub fn ygo_env_player(env: *const RawEnv) -> i32;
    pub fn ygo_env_done(env: *const RawEnv) -> i32;
    pub fn ygo_env_winner(env: *const RawEnv) -> i32;
    pub fn ygo_env_decisions(env: *const RawEnv) -> i32;
    pub fn ygo_env_truncated(env: *const RawEnv) -> i32;
    pub fn ygo_env_num_errors(env: *const RawEnv) -> usize;
    pub fn ygo_env_error(env: *const RawEnv, i: usize, buf: *mut c_char, len: usize) -> usize;
    pub fn ygo_env_set_record_trace(env: *mut RawEnv, on: c_int);
    pub fn ygo_env_msg_type(env: *const RawEnv) -> i32;
    pub fn ygo_env_dec_step(env: *const RawEnv) -> i32;
    pub fn ygo_env_dec_seq(env: *const RawEnv) -> i64;
    pub fn ygo_env_selected(env: *const RawEnv, out: *mut CardRef, cap: usize) -> usize;
    pub fn ygo_env_options(env: *const RawEnv, out: *mut Opt, cap: usize) -> usize;
    pub fn ygo_env_pending_msg(env: *const RawEnv, buf: *mut u8, cap: usize) -> usize;
    pub fn ygo_env_trace_len(env: *const RawEnv) -> usize;
    pub fn ygo_env_trace_msg(env: *const RawEnv, i: usize, buf: *mut u8, cap: usize) -> usize;
    pub fn ygo_env_duel_state(env: *const RawEnv, out: *mut DuelState, err: *mut c_char, err_len: usize) -> c_int;
    pub fn ygo_env_query_field(env: *const RawEnv, buf: *mut u8, cap: usize, err: *mut c_char, err_len: usize)
    -> usize;
    pub fn ygo_env_query_location(
        env: *const RawEnv, controller: u8, location: u32, flags: u32, buf: *mut u8, cap: usize,
        err: *mut c_char, err_len: usize,
    ) -> usize;
    pub fn ygo_env_history_len(env: *const RawEnv) -> usize;
    pub fn ygo_env_history_log(env: *const RawEnv, out: *mut i32, cap_rows: usize) -> usize;
    pub fn ygo_env_num_choices(env: *const RawEnv) -> usize;
    pub fn ygo_env_choices(env: *const RawEnv, out: *mut i32, cap: usize) -> usize;
    pub fn ygo_env_cards(env: *mut RawEnv, out: *mut DebugCard, cap: usize) -> usize;
    pub fn ygo_env_replay_from_reset(
        env: *mut RawEnv, choices: *const i32, n: usize, obs: *mut Obs, err: *mut c_char, err_len: usize,
    ) -> c_int;
    pub fn ygo_env_observe_as(env: *mut RawEnv, viewer: i32, obs: *mut Obs, err: *mut c_char, err_len: usize)
    -> c_int;
    pub fn ygo_env_replay_keys(
        env: *mut RawEnv, keys: *const i32, ranks: *const i32, n: usize, obs: *mut Obs, resolved: *mut i32,
        resolved_cap: usize, n_resolved: *mut usize, reason: *mut c_char, reason_len: usize, err: *mut c_char,
        err_len: usize,
    ) -> c_int;
    pub fn ygo_env_num_responses(env: *const RawEnv) -> usize;
    pub fn ygo_env_response(env: *const RawEnv, i: usize, buf: *mut u8, cap: usize) -> usize;

    pub fn ygo_decision_new(
        e: *mut RawEngine, msg: *const u8, len: usize, pool: *const u32, n_pool: usize, err: *mut c_char,
        err_len: usize,
    ) -> *mut RawDecision;
    pub fn ygo_decision_clone(d: *const RawDecision, err: *mut c_char, err_len: usize) -> *mut RawDecision;
    pub fn ygo_env_decision_clone(env: *const RawEnv, err: *mut c_char, err_len: usize) -> *mut RawDecision;
    pub fn ygo_decision_free(d: *mut RawDecision);
    pub fn ygo_decision_msg_type(d: *const RawDecision) -> i32;
    pub fn ygo_decision_player(d: *const RawDecision) -> i32;
    pub fn ygo_decision_step(d: *const RawDecision) -> i32;
    pub fn ygo_decision_options(d: *const RawDecision, out: *mut Opt, cap: usize) -> usize;
    pub fn ygo_decision_choose(
        d: *mut RawDecision, index: i32, buf: *mut u8, cap: usize, out_len: *mut usize, err: *mut c_char,
        err_len: usize,
    ) -> i32;
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardRef {
    pub code: u32,
    pub controller: u32,
    pub location: u32,
    pub sequence: u32,
    pub position: u32,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opt {
    pub kind: i32,
    pub has_card: i32,
    pub code: u32,
    pub controller: u32,
    pub location: u32,
    pub sequence: u32,
    pub position: u32,
    pub opt_code: u32,
    pub desc: u64,
    pub arg: i64,
}

/// One card anywhere on the board (`Env::cards`, for honest-search determinization).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugCard {
    pub controller: u32,
    pub location: u32,
    pub sequence: u32,
    pub position: u32,
    pub code: u32,
    pub cardid: u32,
    pub is_public: u8,
    pub known: u8,
}

/// Outcome of a semantic replay (`Env::replay_keys`), exactly `Env::ReplayKeysResult` in C++.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayKeysResult {
    /// true: every key resolved and `obs` holds the final decision; false: the duel ended or a key failed.
    pub ok: bool,
    pub resolved: Vec<i32>,
    /// "" on success, else "no_key" / "replay_error" (empty also when the duel ended exactly at the end).
    pub reason: String,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DuelState {
    pub turn: i32,
    pub turn_player: i32,
    pub first_player: i32,
    pub phase: i32,
    pub winner: i32,
    pub win_reason: i32,
    pub select_hint: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionState {
    pub msg_type: i32,
    pub player: i32,
    pub step: i32,
    pub selected: Vec<CardRef>,
    pub options: Vec<Opt>,
}

/// A pending selection (`make_decision`) independent of a duel, for client mode and response inversion.
pub struct Decision {
    raw: *mut RawDecision,
}

// SAFETY: a Decision owns its C++ object; the engine and message it points at stay shared and read-only.
unsafe impl Send for Decision {}

impl Decision {
    /// `make_decision(msg, engine, pool)`; `pool` is the ANNOUNCE_CARD card-code pool.
    pub fn new(engine: &Engine, msg: &[u8], pool: &[u32]) -> Result<Decision, String> {
        let mut err = err_buf();
        let raw = unsafe {
            ygo_decision_new(
                engine.raw,
                msg.as_ptr(),
                msg.len(),
                if pool.is_empty() { std::ptr::null() } else { pool.as_ptr() },
                pool.len(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if raw.is_null() { Err(err_string(&err)) } else { Ok(Decision { raw }) }
    }

    /// The env's current pending decision, copied.
    pub fn from_env(env: &Env<'_>) -> Result<Option<Decision>, String> {
        let mut err = err_buf();
        let raw = unsafe { ygo_env_decision_clone(env.raw, err.as_mut_ptr() as *mut c_char, err.len()) };
        if raw.is_null() {
            if err[0] != 0 { Err(err_string(&err)) } else { Ok(None) }
        } else {
            Ok(Some(Decision { raw }))
        }
    }

    pub fn clone_handle(&self) -> Result<Decision, String> {
        let mut err = err_buf();
        let raw = unsafe { ygo_decision_clone(self.raw, err.as_mut_ptr() as *mut c_char, err.len()) };
        if raw.is_null() { Err(err_string(&err)) } else { Ok(Decision { raw }) }
    }

    pub fn msg_type(&self) -> i32 {
        unsafe { ygo_decision_msg_type(self.raw) }
    }
    pub fn player(&self) -> i32 {
        unsafe { ygo_decision_player(self.raw) }
    }
    pub fn step(&self) -> i32 {
        unsafe { ygo_decision_step(self.raw) }
    }

    pub fn options(&self) -> Vec<Opt> {
        let n = unsafe { ygo_decision_options(self.raw, std::ptr::null_mut(), 0) };
        let mut out = vec![Opt { kind: 0, has_card: 0, code: 0, controller: 0, location: 0, sequence: 0, position: 0, opt_code: 0, desc: 0, arg: 0 }; n];
        if n > 0 {
            unsafe { ygo_decision_options(self.raw, out.as_mut_ptr(), n) };
        }
        out
    }

    /// Ok(Some(response)) when this pick completes the selection, Ok(None) when more picks are needed.
    /// One call: a response is at most 8 + 255 bytes (card sets) or 2 x cards (counters), so 4096 is ample.
    pub fn choose(&mut self, index: i32) -> Result<Option<Vec<u8>>, String> {
        let mut err = err_buf();
        let mut out = vec![0u8; 4096];
        let mut n = 0usize;
        let rc = unsafe {
            ygo_decision_choose(
                self.raw,
                index,
                out.as_mut_ptr(),
                out.len(),
                &mut n,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        match rc {
            0 => Ok(None),
            1 => {
                if n > out.len() {
                    return Err(format!("response of {n} bytes exceeds the {} byte buffer", out.len()));
                }
                out.truncate(n);
                Ok(Some(out))
            }
            _ => Err(err_string(&err)),
        }
    }
}

impl Drop for Decision {
    fn drop(&mut self) {
        unsafe { ygo_decision_free(self.raw) }
    }
}

/// Fails if the Rust `Obs` does not match the C++ one byte for byte (size, every field offset, version).
pub fn check_layout() -> Result<(), String> {
    let rust = [
        std::mem::offset_of!(Obs, cards),
        std::mem::offset_of!(Obs, card_mask),
        std::mem::offset_of!(Obs, card_known),
        std::mem::offset_of!(Obs, known_code),
        std::mem::offset_of!(Obs, priv_code),
        std::mem::offset_of!(Obs, global),
        std::mem::offset_of!(Obs, chain),
        std::mem::offset_of!(Obs, chain_mask),
        std::mem::offset_of!(Obs, actions),
        std::mem::offset_of!(Obs, action_mask),
        std::mem::offset_of!(Obs, history),
        std::mem::offset_of!(Obs, history_mask),
        std::mem::offset_of!(Obs, hist_long),
        std::mem::offset_of!(Obs, hist_long_mask),
        std::mem::offset_of!(Obs, obs_history),
        std::mem::offset_of!(Obs, obs_history_mask),
    ];
    // SAFETY: pure functions without arguments that point anywhere.
    unsafe {
        if ygo_obs_version() != OBS_VERSION {
            return Err(format!("obs version: C++ {} vs Rust {OBS_VERSION}", ygo_obs_version()));
        }
        if ygo_obs_size() != std::mem::size_of::<Obs>() {
            return Err(format!("obs size: C++ {} vs Rust {}", ygo_obs_size(), std::mem::size_of::<Obs>()));
        }
        for (i, off) in rust.iter().enumerate() {
            let c = ygo_obs_offset(i as u32);
            if c != *off {
                return Err(format!("obs field {i} offset: C++ {c} vs Rust {off}"));
            }
        }
    }
    Ok(())
}

/// Card stats exactly as the core takes them (OCG_CardData; `decklight.engine.resources.CardData`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardData {
    pub code: u32,
    pub alias: u32,
    pub setcodes: Vec<u16>,
    pub type_: u32,
    pub level: u32,
    pub attribute: u32,
    pub race: u64,
    pub attack: i32,
    pub defense: i32,
    pub lscale: u32,
    pub rscale: u32,
    pub link_marker: u32,
}

/// Card data + script index for the core. Must outlive every [`Env`] made from it (enforced by the borrow).
pub struct Engine {
    raw: *mut RawEngine,
}

// SAFETY: after `finalize` the C++ Engine is only read, and its script cache is guarded by a shared_mutex.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    pub fn new() -> Engine {
        check_layout().expect("ygoenv-sys: Obs layout mismatch");
        // SAFETY: returns a fresh heap object owned by this wrapper.
        Engine { raw: unsafe { ygo_engine_new() } }
    }

    pub fn set_precompile(&mut self, on: bool) {
        unsafe { ygo_engine_set_precompile(self.raw, on as c_int) }
    }

    pub fn add_card(&mut self, c: &CardData) {
        // SAFETY: setcodes points at c.setcodes.len() u16s for the duration of the call.
        unsafe {
            ygo_engine_add_card(
                self.raw, c.code, c.alias, c.setcodes.as_ptr(), c.setcodes.len(), c.type_, c.level, c.attribute,
                c.race, c.attack, c.defense, c.lscale, c.rscale, c.link_marker,
            )
        }
    }

    pub fn set_script(&mut self, name: &str, path: &str) {
        let (n, p) = (CString::new(name).unwrap(), CString::new(path).unwrap());
        unsafe { ygo_engine_set_script(self.raw, n.as_ptr(), p.as_ptr()) }
    }

    /// Like [`Engine::set_script`], but the bytes are `[offset, offset + len)` of `path`.
    pub fn set_script_range(&mut self, name: &str, path: &str, offset: u64, len: u32) {
        let (n, p) = (CString::new(name).unwrap(), CString::new(path).unwrap());
        unsafe { ygo_engine_set_script_range(self.raw, n.as_ptr(), p.as_ptr(), offset, len) }
    }

    /// The script bytes the engine would hand the core (precompiled when enabled), or `None` if unknown.
    pub fn script(&mut self, name: &str) -> Option<Vec<u8>> {
        let n = CString::new(name).unwrap();
        let len = unsafe { ygo_engine_script(self.raw, n.as_ptr(), std::ptr::null_mut(), 0) };
        if len == 0 {
            return None;
        }
        let mut out = vec![0u8; len];
        unsafe { ygo_engine_script(self.raw, n.as_ptr(), out.as_mut_ptr(), len) };
        Some(out)
    }

    pub fn finalize(&mut self) {
        unsafe { ygo_engine_finalize(self.raw) }
    }

    pub fn num_cards(&self) -> usize {
        unsafe { ygo_engine_num_cards(self.raw) }
    }

    /// The card's `TYPE_*` flags, or `None` when the code is not in the card DB.
    pub fn card_type(&self, code: u32) -> Option<u32> {
        let mut t = 0u32;
        if unsafe { ygo_engine_card_type(self.raw, code, &mut t) } == 0 { Some(t) } else { None }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ygo_engine_free(self.raw) }
    }
}

/// `ygo::DuelConfig`; the main decks in the order they reach the core (see `decklight-cards::shuffle_main`).
#[derive(Clone, Debug)]
pub struct DuelConfig {
    pub main0: Vec<u32>,
    pub extra0: Vec<u32>,
    pub main1: Vec<u32>,
    pub extra1: Vec<u32>,
    pub seed: u64,
    pub flags: u64,
    pub lp: u32,
    pub hand: u32,
    pub draw: u32,
    /// The core's 4 x u64 seed state, used instead of `seed_words(seed)` when set (ADR 0010 replays).
    pub seed_state: Option<[u64; 4]>,
    /// The order the teams' cards are added to the core (default `[0, 1]`; changes the duel).
    pub team_order: [u8; 2],
    /// Every main-deck card added at sequence 0, like EDOPro/Multirole (C++ `DuelConfig::edopro_add`).
    pub edopro_add: bool,
}

impl Default for DuelConfig {
    fn default() -> Self {
        DuelConfig {
            main0: Vec::new(),
            extra0: Vec::new(),
            main1: Vec::new(),
            extra1: Vec::new(),
            seed: 0,
            flags: TCG_FLAGS,
            lp: 8000,
            hand: 5,
            draw: 1,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        }
    }
}

/// `TCG_FLAGS` of decklight.engine.duel: DUEL_MODE_MR5 | DUEL_TCG_SEGOC_NONPUBLIC.
pub const TCG_FLAGS: u64 = 0x2E800 | 0x1_0000_0000;

/// One duel driven decision by decision (the C++ `Env`, same as `CppEnv` in Python).
pub struct Env<'e> {
    raw: *mut RawEnv,
    _engine: PhantomData<&'e Engine>,
}

unsafe impl Send for Env<'_> {}

fn err_buf() -> Vec<u8> {
    vec![0u8; 1024]
}

fn err_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

impl<'e> Env<'e> {
    pub fn new(engine: &'e Engine, c: &DuelConfig, auto_single: bool) -> Result<Env<'e>, String> {
        let mut err = err_buf();
        let seed_state = c.seed_state;
        // SAFETY: slices live for the call; the engine outlives the env through the 'e borrow.
        let raw = unsafe {
            ygo_env_new(
                engine.raw, c.main0.as_ptr(), c.main0.len(), c.extra0.as_ptr(), c.extra0.len(), c.main1.as_ptr(),
                c.main1.len(), c.extra1.as_ptr(), c.extra1.len(), c.seed, c.flags, c.lp, c.hand, c.draw,
                seed_state.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                seed_state.map_or(0, |_| 4), c.team_order.as_ptr(), 2, c.edopro_add as c_int, auto_single as c_int,
                err.as_mut_ptr() as *mut c_char, err.len(),
            )
        };
        if raw.is_null() { Err(err_string(&err)) } else { Ok(Env { raw, _engine: PhantomData }) }
    }

    /// Ok(true): a decision is pending and `obs` holds it; Ok(false): the duel ended.
    pub fn reset(&mut self, obs: &mut Obs) -> Result<bool, String> {
        let mut err = err_buf();
        match unsafe { ygo_env_reset(self.raw, obs, err.as_mut_ptr() as *mut c_char, err.len()) } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(err_string(&err)),
        }
    }

    pub fn step(&mut self, index: i32, obs: &mut Obs) -> Result<bool, String> {
        let mut err = err_buf();
        match unsafe { ygo_env_step(self.raw, index, obs, err.as_mut_ptr() as *mut c_char, err.len()) } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(err_string(&err)),
        }
    }

    pub fn num_options(&self) -> i32 {
        unsafe { ygo_env_num_options(self.raw) }
    }
    pub fn player(&self) -> i32 {
        unsafe { ygo_env_player(self.raw) }
    }
    /// Monotonic id of the pending decision object (changes when a new core message arrives).
    pub fn decision_seq(&self) -> i64 {
        unsafe { ygo_env_dec_seq(self.raw) }
    }
    pub fn done(&self) -> bool {
        unsafe { ygo_env_done(self.raw) != 0 }
    }
    pub fn winner(&self) -> i32 {
        unsafe { ygo_env_winner(self.raw) }
    }
    pub fn decisions(&self) -> i32 {
        unsafe { ygo_env_decisions(self.raw) }
    }
    pub fn truncated(&self) -> i32 {
        unsafe { ygo_env_truncated(self.raw) }
    }
    /// Lua/script errors the core reported so far.
    pub fn errors(&self) -> Vec<String> {
        let n = unsafe { ygo_env_num_errors(self.raw) };
        (0..n)
            .map(|i| {
                let mut buf = vec![0u8; 4096];
                unsafe { ygo_env_error(self.raw, i, buf.as_mut_ptr() as *mut c_char, buf.len()) };
                err_string(&buf)
            })
            .collect()
    }

    pub fn set_record_trace(&mut self, on: bool) {
        unsafe { ygo_env_set_record_trace(self.raw, on as c_int) }
    }

    /// The pending decision, or `None` once the duel ended (before the first reset too).
    pub fn decision(&self) -> Option<DecisionState> {
        let msg_type = unsafe { ygo_env_msg_type(self.raw) };
        if msg_type < 0 {
            return None;
        }
        let step = unsafe { ygo_env_dec_step(self.raw) };
        // SAFETY: msg_type >= 0 means has_decision, so the selected/options/player calls are valid.
        let n_sel = unsafe { ygo_env_selected(self.raw, std::ptr::null_mut(), 0) };
        let mut selected = vec![
            CardRef { code: 0, controller: 0, location: 0, sequence: 0, position: 0 };
            n_sel
        ];
        if n_sel > 0 {
            unsafe { ygo_env_selected(self.raw, selected.as_mut_ptr(), n_sel) };
        }
        let n_opt = unsafe { ygo_env_options(self.raw, std::ptr::null_mut(), 0) };
        let mut options = vec![
            Opt {
                kind: 0,
                has_card: 0,
                code: 0,
                controller: 0,
                location: 0,
                sequence: 0,
                position: 0,
                opt_code: 0,
                desc: 0,
                arg: 0,
            };
            n_opt
        ];
        if n_opt > 0 {
            unsafe { ygo_env_options(self.raw, options.as_mut_ptr(), n_opt) };
        }
        // Player: the C side's dec_player; ygo_env_player is only valid with a decision, which we have.
        let pl = unsafe { ygo_env_player(self.raw) };
        Some(DecisionState { msg_type, player: pl, step, selected, options })
    }

    pub fn pending_msg(&self) -> Vec<u8> {
        let n = unsafe { ygo_env_pending_msg(self.raw, std::ptr::null_mut(), 0) };
        let mut out = vec![0u8; n];
        if n > 0 {
            unsafe { ygo_env_pending_msg(self.raw, out.as_mut_ptr(), n) };
        }
        out
    }

    pub fn trace_len(&self) -> usize {
        unsafe { ygo_env_trace_len(self.raw) }
    }

    pub fn trace_msg(&self, i: usize) -> Vec<u8> {
        let n = unsafe { ygo_env_trace_msg(self.raw, i, std::ptr::null_mut(), 0) };
        let mut out = vec![0u8; n];
        if n > 0 {
            unsafe { ygo_env_trace_msg(self.raw, i, out.as_mut_ptr(), n) };
        }
        out
    }

    /// Err before the first `reset` (no duel yet).
    pub fn duel_state(&self) -> Result<DuelState, String> {
        let mut out = DuelState {
            turn: 0,
            turn_player: 0,
            first_player: 0,
            phase: 0,
            winner: 0,
            win_reason: 0,
            select_hint: 0,
        };
        let mut err = err_buf();
        let rc =
            unsafe { ygo_env_duel_state(self.raw, &mut out, err.as_mut_ptr() as *mut c_char, err.len()) };
        if rc != 0 {
            return Err(format!("duel_state: {}", err_string(&err)));
        }
        Ok(out)
    }

    /// Raw `OCG_DuelQueryField` bytes; Err before the first `reset`.
    pub fn query_field(&self) -> Result<Vec<u8>, String> {
        let mut err = err_buf();
        let n = unsafe {
            ygo_env_query_field(self.raw, std::ptr::null_mut(), 0, err.as_mut_ptr() as *mut c_char, err.len())
        };
        if err[0] != 0 {
            return Err(format!("query_field: {}", err_string(&err)));
        }
        let mut out = vec![0u8; n];
        if n > 0 {
            let mut err2 = err_buf();
            unsafe {
                ygo_env_query_field(
                    self.raw,
                    out.as_mut_ptr(),
                    n,
                    err2.as_mut_ptr() as *mut c_char,
                    err2.len(),
                )
            };
        }
        Ok(out)
    }

    /// Raw `OCG_DuelQueryLocation` bytes with `flags`; Err before the first `reset`.
    pub fn query_location(&self, controller: u8, location: u32, flags: u32) -> Result<Vec<u8>, String> {
        let mut err = err_buf();
        let n = unsafe {
            ygo_env_query_location(
                self.raw,
                controller,
                location,
                flags,
                std::ptr::null_mut(),
                0,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if err[0] != 0 {
            return Err(format!("query_location: {}", err_string(&err)));
        }
        let mut out = vec![0u8; n];
        if n > 0 {
            let mut err2 = err_buf();
            unsafe {
                ygo_env_query_location(
                    self.raw,
                    controller,
                    location,
                    flags,
                    out.as_mut_ptr(),
                    n,
                    err2.as_mut_ptr() as *mut c_char,
                    err2.len(),
                )
            };
        }
        Ok(out)
    }

    pub fn history_log(&self) -> Vec<[i32; 14]> {
        let n = unsafe { ygo_env_history_len(self.raw) };
        let mut flat = vec![0i32; n * 14];
        if n > 0 {
            unsafe { ygo_env_history_log(self.raw, flat.as_mut_ptr(), n) };
        }
        flat.chunks_exact(14).map(|c| c.try_into().unwrap()).collect()
    }

    pub fn choices(&self) -> Vec<i32> {
        let n = unsafe { ygo_env_num_choices(self.raw) };
        let mut out = vec![0i32; n];
        if n > 0 {
            unsafe { ygo_env_choices(self.raw, out.as_mut_ptr(), n) };
        }
        out
    }

    /// Every card of both players (DECK/HAND/MZONE/SZONE/GRAVE/REMOVED/EXTRA), like the pybind `Env::cards`.
    pub fn cards(&self) -> Vec<DebugCard> {
        let n = unsafe { ygo_env_cards(self.raw, std::ptr::null_mut(), 0) };
        let mut out = vec![
            DebugCard {
                controller: 0,
                location: 0,
                sequence: 0,
                position: 0,
                code: 0,
                cardid: 0,
                is_public: 0,
                known: 0,
            };
            n
        ];
        if n > 0 {
            unsafe { ygo_env_cards(self.raw, out.as_mut_ptr(), n) };
        }
        out
    }

    /// reset + step every choice in one call (`CppEnv.replay_from_reset`). Ok(true): a decision is pending.
    pub fn replay(&mut self, choices: &[i32], obs: &mut Obs) -> Result<bool, String> {
        let mut err = err_buf();
        match unsafe {
            ygo_env_replay_from_reset(
                self.raw,
                if choices.is_empty() { std::ptr::null() } else { choices.as_ptr() },
                choices.len(),
                obs,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(err_string(&err)),
        }
    }

    /// The observation from `viewer`'s side at the current position, whoever is to act (`review.human_obs_dict`):
    /// a pending decision keeps its options, a finished duel has none.
    pub fn observe_as(&mut self, viewer: i32, obs: &mut Obs) -> Result<(), String> {
        let mut err = err_buf();
        match unsafe { ygo_env_observe_as(self.raw, viewer, obs, err.as_mut_ptr() as *mut c_char, err.len()) } {
            0 => Ok(()),
            _ => Err(err_string(&err)),
        }
    }

    /// reset + semantic replay by (key, rank) (`CppEnv.replay_keys`); `keys` are rows of 8 int32.
    pub fn replay_keys(&mut self, keys: &[[i32; 8]], ranks: &[i32], obs: &mut Obs) -> Result<ReplayKeysResult, String> {
        assert_eq!(keys.len(), ranks.len(), "replay_keys: keys and ranks must be the same length");
        let flat: Vec<i32> = keys.iter().flatten().copied().collect();
        let mut resolved = vec![0i32; keys.len()];
        let mut n_resolved = 0usize;
        let mut reason = vec![0u8; 256];
        let mut err = err_buf();
        let rc = unsafe {
            ygo_env_replay_keys(
                self.raw,
                if flat.is_empty() { std::ptr::null() } else { flat.as_ptr() },
                if ranks.is_empty() { std::ptr::null() } else { ranks.as_ptr() },
                keys.len(),
                obs,
                if resolved.is_empty() { std::ptr::null_mut() } else { resolved.as_mut_ptr() },
                resolved.len(),
                &mut n_resolved,
                reason.as_mut_ptr() as *mut c_char,
                reason.len(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        match rc {
            1 | 0 => {
                resolved.truncate(n_resolved.min(resolved.len()));
                Ok(ReplayKeysResult { ok: rc == 1, resolved, reason: err_string(&reason) })
            }
            _ => Err(err_string(&err)),
        }
    }

    /// Raw OCG response bytes so far (match record v1); one entry per completed core selection.
    pub fn num_responses(&self) -> usize {
        unsafe { ygo_env_num_responses(self.raw) }
    }

    pub fn response(&self, i: usize) -> Vec<u8> {
        let n = unsafe { ygo_env_response(self.raw, i, std::ptr::null_mut(), 0) };
        let mut out = vec![0u8; n];
        if n > 0 {
            unsafe { ygo_env_response(self.raw, i, out.as_mut_ptr(), n) };
        }
        out
    }
}

impl Drop for Env<'_> {
    fn drop(&mut self) {
        unsafe { ygo_env_free(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripts_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../third_party/CardScripts")
    }

    fn tiny_engine() -> Engine {
        let mut e = Engine::new();
        let d = scripts_dir();
        e.set_script("constant.lua", &d.join("constant.lua").to_string_lossy());
        e.set_script("utility.lua", &d.join("utility.lua").to_string_lossy());
        e.finalize();
        e
    }

    fn tiny_config() -> DuelConfig {
        DuelConfig {
            main0: vec![],
            extra0: vec![],
            main1: vec![],
            extra1: vec![],
            seed: 1,
            flags: TCG_FLAGS,
            lp: 8000,
            hand: 0,
            draw: 0,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        }
    }

    #[test]
    fn no_duel_is_safe() {
        let e = tiny_engine();
        let env = Env::new(&e, &tiny_config(), false).unwrap();
        // Before reset there is no duel: safe accessors return empty/None, never crash.
        assert!(env.decision().is_none());
        assert!(env.pending_msg().is_empty());
        assert_eq!(env.trace_len(), 0);
        assert!(env.trace_msg(0).is_empty());
        assert_eq!(unsafe { ygo_env_selected(env.raw, std::ptr::null_mut(), 0) }, 0);
        assert_eq!(unsafe { ygo_env_options(env.raw, std::ptr::null_mut(), 0) }, 0);
        assert_eq!(unsafe { ygo_env_pending_msg(env.raw, std::ptr::null_mut(), 0) }, 0);
        assert_eq!(unsafe { ygo_env_trace_len(env.raw) }, 0);
        assert_eq!(unsafe { ygo_env_trace_msg(env.raw, 0, std::ptr::null_mut(), 0) }, 0);
        assert_eq!(unsafe { ygo_env_history_len(env.raw) }, 0);
        assert_eq!(unsafe { ygo_env_num_choices(env.raw) }, 0);
        // duel_state / queries report no-duel via err.
        let mut st = DuelState {
            turn: 0,
            turn_player: 0,
            first_player: 0,
            phase: 0,
            winner: 0,
            win_reason: 0,
            select_hint: 0,
        };
        let mut err = err_buf();
        assert_eq!(
            unsafe { ygo_env_duel_state(env.raw, &mut st, err.as_mut_ptr() as *mut c_char, err.len()) },
            -1
        );
        let mut err2 = err_buf();
        assert_eq!(
            unsafe {
                ygo_env_query_field(env.raw, std::ptr::null_mut(), 0, err2.as_mut_ptr() as *mut c_char, err2.len())
            },
            0
        );
        assert_ne!(err2[0], 0);
        // The safe wrappers turn that into Err, never a panic.
        assert!(env.duel_state().is_err());
        assert!(env.query_field().is_err());
        assert!(env.query_location(0, 0x2, 0x1).is_err());
    }

    #[test]
    fn buffer_size_protocol() {
        let e = tiny_engine();
        let mut env = Env::new(&e, &tiny_config(), false).unwrap();
        env.set_record_trace(true);
        let mut obs = Obs::zeroed();
        let _ = env.reset(&mut obs);
        // Every sized query: cap 0 gives the needed size; a second call with that size returns the same size
        // and identical bytes/rows. Undersized calls return the needed size without writing past `cap`.
        let check_bytes = |get_needed: &dyn Fn(*mut u8, usize) -> usize, get_full: &dyn Fn() -> Vec<u8>| {
            let n = get_needed(std::ptr::null_mut(), 0);
            let full = get_full();
            assert_eq!(n, full.len());
            if n > 0 {
                // Undersized: returns needed size; buffer untouched beyond cap (we pass a 1-byte buf).
                let mut one = vec![0xAAu8; 1];
                let m = get_needed(one.as_mut_ptr(), 1);
                assert_eq!(m, n);
                // Exact: copies fully.
                let mut buf = vec![0u8; n];
                let k = get_needed(buf.as_mut_ptr(), n);
                assert_eq!(k, n);
                assert_eq!(buf, full);
            }
        };
        let raw = env.raw as *const RawEnv;
        check_bytes(
            &|p, c| unsafe { ygo_env_pending_msg(raw, p, c) },
            &|| env.pending_msg(),
        );
        if env.trace_len() > 0 {
            let n = unsafe { ygo_env_trace_msg(raw, 0, std::ptr::null_mut(), 0) };
            let full = env.trace_msg(0);
            assert_eq!(n, full.len());
            if n > 0 {
                let mut buf = vec![0u8; n];
                assert_eq!(unsafe { ygo_env_trace_msg(raw, 0, buf.as_mut_ptr(), n) }, n);
                assert_eq!(buf, full);
            }
        }
        check_bytes(&|p, c| unsafe {
            let mut err = err_buf();
            ygo_env_query_field(raw, p, c, err.as_mut_ptr() as *mut c_char, err.len())
        }, &|| env.query_field().unwrap());
        const CARD_FLAGS: u32 = 0x1 | 0x2 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800
            | 0x10000
            | 0x40000
            | 0x80000
            | 0x100000
            | 0x200000
            | 0x400000
            | 0x800000;
        for con in [0u8, 1] {
            for loc in [0x1u32, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40] {
                check_bytes(&|p, c| unsafe {
                    let mut err = err_buf();
                    ygo_env_query_location(raw, con, loc, CARD_FLAGS, p, c, err.as_mut_ptr() as *mut c_char, err.len())
                }, &|| env.query_location(con, loc, CARD_FLAGS).unwrap());
            }
        }
        // selected / options / history / choices: cap 0 gives the count; sized call matches safe wrappers.
        let n_sel = unsafe { ygo_env_selected(raw, std::ptr::null_mut(), 0) };
        let mut sel = vec![CardRef { code: 0, controller: 0, location: 0, sequence: 0, position: 0 }; n_sel];
        if n_sel > 0 {
            assert_eq!(unsafe { ygo_env_selected(raw, sel.as_mut_ptr(), n_sel) }, n_sel);
        }
        if let Some(d) = env.decision() {
            assert_eq!(d.selected.len(), n_sel);
            assert_eq!(d.selected, sel);
        }
        let n_opt = unsafe { ygo_env_options(raw, std::ptr::null_mut(), 0) };
        let mut opts = vec![
            Opt {
                kind: 0,
                has_card: 0,
                code: 0,
                controller: 0,
                location: 0,
                sequence: 0,
                position: 0,
                opt_code: 0,
                desc: 0,
                arg: 0,
            };
            n_opt
        ];
        if n_opt > 0 {
            assert_eq!(unsafe { ygo_env_options(raw, opts.as_mut_ptr(), n_opt) }, n_opt);
        }
        if let Some(d) = env.decision() {
            assert_eq!(d.options.len(), n_opt);
            assert_eq!(d.options, opts);
        }
        let n_h = unsafe { ygo_env_history_len(raw) };
        let mut flat = vec![0i32; n_h * 14];
        if n_h > 0 {
            assert_eq!(unsafe { ygo_env_history_log(raw, flat.as_mut_ptr(), n_h) }, n_h);
        }
        assert_eq!(env.history_log().len(), n_h);
        let n_c = unsafe { ygo_env_num_choices(raw) };
        let mut ch = vec![0i32; n_c];
        if n_c > 0 {
            assert_eq!(unsafe { ygo_env_choices(raw, ch.as_mut_ptr(), n_c) }, n_c);
        }
        assert_eq!(env.choices().len(), n_c);
        // duel_state ok after reset.
        let st = env.duel_state().unwrap();
        let _ = (st.turn, st.phase, st.select_hint);
    }
}
