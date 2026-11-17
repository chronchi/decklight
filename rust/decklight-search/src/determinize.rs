//! Port of `decklight.search.determinize`: honest (determinized) worlds by rejection sampling.
//!
//! The Rust search uses prefix replays (the C++ core cannot clone), which is bit-identical to the Python
//! `YGO_SEARCH_CLONE=0` / C++-core path. Every source of nondeterminism (the permutation RNG, the slot order used by
//! `_propose_mains`) is reproduced exactly so the worlds match the Python oracle.

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};
use ygoenv_sys::{DebugCard, DuelConfig, Env, Engine, Obs, TCG_FLAGS};

use crate::pyrandom::{Random, State};

const _DECK: u32 = 0x1;
const _HAND: u32 = 0x2;
const _MZONE: u32 = 0x4;
const _SZONE: u32 = 0x8;
const _GRAVE: u32 = 0x10;
const _REMOVED: u32 = 0x20;
const _EXTRA: u32 = 0x40;
const _POS_FACEUP: u32 = 0x5;
const TYPE_MONSTER: u32 = 0x1;
const TYPE_SPELL: u32 = 0x2;
const TYPE_TRAP: u32 = 0x4;

/// Columns of an option row that describe what the option *means* (`_CMP_COLS`).
pub const CMP_COLS: [usize; 8] = [0, 2, 3, 4, 5, 6, 7, 8];

pub fn option_key(row: &[i32]) -> [i32; 8] {
    [row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]
}

fn action_row(obs: &Obs, i: usize) -> &[i32] {
    &obs.actions[i * 10..i * 10 + 10]
}

pub fn key_rank_of_choice(obs: &Obs, idx: usize) -> ([i32; 8], i32) {
    let key = option_key(action_row(obs, idx));
    let mut rank = 0i32;
    for j in 0..idx {
        if obs.action_mask[j] != 0 && option_key(action_row(obs, j)) == key {
            rank += 1;
        }
    }
    (key, rank)
}

/// Legal option with this key and rank; the last copy when fewer exist; None when no legal option has the key.
pub fn find_by_key_rank(obs: &Obs, key: &[i32; 8], rank: i32) -> Option<usize> {
    let mut matches = Vec::new();
    for j in 0..obs.action_mask.len() {
        if obs.action_mask[j] != 0 && option_key(action_row(obs, j)) == *key {
            matches.push(j);
        }
    }
    if matches.is_empty() {
        return None;
    }
    if (rank as usize) < matches.len() {
        Some(matches[rank as usize])
    } else {
        Some(*matches.last().unwrap())
    }
}

fn cmp_actions_equal(a: &Obs, b: &Obs) -> bool {
    for row in 0..a.actions.len() / 10 {
        for &c in &CMP_COLS {
            if a.actions[row * 10 + c] != b.actions[row * 10 + c] {
                return false;
            }
        }
    }
    true
}

/// Exact equality of two observations except `priv_code` (the privileged-critic channel the agent never sees).
pub fn obs_equal(a: Option<&Obs>, b: Option<&Obs>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.cards == b.cards
                && a.card_mask == b.card_mask
                && a.card_known == b.card_known
                && a.known_code == b.known_code
                && a.global == b.global
                && a.chain == b.chain
                && a.chain_mask == b.chain_mask
                && a.actions == b.actions
                && a.action_mask == b.action_mask
                && a.history == b.history
                && a.history_mask == b.history_mask
                && a.hist_long == b.hist_long
                && a.hist_long_mask == b.hist_long_mask
                && a.obs_history == b.obs_history
                && a.obs_history_mask == b.obs_history_mask
        }
        _ => false,
    }
}

/// `me`'s root-obs comparison for semantic replay: `actions` matches on the meaning columns only, everything else
/// exactly (except `priv_code`).
pub fn root_obs_equal(a: Option<&Obs>, b: Option<&Obs>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.cards == b.cards
                && a.card_mask == b.card_mask
                && a.card_known == b.card_known
                && a.known_code == b.known_code
                && a.global == b.global
                && a.chain == b.chain
                && a.chain_mask == b.chain_mask
                && cmp_actions_equal(a, b)
                && a.action_mask == b.action_mask
                && a.history == b.history
                && a.history_mask == b.history_mask
                && a.hist_long == b.hist_long
                && a.hist_long_mask == b.hist_long_mask
                && a.obs_history == b.obs_history
                && a.obs_history_mask == b.obs_history_mask
        }
        _ => false,
    }
}

fn is_public(loc: u32, pos: u32, is_pub: bool) -> bool {
    if loc == _GRAVE {
        return true;
    }
    if is_pub {
        return true;
    }
    if loc == _HAND {
        return false;
    }
    (pos & _POS_FACEUP) != 0
}

/// Cardids player `me` can identify; `update` after every step (once seen, always seen).
#[derive(Clone, Debug, Default)]
pub struct SeenTracker {
    pub me: u8,
    pub seen: HashSet<u32>,
}

impl SeenTracker {
    pub fn new(me: u8) -> SeenTracker {
        SeenTracker { me, seen: HashSet::new() }
    }

    pub fn update(&mut self, env: &Env<'_>) {
        for c in env.cards() {
            let p = c.controller;
            let loc = c.location;
            if p as u8 == self.me {
                if loc == _HAND || loc == _MZONE || loc == _SZONE || loc == _GRAVE || loc == _REMOVED {
                    self.seen.insert(c.cardid);
                } else if loc == _EXTRA && (c.position & _POS_FACEUP) != 0 {
                    self.seen.insert(c.cardid);
                }
            } else if loc == _HAND {
                if c.known != 0 {
                    self.seen.insert(c.cardid);
                }
            } else if is_public(loc, c.position, c.is_public != 0) {
                self.seen.insert(c.cardid);
            }
        }
    }
}

/// The config the world replay needs. Main decks are in the order they reach the core (already shuffled by the
/// caller), exactly the Python `DuelConfig` after `main_of`. As in Python's `_build_config`, determinized worlds
/// reset `seed_state` / `team_order` / `edopro_add` to their defaults; only `seed, flags, lp, hand, draw` carry over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldConfig {
    pub main0: Vec<u32>,
    pub extra0: Vec<u32>,
    pub main1: Vec<u32>,
    pub extra1: Vec<u32>,
    pub seed: u64,
    pub flags: u64,
    pub lp: u32,
    pub hand: u32,
    pub draw: u32,
    pub seed_state: Option<[u64; 4]>,
    pub team_order: [u8; 2],
    pub edopro_add: bool,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
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

impl WorldConfig {
    pub fn to_env_config(&self) -> DuelConfig {
        DuelConfig {
            main0: self.main0.clone(),
            extra0: self.extra0.clone(),
            main1: self.main1.clone(),
            extra1: self.extra1.clone(),
            seed: self.seed,
            flags: self.flags,
            lp: self.lp,
            hand: self.hand,
            draw: self.draw,
            seed_state: self.seed_state,
            team_order: self.team_order,
            edopro_add: self.edopro_add,
        }
    }

    /// `_build_config(mains)`: the proposed main orders, keeping the other duel fields but resetting the replay-only
    /// fields exactly like the Python constructor call.
    pub fn with_mains(&self, mains: &[Vec<u32>; 2]) -> WorldConfig {
        WorldConfig {
            main0: mains[0].clone(),
            extra0: self.extra0.clone(),
            main1: mains[1].clone(),
            extra1: self.extra1.clone(),
            seed: self.seed,
            flags: self.flags,
            lp: self.lp,
            hand: self.hand,
            draw: self.draw,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotState {
    location: u32,
    position: u32,
    code: u32,
    known: bool,
    is_public: bool,
}

/// One accepted world and its own option indices.
#[derive(Clone, Debug, PartialEq)]
pub struct DetResult {
    pub world: WorldConfig,
    pub resolved: Vec<i32>,
}

impl DetResult {
    pub fn mains_hash(&self) -> [u8; 32] {
        hash_mains(&self.world.main0, &self.world.main1)
    }
}

/// One parallel proposal check: the rejection reason, or the accepted world.
struct CheckOutcome {
    reason: Option<String>,
    result: Option<DetResult>,
}

pub fn hash_mains(main0: &[u32], main1: &[u32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((main0.len() as u64).to_le_bytes());
    for &x in main0 {
        h.update(x.to_le_bytes());
    }
    h.update((main1.len() as u64).to_le_bytes());
    for &x in main1 {
        h.update(x.to_le_bytes());
    }
    h.finalize().into()
}

/// SHA-256 of `getstate()`'s 625 words (little-endian); `gauss_next` is unused by the search.
pub fn hash_state(s: &State) -> [u8; 32] {
    let mut h = Sha256::new();
    for w in s.words() {
        h.update(w.to_le_bytes());
    }
    h.finalize().into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    Monster,
    Spell,
    None,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reasons {
    pub no_key: usize,
    pub replay_error: usize,
    pub obs_mismatch: usize,
}

impl Reasons {
    pub fn add(&mut self, key: &str) {
        match key {
            "no_key" => self.no_key += 1,
            "replay_error" => self.replay_error += 1,
            "obs_mismatch" => self.obs_mismatch += 1,
            _ => {}
        }
    }
}

pub struct Determinizer<'e> {
    pub engine: &'e Engine,
    pub config: WorldConfig,
    pub auto_single: bool,
    pub me: u8,
    pub prefix: Vec<i32>,
    pub prefix_keys: Option<Vec<([i32; 8], i32)>>,
    pub seen: HashSet<u32>,
    pub root_obs: Obs,
    pub rng: Random,
    pub tries: usize,
    pub proposals: usize,
    pub accepted: usize,
    pub fallbacks: usize,
    pub reasons: Reasons,
    /// Speed plan E6: replay/check proposals on scoped threads (1 = sequential, identical).
    pub replay_threads: usize,
    /// One entry per attempted proposal: the mains used (`Some`, hash) or the identity fallback (`None`).
    pub proposals_log: Vec<Option<[u8; 32]>>,
    pub base: [u32; 2],
    pub true_main: [Vec<u32>; 2],
    cur: [Vec<(u32, SlotState)>; 2],
    pending_proposal: Option<[Vec<u32>; 2]>,
}

impl<'e> Determinizer<'e> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &'e Engine,
        config: WorldConfig,
        auto_single: bool,
        me: u8,
        prefix: &[i32],
        seen: &HashSet<u32>,
        root_obs: &Obs,
        rng: Random,
        tries: usize,
        prefix_keys: Option<&[([i32; 8], i32)]>,
    ) -> Result<Determinizer<'e>, String> {
        let env_cfg = config.to_env_config();
        let mut probe = Env::new(engine, &env_cfg, auto_single)?;
        let mut obs = Obs::zeroed();
        let _ = probe.reset(&mut obs)?;
        let table: Vec<DebugCard> = probe.cards();

        let mut base = [0u32; 2];
        let mut true_main: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for team in 0..2usize {
            let main = if team == 0 { config.main0.clone() } else { config.main1.clone() };
            let mut deck_ids: Vec<u32> =
                table.iter().filter(|c| c.controller as usize == team && c.location == _DECK).map(|c| c.cardid).collect();
            deck_ids.sort_unstable();
            let hand = table
                .iter()
                .filter(|c| c.controller as usize == team && c.location == _HAND)
                .count();
            if deck_ids.len() + hand < main.len() {
                return Err(format!("determinizer: team {team} has too few main-deck cards"));
            }
            let b = *deck_ids.first().ok_or_else(|| format!("determinizer: team {team} has no deck card"))?;
            let want: HashSet<u32> = (b..b + main.len() as u32).collect();
            let mut got: Vec<(u32, u32)> = Vec::new();
            for loc in [_DECK, _HAND] {
                for c in table.iter().filter(|c| c.controller as usize == team && c.location == loc) {
                    if want.contains(&c.cardid) {
                        got.push((c.cardid, c.code));
                    }
                }
            }
            if got.len() != want.len() {
                return Err(format!("team {team}: main block is not consecutive"));
            }
            got.sort_unstable_by_key(|(id, _)| *id);
            let codes: Vec<u32> = got.iter().map(|(_, c)| *c).collect();
            if codes != main {
                return Err(format!("team {team}: codes != main"));
            }
            base[team] = b;
            true_main[team] = main;
        }

        let mut live = Env::new(engine, &env_cfg, auto_single)?;
        let mut obs = Obs::zeroed();
        let _ = live.reset(&mut obs)?;
        for c in prefix {
            let _ = live.step(*c, &mut obs)?;
        }
        let cur_table = live.cards();
        let mut cur: [Vec<(u32, SlotState)>; 2] = [Vec::new(), Vec::new()];
        for c in cur_table {
            let p = c.controller as usize;
            if p > 1 {
                continue;
            }
            let slot = c.cardid.wrapping_sub(base[p]);
            if (slot as usize) < true_main[p].len() {
                cur[p].push((
                    slot,
                    SlotState {
                        location: c.location,
                        position: c.position,
                        code: c.code,
                        known: c.known != 0,
                        is_public: c.is_public != 0,
                    },
                ));
            }
        }

        Ok(Determinizer {
            engine,
            config,
            auto_single,
            me,
            prefix: prefix.to_vec(),
            prefix_keys: prefix_keys.map(|k| k.to_vec()),
            seen: seen.clone(),
            root_obs: (*root_obs).clone(),
            rng,
            tries,
            proposals: 0,
            accepted: 0,
            fallbacks: 0,
            reasons: Reasons::default(),
            replay_threads: 1,
            proposals_log: Vec::new(),
            base,
            true_main,
            cur,
            pending_proposal: None,
        })
    }

    fn slot(&self, team: usize, slot: u32) -> Option<&SlotState> {
        self.cur[team].iter().find(|(s, _)| *s == slot).map(|(_, st)| st)
    }

    pub fn hidden_slots(&self, team: usize) -> Vec<u32> {
        let mut out = Vec::new();
        for (slot, st) in &self.cur[team] {
            if self.seen.contains(&(self.base[team] + slot)) {
                continue;
            }
            let loc = st.location;
            let hidden = if team == self.me as usize {
                loc == _DECK
            } else if loc == _DECK || loc == _HAND {
                true
            } else {
                (loc == _MZONE || loc == _SZONE) && !is_public(loc, st.position, st.is_public)
            };
            if hidden {
                out.push(*slot);
            }
        }
        out
    }

    fn code_kind(&self, code: u32) -> u8 {
        match self.engine.card_type(code) {
            None => 0,
            Some(t) if t & TYPE_MONSTER != 0 => 1,
            Some(t) if t & (TYPE_SPELL | TYPE_TRAP) != 0 => 2,
            Some(_) => 0,
        }
    }

    fn slot_need(&self, team: usize, slot: u32) -> Need {
        match self.slot(team, slot).map(|s| s.location) {
            Some(_MZONE) => Need::Monster,
            Some(_SZONE) => Need::Spell,
            _ => Need::None,
        }
    }

    /// Permuted main-deck orders, or None when no valid permutation was found (identity).
    fn propose_mains(&mut self) -> Option<[Vec<u32>; 2]> {
        let mut out: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for team in 0..2usize {
            let hidden = self.hidden_slots(team);
            if hidden.is_empty() {
                out[team] = self.true_main[team].clone();
                continue;
            }
            let codes: Vec<u32> =
                hidden.iter().map(|&s| self.slot(team, s).map(|x| x.code).unwrap_or(0)).collect();
            let needs: Vec<(u32, Need)> = hidden.iter().map(|&s| (s, self.slot_need(team, s))).collect();
            let m_slots: Vec<u32> =
                hidden.iter().copied().filter(|s| self.slot_need(team, *s) == Need::Monster).collect();
            let s_slots: Vec<u32> =
                hidden.iter().copied().filter(|s| self.slot_need(team, *s) == Need::Spell).collect();
            let kinds: HashMap<u32, u8> = {
                let uniq: HashSet<u32> = codes.iter().copied().collect();
                uniq.into_iter().map(|c| (c, self.code_kind(c))).collect()
            };
            let mut m_codes: Vec<u32> = codes.iter().copied().filter(|c| kinds[c] == 1).collect();
            let mut s_codes: Vec<u32> = codes.iter().copied().filter(|c| kinds[c] == 2).collect();
            if m_codes.len() < m_slots.len() || s_codes.len() < s_slots.len() {
                return None; // keep identity (always a valid world)
            }
            self.rng.shuffle(&mut m_codes);
            self.rng.shuffle(&mut s_codes);
            let mut rest: Vec<u32> = codes.iter().copied().filter(|c| kinds[c] == 0).collect();
            rest.extend_from_slice(&m_codes[m_slots.len()..]);
            rest.extend_from_slice(&s_codes[s_slots.len()..]);
            self.rng.shuffle(&mut rest);
            let free: Vec<u32> = hidden
                .iter()
                .copied()
                .filter(|s| needs.iter().any(|(n, need)| n == s && *need == Need::None))
                .collect();
            if rest.len() != free.len() {
                return None;
            }
            let mut assign: Vec<(u32, u32)> = Vec::new();
            for (i, &s) in m_slots.iter().enumerate() {
                assign.push((s, m_codes[i]));
            }
            for (i, &s) in s_slots.iter().enumerate() {
                assign.push((s, s_codes[i]));
            }
            for (i, &s) in free.iter().enumerate() {
                assign.push((s, rest[i]));
            }
            let mut new_main = self.true_main[team].clone();
            for (s, co) in assign {
                new_main[s as usize] = co;
            }
            out[team] = new_main;
        }
        Some(out)
    }

    fn log_proposal(&mut self) {
        let h = self.propose_mains();
        self.proposals_log.push(h.as_ref().map(|m| hash_mains(&m[0], &m[1])));
        self.pending_proposal = h;
    }

    /// One honest world + its own option indices, or None after `tries` rejections.
    pub fn sample(&mut self) -> Option<DetResult> {
        let semantic = self.prefix_keys.is_some();
        for _ in 0..self.tries {
            self.proposals += 1;
            self.log_proposal();
            let mains = self.pending_proposal.clone().unwrap_or_else(|| {
                [self.true_main[0].clone(), self.true_main[1].clone()]
            });
            let world = self.config.with_mains(&mains);
            let built = Env::new(self.engine, &world.to_env_config(), self.auto_single);
            let mut env = match built {
                Ok(e) => e,
                Err(_) => {
                    self.reasons.replay_error += 1;
                    continue;
                }
            };
            let mut obs = Obs::zeroed();
            let outcome: Result<(Option<Obs>, Vec<i32>, Option<String>), String> = if semantic {
                let keys_ranks = self.prefix_keys.as_ref().unwrap();
                let keys: Vec<[i32; 8]> = keys_ranks.iter().map(|(k, _)| *k).collect();
                let ranks: Vec<i32> = keys_ranks.iter().map(|(_, r)| *r).collect();
                env.replay_keys(&keys, &ranks, &mut obs).map(|r| {
                    let reason = if r.reason.is_empty() { None } else { Some(r.reason) };
                    let o = if r.ok { Some((*obs).clone()) } else { None };
                    (o, r.resolved, reason)
                })
            } else {
                let resolved = self.prefix.clone();
                env.replay(&self.prefix, &mut obs).map(|ok| {
                    let o = if ok { Some((*obs).clone()) } else { None };
                    (o, resolved, None)
                })
            };
            let (observed, resolved, failed) = match outcome {
                Ok(v) => v,
                Err(_) => {
                    self.reasons.replay_error += 1;
                    continue;
                }
            };
            if let Some(reason) = failed {
                self.reasons.add(&reason);
                continue;
            }
            let ok = if semantic {
                root_obs_equal(observed.as_ref(), Some(&self.root_obs))
            } else {
                obs_equal(observed.as_ref(), Some(&self.root_obs))
            };
            if ok {
                self.accepted += 1;
                return Some(DetResult { world, resolved });
            }
            self.reasons.obs_mismatch += 1;
        }
        self.fallbacks += 1;
        None
    }

    /// Replay + root-obs check for one proposed main-deck order. Read-only (fresh `Env` per call), so proposals can
    /// be checked on scoped threads.
    fn check_proposal(&self, mains: &[Vec<u32>; 2]) -> CheckOutcome {
        let world = self.config.with_mains(mains);
        let mut env = match Env::new(self.engine, &world.to_env_config(), self.auto_single) {
            Ok(e) => e,
            Err(_) => return CheckOutcome { reason: Some("replay_error".to_string()), result: None },
        };
        let mut obs = Obs::zeroed();
        let (observed, resolved, failed) = if let Some(keys_ranks) = self.prefix_keys.as_ref() {
            let keys: Vec<[i32; 8]> = keys_ranks.iter().map(|(k, _)| *k).collect();
            let ranks: Vec<i32> = keys_ranks.iter().map(|(_, r)| *r).collect();
            match env.replay_keys(&keys, &ranks, &mut obs) {
                Ok(r) => {
                    let reason = if r.reason.is_empty() { None } else { Some(r.reason) };
                    let o = if r.ok { Some((*obs).clone()) } else { None };
                    (o, r.resolved, reason)
                }
                Err(_) => return CheckOutcome { reason: Some("replay_error".to_string()), result: None },
            }
        } else {
            let resolved = self.prefix.clone();
            match env.replay(&self.prefix, &mut obs) {
                Ok(ok) => ((if ok { Some((*obs).clone()) } else { None }), resolved, None),
                Err(_) => return CheckOutcome { reason: Some("replay_error".to_string()), result: None },
            }
        };
        if let Some(reason) = failed {
            return CheckOutcome { reason: Some(reason), result: None };
        }
        let ok = if self.prefix_keys.is_some() {
            root_obs_equal(observed.as_ref(), Some(&self.root_obs))
        } else {
            obs_equal(observed.as_ref(), Some(&self.root_obs))
        };
        if ok {
            CheckOutcome { reason: None, result: Some(DetResult { world, resolved }) }
        } else {
            CheckOutcome { reason: Some("obs_mismatch".to_string()), result: None }
        }
    }

    /// `n` results of `sample()`, bit-identical to `(0..n).map(|_| self.sample())`.
    ///
    /// With `replay_threads > 1`: proposals are still generated sequentially with the RNG (recording its state
    /// after each), in chunks of the thread count, and only the replay+check runs in parallel. Results are scanned
    /// in proposal order: at the first accepted proposal `i` only proposals `0..i` are counted, the RNG is rewound to
    /// the state after `i`, and sampling continues. A chunk with no acceptance counts all its proposals and the next
    /// chunk is generated, until `tries` is used up (then a fallback with the RNG after the last proposal).
    pub fn sample_many(&mut self, n: usize) -> Vec<Option<DetResult>> {
        let threads = self.replay_threads;
        if threads <= 1 {
            return (0..n).map(|_| self.sample()).collect();
        }
        let chunk_size = threads.max(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut result = None;
            let mut remaining = self.tries;
            let mut done = false;
            while remaining > 0 && !done {
                let size = chunk_size.min(remaining);
                let mut generated: Vec<([Vec<u32>; 2], State)> = Vec::with_capacity(size);
                for _ in 0..size {
                    let mains = self.propose_mains().unwrap_or_else(|| {
                        [self.true_main[0].clone(), self.true_main[1].clone()]
                    });
                    self.proposals_log.push(Some(hash_mains(&mains[0], &mains[1])));
                    generated.push((mains, self.rng.getstate()));
                }
                remaining -= size;
                let this: &Determinizer<'e> = self;
                let checked = crate::parallel::map(threads, size, |i| this.check_proposal(&generated[i].0));
                for (i, c) in checked.iter().enumerate() {
                    if c.result.is_none() {
                        continue;
                    }
                    for prev in &checked[..i] {
                        self.proposals += 1;
                        if let Some(r) = prev.reason.as_deref() {
                            self.reasons.add(r);
                        }
                    }
                    self.proposals += 1;
                    self.accepted += 1;
                    self.rng.setstate(&generated[i].1);
                    result = c.result.clone();
                    done = true;
                    break;
                }
                if !done {
                    for c in &checked {
                        debug_assert!(c.result.is_none());
                        self.proposals += 1;
                        if let Some(r) = c.reason.as_deref() {
                            self.reasons.add(r);
                        }
                    }
                    if remaining == 0 {
                        self.fallbacks += 1;
                        done = true;
                    }
                }
            }
            out.push(result);
        }
        out
    }

    pub fn rng_state_hash(&self) -> [u8; 32] {
        hash_state(&self.rng.getstate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_key_skips_row_of_and_step() {
        let row = [19, 7, 123, 4, 5, 6, 16, 3, 1, 99];
        assert_eq!(option_key(&row), [19, 123, 4, 5, 6, 16, 3, 1]);
    }

    #[test]
    fn find_by_key_rank_uses_rank_then_last_copy() {
        let mut o = Obs::zeroed();
        o.action_mask[0] = 1;
        o.action_mask[2] = 1;
        o.actions[0] = 19;
        o.actions[2] = 19;
        o.actions[2 + 2] = 123;
        let key = option_key(&o.actions[0..10]);
        assert_eq!(find_by_key_rank(&o, &key, 0), Some(0));
        // The second copy has a different meaning column, so only one match: rank 1 falls back to the last copy.
        assert_eq!(find_by_key_rank(&o, &key, 1), Some(0));
        let key2 = option_key(&o.actions[20..30]);
        assert_eq!(find_by_key_rank(&o, &key2, 0), Some(2));
        let missing = [7; 8];
        assert_eq!(find_by_key_rank(&o, &missing, 0), None);
    }

    #[test]
    fn obs_equal_skips_priv_code_and_root_skips_action_rows() {
        let a = Obs::zeroed();
        let mut b = Obs::zeroed();
        b.priv_code[0] = 12345;
        assert!(obs_equal(Some(&a), Some(&b)));
        b.actions[1] = 99; // column 1 (row_of) is not compared by root_obs_equal
        assert!(!obs_equal(Some(&a), Some(&b)));
        assert!(root_obs_equal(Some(&a), Some(&b)));
        b.actions[2] = 7; // meaning column
        assert!(!root_obs_equal(Some(&a), Some(&b)));
        assert!(obs_equal(None, None));
        assert!(!obs_equal(Some(&a), None));
    }

    #[test]
    fn mains_hash_distinguishes_order_and_boundary() {
        let a = hash_mains(&[1, 2], &[3]);
        let b = hash_mains(&[1], &[2, 3]);
        let c = hash_mains(&[1, 2], &[3]);
        assert_ne!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn state_hash_is_stable() {
        let s = Random::from_i128(5).getstate();
        assert_eq!(hash_state(&s), hash_state(&s.clone()));
    }
}
