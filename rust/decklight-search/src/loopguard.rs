//! Port of `decklight.eval.loopguard`: mask options already taken in an exact decision state.
//!
//! The Python `state_key` is `hash(bytes)` (salted per process); it is only ever used as an in-process dict key, so
//! this port returns a stable SHA-256 digest of the same concatenated bytes. Equality of keys -- the only property the
//! guard relies on -- is identical, and the digest is what the parity dump compares across languages.

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};
use ygoenv_sys::Obs;

use crate::select::KIND_PASS;

pub const MSG_SELECT_CHAIN: i32 = 0x10;
pub const STATE_KEYS: [&str; 7] =
    ["cards", "card_mask", "global", "chain", "chain_mask", "actions", "action_mask"];

pub type StateKey = [u8; 32];

/// The key's bytes in `STATE_KEYS` order (exactly `b"".join(obs[k].tobytes() ...)`).
pub fn state_key_bytes(obs: &Obs) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        obs.cards.len() * 4
            + obs.card_mask.len()
            + obs.global.len() * 4
            + obs.chain.len() * 4
            + obs.chain_mask.len()
            + obs.actions.len() * 4
            + obs.action_mask.len(),
    );
    for x in &obs.cards {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(&obs.card_mask);
    for x in &obs.global {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for x in &obs.chain {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(&obs.chain_mask);
    for x in &obs.actions {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(&obs.action_mask);
    out
}

pub fn state_key(obs: &Obs) -> StateKey {
    let mut h = Sha256::new();
    h.update(state_key_bytes(obs));
    h.finalize().into()
}

#[derive(Clone, Debug, Default)]
pub struct LoopGuard {
    pub turn: HashMap<i32, i32>,
    pub seen: HashMap<(i32, StateKey), HashSet<i32>>,
    pub masked: u64,
}

impl LoopGuard {
    pub fn new() -> LoopGuard {
        LoopGuard::default()
    }

    pub fn reset(&mut self) {
        self.turn.clear();
        self.seen.clear();
    }

    /// `z` with the options already taken in this exact state set to -inf, plus the state key. `z` holds -inf on
    /// illegal options. Call [`note`](Self::note) with the option finally played.
    pub fn mask(&mut self, slot: i32, obs: &Obs, z: &[f64]) -> (Vec<f64>, StateKey) {
        let turn = obs.global[2];
        if self.turn.get(&slot) != Some(&turn) {
            self.turn.insert(slot, turn);
            self.seen.retain(|(s, _), _| *s != slot);
        }
        let key = state_key(obs);
        let mut taken = self.seen.get(&(slot, key)).cloned().unwrap_or_default();
        if !taken.is_empty() && obs.global[21] == MSG_SELECT_CHAIN {
            let actions = &obs.actions;
            let rows = actions.len() / 10;
            taken.retain(|&a| a < 0 || a as usize >= rows || actions[a as usize * 10] != KIND_PASS);
        }
        if !taken.is_empty() {
            let mut z2 = z.to_vec();
            for &a in &taken {
                if a >= 0 && (a as usize) < z2.len() {
                    z2[a as usize] = f64::NEG_INFINITY;
                }
            }
            if z2.iter().any(|v| v.is_finite()) {
                self.masked += 1;
                return (z2, key);
            }
        }
        (z.to_vec(), key)
    }

    pub fn note(&mut self, slot: i32, key: StateKey, action: i32) {
        self.seen.entry((slot, key)).or_default().insert(action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs() -> Obs {
        let mut o = Obs::zeroed();
        o.global[2] = 1;
        o.action_mask[0] = 1;
        o.action_mask[1] = 1;
        *o
    }

    #[test]
    fn masks_options_already_taken() {
        let o = obs();
        let mut g = LoopGuard::new();
        let z = [0.5, 0.4, f64::NEG_INFINITY];
        let (z1, k1) = g.mask(0, &o, &z);
        assert_eq!(z1, z);
        g.note(0, k1, 0);
        let (z2, _) = g.mask(0, &o, &z);
        assert_eq!(z2[0], f64::NEG_INFINITY);
        assert_eq!(z2[1], 0.4);
        assert_eq!(g.masked, 1);
    }

    #[test]
    fn all_options_taken_leaves_z_untouched() {
        let o = obs();
        let mut g = LoopGuard::new();
        let z = [0.5, 0.4];
        let (_, k) = g.mask(0, &o, &z);
        g.note(0, k, 0);
        g.note(0, k, 1);
        let (z2, _) = g.mask(0, &o, &z);
        assert_eq!(z2, z);
        assert_eq!(g.masked, 0);
    }

    #[test]
    fn pass_at_select_chain_is_not_masked() {
        let mut o = obs();
        o.global[21] = MSG_SELECT_CHAIN;
        o.actions[0] = KIND_PASS;
        let mut g = LoopGuard::new();
        let z = [0.5, 0.4];
        let (_, k) = g.mask(0, &o, &z);
        g.note(0, k, 0);
        let (z2, _) = g.mask(0, &o, &z);
        assert_eq!(z2, z);
    }

    #[test]
    fn turn_change_clears_slot() {
        let mut o = obs();
        let mut g = LoopGuard::new();
        let z = [0.5, 0.4];
        let (_, k) = g.mask(0, &o, &z);
        g.note(0, k, 0);
        o.global[2] = 2;
        let (z2, _) = g.mask(0, &o, &z);
        assert_eq!(z2, z);
    }
}
