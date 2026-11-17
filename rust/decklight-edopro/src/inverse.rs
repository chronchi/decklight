//! Raw EDOPro response bytes -> env option-index sequences.
//!
//! Port of `src/decklight/engine/inverse.py`. The search runs on cloned C++ `Decision` objects
//! (`ygo_decision_clone`), never on the env: the engine, card DB and message stay shared and untouched.
//! It mirrors `PyEnv._settle`: single-option steps are taken automatically (and take no index) when
//! `auto_single`, otherwise options are cut at `MAX_OPTIONS`. Depth-first in ascending index order, so
//! the first hit is the same sequence Python returns (fewest steps in practice; ties lexicographic).

use ygoenv_sys::Decision;

pub const NODE_BUDGET: usize = 200_000;
const MAX_OPTIONS: usize = 64;

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub nodes: usize,
    pub bounded: bool,
    pub msg_type: i32,
}

/// Option indices that make the pending decision send exactly `resp`, or None.
pub fn choices_for_response(dec: &Decision, resp: &[u8], auto_single: bool) -> Option<Vec<usize>> {
    choices_for_response_stats(dec, resp, auto_single, None)
}

pub fn choices_for_response_stats(
    dec: &Decision,
    resp: &[u8],
    auto_single: bool,
    stats: Option<&mut Stats>,
) -> Option<Vec<usize>> {
    let mut local = Stats { msg_type: dec.msg_type(), ..Default::default() };
    let out = search(dec, auto_single, resp, &mut local);
    if let Some(s) = stats {
        *s = local;
    }
    out
}

fn search(dec: &Decision, auto_single: bool, resp: &[u8], stats: &mut Stats) -> Option<Vec<usize>> {
    let mut stack: Vec<(Decision, Vec<usize>)> = vec![(dec.clone_handle().ok()?, Vec::new())];
    let mut nodes = 0usize;
    while let Some((mut d, seq)) = stack.pop() {
        nodes += 1;
        if nodes > NODE_BUDGET {
            stats.bounded = true;
            stats.nodes = nodes;
            return None;
        }
        let opts = d.options();
        if opts.is_empty() {
            continue;
        }
        if auto_single && opts.len() == 1 {
            match d.choose(0) {
                Ok(Some(out)) => {
                    if out == resp {
                        stats.nodes = nodes;
                        return Some(seq);
                    }
                }
                Ok(None) => stack.push((d, seq)),
                Err(_) => {}
            }
            continue;
        }
        let n = opts.len().min(MAX_OPTIONS);
        for i in (0..n).rev() {
            let mut d2 = match d.clone_handle() {
                Ok(x) => x,
                Err(_) => continue,
            };
            match d2.choose(i as i32) {
                Ok(Some(out)) => {
                    if out == resp {
                        stats.nodes = nodes + 1;
                        let mut s = seq.clone();
                        s.push(i);
                        return Some(s);
                    }
                }
                Ok(None) => {
                    let mut s = seq.clone();
                    s.push(i);
                    stack.push((d2, s));
                }
                Err(_) => {}
            }
        }
    }
    stats.nodes = nodes;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::engine;

    #[test]
    fn fixed_decision_inverts_its_response() {
        let e = engine();
        // MSG_SELECT_YESNO: type, player, desc u64 -> two options (YES=1, NO=0).
        let mut msg = vec![0x0D, 0];
        msg.extend_from_slice(&0u64.to_le_bytes());
        let d = Decision::new(&e, &msg, &[]).unwrap();
        assert_eq!(d.options().len(), 2);
        assert_eq!(choices_for_response(&d, &1i32.to_le_bytes(), false), Some(vec![0]));
        assert_eq!(choices_for_response(&d, &0i32.to_le_bytes(), false), Some(vec![1]));
        assert_eq!(choices_for_response(&d, &[9, 9], false), None);
    }

    /// Client mode builds decisions from a remote server's bytes: truncated or random select messages must be
    /// errors (the C++ reader is bounds-checked), never an over-read or a crash.
    #[test]
    fn malformed_select_messages_are_errors() {
        let e = engine();
        // MSG_SELECT_EFFECTYN: type, player, card info (code u32, con u8, loc u8, seq u32, pos u32), desc u64.
        let mut full = vec![0x0C, 0];
        full.extend_from_slice(&89631139u32.to_le_bytes());
        full.extend_from_slice(&[0, 4]);
        full.extend_from_slice(&0u32.to_le_bytes());
        full.extend_from_slice(&1u32.to_le_bytes());
        full.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(Decision::new(&e, &full, &[]).unwrap().options().len(), 2);
        for n in 1..full.len() {
            assert!(Decision::new(&e, &full[..n], &[]).is_err(), "truncated at {n} accepted");
        }
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for t in 0..=255u8 {
            for _ in 0..200 {
                let len = 1 + (next() % 48) as usize;
                let mut m: Vec<u8> = (0..len).map(|_| next() as u8).collect();
                m[0] = t;
                if let Ok(mut d) = Decision::new(&e, &m, &[89631139]) {
                    let n = d.options().len();
                    for i in 0..n.min(3) {
                        let _ = d.choose(i as i32);
                    }
                    let _ = d.choose(n as i32 + 5);
                }
            }
        }
    }
}
