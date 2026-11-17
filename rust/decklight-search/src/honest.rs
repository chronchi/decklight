//! Port of `decklight.search.honest`: merge semantically identical candidates and override only on a paired,
//! significant gap (same float order as NumPy/Python).

use std::collections::HashSet;

use ygoenv_sys::Obs;

use crate::determinize::option_key;
use crate::select::{ACTION_COLS, mean, std_ddof1};

pub const TYPE_LINK: i32 = 0x0400_0000;
pub const KIND_PLACE: i32 = 19;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HonestRule {
    pub z: f64,
    pub min_gap: f64,
    pub s_max: usize,
    pub step: usize,
    pub merge: bool,
}

impl Default for HonestRule {
    fn default() -> Self {
        HonestRule { z: 2.0, min_gap: 0.05, s_max: 8, step: 4, merge: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Zone {
    Mz,
    Sz,
    Zn(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemKey {
    Option([i32; 8]),
    Place { kind: i32, code: i32, zone: Zone },
}

impl SemKey {
    /// Cross-language representation: `[0, k0..k7]` or `[1, kind, code, zone]` with zone 2=mz, 3=sz, else the
    /// sequence number.
    pub fn as_ints(&self) -> Vec<i64> {
        match self {
            SemKey::Option(k) => {
                let mut v = vec![0i64];
                v.extend(k.iter().map(|&x| x as i64));
                v
            }
            SemKey::Place { kind, code, zone } => {
                let z = match zone {
                    Zone::Mz => 2,
                    Zone::Sz => 3,
                    Zone::Zn(n) => *n as i64,
                };
                vec![1, *kind as i64, *code as i64, z]
            }
        }
    }
}

/// `_own_link_on_field`: a face-up Link monster I control (cards col 1 == 3 mzone, col 3 == 0 me, col 7 type).
pub fn own_link_on_field(obs: &Obs) -> bool {
    for i in 0..obs.card_mask.len() {
        if obs.card_mask[i] == 0 {
            continue;
        }
        let r = &obs.cards[i * 21..(i + 1) * 21];
        if r[1] == 3 && r[3] == 0 && (r[7] & TYPE_LINK) != 0 {
            return true;
        }
    }
    false
}

fn row(obs: &Obs, i: usize) -> [i32; ACTION_COLS] {
    let mut out = [0i32; ACTION_COLS];
    out.copy_from_slice(&obs.actions[i * ACTION_COLS..(i + 1) * ACTION_COLS]);
    out
}

/// Meaning of option `i` for merging (`option_key`, except PLACE options when no own Link monster is on the field).
pub fn semantic_key(obs: &Obs, i: usize, links: Option<bool>) -> SemKey {
    let row = row(obs, i);
    if row[0] == KIND_PLACE {
        let links = links.unwrap_or_else(|| own_link_on_field(obs));
        if !links {
            let zn = row[5];
            let zk = if zn <= 4 && zn != 2 {
                Zone::Mz
            } else if (8..=12).contains(&zn) {
                Zone::Sz
            } else {
                Zone::Zn(zn)
            };
            return SemKey::Place { kind: KIND_PLACE, code: row[8], zone: zk };
        }
    }
    SemKey::Option(option_key(&row))
}

/// Keep the first option of each semantic key, in `order`'s order.
pub fn merge_candidates(order: &[usize], obs: &Obs, links: Option<bool>) -> Vec<usize> {
    let links = links.unwrap_or_else(|| own_link_on_field(obs));
    let mut seen: HashSet<SemKey> = HashSet::new();
    let mut out = Vec::new();
    for &i in order {
        let k = semantic_key(obs, i, Some(links));
        if seen.insert(k) {
            out.push(i);
        }
    }
    out
}

/// Mean and SE of `a[s] - b[s]` over the common samples. SE = inf for n < 2.
pub fn paired_gap(a: &[f64], b: &[f64]) -> (f64, f64, usize) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, f64::INFINITY, 0);
    }
    let d: Vec<f64> = (0..n).map(|i| a[i] - b[i]).collect();
    let m = mean(&d);
    let se = if n >= 2 { std_ddof1(&d) / (n as f64).sqrt() } else { f64::INFINITY };
    (m, se, n)
}

pub fn significant(m: f64, se: f64, rule: &HonestRule) -> bool {
    m >= (rule.z * se).max(rule.min_gap)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub chosen_j: usize,
    pub challenger_j: Option<usize>,
    pub gap: f64,
    pub se: f64,
    pub n: usize,
    pub status: &'static str,
}

fn means(samples: &[Vec<f64>]) -> Vec<f64> {
    samples
        .iter()
        .map(|s| if s.is_empty() { f64::NEG_INFINITY } else { mean(s) })
        .collect()
}

fn best_other(means: &[f64], samples: &[Vec<f64>], pick_j: usize) -> Option<usize> {
    let mut c: Option<usize> = None;
    for j in 0..samples.len() {
        if j == pick_j || samples[j].is_empty() {
            continue;
        }
        c = Some(match c {
            None => j,
            Some(c) => {
                if means[j] > means[c] || (means[j] == means[c] && j < c) {
                    j
                } else {
                    c
                }
            }
        });
    }
    c
}

pub fn decide(samples: &[Vec<f64>], pick_j: usize, rule: &HonestRule) -> Decision {
    let means = means(samples);
    let Some(c) = best_other(&means, samples, pick_j) else {
        return Decision {
            chosen_j: pick_j,
            challenger_j: None,
            gap: 0.0,
            se: f64::INFINITY,
            n: 0,
            status: "keep",
        };
    };
    let (m, se, n) = paired_gap(&samples[c], &samples[pick_j]);
    if m <= 0.0 {
        Decision { chosen_j: pick_j, challenger_j: Some(c), gap: m, se, n, status: "keep" }
    } else if significant(m, se, rule) {
        Decision { chosen_j: c, challenger_j: Some(c), gap: m, se, n, status: "override" }
    } else {
        Decision { chosen_j: pick_j, challenger_j: Some(c), gap: m, se, n, status: "undecided" }
    }
}

pub fn decide_pair(samples: &[Vec<f64>], pick_j: usize, c: usize, rule: &HonestRule) -> Decision {
    let (m, se, n) = paired_gap(&samples[c], &samples[pick_j]);
    if m <= 0.0 {
        Decision { chosen_j: pick_j, challenger_j: Some(c), gap: m, se, n, status: "keep" }
    } else if significant(m, se, rule) {
        Decision { chosen_j: c, challenger_j: Some(c), gap: m, se, n, status: "override" }
    } else {
        Decision { chosen_j: pick_j, challenger_j: Some(c), gap: m, se, n, status: "undecided" }
    }
}

pub fn decisive_label(samples: &[Vec<f64>], rule: &HonestRule) -> Decision {
    let means = means(samples);
    let live: Vec<usize> = (0..samples.len()).filter(|&j| !samples[j].is_empty()).collect();
    if live.len() < 2 {
        return Decision {
            chosen_j: live.first().copied().unwrap_or(0),
            challenger_j: None,
            gap: 0.0,
            se: f64::INFINITY,
            n: 0,
            status: "undecided",
        };
    }
    let mut order = live;
    order.sort_by(|&a, &b| match means[b].partial_cmp(&means[a]) {
        Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
        Some(o) => o,
    });
    let (best, second) = (order[0], order[1]);
    let (m, se, n) = paired_gap(&samples[best], &samples[second]);
    let status = if significant(m, se, rule) { "label" } else { "undecided" };
    Decision { chosen_j: best, challenger_j: Some(second), gap: m, se, n, status }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_gap_uses_common_samples() {
        let (m, se, n) = paired_gap(&[1.0, 2.0, 3.0], &[0.0, 0.0]);
        assert_eq!(n, 2);
        assert!((m - 1.5).abs() < 1e-12);
        assert!(se.is_finite());
    }

    #[test]
    fn fewer_than_two_samples_has_infinite_se() {
        let (_, se, n) = paired_gap(&[1.0], &[0.0]);
        assert_eq!(n, 1);
        assert!(se.is_infinite());
    }

    #[test]
    fn decide_keeps_without_others() {
        let d = decide(&[vec![0.0], vec![]], 0, &HonestRule::default());
        assert_eq!(d.status, "keep");
        assert_eq!(d.challenger_j, None);
    }
}
