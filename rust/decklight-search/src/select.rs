//! Selection helpers: `_masked` (search/turn1.py), `argmax_select` / `softmax_conf` (search/play.py), and the
//! NumPy-compatible float64 reduction they depend on.
//!
//! `np.sum` is a pairwise summation (blocks of 8 over <= 128 elements, recursively halved above), not a left fold;
//! matching it bit for bit is required by the parity gate.

use std::cmp::Ordering;

pub const KIND_PASS: i32 = 11;
pub const ACTION_COLS: usize = 10;
pub const MAX_OPTIONS: usize = 64;

/// `_masked`: truncate to the action-mask length, float64, and set illegal entries to -inf.
pub fn masked(logits: &[f32], action_mask: &[bool]) -> Result<Vec<f64>, String> {
    let n = action_mask.len();
    if logits.len() < n {
        return Err(format!("masked: {} logits for a {n}-entry action mask", logits.len()));
    }
    let mut z: Vec<f64> = logits[..n].iter().map(|&x| x as f64).collect();
    for (i, &legal) in action_mask.iter().enumerate() {
        if !legal {
            z[i] = f64::NEG_INFINITY;
        }
    }
    Ok(z)
}

/// `_masked` when the logits are already float64 (fixtures, tests).
pub fn masked_f64(logits: &[f64], action_mask: &[bool]) -> Result<Vec<f64>, String> {
    let n = action_mask.len();
    if logits.len() < n {
        return Err(format!("masked_f64: {} logits for a {n}-entry action mask", logits.len()));
    }
    let mut z = logits[..n].to_vec();
    for (i, &legal) in action_mask.iter().enumerate() {
        if !legal {
            z[i] = f64::NEG_INFINITY;
        }
    }
    Ok(z)
}

/// `np.argmax`: the first index holding the maximum.
pub fn argmax_select(z: &[f64]) -> Option<usize> {
    let mut best = 0usize;
    let mut found = !z.is_empty();
    for (i, &v) in z.iter().enumerate() {
        if !found {
            best = i;
            found = true;
        } else if v > z[best] {
            best = i;
        }
    }
    found.then_some(best)
}

/// NumPy's pairwise summation for float64 (`npy_pairwise_sum`).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8usize;
        while i + 8 <= n {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// `float(np.mean(a))` for a non-empty float64 array.
pub fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `float(np.std(a, ddof=1))`: two-pass NumPy variance, then sqrt.
pub fn std_ddof1(a: &[f64]) -> f64 {
    let n = a.len();
    if n - 1 <= 0 {
        return f64::NAN;
    }
    let m = mean(a);
    let sq: Vec<f64> = a.iter().map(|&x| {
        let d = x - m;
        d * d
    }).collect();
    (pairwise_sum(&sq) / (n - 1) as f64).sqrt()
}

/// `softmax_conf`: max softmax probability over the finite entries (1.0 when <= 1 finite entry).
pub fn softmax_conf(z: &[f64]) -> f64 {
    let finite: Vec<f64> = z.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() <= 1 {
        return 1.0;
    }
    let mx = finite.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap();
    let v: Vec<f64> = finite.iter().map(|&x| x - mx).collect();
    let e: Vec<f64> = v.iter().map(|&x| x.exp()).collect();
    let s = pairwise_sum(&e);
    let mut out = f64::NEG_INFINITY;
    for &x in &e {
        let p = x / s;
        if p > out {
            out = p;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_uses_first_mask_length() {
        let mask = [true, false, true, false];
        let z = masked(&[1.5, 2.5, 3.5, 4.5, 5.5], &mask).unwrap();
        assert_eq!(z, vec![1.5, f64::NEG_INFINITY, 3.5, f64::NEG_INFINITY]);
        assert!(masked(&[1.5], &mask).is_err());
    }

    #[test]
    fn argmax_first_max() {
        assert_eq!(argmax_select(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax_select(&[]), None);
    }

    #[test]
    fn pairwise_small_is_left_fold() {
        assert_eq!(pairwise_sum(&[0.1, 0.2, 0.3]), (0.0 + 0.1 + 0.2 + 0.3) - 0.0);
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;

    #[test]
    fn softmax_uniform_and_single() {
        assert!((softmax_conf(&[0.0, 0.0, 0.0, 0.0]) - 0.25).abs() < 1e-15);
        assert_eq!(softmax_conf(&[f64::NEG_INFINITY, 2.0, f64::NEG_INFINITY]), 1.0);
        assert_eq!(softmax_conf(&[f64::NEG_INFINITY, f64::NEG_INFINITY]), 1.0);
    }

    #[test]
    fn pairwise_sum_128_path() {
        let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(pairwise_sum(&x), 5050.0);
    }

    #[test]
    fn std_of_constant_is_zero() {
        assert_eq!(std_ddof1(&[3.0, 3.0, 3.0]), 0.0);
    }
}
