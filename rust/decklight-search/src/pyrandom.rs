//! An exact port of CPython 3.14's `random.Random` (Mersenne Twister) as the search uses it.
//!
//! The search draws `random()`, `shuffle()` and saves/restores the state (adaptive rounds, `sample_many`
//! rewinding); seeds are ints (`seed * 1_000_003 + game_seed`) and strings (`f"{seed}:{game_seed}:{k}:det"`).
//! Every operation mirrors `Modules/_randommodule.c`: `random()` is the two-draw 27/26-bit expression, not
//! `getrandbits(53)`; int seeds take the absolute value and little-endian 32-bit chunks; string seeds append the
//! SHA-512 digest and read the result big-endian (seed version 2).

use sha2::{Digest, Sha512};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

const INIT_GENRAND: u32 = 18_124_332_53;
const INIT_BY_ARRAY_1: u32 = 1_664_525;
const INIT_BY_ARRAY_2: u32 = 1_566_083_941;

/// A cloneable `random.Random` state: the 624 words, the index, and `gauss_next` (unused by the search).
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub mt: [u32; N],
    pub index: usize,
    pub gauss_next: Option<f64>,
}

impl State {
    /// CPython `getstate()[1]`: the 624 words followed by the index.
    pub fn words(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(N + 1);
        out.extend_from_slice(&self.mt);
        out.push(self.index as u32);
        out
    }
}

#[derive(Clone, Debug)]
pub struct Random {
    mt: [u32; N],
    index: usize,
    gauss_next: Option<f64>,
}

impl Random {
    /// Low-level `init_by_array` (the key is little-endian 32-bit chunks, at least one).
    pub fn from_words(key: &[u32]) -> Random {
        let key: &[u32] = if key.is_empty() { &[0] } else { key };
        let mut mt = [0u32; N];
        init_genrand(&mut mt, 19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = N.max(key.len());
        while k > 0 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(INIT_BY_ARRAY_1)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(INIT_BY_ARRAY_2)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x8000_0000;
        Random { mt, index: N, gauss_next: None }
    }

    /// `random.seed(int)`: absolute value, little-endian 32-bit chunks, `keyused = ceil(bits/32)` (min 1).
    pub fn from_i128(seed: i128) -> Random {
        let mut v = seed.unsigned_abs();
        let mut words = Vec::new();
        while v > 0 {
            words.push((v & 0xffff_ffff) as u32);
            v >>= 32;
        }
        if words.is_empty() {
            words.push(0);
        }
        Random::from_words(&words)
    }

    pub fn from_u128(seed: u128) -> Random {
        Random::from_i128(seed as i128)
    }

    /// `random.seed(str)` (version 2): `int.from_bytes(s.encode() + sha512(s.encode()).digest(), "big")`.
    pub fn from_str(s: &str) -> Random {
        let bytes = s.as_bytes();
        let mut digest = Sha512::new();
        digest.update(bytes);
        let mut all = bytes.to_vec();
        all.extend_from_slice(&digest.finalize());
        Random::from_words(&big_endian_words(&all))
    }

    /// CPython `random_random`: `(a * 2^27 + b) * 2^-53` with `a = genrand >> 5`, `b = genrand >> 6`.
    pub fn random(&mut self) -> f64 {
        let a = self.genrand_uint32() >> 5;
        let b = self.genrand_uint32() >> 6;
        (a as f64 * 67_108_864.0 + b as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// CPython `getrandbits(k)`, for any k up to 128.
    pub fn getrandbits(&mut self, k: u32) -> u128 {
        if k == 0 {
            return 0;
        }
        let words = k.div_ceil(32);
        let mut out = 0u128;
        for i in 0..words {
            let mut r = self.genrand_uint32();
            let rem = k - 32 * i;
            if rem < 32 {
                r >>= 32 - rem;
            }
            out |= (r as u128) << (32 * i);
        }
        out
    }

    /// CPython `_randbelow_with_getrandbits`: rejection sampling on `getrandbits(n.bit_length())`.
    pub fn randbelow(&mut self, n: u128) -> u128 {
        if n == 0 {
            return 0;
        }
        let k = 128 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    pub fn randrange(&mut self, n: u128) -> u128 {
        self.randbelow(n)
    }

    /// CPython `shuffle`: `for i in reversed(range(1, len)): j = _randbelow(i+1); swap`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u128 + 1) as usize;
            x.swap(i, j);
        }
    }

    pub fn getstate(&self) -> State {
        State { mt: self.mt, index: self.index, gauss_next: self.gauss_next }
    }

    pub fn setstate(&mut self, s: &State) {
        self.mt = s.mt;
        self.index = s.index;
        self.gauss_next = s.gauss_next;
    }

    fn genrand_uint32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn twist(&mut self) {
        let mt = &mut self.mt;
        let mut kk = 0usize;
        while kk < N - M {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            kk += 1;
        }
        while kk < N - 1 {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            kk += 1;
        }
        let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
        self.index = 0;
    }
}

fn init_genrand(mt: &mut [u32; N], s: u32) {
    mt[0] = s;
    for i in 1..N {
        // CPython: (1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i), the xor inside the multiply.
        mt[i] = INIT_GENRAND.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
    }
}

/// Big-endian byte string -> little-endian 32-bit words, high zero words trimmed (at least one word).
fn big_endian_words(bytes: &[u8]) -> Vec<u32> {
    let mut words = Vec::new();
    let mut idx = bytes.len();
    while idx > 0 {
        let start = idx.saturating_sub(4);
        let mut w = 0u32;
        for &b in &bytes[start..idx] {
            w = (w << 8) | b as u32;
        }
        words.push(w);
        idx = start;
    }
    while words.len() > 1 && *words.last().unwrap() == 0 {
        words.pop();
    }
    if words.is_empty() {
        words.push(0);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_zero_and_one_differ_from_genrand() {
        let a = Random::from_i128(0).random();
        let b = Random::from_i128(1).random();
        assert_ne!(a, b);
    }

    #[test]
    fn negative_seed_is_absolute_value() {
        let mut a = Random::from_i128(-12345);
        let mut b = Random::from_i128(12345);
        for _ in 0..50 {
            assert_eq!(a.random(), b.random());
        }
    }

    #[test]
    fn state_round_trip() {
        let mut r = Random::from_str("7:3:5:det");
        for _ in 0..7 {
            r.random();
        }
        let s = r.getstate();
        let a: Vec<f64> = (0..10).map(|_| r.random()).collect();
        let mut r2 = Random::from_i128(999);
        r2.setstate(&s);
        let b: Vec<f64> = (0..10).map(|_| r2.random()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn cpython_known_values() {
        let mut r = Random::from_i128(0);
        let bits: Vec<u64> = (0..3).map(|_| r.random().to_bits()).collect();
        assert_eq!(bits, vec![4605781095417019838, 4605002265878567962, 4601247963976755576]);
        let mut r = Random::from_i128(0);
        assert_eq!(
            [1u32, 5, 32, 53, 70].iter().map(|&k| r.getrandbits(k)).collect::<Vec<u128>>(),
            vec![1, 12, 3255389356, 3788173568575106, 1130027559504484052788]
        );
        let mut r = Random::from_i128(0);
        let mut l: Vec<u32> = (0..10).collect();
        r.shuffle(&mut l);
        assert_eq!(l, vec![7, 8, 1, 5, 3, 4, 2, 0, 9, 6]);
        let mut r = Random::from_str("7:3:5:det");
        assert_eq!(r.random().to_bits(), 4602303679683967182);
        let mut r = Random::from_i128((1i128 << 64) + 5);
        assert_eq!(r.random().to_bits(), 4602774100721505825);
    }

    #[test]
    fn shuffle_empty_and_one() {
        let mut r = Random::from_i128(3);
        let mut a: [i32; 0] = [];
        r.shuffle(&mut a);
        let mut b = [1];
        r.shuffle(&mut b);
        assert_eq!(b, [1]);
    }
}
