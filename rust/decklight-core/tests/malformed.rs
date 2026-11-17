//! Malformed bytes must give `Err`, never a panic: the app parses what a remote peer or an old record sends.
use decklight_core::messages::{parse_one, split_buffer};
use decklight_core::query::{lps, parse_chain, parse_location, CardState, parse_card, QUERY_LOCATIONS};

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

#[test]
fn random_and_truncated_bytes_never_panic() {
    let mut rng = XorShift(0x9e3779b97f4a7c15);
    for ty in 0..=255u8 {
        for len in 0..64 {
            let mut raw = rng.bytes(len + 1);
            raw[0] = ty;
            let _ = parse_one(&raw);
            // small counts make the list paths go deep before running out
            for b in raw.iter_mut().skip(1) {
                *b &= 0x03;
            }
            let _ = parse_one(&raw);
        }
    }
    for len in 0..512 {
        let buf = rng.bytes(len);
        let _ = split_buffer(&buf);
        let _ = parse_chain(&buf);
        let _ = lps(&buf);
        let _ = parse_card(&buf, 0, &mut CardState::default());
        for &loc in QUERY_LOCATIONS.iter() {
            let _ = parse_location(&buf, 0, loc);
        }
        let zeros = vec![0u8; len];
        let _ = parse_chain(&zeros);
        let _ = lps(&zeros);
    }
}
