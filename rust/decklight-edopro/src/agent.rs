//! Agents for the network host and client.
//!
//! `StreamBot` is the deterministic test player of `src/decklight/edopro/bot.py` (and `scripts/app/play_dump.py`):
//! option = splitmix64 stream % number of options, RPS = 1 + stream % 3. The same stream on both ends lets a Python
//! and a Rust process play the same game, which is what the parity gates compare.

/// One seat's decision policy, independent of where the options come from (host env or received select message).
pub trait Agent: Send {
    /// Index into the `n_options` legal options of the pending decision.
    fn act(&mut self, n_options: usize) -> usize;
    /// Rock/paper/scissors throw (1 scissors / 2 rock / 3 paper).
    fn rps(&mut self) -> u8;
    /// Turn-order choice after winning RPS (the host's `bot_second` overrides this for a bot seat).
    fn go_first(&mut self) -> bool {
        true
    }
}

/// `bot.StreamBot`: splitmix64 stream, byte-identical to the Python and `scripts/app/play_dump.py` streams.
#[derive(Clone, Debug)]
pub struct StreamBot {
    state: u64,
}

impl StreamBot {
    pub fn new(seed: u64) -> StreamBot {
        StreamBot { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Agent for StreamBot {
    fn act(&mut self, n_options: usize) -> usize {
        if n_options == 0 {
            return 0;
        }
        (self.next_u64() % n_options as u64) as usize
    }

    fn rps(&mut self) -> u8 {
        1 + (self.next_u64() % 3) as u8
    }
}

/// A uniform-ish random legal player (xorshift64*); not used by the parity gates, which need `StreamBot`.
#[derive(Clone, Debug)]
pub struct RandomAgent {
    state: u64,
}

impl RandomAgent {
    pub fn new(seed: u64) -> RandomAgent {
        RandomAgent { state: seed | 1 }
    }

    pub fn from_time() -> RandomAgent {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        RandomAgent::new(ns ^ (std::process::id() as u64).rotate_left(32))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

impl Agent for RandomAgent {
    fn act(&mut self, n_options: usize) -> usize {
        if n_options == 0 {
            return 0;
        }
        (self.next_u64() % n_options as u64) as usize
    }

    fn rps(&mut self) -> u8 {
        1 + (self.next_u64() % 3) as u8
    }

    fn go_first(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streambot_matches_the_python_stream() {
        // bot.StreamBot(0)._next() sequence (splitmix64 with the golden increment).
        let mut b = StreamBot::new(0);
        assert_eq!(b.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(b.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(b.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn act_and_rps_are_in_range() {
        let mut b = StreamBot::new(7);
        for n in 1..=64 {
            for _ in 0..20 {
                assert!(b.act(n) < n);
                assert!((1..=3).contains(&b.rps()));
            }
        }
    }
}
