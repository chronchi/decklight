//! Offline EDOPro protocol core (A2b-1): framing/structs, per-seat filter, response inverse, decisions
//! from raw select messages, and `.yrpX` read/write/import. The Python implementation is the oracle;
//! every function here reproduces `src/decklight/edopro/*` and `src/decklight/play/yrp.py` byte for byte.

pub mod agent;
pub mod client;
pub mod discovery;
pub mod filter;
pub mod host;
pub mod inverse;
pub mod log;
pub mod protocol;
pub mod record;
pub mod yrp;

#[cfg(test)]
mod test_support {
    use ygoenv_sys::Engine;

    /// An empty engine: enough for decisions that do not need the card DB (yes/no, idle/battle, ...).
    pub fn engine() -> Engine {
        Engine::new()
    }
}
