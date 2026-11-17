//! Deterministic building blocks of the product search, bit-identical to the Python oracle
//! (`src/decklight/search/{play,window,determinize,honest}.py`, `src/decklight/eval/loopguard.py`).

pub mod determinize;
pub mod honest;
pub mod loopguard;
pub mod parallel;
pub mod pyrandom;
pub mod select;
pub mod play;
pub mod window;
