//! The play-time core shared by the phone app, the laptop server and the research code (ADR 0009).
//!
//! `constants` is generated from the Python module; `messages` and `query` are behaviour-for-behaviour ports of
//! `decklight.engine.messages` / `decklight.engine.query`, with canonical JSON used by the parity harnesses.

pub mod constants;
pub mod messages;
pub mod query;
