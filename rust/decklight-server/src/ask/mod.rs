//! Ask tab (docs/plans/ask-tab.md): the chat loop, its provider and retry seams, and the tools.

pub mod attach;
pub mod chat;
pub mod eval;
pub mod export;
pub mod games;
pub mod host;
pub mod keys;
pub mod odds;
pub mod ondevice;
pub mod presets;
pub mod provider;
pub mod replay;
pub mod retry;
pub mod store;
pub mod turn;
pub mod ws;
pub mod stats;

#[cfg(test)]
mod tests;
