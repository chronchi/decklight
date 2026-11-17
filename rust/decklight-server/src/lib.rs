//! decklight web server (A2c): the unchanged `web/` client plays a full game over the Python protocol, plus match
//! record v1 and the SQLite history index.

pub mod artsync;
pub mod bgjob;
pub mod ask;
pub mod cards_search;
pub mod decks;
pub mod describe;
pub mod describe_enc;
pub mod images;
pub mod product;
pub mod raw;
pub mod record;
pub mod room;
pub mod server;
pub mod userdecks;

pub use server::{App, Config, build_app, build_state, run};
