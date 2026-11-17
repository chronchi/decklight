//! Match record v1 (ADR 0010): the shared writer lives in `decklight-record` so the EDOPro LAN host writes the
//! exact same document. Re-exported here so existing `crate::record::*` callers are unchanged.

pub use decklight_record::*;
