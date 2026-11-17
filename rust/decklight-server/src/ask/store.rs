//! The chat store (docs/plans/ask-tab.md §Core design 2–3, decision 1 "yes"): SQLite in the app data folder, chats
//! never leave the device. Messages form a tree (each row names its parent), so edit-and-resend and regenerate are
//! branches; a chat's `head` is the leaf the next message goes under. Rows are append-only with stable ids (uuid7), in
//! OpenAI message format, so export or a later sync stay possible; only the chat row (head, title, tokens) changes.
//!
//! The context builder is a pure function (branch, budget) → request messages. Compaction v1: old tool results are
//! left out first (tools can be called again), then whole old turns; both are counted so the chat shows it.

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};

/// Context and cost limits (decision 3: a token cap per chat).
#[derive(Clone, Debug)]
pub struct Policy {
    /// Estimated tokens per request (system prompt and tool specs not counted).
    pub context_tokens: usize,
    /// Prompt + completion tokens a chat may spend in all; past it, the chat asks for a new one.
    pub chat_token_cap: u64,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy { context_tokens: 24_000, chat_token_cap: 300_000 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chat {
    pub id: String,
    pub created: String,
    pub updated: String,
    pub title: String,
    pub head: Option<String>,
    pub preset: String,
    pub model: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
}

impl Chat {
    pub fn tokens(&self) -> u64 {
        self.tokens_in + self.tokens_out
    }
}

/// One stored message.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub id: String,
    pub parent: Option<String>,
    pub created: String,
    /// user | assistant | tool, or `error` (a failed turn: shown, never sent to the model).
    pub role: String,
    /// The OpenAI-format message, or `{"error", "action"}` for an error row.
    pub body: Value,
    /// provider, model, prompt and tool versions, tokens, latency.
    pub meta: Value,
}

pub struct ChatStore {
    path: PathBuf,
}

/// uuid7 ids, strictly increasing in this process: two made in the same millisecond would otherwise sort at random,
/// and row order (oldest first) relies on id order.
fn next_id() -> String {
    static LAST: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
    let mut last = LAST.lock().unwrap();
    let mut id = decklight_record::uuid7();
    while id <= *last {
        id = decklight_record::uuid7();
    }
    last.clone_from(&id);
    id
}

fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

fn db_err(e: rusqlite::Error) -> String {
    format!("the chat history could not be read or saved ({e})")
}

/// The chat's title: its first question, cut at a word near 60 characters.
fn title_of(text: &str) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= 60 {
        return one_line;
    }
    let cut: String = one_line.chars().take(60).collect();
    let at = cut.rfind(' ').filter(|&i| i > 30).unwrap_or(cut.len());
    format!("{}…", &cut[..at])
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY, created TEXT NOT NULL, updated TEXT NOT NULL, title TEXT NOT NULL, head TEXT,
    preset TEXT NOT NULL, model TEXT NOT NULL, tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0,
    seq INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, chat TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, parent TEXT, created TEXT NOT NULL,
    role TEXT NOT NULL, body TEXT NOT NULL, meta TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat);
";

impl ChatStore {
    /// The store at `dir/chats.sqlite` (the folder is created).
    pub fn open(dir: &Path) -> Result<ChatStore, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let store = ChatStore { path: dir.join("chats.sqlite") };
        store.db()?.execute_batch(&format!("PRAGMA journal_mode = WAL; {SCHEMA}")).map_err(db_err)?;
        Ok(store)
    }

    fn db(&self) -> Result<Connection, String> {
        let con = Connection::open(&self.path).map_err(db_err)?;
        con.execute_batch("PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 3000;").map_err(db_err)?;
        Ok(con)
    }

    pub fn create(&self, preset: &str, model: &str) -> Result<Chat, String> {
        let t = now();
        let chat = Chat {
            id: next_id(),
            created: t.clone(),
            updated: t,
            title: String::new(),
            head: None,
            preset: preset.into(),
            model: model.into(),
            tokens_in: 0,
            tokens_out: 0,
        };
        self.db()?
            .execute(
                "INSERT INTO chats (id, created, updated, title, head, preset, model, seq) \
                 VALUES (?1, ?2, ?3, '', NULL, ?4, ?5, (SELECT coalesce(max(seq), 0) + 1 FROM chats))",
                params![chat.id, chat.created, chat.updated, chat.preset, chat.model],
            )
            .map_err(db_err)?;
        Ok(chat)
    }

    fn chat_from(r: &rusqlite::Row) -> rusqlite::Result<Chat> {
        Ok(Chat {
            id: r.get(0)?,
            created: r.get(1)?,
            updated: r.get(2)?,
            title: r.get(3)?,
            head: r.get(4)?,
            preset: r.get(5)?,
            model: r.get(6)?,
            tokens_in: r.get::<_, i64>(7)? as u64,
            tokens_out: r.get::<_, i64>(8)? as u64,
        })
    }

    const CHAT_COLS: &str = "id, created, updated, title, head, preset, model, tokens_in, tokens_out";

    pub fn get(&self, id: &str) -> Result<Chat, String> {
        self.db()?
            .query_row(&format!("SELECT {} FROM chats WHERE id = ?1", Self::CHAT_COLS), [id], Self::chat_from)
            .optional()
            .map_err(db_err)?
            .ok_or(format!("no chat {id:?}; it may have been deleted"))
    }

    /// Most recently changed first (by a change counter: timestamps tie within a millisecond).
    pub fn list(&self) -> Result<Vec<Chat>, String> {
        let con = self.db()?;
        let mut st = con
            .prepare(&format!("SELECT {} FROM chats ORDER BY seq DESC", Self::CHAT_COLS))
            .map_err(db_err)?;
        let rows = st.query_map([], Self::chat_from).and_then(|r| r.collect::<Result<Vec<_>, _>>()).map_err(db_err)?;
        Ok(rows)
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        let n = self.db()?.execute("DELETE FROM chats WHERE id = ?1", [id]).map_err(db_err)?;
        if n == 0 {
            return Err(format!("no chat {id:?}; it may have been deleted"));
        }
        Ok(())
    }

    fn row_from(r: &rusqlite::Row) -> rusqlite::Result<Row> {
        let body: String = r.get(4)?;
        let meta: String = r.get(5)?;
        Ok(Row {
            id: r.get(0)?,
            parent: r.get(1)?,
            created: r.get(2)?,
            role: r.get(3)?,
            body: serde_json::from_str(&body).unwrap_or(Value::Null),
            meta: serde_json::from_str(&meta).unwrap_or(Value::Null),
        })
    }

    /// Every message of a chat, oldest first (all branches).
    pub fn rows(&self, chat: &str) -> Result<Vec<Row>, String> {
        let con = self.db()?;
        let mut st = con
            .prepare("SELECT id, parent, created, role, body, meta FROM messages WHERE chat = ?1 ORDER BY id")
            .map_err(db_err)?;
        let rows = st.query_map([chat], Self::row_from).and_then(|r| r.collect::<Result<Vec<_>, _>>()).map_err(db_err)?;
        Ok(rows)
    }

    /// The path from the root to `leaf` (default: the chat's head), oldest first.
    pub fn branch(&self, chat: &str, leaf: Option<&str>) -> Result<Vec<Row>, String> {
        let c = self.get(chat)?;
        let Some(mut at) = leaf.map(str::to_string).or(c.head) else { return Ok(Vec::new()) };
        let all = self.rows(chat)?;
        let by_id: std::collections::HashMap<&str, &Row> = all.iter().map(|r| (r.id.as_str(), r)).collect();
        let mut path = Vec::new();
        loop {
            let r = by_id.get(at.as_str()).ok_or(format!("message {at:?} is not in this chat"))?;
            path.push((*r).clone());
            match &r.parent {
                Some(p) if path.len() <= all.len() => at = p.clone(),
                Some(_) => return Err("the chat history has a loop; delete this chat".into()),
                None => break,
            }
        }
        path.reverse();
        Ok(path)
    }

    /// Append `bodies` as a chain under `parent` (`None` = a new root; for the first message, or an edit of it) and
    /// make the last one the head. Returns the new ids. `role` of each row comes from its body (`error` rows set it).
    pub fn append(&self, chat: &str, parent: Option<&str>, bodies: &[Value], meta: &Value) -> Result<Vec<String>, String> {
        if bodies.is_empty() {
            return Err("nothing to store".into());
        }
        let mut con = self.db()?;
        // IMMEDIATE: a deferred transaction that reads first fails at once with "database is locked" when another
        // writer is active, whatever busy_timeout says (review 2026-11-14, measured 0.06 s).
        let tx = con.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(db_err)?;
        let exists: bool =
            tx.query_row("SELECT count(*) FROM chats WHERE id = ?1", [chat], |r| r.get::<_, i64>(0)).map_err(db_err)? > 0;
        if !exists {
            return Err(format!("no chat {chat:?}; it may have been deleted"));
        }
        if let Some(p) = parent {
            let ok: bool = tx
                .query_row("SELECT count(*) FROM messages WHERE id = ?1 AND chat = ?2", [p, chat], |r| r.get::<_, i64>(0))
                .map_err(db_err)?
                > 0;
            if !ok {
                return Err(format!("message {p:?} is not in this chat"));
            }
        }
        let mut ids = Vec::new();
        let mut up = parent.map(str::to_string);
        let t = now();
        for b in bodies {
            let id = next_id();
            let role = if b.get("error").is_some() { "error" } else { b["role"].as_str().unwrap_or("") };
            if !["user", "assistant", "tool", "error"].contains(&role) {
                return Err(format!("cannot store a message with role {role:?}"));
            }
            tx.execute(
                "INSERT INTO messages (id, chat, parent, created, role, body, meta) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![id, chat, up, t, role, b.to_string(), meta.to_string()],
            )
            .map_err(db_err)?;
            up = Some(id.clone());
            ids.push(id);
        }
        // The first question names the chat.
        // The first question names the chat; one with only attachments, by what is attached ("Photo", the deck's name).
        let first_user = bodies.iter().find(|b| b["role"] == "user").map(|b| match b["content"].as_str().map(str::trim) {
            Some(q) if !q.is_empty() => q.to_string(),
            _ => b["attach"].as_array().into_iter().flatten().filter_map(|a| a["label"].as_str()).collect::<Vec<_>>().join(", "),
        });
        if let Some(q) = first_user.filter(|q| !q.is_empty()) {
            tx.execute("UPDATE chats SET title = ?2 WHERE id = ?1 AND title = ''", params![chat, title_of(&q)])
                .map_err(db_err)?;
        }
        tx.execute(
            "UPDATE chats SET head = ?2, updated = ?3, seq = (SELECT max(seq) + 1 FROM chats) WHERE id = ?1",
            params![chat, up, t],
        )
        .map_err(db_err)?;
        tx.commit().map_err(db_err)?;
        Ok(ids)
    }

    /// Move the head (pick a branch, or go back to a message to edit and resend from it).
    pub fn set_head(&self, chat: &str, message: Option<&str>) -> Result<(), String> {
        if let Some(m) = message {
            self.branch(chat, Some(m))?;
        }
        let n = self
            .db()?
            .execute(
                "UPDATE chats SET head = ?2, updated = ?3, seq = (SELECT max(seq) + 1 FROM chats) WHERE id = ?1",
                params![chat, message, now()],
            )
            .map_err(db_err)?;
        if n == 0 {
            return Err(format!("no chat {chat:?}; it may have been deleted"));
        }
        Ok(())
    }

    /// The provider and model the chat's next answers use (switching after a limit error keeps the chat).
    pub fn set_model(&self, chat: &str, preset: &str, model: &str) -> Result<(), String> {
        let n = self
            .db()?
            .execute("UPDATE chats SET preset = ?2, model = ?3 WHERE id = ?1", params![chat, preset, model])
            .map_err(db_err)?;
        if n == 0 {
            return Err(format!("no chat {chat:?}; it may have been deleted"));
        }
        Ok(())
    }

    pub fn add_usage(&self, chat: &str, prompt: u64, completion: u64) -> Result<(), String> {
        let n = self
            .db()?
            .execute(
                "UPDATE chats SET tokens_in = tokens_in + ?2, tokens_out = tokens_out + ?3 WHERE id = ?1",
                params![chat, prompt as i64, completion as i64],
            )
            .map_err(db_err)?;
        if n == 0 {
            return Err(format!("no chat {chat:?}; it may have been deleted"));
        }
        Ok(())
    }
}

/// Rough token count of a message: 4 characters per token (only for the budget, never billed by it).
pub fn estimate(v: &Value) -> usize {
    v.to_string().chars().count().div_ceil(4)
}

/// The request messages for a branch, after compaction.
#[derive(Debug, PartialEq)]
pub struct Context {
    pub messages: Vec<Value>,
    /// Old tool results left out (the model is told it can call the tool again).
    pub elided: usize,
    /// Old messages not sent at all.
    pub dropped: usize,
}

/// What an elided tool result says in place of the result.
pub const ELIDED: &str = "result left out to save space; call the tool again if you need it";

/// `system` first, then the branch's sendable rows (error rows are not sent). Over `budget` estimated tokens: tool
/// results of earlier turns are elided, oldest first; then earlier turns are dropped whole, oldest first. The last
/// turn (from the last user message) is always sent whole.
pub fn build_context(system: &Value, branch: &[Row], budget: usize) -> Context {
    let mut msgs: Vec<Value> = branch.iter().filter(|r| r.role != "error").map(|r| r.body.clone()).collect();
    let last_turn = msgs.iter().rposition(|m| m["role"] == "user").unwrap_or(0);
    let total = |m: &[Value]| m.iter().map(estimate).sum::<usize>();
    let mut elided = 0;
    for i in 0..last_turn {
        if total(&msgs) <= budget {
            break;
        }
        if msgs[i]["role"] == "tool" && msgs[i]["content"] != json!(json!({"elided": true, "note": ELIDED}).to_string()) {
            msgs[i]["content"] = json!(json!({"elided": true, "note": ELIDED}).to_string());
            elided += 1;
        }
    }
    // Drop whole turns (a user message and everything up to the next one), so no tool result loses its call.
    let mut dropped = 0;
    let mut start = 0;
    let last_turn_len = msgs.len() - last_turn;
    while total(&msgs[start..]) > budget && msgs.len() - start > last_turn_len {
        let next = msgs[start + 1..].iter().position(|m| m["role"] == "user").map_or(msgs.len(), |j| start + 1 + j);
        dropped += next - start;
        start = next;
    }
    let mut out = vec![system.clone()];
    if dropped > 0 {
        let note = format!(
            "{}\n\n({dropped} earlier messages of this chat were left out to save space.)",
            system["content"].as_str().unwrap_or("")
        );
        out[0]["content"] = json!(note);
    }
    out.extend(msgs.drain(start..));
    Context { messages: out, elided, dropped }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (ChatStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("decklight-chat-{}", decklight_record::uuid7()));
        (ChatStore::open(&dir).unwrap(), dir)
    }

    fn user(t: &str) -> Value {
        json!({"role": "user", "content": t})
    }

    fn row(role: &str, body: Value) -> Row {
        Row { id: String::new(), parent: None, created: String::new(), role: role.into(), body, meta: json!({}) }
    }

    #[test]
    fn a_chat_is_a_tree_with_a_head() {
        let (s, dir) = store();
        let c = s.create("opencode-go", "deepseek-v4.1-flash").unwrap();
        let q1 = s.append(&c.id, None, &[user("What does Ash Blossom do?")], &json!({})).unwrap();
        let a1 = s.append(&c.id, Some(&q1[0]), &[json!({"role": "assistant", "content": "It negates searches."})], &json!({})).unwrap();
        // Edit and resend: a second question under the same parent (the root) is a new branch.
        let q1b = s.append(&c.id, None, &[user("What does Maxx C do?")], &json!({})).unwrap();
        let roles = |b: Vec<Row>| b.iter().map(|r| r.body["content"].as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert_eq!(roles(s.branch(&c.id, None).unwrap()), vec!["What does Maxx C do?"]);
        assert_eq!(roles(s.branch(&c.id, Some(&a1[0])).unwrap()), vec!["What does Ash Blossom do?", "It negates searches."]);
        s.set_head(&c.id, Some(&a1[0])).unwrap();
        assert_eq!(s.get(&c.id).unwrap().head.as_deref(), Some(a1[0].as_str()));
        assert_eq!(s.rows(&c.id).unwrap().len(), 3, "append-only: nothing replaced");
        assert_eq!(s.get(&c.id).unwrap().title, "What does Ash Blossom do?", "the first question names the chat");
        assert!(q1b[0] > q1[0], "uuid7 ids sort by time");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn wrong_ids_are_errors_with_a_reason() {
        let (s, dir) = store();
        let a = s.create("opencode-go", "m").unwrap();
        let b = s.create("opencode-go", "m").unwrap();
        let qa = s.append(&a.id, None, &[user("hi")], &json!({})).unwrap();
        assert!(s.append(&b.id, Some(&qa[0]), &[user("x")], &json!({})).unwrap_err().contains("not in this chat"));
        assert!(s.append("nope", None, &[user("x")], &json!({})).unwrap_err().contains("deleted"));
        assert!(s.append(&a.id, None, &[json!({"role": "system", "content": "x"})], &json!({})).unwrap_err().contains("role"));
        assert!(s.set_head(&b.id, Some(&qa[0])).is_err());
        s.delete(&a.id).unwrap();
        assert!(s.rows(&a.id).unwrap().is_empty(), "messages go with their chat");
        assert!(s.delete(&a.id).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn usage_adds_up_and_lists_are_newest_first() {
        let (s, dir) = store();
        let a = s.create("opencode-go", "m").unwrap();
        let b = s.create("opencode-go", "m").unwrap();
        s.add_usage(&a.id, 7000, 250).unwrap();
        s.add_usage(&a.id, 3000, 50).unwrap();
        assert_eq!(s.get(&a.id).unwrap().tokens(), 10_300);
        s.append(&a.id, None, &[user("later")], &json!({})).unwrap();
        let ids: Vec<String> = s.list().unwrap().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![a.id.clone(), b.id.clone()]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn titles_are_one_line_and_cut_at_a_word() {
        assert_eq!(title_of("  Is my\nBranded deck legal? "), "Is my Branded deck legal?");
        let long = "If I play nine hand traps in a forty card deck how often do I open at least one of them in five";
        let t = title_of(long);
        assert!(t.ends_with('…') && t.chars().count() <= 61 && !t.contains("of them"), "{t}");
    }

    fn turn(q: &str, tool_chars: usize) -> Vec<Row> {
        vec![
            row("user", user(q)),
            row("assistant", json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "card", "arguments": "{}"}}]})),
            row("tool", json!({"role": "tool", "tool_call_id": "c1", "content": "x".repeat(tool_chars)})),
            row("assistant", json!({"role": "assistant", "content": "answer"})),
        ]
    }

    #[test]
    fn context_is_whole_under_budget_and_skips_error_rows() {
        let sys = json!({"role": "system", "content": "be brief"});
        let mut b = turn("q1", 100);
        b.push(row("error", json!({"error": "HTTP 503", "action": "Retry"})));
        b.extend(turn("q2", 100));
        let c = build_context(&sys, &b, 10_000);
        assert_eq!((c.messages.len(), c.elided, c.dropped), (9, 0, 0));
        assert!(c.messages.iter().all(|m| m.get("error").is_none()));
    }

    #[test]
    fn compaction_elides_old_tool_results_then_drops_old_turns() {
        let sys = json!({"role": "system", "content": "be brief"});
        let mut b = turn("q1", 4000);
        b.extend(turn("q2", 4000));
        b.extend(turn("q3", 4000));
        // Budget for about one and a half turns: both older tool results are elided, nothing dropped.
        let c = build_context(&sys, &b, 1600);
        assert_eq!((c.elided, c.dropped), (2, 0));
        assert!(c.messages[3]["content"].as_str().unwrap().contains("elided"));
        assert_eq!(c.messages.last().unwrap()["content"], "answer");
        assert_eq!(c.messages[11]["content"].as_str().unwrap().len(), 4000, "the last turn is sent whole");
        // A budget below one turn: everything older goes, whole turns at a time, and the system prompt says so.
        let c = build_context(&sys, &b, 100);
        assert_eq!((c.dropped, c.messages.len()), (8, 5));
        assert_eq!(c.messages[1]["content"], "q3");
        assert!(c.messages[0]["content"].as_str().unwrap().contains("8 earlier messages"));
    }
}
