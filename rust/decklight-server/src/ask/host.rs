//! The v1 tools on the real server (config/ask/tools.v1.json): card search and card text from the search index,
//! the user's saved decks with legality, `deck_odds` on a saved deck, `my_stats` on the record index, `replay` and
//! `bot_view` on saved games (`replay.rs`). Results are
//! compact JSON for the model; specs and the system prompt are compiled in, so the APK needs no extra files.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::{Value, json};

use super::chat::Tools;
use crate::server::App;
use crate::userdecks::{Format as DeckFormat, StoredDeck};

pub const TOOLS_JSON: &str = include_str!("../../../../config/ask/tools.v1.json");
pub const SYSTEM_PROMPT: &str = include_str!("../../../../config/ask/prompts/system.v2.md");
pub const SYSTEM_PROMPT_VERSION: &str = "system.v2";

/// The tool specs in OpenAI function format, from tools.v1.json.
pub fn specs() -> Vec<Value> {
    let cfg: Value = serde_json::from_str(TOOLS_JSON).expect("config/ask/tools.v1.json is valid JSON");
    cfg["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            json!({"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"],
            }})
        })
        .collect()
}

/// Effect text shown to the model is capped (one card can run to 1500 characters).
const TEXT_CAP: usize = 1200;

/// BabelCDB ends some card texts with a provenance note ("* The above text is unofficial and describes the card's
/// functionality in the OCG.", and variants); models repeat it in every answer (eval 2026-11-14), so it is dropped.
const UNOFFICIAL_NOTE: &str = "* The above text is unofficial";

/// A whole number, also when the model writes it as a float (6.0).
pub(crate) fn whole(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().filter(|f| f.fract() == 0.0 && *f >= 0.0 && *f <= u32::MAX as f64).map(|f| f as u64))
}

fn plain(text: &str) -> &str {
    let t = text.trim_end();
    match t.rfind(UNOFFICIAL_NOTE) {
        Some(i) if !t[i..].contains('\n') => t[..i].trim_end(),
        _ => t,
    }
}

pub struct AppTools {
    app: Arc<App>,
    specs: Vec<Value>,
}

impl AppTools {
    pub fn new(app: Arc<App>) -> AppTools {
        AppTools { app, specs: specs() }
    }

    fn names(&self) -> &BTreeMap<u32, String> {
        self.app.ctx.texts.names()
    }

    /// A saved deck by id, or by its name (case-insensitive): models often pass the name they were shown.
    fn find_deck(&self, key: &str) -> Result<StoredDeck, String> {
        let key = key.trim();
        if let Ok(d) = self.app.user_decks.get(key) {
            return Ok(d);
        }
        let all = self.app.user_decks.list();
        let found: Vec<&StoredDeck> = all.iter().filter(|d| d.meta.name.trim().eq_ignore_ascii_case(key)).collect();
        match found[..] {
            [d] => Ok(d.clone()),
            [] => Err(format!("no saved deck {key:?}; call deck without an id to list them")),
            _ => Err(format!("several decks are named {key:?}; use the id from the deck list")),
        }
    }

    fn search_cards(&self, args: &Value) -> Result<Value, String> {
        let text = match args.get("text") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.trim().to_string(),
            Some(v) => return Err(format!("`text` must be text, got {v}")),
        };
        let filters = args.get("filters").cloned().unwrap_or(Value::Null);
        if text.is_empty() && filters.as_object().is_none_or(|f| f.is_empty()) {
            return Err("give `text` (a card name or words from its text) or `filters`".into());
        }
        let limit = match args.get("limit") {
            None | Some(Value::Null) => 10,
            Some(v) => whole(v).filter(|n| (1..=20).contains(n)).ok_or(format!("`limit` must be 1 to 20, got {v}"))?,
        } as usize;
        let index = self.app.search_index();
        let want_describe = args.get("mode").and_then(Value::as_str) == Some("describe");
        let mut note = None;
        let result = if want_describe && self.app.describe.state.installed() {
            let emb = self.app.describe.embeddings().map_err(|e| format!("describe search unavailable: {e}"))?;
            let app = self.app.clone();
            index.search_describe(&text, &filters, 0, limit, &emb, |clause| {
                app.describe.encode(&format!("{}{clause}", crate::describe_enc::INSTRUCT))
            })?
        } else {
            if want_describe {
                note = Some("describe search is not installed on this device; this is a name and text search");
            }
            index.search(&text, &filters, 0, limit)?
        };
        let rows: Vec<Value> = result
            .rows
            .iter()
            .map(|r| {
                let mut row = json!({"code": r.code, "name": r.name, "line": r.line});
                let snippet = plain(&r.snippet);
                if !snippet.is_empty() {
                    row["snippet"] = json!(snippet);
                }
                row
            })
            .collect();
        let mut out = json!({"total": result.total, "rows": rows});
        if let Some(n) = note {
            out["note"] = json!(n);
        }
        Ok(out)
    }

    fn card(&self, args: &Value) -> Result<Value, String> {
        let code = args
            .get("code")
            .and_then(|v| whole(v).or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())))
            .filter(|n| *n > 0 && *n <= u32::MAX as u64)
            .ok_or("code must be a card code from search_cards")?;
        let c = self.app.search_index().card_json(code as u32).ok_or(format!("unknown card {code}"))?;
        let mut text = plain(c["text"].as_str().unwrap_or("")).to_string();
        if text.chars().count() > TEXT_CAP {
            text = text.chars().take(TEXT_CAP).collect::<String>() + " …";
        }
        Ok(json!({
            "code": c["code"], "name": c["name"], "kind": c["kind"], "line": c["line"], "text": text,
            "archetypes": c["archetypes"],
        }))
    }

    fn deck(&self, args: &Value) -> Result<Value, String> {
        let format = DeckFormat::default();
        let legal = |d: &StoredDeck| crate::userdecks::legality(&d.deck, &self.app.cards, self.names(), &format);
        let key = match args.get("id") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.trim()).filter(|s| !s.is_empty()),
            Some(v) => return Err(format!("`id` must be a deck id from the deck list, got {v}")),
        };
        let Some(key) = key else {
            let decks: Vec<Value> = self
                .app
                .user_decks
                .list()
                .iter()
                .map(|d| {
                    json!({"id": d.id, "name": d.meta.name, "main": d.deck.main.len(), "extra": d.deck.extra.len(),
                           "side": d.deck.side.len(), "legal": legal(d).ok})
                })
                .collect();
            return Ok(json!({"decks": decks}));
        };
        let d = self.find_deck(key)?;
        let group = |codes: &[u32]| -> Vec<Value> {
            let mut order: Vec<u32> = Vec::new();
            let mut n: BTreeMap<u32, u32> = BTreeMap::new();
            for &c in codes {
                if !n.contains_key(&c) {
                    order.push(c);
                }
                *n.entry(c).or_default() += 1;
            }
            order
                .iter()
                .map(|c| json!({"code": c, "name": self.names().get(c).cloned().unwrap_or_else(|| c.to_string()), "n": n[c]}))
                .collect()
        };
        let l = legal(&d);
        Ok(json!({
            "id": d.id, "name": d.meta.name,
            "main": group(&d.deck.main), "extra": group(&d.deck.extra), "side": group(&d.deck.side),
            "counts": {"main": d.deck.main.len(), "extra": d.deck.extra.len(), "side": d.deck.side.len()},
            "legal": {"ok": l.ok, "problems": l.problems.iter().map(|p| p.text.clone()).collect::<Vec<_>>()},
        }))
    }

    fn deck_odds(&self, args: &Value) -> Result<Value, String> {
        match args.get("deck").and_then(Value::as_str).filter(|s| !s.trim().is_empty()) {
            Some(key) => {
                let d = self.find_deck(key)?;
                super::odds::deck_odds(args, Some(&d.deck.main))
            }
            None => super::odds::deck_odds(args, None),
        }
    }

    fn my_stats(&self, args: &Value) -> Result<Value, String> {
        let out = super::stats::my_stats(args, &self.app.records.root.join("index.sqlite"))?;
        // The record ids are for the app's block, not for the model.
        Ok(out["summary"].clone())
    }
}

/// The cards an answer names as `[[Name]]` (system.v2): `{"cards": {name: code}, "unresolved": [name]}`, so the app
/// can open their details and ask the user to pick a card for the rest (option 1, user 2026-11-14). Only an
/// exact card name resolves (ignoring case, spaces and punctuation): a near match is a guess, and "[[Branded]]" (the
/// deck) once linked the card Branded in Red. Unresolved names stay plain text. (User 2026-11-14: the answer named
/// Ghost Belle, Veiler and Fuwalos but only Ash, found by a tool, could be opened.)
pub fn card_refs(app: &App, text: &str) -> Value {
    let mut out = serde_json::Map::new();
    let mut unresolved: Vec<String> = Vec::new();
    let mut rest = text;
    let mut seen = 0;
    while let Some(i) = rest.find("[[") {
        let Some(j) = rest[i + 2..].find("]]") else { break };
        let name = rest[i + 2..i + 2 + j].trim().to_string();
        rest = &rest[i + 2 + j + 2..];
        seen += 1;
        if seen > 40 || name.len() < 2 || name.len() > 80 || out.contains_key(&name) || unresolved.contains(&name) {
            continue;
        }
        let want = name_key(&name);
        match app.ctx.texts.names().iter().find(|(_, n)| name_key(n) == want) {
            Some((code, _)) => {
                out.insert(name, json!(code));
            }
            // Models bracket deck names despite the prompt ("[[Branded]]", 2026-11-14): one of the user's decks is
            // not a card to ask about, so it stays plain text.
            None if app.user_decks.list().iter().any(|d| name_key(&d.meta.name) == want) => {}
            None => unresolved.push(name),
        }
    }
    json!({"cards": out, "unresolved": unresolved})
}

/// A card name compared without case, spaces or punctuation ("Maxx \"C\"" = "maxx c").
fn name_key(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

impl Tools for AppTools {
    fn specs(&self) -> Vec<Value> {
        self.specs.clone()
    }

    // `replay` polls `cancel` per replayed choice and `bot_view` per lookahead candidate; the rest answer in
    // milliseconds (the turn checks it between calls).
    fn call(&mut self, name: &str, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
        match name {
            "search_cards" => self.search_cards(args),
            "card" => self.card(args),
            "deck" => self.deck(args),
            "deck_odds" => self.deck_odds(args),
            "my_stats" => self.my_stats(args),
            "replay" => super::replay::replay(&self.app, args, cancel),
            "bot_view" => super::replay::bot_view(&self.app, args, cancel),
            _ => Err(format!(
                "unknown tool {name}; the tools are search_cards, card, deck, deck_odds, my_stats, replay, bot_view"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_cover_the_v1_tools_in_function_format() {
        let names: Vec<String> = specs().iter().map(|s| s["function"]["name"].as_str().unwrap().to_string()).collect();
        assert_eq!(names, vec!["search_cards", "card", "deck", "deck_odds", "my_stats", "replay", "bot_view"]);
        assert!(specs().iter().all(|s| s["type"] == "function" && s["function"]["parameters"]["type"] == "object"));
    }

    #[test]
    fn card_names_compare_without_case_or_punctuation() {
        assert_eq!(name_key("Maxx \"C\""), name_key("maxx c"));
        assert_eq!(name_key("Ash Blossom & Joyous Spring"), "ashblossomjoyousspring");
        assert_ne!(name_key("Branded"), name_key("Branded in Red"));
    }

    #[test]
    fn the_unofficial_note_is_dropped() {
        let ocg = "Negate it. * The above text is unofficial and describes the card's functionality in the OCG.\n";
        assert_eq!(plain(ocg), "Negate it.");
        let speed = "Negate it.\n\r\n* The above text is unofficial and describes the card's presumed intended \
                     functionality in the TCG Speed Duel format.";
        assert_eq!(plain(speed), "Negate it.");
        assert_eq!(plain("Negate it."), "Negate it.");
    }
}
