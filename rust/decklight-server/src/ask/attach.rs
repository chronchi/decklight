//! What the user attaches to a question (docs/plans/ask-tab.md §Attach; mocks ask-attach*): a deck, a game, a card or a
//! photo of a card. The question row stores the references (`attach`); the context gets them as a tool round the model
//! did not have to ask for (the same `deck` / `replay` / `card` tools it calls itself), so Retry and later turns read
//! the data as it is then. A photo is saved under the chat store's `images/` and goes to the model as an image part.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};

use super::chat::Tools;

/// At most this many attachments per question, and one photo.
pub const MAX: usize = 5;
/// A photo's size limit after the client's downscale (≤1024 px JPEG is ~100-300 KB).
pub const MAX_PHOTO_BYTES: usize = 2 * 1024 * 1024;
/// A thumbnail data URL kept in the question's meta for the bubble.
const MAX_THUMB_CHARS: usize = 40_000;
/// One attached result in the context (a whole game turn by turn can be long), and all of a question's together: the
/// reads come after the context budget is applied, so they are capped here (review 2026-11-15).
const MAX_RESULT_CHARS: usize = 12_000;
const MAX_ROUND_CHARS: usize = 30_000;

pub fn images_dir(store_dir: &Path) -> PathBuf {
    store_dir.join("images")
}

fn label(a: &Value) -> String {
    a["label"].as_str().unwrap_or("").chars().take(120).collect()
}

/// A saved photo's name as `accept` makes it (a uuid7 and .jpg): anything else is refused, so a name never leaves `images/`.
pub fn is_photo_name(f: &str) -> bool {
    f.strip_suffix(".jpg").is_some_and(|id| !id.is_empty() && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'))
}

/// The client's attachments, checked; a new photo is saved (only once every attachment passed) and replaced by its
/// file name, and an edited question's photo (`file`) is reused. Returns the references for the question row and its
/// meta (thumbnails for the bubble).
pub fn accept(raw: &Value, images: &Path) -> Result<(Vec<Value>, Value), String> {
    let items = match raw {
        Value::Null => return Ok((vec![], json!({}))),
        Value::Array(a) => a,
        _ => return Err("attachments must be a list: remove them and attach again".into()),
    };
    if items.len() > MAX {
        return Err(format!("{} attachments: a question takes up to {MAX}, remove some", items.len()));
    }
    let mut out = Vec::new();
    let mut thumbs = serde_json::Map::new();
    let mut photos = 0;
    let mut new_photo: Option<(String, Vec<u8>)> = None;
    for a in items {
        let kind = a["kind"].as_str().unwrap_or("");
        let r = match kind {
            "deck" | "game" => {
                let id = a["id"].as_str().map(str::trim).filter(|s| !s.is_empty() && s.len() <= 200);
                let Some(id) = id else { return Err(format!("the attached {kind} has no id: remove it and attach it again")) };
                json!({"kind": kind, "id": id, "label": label(a)})
            }
            "card" => {
                let Some(code) = a["code"].as_u64().filter(|c| *c > 0 && *c < u32::MAX as u64) else {
                    return Err("the attached card has no code: remove it and attach it again".into());
                };
                json!({"kind": "card", "code": code, "label": label(a)})
            }
            "photo" => {
                photos += 1;
                if photos > 1 {
                    return Err("one photo per question: remove the others".into());
                }
                let thumb = a["thumb"].as_str().filter(|t| t.starts_with("data:image/") && t.len() <= MAX_THUMB_CHARS);
                // "Edit and ask again" sends the saved photo back by name.
                if let Some(file) = a["file"].as_str() {
                    if !is_photo_name(file) || !images.join(file).is_file() {
                        return Err("the photo of the edited question is gone: attach it again".into());
                    }
                    if let Some(t) = thumb {
                        thumbs.insert(file.to_string(), json!(t));
                    }
                    out.push(json!({"kind": "photo", "file": file, "label": "Photo"}));
                    continue;
                }
                let data = a["data"].as_str().unwrap_or("");
                let data = data.split_once("base64,").map_or(data, |(_, d)| d);
                let bytes = B64.decode(data).map_err(|_| "the photo could not be read: take or pick it again".to_string())?;
                if bytes.len() > MAX_PHOTO_BYTES {
                    return Err(format!("the photo is {} KB, over the {} KB limit: pick a smaller one", bytes.len() / 1024, MAX_PHOTO_BYTES / 1024));
                }
                if !bytes.starts_with(&[0xFF, 0xD8]) {
                    return Err("the photo is not a JPEG (the app converts it): take or pick it again".into());
                }
                let file = format!("{}.jpg", decklight_record::uuid7());
                if let Some(t) = thumb {
                    thumbs.insert(file.clone(), json!(t));
                }
                new_photo = Some((file.clone(), bytes));
                json!({"kind": "photo", "file": file, "label": "Photo"})
            }
            other => return Err(format!("unknown attachment {other:?}: attach a deck, a game, a card or a photo")),
        };
        out.push(r);
    }
    if let Some((file, bytes)) = new_photo {
        let saved = std::fs::create_dir_all(images).and_then(|_| std::fs::write(images.join(&file), &bytes));
        saved.map_err(|e| format!("could not save the photo ({e}): check the phone's free space"))?;
    }
    let meta = if thumbs.is_empty() { json!({}) } else { json!({"thumbs": thumbs}) };
    Ok((out, meta))
}

/// The photos a question row references (to delete them with their chat, or when the question was never stored).
pub fn photos_of(attach: &[Value]) -> Vec<String> {
    attach.iter().filter_map(|a| a["file"].as_str()).filter(|f| is_photo_name(f)).map(str::to_string).collect()
}

/// The tool call that reads an attachment (`None` for a photo).
fn call_of(a: &Value) -> Option<(&'static str, Value)> {
    match a["kind"].as_str()? {
        "deck" => Some(("deck", json!({"id": a["id"]}))),
        "game" => Some(("replay", json!({"record": a["id"]}))),
        "card" => Some(("card", json!({"code": a["code"]}))),
        _ => None,
    }
}

fn capped(v: Value, max: usize) -> String {
    let s = v.to_string();
    if s.len() <= max {
        return s;
    }
    let mut cut = max.saturating_sub(200);
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    json!({"truncated": true, "note": "the attachment was cut to fit; call the tool with `turn` or `decision` for the rest",
           "start": &s[..cut]}).to_string()
}

/// Every user message's `attach` becomes context. Older questions only name theirs (the answer after them already used
/// the data, and the model can call the tool again), so a chat does not re-read and resend every attachment each turn.
/// The newest question gets the deck, game and card read through the tools in a round right after it, and its photo as
/// an image part; with a tool round the photo follows it as a user message of its own, so the conversation still ends
/// on the user's turn with the image (Gemma takes the last message's image; review 2026-11-15).
pub fn expand(msgs: &mut Vec<Value>, tools: &mut dyn Tools, images: &Path, cancel: &AtomicBool) -> Result<(), String> {
    let last_user = msgs.iter().rposition(|m| m["role"] == "user");
    let mut out = Vec::with_capacity(msgs.len());
    for (i, mut m) in std::mem::take(msgs).into_iter().enumerate() {
        let attach = match m.as_object_mut().and_then(|o| o.remove("attach")) {
            Some(Value::Array(a)) if m["role"] == "user" && !a.is_empty() => a,
            _ => {
                out.push(m);
                continue;
            }
        };
        let names: Vec<String> = attach
            .iter()
            .map(|a| match a["kind"].as_str().unwrap_or("") {
                "photo" => "a photo of a card (identify the card from it, then look it up)".to_string(),
                k => format!("{k} {}", if a["label"].as_str().unwrap_or("").is_empty() { a["id"].to_string() } else { label(a) }),
            })
            .collect();
        let text = format!("{}\n\n(Attached: {}.)", m["content"].as_str().unwrap_or(""), names.join("; "));
        m["content"] = json!(text);
        if Some(i) != last_user {
            out.push(m);
            continue;
        }
        let image = match photos_of(&attach).first() {
            Some(file) => {
                let bytes = std::fs::read(images.join(file))
                    .map_err(|e| format!("the attached photo is missing ({e}): ask again with a new photo"))?;
                Some(json!({"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{}", B64.encode(bytes))}}))
            }
            None => None,
        };
        let calls: Vec<(String, &str, Value)> =
            attach.iter().filter_map(call_of).enumerate().map(|(j, (n, a))| (format!("attach-{i}-{j}"), n, a)).collect();
        if calls.is_empty() {
            if let Some(img) = image {
                m["content"] = json!([{"type": "text", "text": text}, img]);
            }
            out.push(m);
            continue;
        }
        out.push(m);
        out.push(json!({"role": "assistant", "content": Value::Null, "tool_calls": calls.iter().map(|(id, n, a)| json!({
            "id": id, "type": "function", "function": {"name": n, "arguments": a.to_string()},
        })).collect::<Vec<_>>()}));
        let mut room = MAX_ROUND_CHARS;
        for (id, n, a) in &calls {
            let r = tools.call(n, a, cancel).unwrap_or_else(|e| json!({"error": e}));
            let c = capped(r, MAX_RESULT_CHARS.min(room));
            room = room.saturating_sub(c.len());
            out.push(json!({"role": "tool", "tool_call_id": id, "content": c}));
        }
        if let Some(img) = image {
            out.push(json!({"role": "user", "content": [{"type": "text", "text": "The attached photo:"}, img]}));
        }
    }
    *msgs = out;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(Vec<(String, Value)>);
    impl Tools for Fake {
        fn specs(&self) -> Vec<Value> {
            vec![]
        }
        fn call(&mut self, name: &str, args: &Value, _: &AtomicBool) -> Result<Value, String> {
            self.0.push((name.into(), args.clone()));
            if name == "card" { Err("no card 7".into()) } else { Ok(json!({"name": format!("{name} result")})) }
        }
    }

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ask-attach-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn attachments_are_checked_and_a_photo_is_saved() {
        let d = dir("accept");
        let jpeg = format!("data:image/jpeg;base64,{}", B64.encode([0xFF, 0xD8, 0xFF, 0xE0, 1, 2]));
        let raw = json!([{"kind": "deck", "id": "d1", "label": "Snake-Eye"}, {"kind": "card", "code": 14558127},
                         {"kind": "photo", "data": jpeg, "thumb": "data:image/jpeg;base64,AA"}]);
        let (refs, meta) = accept(&raw, &d).unwrap();
        assert_eq!(refs[0], json!({"kind": "deck", "id": "d1", "label": "Snake-Eye"}));
        let file = refs[2]["file"].as_str().unwrap();
        assert_eq!(std::fs::read(d.join(file)).unwrap(), vec![0xFF, 0xD8, 0xFF, 0xE0, 1, 2]);
        assert_eq!(meta["thumbs"][file], "data:image/jpeg;base64,AA");
        let bad = |v: Value| accept(&v, &d).unwrap_err();
        assert!(bad(json!([{"kind": "deck"}])).contains("no id"));
        assert!(bad(json!([{"kind": "zip"}])).contains("unknown attachment"));
        assert!(bad(json!([{"kind": "photo", "data": B64.encode(b"GIF89a")}])).contains("not a JPEG"));
        assert!(bad(json!([{"kind": "photo", "data": jpeg}, {"kind": "photo", "data": jpeg}])).contains("one photo"));
        assert!(bad(json!(vec![json!({"kind": "card", "code": 1}); MAX + 1])).contains("up to"));
        // A later bad attachment leaves no photo file behind.
        let before = std::fs::read_dir(&d).unwrap().count();
        assert!(bad(json!([{"kind": "photo", "data": jpeg}, {"kind": "zip"}])).contains("unknown"));
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), before);
        // An edited question's photo is reused by name; a made-up or escaping name is refused.
        let (again, _) = accept(&json!([{"kind": "photo", "file": file}]), &d).unwrap();
        assert_eq!(again[0]["file"], file);
        assert!(bad(json!([{"kind": "photo", "file": "../x.jpg"}])).contains("gone"));
        assert!(bad(json!([{"kind": "photo", "file": "0190-ab.jpg"}])).contains("gone"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn attachments_become_a_tool_round_and_only_the_newest_photo_is_sent() {
        let d = dir("expand");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("0a.jpg"), [0xFF, 0xD8]).unwrap();
        let mut msgs = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "old", "attach": [{"kind": "photo", "file": "0a.jpg"}]}),
            json!({"role": "assistant", "content": "It is Ash."}),
            json!({"role": "user", "content": "how about these?", "attach": [
                {"kind": "game", "id": "g1", "label": "vs Branded"}, {"kind": "card", "code": 7, "label": "X"},
                {"kind": "photo", "file": "0a.jpg"}]}),
        ];
        msgs[1]["attach"].as_array_mut().unwrap().push(json!({"kind": "deck", "id": "d0", "label": "Old deck"}));
        let mut tools = Fake(vec![]);
        expand(&mut msgs, &mut tools, &d, &AtomicBool::new(false)).unwrap();
        // An older question only names its attachments: no image, no tool round.
        assert!(msgs[1]["content"].as_str().unwrap().starts_with("old\n\n(Attached: a photo of a card"));
        assert!(msgs.iter().all(|m| m.get("attach").is_none()));
        assert!(msgs[3]["content"].as_str().unwrap().contains("game vs Branded; card X"));
        assert_eq!(tools.0, vec![("replay".to_string(), json!({"record": "g1"})), ("card".to_string(), json!({"code": 7}))]);
        let calls = msgs[4]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(msgs[5]["tool_call_id"], calls[0]["id"]);
        assert!(msgs[6]["content"].as_str().unwrap().contains("no card 7"));
        // The photo comes last, after the tool round, as the user's.
        assert_eq!(msgs[7]["role"], "user");
        assert_eq!(msgs[7]["content"][1]["image_url"]["url"], format!("data:image/jpeg;base64,{}", B64.encode([0xFF, 0xD8])));
        assert_eq!(msgs.len(), 8);
        // A photo alone stays in the question.
        let mut one = vec![json!({"role": "user", "content": "this?", "attach": [{"kind": "photo", "file": "0a.jpg"}]})];
        expand(&mut one, &mut tools, &d, &AtomicBool::new(false)).unwrap();
        assert_eq!((one.len(), one[0]["content"][1]["type"].as_str()), (1, Some("image_url")));
        // A big read is cut to fit.
        assert!(capped(json!({"x": "y".repeat(50_000)}), MAX_RESULT_CHARS).len() <= MAX_RESULT_CHARS);
        let _ = std::fs::remove_dir_all(&d);
    }
}
