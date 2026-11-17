//! Games > Export (mocks export-*): a saved game, or a part of it, as files to share. The server makes the EDOPro
//! replay (.yrpX) and the text; the video, GIF and image are drawn by the client from the play screen (`games:watch`
//! frames, replay.rs `watch`).

use std::sync::atomic::{AtomicBool, Ordering};

use base64::Engine as _;
use serde_json::{Value, json};

use super::replay::{self, Game};
use crate::server::App;

/// The most games one export takes (Select all can tick hundreds; one share holds a few dozen files at most).
const MAX_GAMES: usize = 50;

fn turn_arg(args: &Value, key: &str) -> Result<Option<i32>, String> {
    match &args[key] {
        Value::Null => Ok(None),
        v => v.as_u64().filter(|&t| t >= 1).map(|t| Some(t as i32)).ok_or_else(|| format!("`{key}` must be a turn number from 1")),
    }
}

/// A file name without characters a share target or a file system refuses.
fn file_name(base: &str, ext: &str) -> String {
    let clean: String = base.chars().filter(|c| c.is_alphanumeric() || " .,_()-".contains(*c)).collect();
    let clean = clean.split_whitespace().collect::<Vec<_>>().join(" ");
    let clean = clean.trim_matches(|c| c == '.' || c == ' ');
    format!("{}.{ext}", if clean.is_empty() { "game" } else { clean })
}

pub(super) fn title_of(g: &Game) -> String {
    title(g)
}

pub(super) fn day_of(g: &Game) -> String {
    day(g)
}

fn title(g: &Game) -> String {
    let h = g.header();
    format!("{} vs {}", h["your_deck"].as_str().unwrap_or("Your deck"), h["opponent_deck"].as_str().unwrap_or("opponent"))
}

fn day(g: &Game) -> String {
    g.rec["created"].as_str().unwrap_or("").chars().take(10).collect()
}

fn part(from: Option<i32>, to: Option<i32>, last: i32) -> String {
    match (from.unwrap_or(1), to.unwrap_or(last).min(last)) {
        (1, t) if t >= last => String::new(),
        (a, b) if a == b => format!(" - turn {a}"),
        (a, b) => format!(" - turns {a}-{b}"),
    }
}

/// The two duelists' names in core order: the decks' names, marked when both play the same deck.
fn names(g: &Game) -> [String; 2] {
    let decks = g.rec["decks"].as_array().cloned().unwrap_or_default();
    let name = |s: usize| decks.get(s).and_then(|d| d["name"].as_str()).unwrap_or("Duelist").to_string();
    let (a, b) = (name(0), name(1));
    if a != b {
        return [a, b];
    }
    let tag = |s: usize| if s as i32 == g.me { "(you)" } else { "(opp)" };
    [format!("{} {}", a.chars().take(13).collect::<String>(), tag(0)), format!("{} {}", b.chars().take(13).collect::<String>(), tag(1))]
}

/// The responses an EDOPro replay keeps: all of them, the ones up to the end of `to_turn`, or the ones up to the
/// `upto_choice`-th choice of the app (a play of the sequence).
fn cut(app: &App, g: &Game, to_turn: Option<i32>, upto_choice: Option<u64>) -> Result<Option<usize>, String> {
    let responses = g.rec["responses"].as_array().cloned().unwrap_or_default();
    if let Some(c) = upto_choice {
        let mut seen = 0u64;
        for (i, r) in responses.iter().enumerate() {
            seen += r["opt"].as_array().map_or(1, |a| a.len().max(1) as u64);
            if seen >= c {
                return Ok(Some(i + 1));
            }
        }
        return Ok(None);
    }
    let Some(t) = to_turn else { return Ok(None) };
    let raw: Vec<Vec<u8>> = responses.iter().map(|r| hex(r["r"].as_str().unwrap_or(""))).collect();
    let out = decklight_edopro::record::replay_responses(app.engine, &g.rec, &raw).map_err(|e| format!("this game cannot be replayed: {e}"))?;
    Ok(out.turns.iter().position(|&x| x > t))
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).filter_map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()).collect()
}

/// One game's plays as text, turns `from..=to`, from the user's side of the table (the opponent's hand stays hidden).
fn text_of(g: &Game, from: Option<i32>, to: Option<i32>) -> String {
    let h = g.header();
    let mut out = format!(
        "{} · {} · {} turns · you went {} · {}\n",
        title(g),
        h["result"].as_str().unwrap_or(""),
        h["turns"],
        h["you_went"].as_str().unwrap_or("?"),
        day(g)
    );
    for t in g.turns.iter().filter(|t| from.is_none_or(|f| t.turn >= f) && to.is_none_or(|x| t.turn <= x)) {
        out += &format!("\nTurn {} · {} · LP you {} · opponent {}\n", t.turn, if t.yours { "your turn" } else { "opponent's turn" }, t.lp.0, t.lp.1);
        for e in &t.events {
            out += &format!("  {e}\n");
        }
    }
    if to.is_none_or(|x| x >= g.turns.last().map_or(0, |t| t.turn)) {
        out += &format!("\nEnd: {} · LP you {} · opponent {}\n", h["result"].as_str().unwrap_or(""), g.final_lp.0, g.final_lp.1);
    }
    out
}

/// `args`: `ids` (games), `as` ("yrp" or "txt"), optional `from_turn`/`to_turn`, optional `upto_choice` (EDOPro
/// replay up to a play). Result: `files` [{name, mime, data (base64)}], and `text` for a text export.
pub fn export(app: &App, args: &Value, cancel: &AtomicBool) -> Result<Value, String> {
    let ids: Vec<String> = args["ids"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect();
    if ids.is_empty() {
        return Err("no game to export: open a game or tick some in Select".into());
    }
    if ids.len() > MAX_GAMES {
        return Err(format!("{} games ticked: export at most {MAX_GAMES} at a time", ids.len()));
    }
    let (from, to) = (turn_arg(args, "from_turn")?, turn_arg(args, "to_turn")?);
    if let (Some(a), Some(b)) = (from, to) {
        if a > b {
            return Err(format!("from turn {a} is after to turn {b}: pick the turns again"));
        }
    }
    let upto_choice = args["upto_choice"].as_u64();
    let format = args["as"].as_str().unwrap_or("");
    let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
    let mut files = Vec::new();
    let mut texts = Vec::new();
    for id in &ids {
        if cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let g = replay::open(app, id, cancel).map_err(|e| if ids.len() > 1 { format!("game {id}: {e}") } else { e })?;
        let last = g.turns.last().map_or(1, |t| t.turn);
        let name = format!("{} {}{}", title(&g), day(&g), part(from, to, last));
        match format {
            "yrp" => {
                let upto = cut(app, &g, to, upto_choice)?;
                let ts = chrono::DateTime::parse_from_rfc3339(g.rec["created"].as_str().unwrap_or(""))
                    .map(|d| d.timestamp().max(0) as u64)
                    .unwrap_or(0);
                let data = decklight_edopro::yrp::record_to_yrpx(app.engine, &app.root, &g.rec, names(&g), upto, ts)
                    .map_err(|e| format!("could not make the EDOPro replay of {}: {e}", title(&g)))?;
                files.push(json!({"name": file_name(&name, "yrpX"), "mime": "application/octet-stream", "data": b64(&data)}));
            }
            "txt" => texts.push((name, text_of(&g, from, to))),
            other => return Err(format!("cannot export as {other:?}: pick EDOPro replay or Text (video, GIF and image are made on the phone)")),
        }
    }
    // Two games of the same decks on the same day share a name: number them, or one file replaces the other.
    let mut seen = std::collections::HashSet::new();
    for f in files.iter_mut() {
        let name = f["name"].as_str().unwrap_or("").to_string();
        let (stem, ext) = name.rsplit_once('.').unwrap_or((&name, ""));
        let mut k = 1;
        let mut unique = name.clone();
        while !seen.insert(unique.clone()) {
            k += 1;
            unique = format!("{stem} ({k}).{ext}");
        }
        f["name"] = json!(unique);
    }
    if texts.is_empty() {
        return Ok(json!({"files": files}));
    }
    let name = if texts.len() == 1 { texts[0].0.clone() } else { format!("{} games", texts.len()) };
    let text = texts.into_iter().map(|(_, t)| t).collect::<Vec<_>>().join("\n\n");
    files.push(json!({"name": file_name(&name, "txt"), "mime": "text/plain", "data": b64(text.as_bytes())}));
    Ok(json!({"files": files, "text": text}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ask::replay::tests::{app, play_and_save};

    fn bytes(f: &Value) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD.decode(f["data"].as_str().unwrap()).unwrap()
    }

    #[test]
    fn exports_edopro_replays_whole_or_cut_and_text() {
        let dir = std::env::temp_dir().join(format!("ask-export-{}", crate::record::random_hex(6)));
        let app = app(&dir);
        let no = AtomicBool::new(false);
        let id = play_and_save(&app, 7);
        let rec = decklight_record::load_gz(&app.records.root.join(format!("{id}.match.json.gz"))).unwrap();
        let all = rec["responses"].as_array().unwrap().len();

        // The whole game: the file reads back (EDOPro's way) to every response of the game.
        let whole = export(&app, &json!({"ids": [id], "as": "yrp"}), &no).unwrap();
        let f = &whole["files"][0];
        assert!(f["name"].as_str().unwrap().ends_with(".yrpX"), "{f}");
        let x = decklight_edopro::yrp::parse_yrpx(&bytes(f)).unwrap();
        assert_eq!(x.yrp.as_ref().unwrap().responses.len(), all);
        // scripts/app/export_edopro_check.py replays the file with a real EDOPro install's core and scripts.
        if let Ok(out) = std::env::var("DECKLIGHT_EXPORT_DUMP") {
            std::fs::write(out, bytes(f)).unwrap();
        }

        // Turns 1-2: a prefix that stops at the end of turn 2.
        let cut2 = export(&app, &json!({"ids": [id], "as": "yrp", "to_turn": 2}), &no).unwrap();
        let f2 = &cut2["files"][0];
        assert!(f2["name"].as_str().unwrap().contains("turns 1-2"), "{f2}");
        let n2 = decklight_edopro::yrp::parse_yrpx(&bytes(f2)).unwrap().yrp.unwrap().responses.len();
        assert!(n2 > 0 && n2 < all, "{n2} of {all}");
        // Up to a play: the first 5 choices.
        let p5 = export(&app, &json!({"ids": [id], "as": "yrp", "upto_choice": 5}), &no).unwrap();
        let n5 = decklight_edopro::yrp::parse_yrpx(&bytes(&p5["files"][0])).unwrap().yrp.unwrap().responses.len();
        assert!((1..=5).contains(&n5), "{n5}");

        // Text: the turns asked for, from the user's side.
        let t = export(&app, &json!({"ids": [id], "as": "txt", "from_turn": 2, "to_turn": 3}), &no).unwrap();
        let text = t["text"].as_str().unwrap();
        assert!(text.contains("\nTurn 2 ") && text.contains("\nTurn 3 ") && !text.contains("\nTurn 1 ") && !text.contains("\nTurn 4 "), "{text}");
        assert_eq!(String::from_utf8(bytes(&t["files"][0])).unwrap(), text);
        let two = export(&app, &json!({"ids": [id, id], "as": "yrp"}), &no).unwrap();
        let names: Vec<&str> = two["files"].as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap()).collect();
        assert!(names[1].ends_with(" (2).yrpX") && names[0] != names[1], "{names:?}");
        let both = export(&app, &json!({"ids": [id, id], "as": "txt"}), &no).unwrap();
        assert_eq!(both["files"][0]["name"], "2 games.txt");

        // Fail loud with what to do.
        assert!(export(&app, &json!({"ids": [], "as": "yrp"}), &no).unwrap_err().contains("tick"));
        assert!(export(&app, &json!({"ids": [id], "as": "mp4"}), &no).unwrap_err().contains("made on the phone"));
        assert!(export(&app, &json!({"ids": [id], "as": "txt", "from_turn": 3, "to_turn": 2}), &no).unwrap_err().contains("pick the turns again"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_names_are_safe() {
        assert_eq!(file_name("Blue-Eyes vs Branded 2026-11-15 - turns 1-2", "yrpX"), "Blue-Eyes vs Branded 2026-11-15 - turns 1-2.yrpX");
        assert_eq!(file_name("a/b: c?", "txt"), "ab c.txt");
        assert_eq!(file_name("../..", "txt"), "game.txt");
    }
}
