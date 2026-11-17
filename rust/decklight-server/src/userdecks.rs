//! Saved user decks (`.ydk` files + `index.json`) and the deck legality rules shared with `web/lib/deck.js`.
//!
//! Files live in `<root>/decks/user/`, outside every app asset unit, so app updates never touch them. Writes are
//! atomic (temporary file + rename). Deleting moves the file to `decks/user/.trash/`, never to oblivion.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Deck {
    pub main: Vec<u32>,
    pub extra: Vec<u32>,
    pub side: Vec<u32>,
}

impl Deck {
    pub fn total(&self) -> usize {
        self.main.len() + self.extra.len() + self.side.len()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedYdk {
    pub deck: Deck,
    pub name: Option<String>,
    pub unknown_lines: Vec<String>,
}

fn decode_base64(text: &str) -> Option<Vec<u8>> {
    use base64::Engine as _;
    let mut s = text.replace('-', "+").replace('_', "/");
    while s.len() % 4 != 0 {
        s.push('=');
    }
    base64::engine::general_purpose::STANDARD.decode(s).ok()
}

fn encode_base64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn decode_ydke(body: &str) -> ParsedYdk {
    let mut out = ParsedYdk::default();
    let parts: Vec<&str> = body.split('!').collect();
    let sections = [&mut out.deck.main, &mut out.deck.extra, &mut out.deck.side];
    for (i, section) in sections.into_iter().enumerate() {
        let Some(part) = parts.get(i).map(|s| s.trim()).filter(|s| !s.is_empty()) else {
            continue;
        };
        let Some(bytes) = decode_base64(part) else {
            out.unknown_lines.push(part.to_string());
            continue;
        };
        for chunk in bytes.chunks_exact(4) {
            section.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if bytes.len() % 4 != 0 {
            out.unknown_lines.push(part.to_string());
        }
    }
    for part in parts.iter().skip(3) {
        if !part.trim().is_empty() {
            out.unknown_lines.push(part.trim().to_string());
        }
    }
    out
}

/// Parse `.ydk` text, or a `ydke://` URL, keeping every line that is not a section, comment or positive code.
pub fn parse_ydk(text: &str) -> ParsedYdk {
    let trimmed = text.trim();
    if let Some(body) = trimmed.strip_prefix("ydke://") {
        return decode_ydke(body);
    }
    let mut out = ParsedYdk::default();
    let mut current = 0usize;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "#main" => {
                current = 0;
                continue;
            }
            "#extra" => {
                current = 1;
                continue;
            }
            "!side" => {
                current = 2;
                continue;
            }
            _ => {}
        }
        if line.starts_with('#') || line.starts_with('!') {
            if let Some(rest) = line.strip_prefix('#')
                && let Some((key, value)) = rest.trim_start().split_once(char::is_whitespace)
                && key.eq_ignore_ascii_case("name")
            {
                let value = value.trim();
                if !value.is_empty() {
                    out.name = Some(value.to_string());
                }
            }
            continue;
        }
        if let Ok(code) = line.parse::<u64>()
            && code > 0
            && code <= u32::MAX as u64
        {
            let code = code as u32;
            match current {
                0 => out.deck.main.push(code),
                1 => out.deck.extra.push(code),
                _ => out.deck.side.push(code),
            }
            continue;
        }
        out.unknown_lines.push(line.to_string());
    }
    out
}

pub fn to_ydk(deck: &Deck) -> String {
    let mut out = String::from("#created by decklight\n#main\n");
    for code in &deck.main {
        out.push_str(&format!("{code}\n"));
    }
    out.push_str("#extra\n");
    for code in &deck.extra {
        out.push_str(&format!("{code}\n"));
    }
    out.push_str("!side\n");
    for code in &deck.side {
        out.push_str(&format!("{code}\n"));
    }
    out
}

pub fn to_ydke(deck: &Deck) -> String {
    fn section(codes: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(codes.len() * 4);
        for code in codes {
            bytes.extend_from_slice(&code.to_le_bytes());
        }
        encode_base64(&bytes)
    }
    format!("ydke://{}!{}!{}!", section(&deck.main), section(&deck.extra), section(&deck.side))
}

// ------------------------------------------------------------------ legality

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Banlist {
    #[serde(default)]
    pub forbidden: Vec<u32>,
    #[serde(default)]
    pub limited: Vec<u32>,
    #[serde(default)]
    pub semi: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Format {
    #[serde(default)]
    pub name: Option<String>,
    pub main: [usize; 2],
    pub extra: [usize; 2],
    pub side: [usize; 2],
    pub copies: usize,
    #[serde(default)]
    pub banlist: Option<Banlist>,
}

impl Default for Format {
    fn default() -> Self {
        Format {
            name: Some("Standard".to_string()),
            main: [40, 60],
            extra: [0, 15],
            side: [0, 15],
            copies: 3,
            banlist: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    pub kind: &'static str,
    pub text: String,
    pub codes: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Legality {
    pub ok: bool,
    pub problems: Vec<Problem>,
}

fn count_problem(label: &str, kind: &'static str, n: usize, range: [usize; 2]) -> Option<Problem> {
    let [minimum, maximum] = range;
    if n < minimum {
        Some(Problem {
            kind,
            text: format!("{label} {n}/{minimum} · {} more", minimum - n),
            codes: Vec::new(),
        })
    } else if n > maximum {
        Some(Problem {
            kind,
            text: format!("{label} {n}/{maximum} · {} too many", n - maximum),
            codes: Vec::new(),
        })
    } else {
        None
    }
}

/// The same rules and texts as `validate` in `web/lib/deck.js`. Copies count across main+extra+side by alias.
pub fn validate<F, G>(deck: &Deck, format: &Format, name_of: F, alias_of: G) -> Legality
where
    F: Fn(u32) -> String,
    G: Fn(u32) -> u32,
{
    let mut problems = Vec::new();
    for (label, kind, n, range) in [
        ("Main", "main_count", deck.main.len(), format.main),
        ("Extra", "extra_count", deck.extra.len(), format.extra),
        ("Side", "side_count", deck.side.len(), format.side),
    ] {
        if let Some(p) = count_problem(label, kind, n, range) {
            problems.push(p);
        }
    }

    // alias -> (copies, lowest code in the group, which is the name shown)
    let mut groups: BTreeMap<u32, (usize, u32)> = BTreeMap::new();
    for code in deck.main.iter().chain(&deck.extra).chain(&deck.side) {
        let alias = alias_of(*code);
        let group = groups.entry(alias).or_insert((0, *code));
        group.0 += 1;
        group.1 = group.1.min(*code);
    }
    let mut ordered: Vec<(usize, u32)> = groups.values().copied().collect();
    ordered.sort_by_key(|(_, code)| *code);
    for (count, code) in ordered {
        if count > format.copies {
            problems.push(Problem {
                kind: "copies",
                text: format!("{} copies max: {} ×{count}", format.copies, name_of(code)),
                codes: vec![code],
            });
        }
    }

    if let Some(banlist) = &format.banlist {
        for &code in &banlist.forbidden {
            if groups.get(&alias_of(code)).is_some_and(|(count, _)| *count > 0) {
                problems.push(Problem {
                    kind: "banlist",
                    text: format!("Forbidden: {}", name_of(code)),
                    codes: vec![code],
                });
            }
        }
        for &code in &banlist.limited {
            let used = groups.get(&alias_of(code)).map_or(0, |(count, _)| *count);
            if used > 1 {
                problems.push(Problem {
                    kind: "banlist",
                    text: format!("Limited (1): {} ×{used}", name_of(code)),
                    codes: vec![code],
                });
            }
        }
        for &code in &banlist.semi {
            let used = groups.get(&alias_of(code)).map_or(0, |(count, _)| *count);
            if used > 2 {
                problems.push(Problem {
                    kind: "banlist",
                    text: format!("Semi-limited (2): {} ×{used}", name_of(code)),
                    codes: vec![code],
                });
            }
        }
    }

    Legality { ok: problems.is_empty(), problems }
}

// ------------------------------------------------------------------ storage

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct EntryMeta {
    pub name: String,
    pub updated: String,
    #[serde(default)]
    pub cover: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct StoredDeck {
    pub id: String,
    pub meta: EntryMeta,
    pub deck: Deck,
}

pub struct UserDecks {
    pub dir: PathBuf,
}

fn now_iso() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

fn system_time_iso(t: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = t.into();
    dt.format("%Y-%m-%dT%H:%M:%S").to_string()
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn slug(name: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out.truncate(24);
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() { "deck".to_string() } else { out }
}

fn clean_name(name: &str) -> String {
    let name: String = name.trim().chars().take(60).collect();
    let name = name.trim();
    if name.is_empty() { "Untitled deck".to_string() } else { name.to_string() }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

impl UserDecks {
    pub fn new(root: &Path) -> UserDecks {
        UserDecks::at(&root.join("decks/user"))
    }

    pub fn at(dir: &Path) -> UserDecks {
        UserDecks { dir: dir.to_path_buf() }
    }

    /// Copies each `(name, .ydk path)` in as an editable deck, once per install: the `.starters` marker records that it
    /// ran, so starters the user deletes never come back. Missing or empty files are skipped (logged).
    pub fn seed_starters(&self, starters: &[(&str, PathBuf)]) {
        let marker = self.dir.join(".starters");
        if marker.exists() {
            return;
        }
        for (name, path) in starters {
            match std::fs::read_to_string(path) {
                Ok(text) => {
                    let deck = parse_ydk(&text).deck;
                    if deck.total() == 0 {
                        eprintln!("decklight: starter deck {} is empty, skipped", path.display());
                    } else if let Err(e) = self.save(None, name, &deck, None) {
                        eprintln!("decklight: starter deck {name}: {e}");
                    }
                }
                Err(e) => eprintln!("decklight: starter deck {}: {e}", path.display()),
            }
        }
        if let Err(e) = std::fs::create_dir_all(&self.dir).and_then(|_| std::fs::write(&marker, b"1\n")) {
            eprintln!("decklight: {}: {e}", marker.display());
        }
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join("index.json")
    }

    fn load_index(&self) -> BTreeMap<String, EntryMeta> {
        std::fs::read_to_string(self.index_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn save_index(&self, index: &BTreeMap<String, EntryMeta>) -> Result<(), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        let text = serde_json::to_vec_pretty(index).map_err(|e| e.to_string())?;
        write_atomic(&self.index_path(), &text)
    }

    fn read(&self, id: &str, meta: EntryMeta) -> Result<StoredDeck, String> {
        if !valid_id(id) {
            return Err(format!("bad deck id {id:?}"));
        }
        let path = self.dir.join(format!("{id}.ydk"));
        let text = std::fs::read_to_string(&path).map_err(|e| format!("unknown deck {id:?}: {e}"))?;
        Ok(StoredDeck { id: id.to_string(), meta, deck: parse_ydk(&text).deck })
    }

    pub fn get(&self, id: &str) -> Result<StoredDeck, String> {
        let index = self.load_index();
        let meta = index.get(id).cloned().ok_or_else(|| format!("unknown deck {id:?}"))?;
        self.read(id, meta)
    }

    /// Newest first: index entries by their `updated`, then `.ydk` files with no index entry.
    pub fn list(&self) -> Vec<StoredDeck> {
        let index = self.load_index();
        let mut out = Vec::new();
        for (id, meta) in &index {
            if let Ok(d) = self.read(id, meta.clone()) {
                out.push(d);
            }
        }
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            let mut extra: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("ydk"))
                .collect();
            extra.sort();
            for path in extra {
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
                if index.contains_key(stem) {
                    continue;
                }
                let updated = std::fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .map(system_time_iso)
                    .unwrap_or_default();
                let meta = EntryMeta { name: crate::decks::pretty_deck_name(stem), updated, cover: None };
                if let Ok(d) = self.read(stem, meta) {
                    out.push(d);
                }
            }
        }
        out.sort_by(|a, b| {
            b.meta.updated.cmp(&a.meta.updated).then_with(|| a.meta.name.cmp(&b.meta.name))
        });
        out
    }

    fn new_id(&self, name: &str, index: &BTreeMap<String, EntryMeta>) -> String {
        let base = slug(name);
        loop {
            let id = format!("{base}-{}", crate::record::random_hex(3));
            if !index.contains_key(&id) && !self.dir.join(format!("{id}.ydk")).exists() {
                return id;
            }
        }
    }

    pub fn save(&self, id: Option<&str>, name: &str, deck: &Deck, cover: Option<u32>) -> Result<StoredDeck, String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        let mut index = self.load_index();
        let name = clean_name(name);
        let id = match id {
            Some(id) => {
                if !valid_id(id) {
                    return Err(format!("bad deck id {id:?}"));
                }
                id.to_string()
            }
            None => self.new_id(&name, &index),
        };
        let path = self.dir.join(format!("{id}.ydk"));
        write_atomic(&path, to_ydk(deck).as_bytes())?;
        // A cover must be a card in the Main or Extra Deck (a card added, picked as cover and removed again left its
        // art on the deck, user 2026-11-15).
        let cover = cover.filter(|c| deck.main.contains(c) || deck.extra.contains(c));
        let meta = EntryMeta { name, updated: now_iso(), cover };
        index.insert(id.clone(), meta.clone());
        self.save_index(&index)?;
        Ok(StoredDeck { id, meta, deck: deck.clone() })
    }

    /// The art shown for a deck: its cover while that card is still in the Main or Extra Deck, else the first Main
    /// (then Extra) card.
    pub fn cover_of(stored: &StoredDeck) -> Option<u32> {
        let d = &stored.deck;
        stored.meta.cover.filter(|c| d.main.contains(c) || d.extra.contains(c)).or_else(|| d.main.first().or(d.extra.first()).copied())
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        if !valid_id(id) {
            return Err(format!("bad deck id {id:?}"));
        }
        let path = self.dir.join(format!("{id}.ydk"));
        if !path.is_file() {
            return Err(format!("unknown deck {id:?}"));
        }
        let trash = self.dir.join(".trash");
        std::fs::create_dir_all(&trash).map_err(|e| format!("{}: {e}", trash.display()))?;
        let dest = trash.join(format!("{id}.{}.ydk", chrono::Local::now().format("%Y%m%d%H%M%S")));
        std::fs::rename(&path, &dest).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut index = self.load_index();
        index.remove(id);
        self.save_index(&index)?;
        Ok(())
    }
}

/// The lobby's summary of a saved deck: the `deck_index.json` entry shape the Decks browser already renders for the
/// prebuilt decks, so saved rows get the same row and deck-list view.
pub fn deck_summary(
    deck: &Deck,
    cards: &BTreeMap<u32, decklight_cards::CardData>,
    cover: Option<u32>,
) -> Value {
    use decklight_core::constants as C;

    let ty = |code: u32| cards.get(&code).map_or(0, |c| c.type_);

    let mut counts = [0usize; 3];
    for &code in &deck.main {
        let t = ty(code);
        if t & C::TYPE_MONSTER != 0 {
            counts[0] += 1;
        } else if t & C::TYPE_SPELL != 0 {
            counts[1] += 1;
        } else if t & C::TYPE_TRAP != 0 {
            counts[2] += 1;
        }
    }

    let mut mech = Vec::new();
    for (name, bit) in
        [("Fusion", C::TYPE_FUSION), ("Synchro", C::TYPE_SYNCHRO), ("Xyz", C::TYPE_XYZ), ("Link", C::TYPE_LINK)]
    {
        if deck.extra.iter().any(|&code| ty(code) & bit != 0) {
            mech.push(name);
        }
    }
    if deck.main.iter().any(|&code| {
        let t = ty(code);
        t & C::TYPE_MONSTER != 0 && t & C::TYPE_RITUAL != 0
    }) {
        mech.push("Ritual");
    }
    if deck.main.iter().chain(&deck.extra).any(|&code| ty(code) & C::TYPE_PENDULUM != 0) {
        mech.push("Pendulum");
    }

    // Copies per code, in first-appearance order.
    let grouped = |codes: &[u32]| -> Vec<(u32, usize)> {
        let mut out: Vec<(u32, usize)> = Vec::new();
        for &code in codes {
            match out.iter_mut().find(|(c, _)| *c == code) {
                Some((_, n)) => *n += 1,
                None => out.push((code, 1)),
            }
        }
        out
    };

    let mut sig: Vec<u32> = Vec::new();
    if let Some(c) = cover.filter(|c| deck.main.contains(c) || deck.extra.contains(c) || deck.side.contains(c)) {
        sig.push(c);
    }
    let mut main = grouped(&deck.main);
    main.sort_by(|a, b| b.1.cmp(&a.1)); // stable: ties stay in first-appearance order
    for (code, _) in main {
        if sig.len() == 3 {
            break;
        }
        if !sig.contains(&code) {
            sig.push(code);
        }
    }
    for (code, _) in grouped(&deck.extra) {
        if sig.len() == 3 {
            break;
        }
        if !sig.contains(&code) {
            sig.push(code);
        }
    }

    let pairs = |groups: &[(u32, usize)]| groups.iter().map(|&(c, n)| json!([c, n])).collect::<Vec<_>>();
    json!({
        "arch": "",
        "style": "",
        "about": "",
        "origin": "Built in Decks",
        "year": 0,
        "mech": mech,
        "counts": counts,
        "sig": sig,
        "main": pairs(&grouped(&deck.main)),
        "extra": pairs(&grouped(&deck.extra)),
    })
}

/// Codes in the deck that the card store does not know.
pub fn unknown_codes(deck: &Deck, cards: &BTreeMap<u32, decklight_cards::CardData>) -> Vec<u32> {
    deck.main
        .iter()
        .chain(&deck.extra)
        .chain(&deck.side)
        .copied()
        .filter(|c| !cards.contains_key(c))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Legality when card aliases are only known from the card store (the server path).
pub fn legality(
    deck: &Deck,
    cards: &BTreeMap<u32, decklight_cards::CardData>,
    names: &BTreeMap<u32, String>,
    format: &Format,
) -> Legality {
    validate(
        deck,
        format,
        |code| names.get(&code).cloned().unwrap_or_else(|| code.to_string()),
        |code| cards.get(&code).map(|c| if c.alias == 0 { code } else { c.alias }).unwrap_or(code),
    )
}

/// Chain `{main,extra,side}` JSON (or a deck object) into a Deck.
pub fn deck_from_value(v: &Value) -> Result<Deck, String> {
    let codes = |key: &str| -> Result<Vec<u32>, String> {
        let Some(arr) = v.get(key) else { return Ok(Vec::new()) };
        let Some(arr) = arr.as_array() else { return Err(format!("{key} must be an array of card codes")) };
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let Some(n) = item.as_u64() else {
                return Err(format!("{key} must contain positive card codes"));
            };
            if n == 0 || n > u32::MAX as u64 {
                return Err(format!("{key} must contain positive card codes"));
            }
            out.push(n as u32);
        }
        Ok(out)
    };
    Ok(Deck { main: codes("main")?, extra: codes("extra")?, side: codes("side")? })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names_and_aliases(case: &Value) -> (BTreeMap<u32, String>, BTreeMap<u32, u32>) {
        let mut names = BTreeMap::new();
        if let Some(obj) = case.get("names").and_then(Value::as_object) {
            for (k, v) in obj {
                names.insert(k.parse().unwrap(), v.as_str().unwrap().to_string());
            }
        }
        let mut aliases = BTreeMap::new();
        if let Some(obj) = case.get("aliases").and_then(Value::as_object) {
            for (k, v) in obj {
                aliases.insert(k.parse().unwrap(), v.as_u64().unwrap() as u32);
            }
        }
        (names, aliases)
    }

    #[test]
    fn userdecks_vectors_match_js() {
        let cases: Vec<Value> =
            serde_json::from_str(include_str!("../../../tests/data/deck_legality_cases.json")).unwrap();
        assert!(cases.len() >= 10);
        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let deck = deck_from_value(&case["deck"]).unwrap();
            let format: Format = serde_json::from_value(case["format"].clone()).unwrap();
            let (names, aliases) = names_and_aliases(case);
            let got = validate(
                &deck,
                &format,
                |code| names.get(&code).cloned().unwrap_or_else(|| code.to_string()),
                |code| aliases.get(&code).copied().unwrap_or(code),
            );
            let expect = &case["expect"];
            assert_eq!(got.ok, expect["ok"].as_bool().unwrap(), "{name}");
            let kinds: Vec<&str> = got.problems.iter().map(|p| p.kind).collect();
            let want_kinds: Vec<&str> = expect["kinds"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            assert_eq!(kinds, want_kinds, "{name}");
            let texts: Vec<&str> = got.problems.iter().map(|p| p.text.as_str()).collect();
            let want_texts: Vec<&str> = expect["texts"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            assert_eq!(texts, want_texts, "{name}");
        }
    }

    #[test]
    fn userdecks_ydk_parser_keeps_unknown_lines() {
        let parsed = parse_ydk("#created by x\n#main\n000123\nabc\n0\n#extra\n789\n!side\n1\n");
        assert_eq!(parsed.deck, Deck { main: vec![123], extra: vec![789], side: vec![1] });
        assert_eq!(parsed.unknown_lines, vec!["abc".to_string(), "0".to_string()]);
    }

    #[test]
    fn userdecks_ydke_round_trip() {
        let deck = Deck { main: vec![1, 0x0102_0304], extra: vec![100], side: Vec::new() };
        let url = to_ydke(&deck);
        assert!(url.starts_with("ydke://"));
        assert_eq!(parse_ydk(&url).deck, deck);
    }

    fn test_card(cards: &mut BTreeMap<u32, decklight_cards::CardData>, code: u32, type_: u32) {
        cards.insert(code, decklight_cards::decode_row(code as i64, 0, 0, type_ as i64, 0, 0, 4, 1, 1));
    }

    #[test]
    fn deck_summary_counts_mechanics_and_cards() {
        use decklight_core::constants as C;
        let mut cards = BTreeMap::new();
        test_card(&mut cards, 101, C::TYPE_MONSTER);
        test_card(&mut cards, 102, C::TYPE_SPELL);
        test_card(&mut cards, 103, C::TYPE_TRAP);
        test_card(&mut cards, 104, C::TYPE_MONSTER | C::TYPE_RITUAL);
        test_card(&mut cards, 105, C::TYPE_MONSTER | C::TYPE_PENDULUM);
        test_card(&mut cards, 201, C::TYPE_MONSTER | C::TYPE_FUSION);
        test_card(&mut cards, 202, C::TYPE_MONSTER | C::TYPE_SYNCHRO);
        test_card(&mut cards, 203, C::TYPE_MONSTER | C::TYPE_XYZ);
        test_card(&mut cards, 204, C::TYPE_MONSTER | C::TYPE_LINK);
        let deck = Deck {
            main: vec![101, 101, 101, 102, 102, 103, 104, 105],
            extra: vec![201, 202, 203, 204],
            side: Vec::new(),
        };
        let got = deck_summary(&deck, &cards, None);
        assert_eq!(got["counts"], serde_json::json!([5, 2, 1]));
        assert_eq!(
            got["mech"],
            serde_json::json!(["Fusion", "Synchro", "Xyz", "Link", "Ritual", "Pendulum"])
        );
        assert_eq!(got["main"], serde_json::json!([[101, 3], [102, 2], [103, 1], [104, 1], [105, 1]]));
        assert_eq!(got["extra"], serde_json::json!([[201, 1], [202, 1], [203, 1], [204, 1]]));
        assert_eq!(got["sig"], serde_json::json!([101, 102, 103]));
        assert_eq!(got["origin"], "Built in Decks");
        assert_eq!(got["arch"], "");
        assert_eq!(got["year"], 0);
    }

    #[test]
    fn deck_summary_sig_starts_with_the_cover() {
        use decklight_core::constants as C;
        let mut cards = BTreeMap::new();
        test_card(&mut cards, 101, C::TYPE_MONSTER);
        test_card(&mut cards, 102, C::TYPE_SPELL);
        test_card(&mut cards, 103, C::TYPE_TRAP);
        test_card(&mut cards, 201, C::TYPE_MONSTER | C::TYPE_FUSION);
        let deck = Deck { main: vec![101, 101, 102, 102, 103], extra: vec![201], side: Vec::new() };
        // Cover in the Main deck: first, never repeated.
        assert_eq!(deck_summary(&deck, &cards, Some(102))["sig"], serde_json::json!([102, 101, 103]));
        // Cover in the Extra deck.
        assert_eq!(deck_summary(&deck, &cards, Some(201))["sig"], serde_json::json!([201, 101, 102]));
        // A cover that is not in the deck is ignored.
        assert_eq!(deck_summary(&deck, &cards, Some(999))["sig"], serde_json::json!([101, 102, 103]));
    }

    #[test]
    fn deck_summary_of_an_empty_deck_is_empty() {
        let got = deck_summary(&Deck::default(), &BTreeMap::new(), None);
        assert_eq!(got["counts"], serde_json::json!([0, 0, 0]));
        assert_eq!(got["mech"], serde_json::json!([]));
        assert_eq!(got["sig"], serde_json::json!([]));
        assert_eq!(got["main"], serde_json::json!([]));
        assert_eq!(got["extra"], serde_json::json!([]));
    }

    fn temp_root(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dl-userdecks-{tag}-{}-{}", std::process::id(), crate::record::random_hex(3)))
    }

    #[test]
    fn userdecks_deck_value_rejects_bad_codes() {
        assert!(deck_from_value(&serde_json::json!({"main": [1, 2]})).is_ok());
        for bad in [
            serde_json::json!({"main": [0]}),
            serde_json::json!({"main": [-1]}),
            serde_json::json!({"main": [1.5]}),
            serde_json::json!({"main": ["1"]}),
            serde_json::json!({"main": "1"}),
        ] {
            assert!(deck_from_value(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn userdecks_store_round_trip_and_trash() {
        let root = temp_root("rt");
        let user = UserDecks::new(&root);
        let deck = Deck { main: vec![1, 2, 3], extra: vec![10], side: vec![20] };
        let saved = user.save(None, "  My Deck  ", &deck, Some(1)).unwrap();
        assert_eq!(saved.meta.name, "My Deck");
        assert!(saved.id.starts_with("my-deck-"));
        let got = user.get(&saved.id).unwrap();
        assert_eq!(got.deck, deck);
        assert_eq!(user.list().len(), 1);
        let saved2 = user.save(Some(&saved.id), "Renamed", &deck, None).unwrap();
        assert_eq!(saved2.id, saved.id);
        assert_eq!(user.get(&saved.id).unwrap().meta.name, "Renamed");
        user.delete(&saved.id).unwrap();
        assert!(user.list().is_empty());
        assert!(root.join("decks/user/.trash").is_dir());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn userdecks_unindexed_ydk_is_listed() {
        let root = temp_root("unindexed");
        let user = UserDecks::new(&root);
        std::fs::create_dir_all(&user.dir).unwrap();
        std::fs::write(user.dir.join("blue-eyes.ydk"), "#main\n1\n").unwrap();
        let list = user.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].meta.name, "Blue Eyes");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn starters_seed_once() {
        let root = std::env::temp_dir().join(format!("dl-starters-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let ydk = root.join("s.ydk");
        std::fs::write(&ydk, "#main\n89631139\n89631139\n#extra\n!side\n").unwrap();
        let u = UserDecks::new(&root);
        u.seed_starters(&[("Starter", ydk.clone()), ("Missing", root.join("nope.ydk"))]);
        let l = u.list();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].deck.main.len(), 2);
        u.delete(&l[0].id).unwrap();
        u.seed_starters(&[("Starter", ydk)]);
        assert!(u.list().is_empty(), "a deleted starter must not come back");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_cover_is_always_a_card_in_the_deck() {
        let root = std::env::temp_dir().join(format!("dl-cover-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let u = UserDecks::new(&root);
        let deck = |main: Vec<u32>, extra: Vec<u32>| Deck { main, extra, side: vec![7] };
        // A cover picked, then the card removed: the saved cover is dropped, the first Main card shows.
        let d = u.save(None, "Branded", &deck(vec![1, 2], vec![9]), Some(5)).unwrap();
        assert_eq!(d.meta.cover, None);
        assert_eq!(UserDecks::cover_of(&d), Some(1));
        // An older index entry whose cover left the deck.
        let mut stale = u.save(Some(&d.id), "Branded", &deck(vec![1, 2], vec![9]), Some(9)).unwrap();
        assert_eq!(UserDecks::cover_of(&stale), Some(9));
        stale.meta.cover = Some(5);
        assert_eq!(UserDecks::cover_of(&stale), Some(1));
        // A Side Deck card is not a cover; an empty Main falls back to the Extra.
        assert_eq!(u.save(Some(&d.id), "B", &deck(vec![], vec![9]), Some(7)).map(|d| UserDecks::cover_of(&d)).unwrap(), Some(9));
        let _ = std::fs::remove_dir_all(&root);
    }
}
