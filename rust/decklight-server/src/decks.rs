//! Lobby decks: naming, discovery, ydk parsing and the EDOPro host's deck rules (port of `Server.list_decks`,
//! `resolve_deck`, `check_text`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use decklight_cards::CardData;
use serde_json::{Value, json};

#[derive(Clone, Debug, Default)]
pub struct DeckSpec {
    pub main: Vec<u32>,
    pub extra: Vec<u32>,
    pub side: Vec<u32>,
    pub name: String,
}

impl From<decklight_cards::Deck> for DeckSpec {
    fn from(d: decklight_cards::Deck) -> Self {
        DeckSpec { main: d.main, extra: d.extra, side: d.side, name: String::new() }
    }
}

pub fn parse_ydk_named(text: &str, name: &str) -> DeckSpec {
    let mut d: DeckSpec = decklight_cards::parse_ydk(text).into();
    d.name = name.to_string();
    d
}

pub fn load_ydk(path: &Path) -> std::io::Result<DeckSpec> {
    let text = std::fs::read_to_string(path)?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
    Ok(parse_ydk_named(&text, &stem))
}

/// Python `_pretty_deck_name` (web/server.py): drop `-metaN`; a stem with capitals is CamelCase ("SnakeEyeAlter",
/// "Branded60"), so split it at lower→upper and before a trailing number; then `.replace("-", " ").title()`.
pub fn pretty_deck_name(stem: &str) -> String {
    let base = if let Some(i) = stem.rfind("-meta") {
        let tail = &stem[i + 5..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) { &stem[..i] } else { stem }
    } else {
        stem
    };
    let base = if base.chars().any(|c| c.is_uppercase()) { split_camel(base) } else { base.to_string() };
    py_title(&base.replace('-', " "))
}

fn split_camel(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let digits_from = chars.iter().rposition(|c| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    let mut out = String::with_capacity(s.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 {
            let prev = chars[i - 1];
            let camel = prev.is_lowercase() && c.is_uppercase();
            let number = i == digits_from && digits_from < chars.len() && prev.is_alphabetic();
            if camel || number {
                out.push(' ');
            }
        }
        out.push(c);
    }
    out
}

fn py_title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut cap = true;
    for c in s.chars() {
        if c.is_alphabetic() {
            if cap {
                out.extend(c.to_uppercase());
            } else {
                out.extend(c.to_lowercase());
            }
            cap = false;
        } else {
            out.push(c);
            cap = true;
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct DeckDirs {
    pub dirs: Vec<(String, PathBuf)>,
}

impl DeckDirs {
    pub fn default_for(root: &Path) -> DeckDirs {
        // Only the curated pool is listed; `resolve` still searches newcards and the gauntlet so old saved
        // settings and puzzles keep working (user, 2026-11-13).
        DeckDirs { dirs: vec![("pool".to_string(), root.join("decks/pool"))] }
    }

    pub fn from_paths(paths: &[PathBuf]) -> DeckDirs {
        DeckDirs {
            dirs: paths
                .iter()
                .map(|p| {
                    (
                        p.file_name().and_then(|n| n.to_str()).unwrap_or("pool").to_string(),
                        p.clone(),
                    )
                })
                .collect(),
        }
    }
}

pub struct DeckCatalog {
    pub dirs: DeckDirs,
    pub coverage: BTreeMap<String, Value>,
    pub user_decks: BTreeMap<String, String>,
    pub user_dir: PathBuf,
    pub tuned: BTreeMap<String, (String, PathBuf)>,
    /// The curated prebuilt set (`config/app/prebuilt_decks.txt`); `None` keeps the old list-everything behaviour.
    pub prebuilt: Option<Vec<String>>,
    pub root: PathBuf,
}

impl DeckCatalog {
    pub fn new(
        dirs: DeckDirs,
        coverage_path: &Path,
        root: &Path,
        prebuilt_path: &Path,
    ) -> DeckCatalog {
        let coverage = std::fs::read_to_string(coverage_path)
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.get("decks").and_then(Value::as_object).cloned())
            .map(|o| o.into_iter().collect())
            .unwrap_or_else(|| {
                eprintln!("WARN web content: deck coverage unreadable; no Trained/Related tags");
                BTreeMap::new()
            });
        let tuned = discover_tuned_decks(&dirs, root);
        DeckCatalog {
            dirs,
            coverage,
            user_decks: BTreeMap::new(),
            user_dir: root.join("decks/user"),
            tuned,
            prebuilt: load_prebuilt(prebuilt_path),
            root: root.to_path_buf(),
        }
    }

    /// Where an id resolves: the configured dirs first, then the standard bundle dirs so settings and puzzles
    /// that name a deck outside the curated listing keep working.
    fn resolve_path(&self, stem: &str) -> Option<PathBuf> {
        for (_group, dir) in &self.dirs.dirs {
            let p = dir.join(format!("{stem}.ydk"));
            if p.is_file() {
                return Some(p);
            }
        }
        for sub in ["pool", "gauntlet", "newcards"] {
            let p = self.root.join("decks").join(sub).join(format!("{stem}.ydk"));
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    pub fn list_decks(&self) -> Vec<Value> {
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        if let Some(prebuilt) = &self.prebuilt {
            for stem in prebuilt {
                if !seen.insert(stem.clone()) {
                    continue;
                }
                if self.resolve_path(stem).is_none() {
                    continue;
                }
                let mut row = json!({
                    "id": stem,
                    "name": pretty_deck_name(stem),
                    "group": "prebuilt",
                });
                if let Some(cov) = self.coverage.get(stem) {
                    row["forge"] = cov.clone();
                }
                out.push(row);
            }
        } else {
            for (group, dir) in &self.dirs.dirs {
                let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
                    Ok(rd) => rd
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("ydk"))
                        .collect(),
                    Err(_) => Vec::new(),
                };
                paths.sort();
                for p in paths {
                    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                    if !seen.insert(stem.clone()) {
                        continue;
                    }
                    let mut row = json!({
                        "id": stem,
                        "name": pretty_deck_name(&stem),
                        "group": group,
                    });
                    if let Some(cov) = self.coverage.get(&stem) {
                        row["forge"] = cov.clone();
                    }
                    out.push(row);
                }
            }
        }
        let mut tuned: Vec<(&String, &(String, PathBuf))> = self.tuned.iter().collect();
        tuned.sort_by(|a, b| a.1 .0.cmp(&b.1 .0));
        for (id, (name, _p)) in tuned {
            out.push(json!({"id": id, "name": name, "group": "tuned"}));
        }
        for (i, (id, _text)) in self.user_decks.iter().enumerate() {
            out.push(json!({"id": id, "name": format!("Pasted deck {}", i + 1), "group": "user"}));
        }
        out
    }

    /// The saved decks' folder (default `decks/user`), as `Config.user_decks_dir` sets it for `UserDecks` too.
    pub fn set_user_dir(&mut self, dir: &Path) {
        self.user_dir = dir.to_path_buf();
    }

    /// A user deck by id: `{"user": "<id>"}` and the string form `"user:<id>"` share this validation and its errors.
    fn resolve_user(&self, id: &str) -> (Option<DeckSpec>, Option<String>) {
        let valid =
            !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        let path = self.user_dir.join(format!("{id}.ydk"));
        if valid && path.is_file() {
            return match load_ydk(&path) {
                Ok(d) => (Some(d), None),
                Err(e) => (None, Some(format!("unknown deck {id:?}: {e}"))),
            };
        }
        (None, Some(format!("unknown deck {id:?}").replace('"', "'")))
    }

    pub fn resolve(&self, spec: &Value) -> (Option<DeckSpec>, Option<String>) {
        if let Some(obj) = spec.as_object() {
            if let Some(text) = obj.get("ydk").and_then(Value::as_str) {
                return (Some(parse_ydk_named(text, "pasted")), None);
            }
            if let Some(id) = obj.get("user").and_then(Value::as_str) {
                return self.resolve_user(id);
            }
            return (
                None,
                Some("deck must be a deck id, {\"ydk\": \"...\"} or {\"user\": \"<id>\"}".to_string()),
            );
        }
        if let Some(id) = spec.as_str() {
            if let Some(uid) = id.strip_prefix("user:") {
                return self.resolve_user(uid);
            }
            if let Some(p) = self.resolve_path(id) {
                return match load_ydk(&p) {
                    Ok(d) => (Some(d), None),
                    Err(e) => (None, Some(format!("unknown deck {id:?}: {e}"))),
                };
            }
            if let Some((_name, p)) = self.tuned.get(id) {
                return match load_ydk(p) {
                    Ok(d) => (Some(d), None),
                    Err(e) => (None, Some(format!("unknown deck {id:?}: {e}"))),
                };
            }
            if let Some(text) = self.user_decks.get(id) {
                return (Some(parse_ydk_named(text, id)), None);
            }
            return (None, Some(format!("unknown deck {id:?}").replace('"', "'")));
        }
        (None, Some("deck must be a deck id or {\"ydk\": \"...\"}".to_string()))
    }
}

/// `config/app/prebuilt_decks.txt`: one deck stem per line, `#` comments and blank lines skipped. An empty or
/// missing manifest means no curated list (the old behaviour).
fn load_prebuilt(path: &Path) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(path).ok()?;
    let stems: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_string)
        .collect();
    if stems.is_empty() { None } else { Some(stems) }
}

fn split_deck(codes: &[u32], cards: &BTreeMap<u32, CardData>) -> (Vec<u32>, Vec<u32>) {
    use decklight_core::constants as C;
    let extra_bits = C::TYPE_FUSION | C::TYPE_SYNCHRO | C::TYPE_XYZ;
    let mut main = Vec::new();
    let mut extra = Vec::new();
    for &c in codes {
        let t = cards.get(&c).map(|d| d.type_).unwrap_or(0);
        if t & extra_bits != 0 || (t & C::TYPE_LINK != 0 && t & C::TYPE_MONSTER != 0) {
            extra.push(c);
        } else {
            main.push(c);
        }
    }
    (main, extra)
}

/// `Server.check_text`: None when legal, else the EDOPro host's error text.
pub fn check_deck(
    deck: &DeckSpec,
    cards: &BTreeMap<u32, CardData>,
    emb_codes: &BTreeSet<u32>,
    names: &BTreeMap<u32, String>,
) -> Option<String> {
    let mut all = deck.main.clone();
    all.extend(&deck.extra);
    let (main, extra) = split_deck(&all, cards);
    let bad = if !(40..=60).contains(&main.len()) {
        Some((0, main.len(), 40, 60, 0u32))
    } else if extra.len() > 15 {
        Some((1, extra.len(), 0, 15, 0u32))
    } else {
        let mut found = None;
        for &c in main.iter().chain(extra.iter()) {
            if !cards.contains_key(&c) || !emb_codes.contains(&c) {
                found = Some((2, 0, 0, 0, c));
                break;
            }
        }
        found
    }?;
    let (kind, cur, minimum, maximum, code) = bad;
    let extra_txt = if code != 0 {
        let nm = names.get(&code).map(String::as_str).unwrap_or("unknown card");
        format!(": {code} ({nm})")
    } else {
        String::new()
    };
    let what = match kind {
        0 => "main-deck count",
        1 => "extra-deck count",
        _ => "unknown card",
    };
    Some(format!("illegal deck ({what}: have {cur}, allowed {minimum}-{maximum}){extra_txt}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_names_match_python() {
        assert_eq!(pretty_deck_name("blue-eyes"), "Blue Eyes");
        assert_eq!(pretty_deck_name("angelechy-meta1"), "Angelechy");
        assert_eq!(pretty_deck_name("abc-dragon"), "Abc Dragon");
        assert_eq!(pretty_deck_name("ABC-Dragon"), "Abc Dragon");
        assert_eq!(pretty_deck_name("meta2"), "Meta2");
        assert_eq!(pretty_deck_name("SnakeEyeAlter"), "Snake Eye Alter");
        assert_eq!(pretty_deck_name("Branded60"), "Branded 60");
        assert_eq!(pretty_deck_name("BestowedDragonF"), "Bestowed Dragon F");
        assert_eq!(pretty_deck_name("sky-striker-meta1"), "Sky Striker");
    }

    fn temp_root(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "dl-decks-{tag}-{}-{}",
            std::process::id(),
            crate::record::random_hex(3)
        ))
    }

    fn catalog(root: &Path) -> DeckCatalog {
        DeckCatalog::new(
            DeckDirs { dirs: Vec::new() },
            &root.join("missing-coverage.json"),
            root,
            &root.join("missing-prebuilt.txt"),
        )
    }

    #[test]
    fn manifest_lists_only_prebuilt_in_order_and_resolve_sees_the_rest() {
        let root = temp_root("prebuilt");
        let pool = root.join("decks/pool");
        std::fs::create_dir_all(&pool).unwrap();
        for stem in ["a", "b", "c"] {
            std::fs::write(pool.join(format!("{stem}.ydk")), "#main\n1\n").unwrap();
        }
        std::fs::create_dir_all(root.join("decks/newcards")).unwrap();
        std::fs::write(root.join("decks/newcards/n.ydk"), "#main\n1\n").unwrap();
        let manifest = root.join("prebuilt.txt");
        std::fs::write(&manifest, "# curated\nc\n\n# gone\na\n").unwrap();
        let dirs = DeckDirs { dirs: vec![("pool".to_string(), pool)] };
        let cat = DeckCatalog::new(dirs, &root.join("missing-coverage.json"), &root, &manifest);
        let decks = cat.list_decks();
        let ids: Vec<&str> = decks.iter().map(|d| d["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["c", "a"]);
        assert!(decks.iter().all(|d| d["group"] == "prebuilt"));
        // Unlisted pool and newcards stems still resolve for old settings and puzzles.
        for spec in [json!("b"), json!("n")] {
            let (deck, err) = cat.resolve(&spec);
            assert!(deck.is_some(), "{spec}: {err:?}");
            assert_eq!(err, None);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn deck_catalog_resolves_user_string_form() {
        let root = temp_root("user-form");
        std::fs::create_dir_all(root.join("decks/user")).unwrap();
        std::fs::write(root.join("decks/user/fun-deck.ydk"), "#main\n123\n#extra\n456\n!side\n").unwrap();
        let cat = catalog(&root);
        let (deck, err) = cat.resolve(&json!("user:fun-deck"));
        assert_eq!(err, None);
        let deck = deck.unwrap();
        assert_eq!(deck.main, vec![123]);
        assert_eq!(deck.extra, vec![456]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn deck_catalog_rejects_bad_or_missing_user_ids() {
        let root = temp_root("user-bad");
        std::fs::create_dir_all(root.join("decks/user")).unwrap();
        let cat = catalog(&root);
        for (spec, error) in [
            (json!("user:../x"), "unknown deck '../x'"),
            (json!("user:nope"), "unknown deck 'nope'"),
            (json!("user:"), "unknown deck ''"),
        ] {
            let (deck, err) = cat.resolve(&spec);
            assert!(deck.is_none(), "{spec}");
            assert_eq!(err.as_deref(), Some(error), "{spec}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}

fn content_key(d: &DeckSpec) -> (Vec<u32>, Vec<u32>) {
    let mut m = d.main.clone();
    let mut e = d.extra.clone();
    m.sort_unstable();
    e.sort_unstable();
    (m, e)
}

/// `web.models._register_tuned_decks`: fine-tunes' own decks join the lobby as group "tuned" unless a lobby deck has
/// the same content. Pure filesystem discovery, like the Python server.
fn discover_tuned_decks(dirs: &DeckDirs, root: &Path) -> BTreeMap<String, (String, PathBuf)> {
    let mut lobby: BTreeMap<(Vec<u32>, Vec<u32>), String> = BTreeMap::new();
    for (_g, dir) in &dirs.dirs {
        if let Ok(rd) = std::fs::read_dir(dir) {
            let mut paths: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("ydk"))
                .collect();
            paths.sort();
            for p in paths {
                if let Ok(d) = load_ydk(&p) {
                    lobby.entry(content_key(&d)).or_insert_with(|| {
                        p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string()
                    });
                }
            }
        }
    }
    // Python: sorted(glob("*/FINETUNE.json")) + sorted(glob("*/*/FINETUNE.json")).
    let mut direct: Vec<PathBuf> = Vec::new();
    let mut nested: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root.join("runs")) {
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let d = p.join("FINETUNE.json");
            if d.is_file() {
                direct.push(d);
            }
            if let Ok(rd2) = std::fs::read_dir(&p) {
                for e2 in rd2.filter_map(|e| e.ok()) {
                    let n = e2.path().join("FINETUNE.json");
                    if n.is_file() {
                        nested.push(n);
                    }
                }
            }
        }
    }
    direct.sort();
    nested.sort();
    let mut cands = direct;
    cands.extend(nested);
    // `discover_models`: found.sort(key=_mtime, reverse=True) -- newest fine-tune first.
    cands.sort_by(|a, b| {
        let ma = std::fs::metadata(a).and_then(|m| m.modified()).ok();
        let mb = std::fs::metadata(b).and_then(|m| m.modified()).ok();
        mb.cmp(&ma)
    });
    let mut by_key: BTreeMap<(Vec<u32>, Vec<u32>), String> = BTreeMap::new();
    let mut tuned: BTreeMap<String, (String, PathBuf)> = BTreeMap::new();
    for f in cands {
        let parent = f.parent().unwrap_or(Path::new("."));
        if !parent.join("policy/best.pt").is_file() || !parent.join("value/best.pt").is_file() {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        let Ok(data) = serde_json::from_str::<Value>(&text) else { continue };
        let deck_ref = data.get("deck").and_then(Value::as_str).unwrap_or("");
        if deck_ref.is_empty() {
            continue;
        }
        let dp = PathBuf::from(deck_ref);
        let dp = if dp.is_absolute() { dp } else { root.join(dp) };
        if !dp.is_file() {
            continue;
        }
        let Ok(d) = load_ydk(&dp) else { continue };
        let key = content_key(&d);
        if lobby.contains_key(&key) {
            continue;
        }
        let deck_name = dp.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let abs = std::fs::canonicalize(&f).unwrap_or(f.clone());
        let h = crate::record::sha256_hex(abs.to_string_lossy().as_bytes());
        let short = &h[..4];
        let did = by_key.entry(key).or_insert_with(|| format!("tuned-{short}")).clone();
        tuned.insert(did, (deck_name, dp));
    }
    tuned
}
