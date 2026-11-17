//! Card search: a lazily built in-memory index over every card name and effect text.
//!
//! Name search is fuzzy (normalised trigram candidates, prefix/token boosts and Damerau-Levenshtein re-ranking);
//! text search is BM25 with a light stemmer and required quoted phrases. The two rankings are fused with reciprocal
//! rank fusion. Archetype names from `edopro-strings.conf` add their members just below direct name hits.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use decklight_cards::emb::Emb;
use decklight_cards::{CardData, texts::Texts};
use decklight_core::constants as C;
use serde_json::Value;

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
const RRF_K: f64 = 60.0;
const NAME_WEIGHT: f64 = 2.0;
const TEXT_WEIGHT: f64 = 1.0;

#[derive(Clone, Debug)]
struct Row {
    code: u32,
    name: String,
    kind: &'static str,
    line: String,
    extra: bool,
    type_: u32,
    attribute: u32,
    race: u64,
    level: u32,
    link: u32,
    attack: i32,
    defense: i32,
    setcodes: Vec<u16>,
    archetypes: Vec<usize>,
    text: String,
}

#[derive(Clone, Debug)]
struct Doc {
    norm: String,
    tokens: Vec<String>,
    len: usize,
}

#[derive(Clone, Debug)]
struct Archetype {
    code: u16,
    name: String,
    norm: String,
    nospace: String,
    members: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct SearchRow {
    pub code: u32,
    pub name: String,
    pub kind: &'static str,
    pub line: String,
    pub extra: bool,
    pub by: &'static str,
    pub snippet: String,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub total: usize,
    pub rows: Vec<SearchRow>,
}

pub struct SearchIndex {
    rows: Vec<Row>,
    docs: Vec<Doc>,
    name_norm: Vec<String>,
    name_nospace: Vec<String>,
    name_tokens: Vec<Vec<String>>,
    trigrams: HashMap<String, Vec<u32>>,
    postings: HashMap<String, Vec<(u32, u32)>>,
    avgdl: f64,
    archetypes: Vec<Archetype>,
    card_to_row: HashMap<u32, usize>,
    card_alias: BTreeMap<u32, u32>,
    pub size_bytes: usize,
}

// ------------------------------------------------------------------ normalisation

fn fold_char(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'æ' => 'e',
        'ç' => 'c',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'ø' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ý' | 'ÿ' => 'y',
        'ß' => 's',
        'œ' => 'e',
        _ => c,
    }
}

/// Lowercase, fold accents, replace punctuation with spaces, collapse spaces.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.to_lowercase().chars() {
        let c = fold_char(c);
        if c.is_alphanumeric() {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            out.push(c);
        } else {
            space = true;
        }
    }
    out
}

fn nospace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn stem_word(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let strip = |suffix: &str| {
        let n = suffix.chars().count();
        if chars.len() >= n + 3 && word.ends_with(suffix) {
            Some(chars[..chars.len() - n].iter().collect::<String>())
        } else {
            None
        }
    };
    if let Some(s) = strip("ies") {
        return format!("{s}y");
    }
    for suffix in ["ing", "ed", "es", "s"] {
        if let Some(s) = strip(suffix) {
            return s;
        }
    }
    word.to_string()
}

fn tokens(s: &str) -> Vec<String> {
    normalize(s).split_whitespace().map(stem_word).collect()
}

fn name_tokens(s: &str) -> Vec<String> {
    normalize(s).split_whitespace().map(str::to_string).collect()
}

fn trigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return Vec::new();
    }
    chars.windows(3).map(|w| w.iter().collect()).collect()
}

fn damerau(a: &[char], b: &[char]) -> usize {
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=b.len() {
        d[0][j] = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

// ------------------------------------------------------------------ card line formatting

fn attribute_name(a: u32) -> &'static str {
    match a {
        C::ATTRIBUTE_EARTH => "EARTH",
        C::ATTRIBUTE_WATER => "WATER",
        C::ATTRIBUTE_FIRE => "FIRE",
        C::ATTRIBUTE_WIND => "WIND",
        C::ATTRIBUTE_LIGHT => "LIGHT",
        C::ATTRIBUTE_DARK => "DARK",
        C::ATTRIBUTE_DIVINE => "DIVINE",
        _ => "?",
    }
}

fn race_name(r: u64) -> &'static str {
    match r {
        x if x == C::RACE_WARRIOR as u64 => "Warrior",
        x if x == C::RACE_SPELLCASTER as u64 => "Spellcaster",
        x if x == C::RACE_FAIRY as u64 => "Fairy",
        x if x == C::RACE_FIEND as u64 => "Fiend",
        x if x == C::RACE_ZOMBIE as u64 => "Zombie",
        x if x == C::RACE_MACHINE as u64 => "Machine",
        x if x == C::RACE_AQUA as u64 => "Aqua",
        x if x == C::RACE_PYRO as u64 => "Pyro",
        x if x == C::RACE_ROCK as u64 => "Rock",
        x if x == C::RACE_WINGEDBEAST as u64 => "Winged Beast",
        x if x == C::RACE_PLANT as u64 => "Plant",
        x if x == C::RACE_INSECT as u64 => "Insect",
        x if x == C::RACE_THUNDER as u64 => "Thunder",
        x if x == C::RACE_DRAGON as u64 => "Dragon",
        x if x == C::RACE_BEAST as u64 => "Beast",
        x if x == C::RACE_BEASTWARRIOR as u64 => "Beast-Warrior",
        x if x == C::RACE_DINOSAUR as u64 => "Dinosaur",
        x if x == C::RACE_FISH as u64 => "Fish",
        x if x == C::RACE_SEASERPENT as u64 => "Sea Serpent",
        x if x == C::RACE_REPTILE as u64 => "Reptile",
        x if x == C::RACE_PSYCHIC as u64 => "Psychic",
        x if x == C::RACE_DIVINE as u64 => "Divine",
        x if x == C::RACE_CREATORGOD as u64 => "Creator God",
        x if x == C::RACE_WYRM as u64 => "Wyrm",
        x if x == C::RACE_CYBERSE as u64 => "Cyberse",
        x if x == C::RACE_ILLUSION as u64 => "Illusion",
        x if x == C::RACE_CYBORG as u64 => "Cyborg",
        x if x == C::RACE_MAGICALKNIGHT as u64 => "Magical Knight",
        x if x == C::RACE_HIGHDRAGON as u64 => "High Dragon",
        x if x == C::RACE_OMEGAPSYCHIC as u64 => "Omega Psychic",
        x if x == C::RACE_CELESTIALWARRIOR as u64 => "Celestial Warrior",
        x if x == C::RACE_GALAXY as u64 => "Galaxy",
        x if x == C::RACE_YOKAI => "Yokai",
        _ => "?",
    }
}

fn race_bit(name: &str) -> Option<u64> {
    let key = normalize(name).replace(' ', "");
    let values: [(&str, u64); 33] = [
        ("warrior", C::RACE_WARRIOR as u64),
        ("spellcaster", C::RACE_SPELLCASTER as u64),
        ("fairy", C::RACE_FAIRY as u64),
        ("fiend", C::RACE_FIEND as u64),
        ("zombie", C::RACE_ZOMBIE as u64),
        ("machine", C::RACE_MACHINE as u64),
        ("aqua", C::RACE_AQUA as u64),
        ("pyro", C::RACE_PYRO as u64),
        ("rock", C::RACE_ROCK as u64),
        ("wingedbeast", C::RACE_WINGEDBEAST as u64),
        ("plant", C::RACE_PLANT as u64),
        ("insect", C::RACE_INSECT as u64),
        ("thunder", C::RACE_THUNDER as u64),
        ("dragon", C::RACE_DRAGON as u64),
        ("beast", C::RACE_BEAST as u64),
        ("beastwarrior", C::RACE_BEASTWARRIOR as u64),
        ("dinosaur", C::RACE_DINOSAUR as u64),
        ("fish", C::RACE_FISH as u64),
        ("seaserpent", C::RACE_SEASERPENT as u64),
        ("reptile", C::RACE_REPTILE as u64),
        ("psychic", C::RACE_PSYCHIC as u64),
        ("divine", C::RACE_DIVINE as u64),
        ("creatorgod", C::RACE_CREATORGOD as u64),
        ("wyrm", C::RACE_WYRM as u64),
        ("cyberse", C::RACE_CYBERSE as u64),
        ("illusion", C::RACE_ILLUSION as u64),
        ("cyborg", C::RACE_CYBORG as u64),
        ("magicalknight", C::RACE_MAGICALKNIGHT as u64),
        ("highdragon", C::RACE_HIGHDRAGON as u64),
        ("omegapsychic", C::RACE_OMEGAPSYCHIC as u64),
        ("celestialwarrior", C::RACE_CELESTIALWARRIOR as u64),
        ("galaxy", C::RACE_GALAXY as u64),
        ("yokai", C::RACE_YOKAI),
    ];
    values.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn attribute_bit(name: &str) -> Option<u32> {
    let key = normalize(name);
    let values = [
        ("earth", C::ATTRIBUTE_EARTH),
        ("water", C::ATTRIBUTE_WATER),
        ("fire", C::ATTRIBUTE_FIRE),
        ("wind", C::ATTRIBUTE_WIND),
        ("light", C::ATTRIBUTE_LIGHT),
        ("dark", C::ATTRIBUTE_DARK),
        ("divine", C::ATTRIBUTE_DIVINE),
    ];
    values.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn frame_bit(name: &str) -> Option<u32> {
    let key = normalize(name).replace(' ', "");
    let values = [
        ("normal", C::TYPE_NORMAL),
        ("effect", C::TYPE_EFFECT),
        ("ritual", C::TYPE_RITUAL),
        ("fusion", C::TYPE_FUSION),
        ("synchro", C::TYPE_SYNCHRO),
        ("xyz", C::TYPE_XYZ),
        ("pendulum", C::TYPE_PENDULUM),
        ("link", C::TYPE_LINK),
        ("token", C::TYPE_TOKEN),
    ];
    values.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// Inclusive `[min, max]` filter (both integers). `min > max` is an error.
fn range_filter(value: &Value, key: &str) -> Result<(i64, i64), String> {
    let arr = value.as_array().ok_or_else(|| format!("{key} must be [min, max]"))?;
    if arr.len() != 2 {
        return Err(format!("{key} must be [min, max]"));
    }
    let lo = arr[0].as_i64().ok_or_else(|| format!("{key} must be [min, max]"))?;
    let hi = arr[1].as_i64().ok_or_else(|| format!("{key} must be [min, max]"))?;
    if lo > hi {
        return Err(format!("{key} min {lo} is greater than max {hi}"));
    }
    Ok((lo, hi))
}

fn st_type_matches(type_: u32, values: &BTreeSet<String>) -> bool {
    if type_ & (C::TYPE_SPELL | C::TYPE_TRAP) == 0 {
        return false;
    }
    let other_bits = C::TYPE_QUICKPLAY
        | C::TYPE_CONTINUOUS
        | C::TYPE_EQUIP
        | C::TYPE_FIELD
        | C::TYPE_RITUAL
        | C::TYPE_COUNTER;
    values.iter().any(|v| match v.as_str() {
        "normal" => type_ & other_bits == 0,
        "quickplay" => type_ & C::TYPE_QUICKPLAY != 0,
        "continuous" => type_ & C::TYPE_CONTINUOUS != 0,
        "equip" => type_ & C::TYPE_EQUIP != 0,
        "field" => type_ & C::TYPE_FIELD != 0,
        "ritual" => type_ & C::TYPE_RITUAL != 0,
        "counter" => type_ & C::TYPE_COUNTER != 0,
        _ => false,
    })
}

fn is_extra(type_: u32) -> bool {
    type_ & C::TYPE_MONSTER != 0
        && type_ & (C::TYPE_FUSION | C::TYPE_SYNCHRO | C::TYPE_XYZ | C::TYPE_LINK) != 0
}

fn kind_of(type_: u32) -> &'static str {
    if type_ & C::TYPE_MONSTER != 0 {
        "monster"
    } else if type_ & C::TYPE_SPELL != 0 {
        "spell"
    } else if type_ & C::TYPE_TRAP != 0 {
        "trap"
    } else {
        "other"
    }
}

fn spell_trap_line(type_: u32) -> String {
    if type_ & C::TYPE_SPELL != 0 {
        let frame = if type_ & C::TYPE_QUICKPLAY != 0 {
            "Quick-Play"
        } else if type_ & C::TYPE_CONTINUOUS != 0 {
            "Continuous"
        } else if type_ & C::TYPE_EQUIP != 0 {
            "Equip"
        } else if type_ & C::TYPE_FIELD != 0 {
            "Field"
        } else if type_ & C::TYPE_RITUAL != 0 {
            "Ritual"
        } else {
            "Normal"
        };
        format!("{frame} Spell")
    } else if type_ & C::TYPE_COUNTER != 0 {
        "Counter Trap".to_string()
    } else if type_ & C::TYPE_CONTINUOUS != 0 {
        "Continuous Trap".to_string()
    } else {
        "Normal Trap".to_string()
    }
}

fn stat_line(cd: &CardData) -> String {
    let t = cd.type_;
    if t & C::TYPE_MONSTER == 0 {
        return spell_trap_line(t);
    }
    let attr = attribute_name(cd.attribute);
    let race = race_name(cd.race);
    if t & C::TYPE_LINK != 0 {
        return format!("Link-{} · {attr} · {race} · {}", cd.level, cd.attack);
    }
    let level = if t & C::TYPE_XYZ != 0 {
        format!("Rank {}", cd.level)
    } else {
        format!("Level {}", cd.level)
    };
    format!("{level} · {attr} · {race} · {}/{}", cd.attack, cd.defense)
}

// ------------------------------------------------------------------ build

fn parse_archetypes(conf: &Path) -> Vec<(u16, String)> {
    let Ok(text) = std::fs::read_to_string(conf) else { return Vec::new() };
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("!setname ") else { continue };
        let mut parts = rest.splitn(2, char::is_whitespace);
        let (Some(code), Some(name)) = (parts.next(), parts.next()) else { continue };
        let Some(hex) = code.strip_prefix("0x").or_else(|| code.strip_prefix("0X")) else { continue };
        let Ok(code) = u32::from_str_radix(hex, 16) else { continue };
        if code > u16::MAX as u32 {
            continue;
        }
        let name = name.trim();
        if !name.is_empty() {
            out.push((code as u16, name.to_string()));
        }
    }
    out
}

fn setcode_matches(setcodes: &[u16], s: u16) -> bool {
    setcodes.iter().any(|&p| (p & 0x0fff) == (s & 0x0fff) && (p & s) == s)
}

fn estimate_size(rows: &[Row], docs: &[Doc], trigrams: &HashMap<String, Vec<u32>>, names: &[String]) -> usize {
    let mut n = std::mem::size_of::<Row>() * rows.len();
    n += rows.iter().map(|r| r.name.len() + r.line.len() + r.text.len()).sum::<usize>();
    n += names.iter().map(String::len).sum::<usize>();
    n += docs.iter().map(|d| d.norm.len() + d.tokens.iter().map(String::len).sum::<usize>()).sum::<usize>();
    n += trigrams.iter().map(|(k, v)| k.len() + v.len() * 4).sum::<usize>();
    n
}

impl SearchIndex {
    /// Build the index. Archetype names come from `strings_conf` (`!setname` lines).
    pub fn build(cards: &BTreeMap<u32, CardData>, texts: &Texts, strings_conf: &Path) -> SearchIndex {
        let raw_archetypes = parse_archetypes(strings_conf);
        let mut archetypes: Vec<Archetype> = raw_archetypes
            .into_iter()
            .map(|(code, name)| {
                let norm = normalize(&name);
                Archetype { code, name, nospace: nospace(&norm), norm, members: Vec::new() }
            })
            .collect();
        archetypes.sort_by(|a, b| a.code.cmp(&b.code));

        // One row per card NAME: alt arts collapse to the lowest code with that name.
        let mut by_name: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for (&code, _) in cards {
            let name = texts.name(code);
            let name = if name.is_empty() { format!("Card {code}") } else { name.to_string() };
            by_name.entry(name).or_default().push(code);
        }

        let mut rows = Vec::with_capacity(by_name.len());
        let mut card_to_row = HashMap::with_capacity(cards.len());
        let mut card_alias = BTreeMap::new();
        let mut name_norm = Vec::with_capacity(by_name.len());
        let mut name_nospace = Vec::with_capacity(by_name.len());
        let mut name_tokens_v = Vec::with_capacity(by_name.len());
        let mut trigrams_map: HashMap<String, Vec<u32>> = HashMap::new();

        for (name, codes) in by_name {
            let code = *codes.iter().min().unwrap();
            let cd = &cards[&code];
            let idx = rows.len() as u32;
            for c in &codes {
                card_to_row.insert(*c, idx as usize);
                if let Some(d) = cards.get(c) {
                    card_alias.insert(*c, if d.alias == 0 { *c } else { d.alias });
                }
            }
            let row_archetypes: Vec<usize> = archetypes
                .iter()
                .enumerate()
                .filter(|(_, a)| setcode_matches(&cd.setcodes, a.code))
                .map(|(i, _)| i)
                .collect();
            let norm = normalize(&name);
            let nospace_name = nospace(&norm);
            let toks = name_tokens(&name);
            for t in trigrams(&norm).into_iter().chain(trigrams(&nospace_name)) {
                trigrams_map.entry(t).or_default().push(idx);
            }
            name_norm.push(norm);
            name_nospace.push(nospace_name);
            name_tokens_v.push(toks);
            rows.push(Row {
                code,
                name,
                kind: kind_of(cd.type_),
                line: stat_line(cd),
                extra: is_extra(cd.type_),
                type_: cd.type_,
                attribute: cd.attribute,
                race: cd.race,
                level: cd.level,
                link: if cd.type_ & C::TYPE_LINK != 0 { cd.level } else { 0 },
                attack: cd.attack,
                defense: cd.defense,
                setcodes: cd.setcodes.clone(),
                archetypes: row_archetypes,
                text: texts.effect_text(code),
            });
        }
        for row in &rows {
            for &ai in &row.archetypes {
                archetypes[ai].members.push(card_to_row[&row.code]);
            }
        }
        let mut docs = Vec::with_capacity(rows.len());
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut total_len = 0usize;
        for (i, row) in rows.iter().enumerate() {
            let norm = normalize(&row.text);
            // BM25 covers the effect text plus the card name; quoted phrases still match the effect text only.
            let mut toks = tokens(&row.text);
            toks.extend(tokens(&row.name));
            total_len += toks.len();
            let mut tf: HashMap<&String, u32> = HashMap::new();
            for t in &toks {
                *tf.entry(t).or_insert(0) += 1;
            }
            for (t, n) in tf {
                postings.entry(t.clone()).or_default().push((i as u32, n));
            }
            docs.push(Doc { norm, tokens: toks, len: 0 });
            docs[i].len = docs[i].tokens.len();
        }
        let avgdl = if docs.is_empty() { 0.0 } else { total_len as f64 / docs.len() as f64 };
        let size_bytes = estimate_size(&rows, &docs, &trigrams_map, &name_norm);
        SearchIndex {
            rows,
            docs,
            name_norm,
            name_nospace,
            name_tokens: name_tokens_v,
            trigrams: trigrams_map,
            postings,
            avgdl,
            archetypes,
            card_to_row,
            card_alias,
            size_bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn card_text(&self, code: u32) -> Option<&str> {
        self.card_to_row.get(&code).map(|&i| self.rows[i].text.as_str())
    }

    /// A single card's detail, or `None` when the code is not in the card store.
    pub fn card_json(&self, code: u32) -> Option<Value> {
        let &i = self.card_to_row.get(&code)?;
        let row = &self.rows[i];
        let archetypes: Vec<Value> = row
            .archetypes
            .iter()
            .map(|&a| Value::String(self.archetypes[a].name.clone()))
            .collect();
        let alias = self.card_alias.get(&code).copied().unwrap_or(code);
        Some(serde_json::json!({
            "code": code,
            "name": row.name,
            "kind": row.kind,
            "line": row.line,
            "text": row.text,
            "archetypes": archetypes,
            "extra": row.extra,
            "alias": alias,
        }))
    }

    fn candidate_rows(&self, q: &str, q_nospace: &str) -> Vec<usize> {
        if q.chars().count() < 3 {
            return (0..self.rows.len()).collect();
        }
        let mut set: BTreeSet<u32> = BTreeSet::new();
        for t in trigrams(q).into_iter().chain(trigrams(q_nospace)) {
            if let Some(list) = self.trigrams.get(&t) {
                set.extend(list.iter().copied());
            }
        }
        if set.len() < 16 {
            return (0..self.rows.len()).collect();
        }
        set.into_iter().map(|i| i as usize).collect()
    }

    fn name_score(&self, q: &str, q_nospace: &str, q_tokens: &[String], i: usize) -> Option<f32> {
        let norm = &self.name_norm[i];
        if norm == q {
            return Some(1.0);
        }
        if self.name_nospace[i] == q_nospace {
            return Some(0.95);
        }
        if norm.starts_with(q) || self.name_nospace[i].starts_with(q_nospace) {
            return Some(0.9);
        }
        let toks = &self.name_tokens[i];
        if !q_tokens.is_empty() && q_tokens.iter().all(|qt| toks.iter().any(|nt| nt.starts_with(qt))) {
            return Some(0.85);
        }
        let max_edits = match q.chars().count() {
            0..=2 => 0,
            3..=4 => 1,
            _ => 2,
        };
        if max_edits == 0 {
            return None;
        }
        // Fuzzy re-ranking: every query token of 3+ chars must be close to one name token (or, for a one-word
        // query, the name's no-space prefix). Per-token distances are far cheaper than joined windows.
        let mut required = 0usize;
        let mut sim_sum = 0.0f32;
        for qt in q_tokens.iter().filter(|t| t.chars().count() >= 3) {
            let qc: Vec<char> = qt.chars().collect();
            let max_edits = if qc.len() >= 5 { 2 } else { 1 };
            let first = qc[0];
            let best = toks
                .iter()
                .filter(|nt| {
                    let nlen = nt.chars().count();
                    nlen.abs_diff(qc.len()) <= max_edits && nt.starts_with(first)
                })
                .filter_map(|nt| {
                    let nc: Vec<char> = nt.chars().collect();
                    let dist = damerau(&qc, &nc);
                    (dist <= max_edits).then_some(1.0 - dist as f32 / qc.len().max(nc.len()) as f32)
                })
                .fold(None::<f32>, |best, sim| Some(best.map_or(sim, |b: f32| b.max(sim))));
            let Some(sim) = best else { return None };
            required += 1;
            sim_sum += sim;
        }
        let mut best = if required > 0 {
            Some(0.4 + 0.4 * (sim_sum / required as f32))
        } else {
            None
        };
        // A one-word query typed without the hyphens/space ("snakeye" for "Snake-Eye").
        let qn_len = self.name_nospace[i].chars().count();
        if q_tokens.len() == 1 && qn_len >= 3 {
            let qn: Vec<char> = q_nospace.chars().collect();
            let nm: Vec<char> = self.name_nospace[i].chars().collect();
            if nm.len() >= qn.len() && nm[0] == qn[0] {
                let dist = damerau(&qn, &nm[..qn.len()]);
                let max_edits = if qn.len() >= 5 { 2 } else { 1 };
                if dist <= max_edits {
                    let sim = 1.0 - dist as f32 / qn.len().max(nm.len()) as f32;
                    let score = 0.4 + 0.4 * sim;
                    best = Some(best.map_or(score, |b: f32| b.max(score)));
                }
            }
        }
        best
    }

    fn archetype_score(&self, q: &str, q_nospace: &str, q_tokens: &[String]) -> Vec<(usize, f32)> {
        let mut out = Vec::new();
        for a in &self.archetypes {
            let score = if a.norm == q || a.nospace == q_nospace {
                Some(0.49)
            } else if a.norm.starts_with(q) || a.nospace.starts_with(q_nospace) {
                Some(0.46)
            } else if !q_tokens.is_empty()
                && q_tokens.iter().all(|qt| a.norm.split(' ').any(|at| at.starts_with(qt)))
            {
                Some(0.44)
            } else {
                None
            };
            if let Some(s) = score {
                for &m in &a.members {
                    out.push((m, s));
                }
            }
        }
        out
    }

    fn bm25(&self, q_tokens: &[String], phrases: &[String]) -> Vec<(usize, f64)> {
        if q_tokens.is_empty() && phrases.is_empty() {
            return Vec::new();
        }
        let mut scores: HashMap<usize, f64> = HashMap::new();
        let n = self.docs.len() as f64;
        for t in q_tokens {
            let Some(post) = self.postings.get(t) else { continue };
            let df = post.len() as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for &(doc, tf) in post {
                let dl = self.docs[doc as usize].len as f64;
                let denom = tf as f64 + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl.max(1e-9));
                let s = idf * (tf as f64 * (BM25_K1 + 1.0)) / denom;
                *scores.entry(doc as usize).or_insert(0.0) += s;
            }
        }
        let mut out: Vec<(usize, f64)> = scores
            .into_iter()
            .filter(|(i, _)| {
                phrases.iter().all(|p| self.docs[*i].norm.contains(p.as_str()))
            })
            .collect();
        if q_tokens.is_empty() {
            // Phrase-only query: every document that contains every phrase scores the same.
            out = self
                .docs
                .iter()
                .enumerate()
                .filter(|(_, d)| phrases.iter().all(|p| d.norm.contains(p.as_str())))
                .map(|(i, _)| (i, 1.0))
                .collect();
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| self.rows[a.0].name.cmp(&self.rows[b.0].name)));
        out
    }

    fn snippet(&self, i: usize, q_tokens: &[String], phrases: &[String]) -> String {
        let text = &self.rows[i].text;
        if text.is_empty() {
            return String::new();
        }
        let sentence_end = |c: char| matches!(c, '.' | '!' | '?');
        let mut sentences = Vec::new();
        let mut start = 0usize;
        for (pos, c) in text.char_indices() {
            if sentence_end(c) {
                sentences.push(&text[start..pos + c.len_utf8()]);
                start = pos + c.len_utf8();
            }
        }
        if start < text.len() {
            sentences.push(&text[start..]);
        }
        let picked = if phrases.is_empty() {
            sentences
                .iter()
                .find(|s| q_tokens.iter().any(|t| normalize(s).contains(t)))
        } else {
            sentences.iter().find(|s| {
                let norm = normalize(s);
                phrases.iter().all(|p| norm.contains(p.as_str()))
            })
        };
        let picked = picked.copied().unwrap_or(text);
        let trimmed = picked.trim();
        let mut out: String = trimmed.chars().take(160).collect();
        if out.chars().count() < trimmed.chars().count() {
            out.push('…');
        }
        out
    }

    fn filter_rows(&self, filters: &Value) -> Result<Vec<usize>, String> {
        let Some(obj) = filters.as_object() else {
            if filters.is_null() {
                return Ok((0..self.rows.len()).collect());
            }
            return Err("filters must be an object".to_string());
        };
        let mut kinds: Option<BTreeSet<String>> = None;
        let mut attributes: Option<u32> = None;
        let mut races: Option<u64> = None;
        let mut level: Option<(i64, i64)> = None;
        let mut frames: Option<u32> = None;
        let mut atk: Option<(i64, i64)> = None;
        let mut def: Option<(i64, i64)> = None;
        let mut st_types: Option<BTreeSet<String>> = None;
        let mut archetype: Option<u16> = None;
        let mut extra_only = false;

        for (key, value) in obj {
            match key.as_str() {
                "kind" => {
                    let list = string_list(value, "kind")?;
                    let mut set = BTreeSet::new();
                    for k in list {
                        let k = normalize(&k);
                        if !matches!(k.as_str(), "monster" | "spell" | "trap") {
                            return Err(format!("unknown kind {k:?} (monster|spell|trap)"));
                        }
                        set.insert(k);
                    }
                    kinds = Some(set);
                }
                "attribute" => {
                    let mut bits = 0u32;
                    for name in string_list(value, "attribute")? {
                        bits |= attribute_bit(&name)
                            .ok_or_else(|| format!("unknown attribute {name:?}"))?;
                    }
                    attributes = Some(bits);
                }
                "race" => {
                    let mut bits = 0u64;
                    for name in string_list(value, "race")? {
                        bits |= race_bit(&name).ok_or_else(|| format!("unknown race {name:?}"))?;
                    }
                    races = Some(bits);
                }
                "level" => {
                    let arr = value.as_array().ok_or("level must be [min, max]")?;
                    if arr.len() != 2 {
                        return Err("level must be [min, max]".to_string());
                    }
                    let lo = arr[0].as_i64().ok_or("level must be [min, max]")?;
                    let hi = arr[1].as_i64().ok_or("level must be [min, max]")?;
                    level = Some((lo.min(hi), lo.max(hi)));
                }
                "frame" => {
                    let mut bits = 0u32;
                    for name in string_list(value, "frame")? {
                        bits |= frame_bit(&name).ok_or_else(|| format!("unknown frame {name:?}"))?;
                    }
                    frames = Some(bits);
                }
                "atk" => atk = Some(range_filter(value, "atk")?),
                "def" => def = Some(range_filter(value, "def")?),
                "st_type" => {
                    let mut set = BTreeSet::new();
                    for name in string_list(value, "st_type")? {
                        let k = normalize(&name);
                        if !matches!(
                            k.as_str(),
                            "normal" | "quickplay" | "continuous" | "equip" | "field" | "ritual" | "counter"
                        ) {
                            return Err(format!("unknown st_type {name:?}"));
                        }
                        set.insert(k);
                    }
                    st_types = Some(set);
                }
                "archetype" => {
                    let code = match value {
                        Value::Number(n) => n.as_u64(),
                        Value::String(s) => {
                            let s = s.trim();
                            if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                                u32::from_str_radix(hex, 16).ok().map(u64::from)
                            } else {
                                s.parse::<u64>().ok()
                            }
                        }
                        _ => None,
                    }
                    .ok_or("archetype must be a setcode number")?;
                    archetype = Some(code as u16);
                }
                "extra_only" => {
                    extra_only = value.as_bool().ok_or("extra_only must be true|false")?;
                }
                other => return Err(format!("unknown filter {other:?}")),
            }
        }

        Ok((0..self.rows.len())
            .filter(|&i| {
                let row = &self.rows[i];
                if let Some(set) = &kinds
                    && !set.contains(row.kind)
                {
                    return false;
                }
                if let Some(bits) = attributes
                    && row.attribute & bits == 0
                {
                    return false;
                }
                if let Some(bits) = races
                    && row.race & bits == 0
                {
                    return false;
                }
                if let Some((lo, hi)) = level {
                    let value = i64::from(if row.link != 0 { row.link } else { row.level });
                    if value < lo || value > hi {
                        return false;
                    }
                }
                if let Some(bits) = frames
                    && row.type_ & bits == 0
                {
                    return false;
                }
                if let Some((lo, hi)) = atk {
                    // Monsters only; `?` ATK is negative (-2 in the cdb) and never matches.
                    if row.kind != "monster" || row.attack < 0
                        || i64::from(row.attack) < lo || i64::from(row.attack) > hi
                    {
                        return false;
                    }
                }
                if let Some((lo, hi)) = def {
                    // Monsters only; Links have no DEF and `?` DEF is negative (-2).
                    if row.kind != "monster" || row.link != 0 || row.defense < 0
                        || i64::from(row.defense) < lo || i64::from(row.defense) > hi
                    {
                        return false;
                    }
                }
                if let Some(set) = &st_types
                    && !st_type_matches(row.type_, set)
                {
                    return false;
                }
                if let Some(code) = archetype
                    && !setcode_matches(&row.setcodes, code)
                {
                    return false;
                }
                if extra_only && !row.extra {
                    return false;
                }
                true
            })
            .collect())
    }

    pub fn search(&self, q: &str, filters: &Value, offset: usize, limit: usize) -> Result<SearchResult, String> {
        let allowed = self.filter_rows(filters)?;
        let allowed_set: BTreeSet<usize> = allowed.iter().copied().collect();
        let query = q.trim();
        let q_norm = normalize(query);
        let q_nospace = nospace(&q_norm);
        if q_norm.is_empty() {
            let mut rows: Vec<SearchRow> = allowed
                .iter()
                .map(|&i| SearchRow {
                    code: self.rows[i].code,
                    name: self.rows[i].name.clone(),
                    kind: self.rows[i].kind,
                    line: self.rows[i].line.clone(),
                    extra: self.rows[i].extra,
                    by: "name",
                    snippet: String::new(),
                })
                .collect();
            rows.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.code.cmp(&b.code)));
            return Ok(paginate(rows, offset, limit));
        }

        // Name list: direct hits plus archetype members (just below direct hits).
        let q_tokens = name_tokens(query);
        let mut name_scores: HashMap<usize, f32> = HashMap::new();
        for i in self.candidate_rows(&q_norm, &q_nospace) {
            if !allowed_set.contains(&i) {
                continue;
            }
            if let Some(s) = self.name_score(&q_norm, &q_nospace, &q_tokens, i) {
                name_scores.insert(i, s);
            }
        }
        for (i, s) in self.archetype_score(&q_norm, &q_nospace, &q_tokens) {
            if allowed_set.contains(&i) {
                name_scores.entry(i).and_modify(|v| *v = v.max(s)).or_insert(s);
            }
        }
        let mut name_list: Vec<(usize, f32)> = name_scores.into_iter().collect();
        name_list.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| self.rows[a.0].name.cmp(&self.rows[b.0].name)));

        // Text list: BM25 over effect text (with required quoted phrases).
        let (phrase_free, phrases) = split_phrases(query);
        let text_tokens = tokens(&phrase_free);
        let text_list = self.bm25(&text_tokens, &phrases);

        let mut fused: HashMap<usize, f64> = HashMap::new();
        let mut by_name: HashMap<usize, &'static str> = HashMap::new();
        for (rank, &(i, _)) in name_list.iter().enumerate() {
            *fused.entry(i).or_insert(0.0) += NAME_WEIGHT / (RRF_K + rank as f64 + 1.0);
            let direct = self.name_score(&q_norm, &q_nospace, &q_tokens, i).is_some();
            by_name.insert(i, if direct { "name" } else { "archetype" });
        }
        let mut text_hits: BTreeSet<usize> = BTreeSet::new();
        for (rank, &(i, _)) in text_list.iter().enumerate() {
            if !allowed_set.contains(&i) {
                continue;
            }
            *fused.entry(i).or_insert(0.0) += TEXT_WEIGHT / (RRF_K + rank as f64 + 1.0);
            text_hits.insert(i);
        }

        let mut indices: Vec<usize> = fused.keys().copied().collect();
        indices.sort_by(|&a, &b| {
            fused[&b]
                .total_cmp(&fused[&a])
                .then_with(|| self.rows[a].name.cmp(&self.rows[b].name))
        });
        let total = indices.len();
        // Snippets are only built for the returned page: a broad query can fuse thousands of rows.
        let rows: Vec<SearchRow> = indices
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|i| {
                let (by, snippet) = match by_name.get(&i) {
                    Some(by) => (*by, String::new()),
                    None => ("text", self.snippet(i, &text_tokens, &phrases)),
                };
                SearchRow {
                    code: self.rows[i].code,
                    name: self.rows[i].name.clone(),
                    kind: self.rows[i].kind,
                    line: self.rows[i].line.clone(),
                    extra: self.rows[i].extra,
                    by,
                    snippet,
                }
            })
            .collect();
        Ok(SearchResult { total, rows })
    }

    /// "Describe a card" search: encode `q`'s clauses, score every card by its best-matching effect (mean
    /// over clauses), then fuse that ranking with BM25 by reciprocal rank fusion (k = 60).
    ///
    /// `encode` receives one clause and must apply the model's query instruction itself. The caller runs this
    /// off the message thread; encoding is the slow part.
    pub fn search_describe(
        &self,
        q: &str,
        filters: &Value,
        offset: usize,
        limit: usize,
        emb: &Emb,
        mut encode: impl FnMut(&str) -> Result<Vec<f32>, String>,
    ) -> Result<SearchResult, String> {
        let query = q.trim();
        if query.is_empty() {
            return self.search("", filters, offset, limit);
        }
        let allowed = self.filter_rows(filters)?;
        let allowed_set: BTreeSet<usize> = allowed.iter().copied().collect();
        let mut clauses = clause_split(query);
        if clauses.len() < 2 {
            clauses = vec![query.to_string()];
        }
        let mut qvecs = Vec::with_capacity(clauses.len());
        for clause in &clauses {
            let v = encode(clause)?;
            if v.is_empty() || !v.iter().all(|x| x.is_finite()) {
                return Err("the query encoder returned an invalid vector".into());
            }
            qvecs.push(v);
        }
        let dim = qvecs[0].len();
        if qvecs.iter().any(|v| v.len() != dim) || dim != emb.dim {
            return Err(format!("query dimension {dim} != card embeddings {}", emb.dim));
        }

        let mut score = vec![0f32; self.rows.len()];
        let mut semantic = Vec::new();
        let mut first_effect: Vec<Option<(usize, f32)>> = vec![None; self.rows.len()];
        for &i in &allowed {
            let code = self.rows[i].code as i64;
            let Ok(e) = emb.codes.binary_search(&code) else { continue };
            let slots = (emb.n_effects[e].max(1) as usize).min(emb.n_effects_slots);
            let mut total = 0f32;
            let mut best_first = (0usize, f32::NEG_INFINITY);
            for (ci, qv) in qvecs.iter().enumerate() {
                let mut best = f32::NEG_INFINITY;
                let mut best_k = 0usize;
                for k in 0..slots {
                    let base = (e * emb.n_effects_slots + k) * emb.dim;
                    let row = &emb.effects[base..base + emb.dim];
                    let dot: f32 = row.iter().zip(qv).map(|(a, b)| a * b).sum();
                    if dot > best {
                        best = dot;
                        best_k = k;
                    }
                }
                if ci == 0 {
                    best_first = (best_k, best);
                }
                total += best;
            }
            score[i] = total / qvecs.len() as f32;
            first_effect[i] = Some(best_first);
            semantic.push(i);
        }
        semantic.sort_by(|&a, &b| {
            score[b].total_cmp(&score[a]).then_with(|| self.rows[a].name.cmp(&self.rows[b].name))
        });
        semantic.retain(|&i| score[i].is_finite());

        // BM25 over the effect text is the other ranker; quoted phrases stay required.
        let (phrase_free, phrases) = split_phrases(query);
        let text_tokens = tokens(&phrase_free);
        let text_list: Vec<usize> = self
            .bm25(&text_tokens, &phrases)
            .into_iter()
            .map(|(i, _)| i)
            .filter(|i| allowed_set.contains(i))
            .collect();

        let fused = rrf_fuse(&[&semantic, &text_list], RRF_K);
        let mut indices: Vec<usize> = fused.keys().copied().collect();
        indices.sort_by(|&a, &b| {
            fused[&b].total_cmp(&fused[&a]).then_with(|| self.rows[a].name.cmp(&self.rows[b].name))
        });
        let total = indices.len();
        let rows: Vec<SearchRow> = indices
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|i| {
                let snippet = match first_effect[i] {
                    Some((k, _)) => {
                        let parts = effect_sentences(&self.rows[i].text);
                        parts
                            .get(k)
                            .map(|s| truncate_snippet(s))
                            .unwrap_or_else(|| self.snippet(i, &text_tokens, &phrases))
                    }
                    None => self.snippet(i, &text_tokens, &phrases),
                };
                SearchRow {
                    code: self.rows[i].code,
                    name: self.rows[i].name.clone(),
                    kind: self.rows[i].kind,
                    line: self.rows[i].line.clone(),
                    extra: self.rows[i].extra,
                    by: "describe",
                    snippet,
                }
            })
            .collect();
        Ok(SearchResult { total, rows })
    }
}

fn truncate_snippet(text: &str) -> String {
    let trimmed = text.trim();
    let mut out: String = trimmed.chars().take(160).collect();
    if out.chars().count() < trimmed.chars().count() {
        out.push('…');
    }
    out
}

/// `re.split(r"\band\b|,", q)` (case-sensitive, as in `semantic_eval.py`), keeping trimmed parts longer
/// than 3 characters.
pub fn clause_split(q: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let bytes = q.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b',' {
            parts.push(&q[start..i]);
            start = i + 1;
            i += 1;
            continue;
        }
        if bytes[i..].starts_with(b"and") {
            let before_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let after_ok = i + 3 >= bytes.len() || !is_word_byte(bytes[i + 3]);
            if before_ok && after_ok {
                parts.push(&q[start..i]);
                start = i + 3;
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    parts.push(&q[start..]);
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| p.chars().count() > 3)
        .map(str::to_string)
        .collect()
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// `semantic_eval.split_effects` / `decklight.cards.embed.split_effects`: card text sections, then
/// sentences split after `.` followed by whitespace and an uppercase letter, quote or `(`; `●` bullet
/// lines join the sentence before them; parts of 3 characters or fewer are dropped.
pub fn effect_sentences(desc: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for section in split_sections(desc) {
        let section = section.trim();
        if section.is_empty() {
            continue;
        }
        let mut buf: Vec<String> = Vec::new();
        for line in section.split('\n') {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('●') {
                if let Some(last) = buf.last_mut() {
                    last.push(' ');
                    last.push('●');
                    last.push_str(rest);
                    continue;
                }
            }
            for s in split_sentences(line) {
                if !s.is_empty() {
                    buf.push(s);
                }
            }
        }
        parts.extend(buf);
    }
    parts.into_iter().filter(|p| p.chars().count() > 3).collect()
}

fn split_sections(desc: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = desc.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let mut end = None;
        if bytes[i] == b'[' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            let rest = &desc[j..];
            for marker in ["Pendulum Effect", "Monster Effect"] {
                if rest.starts_with(marker) {
                    let mut k = j + marker.len();
                    while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                        k += 1;
                    }
                    if k < bytes.len() && bytes[k] == b']' {
                        end = Some(k + 1);
                        break;
                    }
                }
            }
        }
        if end.is_none() && bytes[i] == b'-' {
            let mut j = i;
            while j < bytes.len() && bytes[j] == b'-' {
                j += 1;
            }
            if j - i >= 5 {
                end = Some(j);
            }
        }
        if let Some(e) = end {
            out.push(&desc[start..i]);
            start = e;
            i = e;
        } else {
            i += 1;
        }
    }
    out.push(&desc[start..]);
    out
}

fn split_sentences(line: &str) -> Vec<String> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while idx < chars.len() {
        let (pos, c) = chars[idx];
        if c == '.' && idx + 1 < chars.len() {
            let mut j = idx + 1;
            let mut saw_space = false;
            while j < chars.len() && chars[j].1.is_whitespace() {
                saw_space = true;
                j += 1;
            }
            if saw_space && j < chars.len() {
                let next = chars[j].1;
                if next.is_ascii_uppercase() || next == '"' || next == '(' {
                    let end = pos + c.len_utf8();
                    out.push(line[start..end].to_string());
                    start = chars[j].0;
                    idx = j;
                    continue;
                }
            }
        }
        idx += 1;
    }
    if start < line.len() {
        out.push(line[start..].to_string());
    }
    out
}

/// Reciprocal rank fusion: `sum(1 / (k + 1 + rank))` over every ranking that contains a row.
fn rrf_fuse(lists: &[&[usize]], k: f64) -> HashMap<usize, f64> {
    let mut out: HashMap<usize, f64> = HashMap::new();
    for list in lists {
        for (rank, &i) in list.iter().enumerate() {
            *out.entry(i).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }
    }
    out
}

fn paginate(rows: Vec<SearchRow>, offset: usize, limit: usize) -> SearchResult {
    let total = rows.len();
    let rows = rows.into_iter().skip(offset).take(limit).collect();
    SearchResult { total, rows }
}

fn string_list(value: &Value, key: &str) -> Result<Vec<String>, String> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(a) => a
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| format!("{key} must be a string or an array of strings")))
            .collect(),
        _ => Err(format!("{key} must be a string or an array of strings")),
    }
}

/// Quoted phrases are required as normalised substrings; they are removed from the plain token query.
fn split_phrases(q: &str) -> (String, Vec<String>) {
    let mut plain = String::new();
    let mut phrases = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for c in q.chars() {
        if c == '"' {
            if quoted {
                let p = normalize(&current);
                if !p.is_empty() {
                    phrases.push(p);
                }
                current.clear();
            }
            quoted = !quoted;
        } else if quoted {
            current.push(c);
        } else {
            plain.push(c);
        }
    }
    if quoted {
        plain.push_str(&current);
    }
    (plain, phrases)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> Option<SearchIndex> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let cdb = root.join("third_party/BabelCDB");
        let strings = root.join("third_party/edopro-strings.conf");
        if !cdb.join("cards.cdb").is_file() {
            eprintln!("skip cards_search: no third_party/BabelCDB/cards.cdb");
            return None;
        }
        let cards = decklight_cards::load_cards(&decklight_cards::default_cdbs(&cdb, &[])).unwrap();
        let texts = Texts::load(&cdb, &[], &strings).unwrap();
        let t = std::time::Instant::now();
        let index = SearchIndex::build(&cards, &texts, &strings);
        eprintln!(
            "cards_search index: {} rows, {:.1} MB, build {} ms",
            index.len(),
            index.size_bytes as f64 / (1024.0 * 1024.0),
            t.elapsed().as_millis()
        );
        Some(index)
    }

    fn names_contain(result: &SearchResult, needle: &str, top: usize) -> bool {
        result.rows.iter().take(top).any(|r| r.name.to_lowercase().contains(needle))
    }

    #[test]
    fn cards_search_ash_blosom_is_first() {
        let Some(index) = fixture() else { return };
        let r = index.search("ash blosom", &Value::Null, 0, 10).unwrap();
        assert_eq!(r.rows.first().map(|x| x.name.as_str()), Some("Ash Blossom & Joyous Spring"));
    }

    #[test]
    fn cards_search_snakeye_finds_snake_eye() {
        let Some(index) = fixture() else { return };
        let r = index.search("snakeye", &Value::Null, 0, 10).unwrap();
        assert!(names_contain(&r, "snake-eye", 1), "top: {:?}", r.rows.first().map(|x| &x.name));
    }

    #[test]
    fn cards_search_sp_little_knight_is_first() {
        let Some(index) = fixture() else { return };
        let r = index.search("S:P little", &Value::Null, 0, 10).unwrap();
        assert_eq!(r.rows.first().map(|x| x.name.as_str()), Some("S:P Little Knight"));
    }

    #[test]
    fn cards_search_accesscode_in_top_three() {
        let Some(index) = fixture() else { return };
        let r = index.search("accesscode", &Value::Null, 0, 10).unwrap();
        assert!(names_contain(&r, "accesscode", 3), "rows: {:?}", r.rows.iter().map(|x| &x.name).collect::<Vec<_>>());
    }

    #[test]
    fn cards_search_quoted_phrase_is_required() {
        let Some(index) = fixture() else { return };
        let r = index.search("\"banish face-down\"", &Value::Null, 0, 50).unwrap();
        assert!(!r.rows.is_empty());
        for row in &r.rows {
            let text = normalize(index.card_text(row.code).unwrap());
            assert!(text.contains("banish face down"), "{}: {text}", row.name);
        }
    }

    #[test]
    fn cards_search_trap_filter_only_traps() {
        let Some(index) = fixture() else { return };
        let filters = serde_json::json!({"kind": "trap"});
        let r = index.search("negate", &filters, 0, 20).unwrap();
        assert!(!r.rows.is_empty());
        assert!(r.rows.iter().all(|x| x.kind == "trap"));
    }

    #[test]
    fn cards_search_empty_query_sorted_by_name() {
        let Some(index) = fixture() else { return };
        let filters = serde_json::json!({"kind": ["spell"]});
        let r = index.search("", &filters, 0, 20).unwrap();
        assert!(!r.rows.is_empty());
        assert!(r.rows.iter().all(|x| x.kind == "spell"));
        let names: Vec<&String> = r.rows.iter().map(|x| &x.name).collect();
        let sorted = {
            let mut n = names.clone();
            n.sort();
            n
        };
        assert_eq!(names, sorted);
    }

    #[test]
    fn cards_search_nonsense_returns_nothing() {
        let Some(index) = fixture() else { return };
        let r = index.search("zzzxqwozz", &Value::Null, 0, 20).unwrap();
        assert_eq!(r.total, 0);
        assert!(r.rows.is_empty());
    }

    #[test]
    fn cards_search_unknown_filter_is_an_error() {
        let Some(index) = fixture() else { return };
        let err = index.search("ash", &serde_json::json!({"colour": "red"}), 0, 10).unwrap_err();
        assert!(err.contains("colour"), "{err}");
    }

    #[test]
    fn clause_split_matches_semantic_eval() {
        assert_eq!(
            clause_split("a light tuner that special summons itself and searches, level 4"),
            vec!["a light tuner that special summons itself", "searches", "level 4"]
        );
        // "and" only at word boundaries, and parts of 3 characters or fewer are dropped.
        assert_eq!(
            clause_split("a card that searches sand and summons a"),
            vec!["a card that searches sand", "summons a"]
        );
    }

    #[test]
    fn effect_sentences_follow_the_reference_splitter() {
        let desc = "[ Pendulum Effect ]\nDuring your Main Phase: You can add 1 card. You can only use this once.\n\
                    ----------------------------------------\n[ Monster Effect ]\nYou can target 1 card; destroy it.\n\
                    ● Bullet line joins the sentence before it.";
        assert_eq!(
            effect_sentences(desc),
            vec![
                "During your Main Phase: You can add 1 card.",
                "You can only use this once.",
                "You can target 1 card; destroy it. ● Bullet line joins the sentence before it.",
            ]
        );
    }

    #[test]
    fn cards_search_atk_range_excludes_unknown_and_non_monsters() {
        let Some(index) = fixture() else { return };
        let allowed = index.filter_rows(&serde_json::json!({"atk": [0, 0]})).unwrap();
        assert!(!allowed.is_empty());
        for &i in &allowed {
            let row = &index.rows[i];
            assert_eq!(row.kind, "monster");
            assert_eq!(row.attack, 0, "{} has ATK {}", row.name, row.attack);
        }
        assert!(index.rows.iter().any(|r| r.kind == "monster" && r.attack < 0), "fixture has no ? ATK card");
        assert!(!allowed.iter().any(|&i| index.rows[i].attack < 0));
    }

    #[test]
    fn cards_search_def_range_excludes_links() {
        let Some(index) = fixture() else { return };
        assert!(index.rows.iter().any(|r| r.link != 0), "fixture has no Link monster");
        let allowed = index.filter_rows(&serde_json::json!({"def": [0, 10000]})).unwrap();
        assert!(!allowed.is_empty());
        for &i in &allowed {
            let row = &index.rows[i];
            assert_eq!(row.kind, "monster");
            assert_eq!(row.link, 0, "{} is a Link and has no DEF", row.name);
            assert!(row.defense >= 0, "{} has unknown DEF", row.name);
        }
    }

    #[test]
    fn cards_search_st_type_quickplay_normal_counter() {
        let Some(index) = fixture() else { return };
        let quickplay = index.filter_rows(&serde_json::json!({"st_type": "quickplay"})).unwrap();
        assert!(!quickplay.is_empty());
        for &i in &quickplay {
            let row = &index.rows[i];
            assert!(row.type_ & C::TYPE_SPELL != 0 && row.type_ & C::TYPE_QUICKPLAY != 0, "{}", row.name);
        }
        let normal = index.filter_rows(&serde_json::json!({"st_type": ["normal"]})).unwrap();
        assert!(!normal.is_empty());
        let subtype_bits = C::TYPE_QUICKPLAY
            | C::TYPE_CONTINUOUS
            | C::TYPE_EQUIP
            | C::TYPE_FIELD
            | C::TYPE_RITUAL
            | C::TYPE_COUNTER;
        for &i in &normal {
            let row = &index.rows[i];
            assert!(row.type_ & (C::TYPE_SPELL | C::TYPE_TRAP) != 0, "{}", row.name);
            assert_eq!(row.type_ & subtype_bits, 0, "{}", row.name);
        }
        let counter = index.filter_rows(&serde_json::json!({"st_type": ["counter"]})).unwrap();
        assert!(!counter.is_empty());
        for &i in &counter {
            let row = &index.rows[i];
            assert!(row.type_ & C::TYPE_TRAP != 0 && row.type_ & C::TYPE_COUNTER != 0, "{}", row.name);
        }
    }

    #[test]
    fn cards_search_st_type_combined_with_kind() {
        let Some(index) = fixture() else { return };
        let allowed = index
            .filter_rows(&serde_json::json!({"kind": "spell", "st_type": ["quickplay", "normal"]}))
            .unwrap();
        assert!(!allowed.is_empty());
        for &i in &allowed {
            let row = &index.rows[i];
            assert_eq!(row.kind, "spell", "{}", row.name);
            assert!(
                row.type_ & (C::TYPE_QUICKPLAY | C::TYPE_CONTINUOUS | C::TYPE_EQUIP | C::TYPE_FIELD | C::TYPE_RITUAL)
                    == row.type_ & C::TYPE_QUICKPLAY
                    || row.type_ & (C::TYPE_QUICKPLAY | C::TYPE_CONTINUOUS | C::TYPE_EQUIP | C::TYPE_FIELD | C::TYPE_RITUAL | C::TYPE_COUNTER)
                        == 0,
                "{}",
                row.name
            );
        }
    }

    #[test]
    fn cards_search_bad_filter_shapes_are_errors() {
        let Some(index) = fixture() else { return };
        for filters in [
            serde_json::json!({"atk": "high"}),
            serde_json::json!({"atk": [1]}),
            serde_json::json!({"atk": [10, 1]}),
            serde_json::json!({"def": [null, 5]}),
            serde_json::json!({"st_type": "spell"}),
            serde_json::json!({"st_type": ["banana"]}),
        ] {
            let err = index.filter_rows(&filters).unwrap_err();
            assert!(!err.is_empty(), "{filters} did not error");
        }
    }

    #[test]
    fn rrf_fusion_sums_each_rankings_score() {
        let fused = rrf_fuse(&[&[1, 2, 3], &[2, 1]], 60.0);
        assert!((fused[&1] - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
        assert!((fused[&2] - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-12);
        assert!((fused[&3] - 1.0 / 63.0).abs() < 1e-12);
    }

    #[test]
    fn cards_search_card_json_has_fields() {
        let Some(index) = fixture() else { return };
        let v = index.card_json(14558127).unwrap();
        assert_eq!(v["name"], "Ash Blossom & Joyous Spring");
        assert_eq!(v["kind"], "monster");
        assert!(v["line"].as_str().unwrap().contains("Level"));
    }

    #[test]
    #[ignore = "micro-benchmark: needs the real cards.cdb and embedding npz"]
    fn cards_search_describe_timing() {
        let Some(index) = fixture() else { return };
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let emb_path = std::env::var_os("YGO_EMB")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("data/cards/emb/qwen3-0.6b-d512.npz"));
        let emb = match decklight_cards::emb::Emb::load(&emb_path) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip describe timing: {e}");
                return;
            }
        };
        let q = "a light tuner that special summons itself and searches";
        let t = std::time::Instant::now();
        let r = index
            .search_describe(q, &Value::Null, 0, 20, &emb, |_clause| {
                let d = emb.dim;
                Ok(vec![1.0 / (d as f32).sqrt(); d])
            })
            .unwrap();
        eprintln!(
            "describe timing: {q:?} -> {} rows in {:.1} ms; top {:?}",
            r.total,
            t.elapsed().as_secs_f64() * 1000.0,
            r.rows.iter().take(3).map(|x| &x.name).collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "micro-benchmark: needs the real cards.cdb"]
    fn cards_search_micro_benchmark() {
        let Some(index) = fixture() else { return };
        let queries = ["ash blosom", "snakeye", "S:P little", "accesscode", "negate", "banish", "dragon", "spellcaster"];
        let mut times = Vec::new();
        for _ in 0..5 {
            for q in queries {
                let t = std::time::Instant::now();
                let _ = index.search(q, &Value::Null, 0, 40).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if times.len() < queries.len() {
                    eprintln!("  {q:?}: {ms:.2}ms");
                }
                times.push(ms);
            }
        }
        times.sort_by(f64::total_cmp);
        let p50 = times[times.len() / 2];
        let p95 = times[(times.len() * 95) / 100];
        eprintln!(
            "cards_search benchmark: rows={} index_mb={:.1} p50={p50:.2}ms p95={p95:.2}ms",
            index.len(),
            index.size_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}
