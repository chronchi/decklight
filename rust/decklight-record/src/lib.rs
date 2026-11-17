//! Match record v1 (ADR 0010): exactly the document `decklight.play.match_record.from_duel` writes.
//!
//! Shared by `decklight-server` (the web server) and `decklight-edopro` (the LAN/EDOPro host) so both write the
//! same on-disk format; moving this module out of the server changed no byte of its output.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const FORMAT: &str = "decklight.match/1";

pub fn seed_words(seed: u64) -> [u64; 4] {
    let mut x = seed;
    let mut out = [0u64; 4];
    for w in out.iter_mut() {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        *w = z ^ (z >> 31);
    }
    out
}

fn random_bytes(n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    if let Ok(mut f) = fs::File::open("/dev/urandom") {
        use std::io::Read;
        if f.read_exact(&mut out).is_ok() {
            return out;
        }
        drop(f);
    }
    // Fallback: derive from time (never used for secrets, only ids).
    let mut x = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
    for b in out.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = x as u8;
    }
    out
}

pub fn random_hex(n: usize) -> String {
    random_bytes(n).iter().map(|b| format!("{b:02x}")).collect()
}

pub fn uuid7() -> String {
    let ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
    uuid7_at(ms)
}

/// RFC 9562 UUIDv7 with an explicit 48-bit timestamp in milliseconds (replay import uses the replay time).
pub fn uuid7_at(ms: u64) -> String {
    let r = random_bytes(10);
    let mut b = [0u8; 16];
    b[..6].copy_from_slice(&ms.to_be_bytes()[2..]);
    b[6..].copy_from_slice(&r);
    b[6] = 0x70 | (b[6] & 0x0F);
    b[8] = 0x80 | (b[8] & 0x3F);
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..])
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> String {
    fs::read(path).map(|b| sha256_hex(&b)).unwrap_or_default()
}

/// `match_record._git_head`: the commit, or "unknown".
pub fn git_head(path: &Path) -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

pub fn dev_pack(root: &Path) -> Value {
    let tp = root.join("third_party");
    let scripts = git_head(&tp.join("CardScripts"));
    let cdb = git_head(&tp.join("BabelCDB"));
    let short = |s: &String| s.chars().take(8).collect::<String>();
    json!({
        "pack": format!("dev-{}-{}", short(&scripts), short(&cdb)),
        "sha256": sha256_hex(format!("{scripts}\n{cdb}\n").as_bytes()),
    })
}

pub fn engine_info(root: &Path) -> Value {
    let tp = root.join("third_party");
    let mut patches: Vec<PathBuf> = match fs::read_dir(tp.join("patches")) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n.starts_with("ygopro-core-") && n.ends_with(".patch")
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    patches.sort();
    let mut h = Sha256::new();
    for p in &patches {
        if let Ok(b) = fs::read(p) {
            h.update(b);
        }
    }
    json!({
        "core": git_head(&tp.join("ygopro-core")),
        "patches": h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>(),
    })
}

pub fn deck_hash(main: &[u32], extra: &[u32]) -> String {
    let mut v: Vec<u32> = main.to_vec();
    v.sort_unstable();
    v.push(0);
    let mut e: Vec<u32> = extra.to_vec();
    e.sort_unstable();
    v.extend(e);
    sha256_hex(v.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",").as_bytes())
}

#[derive(Clone, Debug)]
pub struct ResponseRec {
    pub seat: i32,
    pub r: Vec<u8>,
    pub opt: Vec<i32>,
    pub ms: u64,
}

#[derive(Clone, Debug, Default)]
pub struct RecordGame {
    pub seed: u64,
    pub first: Option<i32>,
    pub main0: Vec<u32>,
    pub extra0: Vec<u32>,
    pub main1: Vec<u32>,
    pub extra1: Vec<u32>,
    pub side0: Vec<u32>,
    pub side1: Vec<u32>,
    pub seats: [Value; 2],
    pub mode: String,
    pub responses: Vec<ResponseRec>,
    pub winner: Option<i32>,
    pub win_reason: i32,
    pub turns: i32,
    pub surrendered: bool,
    pub disconnected: bool,
    pub first_player: Option<i32>,
    /// Deck names in seat order (the deck's display name); empty strings are left out of the record.
    pub deck_names: Option<[String; 2]>,
    /// Web practice metadata (Python reference record writer's `meta`): counts and what-if/puzzle tags.
    pub hints_used: Option<u32>,
    pub undos: Option<u32>,
    pub what_if: Option<Value>,
    pub puzzle: Option<Value>,
    pub puzzle_result: Option<String>,
    pub practice: Option<bool>,
}

pub struct RecordEnv {
    pub root: PathBuf,
    pub app: String,
    pub cards: Value,
    pub engine: Value,
    pub install_id: String,
    pub schema_root: PathBuf,
}

impl RecordEnv {
    pub fn new(repo_root: &Path, records_dir: &Path) -> RecordEnv {
        let install_path = records_dir.parent().unwrap_or(Path::new(".")).join("install_id");
        let install_id = match fs::read_to_string(&install_path) {
            Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                let id = random_hex(16);
                if let Some(p) = install_path.parent() {
                    let _ = fs::create_dir_all(p);
                }
                let _ = fs::write(&install_path, format!("{id}\n"));
                id
            }
        };
        RecordEnv {
            root: records_dir.to_path_buf(),
            app: format!("decklight-server/{}", env!("CARGO_PKG_VERSION")),
            cards: dev_pack(repo_root),
            engine: engine_info(repo_root),
            install_id,
            schema_root: repo_root.to_path_buf(),
        }
    }

    pub fn build(&self, g: &RecordGame, id: &str, created: &str) -> Value {
        let state = seed_words(g.seed);
        let mut decks = Vec::new();
        for (i, (main, extra, side)) in [(&g.main0, &g.extra0, &g.side0), (&g.main1, &g.extra1, &g.side1)].into_iter().enumerate() {
            let mut d = json!({
                "main": main,
                "extra": extra,
                "side": side,
                "hash": deck_hash(main, extra),
            });
            if let Some(name) = g.deck_names.as_ref().map(|n| &n[i]).filter(|n| !n.is_empty()) {
                d["name"] = json!(name);
            }
            decks.push(d);
        }
        let responses: Vec<Value> = g
            .responses
            .iter()
            .map(|r| {
                json!({
                    "seat": r.seat,
                    "r": r.r.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                    "opt": r.opt,
                    "ms": r.ms,
                })
            })
            .collect();
        let mut rec = json!({
            "format": FORMAT,
            "id": id,
            "created": created,
            "app": self.app,
            "engine": self.engine,
            "cards": self.cards,
            "rules": {
                "duel_flags": ygoenv_sys::TCG_FLAGS.to_string(),
                "start_lp": 8000,
                "start_hand": 5,
                "draw_count": 1,
            },
            "kind": "full",
            "engine_seed": state.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
            "decks": decks,
            "seats": g.seats,
            "mode": g.mode,
            "responses": responses,
            "result": {
                "winner": g.winner,
                "reason": g.win_reason,
                "turns": g.turns,
                "surrendered": g.surrendered,
                "disconnected": g.disconnected,
            },
            "source": "engine",
        });
        if let Some(first) = g.first.or(g.first_player) {
            rec["first"] = json!(first);
        }
        if let Some(v) = g.hints_used {
            rec["hints_used"] = json!(v);
        }
        if let Some(v) = g.undos {
            rec["undos"] = json!(v);
        }
        if let Some(v) = &g.what_if {
            rec["what_if"] = v.clone();
        }
        if let Some(v) = &g.puzzle {
            rec["puzzle"] = v.clone();
        }
        if let Some(v) = &g.puzzle_result {
            rec["puzzle_result"] = json!(v);
        }
        if let Some(v) = g.practice {
            rec["practice"] = json!(v);
        }
        rec["seed"] = json!(g.seed.to_string());
        let _ = &self.schema_root;
        rec
    }

    pub fn write(&self, g: &RecordGame) -> Result<PathBuf, String> {
        fs::create_dir_all(&self.root).map_err(|e| format!("{}: {e}", self.root.display()))?;
        let id = uuid7();
        let created = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let rec = self.build(g, &id, &created);
        let path = self.root.join(format!("{id}.match.json.gz"));
        let tmp = self.root.join(format!("{id}.match.json.tmp"));
        {
            let f = fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
            let mut enc = GzEncoder::new(f, Compression::default());
            serde_json::to_writer(&mut enc, &rec).map_err(|e| e.to_string())?;
            enc.finish().map_err(|e| e.to_string())?;
        }
        fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        if let Err(e) = self.index(&rec, &path) {
            eprintln!("ALERT record: index update failed: {e}");
        }
        Ok(path)
    }

    /// The index connection with the table created. v0: a table in an older layout is dropped and rebuilt from the
    /// record files next to it (the records are the source of truth; the index is derived).
    fn index_db(&self) -> Result<rusqlite::Connection, String> {
        let con = rusqlite::Connection::open(self.root.join("index.sqlite")).map_err(|e| e.to_string())?;
        let current: bool = con
            .query_row("SELECT count(*) FROM pragma_table_info('matches') WHERE name = 'kind'", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        // The drop, the new table and the rebuild commit together: a kill midway leaves the old table, so the next
        // start rebuilds again instead of keeping a half-filled index (review 2026-11-15).
        if !current {
            con.execute_batch("BEGIN; DROP TABLE IF EXISTS matches").map_err(|e| e.to_string())?;
        }
        con.execute_batch(
            "CREATE TABLE IF NOT EXISTS matches (\
                id TEXT PRIMARY KEY, created TEXT, mode TEXT, \
                deck0_name TEXT, deck0_hash TEXT, deck1_name TEXT, deck1_hash TEXT, \
                opponent TEXT, result INTEGER, turns INTEGER, path TEXT, \
                me INTEGER, first INTEGER, won INTEGER, my_deck TEXT, opp_deck TEXT, opp_type TEXT, kind TEXT)",
        )
        .map_err(|e| e.to_string())?;
        if !current {
            self.reindex(&con);
            con.execute_batch("COMMIT").map_err(|e| e.to_string())?;
        }
        Ok(con)
    }

    /// Bring the index up to the current layout (rebuilt from the record files when older); readers open it
    /// read-only, so the server calls this at start.
    pub fn check_index(&self) {
        if self.root.is_dir()
            && let Err(e) = self.index_db()
        {
            eprintln!("ALERT record: the game index cannot be opened: {e}");
        }
    }

    /// Delete saved games (Play > Games > Select): each file moves to `trash/` next to the records and its row leaves
    /// the index, so the list, the stats and Ask no longer see it. Recover one by moving it back and deleting
    /// `index.sqlite` (rebuilt from the files at the next start). Unknown ids are an error: the list was out of date.
    pub fn trash(&self, ids: &[String]) -> Result<usize, String> {
        let con = self.index_db()?;
        let bin = self.root.join("trash");
        fs::create_dir_all(&bin).map_err(|e| format!("{}: {e}", bin.display()))?;
        let mut n = 0;
        let mut unknown = Vec::new();
        for id in ids {
            let path: Option<String> = con
                .query_row("SELECT path FROM matches WHERE id = ?1", [id], |r| r.get(0))
                .map(Some)
                .or_else(|e| if e == rusqlite::Error::QueryReturnedNoRows { Ok(None) } else { Err(e.to_string()) })?;
            let Some(path) = path.map(PathBuf::from) else {
                unknown.push(id.as_str());
                continue;
            };
            // Only the file of that name in this records folder: the index's absolute path may be stale (a records
            // folder copied or moved with its index) and must never move a file elsewhere (review 2026-11-15).
            let name = path.file_name().ok_or_else(|| format!("{}: not a game file", path.display()))?;
            let path = self.root.join(name);
            match fs::rename(&path, bin.join(name)) {
                Ok(()) => {}
                // Already gone: the row is all that is left of it.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("{}: {e}", path.display())),
            }
            con.execute("DELETE FROM matches WHERE id = ?1", [id]).map_err(|e| e.to_string())?;
            n += 1;
        }
        if !unknown.is_empty() {
            return Err(format!("{n} deleted; {} not found in your games (the list was out of date): {}", unknown.len(), unknown.join(", ")));
        }
        Ok(n)
    }

    /// Index every record file in the records folder (after the table was rebuilt).
    fn reindex(&self, con: &rusqlite::Connection) {
        let files = fs::read_dir(&self.root).into_iter().flatten().flatten().map(|e| e.path());
        let (mut n, mut bad) = (0, 0);
        for p in files.filter(|p| p.to_string_lossy().ends_with(".match.json.gz")) {
            match load_gz(&p).and_then(|rec| self.index_row(con, &rec, &p)) {
                Ok(()) => n += 1,
                Err(e) => {
                    bad += 1;
                    eprintln!("ALERT record: {} not indexed: {e}", p.display());
                }
            }
        }
        if n + bad > 0 {
            eprintln!("record: index rebuilt from {n} saved games ({bad} unreadable)");
        }
    }

    fn index(&self, rec: &Value, path: &Path) -> Result<(), String> {
        let con = self.index_db()?;
        self.index_row(&con, rec, path)
    }

    /// This install's seat: the human seat with our player id, else the only human seat (the EDOPro LAN host writes
    /// `{"type": "human"}` without an id); -1 when neither (bot vs bot, two humans of other installs).
    pub fn my_seat(&self, rec: &Value) -> i64 {
        let seats = rec["seats"].as_array().cloned().unwrap_or_default();
        let human = |s: &Value| s["type"] == "human";
        if let Some(i) = seats.iter().position(|s| human(s) && s["player"] == self.install_id.as_str()) {
            return i as i64;
        }
        let humans: Vec<usize> = (0..seats.len()).filter(|&i| human(&seats[i])).collect();
        match humans[..] {
            [i] if seats[i].get("player").is_none() => i as i64,
            _ => -1,
        }
    }

    fn index_row(&self, con: &rusqlite::Connection, rec: &Value, path: &Path) -> Result<(), String> {
        let decks = rec["decks"].as_array().cloned().unwrap_or_default();
        let name =
            |i: usize| decks.get(i).and_then(|d| d.get("name")).and_then(Value::as_str).unwrap_or("").to_string();
        let hash =
            |i: usize| decks.get(i).and_then(|d| d.get("hash")).and_then(Value::as_str).unwrap_or("").to_string();
        let seats = rec["seats"].as_array().cloned().unwrap_or_default();
        let me = self.my_seat(rec);
        // The other seat when we know ours; seat 1 otherwise.
        let opp = if me >= 0 { (1 - me) as usize } else { 1 };
        let opponent = seats.get(opp).map(|s| s.to_string()).unwrap_or_default();
        let opp_type = seats.get(opp).and_then(|s| s["type"].as_str()).unwrap_or("").to_string();
        let winner = rec["result"]["winner"].as_i64().unwrap_or(-1);
        let won = if me < 0 || winner < 0 { -1 } else { (winner == me) as i64 };
        let first = rec["first"].as_i64().unwrap_or(-1);
        let (my_deck, opp_deck) = if me >= 0 { (name(me as usize), name(opp)) } else { (String::new(), String::new()) };
        con.execute(
            "INSERT OR REPLACE INTO matches \
             (id, created, mode, deck0_name, deck0_hash, deck1_name, deck1_hash, opponent, result, turns, path, \
              me, first, won, my_deck, opp_deck, opp_type, kind) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)",
            rusqlite::params![
                rec["id"].as_str().unwrap_or(""),
                rec["created"].as_str().unwrap_or(""),
                rec["mode"].as_str().unwrap_or(""),
                name(0),
                hash(0),
                name(1),
                hash(1),
                opponent,
                winner,
                rec["result"]["turns"].as_i64().unwrap_or(0),
                path.to_string_lossy(),
                me,
                first,
                won,
                my_deck,
                opp_deck,
                opp_type,
                kind_of(rec),
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// What kind of game a record is: "puzzle", "practice" (hints and undo, not counted in the stats) or "duel".
pub fn kind_of(rec: &Value) -> &'static str {
    if rec.get("puzzle").is_some_and(|p| !p.is_null()) {
        "puzzle"
    } else if rec["practice"] == true {
        "practice"
    } else {
        "duel"
    }
}

/// Write a gzip-compressed JSON document (temp + rename), the on-disk record rule.
pub fn save_gz(value: &Value, path: &Path) -> Result<(), String> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    let tmp = PathBuf::from(format!("{}.tmp", path.to_string_lossy().trim_end_matches(".gz")));
    {
        let f = fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let mut enc = GzEncoder::new(f, Compression::default());
        serde_json::to_writer(&mut enc, value).map_err(|e| e.to_string())?;
        enc.finish().map_err(|e| e.to_string())?;
    }
    fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Read a gzip-compressed JSON document (a record).
pub fn load_gz(path: &Path) -> Result<Value, String> {
    let f = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_reader(flate2::read::GzDecoder::new(f)).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_words_match_python() {
        // decklight.engine.duel.seed_words(7)
        assert_eq!(
            seed_words(7),
            [
                0x63cb_e1e4_5932_0dd7,
                0x044c_3cd7_f43c_661c,
                0xe698_4080_bab1_2a02,
                0x953a_eb70_673e_29cb
            ]
        );
    }

    #[test]
    fn uuid7_shape_and_order() {
        let a = uuid7();
        let b = uuid7();
        assert_eq!(a.len(), 36);
        assert_eq!(&a[14..15], "7");
        assert!(["8", "9", "a", "b"].contains(&&a[19..20]));
        assert!(a <= b || a > b);
    }

    #[test]
    fn writes_a_valid_full_record() {
        let dir = std::env::temp_dir().join(format!("decklight-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let root = std::path::Path::new(".");
        let env = RecordEnv::new(root, &dir);
        let g = RecordGame {
            seed: 21005,
            first: Some(1),
            main0: vec![1, 2, 3],
            extra0: vec![],
            main1: vec![4, 5],
            extra1: vec![9],
            side0: vec![],
            side1: vec![],
            seats: [
                json!({"type": "bot", "bot_mode": "random", "model": "m"}),
                json!({"type": "human", "player": env.install_id.clone()}),
            ],
            mode: "match".into(),
            responses: vec![ResponseRec { seat: 0, r: vec![0, 0, 0, 1], opt: vec![0], ms: 3 }],
            winner: Some(1),
            win_reason: 1,
            turns: 7,
            surrendered: false,
            disconnected: false,
            first_player: Some(1),
            deck_names: None,
            hints_used: Some(2),
            undos: Some(1),
            what_if: Some(json!({"source": "src", "k": 3, "option": 1, "turn": 4})),
            puzzle: Some(json!({"id": "p1"})),
            puzzle_result: Some("failed".into()),
            practice: Some(true),
        };
        let path = env.write(&g).unwrap();
        assert!(path.file_name().unwrap().to_string_lossy().ends_with(".match.json.gz"));
        let f = std::fs::File::open(&path).unwrap();
        let mut dec = flate2::read::GzDecoder::new(f);
        let mut text = String::new();
        use std::io::Read;
        dec.read_to_string(&mut text).unwrap();
        let rec: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(rec["format"], FORMAT);
        assert_eq!(rec["kind"], "full");
        assert_eq!(rec["engine_seed"][0], seed_words(21005)[0].to_string());
        assert_eq!(rec["responses"][0]["r"], "00000001");
        assert_eq!(rec["responses"][0]["seat"], 0);
        assert_eq!(rec["result"]["winner"], 1);
        assert_eq!(rec["decks"][0]["hash"], deck_hash(&[1, 2, 3], &[]));
        assert_eq!(rec["seats"][1]["player"], env.install_id);
        assert_eq!(rec["hints_used"], 2);
        assert_eq!(rec["undos"], 1);
        assert_eq!(rec["what_if"]["k"], 3);
        assert_eq!(rec["puzzle"]["id"], "p1");
        assert_eq!(rec["puzzle_result"], "failed");
        assert_eq!(rec["practice"], true);
        assert!(dir.join("index.sqlite").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn game(seats: [Value; 2], first: i32, winner: i32, turns: i32) -> RecordGame {
        RecordGame {
            seed: 7,
            first: Some(first),
            main0: vec![1],
            main1: vec![2],
            seats,
            mode: "practice".into(),
            winner: Some(winner),
            turns,
            deck_names: Some(["Blue-Eyes".into(), "Branded".into()]),
            ..Default::default()
        }
    }

    // Rows sort by turns: ids written in the same millisecond have a random order.
    fn rows(dir: &Path) -> Vec<(i64, i64, i64, String, String, String)> {
        let con = rusqlite::Connection::open(dir.join("index.sqlite")).unwrap();
        let mut st = con.prepare("SELECT me, first, won, my_deck, opp_deck, opp_type FROM matches ORDER BY turns").unwrap();
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    #[test]
    fn index_knows_our_seat_first_and_result() {
        let dir = std::env::temp_dir().join(format!("dl-rec-seat-{}", random_hex(4)));
        let env = RecordEnv::new(Path::new("."), &dir);
        let me = json!({"type": "human", "player": env.install_id.clone()});
        let bot = json!({"type": "bot"});
        // We are seat 1, went second (seat 0 first), and won.
        let path = env.write(&game([bot.clone(), me.clone()], 0, 1, 1)).unwrap();
        assert_eq!(load_gz(&path).unwrap()["decks"][1]["name"], "Branded");
        // We are seat 0, went first, and lost.
        env.write(&game([me.clone(), bot.clone()], 0, 1, 2)).unwrap();
        // EDOPro LAN host: a local human without a player id against a remote.
        env.write(&game([json!({"type": "remote"}), json!({"type": "human"})], 1, 0, 3)).unwrap();
        // Bot vs bot: no seat is ours.
        env.write(&game([bot.clone(), bot.clone()], 0, 0, 4)).unwrap();
        let r = rows(&dir);
        assert_eq!(r[0], (1, 0, 1, "Branded".into(), "Blue-Eyes".into(), "bot".into()));
        assert_eq!(r[1], (0, 0, 0, "Blue-Eyes".into(), "Branded".into(), "bot".into()));
        assert_eq!(r[2], (1, 1, 0, "Branded".into(), "Blue-Eyes".into(), "remote".into()));
        assert_eq!(r[3], (-1, 0, -1, "".into(), "".into(), "bot".into()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn old_index_layout_is_replaced() {
        let dir = std::env::temp_dir().join(format!("dl-rec-old-{}", random_hex(4)));
        fs::create_dir_all(&dir).unwrap();
        let con = rusqlite::Connection::open(dir.join("index.sqlite")).unwrap();
        con.execute_batch("CREATE TABLE matches (id TEXT PRIMARY KEY, created TEXT, opponent TEXT)").unwrap();
        drop(con);
        let env = RecordEnv::new(Path::new("."), &dir);
        let me = json!({"type": "human", "player": env.install_id.clone()});
        env.write(&game([json!({"type": "bot"}), me], 1, 1, 5)).unwrap();
        assert_eq!(rows(&dir), vec![(1, 1, 1, "Branded".into(), "Blue-Eyes".into(), "bot".into())]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_rebuilt_from_the_records_knows_practice_and_puzzles() {
        let dir = std::env::temp_dir().join(format!("dl-rec-kind-{}", random_hex(4)));
        let env = RecordEnv::new(Path::new("."), &dir);
        let me = json!({"type": "human", "player": env.install_id.clone()});
        let bot = json!({"type": "bot"});
        env.write(&RecordGame { practice: Some(true), ..game([me.clone(), bot.clone()], 0, 0, 1) }).unwrap();
        env.write(&RecordGame { puzzle: Some(json!({"id": "p1"})), ..game([me.clone(), bot.clone()], 0, 0, 2) }).unwrap();
        env.write(&RecordGame { practice: Some(false), ..game([me.clone(), bot.clone()], 0, 0, 3) }).unwrap();
        let kinds = |dir: &Path| -> Vec<String> {
            let con = rusqlite::Connection::open(dir.join("index.sqlite")).unwrap();
            let mut st = con.prepare("SELECT kind FROM matches ORDER BY turns").unwrap();
            st.query_map([], |r| r.get(0)).unwrap().collect::<Result<_, _>>().unwrap()
        };
        assert_eq!(kinds(&dir), ["practice", "puzzle", "duel"]);
        // An index from before `kind` is rebuilt from the files, not emptied.
        let con = rusqlite::Connection::open(dir.join("index.sqlite")).unwrap();
        con.execute_batch("ALTER TABLE matches DROP COLUMN kind").unwrap();
        drop(con);
        env.write(&game([me, bot], 0, 0, 4)).unwrap();
        assert_eq!(kinds(&dir), ["practice", "puzzle", "duel", "duel"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleted_games_go_to_the_trash_and_leave_the_index() {
        let dir = std::env::temp_dir().join(format!("dl-rec-trash-{}", random_hex(4)));
        let env = RecordEnv::new(Path::new("."), &dir);
        let me = json!({"type": "human", "player": env.install_id.clone()});
        let bot = json!({"type": "bot"});
        let a = env.write(&game([me.clone(), bot.clone()], 0, 0, 1)).unwrap();
        env.write(&game([me, bot], 0, 0, 2)).unwrap();
        let id = load_gz(&a).unwrap()["id"].as_str().unwrap().to_string();
        let ids = |dir: &Path| -> Vec<i64> {
            let con = rusqlite::Connection::open(dir.join("index.sqlite")).unwrap();
            let mut st = con.prepare("SELECT turns FROM matches ORDER BY turns").unwrap();
            st.query_map([], |r| r.get(0)).unwrap().collect::<Result<_, _>>().unwrap()
        };
        assert_eq!(env.trash(std::slice::from_ref(&id)).unwrap(), 1);
        assert_eq!(ids(&dir), [2]);
        assert!(!a.exists() && dir.join("trash").join(a.file_name().unwrap()).is_file());
        // An index path from another folder (records copied with their index) still trashes this folder's file.
        let other = std::env::temp_dir().join(format!("dl-rec-trash-other-{}", random_hex(4)));
        std::fs::create_dir_all(&other).unwrap();
        let con = rusqlite::Connection::open(dir.join("index.sqlite")).unwrap();
        let b_id: String = con.query_row("SELECT id FROM matches", [], |r| r.get(0)).unwrap();
        let b_name: String = con.query_row("SELECT path FROM matches", [], |r| r.get::<_, String>(0)).unwrap();
        let b_name = Path::new(&b_name).file_name().unwrap().to_owned();
        std::fs::write(other.join(&b_name), b"elsewhere").unwrap();
        con.execute("UPDATE matches SET path = ?1", [other.join(&b_name).to_string_lossy()]).unwrap();
        drop(con);
        assert_eq!(env.trash(std::slice::from_ref(&b_id)).unwrap(), 1);
        assert!(other.join(&b_name).is_file(), "the file in the other folder is not touched");
        assert!(dir.join("trash").join(&b_name).is_file());
        let _ = std::fs::remove_dir_all(&other);
        env.write(&game([json!({"type": "human", "player": env.install_id.clone()}), json!({"type": "bot"})], 0, 0, 2)).unwrap();
        // A second delete of the same game says the list was out of date.
        assert!(env.trash(&[id]).unwrap_err().contains("not found"));
        // A rebuild of the index does not bring the trashed game back.
        let con = rusqlite::Connection::open(dir.join("index.sqlite")).unwrap();
        con.execute_batch("ALTER TABLE matches DROP COLUMN kind").unwrap();
        drop(con);
        env.check_index();
        assert_eq!(ids(&dir), [2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deck_hash_ignores_order() {
        assert_eq!(deck_hash(&[1, 2, 3], &[9]), deck_hash(&[3, 1, 2], &[9]));
        assert_ne!(deck_hash(&[1, 2], &[]), deck_hash(&[1, 3], &[]));
    }
}
