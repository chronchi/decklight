//! Card names, effect texts and description strings (port of `decklight.web.view._Db`, behaviour for behaviour).
//!
//! Layering differs per field: `names` and the `str1..str16` strings take the last cdb that has the code;
//! `effect_text` (`desc`) takes the first one. `desc_text(hi, lo)` reads a system string for `hi <= 0`, else
//! card `hi`'s `str{lo+1}` and returns "" when it contains `%` or `lo` is out of range.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::default_cdbs;

const TYPE_XYZ: u32 = 0x800000;
const TYPE_LINK: u32 = 0x4000000;

pub type Stats = (i32, i32, i32, i32, i32, u32);

#[derive(Clone, Debug, Default)]
pub struct Texts {
    names: BTreeMap<u32, String>,
    stats: BTreeMap<u32, Stats>,
    effect: BTreeMap<u32, String>,
    strs: BTreeMap<u32, Vec<String>>,
    system: BTreeMap<u32, String>,
}

fn open(path: &Path) -> Result<rusqlite::Connection, String> {
    rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("{}: {e}", path.display()))
}

impl Texts {
    pub fn load(cdb_dir: &Path, extra_dirs: &[PathBuf], strings_conf: &Path) -> Result<Self, String> {
        Self::load_with_extra_strings(cdb_dir, extra_dirs, strings_conf, &[])
    }

    /// Like `load`, plus pack `strings.conf` files applied after the built-in one.
    pub fn load_with_extra_strings(
        cdb_dir: &Path,
        extra_dirs: &[PathBuf],
        strings_conf: &Path,
        extra_strings: &[PathBuf],
    ) -> Result<Self, String> {
        let mut out = Texts::default();
        for path in default_cdbs(cdb_dir, extra_dirs) {
            let con = open(&path)?;
            {
                let mut st = con
                    .prepare("SELECT id, name FROM texts")
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?)))
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, name) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    // Python's `str(name)`: a NULL column becomes the literal "None".
                    out.names.insert(code as u32, name.unwrap_or_else(|| "None".to_string()));
                }
            }
            {
                let mut st = con
                    .prepare("SELECT id, type, atk, def, level FROM datas")
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, i64>(2)?,
                            r.get::<_, i64>(3)?,
                            r.get::<_, i64>(4)?,
                        ))
                    })
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, type_, atk, mut dfn, level) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    let lv0 = if level < 0 { -(level & 0xFF) } else { level & 0xFF };
                    let mut lv = lv0;
                    let mut rank = 0;
                    let mut link_rating = 0;
                    let t = type_ as u32;
                    if t & TYPE_XYZ != 0 {
                        rank = lv;
                        lv = 0;
                    }
                    if t & TYPE_LINK != 0 {
                        link_rating = lv;
                        lv = 0;
                        dfn = 0;
                    }
                    out.stats.insert(code as u32, (atk as i32, dfn as i32, lv as i32, rank as i32,
                                                   link_rating as i32, t));
                }
            }
            {
                let mut st = con
                    .prepare("SELECT id, desc FROM texts")
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?)))
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, desc) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    // `setdefault`: the first cdb that has the code wins.
                    out.effect.entry(code as u32).or_insert_with(|| desc.unwrap_or_default());
                }
            }
            {
                let cols = (1..=16).map(|i| format!("str{i}")).collect::<Vec<_>>().join(", ");
                let sql = format!("SELECT id, {cols} FROM texts");
                let mut st = con.prepare(&sql).map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| {
                        let code = r.get::<_, i64>(0)?;
                        let mut v = Vec::with_capacity(16);
                        for i in 1..=16 {
                            v.push(r.get::<_, Option<String>>(i)?.unwrap_or_default());
                        }
                        Ok((code, v))
                    })
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, v) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    out.strs.insert(code as u32, v);
                }
            }
        }
        for conf in std::iter::once(strings_conf).chain(extra_strings.iter().map(PathBuf::as_path)) {
            if !conf.exists() {
                continue;
            }
            let text = std::fs::read_to_string(conf)
                .map_err(|e| format!("{}: {e}", conf.display()))?;
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("!system ") {
                    let mut parts = rest.splitn(2, ' ');
                    let num = parts.next().unwrap_or("");
                    if let Some(text) = parts.next()
                        && !num.is_empty()
                        && num.bytes().all(|b| b.is_ascii_digit())
                        && let Ok(n) = num.parse::<u32>()
                    {
                        out.system.insert(n, text.trim().to_string());
                    }
                }
            }
        }
        Ok(out)
    }

    /// Base/custom cdbs first, then pack cdbs whose rows override every field (including `desc`, which is
    /// first-wins between base cdbs so the Python oracle stays unchanged).
    pub fn load_full(
        cdb_dir: &Path,
        extra_dirs: &[PathBuf],
        pack_dirs: &[PathBuf],
        strings_conf: &Path,
        extra_strings: &[PathBuf],
    ) -> Result<Self, String> {
        let mut out = Self::load_with_extra_strings(cdb_dir, extra_dirs, strings_conf, extra_strings)?;
        for path in default_cdbs(cdb_dir, pack_dirs) {
            if !pack_dirs.iter().any(|d| path.starts_with(d)) {
                continue;
            }
            let con = open(&path)?;
            {
                let mut st = con
                    .prepare("SELECT id, name FROM texts")
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?)))
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, name) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    out.names.insert(code as u32, name.unwrap_or_else(|| "None".to_string()));
                }
            }
            {
                let mut st = con
                    .prepare("SELECT id, type, atk, def, level FROM datas")
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, i64>(2)?,
                            r.get::<_, i64>(3)?,
                            r.get::<_, i64>(4)?,
                        ))
                    })
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, type_, atk, mut dfn, level) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    let lv0 = if level < 0 { -(level & 0xFF) } else { level & 0xFF };
                    let mut lv = lv0;
                    let mut rank = 0;
                    let mut link_rating = 0;
                    let t = type_ as u32;
                    if t & TYPE_XYZ != 0 {
                        rank = lv;
                        lv = 0;
                    }
                    if t & TYPE_LINK != 0 {
                        link_rating = lv;
                        lv = 0;
                        dfn = 0;
                    }
                    out.stats.insert(code as u32, (atk as i32, dfn as i32, lv as i32, rank as i32,
                                                   link_rating as i32, t));
                }
            }
            {
                let mut st = con
                    .prepare("SELECT id, desc FROM texts")
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?)))
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, desc) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    out.effect.insert(code as u32, desc.unwrap_or_default());
                }
            }
            {
                let cols = (1..=16).map(|i| format!("str{i}")).collect::<Vec<_>>().join(", ");
                let sql = format!("SELECT id, {cols} FROM texts");
                let mut st = con.prepare(&sql).map_err(|e| format!("{}: {e}", path.display()))?;
                let rows = st
                    .query_map([], |r| {
                        let code = r.get::<_, i64>(0)?;
                        let mut v = Vec::with_capacity(16);
                        for i in 1..=16 {
                            v.push(r.get::<_, Option<String>>(i)?.unwrap_or_default());
                        }
                        Ok((code, v))
                    })
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                for row in rows {
                    let (code, v) = row.map_err(|e| format!("{}: {e}", path.display()))?;
                    out.strs.insert(code as u32, v);
                }
            }
        }
        Ok(out)
    }

    pub fn names(&self) -> &BTreeMap<u32, String> {
        &self.names
    }

    pub fn stats(&self) -> &BTreeMap<u32, Stats> {
        &self.stats
    }

    pub fn name(&self, code: u32) -> &str {
        self.names.get(&code).map(String::as_str).unwrap_or("")
    }

    pub fn stats_of(&self, code: u32) -> Option<&Stats> {
        self.stats.get(&code)
    }

    /// Whitespace runs collapse to one space and the result is stripped (Python `re.sub(r"\\s+", " ", s)`).
    pub fn effect_text(&self, code: u32) -> String {
        collapse(self.effect.get(&code).map(String::as_str).unwrap_or(""))
    }

    /// Text of an engine description id (card `str<lo+1>` or system string `lo`).
    pub fn desc_text(&self, hi: i64, lo: i64) -> String {
        if hi <= 0 {
            if lo < 0 {
                return String::new();
            }
            return self.system.get(&(lo as u32)).cloned().unwrap_or_default();
        }
        if !(0..16).contains(&lo) {
            return String::new();
        }
        let Some(strs) = self.strs.get(&(hi as u32)) else {
            return String::new();
        };
        let t = strs[lo as usize].trim();
        if t.contains('%') { String::new() } else { t.to_string() }
    }

    /// Every system string, as parsed from the conf (the number and its stripped text).
    pub fn system(&self) -> &BTreeMap<u32, String> {
        &self.system
    }
}

fn collapse(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            pending_space = !out.is_empty();
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_and_trim() {
        assert_eq!(collapse("  a \n\t b  "), "a b");
        assert_eq!(collapse(""), "");
        assert_eq!(collapse("x"), "x");
    }

    #[test]
    fn pack_rows_override_changed_text() {
        let dir = std::env::temp_dir().join(format!("texts-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = dir.join("base");
        let pack = dir.join("pack");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&pack).unwrap();
        let cols = "id INT, name TEXT, desc TEXT, str1 TEXT, str2 TEXT, str3 TEXT, str4 TEXT, str5 TEXT,                     str6 TEXT, str7 TEXT, str8 TEXT, str9 TEXT, str10 TEXT, str11 TEXT, str12 TEXT,                     str13 TEXT, str14 TEXT, str15 TEXT, str16 TEXT";
        for (d, name, desc, s1) in [(&base, "Old", "Old desc", "old str"), (&pack, "New", "New desc", "new str")] {
            let con = rusqlite::Connection::open(d.join("cards.cdb")).unwrap();
            con.execute_batch(&format!(
                "CREATE TABLE texts ({cols}); CREATE TABLE datas (id INT, type INT, atk INT, def INT, level INT);"
            ))
            .unwrap();
            con.execute("INSERT INTO datas VALUES (1, 33, 1000, 1000, 4)", []).unwrap();
            con.execute(
                &format!("INSERT INTO texts VALUES ({})", std::iter::repeat_n("?", 19).collect::<Vec<_>>().join(", ")),
                rusqlite::params_from_iter(
                    std::iter::once("1".to_string()).chain(std::iter::once(name.to_string()))
                        .chain(std::iter::once(desc.to_string())).chain(std::iter::once(s1.to_string()))
                        .chain(std::iter::repeat_n(String::new(), 15)),
                ),
            )
            .unwrap();
        }
        let conf = dir.join("strings.conf");
        let t = Texts::load_full(&base, &[], &[pack], &conf, &[]).unwrap();
        assert_eq!(t.name(1), "New");
        assert_eq!(t.effect_text(1), "New desc");
        assert_eq!(t.desc_text(1, 0), "new str");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn system_conf_split_semantics() {
        // Python `line.split(" ", 2)` keeps a double space as an empty number.
        let dir = std::env::temp_dir().join(format!("texts-test-{}", std::process::id()));
        let cdb_dir = dir.join("cdb");
        std::fs::create_dir_all(&cdb_dir).unwrap();
        let con = rusqlite::Connection::open(cdb_dir.join("cards.cdb")).unwrap();
        con.execute_batch(
            "CREATE TABLE texts (id INT, name TEXT, desc TEXT, str1 TEXT, str2 TEXT, str3 TEXT, str4 TEXT, \
             str5 TEXT, str6 TEXT, str7 TEXT, str8 TEXT, str9 TEXT, str10 TEXT, str11 TEXT, str12 TEXT, \
             str13 TEXT, str14 TEXT, str15 TEXT, str16 TEXT); \
             CREATE TABLE datas (id INT, type INT, atk INT, def INT, level INT);",
        )
        .unwrap();
        drop(con);
        let conf = dir.join("strings.conf");
        std::fs::write(&conf, "!system 1 hello world \n!system  2 nope\n!system 3\n").unwrap();
        let t = Texts::load(&cdb_dir, &[], &conf).unwrap();
        assert_eq!(t.desc_text(0, 1), "hello world");
        assert_eq!(t.desc_text(0, 2), "");
        assert_eq!(t.desc_text(0, 3), "");
    }
}
