//! Card store (port of `src/decklight/engine/resources.py` and `decks/ydk.py`, behaviour for behaviour).
//!
//! Cards come in layers, later layers overriding earlier ones: `cards.cdb`, then `release-*.cdb`, then
//! `prerelease-*.cdb` (not rush), then every extra card dir's `*.cdb` (custom cards, and card packs for new cards).
//! Scripts: the CardScripts root (system/proc scripts), `official/`, `pre-release/`, then each extra dir's `script/`;
//! the first file of a name wins, so extras never override official scripts.
pub mod emb;
pub mod pack;
pub mod texts;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use ygoenv_sys::CardData;

const TYPE_LINK: u32 = 0x4000000;

fn sorted_glob(dir: &Path, pred: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(&pred))
            .collect(),
        Err(_) => Vec::new(),
    };
    out.sort();
    out
}

/// `default_cdbs`: the official files in override order, then each extra dir's `*.cdb`.
pub fn default_cdbs(cdb_dir: &Path, extra_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = vec![cdb_dir.join("cards.cdb")];
    paths.extend(sorted_glob(cdb_dir, |n| n.starts_with("release-") && n.ends_with(".cdb")));
    paths.extend(sorted_glob(cdb_dir, |n| n.starts_with("prerelease-") && n.ends_with(".cdb") && !n.contains("rush")));
    for d in extra_dirs.iter().filter(|d| d.is_dir()) {
        paths.extend(sorted_glob(d, |n| n.ends_with(".cdb")));
    }
    paths
}

/// One `datas` row decoded exactly as `load_cards` does.
pub fn decode_row(
    code: i64, alias: i64, setcode: i64, type_: i64, atk: i64, def: i64, level: i64, race: i64, attribute: i64,
) -> CardData {
    let setcode = setcode as u64;
    let setcodes: Vec<u16> = (0..4).map(|i| ((setcode >> (16 * i)) & 0xFFFF) as u16).filter(|&s| s != 0).collect();
    let type_ = type_ as u32;
    let (mut link_marker, mut defense) = (0u32, def);
    if type_ & TYPE_LINK != 0 {
        link_marker = def as u32;
        defense = 0;
    }
    let lv: i64 = if level < 0 { -(level & 0xFF) } else { level & 0xFF };
    CardData {
        code: code as u32,
        alias: alias as u32,
        setcodes,
        type_,
        level: (lv as u64 & 0xFFFF_FFFF) as u32,
        attribute: attribute as u32,
        race: race as u64,
        attack: atk as i32,
        defense: defense as i32,
        lscale: ((level >> 24) & 0xFF) as u32,
        rscale: ((level >> 16) & 0xFF) as u32,
        link_marker,
    }
}

/// `load_cards`: every card of every cdb, later files overriding earlier ones.
pub fn load_cards(paths: &[PathBuf]) -> Result<BTreeMap<u32, CardData>, String> {
    let mut cards = BTreeMap::new();
    for path in paths {
        let con = rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut st = con
            .prepare("SELECT id, alias, setcode, type, atk, def, level, race, attribute FROM datas")
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let rows = st
            .query_map([], |r| {
                Ok(decode_row(r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?,
                              r.get(8)?))
            })
            .map_err(|e| format!("{}: {e}", path.display()))?;
        for row in rows {
            let cd = row.map_err(|e| format!("{}: {e}", path.display()))?;
            cards.insert(cd.code, cd);
        }
    }
    Ok(cards)
}

const PACK_MAGIC: &[u8; 8] = b"DLSPACK1";

/// Where one script comes from: a file on disk, or a byte range of a packed file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScriptRef {
    File(PathBuf),
    Packed { pack: PathBuf, offset: u64, len: u32 },
}

impl ScriptRef {
    /// The script's raw bytes (what the engine reads).
    pub fn read(&self) -> Result<Vec<u8>, String> {
        match self {
            ScriptRef::File(p) => std::fs::read(p).map_err(|e| format!("{}: {e}", p.display())),
            ScriptRef::Packed { pack, offset, len } => read_range(pack, *offset, *len),
        }
    }
}

fn read_range(path: &Path, offset: u64, len: u32) -> Result<Vec<u8>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.seek(SeekFrom::Start(offset)).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = vec![0u8; len as usize];
    f.read_exact(&mut out).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(out)
}

/// `<scripts_dir>.pack` (e.g. `CardScripts` -> `CardScripts.pack`).
pub fn pack_path(scripts_dir: &Path) -> PathBuf {
    let mut s = scripts_dir.as_os_str().to_os_string();
    s.push(".pack");
    PathBuf::from(s)
}

fn u16_at(b: &[u8], pos: usize) -> Result<u16, String> {
    b.get(pos..pos + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| "truncated pack header".to_string())
}

fn u32_at(b: &[u8], pos: usize) -> Result<u32, String> {
    b.get(pos..pos + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "truncated pack header".to_string())
}

fn u64_at(b: &[u8], pos: usize) -> Result<u64, String> {
    b.get(pos..pos + 8)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
        .ok_or_else(|| "truncated pack header".to_string())
}

/// Parse `CardScripts.pack`: magic, u32 count, {u16 name_len, name, u64 offset, u32 len}, then the file bytes.
/// Every entry must lie inside the file; a truncated or malformed pack is an error, never a panic.
pub fn parse_pack(bytes: &[u8]) -> Result<Vec<(String, u64, u32)>, String> {
    if bytes.len() < 12 || &bytes[0..8] != PACK_MAGIC.as_slice() {
        return Err("bad pack magic".into());
    }
    let count = u32_at(bytes, 8)? as usize;
    let mut pos = 12usize;
    let mut out = Vec::with_capacity(count.min(bytes.len() / 6));
    for _ in 0..count {
        let nlen = u16_at(bytes, pos)? as usize;
        pos += 2;
        let name_bytes = bytes.get(pos..pos + nlen).ok_or("truncated pack name")?;
        let name = std::str::from_utf8(name_bytes).map_err(|_| "pack name is not UTF-8")?.to_string();
        pos += nlen;
        let offset = u64_at(bytes, pos)?;
        let len = u32_at(bytes, pos + 8)?;
        pos += 12;
        let end = offset.checked_add(len as u64).ok_or("pack entry overflows")?;
        if end > bytes.len() as u64 {
            return Err(format!("pack entry {name:?} runs past the file"));
        }
        out.push((name, offset, len));
    }
    Ok(out)
}

fn read_pack(pack: &Path) -> Result<Vec<(String, u64, u32)>, String> {
    let bytes = std::fs::read(pack).map_err(|e| format!("{}: {e}", pack.display()))?;
    parse_pack(&bytes).map_err(|e| format!("{}: {e}", pack.display()))
}

/// Root files, then `official/`, then `pre-release/` (the directory order `index_scripts` uses).
fn pack_rank(name: &str) -> Option<u8> {
    match (name.split('/').next(), name.split('/').nth(1), name.split('/').nth(2)) {
        (Some(_), None, _) => Some(0),
        (Some("official"), Some(f), None) if !f.is_empty() => Some(1),
        (Some("pre-release"), Some(f), None) if !f.is_empty() => Some(2),
        _ => None,
    }
}

/// Index a packed script tree (first name wins: root, `official/`, `pre-release/`, then extra dirs).
pub fn index_pack(pack: &Path, extra_dirs: &[PathBuf]) -> Result<BTreeMap<String, ScriptRef>, String> {
    let entries = read_pack(pack)?;
    let mut index = BTreeMap::new();
    for rank in 0..=2u8 {
        for (name, offset, len) in &entries {
            if !name.ends_with(".lua") || pack_rank(name) != Some(rank) {
                continue;
            }
            let leaf = name.rsplit('/').next().unwrap().to_string();
            index.entry(leaf).or_insert(ScriptRef::Packed {
                pack: pack.to_path_buf(),
                offset: *offset,
                len: *len,
            });
        }
    }
    for d in extra_dirs.iter().filter(|d| d.is_dir()) {
        for p in sorted_glob(&d.join("script"), |n| n.ends_with(".lua")) {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            index.entry(name).or_insert(ScriptRef::File(p));
        }
    }
    Ok(index)
}

/// `index_scripts`: script file name -> source (first wins). Uses the directory when it exists, otherwise the
/// adjacent `<scripts_dir>.pack` (the Android app ships only the pack). Neither -> empty (as before).
pub fn index_scripts(
    scripts_dir: &Path,
    extra_dirs: &[PathBuf],
) -> Result<BTreeMap<String, ScriptRef>, String> {
    if scripts_dir.is_dir() {
        let mut index = BTreeMap::new();
        let mut dirs: Vec<PathBuf> =
            ["", "official", "pre-release"].iter().map(|s| scripts_dir.join(s)).collect();
        dirs.extend(extra_dirs.iter().filter(|d| d.is_dir()).map(|d| d.join("script")));
        for d in dirs {
            for p in sorted_glob(&d, |n| n.ends_with(".lua")) {
                let name = p.file_name().unwrap().to_string_lossy().into_owned();
                index.entry(name).or_insert(ScriptRef::File(p));
            }
        }
        Ok(index)
    } else {
        let pack = pack_path(scripts_dir);
        if pack.is_file() { index_pack(&pack, extra_dirs) } else { Ok(BTreeMap::new()) }
    }
}

/// Like `index_scripts`, plus pack dirs whose scripts override the official/custom ones (weekly errata).
pub fn index_scripts_layered(
    scripts_dir: &Path,
    extra_dirs: &[PathBuf],
    pack_dirs: &[PathBuf],
) -> Result<BTreeMap<String, ScriptRef>, String> {
    let mut index = index_scripts(scripts_dir, extra_dirs)?;
    for d in pack_dirs.iter().filter(|d| d.is_dir()) {
        for p in sorted_glob(&d.join("script"), |n| n.ends_with(".lua")) {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            index.insert(name, ScriptRef::File(p));
        }
    }
    Ok(index)
}

/// A C++ engine loaded with `cards` and `scripts` (what `cpp_env.cpp_engine` does from Python).
pub fn build_engine(cards: &BTreeMap<u32, CardData>, scripts: &BTreeMap<String, ScriptRef>) -> ygoenv_sys::Engine {
    let mut e = ygoenv_sys::Engine::new();
    for cd in cards.values() {
        e.add_card(cd);
    }
    for (name, source) in scripts {
        match source {
            ScriptRef::File(p) => e.set_script(name, &p.to_string_lossy()),
            ScriptRef::Packed { pack, offset, len } => {
                e.set_script_range(name, &pack.to_string_lossy(), *offset, *len)
            }
        }
    }
    e.finalize();
    e
}

/// A .ydk deck (`decks/ydk.py`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Deck {
    pub main: Vec<u32>,
    pub extra: Vec<u32>,
    pub side: Vec<u32>,
}

pub fn parse_ydk(text: &str) -> Deck {
    let mut d = Deck::default();
    let mut cur = 0;
    for line in text.lines() {
        let line = line.trim();
        match line {
            "#main" => cur = 0,
            "#extra" => cur = 1,
            "!side" => cur = 2,
            _ if !line.is_empty() && line.bytes().all(|b| b.is_ascii_digit()) => {
                if let Ok(v) = line.parse::<u64>()
                    && v > 0
                {
                    let v = v as u32;
                    match cur {
                        0 => d.main.push(v),
                        1 => d.extra.push(v),
                        _ => d.side.push(v),
                    }
                }
            }
            _ => {}
        }
    }
    d
}

/// `shuffle_main`: seeded Fisher-Yates over a splitmix64 stream, bit-identical to Python and vecenv.cpp.
pub fn shuffle_main(main: &mut [u32], seed: u64, team: u64) {
    let mut st = seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(team + 1);
    for k in (2..=main.len()).rev() {
        st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z % k as u64) as usize;
        main.swap(k - 1, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ydk_sections_and_junk() {
        let d = parse_ydk("#created by x\n#main\n123\n 456 \n#extra\n789\n!side\n1\nabc\n0\n");
        assert_eq!(d, Deck { main: vec![123, 456], extra: vec![789], side: vec![1] });
    }

    #[test]
    fn link_and_scales_decode() {
        // Link: def holds the markers; pendulum scales in level's high bytes; negative level keeps the sign rule.
        let l = decode_row(1, 0, 0x0001_0002, TYPE_LINK as i64 | 1, 2500, 0x1F, 4, 1, 32);
        assert_eq!((l.link_marker, l.defense, l.setcodes.clone()), (0x1F, 0, vec![2, 1]));
        let p = decode_row(2, 0, 0, 1, 0, 0, (5 << 24) | (5 << 16) | 7, 1, 1);
        assert_eq!((p.lscale, p.rscale, p.level), (5, 5, 7));
    }

    #[test]
    fn pack_scripts_override_official() {
        let tmp = std::env::temp_dir().join(format!("dlcards-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let scripts = tmp.join("scripts");
        let pack = tmp.join("pack");
        std::fs::create_dir_all(scripts.join("official")).unwrap();
        std::fs::create_dir_all(pack.join("script")).unwrap();
        std::fs::write(scripts.join("official/c123.lua"), b"official").unwrap();
        std::fs::write(pack.join("script/c123.lua"), b"pack").unwrap();
        let idx = index_scripts_layered(&scripts, &[], &[pack.clone()]).unwrap();
        assert_eq!(idx["c123.lua"], ScriptRef::File(pack.join("script/c123.lua")));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn make_pack(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut entries: Vec<(&str, &[u8])> = entries.to_vec();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let names: Vec<(&[u8], &[u8])> = entries.iter().map(|(n, d)| (n.as_bytes(), *d)).collect();
        let header = 8 + 4 + names.iter().map(|(n, _)| 2 + n.len() + 12).sum::<usize>();
        let mut out = Vec::with_capacity(header + names.iter().map(|(_, d)| d.len()).sum::<usize>());
        out.extend_from_slice(PACK_MAGIC);
        out.extend_from_slice(&(names.len() as u32).to_le_bytes());
        let mut offset = header as u64;
        for (n, d) in &names {
            out.extend_from_slice(&(n.len() as u16).to_le_bytes());
            out.extend_from_slice(n);
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(d.len() as u32).to_le_bytes());
            offset += d.len() as u64;
        }
        for (_, d) in &names {
            out.extend_from_slice(d);
        }
        out
    }

    fn walk_pack_files(dir: &Path, base: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        for ent in std::fs::read_dir(dir).unwrap() {
            let p = ent.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy();
            if matches!(name.as_ref(), ".git" | "__pycache__" | "ci" | ".github") {
                continue;
            }
            if p.is_dir() {
                walk_pack_files(&p, base, out);
            } else if p.is_file() {
                let rel = p.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/");
                out.push((rel, std::fs::read(&p).unwrap()));
            }
        }
    }

    #[test]
    fn pack_round_trip_matches_files_and_first_wins() {
        let tmp = std::env::temp_dir().join(format!("dlcards-pack-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dir = tmp.join("CardScripts");
        std::fs::create_dir_all(dir.join("official")).unwrap();
        std::fs::create_dir_all(dir.join("pre-release")).unwrap();
        let files = [
            ("constant.lua", b"root-constant".as_slice()),
            ("c1.lua", b"root-c1".as_slice()),
            ("official/c1.lua", b"official-c1".as_slice()),
            ("official/c2.lua", b"official-c2".as_slice()),
            ("pre-release/c3.lua", b"pre-release-c3".as_slice()),
        ];
        for (rel, data) in files {
            std::fs::write(dir.join(rel), data).unwrap();
        }
        let mut walked = Vec::new();
        walk_pack_files(&dir, &dir, &mut walked);
        let entries: Vec<(&str, &[u8])> = walked.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        let pack_bytes = make_pack(&entries);
        let scripts = tmp.join("scripts");
        std::fs::write(pack_path(&scripts), &pack_bytes).unwrap();
        let idx = index_scripts(&scripts, &[]).unwrap();
        for (leaf, want) in [
            ("constant.lua", b"root-constant".as_slice()),
            ("c1.lua", b"root-c1".as_slice()),
            ("c2.lua", b"official-c2".as_slice()),
            ("c3.lua", b"pre-release-c3".as_slice()),
        ] {
            assert_eq!(idx[leaf].read().unwrap(), want, "{leaf}");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn truncated_pack_is_an_error() {
        let tmp = std::env::temp_dir().join(format!("dlcards-pack-trunc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let scripts = tmp.join("scripts");
        std::fs::write(pack_path(&scripts), make_pack(&[("a.lua", b"aaa"), ("b.lua", b"bbb")])).unwrap();
        let full = std::fs::read(pack_path(&scripts)).unwrap();
        std::fs::write(pack_path(&scripts), &full[..full.len() - 1]).unwrap();
        assert!(index_scripts(&scripts, &[]).is_err());
        std::fs::write(pack_path(&scripts), &full[..7]).unwrap();
        assert!(index_scripts(&scripts, &[]).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn packed_engine_scripts_match_the_directory() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = root.join("third_party/CardScripts");
        if !dir.is_dir() {
            eprintln!("skip: no CardScripts");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("dlcards-pack-engine-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let pack = root.join("data/cache/app/apk/CardScripts.pack");
        let pack = if pack.is_file() {
            pack
        } else {
            let mut walked = Vec::new();
            walk_pack_files(&dir, &dir, &mut walked);
            let entries: Vec<(&str, &[u8])> = walked.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
            let built = tmp.join("CardScripts.pack");
            std::fs::write(&built, make_pack(&entries)).unwrap();
            built
        };
        let dir_idx = index_scripts(&dir, &[]).unwrap();
        let pack_idx = index_pack(&pack, &[]).unwrap();
        assert_eq!(dir_idx.keys().collect::<Vec<_>>(), pack_idx.keys().collect::<Vec<_>>());
        let mut dir_engine = build_engine(&BTreeMap::new(), &dir_idx);
        let mut pack_engine = build_engine(&BTreeMap::new(), &pack_idx);
        dir_engine.set_precompile(false);
        pack_engine.set_precompile(false);
        let names: Vec<&String> = dir_idx.keys().collect();
        let step = (names.len() / 50).max(1);
        for i in 0..50 {
            let name = names[(i * step) % names.len()];
            let want = dir_idx[name].read().unwrap();
            let got = pack_engine.script(name).expect(name);
            assert_eq!(got, want, "{name}");
            assert_eq!(dir_engine.script(name).unwrap(), want, "{name}");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn shuffle_matches_python() {
        // decklight.engine.duel.shuffle_main(list(range(1, 11)), 7, 1) on the laptop.
        let mut m: Vec<u32> = (1..=10).collect();
        shuffle_main(&mut m, 7, 1);
        assert_eq!(m, PY_SHUFFLE_7_1.to_vec());
    }

    const PY_SHUFFLE_7_1: [u32; 10] = [8, 4, 5, 3, 9, 1, 2, 7, 6, 10];
}
