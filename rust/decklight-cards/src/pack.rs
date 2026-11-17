//! Versioned card packs (A5): a zip with a manifest, verified on load, activated atomically and rolled back
//! to the previous pack when the active one is bad.
//!
//! Layout of a pack:
//! `manifest.json`, `cards.cdb`, `script/*.lua`, `strings.conf`, `emb.npz`, `lua_desc.npz`.
//! The directory is directly usable as an extra card dir (`default_cdbs` / `index_scripts`).

use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FORMAT: &str = "decklight.cardpack/1";
pub const MANIFEST: &str = "manifest.json";
pub const MAX_PACK_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_PACK_UNCOMPRESSED: u64 = 1024 * 1024 * 1024;
pub const MAX_PACK_ENTRIES: usize = 10_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub id: String,
    #[serde(default)]
    pub created: String,
    #[serde(default)]
    pub base: serde_json::Value,
    #[serde(default)]
    pub counts: serde_json::Value,
    #[serde(default)]
    pub codes: Vec<u32>,
    #[serde(default)]
    pub files: Vec<FileEntry>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Marker {
    pub id: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PackStatus {
    pub state: String,
    pub id: Option<String>,
    pub sha256: Option<String>,
    pub created: Option<String>,
    pub cards: u64,
    pub scripts: u64,
    pub strings: u64,
    pub embeddings: u64,
    pub lua_desc: u64,
    pub previous_id: Option<String>,
    pub message: Option<String>,
    pub alerts: Vec<String>,
    pub restart_required: bool,
}

#[derive(Debug)]
pub struct Resolved {
    pub status: PackStatus,
    pub dir: Option<PathBuf>,
    pub manifest: Option<Manifest>,
}

impl Resolved {
    fn stock(alerts: Vec<String>) -> Resolved {
        Resolved {
            status: PackStatus {
                state: "stock".into(),
                alerts,
                ..PackStatus::default()
            },
            dir: None,
            manifest: None,
        }
    }
}

pub struct PackStore {
    root: PathBuf,
}

fn sha256_bytes(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let data = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(sha256_bytes(&data))
}

fn safe_rel(path: &str) -> Result<PathBuf, String> {
    let p = Path::new(path);
    if p.is_absolute() || path.is_empty() {
        return Err(format!("unsafe path {path:?}"));
    }
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(part) => out.push(part),
            _ => return Err(format!("unsafe path {path:?}")),
        }
    }
    Ok(out)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && id != "."
        && id != ".."
        && id != "active.json"
        && id != "previous.json"
        && !id.starts_with(".staging-")
        && !id.starts_with("merged-")
}

fn table_cols(con: &rusqlite::Connection, table: &str) -> Result<BTreeSet<String>, String> {
    let mut st = con.prepare(&format!("PRAGMA table_info({table})")).map_err(|e| e.to_string())?;
    let rows = st.query_map([], |r| r.get::<_, String>(1)).map_err(|e| e.to_string())?;
    let mut out = BTreeSet::new();
    for row in rows {
        out.insert(row.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

fn parse_manifest(data: &[u8]) -> Result<Manifest, String> {
    let m: Manifest = serde_json::from_slice(data).map_err(|e| format!("manifest.json: {e}"))?;
    if m.format != FORMAT {
        return Err(format!("unsupported pack format {:?} (want {FORMAT})", m.format));
    }
    if !valid_id(&m.id) {
        return Err(format!("invalid pack id {:?}", m.id));
    }
    let mut seen = BTreeSet::new();
    for f in &m.files {
        safe_rel(&f.path)?;
        if !seen.insert(f.path.clone()) {
            return Err(format!("duplicate file {:?}", f.path));
        }
        if f.sha256.len() != 64 || !f.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("bad sha256 for {:?}", f.path));
        }
    }
    if !seen.contains("cards.cdb") {
        return Err("pack has no cards.cdb".into());
    }
    Ok(m)
}

fn walk_files(dir: &Path, base: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let rd = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for ent in rd {
        let ent = ent.map_err(|e| format!("{}: {e}", dir.display()))?;
        let p = ent.path();
        let ft = ent.file_type().map_err(|e| format!("{}: {e}", p.display()))?;
        if ft.is_dir() {
            walk_files(&p, base, out)?;
        } else if ft.is_file() {
            let rel = p.strip_prefix(base).map_err(|e| e.to_string())?.to_path_buf();
            out.push(rel);
        }
    }
    Ok(())
}

fn verify_dir(dir: &Path, manifest: &Manifest) -> Result<String, String> {
    let manifest_bytes = fs::read(dir.join(MANIFEST)).map_err(|e| format!("{}: {e}", dir.join(MANIFEST).display()))?;
    let listed: BTreeSet<&str> = manifest.files.iter().map(|f| f.path.as_str()).collect();
    let mut actual = Vec::new();
    walk_files(dir, dir, &mut actual)?;
    for rel in &actual {
        let rel_s = rel.to_string_lossy().replace('\\', "/");
        if rel_s == MANIFEST {
            continue;
        }
        if !listed.contains(rel_s.as_str()) {
            return Err(format!("unlisted file {rel_s:?}"));
        }
    }
    for f in &manifest.files {
        let rel = safe_rel(&f.path)?;
        let p = dir.join(&rel);
        let md = fs::metadata(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        if !md.is_file() {
            return Err(format!("{} is not a file", p.display()));
        }
        if md.len() != f.size {
            return Err(format!("{}: size {} != manifest {}", f.path, md.len(), f.size));
        }
        let got = sha256_file(&p)?;
        if !got.eq_ignore_ascii_case(&f.sha256) {
            return Err(format!("{}: sha256 {got} != manifest {}", f.path, f.sha256));
        }
    }
    let cdb = dir.join("cards.cdb");
    let con = rusqlite::Connection::open_with_flags(&cdb, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("{}: {e}", cdb.display()))?;
    let datas = table_cols(&con, "datas")?;
    for col in ["id", "alias", "setcode", "type", "atk", "def", "level", "race", "attribute"] {
        if !datas.contains(col) {
            return Err(format!("cards.cdb datas has no {col} column"));
        }
    }
    let texts = table_cols(&con, "texts")?;
    for col in std::iter::once("id".to_string())
        .chain(std::iter::once("name".to_string()))
        .chain(std::iter::once("desc".to_string()))
        .chain((1..=16).map(|i| format!("str{i}")))
    {
        if !texts.contains(&col) {
            return Err(format!("cards.cdb texts has no {col} column"));
        }
    }
    for code in &manifest.codes {
        let n: i64 = con
            .query_row("SELECT count(*) FROM datas WHERE id=?", [*code as i64], |r| r.get(0))
            .map_err(|e| format!("{}: {e}", cdb.display()))?;
        if n == 0 {
            return Err(format!("manifest code {code} missing from cards.cdb datas"));
        }
        let n: i64 = con
            .query_row("SELECT count(*) FROM texts WHERE id=?", [*code as i64], |r| r.get(0))
            .map_err(|e| format!("{}: {e}", cdb.display()))?;
        if n == 0 {
            return Err(format!("manifest code {code} missing from cards.cdb texts"));
        }
    }
    let emb = dir.join("emb.npz");
    if emb.is_file() {
        crate::emb::Emb::load(&emb)?;
    }
    Ok(sha256_bytes(&manifest_bytes))
}

fn marker_path(root: &Path, name: &str) -> PathBuf {
    root.join(name)
}

fn read_marker(root: &Path, name: &str) -> Option<Marker> {
    let data = fs::read(marker_path(root, name)).ok()?;
    serde_json::from_slice(&data).ok()
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&tmp, data).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

fn write_marker(root: &Path, name: &str, m: &Marker) -> Result<(), String> {
    fs::create_dir_all(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let data = serde_json::to_vec_pretty(m).map_err(|e| e.to_string())?;
    write_atomic(&marker_path(root, name), &data)
}

fn remove_marker(root: &Path, name: &str) {
    let _ = fs::remove_file(marker_path(root, name));
}

fn counts(manifest: &Manifest) -> (u64, u64, u64, u64, u64) {
    let get = |k: &str| manifest.counts.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    (
        get("cards"),
        get("scripts"),
        get("strings"),
        get("embeddings"),
        get("lua_desc"),
    )
}

fn status_from(root: &Path, marker: &Marker, manifest: &Manifest, state: &str, alerts: Vec<String>) -> PackStatus {
    let (cards, scripts, strings, embeddings, lua_desc) = counts(manifest);
    PackStatus {
        state: state.into(),
        id: Some(marker.id.clone()),
        sha256: Some(marker.sha256.clone()),
        created: Some(manifest.created.clone()).filter(|s| !s.is_empty()),
        cards,
        scripts,
        strings,
        embeddings,
        lua_desc,
        previous_id: read_marker(root, "previous.json").map(|m| m.id),
        message: None,
        alerts,
        restart_required: false,
    }
}

fn load_marker(root: &Path, marker: &Marker) -> Result<(PathBuf, Manifest, String), String> {
    if !valid_id(&marker.id) {
        return Err(format!("invalid pack id {:?}", marker.id));
    }
    let dir = root.join(&marker.id);
    if !dir.is_dir() {
        return Err(format!("{}: missing", dir.display()));
    }
    let data = fs::read(dir.join(MANIFEST)).map_err(|e| format!("{}: {e}", dir.join(MANIFEST).display()))?;
    let manifest = parse_manifest(&data)?;
    if manifest.id != marker.id {
        return Err(format!("manifest id {:?} != marker {:?}", manifest.id, marker.id));
    }
    let sha = verify_dir(&dir, &manifest)?;
    if !sha.eq_ignore_ascii_case(&marker.sha256) {
        return Err(format!("manifest sha256 {sha} != marker {}", marker.sha256));
    }
    Ok((dir, manifest, sha))
}

impl PackStore {
    pub fn new(root: impl Into<PathBuf>) -> PackStore {
        PackStore { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The active pack, verified. A bad active pack rolls back to `previous.json`; a bad previous pack leaves
    /// the store on stock and reports the alerts.
    pub fn resolve(&self) -> Resolved {
        let mut alerts = Vec::new();
        let active_path = marker_path(&self.root, "active.json");
        if active_path.exists() {
            match read_marker(&self.root, "active.json") {
                Some(active) => match load_marker(&self.root, &active) {
                    Ok((dir, manifest, _sha)) => {
                        let status = status_from(&self.root, &active, &manifest, "active", alerts);
                        return Resolved { status, dir: Some(dir), manifest: Some(manifest) };
                    }
                    Err(e) => alerts.push(format!("active pack {} rejected: {e}", active.id)),
                },
                None => alerts.push("active.json is unreadable".into()),
            }
            if let Some(previous) = read_marker(&self.root, "previous.json") {
                match load_marker(&self.root, &previous) {
                    Ok((dir, manifest, sha)) => {
                        let marker = Marker { id: previous.id.clone(), sha256: sha };
                        let _ = write_marker(&self.root, "active.json", &marker);
                        remove_marker(&self.root, "previous.json");
                        alerts.push(format!("rolled back to {}", previous.id));
                        let mut status = status_from(&self.root, &marker, &manifest, "rolled_back", alerts);
                        status.message = Some("The newest card pack was rejected; using the previous pack.".into());
                        return Resolved { status, dir: Some(dir), manifest: Some(manifest) };
                    }
                    Err(e) => alerts.push(format!("previous pack {} rejected: {e}", previous.id)),
                }
            }
            remove_marker(&self.root, "active.json");
            remove_marker(&self.root, "previous.json");
            alerts.push("no usable card pack; using the built-in cards".into());
        }
        Resolved::stock(alerts)
    }

    pub fn install_zip(&self, zip_path: &Path, expected_sha256: Option<&str>) -> Result<Resolved, String> {
        let md = fs::metadata(zip_path).map_err(|e| format!("{}: {e}", zip_path.display()))?;
        if md.len() > MAX_PACK_BYTES {
            return Err(format!("pack is {} bytes (limit {MAX_PACK_BYTES})", md.len()));
        }
        let file = fs::File::open(zip_path).map_err(|e| format!("{}: {e}", zip_path.display()))?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("{}: {e}", zip_path.display()))?;
        if archive.len() > MAX_PACK_ENTRIES {
            return Err(format!("pack has {} entries (limit {MAX_PACK_ENTRIES})", archive.len()));
        }

        let manifest_data = {
            let mut m = archive.by_name(MANIFEST).map_err(|e| format!("{MANIFEST}: {e}"))?;
            let mut buf = Vec::new();
            m.read_to_end(&mut buf).map_err(|e| format!("{MANIFEST}: {e}"))?;
            buf
        };
        let manifest = parse_manifest(&manifest_data)?;

        fs::create_dir_all(&self.root).map_err(|e| format!("{}: {e}", self.root.display()))?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let staging = self.root.join(format!(".staging-{}-{nonce}", std::process::id()));
        if staging.exists() {
            fs::remove_dir_all(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
        }
        fs::create_dir_all(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;

        let extracted = (|| -> Result<(), String> {
            let mut total: u64 = 0;
            for i in 0..archive.len() {
                let mut entry = archive.by_index(i).map_err(|e| format!("zip entry {i}: {e}"))?;
                total = total.saturating_add(entry.size());
                if total > MAX_PACK_UNCOMPRESSED {
                    return Err(format!("pack expands past {MAX_PACK_UNCOMPRESSED} bytes"));
                }
                let Some(name) = entry.enclosed_name() else {
                    return Err(format!("zip entry {:?} escapes the pack", entry.name()));
                };
                let rel = name.to_path_buf();
                if rel.as_os_str().is_empty() {
                    continue;
                }
                let out = staging.join(&rel);
                if entry.is_dir() {
                    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
                    continue;
                }
                if let Some(parent) = out.parent() {
                    fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
                }
                let mut f = fs::File::create(&out).map_err(|e| format!("{}: {e}", out.display()))?;
                std::io::copy(&mut entry, &mut f).map_err(|e| format!("{}: {e}", out.display()))?;
                f.flush().map_err(|e| format!("{}: {e}", out.display()))?;
            }
            fs::write(staging.join(MANIFEST), &manifest_data)
                .map_err(|e| format!("{}: {e}", staging.join(MANIFEST).display()))?;
            Ok(())
        })();
        if let Err(e) = extracted {
            let _ = fs::remove_dir_all(&staging);
            return Err(e);
        }

        let sha = match verify_dir(&staging, &manifest) {
            Ok(s) => s,
            Err(e) => {
                let _ = fs::remove_dir_all(&staging);
                return Err(format!("pack verification failed: {e}"));
            }
        };
        if let Some(want) = expected_sha256
            && !sha.eq_ignore_ascii_case(want)
        {
            let _ = fs::remove_dir_all(&staging);
            return Err(format!("pack sha256 {sha} != expected {want}"));
        }

        let final_dir = self.root.join(&manifest.id);
        if final_dir.exists() {
            let same = load_marker(
                &self.root,
                &Marker { id: manifest.id.clone(), sha256: sha.clone() },
            )
            .is_ok();
            if !same {
                let _ = fs::remove_dir_all(&staging);
                return Err(format!("pack id {} already exists with different content", manifest.id));
            }
            let _ = fs::remove_dir_all(&staging);
        } else {
            fs::rename(&staging, &final_dir).map_err(|e| format!("{}: {e}", final_dir.display()))?;
        }

        let marker = Marker { id: manifest.id.clone(), sha256: sha };
        if let Some(current) = read_marker(&self.root, "active.json")
            && current.id != marker.id
        {
            write_marker(&self.root, "previous.json", &current)?;
        }
        write_marker(&self.root, "active.json", &marker)?;

        let mut out = self.resolve();
        out.status.restart_required = true;
        out.status.message = Some("Card pack installed. Restart the server to use it.".into());
        Ok(out)
    }

    /// Swap the active and previous packs (used when a newly installed pack turns out bad at runtime).
    pub fn rollback(&self) -> Result<Resolved, String> {
        let active = read_marker(&self.root, "active.json").ok_or("no active pack")?;
        let previous = read_marker(&self.root, "previous.json").ok_or("no previous pack to roll back to")?;
        let (dir, manifest, sha) = load_marker(&self.root, &previous)?;
        let marker = Marker { id: previous.id.clone(), sha256: sha };
        write_marker(&self.root, "previous.json", &active)?;
        write_marker(&self.root, "active.json", &marker)?;
        let mut status = status_from(&self.root, &marker, &manifest, "active", Vec::new());
        status.previous_id = Some(active.id);
        status.restart_required = true;
        status.message = Some("Rolled back to the previous card pack. Restart the server to use it.".into());
        Ok(Resolved { status, dir: Some(dir), manifest: Some(manifest) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(data: &[u8]) -> String {
        sha256_bytes(data)
    }

    fn cdb_bytes() -> Vec<u8> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let p = std::env::temp_dir().join(format!("dlpack-cdb-{}-{nonce}.cdb", std::process::id()));
        {
            let con = rusqlite::Connection::open(&p).unwrap();
            con.execute(
                "CREATE TABLE datas (id INT, alias INT, setcode INT, type INT, atk INT, def INT, level INT, race INT, attribute INT)",
                [],
            )
            .unwrap();
            con.execute("INSERT INTO datas VALUES (999999001, 0, 0, 33, 1000, 1000, 4, 1, 16)", []).unwrap();
            let cols = (1..=16).map(|i| format!("str{i} TEXT")).collect::<Vec<_>>().join(", ");
            con.execute(
                &format!("CREATE TABLE texts (id INT, name TEXT, desc TEXT, {cols})"),
                [],
            )
            .unwrap();
            con.execute(
                &format!(
                    "INSERT INTO texts VALUES ({})",
                    std::iter::repeat_n("?", 19).collect::<Vec<_>>().join(", ")
                ),
                rusqlite::params_from_iter(
                    std::iter::once("999999001".to_string())
                        .chain(std::iter::once("Pack Test".to_string()))
                        .chain(std::iter::once("Text.".to_string()))
                        .chain(std::iter::repeat_n(String::new(), 16)),
                ),
            )
            .unwrap();
        }
        let data = fs::read(&p).unwrap();
        let _ = fs::remove_file(&p);
        data
    }

    fn write_pack(path: &Path, id: &str, _cards: &[u8], script: Option<(&str, &[u8])>) {
        let cards = cdb_bytes();
        let mut files: Vec<(String, Vec<u8>)> = vec![("cards.cdb".into(), cards)];
        if let Some((name, data)) = script {
            files.push((format!("script/{name}"), data.to_vec()));
        }
        let entries: Vec<FileEntry> = files
            .iter()
            .map(|(p, d)| FileEntry { path: p.clone(), size: d.len() as u64, sha256: sha(d) })
            .collect();
        let manifest = Manifest {
            format: FORMAT.into(),
            id: id.into(),
            created: "2026-11-02T00:00:00Z".into(),
            base: serde_json::Value::Null,
            counts: serde_json::json!({"cards": 1, "scripts": script.is_some() as u64}),
            codes: vec![999999001],
            files: entries,
        };
        let f = fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default();
        z.start_file(MANIFEST, opts).unwrap();
        z.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        for (p, d) in files {
            z.start_file(p, opts).unwrap();
            z.write_all(&d).unwrap();
        }
        z.finish().unwrap();
    }

    #[test]
    fn installs_verifies_and_resolves() {
        let tmp = std::env::temp_dir().join(format!("dlpack-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let zip = tmp.join("p.zip");
        write_pack(&zip, "pack-a", b"sqlite-bytes", Some(("c999999001.lua", b"return true")));
        let store = PackStore::new(tmp.join("packs"));
        let r = store.install_zip(&zip, None).unwrap();
        assert_eq!(r.status.state, "active");
        assert_eq!(r.status.id.as_deref(), Some("pack-a"));
        assert!(r.status.restart_required);
        let again = store.resolve();
        assert_eq!(again.status.state, "active");
        assert_eq!(again.manifest.unwrap().codes, vec![999999001]);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn bad_hash_is_rejected_and_keeps_previous() {
        let tmp = std::env::temp_dir().join(format!("dlpack-bad-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let store = PackStore::new(tmp.join("packs"));
        let good = tmp.join("good.zip");
        write_pack(&good, "pack-a", b"good", None);
        store.install_zip(&good, None).unwrap();

        let bad = tmp.join("bad.zip");
        let data = cdb_bytes();
        let manifest = Manifest {
            format: FORMAT.into(),
            id: "pack-b".into(),
            created: String::new(),
            base: serde_json::Value::Null,
            counts: serde_json::Value::Null,
            codes: Vec::new(),
            files: vec![FileEntry { path: "cards.cdb".into(), size: data.len() as u64, sha256: sha(b"not-the-content") }],
        };
        let f = fs::File::create(&bad).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default();
        z.start_file(MANIFEST, opts).unwrap();
        z.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        z.start_file("cards.cdb", opts).unwrap();
        z.write_all(&data).unwrap();
        z.finish().unwrap();
        let err = store.install_zip(&bad, None).unwrap_err();
        assert!(err.contains("verification failed"), "{err}");
        assert_eq!(store.resolve().status.id.as_deref(), Some("pack-a"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn bad_active_pack_rolls_back_to_previous() {
        let tmp = std::env::temp_dir().join(format!("dlpack-rb-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let store = PackStore::new(tmp.join("packs"));
        let a = tmp.join("a.zip");
        write_pack(&a, "pack-a", b"a", None);
        store.install_zip(&a, None).unwrap();
        let b = tmp.join("b.zip");
        write_pack(&b, "pack-b", b"b", None);
        store.install_zip(&b, None).unwrap();
        assert_eq!(store.resolve().status.id.as_deref(), Some("pack-b"));
        fs::write(store.root().join("pack-b/cards.cdb"), b"tampered").unwrap();
        let r = store.resolve();
        assert_eq!(r.status.state, "rolled_back");
        assert_eq!(r.status.id.as_deref(), Some("pack-a"));
        assert!(r.status.alerts.iter().any(|a| a.contains("rejected")));
        assert_eq!(store.resolve().status.id.as_deref(), Some("pack-a"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_cdb_without_texts_is_rejected() {
        let tmp = std::env::temp_dir().join(format!("dlpack-notx-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let cdb_path = tmp.join("cards.cdb");
        {
            let con = rusqlite::Connection::open(&cdb_path).unwrap();
            con.execute(
                "CREATE TABLE datas (id INT, alias INT, setcode INT, type INT, atk INT, def INT, level INT, race INT, attribute INT)",
                [],
            )
            .unwrap();
        }
        let data = fs::read(&cdb_path).unwrap();
        let manifest = Manifest {
            format: FORMAT.into(),
            id: "pack-notx".into(),
            created: String::new(),
            base: serde_json::Value::Null,
            counts: serde_json::Value::Null,
            codes: Vec::new(),
            files: vec![FileEntry { path: "cards.cdb".into(), size: data.len() as u64, sha256: sha(&data) }],
        };
        let zip = tmp.join("p.zip");
        let f = fs::File::create(&zip).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default();
        z.start_file(MANIFEST, opts).unwrap();
        z.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        z.start_file("cards.cdb", opts).unwrap();
        z.write_all(&data).unwrap();
        z.finish().unwrap();
        let err = PackStore::new(tmp.join("packs")).install_zip(&zip, None).unwrap_err();
        assert!(err.contains("texts"), "{err}");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn unsafe_paths_and_wrong_format_are_rejected() {
        let tmp = std::env::temp_dir().join(format!("dlpack-fmt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let zip = tmp.join("x.zip");
        let mut manifest = Manifest {
            format: FORMAT.into(),
            id: "pack-x".into(),
            created: String::new(),
            base: serde_json::Value::Null,
            counts: serde_json::Value::Null,
            codes: Vec::new(),
            files: vec![FileEntry { path: "../escape".into(), size: 1, sha256: sha(b"x") }],
        };
        let f = fs::File::create(&zip).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default();
        z.start_file(MANIFEST, opts).unwrap();
        z.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        z.start_file("cards.cdb", opts).unwrap();
        z.write_all(b"x").unwrap();
        z.finish().unwrap();
        assert!(PackStore::new(tmp.join("packs")).install_zip(&zip, None).unwrap_err().contains("unsafe"));
        manifest.files[0].path = "cards.cdb".into();
        manifest.format = "other/9".into();
        let zip2 = tmp.join("y.zip");
        let f = fs::File::create(&zip2).unwrap();
        let mut z = zip::ZipWriter::new(f);
        z.start_file(MANIFEST, opts).unwrap();
        z.write_all(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        z.start_file("cards.cdb", opts).unwrap();
        z.write_all(b"x").unwrap();
        z.finish().unwrap();
        assert!(PackStore::new(tmp.join("packs")).install_zip(&zip2, None).unwrap_err().contains("format"));
        let _ = fs::remove_dir_all(&tmp);
    }
}
