//! The optional "describe a card" model download: Qwen3-Embedding-0.6B (fp16 ONNX), pinned by revision and
//! SHA-256. One download at a time on its own thread, resumable (an HTTP `Range` request against the `.part`
//! file), cancellable between chunks, and hashed while writing. `installed.json` marks a complete install.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use decklight_cards::emb::Emb;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Hugging Face revision of `onnx-community/Qwen3-Embedding-0.6B-ONNX` (NOTEBOOK 2026-11-13).
pub const REV: &str = "c25a394dd583836952667c12f008335071b3f43d";
/// Short revision shown in `describe_status`.
pub const REV_SHORT: &str = "c25a394";
const PROGRESS_EVERY: Duration = Duration::from_millis(500);
const FREE_MARGIN: u64 = 200 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct ModelFile {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const FILES: &[ModelFile] = &[
    ModelFile {
        path: "onnx/model_fp16.onnx",
        size: 583_522,
        sha256: "f376cf065d912675dd559270e19361367ebc28c938b9811dd61d5834c78d0834",
    },
    ModelFile {
        path: "onnx/model_fp16.onnx_data",
        size: 1_199_927_296,
        sha256: "06fbfded30166e3a33fe5252d7002ed1c370a4ec1bf45aba778cdf6cf7d31439",
    },
    ModelFile {
        path: "tokenizer.json",
        size: 11_423_705,
        sha256: "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    },
];

pub fn total_bytes() -> u64 {
    FILES.iter().map(|f| f.size).sum()
}

/// `<root>/models/qwen3-embed-0.6b-fp16/` (keeps the `onnx/` sub-path the fp16 graph references).
pub fn install_dir(root: &Path) -> PathBuf {
    root.join("models/qwen3-embed-0.6b-fp16")
}

fn url(path: &str) -> String {
    format!("https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/{REV}/{path}")
}

fn file_path(dir: &Path, path: &str) -> PathBuf {
    dir.join(path)
}

fn part_path(dir: &Path, path: &str) -> PathBuf {
    dir.join(format!("{path}.part"))
}

fn bytes_on_disk(dir: &Path) -> u64 {
    let mut n = 0u64;
    for f in FILES {
        for p in [file_path(dir, f.path), part_path(dir, f.path)] {
            if let Ok(m) = std::fs::metadata(&p) {
                n += m.len();
            }
        }
    }
    n.min(total_bytes())
}

fn free_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    let c = CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

/// One HTTP response used by the downloader. `ranged` is false when the server ignored a `Range` request
/// (HTTP 200): the caller then starts that file from byte 0 again.
pub struct DownloadResponse {
    pub reader: Box<dyn Read + Send>,
    pub ranged: bool,
}

pub trait Downloader: Send + Sync {
    fn fetch(&self, url: &str, start: u64) -> Result<DownloadResponse, String>;
}

pub struct HttpDownloader;

impl Downloader for HttpDownloader {
    fn fetch(&self, url: &str, start: u64) -> Result<DownloadResponse, String> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(30))
            .user_agent("decklight/1.0")
            .redirects(5)
            .build();
        let mut req = agent.get(url);
        if start > 0 {
            req = req.set("Range", &format!("bytes={start}-"));
        }
        let resp = req.call().map_err(|e| match e {
            ureq::Error::Status(code, _) => format!("HTTP {code}"),
            ureq::Error::Transport(t) => t.to_string(),
        })?;
        let ranged = start > 0 && resp.status() == 206;
        Ok(DownloadResponse { reader: resp.into_reader(), ranged })
    }
}

/// Download (or resume) one pinned file into `<dir>/<path>`, then atomically move the `.part` into place.
/// Returns `Ok(false)` when cancelled (the `.part` is kept for a later resume).
pub fn download_one(
    dir: &Path,
    spec: &ModelFile,
    url: &str,
    source: &dyn Downloader,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(),
) -> Result<bool, String> {
    let final_path = file_path(dir, spec.path);
    if let Ok(meta) = std::fs::metadata(&final_path) {
        if meta.len() == spec.size {
            return Ok(true);
        }
        let _ = std::fs::remove_file(&final_path);
    }
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let part = part_path(dir, spec.path);
    let mut start = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if start > spec.size {
        let _ = std::fs::remove_file(&part);
        start = 0;
    }

    // Hash the bytes already on disk so a resumed file is verified over its whole length.
    let mut hasher = Sha256::new();
    if start > 0 {
        let mut f = std::fs::File::open(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", part.display()))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }

    if start == spec.size {
        // Crashed after the last byte but before the atomic rename: only the hash is left.
        let got = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();
        if got != spec.sha256 {
            let _ = std::fs::remove_file(&part);
            return Err(format!("{}: sha256 mismatch (got {got})", spec.path));
        }
        std::fs::rename(&part, &final_path).map_err(|e| format!("{}: {e}", final_path.display()))?;
        return Ok(true);
    }

    let mut resp = source.fetch(url, start)?;
    if start > 0 && !resp.ranged {
        // The server sent the whole file: restart this one from the beginning.
        start = 0;
        hasher = Sha256::new();
        let _ = std::fs::remove_file(&part);
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part)
        .map_err(|e| format!("{}: {e}", part.display()))?;
    let mut written = start;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        if cancel.load(Ordering::SeqCst) {
            file.flush().map_err(|e| e.to_string())?;
            return Ok(false);
        }
        let n = resp.reader.read(&mut buf).map_err(|e| format!("{}: {e}", spec.path))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", part.display()))?;
        hasher.update(&buf[..n]);
        written += n as u64;
        if written > spec.size {
            drop(file);
            let _ = std::fs::remove_file(&part);
            return Err(format!("{}: server sent more than {} bytes", spec.path, spec.size));
        }
        progress();
    }
    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    if written != spec.size {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{}: got {written} of {} bytes", spec.path, spec.size));
    }
    let got = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>();
    if got != spec.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{}: sha256 mismatch (got {got})", spec.path));
    }
    std::fs::rename(&part, &final_path).map_err(|e| format!("{}: {e}", final_path.display()))?;
    Ok(true)
}

/// Download/status state shared by the server and the download thread.
pub struct DescribeState {
    pub dir: PathBuf,
    downloading: AtomicBool,
    cancel: AtomicBool,
    error: Mutex<Option<String>>,
    bytes: AtomicU64,
}

impl DescribeState {
    pub fn new(root: &Path) -> DescribeState {
        let dir = install_dir(root);
        let bytes = bytes_on_disk(&dir);
        DescribeState {
            dir,
            downloading: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            error: Mutex::new(None),
            bytes: AtomicU64::new(bytes),
        }
    }

    pub fn installed(&self) -> bool {
        let p = self.dir.join("installed.json");
        let Ok(text) = std::fs::read_to_string(&p) else { return false };
        let Ok(v) = serde_json::from_str::<Value>(&text) else { return false };
        if v.get("rev").and_then(Value::as_str) != Some(REV) {
            return false;
        }
        FILES.iter().all(|f| {
            std::fs::metadata(file_path(&self.dir, f.path))
                .map(|m| m.len() == f.size)
                .unwrap_or(false)
        })
    }

    pub fn is_downloading(&self) -> bool {
        self.downloading.load(Ordering::SeqCst)
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    pub fn refresh_bytes(&self) -> u64 {
        let n = bytes_on_disk(&self.dir);
        self.bytes.store(n, Ordering::SeqCst);
        n
    }

    pub fn status(&self) -> Value {
        let bytes = self.refresh_bytes();
        json!({
            "t": "describe_status",
            "installed": self.installed(),
            "downloading": self.is_downloading(),
            "bytes": bytes,
            "total": total_bytes(),
            "error": self.error(),
            "rev": REV_SHORT,
        })
    }

    /// Start (or resume) the download. A second call while running is a no-op. The sink receives status
    /// frames at most every 500 ms and once at the end.
    pub fn start(self: &Arc<Self>, sink: Arc<dyn Fn(Value) + Send + Sync>) -> Result<(), String> {
        if self.installed() || self.is_downloading() {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        let have = bytes_on_disk(&self.dir);
        let remaining = total_bytes().saturating_sub(have);
        if let Some(free) = free_bytes(&self.dir) {
            if free < remaining + FREE_MARGIN {
                let need = (remaining + FREE_MARGIN) as f64 / 1e9;
                return Err(format!("Not enough free space: needs {need:.1} GB"));
            }
        }
        self.cancel.store(false, Ordering::SeqCst);
        *self.error.lock().unwrap() = None;
        self.downloading.store(true, Ordering::SeqCst);
        let st = self.clone();
        let spawned = std::thread::Builder::new().name("describe-download".into()).spawn(move || {
            let result = st.run(&HttpDownloader, &*sink);
            if let Err(e) = &result {
                *st.error.lock().unwrap() = Some(e.clone());
            }
            st.downloading.store(false, Ordering::SeqCst);
            st.refresh_bytes();
            sink(st.status());
        });
        if spawned.is_err() {
            self.downloading.store(false, Ordering::SeqCst);
            return Err("could not start the model download".into());
        }
        Ok(())
    }

    fn run(&self, source: &dyn Downloader, sink: &(dyn Fn(Value) + Send + Sync)) -> Result<(), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        let mut last = Instant::now();
        let mut progress = || {
            if last.elapsed() >= PROGRESS_EVERY {
                last = Instant::now();
                sink(self.status());
            }
        };
        for spec in FILES {
            if !download_one(&self.dir, spec, &url(spec.path), source, &self.cancel, &mut progress)? {
                return Ok(());
            }
            progress();
        }
        let installed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let files: Vec<&str> = FILES.iter().map(|f| f.path).collect();
        let manifest = json!({"rev": REV, "files": files, "installed": installed});
        let path = self.dir.join("installed.json");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(())
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn remove(&self) {
        self.cancel();
        // Let a running download notice the cancel and close its .part before the directory goes (up to 5 s).
        for _ in 0..50 {
            if !self.downloading.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = std::fs::remove_dir_all(&self.dir);
        self.bytes.store(0, Ordering::SeqCst);
        *self.error.lock().unwrap() = None;
    }
}

const ENCODER_IDLE: Duration = Duration::from_secs(180);

/// The installed model as the server uses it: download state, the lazily loaded card embedding table and
/// the query encoder (dropped after 3 minutes idle; it is ~0.9 GB on the phone).
pub struct Describe {
    pub state: Arc<DescribeState>,
    emb_path: PathBuf,
    emb: Arc<Mutex<Option<Arc<Emb>>>>,
    enc: Arc<Mutex<Option<(crate::describe_enc::Encoder, Instant)>>>,
}

impl Describe {
    pub fn new(root: &Path) -> Describe {
        let emb_path = std::env::var_os("YGO_EMB")
            .map(PathBuf::from)
            .filter(|p| p.is_file())
            .unwrap_or_else(|| root.join("data/cards/emb/qwen3-0.6b-d512.npz"));
        let enc: Arc<Mutex<Option<(crate::describe_enc::Encoder, Instant)>>> = Arc::new(Mutex::new(None));
        let emb: Arc<Mutex<Option<Arc<Emb>>>> = Arc::new(Mutex::new(None));
        // The fp16 encoder (~0.9 GB) and the f32 effect table (~270 MB) are released together after 3 idle minutes,
        // even if no query arrives to notice.
        {
            let enc = enc.clone();
            let emb = emb.clone();
            let _ = std::thread::Builder::new().name("describe-idle".into()).spawn(move || loop {
                std::thread::sleep(Duration::from_secs(30));
                let mut guard = enc.lock().unwrap();
                if guard.as_ref().is_some_and(|(_, t)| t.elapsed() >= ENCODER_IDLE) {
                    *guard = None;
                    *emb.lock().unwrap() = None;
                }
            });
        }
        Describe {
            state: Arc::new(DescribeState::new(root)),
            emb_path,
            emb,
            enc,
        }
    }

    pub fn embeddings(&self) -> Result<Arc<Emb>, String> {
        let mut guard = self.emb.lock().unwrap();
        if let Some(e) = guard.as_ref() {
            return Ok(e.clone());
        }
        let e = Arc::new(Emb::load(&self.emb_path)?);
        *guard = Some(e.clone());
        Ok(e)
    }

    fn drop_idle(&self) {
        let mut guard = self.enc.lock().unwrap();
        if guard.as_ref().is_some_and(|(_, t)| t.elapsed() >= ENCODER_IDLE) {
            *guard = None;
        }
    }

    /// Encode one query (the caller passes `INSTRUCT + text`).
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, String> {
        self.drop_idle();
        let mut guard = self.enc.lock().unwrap();
        if guard.is_none() {
            let enc = crate::describe_enc::Encoder::load(&self.state.dir)?;
            *guard = Some((enc, Instant::now()));
        }
        let (enc, used) = guard.as_mut().unwrap();
        let v = enc.encode(text, 512)?;
        *used = Instant::now();
        Ok(v)
    }

    pub fn drop_encoder(&self) {
        *self.enc.lock().unwrap() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    struct VecSource {
        data: Vec<u8>,
        honor_range: bool,
        calls: AtomicU64,
    }

    impl Downloader for VecSource {
        fn fetch(&self, _url: &str, start: u64) -> Result<DownloadResponse, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if start > 0 && self.honor_range {
                Ok(DownloadResponse {
                    reader: Box::new(std::io::Cursor::new(self.data[start as usize..].to_vec())),
                    ranged: true,
                })
            } else {
                Ok(DownloadResponse { reader: Box::new(std::io::Cursor::new(self.data.clone())), ranged: false })
            }
        }
    }

    fn sha(data: &[u8]) -> String {
        Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "decklight-describe-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn spec(data: &[u8]) -> ModelFile {
        // Leak so the &'static str fields live for the test (tiny, test-only).
        let path: &'static str = Box::leak("file.bin".to_string().into_boxed_str());
        let sha: &'static str = Box::leak(sha(data).into_boxed_str());
        ModelFile { path, size: data.len() as u64, sha256: sha }
    }

    #[test]
    fn resumes_a_part_with_a_range_and_verifies_sha() {
        let dir = temp_dir("resume");
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let s = spec(&data);
        std::fs::write(part_path(&dir, s.path), &data[..40_000]).unwrap();
        let src = VecSource { data: data.clone(), honor_range: true, calls: AtomicU64::new(0) };
        let cancel = AtomicBool::new(false);
        let ok = download_one(&dir, &s, "http://test/x", &src, &cancel, &mut || {}).unwrap();
        assert!(ok);
        assert_eq!(std::fs::read(file_path(&dir, s.path)).unwrap(), data);
        assert!(!part_path(&dir, s.path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restarts_when_the_server_ignores_the_range() {
        let dir = temp_dir("range");
        let data: Vec<u8> = (0..30_000u32).map(|i| (i % 97) as u8).collect();
        let s = spec(&data);
        std::fs::write(part_path(&dir, s.path), &data[..10_000]).unwrap();
        let src = VecSource { data: data.clone(), honor_range: false, calls: AtomicU64::new(0) };
        let cancel = AtomicBool::new(false);
        assert!(download_one(&dir, &s, "http://test/x", &src, &cancel, &mut || {}).unwrap());
        assert_eq!(std::fs::read(file_path(&dir, s.path)).unwrap(), data);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_complete_part_is_renamed_without_a_fetch() {
        let dir = temp_dir("complete");
        let data: Vec<u8> = (0..8_000u32).map(|i| (i % 41) as u8).collect();
        let s = spec(&data);
        std::fs::write(part_path(&dir, s.path), &data).unwrap();
        let src = VecSource { data: data.clone(), honor_range: true, calls: AtomicU64::new(0) };
        let cancel = AtomicBool::new(false);
        assert!(download_one(&dir, &s, "http://test/x", &src, &cancel, &mut || {}).unwrap());
        assert_eq!(src.calls.load(Ordering::SeqCst), 0, "no request when the .part is complete");
        assert_eq!(std::fs::read(file_path(&dir, s.path)).unwrap(), data);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_sha_mismatch_removes_the_part() {
        let dir = temp_dir("sha");
        let data = vec![7u8; 5000];
        let mut s = spec(&data);
        s.sha256 = "00";
        let src = VecSource { data: data.clone(), honor_range: true, calls: AtomicU64::new(0) };
        let cancel = AtomicBool::new(false);
        let err = download_one(&dir, &s, "http://test/x", &src, &cancel, &mut || {}).unwrap_err();
        assert!(err.contains("sha256 mismatch"), "{err}");
        assert!(!part_path(&dir, s.path).exists());
        assert!(!file_path(&dir, s.path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_has_the_documented_fields() {
        let root = temp_dir("status");
        let st = DescribeState::new(&root);
        let v = st.status();
        for key in ["t", "installed", "downloading", "bytes", "total", "error", "rev"] {
            assert!(v.get(key).is_some(), "missing {key}: {v}");
        }
        assert_eq!(v["total"].as_u64(), Some(total_bytes()));
        assert_eq!(v["installed"], json!(false));
        let _ = std::fs::remove_dir_all(&root);
    }
}
