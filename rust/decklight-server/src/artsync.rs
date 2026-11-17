//! Background card-art sync: fetch every playable card's small image once per install, in the user's order
//! (the current duel and the deck catalog first), over a shared throttled HTTP path, and keep only the missing
//! list between runs. The Android app freezes the process when backgrounded; `.part` files and a rescan on the
//! next start lose nothing.
//!
//! The sync is idle until the client enables it (card art on and not paused), waits on cellular until the user
//! allows mobile data once per install, stops when free space is low, and yields every request to `/img`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::CString;
use std::fs;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::images::{ImgThrottle, MIN_JPEG_BYTES};

/// A 404 is not retried for seven days (or until the card pack / app assets change).
pub const MISSING_TTL_SECS: u64 = 7 * 24 * 3600;
pub const MIN_FREE_BYTES: u64 = 1024 * 1024 * 1024;
const SPACE_CHECK_EVERY: u64 = 200;
const SPACE_RECHECK: Duration = Duration::from_secs(300);
const LOW_SPACE_RECHECK: Duration = Duration::from_secs(5);
const TEMP_RETRY: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Network {
    Wifi,
    Cellular,
    Unknown,
}

impl Network {
    pub fn parse(s: &str) -> Network {
        match s {
            "cellular" => Network::Cellular,
            "wifi" => Network::Wifi,
            _ => Network::Unknown,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Network::Wifi => "wifi",
            Network::Cellular => "cellular",
            Network::Unknown => "unknown",
        }
    }
}

#[derive(Clone)]
pub struct ArtSyncOptions {
    pub pics_dir: PathBuf,
    /// `.../images/cards_small` (the test hook points this at a local server).
    pub small_base: String,
    pub target: BTreeSet<u32>,
    /// Codes from the current duel and the deck catalog, fetched before the rest.
    pub priority: Vec<u32>,
    /// Card pack id + app assets id: a change expires the missing list.
    pub tokens: String,
    pub min_free: u64,
}

struct State {
    enabled: bool,
    paused: bool,
    network: Network,
    allow_data: bool,
    tokens: String,
    target: BTreeSet<u32>,
    remaining: BTreeSet<u32>,
    priority: Vec<u32>,
    missing: HashMap<u32, u64>,
    retry_after: HashMap<u32, Instant>,
    have: u64,
    bytes: u64,
    unavailable: u64,
    state: &'static str,
    rate: f64,
    rate_start: Instant,
    rate_files: u64,
    last_emit: Instant,
    next_space_check: Instant,
    space_blocked: bool,
    files_since_space: u64,
    sink: Option<tokio::sync::mpsc::UnboundedSender<Value>>,
}

pub struct ArtSync {
    pics_dir: PathBuf,
    small_base: String,
    min_free: u64,
    throttle: Arc<ImgThrottle>,
    state: Mutex<State>,
    cv: Condvar,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn valid_cached(path: &Path) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    if meta.len() < MIN_JPEG_BYTES as u64 {
        let _ = fs::remove_file(path);
        return false;
    }
    let mut head = [0u8; 2];
    let ok = fs::File::open(path).and_then(|mut f| f.read_exact(&mut head)).is_ok();
    if !ok || head != [0xFF, 0xD8] {
        let _ = fs::remove_file(path);
        return false;
    }
    true
}

fn free_bytes(path: &Path) -> Option<u64> {
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0 {
        Some(s.f_bavail as u64 * s.f_frsize as u64)
    } else {
        None
    }
}

fn atomic_write(path: &Path, data: &[u8]) {
    let tmp = path.with_extension("tmp");
    if fs::write(&tmp, data).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

impl ArtSync {
    pub fn start(opts: ArtSyncOptions, throttle: Arc<ImgThrottle>) -> Arc<ArtSync> {
        let _ = fs::create_dir_all(opts.pics_dir.join("small"));
        let _ = fs::create_dir_all(opts.pics_dir.join("full"));

        let mut missing: HashMap<u32, u64> = fs::read_to_string(opts.pics_dir.join("missing.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<BTreeMap<String, u64>>(&s).ok())
            .map(|m| m.into_iter().filter_map(|(k, t)| k.parse().ok().map(|c| (c, t))).collect())
            .unwrap_or_default();

        let sync: SyncFile = fs::read_to_string(opts.pics_dir.join("sync.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if !sync.tokens.is_empty() && sync.tokens != opts.tokens {
            // The pack or the app's assets changed: cards that were unavailable may exist now.
            missing.clear();
        }
        let cutoff = now_secs().saturating_sub(MISSING_TTL_SECS);
        missing.retain(|_c, t| *t >= cutoff);

        let mut have = 0u64;
        let mut bytes = 0u64;
        let mut remaining = BTreeSet::new();
        for &code in &opts.target {
            let path = opts.pics_dir.join("small").join(format!("{code}.jpg"));
            if valid_cached(&path) {
                have += 1;
                bytes += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                missing.remove(&code);
            } else if missing.contains_key(&code) {
                // counted as unavailable below
            } else {
                remaining.insert(code);
            }
        }
        let unavailable = missing.keys().filter(|c| opts.target.contains(c)).count() as u64;
        let priority: Vec<u32> = {
            let mut seen = BTreeSet::new();
            opts.priority
                .into_iter()
                .filter(|c| opts.target.contains(c) && seen.insert(*c))
                .collect()
        };
        let art = Arc::new(ArtSync {
            pics_dir: opts.pics_dir,
            small_base: opts.small_base,
            min_free: opts.min_free,
            throttle,
            state: Mutex::new(State {
                enabled: false,
                paused: true,
                network: Network::Unknown,
                allow_data: sync.allow_mobile,
                tokens: opts.tokens,
                target: opts.target,
                remaining,
                priority,
                missing,
                retry_after: HashMap::new(),
                have,
                bytes,
                unavailable,
                state: "idle",
                rate: 0.0,
                rate_start: Instant::now(),
                rate_files: 0,
                last_emit: Instant::now() - Duration::from_secs(2),
                next_space_check: Instant::now(),
                space_blocked: false,
                files_since_space: 0,
                sink: None,
            }),
            cv: Condvar::new(),
        });
        art.save_sync();
        art.save_missing();
        let (total, have, unavailable) = {
            let st = art.state.lock().unwrap();
            (st.target.len(), st.have, st.unavailable)
        };
        println!("art-sync: {total} target codes ({have} saved, {unavailable} unavailable)");
        let a = art.clone();
        std::thread::Builder::new()
            .name("art-sync".into())
            .spawn(move || a.run())
            .expect("could not start the art-sync thread");
        art
    }

    fn small_path(&self, code: u32) -> PathBuf {
        self.pics_dir.join("small").join(format!("{code}.jpg"))
    }

    fn save_missing(&self) {
        let map: BTreeMap<String, u64> = {
            let st = self.state.lock().unwrap();
            st.missing.iter().map(|(c, t)| (c.to_string(), *t)).collect()
        };
        let data = serde_json::to_vec(&map).unwrap_or_default();
        atomic_write(&self.pics_dir.join("missing.json"), &data);
    }

    fn save_sync(&self) {
        let (allow, tokens) = {
            let st = self.state.lock().unwrap();
            (st.allow_data, st.tokens.clone())
        };
        let data = serde_json::to_vec(&json!({"allow_mobile": allow, "tokens": tokens})).unwrap_or_default();
        atomic_write(&self.pics_dir.join("sync.json"), &data);
    }

    fn wake(&self) {
        self.cv.notify_all();
    }

    /// The client's current progress (also sent once on connect).
    pub fn status(&self) -> Value {
        let st = self.state.lock().unwrap();
        self.snapshot(&st)
    }

    fn snapshot(&self, st: &State) -> Value {
        let resume_in = self
            .throttle
            .paused_until()
            .map(|u| u.saturating_duration_since(Instant::now()).as_secs())
            .unwrap_or(0);
        json!({
            "t": "art_sync",
            "state": st.state,
            "have": st.have,
            "unavailable": st.unavailable,
            "total": st.target.len(),
            "bytes": st.bytes,
            "rate_per_s": (st.rate * 10.0).round() / 10.0,
            "network": st.network.as_str(),
            "allow_data": st.allow_data,
            "resume_in_s": resume_in,
        })
    }

    fn emit(&self, st: &mut State, force: bool) {
        let now = Instant::now();
        if !force && st.state == "running" && now.duration_since(st.last_emit) < Duration::from_secs(1) {
            return;
        }
        st.last_emit = now;
        let frame = self.snapshot(st);
        if let Some(tx) = st.sink.as_ref() {
            let _ = tx.send(frame);
        }
    }

    pub fn set_sink(&self, sink: Option<tokio::sync::mpsc::UnboundedSender<Value>>) {
        let mut st = self.state.lock().unwrap();
        st.sink = sink;
        self.emit(&mut st, true);
    }

    pub fn set_network(&self, network: Network) {
        let mut st = self.state.lock().unwrap();
        st.network = network;
        self.emit(&mut st, true);
        drop(st);
        self.wake();
    }

    pub fn pause(&self) {
        let mut st = self.state.lock().unwrap();
        st.enabled = true;
        st.paused = true;
        self.emit(&mut st, true);
        drop(st);
        self.wake();
    }

    pub fn resume(&self) {
        let mut st = self.state.lock().unwrap();
        st.enabled = true;
        st.paused = false;
        st.rate_start = Instant::now();
        st.rate_files = 0;
        st.rate = 0.0;
        self.emit(&mut st, true);
        drop(st);
        self.wake();
    }

    pub fn allow_data(&self) {
        let mut st = self.state.lock().unwrap();
        st.allow_data = true;
        self.emit(&mut st, true);
        drop(st);
        self.save_sync();
        self.wake();
    }

    /// "Delete saved art": remove both sizes and the missing list, and stay stopped until resumed.
    pub fn delete_all(&self) {
        let mut st = self.state.lock().unwrap();
        let _ = fs::remove_dir_all(self.pics_dir.join("small"));
        let _ = fs::remove_dir_all(self.pics_dir.join("full"));
        let _ = fs::remove_file(self.pics_dir.join("missing.json"));
        let _ = fs::create_dir_all(self.pics_dir.join("small"));
        let _ = fs::create_dir_all(self.pics_dir.join("full"));
        st.missing.clear();
        st.retry_after.clear();
        st.remaining = st.target.clone();
        st.have = 0;
        st.bytes = 0;
        st.unavailable = 0;
        st.paused = true;
        st.enabled = true;
        st.state = "paused";
        st.rate = 0.0;
        self.emit(&mut st, true);
        drop(st);
        self.save_missing();
        self.wake();
    }

    /// Codes of a duel that just started go to the front of the queue.
    pub fn note_deck(&self, codes: &[u32]) {
        let mut st = self.state.lock().unwrap();
        let mut at_front = Vec::new();
        for &c in codes {
            if st.target.contains(&c) && !st.priority.contains(&c) {
                st.priority.insert(0, c);
                at_front.push(c);
            }
        }
        if !at_front.is_empty() {
            self.wake();
        }
    }

    pub fn is_missing(&self, code: u32) -> bool {
        let st = self.state.lock().unwrap();
        st.missing
            .get(&code)
            .is_some_and(|t| *t >= now_secs().saturating_sub(MISSING_TTL_SECS))
    }

    /// `/img` found a cached file that fails the JPEG check and deleted it: it is missing again.
    pub fn file_removed(&self, code: u32) {
        let mut st = self.state.lock().unwrap();
        if st.target.contains(&code) {
            st.have = st.have.saturating_sub(1);
            if !st.missing.contains_key(&code) {
                st.remaining.insert(code);
            }
        }
        drop(st);
        self.wake();
    }

    /// `/img` learned that the upstream has no image: keep it out of the bulk retries too.
    pub fn record_missing(&self, code: u32) {
        let now = now_secs();
        let cutoff = now.saturating_sub(MISSING_TTL_SECS);
        let mut st = self.state.lock().unwrap();
        if st.missing.get(&code).is_some_and(|t| *t >= cutoff) {
            return;
        }
        st.missing.insert(code, now);
        if st.target.contains(&code) {
            if !st.remaining.remove(&code) {
                // It was counted as saved, but the file is gone (the caller removed it).
                st.have = st.have.saturating_sub(1);
            }
            st.unavailable += 1;
        }
        drop(st);
        self.save_missing();
        self.wake();
    }

    /// `/img` fetched a card that the bulk sync still had to fetch (or had given up on).
    pub fn recovered(&self, code: u32, bytes: u64) {
        let mut st = self.state.lock().unwrap();
        if !st.target.contains(&code) {
            if st.missing.remove(&code).is_some() {
                drop(st);
                self.save_missing();
            }
            return;
        }
        let was_missing = st.missing.remove(&code).is_some();
        let was_remaining = st.remaining.remove(&code);
        if was_missing || was_remaining {
            if was_missing {
                st.unavailable = st.unavailable.saturating_sub(1);
            }
            st.have += 1;
            st.bytes += bytes;
        }
        drop(st);
        if was_missing {
            self.save_missing();
        }
        self.wake();
    }

    fn classify(&self, st: &mut State) {
        if !st.enabled {
            st.state = "idle";
            st.rate = 0.0;
            return;
        }
        if st.paused {
            st.state = "paused";
            st.rate = 0.0;
            return;
        }
        if st.network == Network::Cellular && !st.allow_data {
            st.state = "waiting_wifi";
            st.rate = 0.0;
            return;
        }
        if self.throttle.paused_until().is_some() {
            st.state = "rate_limited";
            st.rate = 0.0;
            return;
        }
        let now = Instant::now();
        if now >= st.next_space_check || st.files_since_space >= SPACE_CHECK_EVERY {
            st.files_since_space = 0;
            st.space_blocked = free_bytes(&self.pics_dir).is_some_and(|free| free < self.min_free);
            st.next_space_check = now + if st.space_blocked { LOW_SPACE_RECHECK } else { SPACE_RECHECK };
        }
        if st.space_blocked {
            st.state = "low_space";
            st.rate = 0.0;
            return;
        }
        if st.remaining.is_empty() {
            st.state = "done";
            st.rate = 0.0;
            return;
        }
        st.state = "running";
    }

    fn next_code(&self, st: &State) -> Option<u32> {
        let now = Instant::now();
        for &c in &st.priority {
            if st.remaining.contains(&c) && !st.retry_after.get(&c).is_some_and(|t| now < *t) {
                return Some(c);
            }
        }
        st.remaining
            .iter()
            .find(|c| !st.retry_after.get(c).is_some_and(|t| now < *t))
            .copied()
    }

    fn run(self: Arc<Self>) {
        let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
            Ok(rt) => rt,
            Err(e) => {
                let mut st = self.state.lock().unwrap();
                st.state = "error";
                eprintln!("ALERT art-sync: could not start its runtime: {e}");
                self.emit(&mut st, true);
                return;
            }
        };
        loop {
            let mut st = self.state.lock().unwrap();
            self.classify(&mut st);
            let wait = if st.state == "running" {
                match self.next_code(&st) {
                    Some(code) => {
                        drop(st);
                        self.fetch_one(&rt, code);
                        continue;
                    }
                    None => {
                        // Only temporary failures are left: retry after their back-off.
                        st.state = "error";
                        Duration::from_secs(30)
                    }
                }
            } else {
                match st.state {
                    "done" => Duration::from_secs(60),
                    "low_space" => LOW_SPACE_RECHECK,
                    "rate_limited" => Duration::from_secs(5),
                    _ => Duration::from_secs(1),
                }
            };
            self.emit(&mut st, false);
            let (_guard, _) = self.cv.wait_timeout(st, wait).unwrap();
        }
    }

    fn fetch_one(&self, rt: &tokio::runtime::Runtime, code: u32) {
        let path = self.small_path(code);
        if valid_cached(&path) {
            let len = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mut st = self.state.lock().unwrap();
            if st.remaining.remove(&code) {
                st.have += 1;
                st.bytes += len;
            }
            self.emit(&mut st, false);
            return;
        }
        let url = format!("{}/{code}.jpg", self.small_base.trim_end_matches('/'));
        let outcome = rt.block_on(self.throttle.fetch_bulk(&url));
        let mut st = self.state.lock().unwrap();
        if outcome.pause {
            st.state = "rate_limited";
            st.rate = 0.0;
            self.emit(&mut st, true);
            return;
        }
        if let Some(data) = outcome.data {
            let Ok(dir) = fs::create_dir_all(self.pics_dir.join("small")) else {
                return;
            };
            let _ = dir;
            let part = self.pics_dir.join("small").join(format!("{code}.jpg.part"));
            let ok = fs::write(&part, &data).is_ok() && valid_cached(&part);
            if ok && fs::rename(&part, &path).is_ok() {
                if st.remaining.remove(&code) {
                    st.have += 1;
                    st.bytes += data.len() as u64;
                }
                if st.missing.remove(&code).is_some() {
                    st.unavailable = st.unavailable.saturating_sub(1);
                }
                let now = Instant::now();
                if st.rate_files == 0 {
                    st.rate_start = now;
                }
                st.rate_files += 1;
                let secs = now.duration_since(st.rate_start).as_secs_f64().max(0.001);
                st.rate = st.rate_files as f64 / secs;
                st.files_since_space += 1;
            } else {
                let _ = fs::remove_file(&part);
                st.retry_after.insert(code, Instant::now() + TEMP_RETRY);
            }
            self.emit(&mut st, false);
            return;
        }
        if outcome.not_found {
            st.missing.insert(code, now_secs());
            if st.remaining.remove(&code) {
                st.unavailable += 1;
            }
            drop(st);
            self.save_missing();
            let mut st = self.state.lock().unwrap();
            self.emit(&mut st, false);
        } else {
            st.retry_after.insert(code, Instant::now() + TEMP_RETRY);
        }
    }
}

#[derive(Default, serde::Deserialize, serde::Serialize)]
struct SyncFile {
    #[serde(default)]
    allow_mobile: bool,
    #[serde(default)]
    tokens: String,
}

/// Pull the codes out of every `.ydk` under the given deck directories (the catalog plus the gauntlet).
pub fn deck_codes(root: &Path, dirs: &[(String, PathBuf)]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut all: Vec<PathBuf> = dirs.iter().map(|(_, p)| p.clone()).collect();
    all.push(root.join("decks/gauntlet"));
    all.push(root.join("decks/custom"));
    for dir in all {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("ydk"))
            .collect();
        paths.sort();
        for p in paths {
            let Ok(text) = fs::read_to_string(&p) else {
                continue;
            };
            let d = crate::decks::parse_ydk_named(&text, "");
            for c in d.main.into_iter().chain(d.extra).chain(d.side) {
                if seen.insert(c) {
                    out.push(c);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::FetchOutcome;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempRoot(PathBuf);
    impl TempRoot {
        fn new(tag: &str) -> TempRoot {
            let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
            let p = std::env::temp_dir().join(format!("decklight-art-{tag}-{}-{n}", std::process::id()));
            fs::create_dir_all(&p).unwrap();
            TempRoot(p)
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn jpeg() -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8];
        v.resize(MIN_JPEG_BYTES, 3);
        v
    }

    fn opts(root: &TempRoot, target: &[u32], priority: &[u32], tokens: &str) -> ArtSyncOptions {
        ArtSyncOptions {
            pics_dir: root.0.clone(),
            small_base: "http://127.0.0.1:9/images/cards_small".into(),
            target: target.iter().copied().collect(),
            priority: priority.to_vec(),
            tokens: tokens.to_string(),
            min_free: 0,
        }
    }

    fn throttle(f: impl Fn(&str) -> FetchOutcome + Send + Sync + 'static) -> Arc<ImgThrottle> {
        Arc::new(ImgThrottle::new(Arc::new(f)))
    }

    fn wait_state(art: &ArtSync, want: &str, timeout: Duration) -> bool {
        let t0 = Instant::now();
        while t0.elapsed() < timeout {
            if art.status()["state"].as_str() == Some(want) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        false
    }

    #[test]
    fn deck_codes_come_first_and_present_files_are_skipped() {
        let root = TempRoot::new("order");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let art = ArtSync::start(
            opts(&root, &[1, 2, 3, 4, 5], &[4, 2], "t"),
            throttle(move |u: &str| {
                let code: u32 = u.rsplit('/').next().unwrap().strip_suffix(".jpg").unwrap().parse().unwrap();
                s.lock().unwrap().push(code);
                FetchOutcome { data: Some(jpeg()), ..FetchOutcome::default() }
            }),
        );
        std::fs::write(root.0.join("small/3.jpg"), jpeg()).unwrap();
        art.resume();
        art.set_network(Network::Wifi);
        assert!(wait_state(&art, "done", Duration::from_secs(10)));
        let seen = seen.lock().unwrap().clone();
        assert_eq!(&seen[..2], &[4, 2], "deck codes first");
        assert!(!seen.contains(&3), "an existing file is not fetched");
        assert_eq!(art.status()["have"], 5);
    }

    #[test]
    fn part_files_are_renamed_and_bad_bodies_rejected() {
        let root = TempRoot::new("part");
        let art = ArtSync::start(
            opts(&root, &[7], &[], "t"),
            throttle(|_u: &str| FetchOutcome { data: Some(b"<html>".to_vec()), ..FetchOutcome::default() }),
        );
        art.resume();
        art.set_network(Network::Wifi);
        // A bad body is never written; the state settles on error (temporary failure).
        assert!(wait_state(&art, "error", Duration::from_secs(5)));
        assert!(!root.0.join("small/7.jpg").exists());
        assert_eq!(
            std::fs::read_dir(root.0.join("small")).unwrap().filter_map(|e| e.ok()).count(),
            0,
            "no .part or bad file remains"
        );
    }

    #[test]
    fn a_404_is_recorded_and_not_retried_on_restart() {
        let root = TempRoot::new("missing");
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let art = ArtSync::start(
            opts(&root, &[8], &[], "pack1"),
            throttle(move |_u: &str| {
                c.fetch_add(1, Ordering::SeqCst);
                FetchOutcome { not_found: true, ..FetchOutcome::default() }
            }),
        );
        art.resume();
        art.set_network(Network::Wifi);
        assert!(wait_state(&art, "done", Duration::from_secs(5)));
        assert_eq!(art.status()["unavailable"], 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(art);

        let c2 = calls.clone();
        let art2 = ArtSync::start(
            opts(&root, &[8], &[], "pack1"),
            throttle(move |_u: &str| {
                c2.fetch_add(1, Ordering::SeqCst);
                FetchOutcome { not_found: true, ..FetchOutcome::default() }
            }),
        );
        art2.resume();
        art2.set_network(Network::Wifi);
        assert!(wait_state(&art2, "done", Duration::from_secs(5)));
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the missing list suppresses the restart's fetch");
    }

    #[test]
    fn a_pack_or_assets_change_expires_the_missing_list() {
        let root = TempRoot::new("tokens");
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let art = ArtSync::start(
            opts(&root, &[9], &[], "pack1"),
            throttle(move |_u: &str| {
                c.fetch_add(1, Ordering::SeqCst);
                FetchOutcome { not_found: true, ..FetchOutcome::default() }
            }),
        );
        art.resume();
        art.set_network(Network::Wifi);
        assert!(wait_state(&art, "done", Duration::from_secs(5)));
        drop(art);
        let c2 = calls.clone();
        let art2 = ArtSync::start(
            opts(&root, &[9], &[], "pack2"),
            throttle(move |_u: &str| {
                c2.fetch_add(1, Ordering::SeqCst);
                FetchOutcome { data: Some(jpeg()), ..FetchOutcome::default() }
            }),
        );
        art2.resume();
        art2.set_network(Network::Wifi);
        assert!(wait_state(&art2, "done", Duration::from_secs(5)));
        assert_eq!(calls.load(Ordering::SeqCst), 2, "a new pack id retries the code");
        assert_eq!(art2.status()["have"], 1);
    }

    #[test]
    fn cellular_waits_for_permission_and_pause_resume_work() {
        let root = TempRoot::new("net");
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let art = ArtSync::start(
            opts(&root, &[10], &[], "t"),
            throttle(move |_u: &str| {
                c.fetch_add(1, Ordering::SeqCst);
                FetchOutcome { data: Some(jpeg()), ..FetchOutcome::default() }
            }),
        );
        art.resume();
        art.set_network(Network::Cellular);
        assert!(wait_state(&art, "waiting_wifi", Duration::from_secs(5)));
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        art.allow_data();
        assert!(wait_state(&art, "done", Duration::from_secs(5)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        art.pause();
        assert!(wait_state(&art, "paused", Duration::from_secs(2)));
        art.resume();
        assert!(wait_state(&art, "done", Duration::from_secs(2)));

        art.delete_all();
        assert!(wait_state(&art, "paused", Duration::from_secs(2)));
        assert_eq!(art.status()["have"], 0);
        assert!(!root.0.join("small/10.jpg").exists());
        art.resume();
        assert!(wait_state(&art, "done", Duration::from_secs(5)));
        assert_eq!(calls.load(Ordering::SeqCst), 2, "delete + resume fetches it again");
    }

    #[test]
    fn low_space_stops_before_fetching() {
        let root = TempRoot::new("space");
        let mut o = opts(&root, &[11], &[], "t");
        o.min_free = u64::MAX;
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let art = ArtSync::start(
            o,
            throttle(move |_u: &str| {
                c.fetch_add(1, Ordering::SeqCst);
                FetchOutcome { data: Some(jpeg()), ..FetchOutcome::default() }
            }),
        );
        art.resume();
        art.set_network(Network::Wifi);
        assert!(wait_state(&art, "low_space", Duration::from_secs(5)));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_403_pauses_the_sync() {
        let root = TempRoot::new("403");
        let art = ArtSync::start(
            opts(&root, &[12], &[], "t"),
            throttle(|_u: &str| FetchOutcome { pause: true, ..FetchOutcome::default() }),
        );
        art.resume();
        art.set_network(Network::Wifi);
        assert!(wait_state(&art, "rate_limited", Duration::from_secs(5)));
        assert!(art.status()["resume_in_s"].as_u64().unwrap() > 3000);
    }
}
