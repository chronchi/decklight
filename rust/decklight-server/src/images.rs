//! Card-image throttle (port of `ImgThrottle`): one shared upstream limiter for the on-demand `/img` path (<= 8
//! fetches/s) and the background art sync (<= 15/s overall, on-demand always first), one at a time, concurrent
//! requests for the same URL share one fetch, failures are not retried for 10 min, an HTTP 403/429 pauses
//! everything for an hour.

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

pub const RATE: f64 = 8.0;
pub const BULK_RATE: f64 = 15.0;
pub const RETRY: Duration = Duration::from_secs(600);
pub const PAUSE: Duration = Duration::from_secs(3600);
pub const MIN_JPEG_BYTES: usize = 1024;
const MAX_IMAGE_BYTES: usize = 2 * 1024 * 1024; // full art is ~170 KB; anything bigger is not card art

/// One upstream fetch's result. `data` is None for every failure; `not_found` distinguishes a real 404/empty body
/// (the art sync records it), `pause` asks the throttle to stop fetching for an hour (403/429).
#[derive(Clone, Default)]
pub struct FetchOutcome {
    pub data: Option<Vec<u8>>,
    pub pause: bool,
    pub not_found: bool,
}

impl FetchOutcome {
    fn fail() -> FetchOutcome {
        FetchOutcome::default()
    }
}

pub type FetchFn = Arc<dyn Fn(&str) -> FetchOutcome + Send + Sync>;

struct Flight {
    result: Mutex<Option<FetchOutcome>>,
    done: Notify,
}

struct State {
    last: Instant,
    last_demand: Instant,
    demand_waiters: usize,
    inflight: HashMap<String, Arc<Flight>>,
    failed: HashMap<String, Instant>,
    paused_until: Option<Instant>,
}

pub struct ImgThrottle {
    state: Mutex<State>,
    rate: tokio::sync::Mutex<()>,
    fetch: FetchFn,
}

impl ImgThrottle {
    pub fn new(fetch: FetchFn) -> ImgThrottle {
        ImgThrottle {
            state: Mutex::new(State {
                last: Instant::now() - Duration::from_secs(1),
                last_demand: Instant::now() - Duration::from_secs(1),
                demand_waiters: 0,
                inflight: HashMap::new(),
                failed: HashMap::new(),
                paused_until: None,
            }),
            rate: tokio::sync::Mutex::new(()),
            fetch,
        }
    }

    fn store_result(flight: &Flight, value: FetchOutcome) {
        *flight.result.lock().unwrap() = Some(value);
        flight.done.notify_waiters();
    }

    /// On-demand fetch (`/img`): at most 8/s, ahead of every bulk request.
    pub async fn fetch(&self, url: &str) -> Option<Vec<u8>> {
        self.fetch_outcome(url, false).await.data
    }

    /// Background sync fetch: shares the 15/s upstream limit and yields to on-demand requests.
    pub async fn fetch_bulk(&self, url: &str) -> FetchOutcome {
        self.fetch_outcome(url, true).await
    }

    /// The on-demand fetch with its outcome (the `/img` route records 404s and served files).
    pub async fn fetch_art(&self, url: &str) -> FetchOutcome {
        self.fetch_outcome(url, false).await
    }

    async fn fetch_outcome(&self, url: &str, bulk: bool) -> FetchOutcome {
        let flight = {
            let mut st = self.state.lock().unwrap();
            let now = Instant::now();
            if st.paused_until.is_some_and(|p| now < p) {
                return FetchOutcome { pause: true, ..FetchOutcome::default() };
            }
            if st.failed.get(url).is_some_and(|t| now < *t) {
                return FetchOutcome::fail();
            }
            if let Some(f) = st.inflight.get(url) {
                Some(f.clone())
            } else {
                let f = Arc::new(Flight { result: Mutex::new(None), done: Notify::new() });
                st.inflight.insert(url.to_string(), f.clone());
                None
            }
        };
        if let Some(f) = flight {
            // Someone else is fetching: wait for that result (never start a second request).
            loop {
                let notified = f.done.notified();
                if let Some(v) = f.result.lock().unwrap().clone() {
                    return v;
                }
                notified.await;
                if let Some(v) = f.result.lock().unwrap().clone() {
                    return v;
                }
            }
        }

        let f = {
            let st = self.state.lock().unwrap();
            st.inflight.get(url).cloned().unwrap()
        };
        let result = self.owner_fetch(url, bulk).await;
        {
            let mut st = self.state.lock().unwrap();
            if result.data.is_none() {
                st.failed.insert(url.to_string(), Instant::now() + RETRY);
            }
            st.inflight.remove(url);
        }
        Self::store_result(&f, result.clone());
        result
    }

    async fn acquire(&self, bulk: bool) {
        if !bulk {
            self.state.lock().unwrap().demand_waiters += 1;
        }
        loop {
            let wait = {
                let mut st = self.state.lock().unwrap();
                let now = Instant::now();
                if bulk && st.demand_waiters > 0 {
                    Some(Duration::from_millis(20))
                } else {
                    let mut due = st.last + Duration::from_secs_f64(1.0 / BULK_RATE);
                    if !bulk {
                        let d2 = st.last_demand + Duration::from_secs_f64(1.0 / RATE);
                        if d2 > due {
                            due = d2;
                        }
                    }
                    if now >= due {
                        st.last = now;
                        if !bulk {
                            st.last_demand = now;
                        }
                        None
                    } else {
                        Some(due - now)
                    }
                }
            };
            match wait {
                None => break,
                Some(d) => tokio::time::sleep(d).await,
            }
        }
        if !bulk {
            let mut st = self.state.lock().unwrap();
            st.demand_waiters = st.demand_waiters.saturating_sub(1);
        }
    }

    async fn owner_fetch(&self, url: &str, bulk: bool) -> FetchOutcome {
        let _gate = self.rate.lock().await;
        {
            let st = self.state.lock().unwrap();
            if st.paused_until.is_some_and(|p| Instant::now() < p) {
                return FetchOutcome { pause: true, ..FetchOutcome::default() };
            }
        }
        self.acquire(bulk).await;
        let f = self.fetch.clone();
        let u = url.to_string();
        let mut result = tokio::task::spawn_blocking(move || f(&u))
            .await
            .unwrap_or_else(|_| FetchOutcome::fail());
        if let Some(d) = result.data.as_ref() {
            if !valid_jpeg(d) {
                result.data = None;
            }
        }
        {
            let mut st = self.state.lock().unwrap();
            st.last = Instant::now();
            if result.pause {
                st.paused_until = Some(Instant::now() + PAUSE);
            }
        }
        result
    }

    pub fn pause(&self) {
        self.state.lock().unwrap().paused_until = Some(Instant::now() + PAUSE);
    }

    /// When the 403/429 pause ends, if one is running.
    pub fn paused_until(&self) -> Option<Instant> {
        let st = self.state.lock().unwrap();
        st.paused_until.filter(|p| Instant::now() < *p)
    }
}

/// A body is card art only when it is a real JPEG of at least 1 KB (YGOPRODeck's error pages and HTML 200s are not).
pub fn valid_jpeg(data: &[u8]) -> bool {
    data.len() >= MIN_JPEG_BYTES && data.starts_with(&[0xFF, 0xD8])
}

fn image_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(5))
            .user_agent("decklight/1.0")
            .redirects(0) // YGOPRODeck serves art directly; a redirect is not followed anywhere
            .build()
    })
}

/// One upstream image fetch with `ureq` (rustls, no curl binary). The throttle owns the rate.
pub fn http_fetch(url: &str) -> FetchOutcome {
    match image_agent().get(url).call() {
        Ok(resp) => {
            let mut data = Vec::new();
            let read = resp.into_reader().take(MAX_IMAGE_BYTES as u64 + 1).read_to_end(&mut data);
            if read.is_err() || data.len() > MAX_IMAGE_BYTES {
                return FetchOutcome::fail();
            }
            if data.is_empty() {
                return FetchOutcome { not_found: true, ..FetchOutcome::default() };
            }
            FetchOutcome { data: Some(data), ..FetchOutcome::default() }
        }
        Err(ureq::Error::Status(404, _)) => FetchOutcome { not_found: true, ..FetchOutcome::default() },
        Err(ureq::Error::Status(403, _)) | Err(ureq::Error::Status(429, _)) => {
            FetchOutcome { pause: true, ..FetchOutcome::default() }
        }
        Err(_) => FetchOutcome::fail(),
    }
}

/// A plain HTTP(S) GET for the card-pack download (longer timeout; the caller enforces the size cap).
pub fn http_get(url: &str, timeout: Duration, max_bytes: usize) -> Result<Vec<u8>, String> {
    let agent = ureq::AgentBuilder::new().timeout(timeout).user_agent("decklight/1.0").redirects(3).build();
    let resp = agent.get(url).call().map_err(|e| match e {
        ureq::Error::Status(code, _) => format!("HTTP {code}"),
        ureq::Error::Transport(t) => t.to_string(),
    })?;
    let mut data = Vec::new();
    resp.into_reader()
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() > max_bytes {
        return Err(format!("download exceeds {max_bytes} bytes"));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn jpeg() -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8];
        v.resize(MIN_JPEG_BYTES, 7);
        v
    }

    fn ok() -> FetchOutcome {
        FetchOutcome { data: Some(jpeg()), ..FetchOutcome::default() }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dedupes_concurrent_fetches() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let t = ImgThrottle::new(Arc::new(move |_u: &str| {
            c.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(50));
            ok()
        }));
        let (a, b) = tokio::join!(t.fetch("x"), t.fetch("x"));
        assert_eq!(a, Some(jpeg()));
        assert_eq!(b, Some(jpeg()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rates_and_caches_failures() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let t = ImgThrottle::new(Arc::new(move |u: &str| {
            c.fetch_add(1, Ordering::SeqCst);
            if u == "bad" { FetchOutcome::fail() } else { ok() }
        }));
        let t0 = Instant::now();
        assert_eq!(t.fetch("a").await, Some(jpeg()));
        assert_eq!(t.fetch("b").await, Some(jpeg()));
        assert!(t0.elapsed() >= Duration::from_millis(120), "two fetches are >= 1/8 s apart");
        assert_eq!(t.fetch("bad").await, None);
        assert_eq!(t.fetch("bad").await, None);
        assert_eq!(calls.load(Ordering::SeqCst), 3, "a failed URL is not retried");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_403_pauses_every_fetch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let t = ImgThrottle::new(Arc::new(move |_u: &str| {
            c.fetch_add(1, Ordering::SeqCst);
            FetchOutcome { pause: true, ..FetchOutcome::default() }
        }));
        assert_eq!(t.fetch("a").await, None);
        assert_eq!(t.fetch("b").await, None);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "paused: b is not fetched");
    }

    #[test]
    fn rejects_non_jpeg_and_small_bodies() {
        let t = ImgThrottle::new(Arc::new(|_u: &str| FetchOutcome {
            data: Some(b"<svg>".to_vec()),
            ..FetchOutcome::default()
        }));
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        assert_eq!(rt.block_on(t.fetch("x")), None);
        let small = ImgThrottle::new(Arc::new(|_u: &str| FetchOutcome {
            data: Some(vec![0xFF, 0xD8, 1, 2, 3]),
            ..FetchOutcome::default()
        }));
        assert_eq!(rt.block_on(small.fetch("x")), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bulk_is_at_most_15_per_second_and_yields_to_on_demand() {
        use std::sync::atomic::AtomicBool;
        let done = Arc::new(AtomicBool::new(false));
        let order = Arc::new(Mutex::new(Vec::new()));
        let o = order.clone();
        let d = done.clone();
        let t = Arc::new(ImgThrottle::new(Arc::new(move |u: &str| {
            o.lock().unwrap().push(u.to_string());
            if u == "demand" {
                d.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(2));
            ok()
        })));
        let t0 = Instant::now();
        let mut task = {
            let t = t.clone();
            tokio::spawn(async move {
                for i in 0..12 {
                    t.fetch_bulk(&format!("bulk-{i}")).await;
                }
            })
        };
        // Wait until the bulk loop is under way, then ask for one on-demand image.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(t.fetch("demand").await, Some(jpeg()));
        tokio::time::timeout(Duration::from_secs(5), &mut task).await.unwrap().unwrap();
        let elapsed = t0.elapsed();
        assert!(elapsed >= Duration::from_secs_f64(13.0 / BULK_RATE - 0.05), "13 fetches obey the shared 15/s");
        let order = order.lock().unwrap().clone();
        let demand_at = order.iter().position(|u| u == "demand").unwrap();
        assert!(demand_at < order.len() - 1, "the on-demand request is served before the remaining bulk work");
    }
}
