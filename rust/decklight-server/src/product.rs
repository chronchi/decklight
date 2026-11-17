//! `mode: product`'s search worker: the A3b-1 game loop (`decklight_search::play::search_game`) in a thread, the
//! human's choices through channels, exactly like `ProductDriver` in `decklight.edopro.product_bot`.
//!
//! The host env (`auto_single=false`) stays in the server's duel thread. The worker's product env
//! (`auto_single=true`) skips single-option prompts, so only multi-option decisions are exchanged. The worker
//! reports every decision before answering it; the host compares `(actor, options)` with its own pre-step state
//! (lockstep) and the worker's bot choices arrive in order.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use decklight_onnx::Pooled;
use decklight_search::play::{GameResult, PlayParams, PlayerNets, search_game};
use decklight_search::window::{Nets, WindowSearch};
use ygoenv_sys::{DuelConfig, Engine, Obs};

#[derive(Default)]
pub struct ProductControl {
    /// Set by a host-side stop that ends the duel thread (disconnect, new game, server shutdown).
    pub cancel: AtomicBool,
    /// A surrender on its way to the duel thread: stops the search like `cancel`, but the thread lives on (review,
    /// what-if) and clears it once the surrender is recorded.
    pub surrender: AtomicBool,
    pub disconnect: AtomicBool,
    /// Commands that can make the bot play (the opening advance, `choose`, `undo`) sent to the duel thread and not
    /// yet finished: the socket adds one per send, the duel thread takes it off when the command is done. Non-zero
    /// = the bot may be playing, so `hint`/`undo` answer at once like the Python server (`self.advancing`) instead
    /// of queueing behind the search.
    pub advancing: AtomicUsize,
    /// The session's game has ended (the duel thread keeps running for review and what-if): the socket answers
    /// `surrender` itself instead of stopping the thread.
    pub game_over: AtomicBool,
    /// Counts of `advancing` the duel thread has taken on and not yet given back. They are given back the moment
    /// the human's prompt (or the end) is reached, before its frame is sent, so a client that answers that frame
    /// with `hint`/`undo` never finds the count still up.
    pub owed: AtomicUsize,
    /// A worker's control points to its duel session's: the socket stops the session, and every worker the session
    /// starts (a new one per undo) sees it, while a worker closing (`close` sets its own `cancel`) stops only itself.
    pub parent: Option<Arc<ProductControl>>,
}

impl ProductControl {
    /// Give back every count the duel thread owes (see `owed`). Saturating: a mispaired count can never wrap the
    /// `usize` to a huge value and leave `hint`/`undo` refused for the rest of the session.
    pub fn settle_advancing(&self) {
        let n = self.owed.swap(0, Ordering::SeqCst);
        if n == 0 {
            return;
        }
        let mut cur = self.advancing.load(Ordering::SeqCst);
        loop {
            debug_assert!(cur >= n, "settle_advancing: owing {n} of only {cur} advancing");
            let next = cur.saturating_sub(n);
            match self.advancing.compare_exchange_weak(cur, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    pub fn stopping(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
            || self.surrender.load(Ordering::SeqCst)
            || self.parent.as_ref().is_some_and(|p| p.stopping())
    }
}

#[derive(Default)]
pub struct Share {
    decisions: Mutex<Vec<(i32, usize)>>,
    cond: Condvar,
    error: Mutex<Option<String>>,
    finished: AtomicBool,
}

impl Share {
    fn push_decision(&self, actor: i32, n: usize) {
        self.decisions.lock().unwrap().push((actor, n));
        self.cond.notify_all();
    }

    fn set_error(&self, e: String) {
        *self.error.lock().unwrap() = Some(e);
        self.finished.store(true, Ordering::SeqCst);
        self.cond.notify_all();
    }

    fn set_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
        self.cond.notify_all();
    }

    fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
}

/// The host-side handle of one duel's product worker.
pub struct ProductDriver {
    human_tx: mpsc::Sender<i32>,
    bot_rx: mpsc::Receiver<i32>,
    pub share: Arc<Share>,
    pub control: Arc<ProductControl>,
    join: Option<JoinHandle<Result<GameResult, String>>>,
    pub game_seed: u64,
    pub ws_seed: u64,
    pub replay_threads: usize,
    pub policy_sha256: String,
    pub value_sha256: String,
}

impl ProductDriver {
    /// Load the graphs and start the worker. Errors (missing/invalid ONNX files) surface here, before any duel.
    ///
    /// `prefix` (product-space choices) is replayed verbatim, never searched (undo / what-if: Python
    /// `ProductDriver(prefix=..., prefix_decisions=...)`); `prefix_decisions` holds its `(actor, options)` per
    /// choice and pre-fills the lockstep log, so outer decision `k` keeps its index across the restart.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        engine: &'static Engine,
        config: &DuelConfig,
        bot_seat: i32,
        game_seed: u64,
        ws_seed: u64,
        onnx: Pooled,
        replay_threads: usize,
        prefix: Vec<i32>,
        prefix_decisions: Vec<(i32, usize)>,
        session: Option<Arc<ProductControl>>,
    ) -> Result<ProductDriver, String> {
        if replay_threads < 1 {
            return Err(format!("replay_threads must be >= 1, got {replay_threads}"));
        }
        let policy_sha256 = onnx.policy_sha256.clone();
        let value_sha256 = onnx.value_sha256.clone();
        let (human_tx, human_rx) = mpsc::channel::<i32>();
        let (bot_tx, bot_rx) = mpsc::channel::<i32>();
        if prefix.len() != prefix_decisions.len() {
            return Err(format!(
                "prefix_decisions has {} entries for {} prefix choices",
                prefix_decisions.len(),
                prefix.len()
            ));
        }
        let share = Arc::new(Share::default());
        *share.decisions.lock().unwrap() = prefix_decisions;
        let control = Arc::new(ProductControl { parent: session, ..Default::default() });
        let cfg = config.clone();
        let share_w = share.clone();
        let control_w = control.clone();
        let join = std::thread::Builder::new()
            .name("product-search".into())
            .spawn(move || {
                run_worker(
                    engine, cfg, bot_seat, game_seed, ws_seed, onnx, human_rx, bot_tx, share_w, control_w,
                    replay_threads, prefix,
                )
            })
            .map_err(|e| format!("could not start the product worker: {e}"))?;
        Ok(ProductDriver {
            human_tx,
            bot_rx,
            share,
            control,
            join: Some(join),
            game_seed,
            ws_seed,
            replay_threads,
            policy_sha256,
            value_sha256,
        })
    }

    /// The next bot choice (blocking, cancel-aware).
    pub fn next_bot(&self) -> Result<i32, String> {
        loop {
            if self.control.stopping() {
                return Err("product worker stopped".into());
            }
            if let Some(e) = self.share.error() {
                return Err(e);
            }
            match self.bot_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(c) => return Ok(c),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(self
                        .share
                        .error()
                        .unwrap_or_else(|| "product worker ended before its next decision".into()));
                }
            }
        }
    }

    /// Enqueue a human choice (host/product option index are the same at a multi-option decision).
    pub fn feed_human(&self, index: i32) -> Result<(), String> {
        self.human_tx.send(index).map_err(|_| "product worker is gone".to_string())
    }

    /// The worker's outer decision `k` must match the host's pre-step `(actor, options)`.
    pub fn check_decision(&self, k: usize, pre: (i32, usize)) -> Result<(), String> {
        let t0 = Instant::now();
        loop {
            if self.control.stopping() {
                return Err("product worker stopped".into());
            }
            if let Some(e) = self.share.error() {
                return Err(e);
            }
            {
                let d = self.share.decisions.lock().unwrap();
                if let Some(&(actor, n)) = d.get(k) {
                    if (actor, n) != pre {
                        return Err(format!(
                            "outer decision {k}: host (player={}, options={}) vs product (player={actor}, options={n})",
                            pre.0, pre.1
                        ));
                    }
                    return Ok(());
                }
            }
            if self.share.finished() {
                return Err(format!("product game ended before outer decision {k}"));
            }
            if t0.elapsed() > Duration::from_secs(600) {
                return Err(format!("timed out waiting for product outer decision {k}"));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Stop the worker and join it (idempotent). Returns the worker's error, if any.
    pub fn close(&mut self) -> Option<String> {
        self.control.cancel.store(true, Ordering::SeqCst);
        let _ = self.human_tx.send(-1);
        let join = self.join.take()?;
        match join.join() {
            Ok(Ok(_)) => None,
            Ok(Err(e)) => Some(e),
            Err(_) => Some("product worker panicked".into()),
        }
    }
}

impl Drop for ProductDriver {
    fn drop(&mut self) {
        if let Some(e) = self.close() {
            if !self.control.stopping() {
                eprintln!("ALERT product: worker error on drop: {e}");
            }
        }
    }
}

struct WorkerNets {
    onnx: Pooled,
    human_rx: mpsc::Receiver<i32>,
    bot_tx: mpsc::Sender<i32>,
    share: Arc<Share>,
    control: Arc<ProductControl>,
}

impl WorkerNets {
    fn check(&self) -> Result<(), String> {
        if self.control.stopping() { Err("product worker stopped".into()) } else { Ok(()) }
    }
}

fn n_legal(obs: &Obs) -> usize {
    obs.action_mask.iter().filter(|&&x| x != 0).count()
}

impl Nets for WorkerNets {
    fn forward(&mut self, obs: &[&Obs]) -> Result<Vec<Vec<f32>>, String> {
        self.check()?;
        self.onnx.policy_logits(obs)
    }
    fn forward_threads(&mut self, obs: &[&Obs], threads: usize) -> Result<Vec<Vec<f32>>, String> {
        self.check()?;
        self.onnx.policy_logits_threads(obs, threads)
    }
    fn value(&mut self, obs: &[&Obs]) -> Result<Vec<f32>, String> {
        self.check()?;
        self.onnx.values(obs)
    }
}

impl PlayerNets for WorkerNets {
    fn human(&mut self, obs: &Obs) -> Result<i32, String> {
        let n = n_legal(obs);
        loop {
            self.check()?;
            match self.human_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(-1) => return Err("product worker stopped".into()),
                Ok(i) => {
                    if i < 0 || i as usize >= n {
                        return Err(format!("human returned illegal index {i} of {n}"));
                    }
                    return Ok(i);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("product host went away".into())
                }
            }
        }
    }

    fn decision(&mut self, obs: &Obs, actor: i32) -> Result<(), String> {
        self.check()?;
        self.share.push_decision(actor, n_legal(obs));
        Ok(())
    }

    fn bot_choice(&mut self, _obs: &Obs, choice: i32) -> Result<(), String> {
        self.check()?;
        self.bot_tx.send(choice).map_err(|_| "product host went away".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    engine: &'static Engine,
    config: DuelConfig,
    bot_seat: i32,
    game_seed: u64,
    ws_seed: u64,
    onnx: Pooled,
    human_rx: mpsc::Receiver<i32>,
    bot_tx: mpsc::Sender<i32>,
    share: Arc<Share>,
    control: Arc<ProductControl>,
    replay_threads: usize,
    prefix: Vec<i32>,
) -> Result<GameResult, String> {
    let mut nets = WorkerNets { onnx, human_rx, bot_tx, share: share.clone(), control };
    let mut search = WindowSearch::new(engine, 4, 1, 4, 1.0, 300, ws_seed, None, true);
    search.replay_threads = replay_threads;
    // DECKLIGHT_PLAY_PROF=1 aggregates WindowSearch's timing fields over the game (report-only, never a choice).
    if std::env::var_os("DECKLIGHT_PLAY_PROF").is_some() {
        search.prof = Some(decklight_search::window::Prof::default());
    }
    let mut params = PlayParams::product(bot_seat, game_seed);
    params.prefix = prefix;
    // The app's settings.json passes this through the environment (server.rs is not touched by this change).
    params.search_threads = std::env::var("DECKLIGHT_SEARCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    // A panic in the search ends the game with an error the client shows, instead of a bot that never answers.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| search_game(&mut search, &config, &mut nets, &params)))
        .unwrap_or_else(|p| Err(format!("the bot stopped: {}", decklight_onnx::panic_text(&*p))));
    match &res {
        Ok(r) => {
            // One line per game for speed checks on devices (the app writes stderr to its log).
            let lat = &r.lat_search;
            if !lat.is_empty() {
                let mut s = lat.clone();
                s.sort_by(f64::total_cmp);
                let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
                eprintln!(
                    "product: searched={} s/search mean={:.3} p50={:.3} p90={:.3} max={:.3} replay_threads={replay_threads}",
                    lat.len(),
                    lat.iter().sum::<f64>() / lat.len() as f64,
                    q(0.5),
                    q(0.9),
                    s[s.len() - 1]
                );
            }
            if let Some(p) = search.prof.as_ref() {
                eprintln!(
                    "product: prof replay={:.3} step={:.3} fwd={:.3} value={:.3} rollout={:.3} steps={} fwd_calls={} fwd_obs={} value_calls={} value_obs={}",
                    p.replay_s, p.step_s, p.fwd_s, p.value_s, p.rollout_s, p.steps, p.fwd_calls, p.fwd_obs,
                    p.value_calls, p.value_obs
                );
            }
            share.set_finished()
        }
        Err(e) => share.set_error(e.clone()),
    }
    res
}

// ---------------------------------------------------------------- win chance calibration (port of decklight.eval.winchance)

/// The first `[...]` after `text`, as f64s parsed by Rust (correctly rounded), plus the remaining text.
fn parse_number_array(text: &str) -> Result<(Vec<f64>, &str), String> {
    let lo = text.find('[').ok_or_else(|| "winchance map: missing [".to_string())?;
    let hi = text[lo..].find(']').ok_or_else(|| "winchance map: missing ]".to_string())? + lo;
    let body = &text[lo + 1..hi];
    let mut out = Vec::new();
    for part in body.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        out.push(p.parse::<f64>().map_err(|e| format!("winchance map: {p:?}: {e}"))?);
    }
    Ok((out, &text[hi + 1..]))
}

/// `CalibMap`: one piecewise-linear knot map per turn bucket, flat extrapolation, clipped to [P_LO, P_HI].
#[derive(Clone)]
pub struct WcMap {
    knots: [(Vec<f64>, Vec<f64>); 3],
}

pub const P_LO: f64 = 0.03;
pub const P_HI: f64 = 0.97;

impl WcMap {
    pub fn load(path: &Path) -> Result<WcMap, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // Parse the decimal strings ourselves: serde_json's fast float path can be one ulp off
        // (`0.9127155172413793` -> ...92), and the map is compared bit-for-bit with numpy.interp.
        let start = text.find("\"knots\"").ok_or_else(|| "winchance map: no map.knots".to_string())?;
        let mut rest = &text[start..];
        let mut out: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        for _ in 0..3 {
            let xi = rest.find("\"x\"").ok_or_else(|| "winchance map: knots.x".to_string())?;
            let (xs, after_x) = parse_number_array(&rest[xi + 3..])?;
            let yi = after_x.find("\"y\"").ok_or_else(|| "winchance map: knots.y".to_string())?;
            let (ys, after_y) = parse_number_array(&after_x[yi + 3..])?;
            if xs.is_empty() || xs.len() != ys.len() {
                return Err("winchance map: bad knots".into());
            }
            if ys.windows(2).any(|w| w[1] < w[0]) {
                return Err("winchance map: non-monotone map".into());
            }
            out.push((xs, ys));
            rest = after_y;
        }
        let [a, b, c] = <[(Vec<f64>, Vec<f64>); 3]>::try_from(out)
            .map_err(|_| "winchance map: bucket count".to_string())?;
        Ok(WcMap { knots: [a, b, c] })
    }

    pub fn bucket_of(turn: i32) -> usize {
        if turn <= 2 {
            0
        } else if turn <= 4 {
            1
        } else {
            2
        }
    }

    /// `CalibMap.p`: raw value -> p_raw, then the turn bucket's map.
    pub fn p(&self, v: f64, turn: i32) -> f64 {
        let p_raw = (v + 1.0) / 2.0;
        let (xs, ys) = &self.knots[Self::bucket_of(turn)];
        interp(p_raw, xs, ys).clamp(P_LO, P_HI)
    }
}

/// `numpy.interp` for one point: linear between knots, flat extrapolation outside.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let hi = xs.partition_point(|&v| v <= x);
    let lo = hi - 1;
    let dx = xs[hi] - xs[lo];
    if dx == 0.0 {
        ys[hi]
    } else {
        // numpy.interp's exact operation order: slope * (x - x_lo) + y_lo.
        let slope = (ys[hi] - ys[lo]) / dx;
        slope * (x - xs[lo]) + ys[lo]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use decklight_onnx::NetsPool;

    #[test]
    fn bucket_edges_match_python() {
        assert_eq!(WcMap::bucket_of(1), 0);
        assert_eq!(WcMap::bucket_of(2), 0);
        assert_eq!(WcMap::bucket_of(3), 1);
        assert_eq!(WcMap::bucket_of(4), 1);
        assert_eq!(WcMap::bucket_of(5), 2);
    }

    #[test]
    fn interp_matches_the_observed_point() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let m = WcMap::load(&root.join("src/decklight/web/winchance_calib.json")).unwrap();
        let (xs, ys) = &m.knots[1];
        assert_eq!(ys[32].to_bits(), 0x3fed34f72c234f73, "float parsed one ulp off");
        let x = 0.8197485208511353f64;
        let p = interp(x, xs, ys);
        assert_eq!(p.to_bits(), 0x3feb4cebe291253b);
    }

    #[test]
    fn interp_matches_numpy_examples() {
        let xs = [0.0, 0.5, 1.0];
        let ys = [0.1, 0.4, 0.9];
        assert!((interp(0.25, &xs, &ys) - 0.25).abs() < 1e-12);
        assert_eq!(interp(-1.0, &xs, &ys), 0.1);
        assert_eq!(interp(2.0, &xs, &ys), 0.9);
    }

    #[test]
    fn settling_advancing_gives_back_only_what_is_owed() {
        let c = ProductControl::default();
        c.advancing.store(2, Ordering::SeqCst);
        c.owed.store(1, Ordering::SeqCst);
        c.settle_advancing();
        assert_eq!(c.advancing.load(Ordering::SeqCst), 1);
        c.settle_advancing(); // nothing owed now: must not subtract again
        assert_eq!(c.advancing.load(Ordering::SeqCst), 1);
        c.owed.store(1, Ordering::SeqCst);
        c.settle_advancing();
        assert_eq!(c.advancing.load(Ordering::SeqCst), 0);
        c.settle_advancing(); // must not wrap around to usize::MAX
        assert_eq!(c.advancing.load(Ordering::SeqCst), 0);
    }

    fn threads_named(name: &str) -> usize {
        let mut n = 0;
        if let Ok(rd) = std::fs::read_dir("/proc/self/task") {
            for e in rd.flatten() {
                if let Ok(comm) = std::fs::read_to_string(e.path().join("comm")) {
                    if comm.trim() == name {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    #[test]
    fn cancel_joins_the_search_worker() {
        use decklight_cards::{build_engine, default_cdbs, index_scripts, load_cards, parse_ydk, shuffle_main};
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            let dylib = root.join(
                ".venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.1",
            );
            if !dylib.is_file() {
                eprintln!("skip: no ONNX Runtime dylib (set ORT_DYLIB_PATH)");
                return;
            }
            // SAFETY: set before any ORT session is created (the only ORT user in these tests).
            unsafe { std::env::set_var("ORT_DYLIB_PATH", &dylib) };
        }
        let cards = load_cards(&default_cdbs(&root.join("third_party/BabelCDB"), &[])).unwrap();
        let index = index_scripts(&root.join("third_party/CardScripts"), &[]).expect("card scripts (dir or .pack) unreadable");
        let engine: &'static Engine = Box::leak(Box::new(build_engine(&cards, &index)));
        let read = |name: &str| {
            parse_ydk(&std::fs::read_to_string(root.join("decks/pool").join(name)).unwrap())
        };
        let (d0, d1) = (read("blue-eyes.ydk"), read("dark-magician.ydk"));
        let (mut main0, mut main1) = (d0.main.clone(), d1.main.clone());
        shuffle_main(&mut main0, 7600, 0);
        shuffle_main(&mut main1, 7600, 1);
        let cfg = DuelConfig {
            main0,
            extra0: d0.extra.clone(),
            main1,
            extra1: d1.extra.clone(),
            seed: 7600,
            ..DuelConfig::default()
        };
        let mut driver = ProductDriver::start(
            engine,
            &cfg,
            0,
            7600,
            7600,
            NetsPool::new(&root.join("data/cache/onnx/r1"), &root, 1).take().unwrap(),
            1,
            Vec::new(),
            Vec::new(),
            None,
        )
        .unwrap();
        let t0 = Instant::now();
        while threads_named("product-search") == 0 && t0.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(threads_named("product-search"), 1, "worker thread not running");
        std::thread::sleep(Duration::from_millis(200));
        driver.control.cancel.store(true, Ordering::SeqCst);
        let _ = driver.close();
        assert_eq!(threads_named("product-search"), 0, "worker thread leaked");
    }

    #[test]
    fn missing_onnx_files_are_a_clear_error() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let engine: &'static Engine = Box::leak(Box::new(Engine::new()));
        let _ = engine;
        let e = match NetsPool::new(&root.join("data/cache/onnx/definitely-missing"), &root, 1).take() {
            Ok(_) => panic!("missing ONNX files must fail"),
            Err(e) => e,
        };
        assert!(e.contains("manifest") || e.contains("No such file"), "unclear error: {e}");
    }

    #[test]
    fn loads_the_committed_map() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let m = WcMap::load(&root.join("src/decklight/web/winchance_calib.json")).unwrap();
        let p = m.p(0.0, 1);
        assert!((P_LO..=P_HI).contains(&p));
    }
}
