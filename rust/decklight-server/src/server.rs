//! HTTP + WebSocket server and the duel loop (port of `decklight.web.server` and `play.session`).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use decklight_cards::CardData;
use decklight_cards::pack::{PackStatus, PackStore};
use decklight_cards::texts::Texts;
use decklight_core::constants as C;
use decklight_core::messages::{self as M, Body as MsgBody};
use decklight_view::{Ctx, DuelStats};
use serde_json::{Value, json};
use sha1::{Digest, Sha1};
use tokio::sync::mpsc;

use decklight_search::play::round4;
use decklight_search::select::{argmax_select, pairwise_sum};

use crate::artsync::{ArtSync, ArtSyncOptions, MIN_FREE_BYTES, Network, deck_codes};
use crate::cards_search::SearchIndex;
use crate::decks::{DeckCatalog, DeckDirs, DeckSpec, check_deck, pretty_deck_name};
use crate::images::{ImgThrottle, http_fetch, http_get, valid_jpeg};
use crate::product::{ProductControl, ProductDriver, WcMap};
use crate::raw::RawPolicy;
use crate::record::{RecordEnv, RecordGame, ResponseRec, random_hex};
use crate::room::{HostedRoom, Role, RoomCmd, RoomOut, RoomSession, RoomSpec};
use crate::userdecks::{Format as DeckFormat, UserDecks, deck_from_value, to_ydk, to_ydke, unknown_codes};
use ygoenv_sys::{DuelConfig, Engine, Env, Obs, TCG_FLAGS};

const CARD_BACK_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="146" viewBox="0 0 100 146">
<rect width="100" height="146" rx="6" fill="#1d2a5e"/>
<rect x="8" y="8" width="84" height="130" rx="4" fill="none" stroke="#c8a24a" stroke-width="2"/>
<circle cx="50" cy="73" r="22" fill="none" stroke="#c8a24a" stroke-width="2"/>
<text x="50" y="77" text-anchor="middle" font-size="10" fill="#c8a24a">decklight</text>
</svg>
"##;

const IMG_SMALL: &str = "https://images.ygoprodeck.com/images/cards_small/{code}.jpg";
const IMG_FULL: &str = "https://images.ygoprodeck.com/images/cards/{code}.jpg";

/// The image host, overridable for tests (`DECKLIGHT_IMG_BASE`, with the `/images/...` paths appended). Only a
/// loopback test server may replace YGOPRODeck: any other value is ignored (no fetches to arbitrary hosts).
fn image_host() -> String {
    const UPSTREAM: &str = "https://images.ygoprodeck.com";
    match std::env::var("DECKLIGHT_IMG_BASE").ok().filter(|s| !s.trim().is_empty()) {
        Some(s) if s.starts_with("http://127.0.0.1:") || s.starts_with("http://localhost:") => {
            s.trim_end_matches('/').to_string()
        }
        Some(s) => {
            eprintln!("ALERT server: DECKLIGHT_IMG_BASE {s:?} ignored (loopback test servers only)");
            UPSTREAM.to_string()
        }
        None => UPSTREAM.to_string(),
    }
}

fn small_base() -> String {
    format!("{}/images/cards_small", image_host())
}

/// `DECKLIGHT_TIMING=1` prints one `TIMING <label> <seconds>` line per startup / new-game phase on stderr.
fn timing_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DECKLIGHT_TIMING").is_some())
}

fn timing(label: &str, t0: Instant) {
    if timing_on() {
        eprintln!("TIMING {} {:.6}", label, t0.elapsed().as_secs_f64());
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub root: PathBuf,
    pub port: u16,
    pub token: String,
    pub web_dir: PathBuf,
    pub decks_dirs: Option<Vec<PathBuf>>,
    pub onnx_dir: PathBuf,
    pub records_dir: PathBuf,
    pub pics_dir: PathBuf,
    pub fixed_seed: Option<u64>,
    pub test_opponent: Option<String>,
    pub default_opponent: String,
    /// Speed plan E6: search thread count (1 = the sequential replays).
    pub replay_threads: usize,
    /// ORT intra-op threads for product duels (1 = the parity setting).
    pub ort_threads: usize,
    /// A5: installed card packs (active/previous markers live here).
    pub packs_dir: PathBuf,
    /// A5: a pack URL fetched and installed at startup (also the update source the UI offers).
    pub pack_url: Option<String>,
    /// The user's saved decks (the Ask eval points this at a fixture copy).
    pub user_decks_dir: PathBuf,
    /// Where Ask API keys live; `None` = the OS credential store, opened on first use (tests and the eval pass one).
    pub secrets: Option<Arc<dyn crate::ask::keys::SecretStore>>,
    /// Ask chats (`chats.sqlite`) and provider settings.
    pub ask_dir: PathBuf,
}

impl Config {
    pub fn defaults(root: &Path) -> Config {
        Config {
            root: root.to_path_buf(),
            port: 0,
            token: String::new(),
            web_dir: root.join("web"),
            decks_dirs: None,
            onnx_dir: root.join("data/cache/onnx/y"),
            records_dir: root.join("runs/app/records"),
            pics_dir: root.join("data/cache/pics"),
            fixed_seed: None,
            test_opponent: None,
            default_opponent: "fast".to_string(),
            replay_threads: 1,
            ort_threads: 1,
            packs_dir: root.join("data/cards/packs"),
            pack_url: None,
            user_decks_dir: root.join("decks/user"),
            ask_dir: root.join("runs/app/ask"),
            secrets: None,
        }
    }
}

pub struct App {
    pub root: PathBuf,
    pub token: String,
    pub web_dir: PathBuf,
    pub pics_dir: PathBuf,
    pub engine: &'static Engine,
    pub ctx: Ctx,
    pub cards: BTreeMap<u32, CardData>,
    pub emb_codes: BTreeSet<u32>,
    pub records: RecordEnv,
    pub onnx_dir: PathBuf,
    /// Product duels' policy/value sets, loaded once and reused across games.
    pub onnx_pool: Arc<decklight_onnx::NetsPool>,
    raw: Mutex<RawState>,
    pub models: Vec<(String, String)>,
    pub model_id: String,
    pub model_label: String,
    pub default_opponent: String,
    pub fixed_seed: Option<u64>,
    pub replay_threads: usize,
    pub test_stream: bool,
    pub img: Arc<ImgThrottle>,
    /// Background small-art sync (idle until the client enables card art).
    pub art: Arc<ArtSync>,
    /// Lobby puzzles, primers and the one-at-a-time deck study (Python's lazy caches).
    pub puzzles: Mutex<Option<Vec<Value>>>,
    pub primers: Mutex<Option<Value>>,
    pub study_busy: AtomicBool,
    /// A5: the pack store and the status of the pack loaded at startup (or installed since).
    pub packs: PackStore,
    pub pack_status: Mutex<PackStatus>,
    pub pack_update_busy: AtomicBool,
    pub pack_url: Option<String>,
    /// Saved user decks (`decks/user/`) and the lazily built card-search index.
    pub user_decks: UserDecks,
    /// Ask provider keys (write-only to clients: only hints leave the server).
    pub api_keys: crate::ask::keys::ApiKeys,
    pub ask: crate::ask::ws::Ask,
    pub search: Mutex<Option<Arc<SearchIndex>>>,
    pub search_conf: PathBuf,
    /// "Describe a card": the optional semantic model download and its lazy encoder.
    pub describe: Arc<crate::describe::Describe>,
    /// On-device models: the LiteRT-LM runtime and the model downloads (ask/ondevice.rs).
    pub ondevice: Arc<crate::ask::ondevice::Runtime>,
    pub device_models: Arc<crate::ask::ondevice::Downloads>,
    pub inner: tokio::sync::Mutex<Inner>,
}

enum RawState {
    Uninit,
    Ready(Box<RawPolicy>),
    Failed(String),
}

impl App {
    fn ensure_raw(&self) -> Result<(), String> {
        let mut st = self.raw.lock().unwrap();
        match &*st {
            RawState::Ready(_) => Ok(()),
            RawState::Failed(e) => Err(e.clone()),
            RawState::Uninit => match RawPolicy::load(&self.onnx_dir, &self.root) {
                Ok(p) => {
                    *st = RawState::Ready(Box::new(p));
                    Ok(())
                }
                Err(e) => {
                    *st = RawState::Failed(e.clone());
                    Err(e)
                }
            },
        }
    }

    fn raw_choose(&self, obs: &Obs, n: usize) -> Result<i32, String> {
        self.ensure_raw()?;
        let mut st = self.raw.lock().unwrap();
        match &mut *st {
            RawState::Ready(p) => p.choose(obs, n),
            RawState::Failed(e) => Err(e.clone()),
            RawState::Uninit => Err("raw policy not loaded".into()),
        }
    }

    pub fn model_sha(&self) -> String {
        let p = self.onnx_dir.join("policy.onnx");
        if p.is_file() { crate::record::sha256_file(&p) } else { String::new() }
    }

    pub fn lobby_decks(&self, inner: &Inner) -> Vec<Value> {
        let names = self.ctx.texts.names();
        let mut saved = Vec::new();
        for stored in self.user_decks.list() {
            let legal = crate::userdecks::legality(&stored.deck, &self.cards, names, &DeckFormat::default());
            saved.push(json!({
                "id": format!("user:{}", stored.id),
                "name": stored.meta.name,
                "group": "saved",
                "user": true,
                "legal": legal.ok,
                "problems": legal.problems.iter().map(|p| p.text.clone()).collect::<Vec<_>>(),
                "meta": crate::userdecks::deck_summary(&stored.deck, &self.cards, stored.meta.cover),
            }));
        }
        merge_lobby_decks(saved, inner.deck_catalog.list_decks())
    }

    /// The card-search index, built once on first use (and reported once so its cost is visible in the log).
    pub fn search_index(&self) -> Arc<SearchIndex> {
        let mut guard = self.search.lock().unwrap();
        if let Some(index) = guard.as_ref() {
            return index.clone();
        }
        let t = Instant::now();
        let index = Arc::new(SearchIndex::build(&self.cards, &self.ctx.texts, &self.search_conf));
        eprintln!(
            "decklight: card search index built in {} ms ({} names, {:.1} MB)",
            t.elapsed().as_millis(),
            index.len(),
            index.size_bytes as f64 / (1024.0 * 1024.0)
        );
        *guard = Some(index.clone());
        index
    }
}

/// Saved user decks come first (newest first), then the prebuilt and tuned rows.
fn merge_lobby_decks(mut saved: Vec<Value>, catalog: Vec<Value>) -> Vec<Value> {
    saved.extend(catalog);
    saved
}

pub struct Inner {
    conn_id: u64,
    conn_tx: Option<mpsc::UnboundedSender<Outgoing>>,
    pub game: Option<Game>,
    product: Option<ProductSession>,
    chain_mode: String,
    winchance_on: bool,
    games_started: u64,
    pub deck_catalog: DeckCatalog,
    /// A2b-3c: the room this page is in (host or guest), the app's own room host, and the room's frame sink.
    room: Option<RoomSession>,
    hosted: Option<HostedRoom>,
    room_out: RoomOut,
    /// The live game's decks in human/bot order (Python's `game_decks` / `game_deck_names`).
    live_decks: Option<(DeckSpec, DeckSpec)>,
    live_deck_names: (String, String),
}

pub enum Outgoing {
    Close(u16),
    Frame(Value),
}

/// The current connection's frame sink. A duel thread outlives the socket that started it, so it keeps this handle
/// and a takeover swaps the sender inside it (Python's `self.ws`): the game's frames follow the person, never the
/// closed connection.
#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Option<mpsc::UnboundedSender<Outgoing>>>>);

impl Sink {
    fn set(&self, tx: Option<mpsc::UnboundedSender<Outgoing>>) {
        *self.0.lock().unwrap() = tx;
    }

    fn send(&self, out: Outgoing) {
        if let Some(tx) = self.0.lock().unwrap().as_ref() {
            let _ = tx.send(out);
        }
    }
}

/// Commands from the socket loop to a product duel's thread (the duel owns the host env there).
enum ProductCmd {
    Choose { prompt: i64, index: i64 },
    Resync,
    Hint,
    Undo,
    Cancel,
    Settings { chain: String, winchance: bool },
    Review,
    ReviewFrame(Value),
    WhatIf(Value),
    Surrender,
    Abandon,
}

/// The socket side of a running product duel: its thread owns the game and sends frames through `Outgoing`.
struct ProductSession {
    cmd_tx: std::sync::mpsc::Sender<ProductCmd>,
    control: Arc<ProductControl>,
    join: Option<std::thread::JoinHandle<()>>,
    /// Frames follow the current connection (takeover swaps the sender inside).
    sink: Sink,
}

/// What `Game::new` needs to start a product worker.
struct ProductLaunch {
    search: decklight_onnx::Pooled,
    replay_threads: usize,
    value: decklight_onnx::Pooled,
    /// The duel session's control (the socket's stop flags); the worker's own control points to it.
    session: Arc<ProductControl>,
}

pub struct Game {
    pub human_seat: i32,
    /// Deck display names in seat order, for the match record (set by the lobby after `Game::new`).
    pub deck_names: [String; 2],
    pub bot_seat: i32,
    cfg: DuelConfig,
    env: Env<'static>,
    obs: Box<Obs>,
    bot: Bot,
    stats: DuelStats,
    pub result: Option<DuelResult>,
    pub finished: bool,
    pub end_reason: String,
    advancing: bool,
    cue_sent: usize,
    log_sent: usize,
    prompt_id: i64,
    last_prompt_seq: i64,
    auto_passed: BTreeSet<usize>,
    human_decisions: Vec<usize>,
    practice: bool,
    /// Practice counters carried across undos (Python `hints_used`, record meta `undos`).
    hints_used: u32,
    undos: u32,
    /// Cancel (any bot game, not only practice): the human's action in progress, started at an idle/battle prompt:
    /// (index into `human_decisions` of that pick, `bot_moves` then). None once the action ends (the next idle/battle
    /// prompt) or before the first one.
    action: Option<(usize, usize)>,
    /// Bot decisions stepped so far: a Cancel is refused once the bot answered during the action (it saw it).
    bot_moves: usize,
    model_label: String,
    last_sig: (usize, i32, i32, bool, bool, i64),
    responses: Vec<ResponseRec>,
    cur_opts: Vec<i32>,
    resp_start: Instant,
    seed: u64,
    main0: Vec<u32>,
    extra0: Vec<u32>,
    main1: Vec<u32>,
    extra1: Vec<u32>,
    side0: Vec<u32>,
    side1: Vec<u32>,
    mode: String,
    bot_mode: String,
    product: Option<ProductDriver>,
    prod_len: usize,
    control: Option<Arc<ProductControl>>,
    value: Option<decklight_onnx::Pooled>,
    wc_map: Option<Option<WcMap>>,
    winchance: Option<Value>,
    wc_n: u64,
    /// Product duels only: the socket channel, so frames go out as they happen (not after the bot's whole turn)
    /// and the thinking ticker can run while the search blocks.
    live_tx: Option<Sink>,
    /// The finished game's record (what-if names it as its source).
    record_path: Option<PathBuf>,
    /// The finished game's review (Python's review.json cache); review_frame and what-if need it.
    review: Option<Value>,
    /// Practice games started by what-if: `{source, k, option, turn}` (carried across undos).
    what_if: Option<Value>,
    /// A puzzle game: `{id, start_turn, goal, n_plays, status}` (Python's `self.puzzle`).
    puzzle: Option<Value>,
    /// Record meta `puzzle_result` ("solved"/"failed").
    puzzle_result: Option<String>,
}

/// One human decision with more than one option, from a replay of the finished game
/// (`review.collect_human_decisions`), plus what the review needs at that position.
struct ReviewDecision {
    prefix: Vec<i32>,
    choice: usize,
    opt_kinds: Vec<i32>,
    turn: i32,
    phase: &'static str,
    kind: String,
    auto: bool,
    is_chain: bool,
    /// Masked softmax of the policy on the human's observation there.
    probs: Vec<f64>,
    legal: Vec<usize>,
    /// The prompt's options (labels and cards), as `build_prompt` gives them.
    options: Vec<Value>,
}

const KIND_ACTIVATE: i32 = 6;
const KIND_PASS: i32 = 11;
const KIND_NO: i32 = 13;
const REVIEW_MAX_CANDS: usize = 5;

/// `review.mark_for_delta`.
pub(crate) fn mark_for_delta(delta_pts: f64) -> &'static str {
    if delta_pts >= -3.0 {
        "Good"
    } else if delta_pts >= -7.0 {
        "Inaccuracy"
    } else if delta_pts >= -15.0 {
        "Mistake"
    } else {
        "Blunder"
    }
}

/// Python's `round(x, n)` (via the shortest decimal at n places, like `round4`).
fn round_n(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.n$}").parse::<f64>().unwrap_or(x)
}

/// Python's `int(msg.get(key))` for the review messages: ints, floats (truncated) and numeric strings.
fn int_field(msg: &Value, key: &str) -> Option<i64> {
    match msg.get(key)? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// The thinking ticker (Python `_think_loop` / `_send_peek`): once a second while the bot searches, the pre-move
/// snapshot with `thinking.seconds`. It never touches the duel; dropping it stops and joins the thread.
struct Ticker {
    stop: Option<std::sync::mpsc::Sender<()>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Ticker {
    fn start(peek: Value, sink: Sink) -> Ticker {
        let (stop, stop_rx) = std::sync::mpsc::channel::<()>();
        let since = Instant::now();
        let join = std::thread::Builder::new()
            .name("thinking-ticker".into())
            .spawn(move || {
                while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                    stop_rx.recv_timeout(Duration::from_secs(1))
                {
                    let mut msg = peek.clone();
                    msg["thinking"] = json!({"seconds": since.elapsed().as_secs_f64()});
                    sink.send(Outgoing::Frame(msg));
                }
            })
            .map_err(|e| eprintln!("ALERT web: could not start the thinking ticker: {e}"))
            .ok();
        Ticker { stop: Some(stop), join }
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DuelResult {
    pub winner: Option<i32>,
    pub turns: i32,
    pub win_reason: i32,
    pub surrendered: bool,
    pub disconnected: bool,
}

pub enum Bot {
    Stream(SplitMix),
    Random(SplitMix),
    Raw,
}

pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn random_u64() -> u64 {
    let h = random_hex(8);
    u64::from_str_radix(&h, 16).unwrap_or(1)
}

impl Game {
    #[allow(clippy::too_many_arguments)]
    fn new(
        app: &App,
        seed: u64,
        human_seat: i32,
        first: &DeckSpec,
        second: &DeckSpec,
        bot: Bot,
        mode: &str,
        practice: bool,
        product_launch: Option<ProductLaunch>,
    ) -> Result<Game, String> {
        let mut main0 = first.main.clone();
        let mut main1 = second.main.clone();
        decklight_cards::shuffle_main(&mut main0, seed, 0);
        decklight_cards::shuffle_main(&mut main1, seed, 1);
        let cfg = DuelConfig {
            main0: main0.clone(),
            extra0: first.extra.clone(),
            main1: main1.clone(),
            extra1: second.extra.clone(),
            seed,
            flags: TCG_FLAGS,
            lp: 8000,
            hand: 5,
            draw: 1,
            seed_state: None,
            team_order: [0, 1],
            edopro_add: false,
        };
        let sides = (first.side.clone(), second.side.clone());
        Game::with_config(app, cfg, seed, human_seat, sides, bot, mode, practice, product_launch, (Vec::new(), Vec::new()))
    }

    /// A duel on an already shuffled config; `prefix` = (product-space choices, their `(actor, options)`) the
    /// product worker replays without searching (undo).
    #[allow(clippy::too_many_arguments)]
    fn with_config(
        app: &App,
        cfg: DuelConfig,
        seed: u64,
        human_seat: i32,
        sides: (Vec<u32>, Vec<u32>),
        bot: Bot,
        mode: &str,
        practice: bool,
        product_launch: Option<ProductLaunch>,
        prefix: (Vec<i32>, Vec<(i32, usize)>),
    ) -> Result<Game, String> {
        let mut env = Env::new(app.engine, &cfg, false).map_err(|e| format!("could not start the duel: {e}"))?;
        env.set_record_trace(true); // cues read the raw message trace
        let mut obs = Obs::zeroed();
        env.reset(&mut obs).map_err(|e| format!("could not start the duel: {e}"))?;
        let (product, control, value) = match product_launch {
            Some(p) => {
                let driver = ProductDriver::start(
                    app.engine,
                    &cfg,
                    1 - human_seat,
                    seed,
                    seed,
                    p.search,
                    p.replay_threads,
                    prefix.0,
                    prefix.1,
                    Some(p.session),
                )?;
                let control = driver.control.clone();
                (Some(driver), Some(control), Some(p.value))
            }
            None => (None, None, None),
        };
        let bot_mode = match mode {
            "random" => "random",
            "fast" => "raw",
            "product" => "product",
            _ => "raw",
        }
        .to_string();
        Ok(Game {
            human_seat,
            deck_names: Default::default(),
            bot_seat: 1 - human_seat,
            main0: cfg.main0.clone(),
            extra0: cfg.extra0.clone(),
            main1: cfg.main1.clone(),
            extra1: cfg.extra1.clone(),
            side0: sides.0,
            side1: sides.1,
            cfg,
            env,
            obs,
            bot,
            stats: DuelStats::new(human_seat, 8000),
            result: None,
            finished: false,
            end_reason: String::new(),
            advancing: false,
            cue_sent: 0,
            log_sent: 0,
            prompt_id: 0,
            last_prompt_seq: -1,
            auto_passed: BTreeSet::new(),
            human_decisions: Vec::new(),
            practice,
            hints_used: 0,
            undos: 0,
            action: None,
            bot_moves: 0,
            model_label: app.model_label.clone(),
            last_sig: (usize::MAX, -1, -1, false, false, 0),
            responses: Vec::new(),
            cur_opts: Vec::new(),
            resp_start: Instant::now(),
            seed,
            // Python's record meta uses the game mode ("product"/"fast"/"random"); `practice` is a separate flag.
            mode: mode.to_string(),
            bot_mode,
            product,
            prod_len: 0,
            control,
            value,
            wc_map: None,
            winchance: None,
            wc_n: 0,
            live_tx: None,
            record_path: None,
            review: None,
            what_if: None,
            puzzle: None,
            puzzle_result: None,
        })
    }

    fn need_human(&self) -> bool {
        !self.env.done() && self.env.player() == self.human_seat && self.env.num_options() > 1
    }

    fn step_index(&mut self, index: i32) -> Result<(), String> {
        let dec = self.env.decision().ok_or("no pending decision")?;
        let player = dec.player;
        let before = self.env.num_responses();
        if self.cur_opts.is_empty() {
            self.resp_start = Instant::now();
        }
        self.cur_opts.push(index);
        self.env.step(index, &mut self.obs).map_err(|e| e.to_string())?;
        let after = self.env.num_responses();
        for k in before..after {
            let r = self.env.response(k);
            let opt = std::mem::take(&mut self.cur_opts);
            let ms = self.resp_start.elapsed().as_millis() as u64;
            self.responses.push(ResponseRec { seat: player, r, opt, ms });
            self.resp_start = Instant::now();
        }
        Ok(())
    }

    fn settle(&mut self) -> Result<usize, String> {
        let mut n = 0;
        while !self.env.done() && self.env.num_options() == 1 {
            self.step_index(0)?;
            n += 1;
        }
        Ok(n)
    }

    fn step_bot(&mut self, app: &App) -> Result<(), String> {
        let n = self.env.num_options();
        if n <= 0 {
            return Err("bot decision with no options".into());
        }
        if n > 1 {
            self.bot_moves += 1;
        }
        if let Some(driver) = self.product.as_ref() {
            let pre = (self.env.player(), n as usize);
            let idx = driver.next_bot()?;
            driver.check_decision(self.prod_len, pre)?;
            self.step_index(idx)?;
            self.prod_len += 1;
            return Ok(());
        }
        let idx = match &mut self.bot {
            Bot::Stream(r) | Bot::Random(r) => (r.next() % n as u64) as i32,
            Bot::Raw => app.raw_choose(&self.obs, n as usize)?,
        };
        self.step_index(idx)
    }

    /// One human pick. In product mode a multi-option pick is also fed to the search worker (singletons are not in
    /// its product space, like `DuelSession.step_human`).
    fn human_step(&mut self, index: i32) -> Result<(), String> {
        let pre = (self.env.player(), self.env.num_options() as usize);
        self.step_index(index)?;
        if let Some(driver) = self.product.as_ref()
            && pre.1 > 1
        {
            driver.feed_human(index)?;
            driver.check_decision(self.prod_len, pre)?;
            self.prod_len += 1;
        }
        Ok(())
    }

    /// Value_p2 on the human's pending observation, calibrated (port of `Server._compute_winchance`).
    fn compute_winchance(&mut self, app: &App) {
        if !self.need_human() {
            return;
        }
        if self.wc_map.is_none() {
            let path = app.root.join("src/decklight/web/winchance_calib.json");
            self.wc_map = Some(WcMap::load(&path).ok());
        }
        let Some(Some(map)) = self.wc_map.as_ref() else {
            return;
        };
        let Some(value) = self.value.as_mut() else { return };
        let v = match value.values(&[&self.obs]) {
            Ok(mut xs) if !xs.is_empty() => xs.remove(0) as f64,
            _ => return,
        };
        let turn = self.obs.global[2];
        let p = map.p(v, turn);
        self.wc_n += 1;
        let entry = self.winchance.get_or_insert_with(|| json!({"p": 0.0, "turn": 0, "history": []}));
        entry["p"] = json!(p);
        entry["turn"] = json!(turn);
        if let Some(h) = entry.get_mut("history").and_then(Value::as_array_mut) {
            h.push(json!([turn, p]));
        }
        self.stats.note_winchance(turn as i64, p);
    }

    /// Practice "Ask the bot" (port of `Server.on_hint` + `analysis.hints.compute_hint`): the policy's top 3 at the
    /// human's prompt, each with its one-step-lookahead win chance when the win-chance switch is on. Only the
    /// human's own observations reach the nets (the review's fairness rule).
    fn hint(&mut self, app: &App, winchance_on: bool) -> Vec<Value> {
        if !self.practice {
            return Inner::error("Hints are only in practice mode");
        }
        // Python checks the env, not its own `finished`: a surrendered game is still at its last prompt, so a
        // hint there answers (the review's fairness rule only needs the human's observation).
        if self.env.done() || !self.need_human() || self.advancing {
            return Inner::error("no prompt is waiting for you right now");
        }
        if self.value.is_none() {
            return Inner::error("Hints need the full bot");
        }
        let calib = if winchance_on {
            if self.wc_map.is_none() {
                let path = app.root.join("src/decklight/web/winchance_calib.json");
                self.wc_map = Some(match WcMap::load(&path) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!("ALERT web: hint calib unavailable: {e}");
                        None
                    }
                });
            }
            self.wc_map.clone().flatten()
        } else {
            None
        };
        let t0 = Instant::now();
        let out = match self.hint_options(app, calib.as_ref()) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ALERT web: hint crashed: {e}");
                return Inner::error(format!("hint failed: {e} (see server log)"));
            }
        };
        let ms = (t0.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;
        let pid = self.prompt_id;
        let labels: HashMap<i64, String> = app
            .ctx
            .build_prompt(&self.env, self.human_seat, pid)
            .and_then(|p| p.get("options").and_then(Value::as_array).cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|o| Some((o.get("index")?.as_i64()?, o.get("label")?.as_str()?.to_string())))
            .collect();
        let options: Vec<Value> = out
            .into_iter()
            .map(|(i, p, est)| {
                json!({
                    "index": i,
                    "label": labels.get(&(i as i64)).cloned().unwrap_or_else(|| format!("option {i}")),
                    "p": p,
                    "est": est,
                })
            })
            .collect();
        self.hints_used += 1;
        vec![json!({"t": "hint", "prompt": pid, "options": options, "ms": ms})]
    }

    /// `(index, p, est)` for the top 3 legal options by policy probability (ties: lower index first).
    fn hint_options(&mut self, app: &App, calib: Option<&WcMap>) -> Result<Vec<(usize, f64, Option<f64>)>, String> {
        let nets = self.value.as_mut().ok_or("no nets")?;
        let logits = nets.policy_logits(&[&self.obs])?.into_iter().next().ok_or("no policy output")?;
        let probs = masked_softmax(&logits, &self.obs.action_mask);
        let mut legal: Vec<usize> = (0..self.obs.action_mask.len()).filter(|&j| self.obs.action_mask[j] != 0).collect();
        legal.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        legal.truncate(3);
        let est: Vec<Option<f64>> = match calib {
            Some(map) => self.estimate_candidates(app, &self.env.choices(), &legal, map)?.into_iter().map(Some).collect(),
            None => vec![None; legal.len()],
        };
        Ok(legal
            .into_iter()
            .zip(est)
            .map(|(i, e)| (i, round4(probs[i]), e.map(round4)))
            .collect())
    }

    /// Win chance per candidate by one-step lookahead (see [`estimate_candidates`]).
    fn estimate_candidates(
        &mut self,
        app: &App,
        prefix: &[i32],
        cands: &[usize],
        map: &WcMap,
    ) -> Result<Vec<f64>, String> {
        let nets = self.value.as_mut().ok_or("no nets")?;
        estimate_candidates(app, &self.cfg, self.human_seat, nets, prefix, cands, map, &|| false)
    }

    /// The pending human prompt is a step inside an action the human started (an idle/battle pick), the bot has not
    /// decided since, and the full bot is loaded (the rewind is a replay, like undo).
    fn can_cancel(&self) -> bool {
        let start_kind = |m: i32| m == C::MSG_SELECT_IDLECMD as i32 || m == C::MSG_SELECT_BATTLECMD as i32;
        let Some((hd, bots)) = self.action else { return false };
        !self.advancing
            && self.need_human()
            && self.value.is_some()
            && bots == self.bot_moves
            && hd < self.human_decisions.len()
            && self.env.decision().is_some_and(|d| !start_kind(d.msg_type))
    }

    /// Cancel the action in progress: a replay back to the idle/battle prompt where it started (the undo machinery,
    /// in any bot game). Not counted as an undo: the bot has not seen the action (`can_cancel`).
    fn cancel(&mut self, app: &App) -> Result<Game, String> {
        if self.session_stopping() {
            return Err("product worker stopped".into());
        }
        if !self.can_cancel() {
            return Err("Nothing to cancel".into());
        }
        let (hd, _) = self.action.expect("checked by can_cancel");
        let target = self.human_decisions[hd];
        let prefix_host: Vec<i32> = self.env.choices()[..target].to_vec();
        let mut g = self.restart_from(app, &prefix_host, None, self.what_if.clone())?;
        g.practice = self.practice;
        g.human_decisions = self.human_decisions[..hd].to_vec();
        g.auto_passed = self.auto_passed.iter().copied().filter(|&i| i <= target).collect();
        g.hints_used = self.hints_used;
        g.undos = self.undos;
        g.bot_moves = self.bot_moves;
        g.puzzle = self.puzzle.clone();
        g.puzzle_result = self.puzzle_result.clone();
        Ok(g)
    }

    /// Practice undo (port of `Server.on_undo` + `_restart_from`): a new game replayed to the human's previous real
    /// decision. The host replays the choices; the product worker gets them as a prefix it never re-searches, with
    /// its lockstep log pre-filled. On any failure the current game stays (the caller sends the error).
    fn undo(&mut self, app: &App) -> Result<Game, String> {
        if self.session_stopping() {
            // A surrender already cut the search: do not load a fresh worker (seconds on a cold pool) for it.
            return Err("product worker stopped".into());
        }
        if !self.practice {
            return Err("Undo is only in practice mode".into());
        }
        if self.env.done() {
            return Err("no active game (send new_game first)".into());
        }
        // Python tests the launcher, which a surrendered game still has; here the value nets are what restart_from
        // needs (the worker is closed on surrender and a fresh one is started by the restart).
        if self.value.is_none() {
            return Err("Undo needs the full bot".into());
        }
        if self.advancing {
            return Err("Wait for the bot to finish".into());
        }
        if !self.need_human() {
            return Err("no prompt is waiting for you right now".into());
        }
        let Some(&target) = self.human_decisions.last() else {
            return Err("Nothing to undo".into());
        };
        let prefix_host: Vec<i32> = self.env.choices()[..target].to_vec();
        let mut g = self.restart_from(app, &prefix_host, None, self.what_if.clone())?;
        g.human_decisions = self.human_decisions[..self.human_decisions.len() - 1].to_vec();
        g.auto_passed = self.auto_passed.iter().copied().filter(|&i| i <= target).collect();
        g.hints_used = self.hints_used;
        g.undos = self.undos + 1;
        g.puzzle = self.puzzle.clone();
        g.puzzle_result = self.puzzle_result.clone();
        Ok(g)
    }

    /// Port of `Server._restart_from` (undo and what-if): a new practice game on this game's deal, replayed through
    /// `prefix_host`. The product worker gets the prefix's multi-option choices as a prefix it never re-searches,
    /// with its lockstep log pre-filled. With `apply_option` (what-if) that option is then played at the prefix's
    /// end; without it the position must be a human prompt (undo). On any failure this game stays as it was.
    fn restart_from(
        &mut self,
        app: &App,
        prefix_host: &[i32],
        apply_option: Option<i64>,
        what_if: Option<Value>,
    ) -> Result<Game, String> {
        // host_to_product + the prefix's (actor, options), from a host-space replay.
        let mut env = Env::new(app.engine, &self.cfg, false)?;
        let mut obs = Obs::zeroed();
        let mut pending = env.reset(&mut obs)?;
        let (mut product_prefix, mut prefix_decisions) = (Vec::new(), Vec::new());
        for &c in prefix_host {
            if !pending {
                break;
            }
            let n = env.num_options();
            if n > 1 {
                product_prefix.push(c);
                prefix_decisions.push((env.player(), n as usize));
            }
            pending = env.step(c, &mut obs)?;
        }
        if !pending || env.done() {
            return Err("the position is already game over".into());
        }
        let pre = (env.player(), env.num_options() as usize);
        let turn = env.duel_state()?.turn;
        drop(env);
        let what_if = what_if.map(|mut w| {
            if w.get("turn").is_none() {
                w["turn"] = json!(turn);
            }
            w
        });
        if self.session_stopping() {
            // Surrender/new game landed while the prefix replayed: skip the cold ONNX load.
            return Err("product worker stopped".into());
        }
        let search = app.onnx_pool.take().map_err(|e| format!("could not start the product opponent: {e}"))?;
        let value = self.value.take().expect("checked above");
        let session = self.control.as_ref().and_then(|c| c.parent.clone()).expect("product game without a session");
        let launch = ProductLaunch { search, replay_threads: app.replay_threads, value, session };
        let sides = (self.side0.clone(), self.side1.clone());
        let n_prefix = product_prefix.len();
        let started = Game::with_config(
            app,
            self.cfg.clone(),
            self.seed,
            self.human_seat,
            sides,
            Bot::Raw,
            "product",
            true,
            Some(launch),
            (product_prefix, prefix_decisions),
        );
        let mut g = match started {
            Ok(mut g) => {
                g.deck_names = self.deck_names.clone();
                g
            }
            Err(e) => {
                // with_config consumed the value nets; take a fresh set so this game keeps its win chance.
                self.value = app.onnx_pool.take().ok();
                return Err(format!("could not start the replay game: {e}"));
            }
        };
        let replayed = (|| -> Result<(), String> {
            for &c in prefix_host {
                if g.env.done() {
                    break;
                }
                g.step_index(c)?;
            }
            if g.env.done() {
                return Err("the replay position is already game over".into());
            }
            let now = (g.env.player(), g.env.num_options() as usize);
            if now != pre {
                return Err(format!(
                    "replay diverged: host (player={}, options={}) vs prefix (player={}, options={})",
                    now.0, now.1, pre.0, pre.1
                ));
            }
            if let Some(opt) = apply_option {
                if opt < 0 || opt >= pre.1 as i64 {
                    return Err(format!("illegal option {opt} of {}", pre.1));
                }
                g.step_index(opt as i32)?;
                g.prod_len = n_prefix + 1;
                let driver = g.product.as_ref().expect("product game");
                driver.feed_human(opt as i32)?;
                return driver.check_decision(n_prefix, pre);
            }
            if !g.need_human() {
                return Err("the rewind position is not a human prompt".into());
            }
            g.prod_len = n_prefix;
            g.product.as_ref().expect("product game").check_decision(n_prefix, pre)
        })();
        if let Err(e) = replayed {
            eprintln!("ALERT web: replay failed: {e}");
            self.value = g.value.take();
            return Err(format!("replay failed: {e}"));
        }
        // The replayed responses are this game's own: keep their original think times.
        for (new, old) in g.responses.iter_mut().zip(&self.responses) {
            if new.seat == old.seat && new.r == old.r && new.opt == old.opt {
                new.ms = old.ms;
            }
        }
        g.model_label = self.model_label.clone();
        g.live_tx = self.live_tx.clone();
        g.what_if = what_if;
        Ok(g)
    }

    /// `review.collect_human_decisions` on this game's choices, with the policy and the prompt at each decision.
    fn review_decisions(&mut self, app: &App, with_policy: bool) -> Result<Vec<ReviewDecision>, String> {
        let choices = self.env.choices();
        let hs = self.human_seat;
        let mut env = Env::new(app.engine, &self.cfg, false)?;
        let mut obs = Obs::zeroed();
        let mut pending = env.reset(&mut obs)?;
        let mut out = Vec::new();
        for &ch in &choices {
            if !pending || env.done() {
                break;
            }
            if env.player() == hs && env.num_options() > 1 {
                let dec = env.decision().ok_or("no pending decision")?;
                let st = env.duel_state()?;
                let prompt = app.ctx.build_prompt(&env, hs, 1);
                let kind = prompt.as_ref().and_then(|p| p.get("kind")?.as_str()).unwrap_or("card").to_string();
                let options = prompt
                    .as_ref()
                    .and_then(|p| p.get("options")?.as_array().cloned())
                    .unwrap_or_default();
                let (probs, legal) = if with_policy {
                    let nets = self.value.as_mut().ok_or("no nets")?;
                    let logits = nets.policy_logits(&[&obs])?.into_iter().next().ok_or("no policy output")?;
                    let legal = (0..obs.action_mask.len()).filter(|&j| obs.action_mask[j] != 0).collect();
                    (masked_softmax(&logits, &obs.action_mask), legal)
                } else {
                    (Vec::new(), Vec::new())
                };
                let prefix = env.choices();
                out.push(ReviewDecision {
                    auto: self.auto_passed.contains(&(prefix.len() + 1)),
                    prefix,
                    choice: ch as usize,
                    opt_kinds: dec.options.iter().map(|o| o.kind).collect(),
                    turn: st.turn,
                    phase: decklight_view::phase_name(st.phase).unwrap_or("main1"),
                    kind,
                    is_chain: dec.msg_type == C::MSG_SELECT_CHAIN as i32,
                    probs,
                    legal,
                    options,
                });
            }
            pending = env.step(ch, &mut obs)?;
        }
        Ok(out)
    }

    /// Port of `review.review_game`: for every human decision with more than one option, the one-step-lookahead
    /// win chance of the human's choice against the policy's top picks (plus the activations in a chain window),
    /// marked by the loss in points. `None` when the session stopped part way.
    fn review(
        &mut self,
        app: &App,
        map: &WcMap,
        stop: &dyn Fn() -> bool,
        progress: &dyn Fn(usize, usize),
    ) -> Result<Option<Value>, String> {
        let t0 = Instant::now();
        let t1 = Instant::now();
        let decisions = self.review_decisions(app, true)?;
        let fwd_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let mut val_ms = 0.0;
        let mut entries: Vec<Value> = Vec::new();
        let mut deltas: Vec<f64> = Vec::new();
        let mut counts: BTreeMap<&str, i64> =
            ["Good", "Inaccuracy", "Mistake", "Blunder", "Missed response"].into_iter().map(|m| (m, 0)).collect();
        let mut missed_n = 0;
        for (k, d) in decisions.iter().enumerate() {
            if stop() {
                return Ok(None);
            }
            let probs = &d.probs;
            let mut order = d.legal.clone();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            let mut cands = vec![d.choice];
            for &j in &order {
                if !cands.contains(&j) {
                    cands.push(j);
                }
                if cands.len() >= 3 {
                    break;
                }
            }
            if d.is_chain {
                for (j, &kd) in d.opt_kinds.iter().enumerate() {
                    if kd == KIND_ACTIVATE && !cands.contains(&j) && j != d.choice {
                        cands.push(j);
                    }
                    if cands.len() >= REVIEW_MAX_CANDS {
                        break;
                    }
                }
            }
            cands.truncate(REVIEW_MAX_CANDS);
            let t1 = Instant::now();
            let est = self.estimate_candidates(app, &d.prefix, &cands, map)?;
            val_ms += t1.elapsed().as_secs_f64() * 1000.0;
            // Ties keep the policy order: among equal estimates pick the lower policy rank.
            let best_p = est.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let rank = |c: usize| order.iter().position(|&x| x == c).unwrap_or(999);
            let best_j = (0..est.len())
                .filter(|&j| est[j] == best_p)
                .min_by_key(|&j| rank(cands[j]))
                .ok_or("no candidates")?;
            let agent_idx = cands[best_j];
            let human_p = est[0];
            let agent_p = est[best_j];
            let delta_pts = round_n((human_p - agent_p) * 100.0, 1);
            let had_act = d.opt_kinds.contains(&KIND_ACTIVATE);
            let human_kind = d.opt_kinds.get(d.choice).copied();
            let human_pass = matches!(human_kind, Some(KIND_PASS | KIND_NO));
            let p_activate: f64 = d
                .opt_kinds
                .iter()
                .enumerate()
                .filter(|&(_, &kd)| kd == KIND_ACTIVATE)
                .fold(0.0, |s, (j, _)| s + probs[j]);
            let act_better = cands
                .iter()
                .enumerate()
                .any(|(j, &c)| d.opt_kinds.get(c) == Some(&KIND_ACTIVATE) && est[j] > human_p);
            let missed = d.is_chain && had_act && human_pass && p_activate >= 0.5 && act_better;
            let mark = if missed { "Missed response" } else { mark_for_delta(delta_pts) };
            let opt = |i: usize| d.options.iter().find(|o| o.get("index").and_then(Value::as_u64) == Some(i as u64));
            let label = |i: usize| {
                opt(i)
                    .and_then(|o| o.get("label")?.as_str().map(str::to_string))
                    .unwrap_or_else(|| format!("option {i}"))
            };
            let card = |i: usize| {
                let c = opt(i).and_then(|o| o.get("card")).filter(|c| c.is_object());
                json!({
                    "code": c.and_then(|c| c.get("code")).cloned().unwrap_or(json!(0)),
                    "name": c.and_then(|c| c.get("name")).cloned().unwrap_or(Value::Null),
                })
            };
            let note = if missed {
                let agent_label = opt(agent_idx).and_then(|o| o.get("label")?.as_str()).unwrap_or("");
                format!("You passed; the agent would activate {agent_label} ({delta_pts:+.1} pts)")
            } else {
                String::new()
            };
            entries.push(json!({
                "k": k,
                "turn": d.turn,
                "phase": d.phase,
                "kind": d.kind,
                "mark": mark,
                "delta_pts": delta_pts,
                "your": {"label": label(d.choice), "card": card(d.choice)},
                "agent": {"label": label(agent_idx), "card": card(agent_idx)},
                "policy_p_yours": round4(probs[d.choice]),
                "policy_p_agent": round4(probs[agent_idx]),
                "missed_response": missed,
                "auto_passed": d.auto,
                "note": note,
                "est_yours": round4(human_p),
                "est_agent": round4(agent_p),
            }));
            *counts.entry(mark).or_insert(0) += 1;
            missed_n += missed as i64;
            deltas.push(delta_pts);
            progress(k + 1, decisions.len());
        }
        // Python's `sum` adds left to right.
        let acc = if deltas.is_empty() {
            100.0
        } else {
            deltas.iter().fold(0.0, |s, d| s + (100.0 + d).max(0.0)) / deltas.len() as f64
        };
        let n = entries.len();
        Ok(Some(json!({
            "decisions": entries,
            "counts": counts,
            "accuracy": round_n(acc, 1),
            "missed": missed_n,
            "timing": {
                "seconds": round_n(t0.elapsed().as_secs_f64(), 2),
                "fwd_ms": round_n(fwd_ms, 1),
                "value_ms": round_n(val_ms, 1),
                "decisions": n,
            },
        })))
    }

    /// Port of `Server.on_review_frame`: the board and options at reviewed decision `k`, with its review entry.
    fn review_frame(&mut self, app: &App, msg: &Value) -> Vec<Value> {
        if !self.finished {
            return Inner::error("no finished game to review (finish a game first)");
        }
        let Some(k) = int_field(msg, "k") else {
            return Inner::error("review_frame.k must be a decision number");
        };
        let Some(review) = self.review.as_ref() else {
            return Inner::error("no review yet (send review first)");
        };
        let decs = review.get("decisions").and_then(Value::as_array).cloned().unwrap_or_default();
        if k < 0 || k as usize >= decs.len() {
            return Inner::error(format!("unknown decision {k}"));
        }
        let dlist = match self.review_decisions(app, false) {
            Ok(d) => d,
            Err(e) => return Inner::error(format!("review_frame failed: {e}")),
        };
        let Some(d) = dlist.get(k as usize) else {
            return Inner::error(format!("unknown decision {k}"));
        };
        let replayed = (|| -> Result<Env<'static>, String> {
            let mut env = Env::new(app.engine, &self.cfg, false)?;
            let mut obs = Obs::zeroed();
            env.replay(&d.prefix, &mut obs)?;
            Ok(env)
        })();
        let env = match replayed {
            Ok(e) => e,
            Err(e) => return Inner::error(format!("review_frame failed: {e}")),
        };
        let view = app.ctx.build_view(&env, self.human_seat, false);
        let opts = app
            .ctx
            .build_prompt(&env, self.human_seat, 999)
            .and_then(|p| p.get("options").cloned())
            .unwrap_or(json!([]));
        vec![json!({"t": "review_frame", "k": k, "view": view, "options": opts, "entry": decs[k as usize]})]
    }

    /// Port of `Server.on_whatif`'s checks: the prefix and option to restart from, or the error to send.
    fn whatif_target(&mut self, app: &App, msg: &Value) -> Result<(Vec<i32>, i64, i64), String> {
        if !self.finished || self.result.is_none() {
            return Err("no finished game to review (finish a game first)".into());
        }
        if self.review.is_none() {
            return Err("no review yet (send review first)".into());
        }
        let k = int_field(msg, "k").ok_or("whatif.k must be a decision number")?;
        let option = int_field(msg, "option").ok_or("whatif.option must be an option number")?;
        if self.record_path.is_none() {
            return Err("no finished game to review (finish a game first)".into());
        }
        let dlist = self.review_decisions(app, false)?;
        if k < 0 || k as usize >= dlist.len() {
            return Err(format!("unknown decision {k}"));
        }
        if self.bot_mode != "product" || self.value.is_none() {
            return Err("What-if needs the full bot".into());
        }
        Ok((dlist[k as usize].prefix.clone(), k, option))
    }

    /// What-if: a new practice game from reviewed decision `k` with `option` played instead.
    fn whatif(&mut self, app: &App, msg: &Value) -> Result<Game, String> {
        let (prefix, k, option) = self.whatif_target(app, msg)?;
        let source = self
            .record_path
            .as_ref()
            .and_then(|p| p.file_name()?.to_str()?.split('.').next().map(str::to_string))
            .unwrap_or_default();
        let w = json!({"source": source, "k": k, "option": option});
        let g = self.restart_from(app, &prefix, Some(option), Some(w))?;
        Ok(g)
    }

    fn current_prompt_id(&mut self) -> i64 {
        let seq = self.env.decision_seq();
        bump_prompt_id(&mut self.last_prompt_seq, &mut self.prompt_id, seq)
    }

    fn auto_pass_index(&self) -> Option<i32> {
        if !self.need_human() {
            return None;
        }
        let raw = self.env.pending_msg();
        let msg = M::parse_one(&raw).ok()?;
        let (forced, spe_count, hint_timing, hint_timing_other) = match msg.body {
            MsgBody::SelectChain { forced, spe_count, hint_timing, hint_timing_other, .. } => {
                (forced, spe_count, hint_timing, hint_timing_other)
            }
            _ => return None,
        };
        let chain_empty = decklight_core::query::parse_chain(&self.env.query_field().ok()?)
            .map(|c| c.is_empty())
            .unwrap_or(false);
        if !quiet_chain_window(forced, spe_count, hint_timing, hint_timing_other, chain_empty) {
            return None;
        }
        let history = self.env.history_log();
        let mut last = None;
        for i in (0..history.len()).rev() {
            if history[i][10] == self.human_seat {
                last = Some(i);
                break;
            }
        }
        let start = last.map(|i| i + 1).unwrap_or(0);
        for e in &history[start..] {
            if !matches!(e[0], 8 | 9 | 10 | 11 | 13) {
                return None;
            }
        }
        let dec = self.env.decision()?;
        for (i, o) in dec.options.iter().enumerate() {
            if o.kind == 11 {
                return Some(i as i32);
            }
        }
        None
    }

    fn sig(&self) -> (usize, i32, i32, bool, bool, i64) {
        let st = self.env.duel_state().ok();
        let (turn, phase) = st.map(|s| (s.turn, s.phase)).unwrap_or((0, 0));
        (
            self.env.history_log().len(),
            turn,
            phase,
            self.env.done(),
            self.need_human() && !self.advancing,
            self.wc_n as i64,
        )
    }

    fn state_message(&mut self, app: &App, full_log: bool, peek: bool, winchance_on: bool) -> Value {
        let view = app.ctx.build_view(&self.env, self.human_seat, false);
        let mut prompt = Value::Null;
        if self.need_human() && !self.advancing {
            let pid = self.current_prompt_id();
            if let Some(p) = app.ctx.build_prompt(&self.env, self.human_seat, pid) {
                prompt = p;
            }
        }
        let mut log = app
            .ctx
            .build_events(&self.env, self.human_seat, if full_log { 0 } else { self.log_sent });
        for e in log.iter_mut() {
            let n = e.get("n").and_then(Value::as_u64).unwrap_or(0) as usize;
            if self.auto_passed.contains(&n) {
                e["actor"] = json!("system");
                e["text"] = json!("(auto-passed)");
                e["card"] = Value::Null;
                e["kind"] = json!("auto_pass");
            }
        }
        let cues: Vec<Value> = if peek {
            Vec::new()
        } else {
            let (c, next) = decklight_view::build_cues(&self.env, &app.ctx, self.human_seat, self.cue_sent, false);
            self.cue_sent = next;
            c
        };
        self.stats.feed_cues(&cues);
        self.stats.note_view(Some(&view));
        if !full_log && !peek {
            self.log_sent = self.env.history_log().len();
        }
        let result = if self.finished {
            self.result.map(|r| {
                let winner = if r.winner == Some(self.human_seat) { "you" } else { "bot" };
                let reason = if self.end_reason.is_empty() { "win" } else { self.end_reason.as_str() };
                json!({"winner": winner, "reason": reason})
            })
        } else {
            None
        };
        json!({
            "t": "state",
            "view": view,
            "prompt": prompt,
            "log": log,
            "cues": cues,
            "stats": self.stats.stats_message(winchance_on),
            "winchance": if winchance_on { self.winchance.clone().unwrap_or(Value::Null) } else { Value::Null },
            "thinking": Value::Null,
            "result": result,
            "practice": self.practice,
            "can_undo": self.practice && !self.advancing && self.need_human() && !self.human_decisions.is_empty(),
            "can_cancel": self.can_cancel(),
            "what_if": self.what_if.clone().unwrap_or(Value::Null),
            "puzzle": self.puzzle.clone().unwrap_or(Value::Null),
            "model_label": BASE_MODEL_LABEL,
        })
    }

    fn send_state(&mut self, app: &App, full_log: bool, peek: bool, winchance_on: bool, frames: &mut Vec<Value>) {
        let msg = self.state_message(app, full_log, peek, winchance_on);
        frames.push(msg);
        if !peek {
            self.last_sig = self.sig();
        }
    }

    fn advance(&mut self, app: &App, chain_mode: &str, winchance_on: bool, frames: &mut Vec<Value>) {
        self.advancing = true;
        loop {
            if self.control.as_ref().is_some_and(|c| c.stopping()) {
                self.advancing = false;
                return;
            }
            if self.puzzle_failed() {
                self.fail_puzzle(app, winchance_on, frames);
                return;
            }
            if self.env.done() {
                self.finish(app, winchance_on, frames);
                return;
            }
            let moved = match self.settle() {
                Ok(n) => n,
                Err(e) => {
                    if self.control.as_ref().is_some_and(|c| c.stopping()) {
                        self.advancing = false;
                        return;
                    }
                    eprintln!("ALERT web: settle failed: {e}");
                    self.finish(app, winchance_on, frames);
                    return;
                }
            };
            if self.puzzle_failed() {
                self.fail_puzzle(app, winchance_on, frames);
                return;
            }
            if self.env.done() {
                self.finish(app, winchance_on, frames);
                return;
            }
            if self.need_human() && chain_mode == "auto" {
                if let Some(i) = self.auto_pass_index() {
                    let had = self
                        .env
                        .decision()
                        .map(|d| d.options.iter().any(|o| o.kind == 6))
                        .unwrap_or(false);
                    if let Err(e) = self.human_step(i) {
                        if self.control.as_ref().is_some_and(|c| c.stopping()) {
                            self.advancing = false;
                            return;
                        }
                        eprintln!("ALERT web: auto-pass step failed: {e}");
                        self.finish(app, winchance_on, frames);
                        return;
                    }
                    self.stats.note_chain_window(had, false);
                    let n = self.env.history_log().len();
                    self.auto_passed.insert(n);
                    continue;
                }
            }
            if self.need_human() {
                self.advancing = false;
                self.release_advancing();
                if winchance_on {
                    self.compute_winchance(app);
                }
                if self.sig() != self.last_sig {
                    self.send_state(app, false, false, winchance_on, frames);
                }
                return;
            }
            if moved > 0 {
                self.send_state(app, false, false, winchance_on, frames);
            }
            let ticker = match self.live_tx.clone() {
                Some(sink) => {
                    for f in frames.drain(..) {
                        sink.send(Outgoing::Frame(f));
                    }
                    Some(Ticker::start(self.state_message(app, false, true, winchance_on), sink))
                }
                None => None,
            };
            let stepped = self.step_bot(app);
            drop(ticker);
            if let Err(e) = stepped {
                if self.control.as_ref().is_some_and(|c| c.stopping()) {
                    self.advancing = false;
                    return;
                }
                eprintln!("ALERT web: bot step failed: {e}");
                self.finish(app, winchance_on, frames);
                return;
            }
            if self.puzzle_failed() {
                self.fail_puzzle(app, winchance_on, frames);
                return;
            }
            self.send_state(app, false, false, winchance_on, frames);
        }
    }

    /// A puzzle fails the moment the turn passes without a win (Python `_puzzle_failed`).
    fn puzzle_failed(&self) -> bool {
        let Some(p) = self.puzzle.as_ref() else { return false };
        if self.finished || self.env.done() {
            return false;
        }
        let Some(start) = p.get("start_turn").and_then(Value::as_i64) else { return false };
        let Ok(st) = self.env.duel_state() else { return false };
        st.turn as i64 != start
    }

    /// Stop a failed puzzle: write the record with `puzzle_result` and a failed puzzle state.
    fn fail_puzzle(&mut self, app: &App, winchance_on: bool, frames: &mut Vec<Value>) {
        if self.finished {
            return;
        }
        self.advancing = false;
        self.release_advancing();
        let turn = self.env.duel_state().map(|s| s.turn).unwrap_or(0);
        // Python writes the record from a local DuelResult and leaves `session.result` None, so the failed
        // puzzle's state frame carries `result: null`.
        self.result = Some(DuelResult {
            winner: None,
            turns: turn,
            win_reason: 0,
            surrendered: false,
            disconnected: false,
        });
        self.puzzle_result = Some("failed".into());
        if let Some(p) = self.puzzle.as_mut() {
            p["status"] = json!("failed");
        }
        self.write_record(app);
        self.result = None;
        self.finished = true;
        self.end_reason = "puzzle failed".into();
        self.log_sent = self.env.history_log().len();
        self.send_state(app, true, false, winchance_on, frames);
    }

    /// The duel session was stopped from the socket (new game / disconnect / surrender). The worker's own cancel
    /// flag also turns true when that worker is closed, so ask the session, never this game's (possibly retired)
    /// worker.
    fn session_stopping(&self) -> bool {
        self.control.as_ref().and_then(|c| c.parent.as_ref()).is_some_and(|s| s.stopping())
    }

    /// The human may act now: give back the socket's `advancing` counts before the frame that says so goes out.
    fn release_advancing(&self) {
        if let Some(session) = self.control.as_ref().and_then(|c| c.parent.as_ref()) {
            session.settle_advancing();
        }
    }

    fn finish(&mut self, app: &App, winchance_on: bool, frames: &mut Vec<Value>) {
        self.release_advancing();
        if self.finished {
            self.send_state(app, true, false, winchance_on, frames);
            return;
        }
        let st = self.env.duel_state().ok();
        self.result = Some(DuelResult {
            winner: st.as_ref().and_then(|s| (0..=1).contains(&s.winner).then_some(s.winner)),
            turns: st.as_ref().map(|s| s.turn).unwrap_or(0),
            win_reason: st.as_ref().map(|s| s.win_reason).unwrap_or(0),
            surrendered: false,
            disconnected: false,
        });
        if let (Some(p), Some(res)) = (self.puzzle.as_ref(), self.result) {
            let start = p.get("start_turn").and_then(Value::as_i64).unwrap_or(0);
            let solved = res.winner == Some(self.human_seat) && res.turns as i64 == start;
            let status = if solved { "solved" } else { "failed" };
            self.puzzle_result = Some(status.into());
            if let Some(p) = self.puzzle.as_mut() {
                p["status"] = json!(status);
            }
        }
        self.write_record(app);
        self.finished = true;
        self.end_reason.clear();
        self.log_sent = self.env.history_log().len();
        self.send_state(app, true, false, winchance_on, frames);
    }

    fn surrender(&mut self, app: &App, winchance_on: bool, frames: &mut Vec<Value>) {
        let st = self.env.duel_state().ok();
        self.result = Some(DuelResult {
            winner: Some(self.bot_seat),
            turns: st.as_ref().map(|s| s.turn).unwrap_or(0),
            win_reason: st.as_ref().map(|s| s.win_reason).unwrap_or(0),
            surrendered: true,
            disconnected: false,
        });
        self.write_record(app);
        self.finished = true;
        self.end_reason = "surrender".to_string();
        self.send_state(app, true, false, winchance_on, frames);
    }

    fn disconnect_record(&mut self, app: &App) {
        if self.finished || self.result.is_some() {
            return;
        }
        let st = self.env.duel_state().ok();
        self.result = Some(DuelResult {
            winner: Some(self.bot_seat),
            turns: st.as_ref().map(|s| s.turn).unwrap_or(0),
            win_reason: st.as_ref().map(|s| s.win_reason).unwrap_or(0),
            surrendered: false,
            disconnected: true,
        });
        self.write_record(app);
        self.finished = true;
    }

    fn write_record(&mut self, app: &App) {
        let Some(res) = self.result else { return };
        let st = self.env.duel_state().ok();
        let first_player = st.map(|s| s.first_player).filter(|p| *p >= 0);
        let mut seats = [Value::Null, Value::Null];
        seats[self.human_seat as usize] =
            json!({"type": "human", "player": app.records.install_id.clone()});
        let mut bot_seat = json!({
            "type": "bot",
            "bot_mode": self.bot_mode,
            "model": app.model_sha(),
        });
        if let Some(p) = self.product.as_ref() {
            bot_seat["backend"] = json!("rust");
            bot_seat["game_seed"] = json!(p.game_seed);
            bot_seat["ws_seed"] = json!(p.ws_seed);
            bot_seat["replay_threads"] = json!(p.replay_threads);
            if app.onnx_pool.intra_threads() > 1 {
                // Absent at 1 so records stay identical to the Python server's.
                bot_seat["ort_threads"] = json!(app.onnx_pool.intra_threads());
            }
            bot_seat["policy_sha256"] = json!(p.policy_sha256);
            bot_seat["value_sha256"] = json!(p.value_sha256);
        }
        seats[self.bot_seat as usize] = bot_seat;
        let rg = RecordGame {
            seed: self.seed,
            first: first_player,
            main0: self.main0.clone(),
            extra0: self.extra0.clone(),
            main1: self.main1.clone(),
            extra1: self.extra1.clone(),
            side0: self.side0.clone(),
            side1: self.side1.clone(),
            seats,
            mode: self.mode.clone(),
            responses: self.responses.clone(),
            winner: res.winner,
            win_reason: res.win_reason,
            turns: res.turns,
            surrendered: res.surrendered,
            disconnected: res.disconnected,
            first_player,
            deck_names: Some(self.deck_names.clone()),
            hints_used: Some(self.hints_used),
            undos: (self.undos > 0).then_some(self.undos),
            what_if: self.what_if.clone(),
            puzzle: self.puzzle.as_ref().map(|p| json!({"id": p.get("id").cloned().unwrap_or(Value::Null)})),
            puzzle_result: self.puzzle_result.clone(),
            practice: Some(self.practice),
        };
        match app.records.write(&rg) {
            Ok(path) => self.record_path = Some(path),
            Err(e) => eprintln!("ALERT web: could not write match record: {e}"),
        }
    }
}

/// What the client shows for the one model this server plays (Python web/server.py names its base model the same).
const BASE_MODEL_LABEL: &str = "Base model";

// ---------------------------------------------------------------- message handling

impl Inner {
    fn error(text: impl Into<String>) -> Vec<Value> {
        vec![json!({"t": "error", "text": text.into()})]
    }

    fn lobby(&self, app: &App) -> Value {
        json!({
            "t": "lobby",
            "decks": app.lobby_decks(self),
            // The Rust server serves one general model: it is the "base" kind to the client (Python get_models), so
            // its manifest label ("r1") is never shown as if it were a deck-tuned model.
            "models": app.models.iter()
                .map(|(id, label)| json!({"id": id, "label": BASE_MODEL_LABEL, "kind": "base", "short": label}))
                .collect::<Vec<_>>(),
            "model_default": app.model_id,
            // Card art on by default (user 2026-11-12: "the default on the app should be card art on"); the client
            // applies it only when the player never chose in Settings.
            "card_art": true,
            "packs": app.pack_status.lock().unwrap().clone(),
            "pack_update_url": app.pack_url,
        })
    }

    fn handle(&mut self, app: &Arc<App>, text: &str) -> Vec<Value> {
        let msg: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return Self::error("not JSON"),
        };
        let t = msg.get("t").and_then(Value::as_str);
        self.reap_product();
        // Card-art sync control works in the lobby and during a duel, before every game message.
        match t {
            Some("art_sync") => {
                let net = msg.get("network").and_then(Value::as_str).unwrap_or("unknown");
                app.art.set_network(Network::parse(net));
                app.device_models.cellular.store(net == "cellular", Ordering::SeqCst);
                return Vec::new();
            }
            Some("art_sync_pause") => {
                app.art.pause();
                return Vec::new();
            }
            Some("art_sync_resume") => {
                app.art.resume();
                return Vec::new();
            }
            Some("art_sync_allow_data") => {
                app.art.allow_data();
                return Vec::new();
            }
            Some("art_sync_delete") => {
                app.art.delete_all();
                return Vec::new();
            }
            _ => {}
        }
        if let Some(frames) = app.ask.handle(app, t, &msg) {
            return frames;
        }
        if let Some(frames) = self.handle_packs(app, t, &msg) {
            return frames;
        }
        if let Some(frames) = self.handle_describe(app, t, &msg) {
            return frames;
        }
        if let Some(frames) = Self::handle_device(app, t, &msg) {
            return frames;
        }
        if let Some(frames) = self.handle_cards_decks(app, t, &msg) {
            return frames;
        }
        if let Some(frames) = self.handle_room(app, t, &msg) {
            return frames;
        }
        if self.product.is_some() {
            match t {
                Some("surrender") => {
                    if self.product.as_ref().is_some_and(|p| p.control.game_over.load(Ordering::SeqCst)) {
                        return Self::error("no active game (send new_game first)");
                    }
                    // Cuts through a running search; the thread records the surrender and stays for the review.
                    if let Some(p) = self.product.as_ref() {
                        p.control.surrender.store(true, Ordering::SeqCst);
                    }
                    return self.send_product(ProductCmd::Surrender);
                }
                Some("choose") => {
                    let index = match msg.get("index") {
                        Some(Value::Number(n)) => n.as_i64(),
                        Some(Value::String(s)) => s.parse::<i64>().ok(),
                        _ => None,
                    };
                    let Some(index) = index else {
                        return Self::error("index must be an option number");
                    };
                    let prompt = msg.get("prompt").and_then(Value::as_i64).unwrap_or(-1);
                    if let Some(p) = self.product.as_ref() {
                        p.control.advancing.fetch_add(1, Ordering::SeqCst);
                    }
                    return self.send_product(ProductCmd::Choose { prompt, index });
                }
                Some("resync") => return self.send_product(ProductCmd::Resync),
                // While the bot plays, answer at once like the Python server (its `advancing` checks come first).
                Some("hint") if self.product_advancing() => {
                    return Self::error("no prompt is waiting for you right now");
                }
                Some("undo") if self.product_advancing() => return Self::error("Wait for the bot to finish"),
                Some("cancel") if self.product_advancing() => return Self::error("Wait for the bot to finish"),
                Some("cancel") => {
                    if let Some(p) = self.product.as_ref() {
                        p.control.advancing.fetch_add(1, Ordering::SeqCst);
                    }
                    return self.send_product(ProductCmd::Cancel);
                }
                Some("hint") => return self.send_product(ProductCmd::Hint),
                Some("undo") => {
                    if let Some(p) = self.product.as_ref() {
                        p.control.advancing.fetch_add(1, Ordering::SeqCst);
                    }
                    return self.send_product(ProductCmd::Undo);
                }
                Some("settings") => {
                    let has_chain = msg.get("chain").is_some();
                    let has_wc = msg.get("winchance").is_some();
                    if !has_chain && !has_wc {
                        return Self::error("settings.chain must be auto|all");
                    }
                    if has_chain {
                        match msg.get("chain").and_then(Value::as_str) {
                            Some(v @ ("auto" | "all")) => self.chain_mode = v.to_string(),
                            _ => return Self::error("settings.chain must be auto|all"),
                        }
                    }
                    if has_wc {
                        match msg.get("winchance") {
                            Some(Value::Bool(b)) => self.winchance_on = *b,
                            _ => return Self::error("settings.winchance must be true|false"),
                        }
                    }
                    return self.send_product(ProductCmd::Settings {
                        chain: self.chain_mode.clone(),
                        winchance: self.winchance_on,
                    });
                }
                Some("new_game") => self.stop_product(false),
                Some("puzzles") => return self.on_puzzles(app),
                Some("primer") => return self.on_primer(app, &msg),
                Some("study") => return self.on_study(app, &msg),
                Some("puzzle_solution") => return self.on_puzzle_solution(app, &msg),
                Some("puzzle") => return self.on_puzzle(app, &msg),
                Some("review") => return self.send_product(ProductCmd::Review),
                Some("review_frame") => return self.send_product(ProductCmd::ReviewFrame(msg.clone())),
                Some("whatif") => {
                    if let Some(p) = self.product.as_ref() {
                        p.control.advancing.fetch_add(1, Ordering::SeqCst);
                    }
                    return self.send_product(ProductCmd::WhatIf(msg.clone()));
                }
                _ => {
                    return Self::error(format!("unknown message {}", py_repr(t)));
                }
            }
        }
        match t {
            Some("surrender") => self.on_surrender(app),
            Some("settings") => self.on_settings(app, &msg),
            Some("new_game") => self.on_new_game(app, &msg),
            Some("choose") => self.on_choose(app, &msg),
            Some("resync") => {
                if self.game.is_none() {
                    vec![self.lobby(app)]
                } else {
                    let mut frames = Vec::new();
                    if let Some(g) = self.game.as_mut() {
                        g.send_state(app, true, false, self.winchance_on, &mut frames);
                    }
                    frames
                }
            }
            Some("puzzles") => self.on_puzzles(app),
            Some("puzzle") => self.on_puzzle(app, &msg),
            Some("puzzle_solution") => self.on_puzzle_solution(app, &msg),
            Some("primer") => self.on_primer(app, &msg),
            Some("study") => self.on_study(app, &msg),
            Some("hint") => self.on_hint_local(),
            Some("undo") => self.on_undo_local(),
            Some("cancel") => Self::error("Cancel needs the full bot"),
            Some(t @ ("review" | "review_frame" | "whatif")) => self.on_review_local(t, &msg),
            _ => Self::error(format!("unknown message {}", py_repr(t))),
        }
    }

    /// A5 card-pack commands. They work in the lobby and during a duel; installing a pack takes effect on the
    /// next server start (the status says `restart_required`).
    fn handle_packs(&mut self, app: &Arc<App>, t: Option<&str>, _msg: &Value) -> Option<Vec<Value>> {
        let frame = |status: &PackStatus| {
            json!({"t": "packs", "packs": status, "update_url": app.pack_url})
        };
        match t? {
            "packs" => Some(vec![frame(&app.pack_status.lock().unwrap().clone())]),
            "packs_update" => {
                let Some(url) = app.pack_url.clone() else {
                    return Some(Self::error("no card-pack update source is configured"));
                };
                if app.pack_update_busy.swap(true, Ordering::SeqCst) {
                    return Some(Self::error("a card-pack update is already running"));
                }
                let app2 = app.clone();
                let tx = self.conn_tx.clone();
                let spawned = std::thread::Builder::new().name("pack-update".into()).spawn(move || {
                    let result = download_pack(&url).and_then(|p| {
                        let r = app2.packs.install_zip(&p, None);
                        let _ = std::fs::remove_file(&p);
                        r
                    });
                    app2.pack_update_busy.store(false, Ordering::SeqCst);
                    let frame = match result {
                        Ok(r) => {
                            *app2.pack_status.lock().unwrap() = r.status.clone();
                            json!({"t": "packs", "packs": r.status, "update_url": app2.pack_url, "installed": true})
                        }
                        Err(e) => json!({"t": "error", "text": format!("card pack update failed: {e}")}),
                    };
                    if let Some(tx) = tx {
                        let _ = tx.send(Outgoing::Frame(frame));
                    }
                });
                if spawned.is_err() {
                    app.pack_update_busy.store(false, Ordering::SeqCst);
                    return Some(Self::error("could not start the card-pack update"));
                }
                Some(vec![json!({
                    "t": "packs", "packs": app.pack_status.lock().unwrap().clone(),
                    "update_url": app.pack_url, "busy": true
                })])
            }
            "packs_rollback" => match app.packs.rollback() {
                Ok(r) => {
                    *app.pack_status.lock().unwrap() = r.status.clone();
                    Some(vec![json!({
                        "t": "packs", "packs": r.status, "update_url": app.pack_url, "rolled_back": true
                    })])
                }
                Err(e) => Some(Self::error(format!("card pack rollback failed: {e}"))),
            },
            _ => None,
        }
    }

    // ---------------------------------------------------------------- describe-a-card model

    /// The optional semantic model: status, download/resume, cancel and remove. Every reply echoes `req`.
    fn handle_describe(&mut self, app: &Arc<App>, t: Option<&str>, msg: &Value) -> Option<Vec<Value>> {
        let req = msg.get("req").cloned();
        let state = &app.describe.state;
        match t? {
            "describe_status" => Some(vec![with_req(state.status(), &req)]),
            "describe_download" => {
                let sink: Arc<dyn Fn(Value) + Send + Sync> = match self.conn_tx.clone() {
                    Some(tx) => Arc::new(move |v: Value| {
                        let _ = tx.send(Outgoing::Frame(v));
                    }),
                    None => Arc::new(|_v: Value| {}),
                };
                match state.start(sink) {
                    Ok(()) => {
                        let st = state.status();
                        let frame = with_req(st, &req);
                        // `start` returns before the worker's first status; tell the client it is running.
                        Some(vec![frame])
                    }
                    Err(e) => Some(vec![error_with_req(&req, e)]),
                }
            }
            "describe_cancel" => {
                state.cancel();
                Some(vec![with_req(state.status(), &req)])
            }
            "describe_remove" => {
                state.remove();
                app.describe.drop_encoder();
                Some(vec![with_req(state.status(), &req)])
            }
            _ => None,
        }
    }

    // ---------------------------------------------------------------- on-device models

    /// The "On this device" tab: the phone check and each model's download (start/resume, pause, remove, Wi-Fi only).
    /// Progress frames go to whichever connection is open (`ask.emit`).
    fn handle_device(app: &Arc<App>, t: Option<&str>, msg: &Value) -> Option<Vec<Value>> {
        let req = msg.get("req").cloned();
        let models = &app.device_models;
        let id = msg.get("model").and_then(Value::as_str).unwrap_or("");
        let r = match t? {
            "device_status" => Ok(()),
            "device_download" => {
                let a = Arc::clone(app);
                models.start(id, Arc::new(move |v: Value| a.ask.emit(v)))
            }
            "device_pause" => {
                models.pause();
                Ok(())
            }
            "device_remove" => {
                // The engine may hold this model's file open on the GPU.
                app.ondevice.unload();
                models.remove(id)
            }
            "device_wifi" => {
                models.wifi_only.store(msg.get("on").and_then(Value::as_bool).unwrap_or(true), Ordering::SeqCst);
                Ok(())
            }
            _ => return None,
        };
        Some(vec![match r {
            Ok(()) => with_req(models.status(), &req),
            Err(e) => error_with_req(&req, e),
        }])
    }

    // ---------------------------------------------------------------- cards and saved decks

    /// Card search/lookup and user-deck messages. Lobby-level, also allowed during a duel; every reply echoes `req`.
    fn handle_cards_decks(&mut self, app: &Arc<App>, t: Option<&str>, msg: &Value) -> Option<Vec<Value>> {
        let req = msg.get("req").cloned();
        if t == Some("cards_search") && msg.get("mode").and_then(Value::as_str) == Some("describe") {
            return Some(self.on_cards_search_describe(app, msg, &req));
        }
        let frame = match t? {
            "cards_search" => self.on_cards_search(app, msg, &req),
            "card_get" => Self::on_card_get(app, msg, &req),
            "deck_list" => Self::on_deck_list(app, &req),
            "deck_get" => Self::on_deck_get(app, msg, &req),
            "deck_save" => Self::on_deck_save(app, msg, &req),
            "deck_delete" => Self::on_deck_delete(app, msg, &req),
            "deck_import" => Self::on_deck_import(app, msg, &req),
            "deck_export" => Self::on_deck_export(app, msg, &req),
            _ => return None,
        };
        let mut frames = vec![frame];
        // A saved/imported/deleted deck changes the lobby pickers: push the full list to this connection.
        let done = frames[0].get("t").and_then(Value::as_str);
        if matches!(done, Some("deck_saved" | "deck_deleted")) {
            frames.push(self.lobby_decks_frame(app));
        }
        Some(frames)
    }

    fn lobby_decks_frame(&self, app: &App) -> Value {
        json!({"t": "lobby_decks", "decks": app.lobby_decks(self)})
    }

    fn on_cards_search(&self, app: &App, msg: &Value, req: &Option<Value>) -> Value {
        match msg.get("mode").and_then(Value::as_str) {
            Some("describe") => return error_with_req(req, "describe mode not installed"),
            Some("names_text") | None => {}
            Some(other) => return error_with_req(req, format!("unknown search mode {other:?}")),
        }
        let q = msg.get("q").and_then(Value::as_str).unwrap_or("");
        let filters = msg.get("filters").cloned().unwrap_or(Value::Null);
        let offset = msg.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit = msg.get("limit").and_then(Value::as_u64).unwrap_or(40).min(200) as usize;
        match app.search_index().search(q, &filters, offset, limit) {
            Ok(result) => cards_frame(q, &result, req),
            Err(e) => error_with_req(req, e),
        }
    }

    /// Semantic search runs on its own thread: encoding one query takes ~0.2-0.4 s and must never block the
    /// duel/message loop. The reply (with `req`) arrives through the connection channel when ranking is done.
    fn on_cards_search_describe(&self, app: &Arc<App>, msg: &Value, req: &Option<Value>) -> Vec<Value> {
        if !app.describe.state.installed() {
            return vec![error_with_req(req, "describe mode not installed")];
        }
        let q = msg.get("q").and_then(Value::as_str).unwrap_or("").to_string();
        let filters = msg.get("filters").cloned().unwrap_or(Value::Null);
        let offset = msg.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit = msg.get("limit").and_then(Value::as_u64).unwrap_or(40).min(200) as usize;
        let Some(tx) = self.conn_tx.clone() else {
            return Self::error("no client connection");
        };
        let app2 = app.clone();
        let req2 = req.clone();
        let spawned = std::thread::Builder::new().name("cards-describe".into()).spawn(move || {
            let frame = match app2.describe.embeddings() {
                Err(e) => error_with_req(&req2, format!("card embeddings could not be loaded: {e}")),
                Ok(emb) => {
                    let index = app2.search_index();
                    let result = index.search_describe(&q, &filters, offset, limit, &emb, |clause| {
                        app2.describe.encode(&format!("{}{clause}", crate::describe_enc::INSTRUCT))
                    });
                    match result {
                        Ok(r) => cards_frame(&q, &r, &req2),
                        Err(e) => error_with_req(&req2, e),
                    }
                }
            };
            let _ = tx.send(Outgoing::Frame(frame));
        });
        match spawned {
            Ok(_) => Vec::new(),
            Err(e) => Self::error(format!("could not start the semantic search: {e}")),
        }
    }

    fn on_card_get(app: &App, msg: &Value, req: &Option<Value>) -> Value {
        let Some(code) = msg.get("code").and_then(Value::as_u64).filter(|n| *n > 0 && *n <= u32::MAX as u64)
        else {
            return error_with_req(req, "card_get.code must be a positive card code");
        };
        match app.search_index().card_json(code as u32) {
            Some(mut card) => {
                card["t"] = json!("card");
                with_req(card, req)
            }
            None => error_with_req(req, format!("unknown card {code}")),
        }
    }

    fn on_deck_list(app: &App, req: &Option<Value>) -> Value {
        let format = DeckFormat::default();
        let names = app.ctx.texts.names();
        let decks: Vec<Value> = app
            .user_decks
            .list()
            .into_iter()
            .map(|stored| {
                let legal = crate::userdecks::legality(&stored.deck, &app.cards, names, &format);
                json!({
                    "id": stored.id,
                    "name": stored.meta.name,
                    "updated": stored.meta.updated,
                    "cover": crate::userdecks::UserDecks::cover_of(&stored),
                    "counts": {
                        "main": stored.deck.main.len(),
                        "extra": stored.deck.extra.len(),
                        "side": stored.deck.side.len(),
                    },
                    "legal": legal_text(&legal),
                })
            })
            .collect();
        with_req(json!({"t": "user_decks", "decks": decks}), req)
    }

    fn deck_id(msg: &Value) -> Result<String, Value> {
        match msg.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            Some(id) => Ok(id.to_string()),
            None => Err(json!({"error": "id must be a deck id"})),
        }
    }

    fn on_deck_get(app: &App, msg: &Value, req: &Option<Value>) -> Value {
        let id = match Self::deck_id(msg) {
            Ok(id) => id,
            Err(e) => return error_with_req(req, e["error"].as_str().unwrap_or("bad id")),
        };
        let stored = match app.user_decks.get(&id) {
            Ok(d) => d,
            Err(e) => return error_with_req(req, e),
        };
        let format = DeckFormat::default();
        let legal = crate::userdecks::legality(&stored.deck, &app.cards, app.ctx.texts.names(), &format);
        let unknown = unknown_codes(&stored.deck, &app.cards);
        with_req(
            json!({
                "t": "user_deck",
                "deck": {
                    "id": stored.id,
                    "name": stored.meta.name,
                    "main": stored.deck.main,
                    "extra": stored.deck.extra,
                    "side": stored.deck.side,
                    "updated": stored.meta.updated,
                    "cover": crate::userdecks::UserDecks::cover_of(&stored),
                },
                "legal": legal_full(&legal),
                "unknown": unknown,
            }),
            req,
        )
    }

    fn on_deck_save(app: &App, msg: &Value, req: &Option<Value>) -> Value {
        let deck = match deck_from_value(msg) {
            Ok(d) => d,
            Err(e) => return error_with_req(req, e),
        };
        if deck.total() > 200 {
            return error_with_req(req, "a deck cannot hold more than 200 cards");
        }
        let id = match msg.get("id") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if !s.is_empty() => Some(s.as_str()),
            _ => return error_with_req(req, "deck_save.id must be a deck id"),
        };
        let name = msg.get("name").and_then(Value::as_str).unwrap_or("");
        let cover = msg.get("cover").and_then(Value::as_u64).map(|n| n as u32);
        let stored = match app.user_decks.save(id, name, &deck, cover) {
            Ok(d) => d,
            Err(e) => return error_with_req(req, e),
        };
        let legal = crate::userdecks::legality(&stored.deck, &app.cards, app.ctx.texts.names(), &DeckFormat::default());
        with_req(
            json!({"t": "deck_saved", "id": stored.id, "updated": stored.meta.updated, "legal": legal_text(&legal)}),
            req,
        )
    }

    fn on_deck_delete(app: &App, msg: &Value, req: &Option<Value>) -> Value {
        let id = match Self::deck_id(msg) {
            Ok(id) => id,
            Err(e) => return error_with_req(req, e["error"].as_str().unwrap_or("bad id")),
        };
        match app.user_decks.delete(&id) {
            Ok(()) => with_req(json!({"t": "deck_deleted", "id": id}), req),
            Err(e) => error_with_req(req, e),
        }
    }

    fn on_deck_import(app: &App, msg: &Value, req: &Option<Value>) -> Value {
        let Some(ydk) = msg.get("ydk").and_then(Value::as_str) else {
            return error_with_req(req, "deck_import.ydk must be the .ydk text");
        };
        let parsed = crate::userdecks::parse_ydk(ydk);
        if parsed.deck.total() > 200 {
            return error_with_req(req, "a deck cannot hold more than 200 cards");
        }
        let name = msg
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string)
            .or_else(|| parsed.name.clone())
            .unwrap_or_else(|| "Imported deck".to_string());
        let stored = match app.user_decks.save(None, &name, &parsed.deck, None) {
            Ok(d) => d,
            Err(e) => return error_with_req(req, e),
        };
        let legal = crate::userdecks::legality(&stored.deck, &app.cards, app.ctx.texts.names(), &DeckFormat::default());
        let unknown = unknown_codes(&parsed.deck, &app.cards);
        with_req(
            json!({
                "t": "deck_saved",
                "id": stored.id,
                "updated": stored.meta.updated,
                "legal": legal_text(&legal),
                "unknown": unknown,
                "unknown_lines": parsed.unknown_lines,
            }),
            req,
        )
    }

    fn on_deck_export(app: &App, msg: &Value, req: &Option<Value>) -> Value {
        let id = match Self::deck_id(msg) {
            Ok(id) => id,
            Err(e) => return error_with_req(req, e["error"].as_str().unwrap_or("bad id")),
        };
        let stored = match app.user_decks.get(&id) {
            Ok(d) => d,
            Err(e) => return error_with_req(req, e),
        };
        with_req(
            json!({
                "t": "deck_ydk",
                "id": stored.id,
                "name": stored.meta.name,
                "ydk": to_ydk(&stored.deck),
                "ydke": to_ydke(&stored.deck),
            }),
            req,
        )
    }

    /// Room messages (A2b-3c), and the bot-duel messages a room takes over while it is open. `None` = not a room
    /// matter.
    fn handle_room(&mut self, app: &Arc<App>, t: Option<&str>, msg: &Value) -> Option<Vec<Value>> {
        let in_room = self.room.is_some();
        let frames = match t? {
            "room_host" => self.on_room_host(app, msg),
            "room_join" => self.on_room_join(app, msg),
            "room_scan" => self.on_room_scan(),
            "room_leave" => {
                self.close_room();
                vec![json!({"t": "room", "state": "closed"}), self.lobby(app)]
            }
            "room_back" => self.on_room_back(app),
            "room_ready" => match msg.get("ready") {
                Some(Value::Bool(b)) => self.send_room(RoomCmd::Ready(*b)),
                _ => Self::error("room_ready.ready must be true|false"),
            },
            "room_start" => self.send_room(RoomCmd::Start),
            "room_rps" => match msg.get("hand").and_then(Value::as_u64) {
                Some(h @ 1..=3) => self.send_room(RoomCmd::Rps(h as u8)),
                _ => Self::error("room_rps.hand must be 1 (scissors), 2 (rock) or 3 (paper)"),
            },
            "room_first" => match msg.get("first") {
                Some(Value::Bool(b)) => self.send_room(RoomCmd::First(*b)),
                _ => Self::error("room_first.first must be true|false"),
            },
            "rematch" => match msg.get("yes") {
                Some(Value::Bool(b)) => self.send_room(RoomCmd::Rematch(*b)),
                _ => Self::error("rematch.yes must be true|false"),
            },
            "choose" if in_room => {
                let index = match msg.get("index") {
                    Some(Value::Number(n)) => n.as_i64(),
                    Some(Value::String(s)) => s.parse::<i64>().ok(),
                    _ => None,
                };
                let Some(index) = index else {
                    return Some(Self::error("index must be an option number"));
                };
                let prompt = msg.get("prompt").and_then(Value::as_i64).unwrap_or(-1);
                self.send_room(RoomCmd::Choose { prompt, index })
            }
            "surrender" if in_room => self.send_room(RoomCmd::Surrender),
            "resync" if in_room => self.send_room(RoomCmd::Resync),
            "new_game" if in_room => Self::error("leave the room first"),
            "settings" if in_room => Vec::new(),
            "hint" | "undo" | "cancel" | "whatif" | "review" | "review_frame" | "study" if in_room => {
                Self::error("Off in rooms")
            }
            _ => return None,
        };
        Some(frames)
    }

    fn product_advancing(&self) -> bool {
        self.product.as_ref().is_some_and(|p| p.control.advancing.load(Ordering::SeqCst) > 0)
    }

    // ---------------------------------------------------------------- puzzles, primer, study

    fn on_puzzles(&self, app: &App) -> Vec<Value> {
        vec![json!({"t": "puzzles", "items": load_puzzles(app).iter().map(puzzle_item).collect::<Vec<_>>()})]
    }

    fn on_puzzle(&mut self, app: &Arc<App>, msg: &Value) -> Vec<Value> {
        let pid = msg.get("id").cloned().unwrap_or(Value::Null);
        let want = py_str_value(&pid);
        let Some(found) = load_puzzles(app).into_iter().find(|p| py_str_value(p.get("id").unwrap_or(&Value::Null)) == want)
        else {
            return Self::error(format!("unknown puzzle {}", py_value_repr(&pid)));
        };
        let deck_names: Vec<String> = found
            .get("decks")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|d| d.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_default();
        if deck_names.len() < 2 {
            return Self::error("puzzle decks unavailable: 'decks'");
        }
        let mut decks = Vec::new();
        for name in &deck_names {
            match load_puzzle_deck(&app.root, name) {
                Ok(d) => decks.push(d),
                Err(e) => return Self::error(format!("puzzle decks unavailable: {e}")),
            }
        }
        let Some(seat) = int_val(found.get("seat"), None) else {
            return Self::error(format!("bad puzzle {}: 'seat'", py_value_repr(&pid)));
        };
        let Some(start_turn) = int_val(found.get("turn"), None) else {
            return Self::error(format!("bad puzzle {}: 'turn'", py_value_repr(&pid)));
        };
        let prefix: Vec<i32> = found
            .get("prefix")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| int_val(Some(v), Some(0)).unwrap_or(0) as i32).collect())
            .unwrap_or_default();
        let seed = int_val(found.get("seed"), Some(0)).unwrap_or(0) as u64;
        let game_seed = int_val(found.get("game_seed"), None)
            .map(|v| v as u64)
            .unwrap_or(seed);
        let cfg = puzzle_cfg(&decks, seed);
        // Host-space replay of the stored prefix (Python's first PyEnv check).
        {
            let mut env = match Env::new(app.engine, &cfg, false) {
                Ok(e) => e,
                Err(e) => return Self::error(format!("could not start the puzzle duel: {e}")),
            };
            let mut obs = Obs::zeroed();
            if let Err(e) = env.reset(&mut obs) {
                return Self::error(format!("could not start the puzzle duel: {e}"));
            }
            for &c in &prefix {
                if env.done() {
                    break;
                }
                if let Err(e) = env.step(c, &mut obs) {
                    return Self::error(format!("replay failed: {e}"));
                }
            }
            if env.done() {
                return Self::error("the puzzle position is already game over");
            }
        }
        let bot = if app.test_stream {
            Bot::Stream(SplitMix(app.fixed_seed.unwrap_or(0)))
        } else {
            if let Err(e) = app.ensure_raw() {
                return Self::error(format!("could not start the fast opponent: {e}"));
            }
            Bot::Raw
        };
        let sides = (decks[0].side.clone(), decks[1].side.clone());
        let human = seat as i32;
        let mut game = match Game::with_config(
            app,
            cfg,
            game_seed,
            human,
            sides,
            bot,
            "fast",
            true,
            None,
            (Vec::new(), Vec::new()),
        ) {
            Ok(g) => g,
            Err(e) => return Self::error(format!("could not start the puzzle duel: {e}")),
        };
        let replayed = (|| -> Result<(), String> {
            for &c in &prefix {
                if game.env.done() {
                    break;
                }
                game.step_index(c)?;
            }
            if game.env.done() {
                return Err("the replay position is already game over".into());
            }
            let turn = game.env.duel_state()?.turn;
            if turn != start_turn as i32 {
                return Err("puzzle prefix does not end on its turn".into());
            }
            Ok(())
        })();
        if let Err(e) = replayed {
            eprintln!("ALERT web: puzzle replay failed: {e}");
            return Self::error(format!("replay failed: {e}"));
        }
        game.practice = true;
        game.hints_used = 0;
        game.undos = 0;
        game.human_decisions.clear();
        game.auto_passed.clear();
        game.what_if = None;
        game.puzzle_result = None;
        game.puzzle = Some(json!({
            "id": py_str_value(found.get("id").unwrap_or(&Value::Null)),
            "start_turn": start_turn,
            "goal": found.get("goal").and_then(Value::as_str).unwrap_or(""),
            "n_plays": int_val(found.get("n_plays").or_else(|| found.get("n_moves")), Some(0)).unwrap_or(0),
            "status": "playing",
        }));
        self.stop_product(false);
        self.game = None;
        let pretty = |s: &str| pretty_deck_name(s);
        let (hname, bname) = (pretty(&deck_names[0]), pretty(&deck_names[1]));
        if human == 0 {
            self.live_decks = Some((decks[0].clone(), decks[1].clone()));
            self.live_deck_names = (hname, bname);
        } else {
            self.live_decks = Some((decks[1].clone(), decks[0].clone()));
            self.live_deck_names = (bname, hname);
        }
        // `decks`/`deck_names` are in seat order (puzzle_cfg builds seat 0 from decks[0]).
        game.deck_names = [pretty(&deck_names[0]), pretty(&deck_names[1])];
        let mut frames = Vec::new();
        game.advance(app, &self.chain_mode, self.winchance_on, &mut frames);
        self.game = Some(game);
        frames
    }

    fn on_puzzle_solution(&self, app: &App, msg: &Value) -> Vec<Value> {
        let pid = msg.get("id").cloned().unwrap_or(Value::Null);
        let want = py_str_value(&pid);
        let Some(found) = load_puzzles(app).into_iter().find(|p| py_str_value(p.get("id").unwrap_or(&Value::Null)) == want)
        else {
            return Self::error(format!("unknown puzzle {}", py_value_repr(&pid)));
        };
        let deck_names: Vec<String> = found
            .get("decks")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|d| d.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_default();
        let mut decks = Vec::new();
        for name in &deck_names {
            match load_puzzle_deck(&app.root, name) {
                Ok(d) => decks.push(d),
                Err(e) => return Self::error(format!("puzzle solution failed: {e}")),
            }
        }
        if decks.len() < 2 {
            return Self::error("puzzle solution failed: 'decks'");
        }
        let seed = int_val(found.get("seed"), Some(0)).unwrap_or(0) as u64;
        let prefix: Vec<i32> = found
            .get("prefix")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| int_val(Some(v), Some(0)).unwrap_or(0) as i32).collect())
            .unwrap_or_default();
        let solution: Vec<i32> = found
            .get("solution")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| int_val(Some(v), Some(0)).unwrap_or(0) as i32).collect())
            .unwrap_or_default();
        let seat = int_val(found.get("seat"), None).unwrap_or(0) as i32;
        let cfg = puzzle_cfg(&decks, seed);
        let mut choices = prefix.clone();
        choices.extend(solution);
        let names = (
            deck_names.first().map(|s| pretty_deck_name(s)).unwrap_or_else(|| "?".into()),
            deck_names.get(1).map(|s| pretty_deck_name(s)).unwrap_or_else(|| "?".into()),
        );
        match replay_frames(app, &cfg, &choices, &names, Some(seat)) {
            Ok(frames) => {
                let frames = frames.into_iter().skip(prefix.len()).collect::<Vec<_>>();
                vec![json!({"t": "puzzle_solution", "id": py_str_value(found.get("id").unwrap_or(&Value::Null)), "frames": frames})]
            }
            Err(e) => {
                eprintln!("ALERT web: puzzle_solution crashed: {e}");
                Self::error(format!("puzzle solution failed: {e} (see server log)"))
            }
        }
    }

    fn on_primer(&mut self, app: &App, msg: &Value) -> Vec<Value> {
        let store = load_primers(app);
        let bot_name;
        let bot_deck;
        let my_deck;
        if let Some(spec) = msg.get("deck").filter(|v| !v.is_null()) {
            let (b, berr) = self.deck_catalog.resolve(spec);
            let Some(b) = b else { return Self::error(berr.unwrap_or_else(|| "unknown deck".into())) };
            bot_deck = b;
            if let Some(my_spec) = msg.get("my_deck").filter(|v| !v.is_null()) {
                let (m, merr) = self.deck_catalog.resolve(my_spec);
                let Some(m) = m else { return Self::error(merr.unwrap_or_else(|| "unknown deck".into())) };
                my_deck = m;
            } else if let Some((m, _)) = self.live_decks.as_ref() {
                my_deck = m.clone();
            } else {
                return Self::error("primer needs my_deck");
            }
            bot_name = if spec.is_string() { pretty_deck_name(spec.as_str().unwrap_or("")) } else { "Pasted deck".into() };
        } else {
            let Some((_, b)) = self.live_decks.as_ref() else {
                return Self::error("no active game (send new_game first)");
            };
            bot_deck = b.clone();
            my_deck = self.live_decks.as_ref().map(|(h, _)| h.clone()).unwrap_or_default();
            bot_name = self.live_deck_names.1.clone();
        }
        let primer = primer_for(app, &store, &bot_deck, &my_deck);
        let mut reply = json!({
            "t": "primer",
            "deck": bot_name,
            "hash": primer_hash(&bot_deck),
            "primer": primer.as_ref().map(trim_primer).unwrap_or(Value::Null),
        });
        if primer.is_none() {
            reply["can_study"] = json!(true);
        }
        vec![reply]
    }

    fn on_study(&mut self, app: &Arc<App>, msg: &Value) -> Vec<Value> {
        let (deck, deck_name, text) = if let Some(spec) = msg.get("deck").filter(|v| !v.is_null()) {
            let (d, err) = self.deck_catalog.resolve(spec);
            let Some(d) = d else { return Self::error(err.unwrap_or_else(|| "unknown deck".into())) };
            let deck_name = if spec.is_string() { pretty_deck_name(spec.as_str().unwrap_or("")) } else { "Pasted deck".into() };
            let text = if let Some(t) = spec.get("ydk").and_then(Value::as_str) {
                Some(t.to_string())
            } else if let Some(id) = spec.as_str() {
                self.deck_catalog
                    .dirs
                    .dirs
                    .iter()
                    .find_map(|(_, dir)| std::fs::read_to_string(dir.join(format!("{id}.ydk"))).ok())
                    .or_else(|| self.deck_catalog.user_decks.get(id).cloned())
            } else {
                None
            };
            let Some(text) = text else { return Self::error(format!("unknown deck {}", py_value_repr(spec))) };
            (d, deck_name, text)
        } else {
            let Some((_, b)) = self.live_decks.as_ref() else {
                return Self::error("no active game (send new_game first)");
            };
            let d = b.clone();
            let mut text = "#main\n".to_string();
            text.push_str(&d.main.iter().map(|c| format!("{c}\n")).collect::<String>());
            text.push_str("#extra\n");
            text.push_str(&d.extra.iter().map(|c| format!("{c}\n")).collect::<String>());
            text.push_str("!side\n");
            text.push_str(&d.side.iter().map(|c| format!("{c}\n")).collect::<String>());
            text.push('\n');
            (d, self.live_deck_names.1.clone(), text)
        };
        let h = primer_hash(&deck);
        if app.study_busy.load(Ordering::SeqCst) {
            return vec![json!({"t": "study_status", "busy": true, "deck": deck_name, "hash": h})];
        }
        let store = load_primers(app);
        if let Some(existing) = store.get("decks").and_then(|d| d.get(&h)) {
            let scored = existing.get("study").is_none()
                || existing.get("study").and_then(|s| s.get("phase")).and_then(Value::as_str) == Some("scored");
            if scored {
                let games = int_val(existing.get("games"), Some(0)).unwrap_or(0);
                if games >= 20 {
                    let which = if existing.get("study").is_none() { "shipped" } else { "scored" };
                    return Self::error(format!("already studied ({which} primer with {games} games)"));
                }
            }
        }
        let tmp = std::env::temp_dir().join(format!("decklight-study-{}.ydk", random_hex(8)));
        if let Err(e) = std::fs::write(&tmp, &text) {
            return Self::error(format!("could not start the study: {e}"));
        }
        let python = std::env::var("PYTHON").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| {
            let v = app.root.join(".venv/bin/python");
            if v.is_file() { v.to_string_lossy().into_owned() } else { "python3".to_string() }
        });
        let child = std::process::Command::new("taskset")
            .args(["-c", "0-3", "nice", "-n", "19"])
            .arg(&python)
            .arg("scripts/study_deck.py")
            .arg("--ydk")
            .arg(&tmp)
            .arg("--out-dir")
            .arg(app.root.join("data/primers/studied"))
            .args(["--procs", "3", "--games", "32"])
            .current_dir(&app.root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Self::error(format!("could not start the study: {e}"));
            }
        };
        app.study_busy.store(true, Ordering::SeqCst);
        let Some(tx) = self.conn_tx.clone() else {
            let _ = child.kill();
            let _ = std::fs::remove_file(&tmp);
            app.study_busy.store(false, Ordering::SeqCst);
            return Self::error("no client connection to run the study");
        };
        let app2 = app.clone();
        let h2 = h.clone();
        let name2 = deck_name.clone();
        std::thread::Builder::new().name("deck-study".into()).spawn(move || {
            use std::io::BufRead;
            const TOTAL: i64 = 32;
            let mut last = String::new();
            let mut total = TOTAL;
            let send = |frame: Value| {
                let _ = tx.send(Outgoing::Frame(frame));
            };
            if let Some(out) = child.stdout.take() {
                for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                    let line = line.trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    last = line.clone();
                    if let Some(rest) = line.strip_prefix("STUDY games ") {
                        if let Some((d, t)) = parse_ratio(rest) {
                            total = t;
                            send(json!({"t": "study_status", "deck": name2, "hash": h2, "phase": "games", "done": d, "total": t}));
                        }
                    } else if line == "STUDY phase games done" {
                        *app2.primers.lock().unwrap() = None;
                        send(json!({"t": "study_status", "deck": name2, "hash": h2, "phase": "games", "done": total, "total": total}));
                    } else if let Some(rest) = line.strip_prefix("STUDY scored ") {
                        if let Some((d, t)) = parse_ratio(rest) {
                            send(json!({"t": "study_status", "deck": name2, "hash": h2, "phase": "scoring", "done": d, "total": t}));
                        }
                    } else if line == "STUDY done" {
                        *app2.primers.lock().unwrap() = None;
                        send(json!({"t": "study_status", "deck": name2, "hash": h2, "phase": "done", "done": total, "total": total}));
                    }
                }
            }
            let rc = child.wait().map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
            let _ = std::fs::remove_file(&tmp);
            app2.study_busy.store(false, Ordering::SeqCst);
            if rc != 0 {
                eprintln!("ALERT web: study {name2} failed (exit {rc}): {last}");
                send(json!({"t": "study_status", "deck": name2, "hash": h2, "phase": "failed", "done": 0, "total": total,
                            "error": if last.is_empty() { format!("exit {rc}") } else { last }}));
            }
        }).ok();
        Vec::new()
    }

    /// `hint` outside a product duel: the Python server's checks, in its order (no full bot here).
    fn on_hint_local(&self) -> Vec<Value> {
        let Some(g) = self.game.as_ref() else {
            return Self::error("Hints are only in practice mode");
        };
        if !g.practice {
            return Self::error("Hints are only in practice mode");
        }
        if g.finished || g.env.done() || !g.need_human() || g.advancing {
            return Self::error("no prompt is waiting for you right now");
        }
        Self::error("Hints need the full bot")
    }

    /// `review` / `review_frame` / `whatif` outside a product duel: the Python server's checks, in its order (no
    /// full bot here, so never a review to show).
    fn on_review_local(&self, t: &str, msg: &Value) -> Vec<Value> {
        let finished = self.game.as_ref().is_some_and(|g| g.finished && g.result.is_some());
        if !finished {
            return Self::error("no finished game to review (finish a game first)");
        }
        match t {
            "review" => Self::error("Review needs the full bot"),
            "review_frame" if int_field(msg, "k").is_none() => {
                Self::error("review_frame.k must be a decision number")
            }
            _ => Self::error("no review yet (send review first)"),
        }
    }

    fn on_undo_local(&self) -> Vec<Value> {
        let Some(g) = self.game.as_ref() else {
            return Self::error("Undo is only in practice mode");
        };
        if !g.practice {
            return Self::error("Undo is only in practice mode");
        }
        if g.finished || g.env.done() {
            return Self::error("no active game (send new_game first)");
        }
        Self::error("Undo needs the full bot")
    }

    fn send_room(&self, cmd: RoomCmd) -> Vec<Value> {
        match self.room.as_ref() {
            Some(r) => match r.send(cmd) {
                Ok(()) => Vec::new(),
                Err(e) => Self::error(e),
            },
            None => Self::error("not in a room"),
        }
    }

    fn close_room(&mut self) {
        if let Some(mut r) = self.room.take() {
            r.leave();
        }
        if let Some(mut h) = self.hosted.take() {
            h.close();
        }
    }

    /// The deck and name a room message carries, checked like a bot duel's deck.
    fn room_deck(&mut self, app: &App, msg: &Value) -> Result<(DeckSpec, String), String> {
        let deck_spec = msg.get("deck").cloned().unwrap_or(Value::Null);
        let (deck, err) = self.deck_catalog.resolve(&deck_spec);
        let deck = deck.ok_or_else(|| err.unwrap_or_else(|| "unknown deck".into()))?;
        if let Some(e) = check_deck(&deck, &app.cards, &app.emb_codes, app.ctx.texts.names()) {
            return Err(e);
        }
        Ok((deck, crate::room::clean_name(msg.get("name").and_then(Value::as_str))))
    }

    fn start_room(&mut self, app: &Arc<App>, spec: RoomSpec) -> Vec<Value> {
        self.stop_product(false);
        self.game = None;
        match RoomSession::start(app.clone(), spec, self.room_out.clone()) {
            Ok(r) => {
                self.room = Some(r);
                Vec::new()
            }
            Err(e) => Self::error(e),
        }
    }

    fn on_room_host(&mut self, app: &Arc<App>, msg: &Value) -> Vec<Value> {
        let (deck, name) = match self.room_deck(app, msg) {
            Ok(v) => v,
            Err(e) => return Self::error(e),
        };
        let password: String = msg.get("password").and_then(Value::as_str).unwrap_or("").chars().take(20).collect();
        self.close_room();
        let room_name = format!("{name}'s room");
        let hosted = match HostedRoom::start(app, &room_name, &password) {
            Ok(h) => h,
            Err(e) => return Self::error(e),
        };
        let spec = RoomSpec {
            role: Role::Host,
            host: "127.0.0.1".into(),
            port: hosted.port,
            gameid: 0,
            password,
            name,
            deck,
            room_name,
            addresses: hosted.addresses.clone(),
        };
        self.hosted = Some(hosted);
        let frames = self.start_room(app, spec);
        if self.room.is_none() {
            // The session thread did not start: close our listener too (review a2b3c: it leaked until the next host).
            if let Some(mut h) = self.hosted.take() {
                h.close();
            }
        }
        frames
    }

    fn on_room_join(&mut self, app: &Arc<App>, msg: &Value) -> Vec<Value> {
        let (deck, name) = match self.room_deck(app, msg) {
            Ok(v) => v,
            Err(e) => return Self::error(e),
        };
        let address = msg.get("address").and_then(Value::as_str).unwrap_or("");
        let (host, port, gameid) = match crate::room::parse_address(address) {
            Ok(v) => v,
            Err(e) => return vec![json!({"t": "room", "state": "closed", "error": e, "field": "address"})],
        };
        let password: String = msg.get("password").and_then(Value::as_str).unwrap_or("").chars().take(20).collect();
        self.close_room();
        let room_name = format!("{host}:{port}");
        let spec = RoomSpec { role: Role::Guest, host, port, gameid, password, name, deck, room_name, addresses: Vec::new() };
        self.start_room(app, spec)
    }

    /// After a duel: the host's app rejoins its own (still open) room, a guest rejoins the same address.
    fn on_room_back(&mut self, app: &Arc<App>) -> Vec<Value> {
        let Some(mut old) = self.room.take() else { return Self::error("not in a room") };
        let mut spec = old.spec.clone();
        old.leave();
        if spec.role == Role::Host {
            match self.hosted.as_ref() {
                Some(h) => spec.port = h.port,
                None => return Self::error("the room has closed"),
            }
        }
        self.start_room(app, spec)
    }

    /// LAN discovery takes ~1.5 s: it runs on its own thread and answers through the socket channel.
    fn on_room_scan(&self) -> Vec<Value> {
        let Some(tx) = self.conn_tx.clone() else { return Self::error("no client connection") };
        let spawned = std::thread::Builder::new().name("room-scan".into()).spawn(move || {
            let frame = match crate::room::scan(Duration::from_millis(1500)) {
                Ok(rooms) => json!({"t": "rooms", "rooms": rooms}),
                // The reply port is fixed (7921): a second app on the same device scanning at once gets "in use".
                Err(e) if e.contains("in use") => {
                    json!({"t": "rooms", "rooms": [], "error": "Another app on this device is scanning. Try again in a moment."})
                }
                Err(e) => json!({"t": "rooms", "rooms": [], "error": format!("Could not scan this network ({e})")}),
            };
            let _ = tx.send(Outgoing::Frame(frame));
        });
        match spawned {
            Ok(_) => Vec::new(),
            Err(e) => Self::error(format!("could not scan: {e}")),
        }
    }

    /// Drop a product session whose thread has finished (so settings/resync/new_game do not talk to a dead thread).
    fn reap_product(&mut self) {
        let done = self.product.as_ref().is_some_and(|p| p.join.as_ref().is_some_and(|j| j.is_finished()));
        if done && let Some(mut s) = self.product.take() {
            if let Some(j) = s.join.take() {
                let _ = j.join();
            }
        }
    }

    fn send_product(&self, cmd: ProductCmd) -> Vec<Value> {
        match self.product.as_ref() {
            Some(p) => match p.cmd_tx.send(cmd) {
                Ok(()) => Vec::new(),
                Err(_) => {
                    // The thread is gone: nobody will give back the counts this command added, and they must not
                    // leave `hint`/`undo` refused if this session is somehow reused.
                    p.control.advancing.store(0, Ordering::SeqCst);
                    p.control.owed.store(0, Ordering::SeqCst);
                    Self::error("the product worker is gone")
                }
            },
            None => Self::error("no active game (send new_game first)"),
        }
    }

    /// Stop a running product duel: cancel the search, then join its thread (and the search worker inside it).
    fn stop_product(&mut self, disconnect: bool) {
        let Some(mut s) = self.product.take() else { return };
        if disconnect {
            s.control.disconnect.store(true, Ordering::SeqCst);
        }
        s.control.cancel.store(true, Ordering::SeqCst);
        let _ = s.cmd_tx.send(ProductCmd::Abandon);
        if let Some(j) = s.join.take() {
            let _ = j.join();
        }
    }

    fn on_new_game(&mut self, app: &Arc<App>, msg: &Value) -> Vec<Value> {
        let t_decks = Instant::now();
        let deck_spec = msg.get("deck").cloned().unwrap_or(Value::Null);
        let bot_spec = msg.get("bot_deck").cloned().unwrap_or_else(|| deck_spec.clone());
        let first = msg.get("first").and_then(Value::as_str).unwrap_or("rps");
        let mut mode = msg
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or(&app.default_opponent)
            .to_string();
        if let Some(obj) = deck_spec.as_object()
            && let Some(text) = obj.get("ydk").and_then(Value::as_str)
        {
            let id = format!("pasted-{}", chrono::Local::now().format("%Y%m%d-%H%M%S"));
            self.deck_catalog.user_decks.insert(id, text.to_string());
        }
        if app.test_stream {
            mode = "stream".to_string();
        } else if !matches!(mode.as_str(), "product" | "fast" | "random") {
            return Self::error(format!("unknown mode {} (product|fast|random)", py_repr(Some(&mode))));
        }
        let practice = match msg.get("practice") {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Self::error("practice must be true|false"),
        };
        if let Some(c) = msg.get("chain").and_then(Value::as_str) {
            if c == "auto" || c == "all" {
                self.chain_mode = c.to_string();
            }
        }
        let model_id = msg.get("model").and_then(Value::as_str).unwrap_or(&app.model_id);
        if model_id != app.model_id && mode != "stream" && mode != "random" {
            return Self::error(format!("unknown model {}", py_repr(Some(model_id))));
        }
        let (human_deck, err) = self.deck_catalog.resolve(&deck_spec);
        let Some(human_deck) = human_deck else {
            return Self::error(err.unwrap_or_else(|| "unknown deck".into()));
        };
        if let Some(e) = check_deck(&human_deck, &app.cards, &app.emb_codes, app.ctx.texts.names()) {
            return Self::error(e);
        }
        let (bot_deck, err) = self.deck_catalog.resolve(&bot_spec);
        let Some(bot_deck) = bot_deck else {
            return Self::error(err.unwrap_or_else(|| "unknown deck".into()));
        };
        if let Some(e) = check_deck(&bot_deck, &app.cards, &app.emb_codes, app.ctx.texts.names()) {
            return Self::error(e);
        }
        let mut art_codes = human_deck.main.clone();
        art_codes.extend(human_deck.extra.iter().copied());
        art_codes.extend(bot_deck.main.iter().copied());
        art_codes.extend(bot_deck.extra.iter().copied());
        app.art.note_deck(&art_codes);
        timing("newgame.decks", t_decks);
        let deck_name = |spec: &Value| {
            spec.as_str().filter(|s| *s != "pasted").map(pretty_deck_name).unwrap_or_else(|| "Pasted deck".into())
        };
        let (hname, bname) = (deck_name(&deck_spec), deck_name(&bot_spec));
        let names = (hname.clone(), bname.clone());
        // Human/bot order whoever goes first: the primer and "opponent's deck" read .0 as the human's, .1 as the bot's.
        self.live_decks = Some((human_deck.clone(), bot_deck.clone()));
        self.live_deck_names = (hname, bname);
        let (human_seat, first_deck, second_deck) = match first {
            "me" => (0, human_deck, bot_deck),
            "bot" => (1, bot_deck, human_deck),
            "rps" => {
                if random_u64() & 1 == 0 {
                    (0, human_deck, bot_deck)
                } else {
                    (1, bot_deck, human_deck)
                }
            }
            _ => return Self::error(format!("unknown first {} (rps|me|bot)", py_repr(Some(first)))),
        };
        let seed = match app.fixed_seed {
            Some(n) => n.wrapping_add(self.games_started),
            None => random_u64() >> 1,
        };
        self.games_started += 1;
        let t_onnx = Instant::now();
        let product_value = if mode == "product" {
            match app.onnx_pool.take().and_then(|v| Ok((app.onnx_pool.take()?, v))) {
                Ok(v) => Some(v),
                Err(e) => return Self::error(format!("could not start the product opponent: {e}")),
            }
        } else {
            None
        };
        if mode == "product" {
            timing("newgame.onnx", t_onnx);
        }
        let bot = if app.test_stream || mode == "stream" {
            // StreamBot(N): the opponent stream is seeded by the server's fixed seed, not the duel seed.
            Bot::Stream(SplitMix(app.fixed_seed.unwrap_or(0)))
        } else if mode == "random" {
            Bot::Random(SplitMix(random_u64()))
        } else if mode == "product" {
            Bot::Raw // unused: `Game::step_bot` takes the choices from the product worker
        } else {
            if let Err(e) = app.ensure_raw() {
                return Self::error(format!("could not start the {mode} opponent: {e}"));
            }
            Bot::Raw
        };
        let session = Arc::new(ProductControl::default());
        let launch = product_value.map(|(search, value)| ProductLaunch {
            search,
            replay_threads: app.replay_threads,
            value,
            session: session.clone(),
        });
        let t_engine = Instant::now();
        let mut game = match Game::new(
            app,
            seed,
            human_seat,
            &first_deck,
            &second_deck,
            bot,
            &mode,
            practice,
            launch,
        ) {
            Ok(g) => g,
            Err(e) => return Self::error(e),
        };
        timing("newgame.engine", t_engine);
        // Seat order from the seat the human actually got (with "rps" that is only known here).
        game.deck_names = if human_seat == 0 { [names.0, names.1] } else { [names.1, names.0] };
        if mode == "product" {
            let t_spawn = Instant::now();
            let Some(conn_tx) = self.conn_tx.clone() else {
                return Self::error("no client connection to run the product duel");
            };
            let sink = Sink::default();
            sink.set(Some(conn_tx));
            let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
            let control = session;
            control.advancing.fetch_add(1, Ordering::SeqCst); // the opening advance
            let control_thread = control.clone();
            let app2 = app.clone();
            let chain = self.chain_mode.clone();
            let wc = self.winchance_on;
            let sink_thread = sink.clone();
            let join = std::thread::Builder::new()
                .name("product-duel".into())
                .spawn(move || run_product_session(app2, game, sink_thread, cmd_rx, control_thread, chain, wc))
                .map_err(|e| format!("could not start the product duel: {e}"));
            match join {
                Ok(join) => {
                    self.product = Some(ProductSession { cmd_tx, control, join: Some(join), sink });
                    timing("newgame.spawn", t_spawn);
                    Vec::new()
                }
                Err(e) => Self::error(e),
            }
        } else {
            let mut frames = Vec::new();
            game.advance(app, &self.chain_mode, self.winchance_on, &mut frames);
            self.game = Some(game);
            frames
        }
    }

    fn on_choose(&mut self, app: &App, msg: &Value) -> Vec<Value> {
        let games = self.game.is_some();
        if !games {
            return Self::error("no active game (send new_game first)");
        }
        let g = self.game.as_mut().unwrap();
        if g.finished {
            return Self::error("no active game (send new_game first)");
        }
        if !g.need_human() || g.advancing {
            return Self::error("no prompt is waiting for you right now");
        }
        let want = msg.get("prompt");
        let same = match want {
            Some(Value::Number(n)) => n.as_i64() == Some(g.prompt_id),
            Some(Value::Null) | None => false,
            _ => false,
        };
        if !same {
            return Self::error("stale prompt (a newer one is waiting)");
        }
        let index = match msg.get("index") {
            Some(Value::Number(n)) => n.as_i64(),
            Some(Value::String(s)) => s.parse::<i64>().ok(),
            _ => None,
        };
        let Some(index) = index else {
            return Self::error("index must be an option number");
        };
        let n = g.env.num_options();
        if !(0..n as i64).contains(&index) {
            return Self::error(format!("illegal option: option {index} of {n}"));
        }
        let before = g.env.choices().len();
        let dec = g.env.decision();
        let is_chain = dec.as_ref().is_some_and(|d| {
            d.msg_type == C::MSG_SELECT_CHAIN as i32 && d.player == g.human_seat
        });
        let had = is_chain && dec.as_ref().is_some_and(|d| d.options.iter().any(|o| o.kind == 6));
        let used = is_chain
            && dec.as_ref().is_some_and(|d| d.options.get(index as usize).is_some_and(|o| o.kind == 6));
        if let Err(e) = g.human_step(index as i32) {
            return Self::error(format!("illegal option: {e}"));
        }
        g.human_decisions.push(before);
        if had {
            g.stats.note_chain_window(true, used);
        }
        let mut frames = Vec::new();
        g.advance(app, &self.chain_mode, self.winchance_on, &mut frames);
        frames
    }

    fn on_surrender(&mut self, app: &App) -> Vec<Value> {
        let Some(g) = self.game.as_mut() else {
            return Self::error("no active game (send new_game first)");
        };
        if g.finished {
            return Self::error("no active game (send new_game first)");
        }
        let mut frames = Vec::new();
        g.surrender(app, self.winchance_on, &mut frames);
        frames
    }

    fn on_settings(&mut self, app: &App, msg: &Value) -> Vec<Value> {
        let has_chain = msg.get("chain").is_some();
        let has_wc = msg.get("winchance").is_some();
        if !has_chain && !has_wc {
            return Self::error("settings.chain must be auto|all");
        }
        if has_chain {
            match msg.get("chain").and_then(Value::as_str) {
                Some("auto") => self.chain_mode = "auto".into(),
                Some("all") => self.chain_mode = "all".into(),
                _ => return Self::error("settings.chain must be auto|all"),
            }
        }
        if has_wc {
            match msg.get("winchance") {
                Some(Value::Bool(b)) => self.winchance_on = *b,
                _ => return Self::error("settings.winchance must be true|false"),
            }
        }
        let mut frames = Vec::new();
        if has_chain
            && self.chain_mode == "auto"
            && let Some(g) = self.game.as_mut()
            && !g.finished
            && g.need_human()
            && !g.advancing
        {
            g.advance(app, &self.chain_mode, self.winchance_on, &mut frames);
        }
        if has_wc
            && let Some(g) = self.game.as_mut()
            && !g.advancing
        {
            g.send_state(app, false, false, self.winchance_on, &mut frames);
        }
        frames
    }
}

/// Gives back what the duel thread still owes of `ProductControl::advancing` when dropped (every exit path of a
/// command; usually the game gave it back already when it reached the human's prompt).
struct Settle<'a>(&'a ProductControl);

impl Drop for Settle<'_> {
    fn drop(&mut self) {
        self.0.settle_advancing();
    }
}

/// A product duel's thread: owns the host env + the search worker, sends every frame to the socket channel, and
/// answers the socket's commands. The search runs here, so the socket keeps reading (surrender cuts through).
fn run_product_session(
    app: Arc<App>,
    mut game: Game,
    sink: Sink,
    cmd_rx: std::sync::mpsc::Receiver<ProductCmd>,
    control: Arc<ProductControl>,
    mut chain: String,
    mut winchance: bool,
) {
    let send = |frames: Vec<Value>| {
        for f in frames {
            sink.send(Outgoing::Frame(f));
        }
    };
    game.live_tx = Some(sink.clone());
    {
        control.owed.fetch_add(1, Ordering::SeqCst); // the opening advance, counted when the session started
        let _done = Settle(&control);
        let mut frames = Vec::new();
        let t = Instant::now();
        game.advance(&app, &chain, winchance, &mut frames);
        timing("newgame.first_advance", t);
        send(frames);
    }
    // The thread outlives the game: after the end it answers review, review_frame and what-if (a what-if swaps in
    // a new game) until the socket abandons it.
    loop {
        control.game_over.store(game.finished, Ordering::SeqCst);
        if control.cancel.load(Ordering::SeqCst) {
            break;
        }
        let cmd = match cmd_rx.recv() {
            Ok(c) => c,
            Err(_) => break,
        };
        // Choose and Undo were counted in `control.advancing` by the socket; uncount when this command is done.
        if matches!(cmd, ProductCmd::Choose { .. } | ProductCmd::Undo | ProductCmd::Cancel | ProductCmd::WhatIf(_)) {
            control.owed.fetch_add(1, Ordering::SeqCst);
        }
        let _done = Settle(&control);
        match cmd {
            ProductCmd::Choose { prompt, index } => {
                if game.finished {
                    send(Inner::error("no active game (send new_game first)"));
                    continue;
                }
                if !game.need_human() || game.advancing {
                    send(Inner::error("no prompt is waiting for you right now"));
                    continue;
                }
                if prompt != game.prompt_id {
                    send(Inner::error("stale prompt (a newer one is waiting)"));
                    continue;
                }
                let n = game.env.num_options() as i64;
                if index < 0 || index >= n {
                    send(Inner::error(format!("illegal option: option {index} of {n}")));
                    continue;
                }
                let before = game.env.choices().len();
                let dec = game.env.decision();
                let is_chain = dec.as_ref().is_some_and(|d| {
                    d.msg_type == C::MSG_SELECT_CHAIN as i32 && d.player == game.human_seat
                });
                let had = is_chain && dec.as_ref().is_some_and(|d| d.options.iter().any(|o| o.kind == 6));
                let used = is_chain
                    && dec
                        .as_ref()
                        .is_some_and(|d| d.options.get(index as usize).is_some_and(|o| o.kind == 6));
                let starts = dec.as_ref().is_some_and(|d| {
                    d.msg_type == C::MSG_SELECT_IDLECMD as i32 || d.msg_type == C::MSG_SELECT_BATTLECMD as i32
                });
                let mut frames = Vec::new();
                match game.human_step(index as i32) {
                    Ok(()) => {
                        if starts {
                            game.action = Some((game.human_decisions.len(), game.bot_moves));
                        }
                        game.human_decisions.push(before);
                        if had {
                            game.stats.note_chain_window(true, used);
                        }
                        game.advance(&app, &chain, winchance, &mut frames);
                    }
                    Err(e) => frames = Inner::error(format!("illegal option: {e}")),
                }
                send(frames);
            }
            ProductCmd::Resync => {
                let mut frames = Vec::new();
                game.send_state(&app, true, false, winchance, &mut frames);
                send(frames);
            }
            ProductCmd::Hint => send(game.hint(&app, winchance)),
            ProductCmd::Undo => {
                // A surrender (or new game / disconnect) can land between the socket's send and this command: it
                // must cancel the undo silently, like Python's generation check, not report a failed replay.
                if control.stopping() {
                    continue;
                }
                match game.undo(&app) {
                    Ok(new_game) => {
                        // The old game's worker stops when it drops; the new one waits at the rewound prompt.
                        game = new_game;
                        let mut frames = Vec::new();
                        game.advance(&app, &chain, winchance, &mut frames);
                        send(frames);
                    }
                    Err(e) => {
                        if !control.stopping() {
                            send(Inner::error(e));
                        }
                    }
                }
            }
            ProductCmd::Cancel => {
                if control.stopping() {
                    continue;
                }
                match game.cancel(&app) {
                    Ok(new_game) => {
                        game = new_game;
                        let mut frames = Vec::new();
                        game.advance(&app, &chain, winchance, &mut frames);
                        send(frames);
                    }
                    Err(e) => {
                        if !control.stopping() {
                            send(Inner::error(e));
                        }
                    }
                }
            }
            ProductCmd::Review => {
                let frames = run_review(&app, &mut game, &control, &sink);
                send(frames);
            }
            ProductCmd::ReviewFrame(msg) => send(game.review_frame(&app, &msg)),
            ProductCmd::WhatIf(msg) => {
                if control.stopping() {
                    continue;
                }
                match game.whatif(&app, &msg) {
                    Ok(new_game) => {
                        game = new_game;
                        let mut frames = Vec::new();
                        game.advance(&app, &chain, winchance, &mut frames);
                        send(frames);
                    }
                    Err(e) => {
                        if !control.stopping() {
                            send(Inner::error(e));
                        }
                    }
                }
            }
            ProductCmd::Settings { chain: c, winchance: w } => {
                chain = c;
                winchance = w;
                let mut frames = Vec::new();
                if chain == "auto" && game.need_human() && !game.advancing {
                    // The socket does not count a Settings-triggered advance: count it here, like Python's
                    // start_advance, so a hint/undo sent meanwhile is refused instead of queued behind it.
                    control.advancing.fetch_add(1, Ordering::SeqCst);
                    control.owed.fetch_add(1, Ordering::SeqCst);
                    game.advance(&app, &chain, winchance, &mut frames);
                } else if winchance && game.need_human() && !game.advancing {
                    game.compute_winchance(&app);
                    game.send_state(&app, false, false, winchance, &mut frames);
                }
                send(frames);
            }
            ProductCmd::Surrender => {
                // A game that ended meanwhile keeps its record (the socket checked `game_over` before stopping).
                if !game.finished {
                    let mut frames = Vec::new();
                    game.surrender(&app, winchance, &mut frames);
                    send(frames);
                }
                // The cancelled search worker is done (the record has its seeds); review and what-if need only the
                // value nets.
                if let Some(mut p) = game.product.take() {
                    p.close();
                }
                control.surrender.store(false, Ordering::SeqCst);
            }
            ProductCmd::Abandon => break,
        }
    }
    if control.stopping() && !game.finished {
        let mut frames = Vec::new();
        if control.surrender.load(Ordering::SeqCst) {
            game.surrender(&app, winchance, &mut frames);
        } else if control.disconnect.load(Ordering::SeqCst) {
            game.disconnect_record(&app);
        }
        send(frames);
    }
    if let Some(mut p) = game.product.take() {
        if let Some(e) = p.close() {
            if !control.stopping() {
                eprintln!("ALERT web: product worker stopped with: {e}");
            }
        }
    }
}

/// The review cache next to a match record: `<record id>.review.json` (Python's `review.json` in the game dir;
/// Rust records share one directory, so the id keeps them apart).
fn review_cache_path(record: &Path) -> PathBuf {
    let name = record.file_name().and_then(|n| n.to_str()).unwrap_or("review");
    let stem = name.strip_suffix(".match.json.gz").unwrap_or(name);
    record.with_file_name(format!("{stem}.review.json"))
}

/// Port of `Server.on_review` + `_run_review` on the duel thread: the cached review, or a new one with
/// `review_progress` frames while it runs. Python's review.json cache lives on the game here (the thread keeps it).
fn run_review(app: &App, game: &mut Game, control: &ProductControl, sink: &Sink) -> Vec<Value> {
    if !game.finished || game.result.is_none() {
        return Inner::error("no finished game to review (finish a game first)");
    }
    if game.value.is_none() {
        return Inner::error("Review needs the full bot");
    }
    if game.review.is_none()
        && let Some(record) = game.record_path.as_ref()
    {
        let cache = review_cache_path(record);
        if let Ok(text) = std::fs::read_to_string(&cache)
            && let Ok(v) = serde_json::from_str::<Value>(&text)
        {
            game.review = Some(v);
        }
    }
    if let Some(r) = game.review.as_ref() {
        let mut out = r.clone();
        out["t"] = json!("review");
        return vec![out];
    }
    if game.record_path.is_none() {
        return Inner::error("no finished game to review (finish a game first)");
    }
    let map = match WcMap::load(&app.root.join("src/decklight/web/winchance_calib.json")) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ALERT web: review calib unavailable: {e}");
            return Inner::error("review is unavailable (calibration missing)");
        }
    };
    let progress = |done: usize, total: usize| {
        sink.send(Outgoing::Frame(json!({"t": "review_progress", "done": done, "total": total})));
    };
    match game.review(app, &map, &|| control.stopping(), &progress) {
        Ok(Some(r)) => {
            if let Some(record) = game.record_path.as_ref() {
                let cache = review_cache_path(record);
                if let Err(e) = std::fs::write(&cache, format!("{}\n", r)) {
                    eprintln!("ALERT web: could not write review cache {}: {e}", cache.display());
                }
            }
            game.review = Some(r.clone());
            let mut out = r;
            out["t"] = json!("review");
            vec![out]
        }
        Ok(None) => Vec::new(),
        Err(e) => {
            eprintln!("ALERT web: review crashed: {e}");
            Inner::error(format!("review failed: {e} (see server log)"))
        }
    }
}

/// Win chance per candidate by one-step lookahead (port of `review.estimate_candidates`): replay the game,
/// take the candidate, then both sides play the policy's argmax on their own observation until the human's
/// next real decision, the end of the turn, the end of the game or 32 steps. The value net reads the human's
/// view of each final position (one batch) and the calibration map turns it into a probability.
/// Shared by the practice hint, the review and the Ask tab's `bot_view`; `stop` is polled between candidates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_candidates(
    app: &App,
    cfg: &DuelConfig,
    seat: i32,
    nets: &mut decklight_onnx::OnnxNets,
    prefix: &[i32],
    cands: &[usize],
    map: &WcMap,
    stop: &dyn Fn() -> bool,
) -> Result<Vec<f64>, String> {
    const MAX_LOOKAHEAD: usize = 32;
    let mut finals: Vec<(Env<'static>, bool)> = Vec::new();
    for &c in cands {
        if stop() {
            return Err("stopped".into());
        }
        let mut env = Env::new(app.engine, cfg, false)?;
        let mut obs = Obs::zeroed();
        if !env.replay(prefix, &mut obs)? {
            finals.push((env, true));
            continue;
        }
        let start_turn = env.duel_state()?.turn;
        let mut steps = 0;
        let mut pending = env.step(c as i32, &mut obs)?;
        while pending && steps < MAX_LOOKAHEAD {
            if env.duel_state()?.turn != start_turn {
                break;
            }
            if env.player() == seat && env.num_options() > 1 {
                break;
            }
            let lg = nets.policy_logits(&[&obs])?.into_iter().next().ok_or("no policy output")?;
            pending = env.step(argmax_masked(&lg, &obs.action_mask) as i32, &mut obs)?;
            steps += 1;
        }
        let done = !pending || env.done();
        finals.push((env, done));
    }
    let mut views: Vec<Box<Obs>> = Vec::new();
    for (env, done) in finals.iter_mut() {
        if !*done {
            let mut o = Obs::zeroed();
            env.observe_as(seat, &mut o)?;
            views.push(o);
        }
    }
    let refs: Vec<&Obs> = views.iter().map(|o| &**o).collect();
    let vals = if refs.is_empty() { Vec::new() } else { nets.values(&refs)? };
    let mut vi = vals.into_iter();
    let mut out = Vec::new();
    for (env, done) in &finals {
        let st = env.duel_state()?;
        out.push(if *done {
            if st.winner == seat { 1.0 } else { 0.0 }
        } else {
            map.p(vi.next().ok_or("value batch too short")? as f64, st.turn)
        });
    }
    Ok(out)
}


/// `review._masked_softmax`: float64 softmax over the legal entries (NumPy's pairwise sum); uniform over the legal
/// entries when the sum is not a positive finite number.
pub(crate) fn masked_softmax(logits: &[f32], action_mask: &[u8]) -> Vec<f64> {
    let legal: Vec<bool> = action_mask.iter().map(|&x| x != 0).collect();
    let n = legal.len();
    let mut z: Vec<f64> = (0..n)
        .map(|i| if legal[i] { logits.get(i).copied().unwrap_or(f32::NEG_INFINITY) as f64 } else { f64::NEG_INFINITY })
        .collect();
    let mx = z.iter().copied().filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, f64::max);
    for v in z.iter_mut() {
        *v -= mx;
    }
    let e: Vec<f64> = (0..n).map(|i| if legal[i] { z[i].exp() } else { 0.0 }).collect();
    let s = pairwise_sum(&e);
    if !s.is_finite() || s <= 0.0 {
        let k = legal.iter().filter(|&&b| b).count().max(1) as f64;
        return legal.iter().map(|&b| if b { 1.0 / k } else { 0.0 }).collect();
    }
    e.iter().map(|&x| x / s).collect()
}

/// `review._argmax_masked`: the first index of the largest legal logit (float64).
pub(crate) fn argmax_masked(logits: &[f32], action_mask: &[u8]) -> usize {
    let z: Vec<f64> = (0..action_mask.len())
        .map(|i| {
            if action_mask[i] != 0 { logits.get(i).copied().unwrap_or(f32::NEG_INFINITY) as f64 } else { f64::NEG_INFINITY }
        })
        .collect();
    argmax_select(&z).unwrap_or(0)
}

// ---------------------------------------------------------------- puzzles / primers / study (A4b)

fn int_val(v: Option<&Value>, default: Option<i64>) -> Option<i64> {
    match v {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)),
        Some(Value::String(s)) => s.trim().parse::<i64>().ok(),
        Some(Value::Bool(b)) => Some(*b as i64),
        Some(Value::Null) | None => default,
        _ => default,
    }
}

/// Python `str(value)` for the JSON scalar kinds our ids use.
fn py_str_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn py_value_repr(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{s}'"),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn load_puzzles(app: &App) -> Vec<Value> {
    if let Some(v) = app.puzzles.lock().unwrap().as_ref() {
        return v.clone();
    }
    let path = app.root.join("src/decklight/web/puzzles.json");
    let out = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    *app.puzzles.lock().unwrap() = Some(out.clone());
    out
}

fn puzzle_item(p: &Value) -> Value {
    let decks: Vec<String> = p
        .get("decks")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|d| d.as_str().unwrap_or("").to_string()).collect())
        .unwrap_or_default();
    let mut names: Vec<String> = decks.iter().take(2).map(|d| pretty_deck_name(d)).collect();
    while names.len() < 2 {
        names.push("?".into());
    }
    json!({
        "id": py_str_value(p.get("id").unwrap_or(&Value::Null)),
        "decks": names,
        "turn": int_val(p.get("turn"), Some(1)).unwrap_or(1),
        "difficulty": p.get("difficulty").and_then(Value::as_str).unwrap_or("medium"),
        "goal": p.get("goal").and_then(Value::as_str).unwrap_or(""),
        "n_plays": int_val(p.get("n_plays").or_else(|| p.get("n_moves")), Some(0)).unwrap_or(0),
        "seat": int_val(p.get("seat"), Some(0)).unwrap_or(0),  // the human plays decks[seat]
    })
}

fn load_puzzle_deck(root: &Path, name: &str) -> Result<DeckSpec, String> {
    for dir in ["pool", "gauntlet", "newcards"] {
        let p = root.join("decks").join(dir).join(format!("{name}.ydk"));
        if p.is_file() {
            return crate::decks::load_ydk(&p).map_err(|e| e.to_string());
        }
    }
    let p = root.join("data/primers/studied/decks").join(format!("{name}.ydk"));
    if p.is_file() {
        return crate::decks::load_ydk(&p).map_err(|e| e.to_string());
    }
    Err(format!("deck {name} not found in decks/pool|gauntlet|newcards"))
}

fn puzzle_cfg(decks: &[DeckSpec], seed: u64) -> DuelConfig {
    let mut main0 = decks[0].main.clone();
    let mut main1 = decks[1].main.clone();
    decklight_cards::shuffle_main(&mut main0, seed, 0);
    decklight_cards::shuffle_main(&mut main1, seed, 1);
    DuelConfig {
        main0,
        extra0: decks[0].extra.clone(),
        main1,
        extra1: decks[1].extra.clone(),
        seed,
        flags: TCG_FLAGS,
        lp: 8000,
        hand: 5,
        draw: 1,
        seed_state: None,
        team_order: [0, 1],
        edopro_add: false,
    }
}

fn replay_log(app: &App, env: &Env<'_>, since: usize) -> Vec<Value> {
    app.ctx
        .build_events(env, 0, since)
        .into_iter()
        .filter(|e| {
            let k = e.get("kind").and_then(Value::as_str).unwrap_or("").to_lowercase();
            k != "select" && k != "position"
        })
        .collect()
}

/// Port of `web.replay.replay_frames`: one spectator frame per choice plus the final board frame.
fn replay_frames(
    app: &App,
    cfg: &DuelConfig,
    choices: &[i32],
    names: &(String, String),
    expect_winner: Option<i32>,
) -> Result<Vec<Value>, String> {
    let mut env = Env::new(app.engine, cfg, false)?;
    env.set_record_trace(true); // spectator cues read the raw message trace
    let mut obs = Obs::zeroed();
    env.reset(&mut obs)?;
    let mut frames = Vec::new();
    let mut stats = DuelStats::new(0, cfg.lp as i64);
    let (mut prev_len, mut prev_trace) = (0usize, 0usize);
    for (k, &idx) in choices.iter().enumerate() {
        if env.done() {
            return Err(format!("replay ended early at step {k} of {}", choices.len()));
        }
        let seat = env.player();
        let prompt = app.ctx.build_prompt(&env, seat, k as i64);
        let mut label = String::new();
        if let Some(p) = prompt.as_ref()
            && let Some(opts) = p.get("options").and_then(Value::as_array)
        {
            for o in opts {
                if o.get("index").and_then(Value::as_u64) == Some(idx as u64) {
                    label = o.get("label").and_then(Value::as_str).unwrap_or("").to_string();
                    break;
                }
            }
        }
        if label.is_empty() {
            label = app.ctx.fallback_label(&env, seat, idx.max(0) as usize);
        }
        let view = app.ctx.build_view(&env, 0, true);
        let log = replay_log(app, &env, prev_len);
        let (cues, _) = decklight_view::build_cues(&env, &app.ctx, 0, prev_trace, true);
        stats.feed_cues(&cues);
        stats.note_view(Some(&view));
        let mut frame = json!({
            "t": "state",
            "view": view,
            "prompt": Value::Null,
            "log": log,
            "cues": cues,
            "stats": stats.stats_message(false),
            "thinking": Value::Null,
            "result": Value::Null,
            "caption": if k == 0 { Value::Null } else { json!({"actor": if seat == 0 { &names.0 } else { &names.1 }, "text": label}) },
        });
        if k == 0 {
            frame["title"] = json!(format!("{} vs {} · seed {} · decklight", names.0, names.1, cfg.seed));
        }
        frame["names"] = json!([names.0, names.1]);
        frames.push(frame);
        prev_len = env.history_log().len();
        prev_trace = env.trace_len();
        env.step(idx, &mut obs)?;
    }
    let (winner, reason) = if env.done() {
        let w = env.winner();
        if let Some(exp) = expect_winner
            && w != exp
        {
            return Err(format!("replay winner {w} != record winner {exp}"));
        }
        (w, "win")
    } else {
        return Err(format!(
            "replay of {} choices did not finish the duel (done=false, winner={})",
            choices.len(),
            env.winner()
        ));
    };
    let view = app.ctx.build_view(&env, 0, true);
    let log = replay_log(app, &env, prev_len);
    let (cues, _) = decklight_view::build_cues(&env, &app.ctx, 0, prev_trace, true);
    stats.feed_cues(&cues);
    stats.note_view(Some(&view));
    frames.push(json!({
        "t": "state",
        "view": view,
        "prompt": Value::Null,
        "log": log,
        "cues": cues,
        "stats": stats.stats_message(false),
        "thinking": Value::Null,
        "result": {"winner": if winner == 0 { &names.0 } else { &names.1 }, "reason": reason},
        "caption": {"actor": if winner == 0 { &names.0 } else { &names.1 }, "text": "Wins the duel"},
        "names": [names.0, names.1],
    }));
    Ok(frames)
}

fn parse_ratio(s: &str) -> Option<(i64, i64)> {
    let (a, b) = s.split_once('/')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// `analysis.primer.hash_of`: sha1 over sorted main + "|" + sorted extra.
fn primer_hash(deck: &DeckSpec) -> String {
    let mut main = deck.main.clone();
    main.sort_unstable();
    let mut extra = deck.extra.clone();
    extra.sort_unstable();
    let mut h = Sha1::new();
    h.update(main.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",").as_bytes());
    h.update(b"|");
    h.update(extra.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",").as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn load_primers(app: &App) -> Value {
    if let Some(v) = app.primers.lock().unwrap().as_ref() {
        return v.clone();
    }
    let path = app.root.join("src/decklight/web/primers.json.gz");
    let mut out = json!({"decks": {}, "negated_by": {}});
    if let Ok(bytes) = std::fs::read(&path) {
        let mut text = String::new();
        if path.to_string_lossy().ends_with(".gz") {
            let _ = flate2::read::GzDecoder::new(&bytes[..]).read_to_string(&mut text);
        } else {
            text = String::from_utf8_lossy(&bytes).into_owned();
        }
        if let Ok(v) = serde_json::from_str::<Value>(&text)
            && v.get("decks").map(Value::is_object) == Some(true)
        {
            out = v;
        }
    }
    if out.get("decks").map(Value::is_object) != Some(true) {
        out = json!({"decks": {}, "negated_by": {}});
    }
    if out.get("negated_by").map(Value::is_object) != Some(true) {
        out["negated_by"] = json!({});
    }
    let studied_path = app.root.join("data/primers/studied/primers.json");
    if let Ok(text) = std::fs::read_to_string(studied_path)
        && let Ok(studied) = serde_json::from_str::<Value>(&text)
    {
        if let Some(sd) = studied.get("decks").and_then(Value::as_object) {
            if let Some(d) = out.get_mut("decks").and_then(Value::as_object_mut) {
                for (h, e) in sd {
                    if !d.contains_key(h) {
                        d.insert(h.clone(), e.clone());
                    }
                }
            }
        }
        if let Some(sneg) = studied.get("negated_by").and_then(Value::as_object)
            && let Some(neg) = out.get_mut("negated_by").and_then(Value::as_object_mut)
        {
            for (k, victims) in sneg {
                let Some(victims) = victims.as_object() else { continue };
                let slot = neg.entry(k.clone()).or_insert_with(|| json!({}));
                if let Some(slot) = slot.as_object_mut() {
                    for (v, n) in victims {
                        let add = n.as_i64().unwrap_or(0);
                        let cur = slot.get(v).and_then(Value::as_i64).unwrap_or(0);
                        slot.insert(v.clone(), json!(cur + add));
                    }
                }
            }
        }
    }
    *app.primers.lock().unwrap() = Some(out.clone());
    out
}

fn canon_code(app: &App, code: u32) -> u32 {
    app.cards.get(&code).map(|c| c.alias).filter(|a| *a != 0).unwrap_or(code)
}

fn main_codes(deck: &DeckSpec) -> Vec<u32> {
    deck.main.clone()
}

fn multiset_overlap(a: &[u32], b: &[u32]) -> usize {
    let mut ca: HashMap<u32, usize> = HashMap::new();
    for &c in a {
        *ca.entry(c).or_insert(0) += 1;
    }
    let mut n = 0;
    for &c in b {
        if let Some(x) = ca.get_mut(&c)
            && *x > 0
        {
            *x -= 1;
            n += 1;
        }
    }
    n
}

fn primer_names_of(app: &App, store: &Value, code: i64) -> String {
    if let Some(decks) = store.get("decks").and_then(Value::as_object) {
        for d in decks.values() {
            for section in ["key_cards", "chokepoints", "hand_traps_seen"] {
                for item in d.get(section).and_then(Value::as_array).into_iter().flatten() {
                    if item.get("code").and_then(Value::as_i64) == Some(code) {
                        return item.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                    }
                    if let Some(subs) = item.get("answered").and_then(Value::as_array) {
                        for sub in subs {
                            if sub.get("code").and_then(Value::as_i64) == Some(code) {
                                return sub.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                            }
                        }
                    }
                }
            }
        }
    }
    app.ctx.texts.names().get(&(code as u32)).cloned().unwrap_or_else(|| format!("#{code}"))
}

const OTHER_DECKS_NOTE: &str = "used against other decks";
const MIN_GAMES: i64 = 20;
const SIMILAR_MIN: f64 = 0.80;
const MIN_NEGATIONS: i64 = 3;

fn primer_your_answers(app: &App, store: &Value, p: &Value, my_deck: &DeckSpec) -> Value {
    let mut mine: Vec<i64> = my_deck
        .main
        .iter()
        .chain(my_deck.extra.iter())
        .map(|&c| canon_code(app, c) as i64)
        .collect();
    mine.sort_unstable();
    mine.dedup();
    let ht_here: HashMap<i64, &Value> = p
        .get("hand_traps_seen")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|h| Some((h.get("code")?.as_i64()?, h)))
        .collect();
    let mut ht_global: BTreeSet<i64> = BTreeSet::new();
    if let Some(decks) = store.get("decks").and_then(Value::as_object) {
        for d in decks.values() {
            for h in d.get("hand_traps_seen").and_then(Value::as_array).into_iter().flatten() {
                if let Some(c) = h.get("code").and_then(Value::as_i64) {
                    ht_global.insert(c);
                }
            }
        }
    }
    let negated = store.get("negated_by").and_then(Value::as_object);
    let mut answers: Vec<Value> = Vec::new();
    for c in mine {
        let here = ht_here.get(&c).copied();
        let hits: Vec<Value> = here
            .and_then(|h| h.get("answered").and_then(Value::as_array).cloned())
            .unwrap_or_default();
        let mut neg_victims: Vec<(i64, i64)> = Vec::new();
        if let Some(v) = negated.and_then(|n| n.get(&c.to_string())).and_then(Value::as_object) {
            let total: i64 = v.values().map(|n| n.as_i64().unwrap_or(0)).sum();
            if total >= MIN_NEGATIONS {
                neg_victims = v.iter().filter_map(|(k, n)| Some((k.parse().ok()?, n.as_i64()?))).collect();
            }
        }
        if !ht_global.contains(&c) && neg_victims.is_empty() {
            continue;
        }
        let name = primer_names_of(app, store, c);
        if !hits.is_empty() {
            let mut a = json!({"code": c, "name": name, "hits": hits});
            if let Some(g) = here.and_then(|h| h.get("gain")) {
                a["gain"] = g.clone();
            }
            answers.push(a);
        } else if !neg_victims.is_empty() {
            neg_victims.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let victims: Vec<Value> = neg_victims
                .iter()
                .take(6)
                .map(|&(v, n)| json!({"code": v, "name": primer_names_of(app, store, v), "times": n}))
                .collect();
            answers.push(json!({"code": c, "name": name, "note": OTHER_DECKS_NOTE, "hits_elsewhere": victims}));
        } else {
            answers.push(json!({"code": c, "name": name, "note": OTHER_DECKS_NOTE, "hits": []}));
        }
    }
    answers.sort_by(|a, b| {
        let ah = a.get("hits").and_then(Value::as_array).is_some_and(|h| !h.is_empty());
        let bh = b.get("hits").and_then(Value::as_array).is_some_and(|h| !h.is_empty());
        (!ah).cmp(&(!bh)).then(
            a.get("code").and_then(Value::as_i64).unwrap_or(0).cmp(&b.get("code").and_then(Value::as_i64).unwrap_or(0)),
        )
    });
    Value::Array(answers)
}

fn primer_for(app: &App, store: &Value, bot_deck: &DeckSpec, my_deck: &DeckSpec) -> Option<Value> {
    let decks = store.get("decks")?.as_object()?;
    let h = primer_hash(bot_deck);
    if let Some(p) = decks.get(&h) {
        let games = p.get("games").and_then(Value::as_i64).unwrap_or(0);
        if games >= MIN_GAMES {
            let mut out = p.clone();
            out["your_answers"] = primer_your_answers(app, store, p, my_deck);
            return Some(out);
        }
    }
    let asked: Vec<u32> = main_codes(bot_deck).iter().map(|&c| canon_code(app, c)).collect();
    let mut best: Option<(&String, &Value, usize, f64, i64)> = None;
    for (hh, cand) in decks {
        let games = cand.get("games").and_then(Value::as_i64).unwrap_or(0);
        if games < MIN_GAMES {
            continue;
        }
        let Some(cmain) = cand.get("main").and_then(Value::as_array) else { continue };
        let cmain: Vec<u32> = cmain
            .iter()
            .filter_map(Value::as_u64)
            .map(|c| canon_code(app, c as u32))
            .collect();
        let larger = asked.len().max(cmain.len());
        if larger == 0 {
            continue;
        }
        let match_n = multiset_overlap(&asked, &cmain);
        let ratio = match_n as f64 / larger as f64;
        if ratio < SIMILAR_MIN {
            continue;
        }
        let better = match best {
            None => true,
            Some((bh, _, _, br, bg)) => ratio > br || (ratio == br && (games > bg || (games == bg && hh < bh))),
        };
        if better {
            best = Some((hh, cand, match_n, ratio, games));
        }
    }
    let Some((_, sp, best_match, _, _)) = best else { return None };
    let mut out = sp.clone();
    out["your_answers"] = primer_your_answers(app, store, sp, my_deck);
    out["similar"] = json!({
        "archetype": sp.get("archetype").and_then(Value::as_str).unwrap_or(""),
        "match": best_match as i64,
        "of": asked.len() as i64,
    });
    Some(out)
}

fn trim_primer(p: &Value) -> Value {
    let key_cards: Vec<Value> = p
        .get("key_cards")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(8)
        .map(|c| {
            let mut o = json!({
                "code": c.get("code").cloned().unwrap_or(Value::Null),
                "name": c.get("name").and_then(Value::as_str).unwrap_or(""),
                "games": int_val(c.get("games"), Some(0)).unwrap_or(0),
                "per_game": c.get("per_game").cloned().unwrap_or(json!(0.0)),
            });
            for k in ["impact", "per_play"] {
                if let Some(v) = c.get(k) {
                    o[k] = v.clone();
                }
            }
            o
        })
        .collect();
    let first_plays: Vec<Value> = p
        .get("first_plays")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(4)
        .map(|c| {
            json!({
                "code": c.get("code").cloned().unwrap_or(Value::Null),
                "name": c.get("name").and_then(Value::as_str).unwrap_or(""),
                "games": int_val(c.get("games"), Some(0)).unwrap_or(0),
                "share": c.get("share").cloned().unwrap_or(json!(0.0)),
            })
        })
        .collect();
    let your_answers: Vec<Value> = p
        .get("your_answers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|a| {
            let hits: Vec<Value> = a
                .get("hits")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|h| {
                    json!({
                        "code": h.get("code").cloned().unwrap_or(Value::Null),
                        "name": h.get("name").and_then(Value::as_str).unwrap_or(""),
                        "times": int_val(h.get("times"), Some(0)).unwrap_or(0),
                    })
                })
                .collect();
            let mut o = json!({
                "code": a.get("code").cloned().unwrap_or(Value::Null),
                "name": a.get("name").and_then(Value::as_str).unwrap_or(""),
                "hits": hits,
            });
            if let Some(n) = a.get("note").and_then(Value::as_str).filter(|n| !n.is_empty()) {
                o["note"] = json!(n);
            }
            o
        })
        .collect();
    let mut out = json!({
        "archetype": p.get("archetype").and_then(Value::as_str).unwrap_or(""),
        "games": int_val(p.get("games"), Some(0)).unwrap_or(0),
        "key_cards": key_cards,
        "first_plays": first_plays,
        "first_n": int_val(p.get("first_n"), Some(0)).unwrap_or(0),
        "your_answers": your_answers,
    });
    if let Some(s) = p.get("similar").filter(|s| !s.is_null()) {
        out["similar"] = json!({
            "archetype": s.get("archetype").and_then(Value::as_str).unwrap_or(""),
            "match": int_val(s.get("match"), Some(0)).unwrap_or(0),
            "of": int_val(s.get("of"), Some(0)).unwrap_or(0),
        });
    }
    if let Some(st) = p.get("study").filter(|s| !s.is_null()) {
        out["study"] = json!({
            "phase": st.get("phase").and_then(Value::as_str).unwrap_or(""),
            "games": int_val(st.get("games"), Some(0)).unwrap_or(0),
        });
    }
    out
}

fn py_repr(v: Option<&str>) -> String {
    match v {
        Some(s) => format!("'{s}'"),
        None => "None".to_string(),
    }
}

/// Echo the request id on a reply (`req`, any JSON), as `docs/design/api/cards-and-decks.md` promises.
fn with_req(mut frame: Value, req: &Option<Value>) -> Value {
    if let Some(req) = req {
        frame["req"] = req.clone();
    }
    frame
}

fn error_with_req(req: &Option<Value>, text: impl Into<String>) -> Value {
    with_req(json!({"t": "error", "error": text.into()}), req)
}

fn cards_frame(q: &str, result: &crate::cards_search::SearchResult, req: &Option<Value>) -> Value {
    let rows: Vec<Value> = result
        .rows
        .iter()
        .map(|row| {
            json!({
                "code": row.code,
                "name": row.name,
                "kind": row.kind,
                "line": row.line,
                "extra": row.extra,
                "match": {"by": row.by, "snippet": row.snippet},
            })
        })
        .collect();
    with_req(json!({"t": "cards", "q": q, "total": result.total, "rows": rows}), req)
}

fn legal_text(legal: &crate::userdecks::Legality) -> Value {
    json!({
        "ok": legal.ok,
        "problems": legal.problems.iter().map(|p| p.text.clone()).collect::<Vec<_>>(),
    })
}

fn legal_full(legal: &crate::userdecks::Legality) -> Value {
    json!({
        "ok": legal.ok,
        "problems": legal.problems.iter().map(|p| json!({
            "kind": p.kind,
            "text": p.text,
            "codes": p.codes,
        })).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------- app construction

fn download_pack(url: &str) -> Result<PathBuf, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("card pack URL must be http(s)".into());
    }
    let out = std::env::temp_dir().join(format!("decklight-pack-{}.zip", crate::record::random_hex(8)));
    let data = http_get(url, Duration::from_secs(120), decklight_cards::pack::MAX_PACK_BYTES as usize)?;
    if data.is_empty() {
        return Err("downloaded pack is empty".into());
    }
    std::fs::write(&out, &data).map_err(|e| format!("could not save the pack: {e}"))?;
    Ok(out)
}

/// The app state without the web side (no warm-ups, no router): the server and the Ask eval runner share it.
pub fn build_state(config: &Config) -> Result<Arc<App>, String> {
    let t_build = Instant::now();
    let root = config.root.clone();
    let scripts = root.join("third_party/CardScripts");
    let cdb = root.join("third_party/BabelCDB");
    let strings = root.join("third_party/edopro-strings.conf");
    let extra: Vec<PathBuf> = std::env::var("YGO_EXTRA_CARDS")
        .ok()
        .map(|s| s.split(':').filter(|p| !p.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default();

    let t_phase = Instant::now();
    let packs = PackStore::new(&config.packs_dir);
    if let Some(url) = &config.pack_url {
        match download_pack(url).and_then(|p| {
            let r = packs.install_zip(&p, None);
            let _ = std::fs::remove_file(&p);
            r
        }) {
            Ok(r) => {
                for a in &r.status.alerts {
                    eprintln!("ALERT server: card pack: {a}");
                }
            }
            Err(e) => {
                let shown = url.split('?').next().unwrap_or(url);
                eprintln!("ALERT server: card pack update from {shown} failed: {e}");
            }
        }
    }
    let resolved = packs.resolve();
    for a in &resolved.status.alerts {
        eprintln!("ALERT server: card pack: {a}");
    }
    let pack_dirs: Vec<PathBuf> = resolved.dir.iter().cloned().collect();
    let mut all_extra = extra.clone();
    all_extra.extend(pack_dirs.iter().cloned());
    timing("ready.packs", t_phase);

    let t_phase = Instant::now();
    let cards = decklight_cards::load_cards(&decklight_cards::default_cdbs(&cdb, &all_extra))?;
    if cards.is_empty() {
        return Err("ALERT server: no cards loaded".into());
    }
    let script_index = decklight_cards::index_scripts_layered(&scripts, &extra, &pack_dirs)?;
    if script_index.is_empty() {
        return Err("ALERT server: no scripts loaded".into());
    }
    let engine: &'static Engine = Box::leak(Box::new(decklight_cards::build_engine(&cards, &script_index)));
    timing("ready.cards_scripts_engine", t_phase);

    let t_phase = Instant::now();
    let pack_strings: Vec<PathBuf> = resolved
        .dir
        .as_ref()
        .map(|d| d.join("strings.conf"))
        .filter(|p| p.is_file())
        .into_iter()
        .collect();
    let texts = Texts::load_full(&cdb, &extra, &pack_dirs, &strings, &pack_strings)?;
    let card_types: BTreeMap<u32, u32> = cards.iter().map(|(c, d)| (*c, d.type_)).collect();
    let ctx = Ctx::new(texts, card_types, 8000);
    timing("ready.texts", t_phase);

    let t_phase = Instant::now();
    let stock_emb = root.join("data/cards/emb/qwen3-0.6b-d512.npz");
    let mut emb_path = stock_emb.clone();
    if let (Some(dir), Some(status)) = (&resolved.dir, &resolved.status.id) {
        let pack_emb = dir.join("emb.npz");
        if pack_emb.is_file() {
            let sha = resolved.status.sha256.clone().unwrap_or_default();
            let tag: String = sha.chars().take(12).collect();
            let merged = config.packs_dir.join(format!("merged-{tag}.npz"));
            let ok = if merged.is_file() && decklight_cards::emb::Emb::load(&merged).is_ok() {
                true
            } else {
                let _ = std::fs::remove_file(&merged);
                decklight_cards::emb::merge_emb(&stock_emb, &pack_emb, &merged).map(|_| ()).is_ok()
            };
            if ok {
                emb_path = merged;
                unsafe { std::env::set_var("YGO_EMB", &emb_path) };
            } else {
                eprintln!("ALERT server: card pack {status}: embeddings could not be merged");
            }
        }
    }
    // Only the vocabulary is needed for deck legality. DECKLIGHT_NO_WARM=1 (perf_check --no-warm) also takes the
    // old full-table read, so the before/after measurement compares against the real previous startup path.
    let emb_vocab = if std::env::var_os("DECKLIGHT_NO_WARM").is_some() {
        decklight_cards::emb::Emb::load(&emb_path).map(|e| e.codes().to_vec())
    } else {
        decklight_cards::emb::Emb::load_codes(&emb_path)
    };
    let mut emb_codes: BTreeSet<u32> = match emb_vocab {
        Ok(codes) => codes.into_iter().map(|c| c as u32).collect(),
        Err(e) => {
            eprintln!("ALERT server: embeddings unreadable ({e}); deck legality checks will reject every card");
            BTreeSet::new()
        }
    };
    if let Some(m) = &resolved.manifest {
        emb_codes.extend(m.codes.iter().copied());
    }
    timing("ready.embeddings", t_phase);

    let t_phase = Instant::now();
    let dirs = match &config.decks_dirs {
        Some(d) => DeckDirs::from_paths(d),
        None => DeckDirs::default_for(&root),
    };
    let mut catalog = DeckCatalog::new(
        dirs,
        &root.join("src/decklight/web/deck_coverage.json"),
        &root,
        &root.join("config/app/prebuilt_decks.txt"),
    );
    // Duels resolve `user:<id>` in the same folder the deck list shows (review 2026-11-14).
    catalog.set_user_dir(&config.user_decks_dir);
    timing("ready.deck_catalog", t_phase);

    // Art sync target: the playable vocabulary (plus the active pack), or the first N codes for an e2e run.
    let art_target: BTreeSet<u32> = match std::env::var("DECKLIGHT_ART_CODES_LIMIT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) => emb_codes.iter().copied().take(n).collect(),
        None => emb_codes.clone(),
    };
    let assets_id = std::fs::read_to_string(root.join(".assets_id"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let pack_id = resolved.status.id.clone().unwrap_or_else(|| "stock".to_string());
    let art_tokens = format!("{pack_id}|{assets_id}");
    let art_priority = deck_codes(&root, &catalog.dirs.dirs);
    let throttle = Arc::new(ImgThrottle::new(Arc::new(http_fetch)));
    let art = ArtSync::start(
        ArtSyncOptions {
            pics_dir: config.pics_dir.clone(),
            small_base: small_base(),
            target: art_target,
            priority: art_priority,
            tokens: art_tokens,
            min_free: MIN_FREE_BYTES,
        },
        throttle.clone(),
    );

    let token = if config.token.is_empty() {
        use base64::Engine as _;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(crate::record::random_hex(16).as_bytes())
    } else {
        config.token.clone()
    };
    let (model_id, model_label) = if config.onnx_dir.join("manifest.json").is_file() {
        (
            config.onnx_dir.file_name().and_then(|n| n.to_str()).unwrap_or("onnx").to_string(),
            crate::raw::manifest_label(&config.onnx_dir),
        )
    } else {
        ("base".to_string(), "Base model".to_string())
    };
    let models = if config.onnx_dir.join("manifest.json").is_file() {
        vec![(model_id.clone(), model_label.clone())]
    } else {
        Vec::new()
    };
    let records = RecordEnv::new(&root, &config.records_dir);
    records.check_index();
    let app = Arc::new(App {
        root: root.clone(),
        token,
        web_dir: config.web_dir.clone(),
        pics_dir: config.pics_dir.clone(),
        engine,
        ctx,
        cards,
        emb_codes,
        records,
        onnx_dir: config.onnx_dir.clone(),
        onnx_pool: decklight_onnx::NetsPool::new(&config.onnx_dir, &root, config.ort_threads.max(1)),
        raw: Mutex::new(RawState::Uninit),
        models,
        model_id,
        model_label,
        default_opponent: config.default_opponent.clone(),
        fixed_seed: config.fixed_seed,
        replay_threads: config.replay_threads.max(1),
        test_stream: config.test_opponent.as_deref() == Some("stream"),
        img: throttle,
        art,
        puzzles: Mutex::new(None),
        primers: Mutex::new(None),
        study_busy: AtomicBool::new(false),
        packs,
        pack_status: Mutex::new(resolved.status.clone()),
        pack_update_busy: AtomicBool::new(false),
        pack_url: config.pack_url.clone(),
        user_decks: {
            // Two ready-made decks in "Your decks" on first start, to edit instead of building from scratch (user, 2026-11-13).
            let u = UserDecks::at(&config.user_decks_dir);
            u.seed_starters(&[("Blue-Eyes", root.join("decks/pool/blue-eyes.ydk")), ("Branded", root.join("decks/pool/branded.ydk"))]);
            u
        },
        search: Mutex::new(None),
        search_conf: root.join("third_party/edopro-strings.conf"),
        ask: crate::ask::ws::Ask::new(&config.ask_dir),
        api_keys: match &config.secrets {
            Some(store) => crate::ask::keys::ApiKeys::with_store(store.clone()),
            None => crate::ask::keys::ApiKeys::default(),
        },
        describe: Arc::new(crate::describe::Describe::new(&root)),
        ondevice: crate::ask::ondevice::Runtime::new(&root),
        device_models: crate::ask::ondevice::Downloads::new(&root),
        inner: tokio::sync::Mutex::new(Inner {
            conn_id: 0,
            conn_tx: None,
            game: None,
            product: None,
            chain_mode: "auto".to_string(),
            winchance_on: false,
            games_started: 0,
            deck_catalog: catalog,
            room: None,
            hosted: None,
            room_out: Arc::new(Mutex::new(None)),
            live_decks: None,
            live_deck_names: ("Pasted deck".into(), "Pasted deck".into()),
        }),
    });
    timing("ready.app", t_build);
    Ok(app)
}

pub fn build_app(config: Config) -> Result<Router, String> {
    let app = build_state(&config)?;

    // Warm the card-search index off the startup path (it is still built lazily on first use).
    if std::env::var_os("DECKLIGHT_NO_WARM").is_none() {
        let app2 = app.clone();
        let _ = std::thread::Builder::new().name("card-search-warm".into()).spawn(move || {
            let _ = app2.search_index();
        });
    }


    // Load the product graphs off the startup path: the first product duel then finds two ready sets in the pool
    // instead of loading them inside its `new_game` handler. Two, because a duel holds one for search and one for
    // the value net. A failure only costs the warm-up (the duel loads its own set and reports the real error).
    {
        let pool = app.onnx_pool.clone();
        // DECKLIGHT_NO_WARM=1 reproduces the pre-warm-up load path (the before/after measurement).
        if std::env::var_os("DECKLIGHT_NO_WARM").is_none() {
            std::thread::Builder::new()
                .name("onnx-warm".into())
                .spawn(move || {
                    let t = Instant::now();
                    // Take both before giving any back: take/drop/take would hand the same set out twice and load
                    // only one, leaving the first duel to load its second set in the new-game handler.
                    let mut held = Vec::with_capacity(2);
                    for _ in 0..2 {
                        match pool.take() {
                            Ok(p) => held.push(p),
                            Err(e) => {
                                eprintln!("ALERT server: product model warm-up failed: {e}");
                                break;
                            }
                        }
                    }
                    drop(held);
                    timing("warm.onnx", t);
                })
                .map_err(|e| format!("could not start the product model warm-up: {e}"))?;
        }
    }

    let router = Router::new()
        .route("/", get(index_page))
        .route("/card/{code}", get(card))
        .route("/deck_index.json", get(deck_index))
        .route("/img/{code}", get(img))
        .route("/packs/status", get(packs_status))
        .route("/ws", get(ws_upgrade))
        .fallback(get(static_file))
        .with_state(app.clone());
    Ok(router)
}

pub async fn run(config: Config) -> Result<u16, String> {
    let port = config.port;
    let router = build_app(config)?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    println!("listening on {port}");
    axum::serve(listener, router).await.map_err(|e| e.to_string())?;
    Ok(port)
}

// ---------------------------------------------------------------- routes

fn token_matches(expected: &str, q: &HashMap<String, String>) -> bool {
    q.get("t").is_some_and(|t| t == expected)
}

fn need_token(app: &App, q: &HashMap<String, String>) -> bool {
    token_matches(&app.token, q)
}

/// Connection takeover: the new id replaces the old one and the old socket is told to close (4400).
fn takeover(
    conn_id: &mut u64,
    conn_tx: &mut Option<mpsc::UnboundedSender<Outgoing>>,
    my: u64,
    tx: mpsc::UnboundedSender<Outgoing>,
) -> Option<mpsc::UnboundedSender<Outgoing>> {
    *conn_id = my;
    conn_tx.replace(tx)
}

/// Prompt ids: a fresh id only when the pending core decision object changes (same message across its picks).
fn bump_prompt_id(last_seq: &mut i64, prompt_id: &mut i64, seq: i64) -> i64 {
    if seq != *last_seq {
        *last_seq = seq;
        *prompt_id += 1;
    }
    *prompt_id
}

/// The "quiet chain window" timing rule of `DuelSession.auto_pass_index`.
pub(crate) fn quiet_chain_window(forced: bool, spe_count: u8, hint_timing: u32, hint_timing_other: u32, empty_chain: bool) -> bool {
    const T_PHASES: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x1000000 | 0x4000000;
    const T_CHAIN_END: u32 = 0x8000;
    const T_DRAW: u32 = 0x10000;
    const T_TOHAND: u32 = 0x200000;
    if forced || spe_count != 0 || !empty_chain {
        return false;
    }
    let t = hint_timing | hint_timing_other;
    let allowed = T_PHASES | T_CHAIN_END | (if t & T_DRAW != 0 { T_DRAW | T_TOHAND } else { 0 });
    t != 0 && t & !allowed == 0
}

async fn index_page(State(app): State<Arc<App>>, Query(q): Query<HashMap<String, String>>) -> Response {
    if !need_token(&app, &q) {
        return text_response(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let page = app.web_dir.join("index.html");
    match std::fs::read_to_string(&page) {
        Ok(html) => {
            let body = versioned_index(&html, &app.web_dir);
            let mut r = Response::new(Body::from(body));
            r.headers_mut()
                .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            r
        }
        Err(_) => text_response(StatusCode::OK, "<!doctype html><title>decklight</title>"),
    }
}

async fn card(
    State(app): State<Arc<App>>,
    AxumPath(code): AxumPath<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !need_token(&app, &q) {
        return json_response(StatusCode::UNAUTHORIZED, json!({"error": "unauthorized"}));
    }
    let Ok(code) = code.parse::<u32>() else {
        return json_response(StatusCode::NOT_FOUND, json!({"error": "bad code"}));
    };
    json_response(StatusCode::OK, card_json(&app, code))
}

async fn deck_index(State(app): State<Arc<App>>, Query(q): Query<HashMap<String, String>>) -> Response {
    if !need_token(&app, &q) {
        return json_response(StatusCode::UNAUTHORIZED, json!({"error": "unauthorized"}));
    }
    let path = app.root.join("src/decklight/web/deck_index.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return json_response(StatusCode::NOT_FOUND, json!({"error": "no deck index"}));
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return json_response(StatusCode::NOT_FOUND, json!({"error": "no deck index"}));
    };
    let compact = serde_json::to_vec(&value).unwrap_or_default();
    use flate2::write::GzEncoder;
    use std::io::Write as _;
    let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::new(5));
    let _ = enc.write_all(&compact);
    let body = enc.finish().unwrap_or_default();
    let mut r = Response::new(Body::from(body));
    r.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r.headers_mut()
        .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    r
}

async fn packs_status(State(app): State<Arc<App>>, Query(q): Query<HashMap<String, String>>) -> Response {
    if !need_token(&app, &q) {
        return json_response(StatusCode::UNAUTHORIZED, json!({"error": "unauthorized"}));
    }
    let status = app.pack_status.lock().unwrap().clone();
    json_response(StatusCode::OK, json!({"t": "packs", "packs": status, "update_url": app.pack_url}))
}

async fn img(
    State(app): State<Arc<App>>,
    AxumPath(code): AxumPath<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !need_token(&app, &q) {
        return json_response(StatusCode::UNAUTHORIZED, json!({"error": "unauthorized"}));
    }
    let Some(code) = code.strip_suffix(".jpg").and_then(|c| c.parse::<i64>().ok()) else {
        return json_response(StatusCode::NOT_FOUND, json!({"error": "bad code"}));
    };
    let size = q.get("size").map(String::as_str).unwrap_or("small");
    if size != "small" && size != "full" {
        return json_response(StatusCode::BAD_REQUEST, json!({"error": "size must be small|full"}));
    }
    if code == 0 {
        return svg_response();
    }
    let cached = app.pics_dir.join(size).join(format!("{code}.jpg"));
    if let Ok(data) = std::fs::read(&cached) {
        if valid_jpeg(&data) {
            return cached_image_response(data);
        }
        // A truncated/HTML body from an older fetch: drop it and fetch again below.
        let _ = std::fs::remove_file(&cached);
        app.art.file_removed(code as u32);
    }
    if let Ok(raw) = std::env::var("YGO_EXTRA_CARDS") {
        for d in raw.split(':').filter(|p| !p.is_empty()) {
            let cand = PathBuf::from(d).join("pics").join(format!("{code}.jpg"));
            if let Ok(data) = std::fs::read(&cand) {
                if valid_jpeg(&data) {
                    return cached_image_response(data);
                }
            }
        }
    }
    // The bulk sync already learned that YGOPRODeck has no small image: don't ask again for seven days.
    if size == "small" && app.art.is_missing(code as u32) {
        return svg_response();
    }
    let url = if size == "full" { IMG_FULL } else { IMG_SMALL }.replace("{code}", &code.to_string());
    let url = url.replace("https://images.ygoprodeck.com", &image_host());
    let outcome = app.img.fetch_art(&url).await;
    match outcome.data {
        Some(data) => {
            if let Some(p) = cached.parent() {
                let _ = std::fs::create_dir_all(p);
            }
            let _ = std::fs::write(&cached, &data);
            if size == "small" {
                app.art.recovered(code as u32, data.len() as u64);
            }
            cached_image_response(data)
        }
        None => {
            if size == "small" && outcome.not_found {
                app.art.record_missing(code as u32);
            }
            svg_response()
        }
    }
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(app): State<Arc<App>>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !need_token(&app, &q) {
        // uvicorn/Starlette reject a close-before-accept as an HTTP 403; match that (the Python oracle).
        return StatusCode::FORBIDDEN.into_response();
    }
    ws.on_upgrade(move |socket| handle_socket(socket, app, true))
}

async fn handle_socket(mut socket: WebSocket, app: Arc<App>, token_ok: bool) {
    if !token_ok {
        let _ = socket
            .send(Message::Close(Some(CloseFrame { code: 4401, reason: "unauthorized".into() })))
            .await;
        return;
    }
    static CONN: AtomicU64 = AtomicU64::new(0);
    let my = CONN.fetch_add(1, Ordering::SeqCst) + 1;
    let (tx, mut rx) = mpsc::unbounded_channel::<Outgoing>();
    let (art_tx, mut art_rx) = mpsc::unbounded_channel::<Value>();
    app.art.set_sink(Some(art_tx));
    {
        let mut inner = app.inner.lock().await;
        let Inner { conn_id, conn_tx, .. } = &mut *inner;
        if let Some(old) = takeover(conn_id, conn_tx, my, tx.clone()) {
            let _ = old.send(Outgoing::Close(4400));
        }
        *inner.room_out.lock().unwrap() = Some(tx.clone());
        // A running Ask answer streams to the current connection.
        let ask_tx = tx.clone();
        app.ask.set_out(Some(Box::new(move |f| {
            let _ = ask_tx.send(Outgoing::Frame(f));
        })));
        // A product duel outlives its socket: its frames (and a running review's progress) follow the new one,
        // starting with its current state.
        if let Some(p) = inner.product.as_ref() {
            p.sink.set(Some(tx.clone()));
            let _ = p.cmd_tx.send(ProductCmd::Resync);
        }
        inner.winchance_on = false;
        let lobby = inner.lobby(&app);
        let mut frames = vec![lobby];
        if let Some(mut g) = inner.game.take() {
            if !(g.env.done() && g.finished) {
                let mut st = Vec::new();
                g.send_state(&app, true, false, inner.winchance_on, &mut st);
                frames.extend(st);
            }
            inner.game = Some(g);
        }
        if let Some(r) = inner.room.as_ref() {
            let _ = r.send(RoomCmd::Resync);
        }
        for f in frames {
            let text = serde_json::to_string(&f).unwrap_or_default();
            if socket.send(Message::Text(text.into())).await.is_err() {
                return;
            }
        }
    }
    loop {
        tokio::select! {
            out = rx.recv() => {
                match out {
                    Some(Outgoing::Close(code)) => {
                        let _ = socket.send(Message::Close(Some(CloseFrame { code, reason: "".into() }))).await;
                        break;
                    }
                    Some(Outgoing::Frame(v)) => {
                        let s = serde_json::to_string(&v).unwrap_or_default();
                        if socket.send(Message::Text(s.into())).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                }
            }
            art = art_rx.recv() => {
                let Some(v) = art else { break };
                let s = serde_json::to_string(&v).unwrap_or_default();
                if socket.send(Message::Text(s.into())).await.is_err() {
                    break;
                }
            }
            msg = socket.recv() => {
                let Some(Ok(msg)) = msg else { break };
                let text = match msg {
                    Message::Text(t) => t.to_string(),
                    Message::Binary(b) => match String::from_utf8(b.to_vec()) { Ok(s) => s, Err(_) => { let f = Inner::error("not JSON"); for v in f { let _ = socket.send(Message::Text(serde_json::to_string(&v).unwrap_or_default().into())).await; } continue; } },
                    Message::Close(_) => break,
                    _ => continue,
                };
                let frames = {
                    let mut inner = app.inner.lock().await;
                    if inner.conn_id != my {
                        break;
                    }
                    inner.handle(&app, &text)
                };
                for v in frames {
                    let s = serde_json::to_string(&v).unwrap_or_default();
                    if socket.send(Message::Text(s.into())).await.is_err() {
                        break;
                    }
                }
            }
        }
    }
    let mut inner = app.inner.lock().await;
    if inner.conn_id == my {
        app.art.set_sink(None);
        inner.conn_tx = None;
        // A room is a person on the other side: it stays open while the page reconnects.
        *inner.room_out.lock().unwrap() = None;
        if let Some(p) = inner.product.as_ref() {
            p.sink.set(None); // nothing may be delivered after the socket is gone
        }
        // The game, finished or not, waits for the page to come back: a reload, or the app sent to the
        // background, drops the socket mid-duel, and recording that as a loss (as the Python server did) ended
        // duels and puzzles the player never left. The next socket takes the game over; a new game replaces it.
    }
}

// ---------------------------------------------------------------- static + json helpers

async fn static_file(State(app): State<Arc<App>>, uri: Uri) -> Response {
    let rel = uri.path().trim_start_matches('/');
    if rel.split('/').any(|p| p == "..") {
        return text_response(StatusCode::NOT_FOUND, "not found");
    }
    let full = app.web_dir.join(rel);
    match std::fs::read(&full) {
        Ok(data) => {
            let mut r = Response::new(Body::from(data));
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_str(content_type(&full)).unwrap_or(HeaderValue::from_static("application/octet-stream")),
            );
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            r
        }
        Err(_) => text_response(StatusCode::NOT_FOUND, "not found"),
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/vnd.microsoft.icon",
        "woff2" => "font/woff2",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

fn text_response(status: StatusCode, text: &str) -> Response {
    let mut r = Response::new(Body::from(text.to_string()));
    *r.status_mut() = status;
    r.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    r
}

fn json_response(status: StatusCode, value: Value) -> Response {
    let body = serde_json::to_vec(&value).unwrap_or_default();
    let mut r = Response::new(Body::from(body));
    *r.status_mut() = status;
    r.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}

fn binary_response(ct: &'static str, data: Vec<u8>) -> Response {
    let mut r = Response::new(Body::from(data));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(ct));
    r
}

/// Saved card art is immutable for the life of the install: the WebView keeps it (the URL token changes per launch).
fn cached_image_response(data: Vec<u8>) -> Response {
    let mut r = binary_response("image/jpeg", data);
    r.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, max-age=31536000, immutable"),
    );
    r
}

fn svg_response() -> Response {
    binary_response("image/svg+xml", CARD_BACK_SVG.as_bytes().to_vec())
}

fn card_json(app: &App, code: u32) -> Value {
    let names = app.ctx.texts.names();
    let text = app.ctx.texts.effect_text(code);
    let cd = app.cards.get(&code);
    let st = app.ctx.texts.stats_of(code).copied().unwrap_or((0, 0, 0, 0, 0, 0));
    let (mut atk, mut dfn, level, rank, link, type_) = st;
    if let Some(cd) = cd {
        atk = cd.attack;
        dfn = cd.defense;
    }
    json!({
        "code": code,
        "name": names.get(&code).cloned().unwrap_or_else(|| format!("#{code}")),
        "type": type_,
        "type_str": type_str(type_),
        "attribute": cd.map(|c| c.attribute).unwrap_or(0),
        "race": cd.map(|c| c.race).unwrap_or(0),
        "level": level,
        "rank": rank,
        "link": link,
        "atk": atk,
        "def": dfn,
        "text": text,
    })
}

fn type_str(type_: u32) -> String {
    if type_ & C::TYPE_MONSTER != 0 {
        let mut kinds = Vec::new();
        for (bit, name) in [
            (C::TYPE_FUSION, "Fusion"),
            (C::TYPE_SYNCHRO, "Synchro"),
            (C::TYPE_XYZ, "Xyz"),
            (C::TYPE_LINK, "Link"),
            (C::TYPE_RITUAL, "Ritual"),
            (C::TYPE_PENDULUM, "Pendulum"),
            (C::TYPE_TUNER, "Tuner"),
            (C::TYPE_EFFECT, "Effect"),
            (C::TYPE_NORMAL, "Normal"),
        ] {
            if type_ & bit != 0 {
                kinds.push(name);
            }
        }
        return if kinds.is_empty() { "Monster".into() } else { format!("{} Monster", kinds.join(" ")) };
    }
    if type_ & C::TYPE_SPELL != 0 {
        return "Spell".into();
    }
    if type_ & C::TYPE_TRAP != 0 {
        return "Trap".into();
    }
    format!("type {type_:#x}")
}

/// `versioned_index`: cache-busting `?v=` for the client modules and stylesheet.
pub fn versioned_index(html: &str, web_dir: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut mods: Vec<PathBuf> = match std::fs::read_dir(web_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("js"))
            .collect(),
        Err(_) => Vec::new(),
    };
    if let Ok(rd) = std::fs::read_dir(web_dir.join("render")) {
        mods.extend(
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("js")),
        );
    }
    mods.sort();
    let mut h = Sha256::new();
    let mut files = mods.clone();
    files.push(web_dir.join("styles.css"));
    for f in &files {
        if let Ok(data) = std::fs::read(f) {
            h.update(f.file_name().and_then(|n| n.to_str()).unwrap_or("").as_bytes());
            h.update([0u8]);
            h.update(data);
        }
    }
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let v = &hex[..12];
    let imports: serde_json::Map<String, Value> = mods
        .iter()
        .filter_map(|f| f.strip_prefix(web_dir).ok())
        .map(|rel| {
            let key = format!("/{}", rel.to_string_lossy());
            (key.clone(), json!(format!("{key}?v={v}")))
        })
        .collect();
    let mut out = html.replace("href=\"styles.css\"", &format!("href=\"styles.css?v={v}\""));
    let imap = json!({"imports": imports});
    out = out.replace(
        "<script type=\"module\" src=\"app.js\"></script>",
        &format!(
            "<script type=\"importmap\">{}</script>\n<script type=\"module\" src=\"/app.js?v={v}\"></script>",
            imap
        ),
    );
    out
}

#[allow(dead_code)]
fn deck_name(stem: &str) -> String {
    pretty_deck_name(stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn ticker_sends_rising_seconds_and_stops_on_drop() {
        let (tx, mut rx) = mpsc::unbounded_channel::<Outgoing>();
        let sink = Sink::default();
        sink.set(Some(tx));
        let peek = json!({"t": "state", "thinking": null, "seq": 7});
        let ticker = Ticker::start(peek, sink);
        std::thread::sleep(Duration::from_millis(2500));
        drop(ticker);
        let mut secs = Vec::new();
        while let Ok(Outgoing::Frame(f)) = rx.try_recv() {
            assert_eq!(f["seq"], 7);
            secs.push(f["thinking"]["seconds"].as_f64().unwrap());
        }
        assert!(secs.len() >= 2, "ticks: {secs:?}");
        assert!(secs.windows(2).all(|w| w[1] > w[0]), "ticks: {secs:?}");
        assert!(secs[0] >= 1.0);
        std::thread::sleep(Duration::from_millis(1200));
        assert!(rx.try_recv().is_err(), "a tick arrived after the ticker was dropped");
    }

    #[test]
    fn ticker_dropped_early_sends_nothing() {
        let (tx, mut rx) = mpsc::unbounded_channel::<Outgoing>();
        let sink = Sink::default();
        sink.set(Some(tx));
        let t0 = Instant::now();
        drop(Ticker::start(json!({"thinking": null}), sink));
        assert!(t0.elapsed() < Duration::from_millis(500), "drop must not wait for a tick");
        std::thread::sleep(Duration::from_millis(1200));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn lobby_decks_puts_saved_first() {
        let saved = vec![
            json!({"id": "user:new", "group": "saved"}),
            json!({"id": "user:old", "group": "saved"}),
        ];
        let catalog = vec![json!({"id": "meta", "group": "prebuilt"}), json!({"id": "tuned-1", "group": "tuned"})];
        let rows = merge_lobby_decks(saved, catalog);
        let ids: Vec<&str> = rows.iter().map(|r| r["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["user:new", "user:old", "meta", "tuned-1"]);
        assert_eq!(rows[0]["group"], "saved");
        assert_eq!(rows[2]["group"], "prebuilt");
    }

    #[test]
    fn token_rule() {
        let mut query = q(&[("t", "secret")]);
        assert!(token_matches("secret", &query));
        assert!(!token_matches("other", &query));
        query.insert("t".into(), "secretx".into());
        assert!(!token_matches("secret", &query));
        assert!(!token_matches("secret", &q(&[])));
    }

    #[test]
    fn takeover_closes_the_old_connection() {
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let (tx2, _rx2) = mpsc::unbounded_channel();
        let mut id = 0;
        let mut cur = None;
        assert!(takeover(&mut id, &mut cur, 1, tx1).is_none());
        let old = takeover(&mut id, &mut cur, 2, tx2).unwrap();
        assert_eq!(id, 2);
        assert!(matches!(old.send(Outgoing::Close(4400)), Ok(())));
        assert!(matches!(rx1.try_recv(), Ok(Outgoing::Close(4400))));
    }

    #[test]
    fn a_product_sink_follows_the_new_connection() {
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let (tx2, mut rx2) = mpsc::unbounded_channel();
        let sink = Sink::default();
        sink.set(Some(tx1));
        sink.send(Outgoing::Frame(json!({"t": "state", "n": 1})));
        sink.set(Some(tx2)); // takeover
        sink.send(Outgoing::Frame(json!({"t": "state", "n": 2})));
        assert!(matches!(rx1.try_recv(), Ok(Outgoing::Frame(v)) if v["n"] == 1));
        assert!(matches!(rx2.try_recv(), Ok(Outgoing::Frame(v)) if v["n"] == 2));
        assert!(rx1.try_recv().is_err(), "the old connection must not receive more frames");
    }

    #[test]
    fn prompt_ids_only_bump_on_a_new_decision() {
        let (mut last, mut id) = (-1, 0);
        assert_eq!(bump_prompt_id(&mut last, &mut id, 5), 1);
        assert_eq!(bump_prompt_id(&mut last, &mut id, 5), 1);
        assert_eq!(bump_prompt_id(&mut last, &mut id, 9), 2);
        assert_eq!(bump_prompt_id(&mut last, &mut id, 12), 3);
    }

    #[test]
    fn auto_pass_timing_rule() {
        assert!(quiet_chain_window(false, 0, 0x1, 0, true));
        assert!(quiet_chain_window(false, 0, 0x8000, 0, true));
        assert!(quiet_chain_window(false, 0, 0x0, 0x10000, true));
        assert!(!quiet_chain_window(false, 0, 0x0, 0x200000, true), "to-hand without draw");
        assert!(!quiet_chain_window(true, 0, 0x1, 0, true), "forced");
        assert!(!quiet_chain_window(false, 1, 0x1, 0, true), "special summon window");
        assert!(!quiet_chain_window(false, 0, 0x1, 0, false), "a chain link is on the field");
        assert!(!quiet_chain_window(false, 0, 0x0, 0, true), "no timing");
        assert!(!quiet_chain_window(false, 0, 0x40, 0, true), "event timing keeps asking");
    }
}
