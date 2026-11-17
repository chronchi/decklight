// decklight Android app (A6): the unchanged web client (`web/`) served by `decklight-server` inside the app process
// on 127.0.0.1, in a Tauri 2 webview (S3 spike: rust/spikes/tauri-shell).
//
// First run unpacks the embedded per-unit data archives (cards, scripts, networks, embeddings, decks, web) into the
// app's data dir. Each unit has its own content id, so an app update only re-unpacks what changed. As soon as the
// `web` unit is in place the gate serves the client itself (the lobby shows a quiet "Loading the bot…" line) while the
// remaining units unpack and the server loads; the websocket and API routes wait for Ready. Before `web` exists (a
// first install) every request gets a startup page that polls /__ready, then reloads into the client. The server's
// stdout and stderr go to `<root>/logs/server.log` (Android drops them otherwise).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Router;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tower::ServiceExt;

mod assets;
use assets::{safe_rel, stale_units};

include!(concat!(env!("OUT_DIR"), "/assets_table.rs"));

const ASSETS_ID: &str = env!("DECKLIGHT_ASSETS_ID");
/// Trees the old single archive owned (before per-unit unpacking), removed once when an existing install upgrades.
/// Everything else under the root is the user's (records, packs, card images, rooms, settings) and survives updates.
const LEGACY_TREES: [&str; 7] = ["web", "third_party", "data/cache/onnx", "data/cards/emb", "decks", "src", "data/primers"];
const LOG_MAX_BYTES: u64 = 4 << 20;
// See the listener in run(): a fixed origin keeps the web view's saved settings. Below Linux's ephemeral range
// (32768-60999), so no outgoing socket is handed this port.
const CLIENT_PORT: u16 = 29431;

#[derive(Clone)]
enum Phase {
    Starting { msg: String, web: Option<PathBuf> },
    Ready(Router),
    Failed { msg: String, web: Option<PathBuf> },
}

#[derive(Clone)]
struct Gate(Arc<Mutex<Phase>>);

impl Gate {
    fn set(&self, p: Phase) {
        *self.0.lock().unwrap() = p;
    }

    fn web(&self) -> Option<PathBuf> {
        match &*self.0.lock().unwrap() {
            Phase::Starting { web, .. } | Phase::Failed { web, .. } => web.clone(),
            Phase::Ready(_) => None,
        }
    }

    /// A new startup message, keeping the client available once its unit is in place.
    fn starting(&self, msg: impl Into<String>) {
        let web = self.web();
        self.set(Phase::Starting { msg: msg.into(), web });
    }

    /// The `web` unit is in place: serve the client from now on, even while the rest loads.
    fn client_ready(&self, web: PathBuf) {
        self.set(Phase::Starting { msg: "Loading the bot…".into(), web: Some(web) });
    }

    /// Keep the client (if it was already available) on a failed start, so a reload shows the failure in the client.
    fn failed(&self, msg: String) {
        let web = self.web();
        self.set(Phase::Failed { msg, web });
    }
}

async fn gate_handler(State(g): State<Gate>, req: Request) -> Response {
    let phase = g.0.lock().unwrap().clone();
    // The startup page and the client poll this instead of reloading themselves (a reload a second flickered).
    if req.uri().path() == "/__ready" {
        let (ready, failed, client, msg) = match &phase {
            Phase::Ready(_) => (true, false, true, String::new()),
            Phase::Starting { msg, web } => (false, false, web.is_some(), msg.clone()),
            Phase::Failed { msg, web } => (false, true, web.is_some(), msg.clone()),
        };
        return axum::Json(serde_json::json!({
            "ready": ready, "failed": failed, "client": client, "msg": msg
        }))
        .into_response();
    }
    match phase {
        Phase::Ready(router) => router.oneshot(req).await.into_response(),
        Phase::Starting { msg, web } => phase_response(req, web, false, &msg),
        Phase::Failed { msg, web } => phase_response(req, web, true, &msg),
    }
}

/// While the server is not ready: serve the client's static files if the web unit exists, otherwise the startup page.
/// Everything that is not a client file (the websocket and the API routes) waits for Ready.
fn phase_response(req: Request, web: Option<PathBuf>, failed: bool, msg: &str) -> Response {
    if let Some(dir) = web {
        if let Some(r) = static_file(&dir, req.uri().path()) {
            return r;
        }
        if !failed {
            let mut r = (StatusCode::SERVICE_UNAVAILABLE, "starting").into_response();
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            return r;
        }
    }
    waiting_page(failed, msg)
}

/// The client while the server starts (the same files decklight-server serves once Ready).
fn static_file(dir: &Path, path: &str) -> Option<Response> {
    let rel = path.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    if !safe_rel(rel) {
        return None;
    }
    let full = dir.join(rel);
    let data = fs::read(&full).ok()?;
    let mut r = Response::new(Body::from(data));
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(content_type(&full)).unwrap_or(HeaderValue::from_static("application/octet-stream")),
    );
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    Some(r)
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

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// The splash mark, framed and weighted like the Android splash (scripts/app/android_icons.sh), so the system splash,
/// this page and the client's first paint (web/index.html #boot) all show it at the same place and size.
const BOOT_MARK: &str = include_str!("../../../../web/assets/boot-mark.svg");

// The theme chosen in Settings → Appearance (web/app.js applyTheme; MainActivity keeps it): the light colours of
// web/styles.css :root[data-theme="light"], so a light choice never starts on the night page. Same lookup as
// web/index.html: this page is served on the client's origin (CLIENT_PORT), so the saved choice is readable here, and
// desktops (no bridge) take System from the OS.
const THEME_JS: &str = r##"<script>
try{var b=window.DecklightTheme,m=null;try{m=localStorage.getItem("ygo-theme");}catch(e){}m=m||(b&&b.mode());
var s=b?b.system():(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark");
if((m==="system"?s:m)==="light"){document.documentElement.className="light";document.documentElement.style.background="#f5f1e8";}}catch(e){}
</script>"##;

// While starting: wait for the server, then load the client (a failed start reloads once, into the failure page).
const POLL_JS: &str = r#"<script>
(function poll(){
  fetch("/__ready",{cache:"no-store"}).then(function(r){return r.json();}).then(function(j){
    if(j.ready||j.failed||j.client){location.reload();return;}
    var m=document.getElementById("bootmsg");
    if(m&&j.msg)m.textContent=j.msg;
    setTimeout(poll,200);
  }).catch(function(){setTimeout(poll,500);});
})();
</script>"#;

/// The startup page: the mark and the wordmark as in web/index.html #boot (same CSS geometry), then, only when the
/// start takes longer than a moment, a hairline and what is being prepared. It polls /__ready and loads the client
/// when the server is up.
fn waiting_page(failed: bool, msg: &str) -> Response {
    let title = if failed { "decklight could not start" } else { "decklight" };
    let status = if failed {
        format!(r#"<p class="bootfail" role="alert"><strong>decklight could not start.</strong><br>{}</p>"#, escape(msg))
    } else {
        format!(r#"<div class="bootline" aria-hidden="true"></div><p class="bootmsg" id="bootmsg">{}</p>"#, escape(msg))
    };
    Html(format!(
        r#"<!doctype html><html lang="en" style="background:#141310"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark"><title>{title}</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#141310;color:#d6cfc0;font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
#boot{{position:fixed;inset:0}}
.bootmark{{position:absolute;left:50%;top:50%;width:288px;height:288px;transform:translate(-50%,-50%)}}
.bootword{{position:absolute;left:0;right:0;top:calc(50% + 66px);text-align:center;font-size:22px;font-weight:300;
letter-spacing:.08em}}
.bootline{{position:absolute;left:50%;top:calc(50% + 116px);width:96px;height:1px;margin-left:-48px;overflow:hidden;
background:rgba(214,207,192,.13);opacity:0;animation:show .4s ease 1s forwards}}
.bootline::after{{content:"";position:absolute;inset:0;width:32px;background:#b89a5e;animation:slide 1.6s ease-in-out infinite}}
.bootmsg,.bootfail{{position:absolute;left:16px;right:16px;top:calc(50% + 132px);margin:0;text-align:center;font-size:13px;
color:#958f82;opacity:0;animation:show .4s ease 1s forwards}}
.bootfail{{opacity:1;animation:none;color:#d6cfc0}}
@keyframes show{{to{{opacity:1}}}}
@keyframes slide{{from{{transform:translateX(-32px)}}to{{transform:translateX(96px)}}}}
@media (prefers-reduced-motion:reduce){{.bootline::after{{animation:none;width:100%;opacity:.5}}}}
html.light,html.light body{{background:#f5f1e8;color:#2a2721;color-scheme:light}}
html.light .bootmsg{{color:#666054}} html.light .bootfail{{color:#2a2721}}
html.light .bootline{{background:rgba(42,39,33,.16)}} html.light .bootline::after{{background:#745a26}}
html.light .bootmark :is(.e1,.e2,.e3,.nd){{fill:#faf7f0}} html.light .bootmark .e1{{stroke:#a9b8cd}}
html.light .bootmark .e2{{stroke:#7d8ea8}} html.light .bootmark .e3{{stroke:#46628c}}
html.light .bootmark :is(.pth,.nd){{stroke:#8a6a2c}} html.light .bootmark .st{{fill:#8a6a2c}}
</style>{THEME_JS}</head><body><div id="boot" role="status" aria-live="polite">{BOOT_MARK}
<div class="bootword">decklight</div>{status}</div>
{script}</body></html>"#,
        script = if failed { "" } else { POLL_JS },
    ))
    .into_response()
}

/// The directory this library was loaded from: the APK's native libs, where `libonnxruntime.so` sits too.
fn own_lib_dir() -> Option<PathBuf> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let f = own_lib_dir as *const libc::c_void;
    if unsafe { libc::dladdr(f, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy().into_owned();
    Path::new(&name).parent().map(Path::to_path_buf)
}

/// Send this process's stdout/stderr to the log file (the server reports ALERTs there).
fn redirect_output(root: &Path) {
    let dir = root.join("logs");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("server.log");
    if fs::metadata(&path).map(|m| m.len() > LOG_MAX_BYTES).unwrap_or(false) {
        let _ = fs::rename(&path, dir.join("server.log.1"));
    }
    use std::os::fd::AsRawFd;
    if let Ok(f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
        unsafe {
            libc::dup2(f.as_raw_fd(), 1);
            libc::dup2(f.as_raw_fd(), 2);
        }
    }
}

/// Rename an old tree/file out of the way (one rename, instant) instead of deleting it in the startup path.
/// The `.trash` directory is removed on a background thread once the server is Ready (`cleanup_trash`).
fn move_to_trash(root: &Path, p: &Path) -> Result<(), String> {
    match fs::symlink_metadata(p) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("{}: {e}", p.display())),
    }
    let trash = root.join(".trash");
    fs::create_dir_all(&trash).map_err(|e| format!("{}: {e}", trash.display()))?;
    let rel = p.strip_prefix(root).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut dst = trash.join(rel);
    let mut n = 0;
    while fs::symlink_metadata(&dst).is_ok() {
        n += 1;
        let mut name = rel.as_os_str().to_os_string();
        name.push(format!(".{n}"));
        dst = trash.join(name);
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fs::rename(p, &dst).map_err(|e| format!("{} -> {}: {e}", p.display(), dst.display()))
}

/// Remove `<root>/.trash` after the server is Ready (never in the critical startup path).
fn cleanup_trash(root: &Path) {
    let trash = root.join(".trash");
    if fs::symlink_metadata(&trash).is_err() {
        return;
    }
    let t0 = Instant::now();
    match fs::remove_dir_all(&trash) {
        Ok(()) => println!("app: trash removed in {:.1}s", t0.elapsed().as_secs_f64()),
        Err(e) => eprintln!("ALERT app: trash cleanup: {e}"),
    }
}

fn unpack_if_needed(root: &Path, gate: &Gate) -> Result<(), String> {
    let marker_dir = root.join(".assets");
    // The old single archive left only `.assets_id` and no per-unit markers: make every unit stale once.
    let migrated = root.join(".assets_id").is_file() && !marker_dir.is_dir();
    if migrated {
        for t in LEGACY_TREES {
            move_to_trash(root, &root.join(t))?;
        }
    }
    fs::create_dir_all(&marker_dir).map_err(|e| format!("{}: {e}", marker_dir.display()))?;
    let stale = stale_units(UNITS, |name| fs::read_to_string(marker_dir.join(format!("{name}.id"))).ok(), migrated);
    let skipped = UNITS.len().saturating_sub(stale.len());
    // An unchanged web unit must still unlock the client (only stale units go through the loop).
    if !stale.iter().any(|u| u.name == "web") && root.join("web").is_dir() {
        gate.client_ready(root.join("web"));
    }
    for unit in stale {
        let t0 = Instant::now();
        gate.starting(format!("Unpacking {}…", unit.name));
        for rel in unit.remove {
            move_to_trash(root, &root.join(rel))?;
        }
        let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(unit.archive));
        let mut n = 0usize;
        for entry in ar.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            if !entry.unpack_in(root).map_err(|e| e.to_string())? {
                return Err(format!("archive entry outside the data dir: {:?}", entry.path()));
            }
            n += 1;
            if n % 2000 == 0 {
                gate.starting(format!("Unpacking {}… {n} files", unit.name));
            }
        }
        fs::write(marker_dir.join(format!("{}.id", unit.name)), unit.id).map_err(|e| e.to_string())?;
        println!("app: unpacked {} {n} entries in {:.1}s", unit.name, t0.elapsed().as_secs_f64());
        if unit.name == "web" {
            gate.client_ready(root.join("web"));
        }
    }
    // The archive's own list of what must be there (fail loud instead of a half-working app).
    let mut need = String::new();
    fs::File::open(root.join("ASSETS.txt"))
        .and_then(|mut f| f.read_to_string(&mut need))
        .map_err(|e| format!("ASSETS.txt: {e}"))?;
    for line in need.lines().filter(|l| !l.trim().is_empty()) {
        if !root.join(line.trim()).exists() {
            return Err(format!("unpacked data is missing {line}"));
        }
    }
    // Kept for decklight-server's art-sync token (it reads `.assets_id`; the app itself uses `.assets/<unit>.id`).
    fs::write(root.join(".assets_id"), ASSETS_ID).map_err(|e| e.to_string())?;
    println!("app: assets up to date ({skipped} units skipped)");
    Ok(())
}

/// `<root>/settings.json`: `{"ort_threads": N, "replay_threads": N, "search_threads": N}` (A6 speed settings;
/// defaults below: replay 4 and search 4 = the same moves 1.6x faster on 4 laptop cores, NOTEBOOK 2026-11-12).
fn settings(root: &Path) -> (usize, usize, usize) {
    let v: serde_json::Value = fs::read_to_string(root.join("settings.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let get = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).map(|x| x.clamp(1, 8) as usize).unwrap_or(d);
    (get("ort_threads", 4), get("replay_threads", 4), get("search_threads", 4))
}

fn prepare(root: &Path, token: &str, gate: &Gate) -> Result<Router, String> {
    unpack_if_needed(root, gate)?;
    gate.starting("Loading the duel engine and the bot…");
    let t0 = Instant::now();
    let (ort_threads, replay_threads, search_threads) = settings(root);
    // The product worker reads this (product.rs); the server's Config is untouched by the rollout-thread setting.
    std::env::set_var("DECKLIGHT_SEARCH_THREADS", search_threads.to_string());
    let mut c = decklight_server::Config::defaults(root);
    c.token = token.to_string();
    c.ort_threads = ort_threads;
    c.replay_threads = replay_threads;
    let router = decklight_server::build_app(c)?;
    println!(
        "app: server ready in {:.1}s (ort_threads {ort_threads}, replay_threads {replay_threads}, \
         search_threads {search_threads})",
        t0.elapsed().as_secs_f64()
    );
    Ok(router)
}

fn serve(root: PathBuf, token: String, gate: Gate, port_tx: std::sync::mpsc::Sender<Result<u16, String>>) {
    let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            let _ = port_tx.send(Err(e.to_string()));
            return;
        }
    };
    rt.block_on(async move {
        // A fixed port: the web view keeps the client's settings (theme, chain mode, duelist name, solved puzzles)
        // per origin, so a new random port each launch reset them all (2026-11-15). Taken by another app: a free
        // port for this launch, and the settings start from their defaults until the next one.
        let fixed = tokio::net::TcpListener::bind(("127.0.0.1", CLIENT_PORT)).await;
        if let Err(e) = &fixed {
            log::warn!("decklight: port {CLIENT_PORT} is taken ({e}); serving on a free port, so saved settings are not read this launch");
        }
        let bound = match fixed {
            Ok(l) => Ok(l),
            Err(_) => tokio::net::TcpListener::bind(("127.0.0.1", 0)).await,
        };
        let listener = match bound {
            Ok(l) => l,
            Err(e) => {
                let _ = port_tx.send(Err(format!("bind 127.0.0.1: {e}")));
                return;
            }
        };
        let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
        let _ = port_tx.send(Ok(port));
        let g = gate.clone();
        tokio::task::spawn_blocking(move || match prepare(&root, &token, &g) {
            Ok(r) => {
                g.set(Phase::Ready(r));
                cleanup_trash(&root);
            }
            Err(e) => {
                eprintln!("ALERT app: {e}");
                log::error!("decklight: {e}");
                g.failed(e);
            }
        });
        let app = Router::new().fallback(gate_handler).with_state(gate);
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("ALERT app: server stopped: {e}");
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = android_logger::init_once(android_logger::Config::default().with_max_level(log::LevelFilter::Info));
    // ort loads ONNX Runtime from the APK's native libs (jniLibs/arm64-v8a/libonnxruntime.so).
    let ort = own_lib_dir().map(|d| d.join("libonnxruntime.so")).filter(|p| p.is_file());
    std::env::set_var("ORT_DYLIB_PATH", ort.as_deref().unwrap_or(Path::new("libonnxruntime.so")));
    let token = {
        use base64::Engine as _;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(rand::random::<[u8; 12]>())
    };
    tauri::Builder::default()
        .setup(move |app| {
            let root = app.path().app_data_dir()?.join("root");
            fs::create_dir_all(&root)?;
            redirect_output(&root);
            println!("app: start, root {}, ORT {:?}", root.display(), std::env::var("ORT_DYLIB_PATH").ok());
            let gate = Gate(Arc::new(Mutex::new(Phase::Starting { msg: "Starting…".into(), web: None })));
            let (tx, rx) = std::sync::mpsc::channel();
            let token_run = token.clone();
            let root_run = root.clone();
            std::thread::Builder::new()
                .name("decklight-server".into())
                .spawn(move || serve(root_run, token_run, gate, tx))?;
            let port = rx
                .recv_timeout(Duration::from_secs(20))
                .map_err(|_| "the local server did not start in 20 s".to_string())??;
            log::info!("decklight: serving on 127.0.0.1:{port}");
            // For device tests over adb (app-private storage: only run-as on a debug build can read it).
            let _ = fs::write(root.join("logs/run.json"), serde_json::json!({"port": port, "token": token}).to_string());
            // Built here with the final URL: navigating a placeholder window later races the webview (S3).
            let url = format!("http://127.0.0.1:{port}/?t={token}");
            WebviewWindowBuilder::new(app.handle(), "main", WebviewUrl::External(url.parse()?))
                .title("decklight")
                // The night colour from the first frame, never a white flash before the page paints.
                .background_color(tauri::webview::Color(20, 19, 16, 255))
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running the decklight app");
}
