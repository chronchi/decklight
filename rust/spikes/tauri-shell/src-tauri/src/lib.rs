// S3 spike: Tauri 2 shell that serves the unchanged web client from an
// in-app HTTP + WebSocket server on 127.0.0.1. See ../README.md.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{
    Router,
    extract::{
        State,
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade},
    },
    http::{Uri, header},
    response::{IntoResponse, Response},
    routing::get,
};
use rust_embed::RustEmbed;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

#[derive(RustEmbed)]
#[folder = "../../../../web"]
struct WebAsset;

#[derive(Clone)]
struct Shared {
    token: Arc<String>,
    lobby: Arc<String>,
}

fn content_type(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".js") {
        // The client uses ES modules: must be JavaScript, not e.g. text/plain.
        "text/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".json") {
        "application/json"
    } else if path.ends_with(".webmanifest") {
        "application/manifest+json"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        "image/jpeg"
    } else {
        "application/octet-stream"
    }
}

async fn static_files(State(st): State<Shared>, uri: Uri) -> Response {
    let _ = st;
    let mut path = uri.path().trim_start_matches('/').to_string();
    if path.is_empty() {
        path = "index.html".to_string();
    }
    match WebAsset::get(&path) {
        Some(asset) => {
            let body = asset.data.into_owned();
            (
                [(header::CONTENT_TYPE, content_type(&path))],
                body,
            )
                .into_response()
        }
        None => (axum::http::StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
    State(st): State<Shared>,
) -> Response {
    if q.get("t").map(|s| s.as_str()) != Some(st.token.as_str()) {
        log::warn!("S3 ws: rejected connection with wrong/missing token (4401)");
        return ws.on_upgrade(|mut socket| async move {
            let _ = socket
                .send(Message::Close(Some(CloseFrame {
                    code: 4401u16,
                    reason: "bad token".into(),
                })))
                .await;
        });
    }
    let st2 = st.clone();
    ws.on_upgrade(move |socket| handle_socket(socket, st2))
}

async fn handle_socket(mut socket: WebSocket, st: Shared) {
    log::info!("S3 ws: client connected, sending lobby");
    if socket
        .send(Message::Text(st.lobby.as_str().to_owned().into()))
        .await
        .is_err()
    {
        return;
    }
    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                log::info!("S3 ws: recv error ({e}), closing");
                break;
            }
        };
        match msg {
            Message::Text(text) => {
                let v: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(_) => {
                        log::info!("S3 ws: ignoring non-JSON message");
                        continue;
                    }
                };
                let kind = v.get("t").and_then(|x| x.as_str()).unwrap_or("?");
                match kind {
                    "ping" => {
                        let mut reply = v.clone();
                        reply["t"] = serde_json::Value::String("pong".to_string());
                        let out = reply.to_string();
                        log::info!("S3 ws: ping -> pong");
                        if socket.send(Message::Text(out.into())).await.is_err() {
                            break;
                        }
                    }
                    "resync" => {
                        // The live server re-sends state on resync; with no game
                        // running that is the lobby (server.py ws_handler).
                        log::info!("S3 ws: resync -> lobby");
                        if socket
                            .send(Message::Text(st.lobby.as_str().to_owned().into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    other => {
                        log::info!("S3 ws: ignoring t={other}");
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    log::info!("S3 ws: client disconnected");
}

struct ServerInfo {
    port: u16,
    token: String,
}

fn run_server_blocking(tx: std::sync::mpsc::Sender<ServerInfo>, token: String, lobby: String) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("s3-server")
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async move {
        let shared = Shared {
            token: Arc::new(token.clone()),
            lobby: Arc::new(lobby),
        };
        let app = Router::new()
            .route("/ws", get(ws_handler))
            .fallback(get(static_files))
            .with_state(shared);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind 127.0.0.1:0");
        let port = listener.local_addr().expect("local addr").port();
        log::info!("S3 server listening on 127.0.0.1:{port}");
        tx.send(ServerInfo { port, token })
            .expect("report server port");
        axum::serve(listener, app).await.expect("serve");
    });
}

fn load_lobby() -> String {
    let asset = WebAsset::get("mock/01-lobby.json").expect("web/mock/01-lobby.json embedded");
    let text = String::from_utf8(asset.data.into_owned()).expect("lobby mock is UTF-8");
    // Fail fast: the client parses this on connect.
    let v: serde_json::Value = serde_json::from_str(&text).expect("lobby mock is valid JSON");
    assert_eq!(
        v.get("t").and_then(|x| x.as_str()),
        Some("lobby"),
        "lobby mock must be the raw lobby message, not a frame wrapper"
    );
    text
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );
    let token = {
        let bytes = rand::random::<[u8; 12]>();
        use base64::Engine as _;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    };
    let lobby = load_lobby();

    tauri::Builder::default()
        .setup(move |app| {
            let (tx, rx) = std::sync::mpsc::channel();
            let (token_run, lobby_run) = (token.clone(), lobby.clone());
            std::thread::spawn(move || run_server_blocking(tx, token_run, lobby_run));
            let info: ServerInfo = rx
                .recv_timeout(Duration::from_secs(20))
                .expect("loopback server did not start in 20 s");
            // The token is the session secret; this is a local spike build and
            // the test needs the token, so it goes to logcat.
            log::info!(
                "S3 ready: http://127.0.0.1:{}/?t={} (ws on /ws)",
                info.port,
                info.token
            );
            let url = format!("http://127.0.0.1:{}/?t={}", info.port, info.token);
            // The port is only known at runtime, so the window itself is built
            // here with the final loopback URL (instead of navigating a
            // placeholder window, which races webview creation).
            WebviewWindowBuilder::new(
                app.handle(),
                "main",
                WebviewUrl::External(url.parse().expect("loopback url")),
            )
            .title("decklight S3")
            .build()
            .expect("create main window");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
