//! On-device models (docs/plans/ask-tab.md §Local models; NOTEBOOK 2026-11-15 15:55): Gemma 4 E4B on the phone's GPU
//! through LiteRT-LM's C API (`liblitert-lm.so`, loaded at run time; the app ships it with the GPU accelerators in
//! jniLibs). GPU only (user, 2026-11-15: "Use only gou, never cpu phone iis slow as hell"): when the GPU cannot load
//! the model the user is told so and pointed at a cloud provider, never moved to the CPU.
//!
//! The model is one pinned `.litertlm` file, downloaded by the same resumable, hashed downloader as the describe model
//! (`describe::download_one`). The engine is loaded on the first question and dropped after `IDLE` without one, so
//! its GPU memory goes back to the phone.

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::provider::{Provider, ProviderError, Reply, Request, ToolCall};
use crate::describe::{self, ModelFile};

/// The preset id of on-device models (config/ask/providers.v1.json).
pub const PRESET: &str = "device";
const IDLE: Duration = Duration::from_secs(300);
const MAX_TOKENS: i32 = 8192;
const FREE_MARGIN: u64 = 500 * 1024 * 1024;

pub struct Model {
    pub id: &'static str,
    pub name: &'static str,
    pub note: &'static str,
    pub repo: &'static str,
    pub rev: &'static str,
    pub file: ModelFile,
}

/// litert-community/gemma-4-E4B-it-litert-lm, measured on the S23 Ultra GPU (NOTEBOOK 2026-11-15 15:55).
pub const MODELS: &[Model] = &[Model {
    id: "gemma-4-e4b",
    name: "Gemma 4 E4B",
    note: "Runs offline on the GPU",
    repo: "litert-community/gemma-4-E4B-it-litert-lm",
    rev: "f254b5ba6e36f7d89cc7892e268e9877bc20f0a0",
    file: ModelFile {
        path: "gemma-4-E4B-it.litertlm",
        size: 3_659_530_240,
        sha256: "0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0",
    },
}];

pub fn model(id: &str) -> Option<&'static Model> {
    MODELS.iter().find(|m| m.id == id)
}

// ------------------------------------------------------------------------------------------------ the C API

type Status = i32;
type P = *mut c_void;

/// The functions of liblitert-lm.so the app uses (C API 1.0, engine.h / conversation.h / error_reporter.h).
struct Lib {
    _lib: libloading::Library,
    settings_create: unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, *const c_char, *mut P) -> Status,
    settings_delete: unsafe extern "C" fn(P),
    settings_max_tokens: unsafe extern "C" fn(P, i32) -> Status,
    settings_cache_dir: unsafe extern "C" fn(P, *const c_char) -> Status,
    engine_create: unsafe extern "C" fn(P, *mut P) -> Status,
    engine_delete: unsafe extern "C" fn(P),
    conv_config_create: unsafe extern "C" fn(*mut P) -> Status,
    conv_config_delete: unsafe extern "C" fn(P),
    conv_config_system: unsafe extern "C" fn(P, *const c_char) -> Status,
    conv_config_tools: unsafe extern "C" fn(P, *const c_char) -> Status,
    conv_config_messages: unsafe extern "C" fn(P, *const c_char) -> Status,
    conv_create: unsafe extern "C" fn(P, P, *mut P) -> Status,
    conv_delete: unsafe extern "C" fn(P),
    conv_send: unsafe extern "C" fn(P, *const c_char, *const c_char, *const c_void, *mut P) -> Status,
    conv_cancel: unsafe extern "C" fn(P) -> Status,
    response_get: unsafe extern "C" fn(P, *mut *const c_char) -> Status,
    response_delete: unsafe extern "C" fn(P),
    last_error: unsafe extern "C" fn() -> *const c_char,
    set_log_level: unsafe extern "C" fn(i32) -> Status,
}

fn lib_path() -> String {
    // Android finds a bare name in the app's own native library directory.
    std::env::var("DECKLIGHT_LITERT_LIB").unwrap_or_else(|_| "liblitert-lm.so".into())
}

impl Lib {
    fn load() -> Result<Lib, String> {
        let path = lib_path();
        let lib = unsafe { libloading::Library::new(&path) }.map_err(|e| format!("{path}: {e}"))?;
        macro_rules! f {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }.map_err(|e| format!("{}: {e}", $name))?
            };
        }
        Ok(Lib {
            settings_create: f!("litert_lm_engine_settings_create"),
            settings_delete: f!("litert_lm_engine_settings_delete"),
            settings_max_tokens: f!("litert_lm_engine_settings_set_max_num_tokens"),
            settings_cache_dir: f!("litert_lm_engine_settings_set_cache_dir"),
            engine_create: f!("litert_lm_engine_create"),
            engine_delete: f!("litert_lm_engine_delete"),
            conv_config_create: f!("litert_lm_conversation_config_create"),
            conv_config_delete: f!("litert_lm_conversation_config_delete"),
            conv_config_system: f!("litert_lm_conversation_config_set_system_message"),
            conv_config_tools: f!("litert_lm_conversation_config_set_tools"),
            conv_config_messages: f!("litert_lm_conversation_config_set_messages"),
            conv_create: f!("litert_lm_conversation_create"),
            conv_delete: f!("litert_lm_conversation_delete"),
            conv_send: f!("litert_lm_conversation_send_message"),
            conv_cancel: f!("litert_lm_conversation_cancel_process"),
            response_get: f!("litert_lm_json_response_get_string"),
            response_delete: f!("litert_lm_json_response_delete"),
            last_error: f!("litert_lm_get_last_error_message"),
            set_log_level: f!("litert_lm_set_min_log_level"),
            _lib: lib,
        })
    }

    fn check(&self, status: Status, doing: &str) -> Result<(), String> {
        if status == 0 {
            return Ok(());
        }
        let msg = unsafe { (self.last_error)() };
        let detail = if msg.is_null() { String::new() } else { unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned() };
        Err(if detail.is_empty() { format!("{doing} failed (code {status})") } else { format!("{doing} failed: {detail}") })
    }
}

fn cstr(s: &str) -> Result<CString, String> {
    CString::new(s).map_err(|_| "text with a NUL byte".to_string())
}

struct Engine {
    ptr: P,
    model: PathBuf,
    /// Loaded with the vision encoder (only when a question has a photo: it costs GPU memory and load time).
    vision: bool,
    used: Instant,
}

// The engine handle is only used under the runtime's mutex.
unsafe impl Send for Engine {}

/// The library and the loaded engine (one model at a time), shared by every chat.
pub struct Runtime {
    lib: Mutex<Option<Arc<Lib>>>,
    engine: Mutex<Option<Engine>>,
    cache_dir: PathBuf,
}

impl Runtime {
    pub fn new(root: &Path) -> Arc<Runtime> {
        let rt = Arc::new(Runtime { lib: Mutex::new(None), engine: Mutex::new(None), cache_dir: root.join("models/cache") });
        // Drop the engine after IDLE without a question: its GPU memory goes back to the phone.
        let weak = Arc::downgrade(&rt);
        let _ = std::thread::Builder::new().name("ondevice-idle".into()).spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(30));
                let Some(rt) = weak.upgrade() else { return };
                rt.unload_if_idle(IDLE);
            }
        });
        rt
    }

    fn lib(&self) -> Result<Arc<Lib>, String> {
        let mut l = self.lib.lock().unwrap();
        if l.is_none() {
            let lib = Lib::load().map_err(|e| format!("On-device models run in the Android app ({e})"))?;
            unsafe { (lib.set_log_level)(4) };
            *l = Some(Arc::new(lib));
        }
        Ok(l.as_ref().unwrap().clone())
    }

    pub fn loaded(&self) -> bool {
        self.engine.lock().unwrap().is_some()
    }

    pub fn unload(&self) {
        let lib = self.lib.lock().unwrap().clone();
        if let (Some(e), Some(lib)) = (self.engine.lock().unwrap().take(), lib) {
            unsafe { (lib.engine_delete)(e.ptr) };
        }
    }

    fn unload_if_idle(&self, idle: Duration) {
        let lib = self.lib.lock().unwrap().clone();
        let mut g = self.engine.lock().unwrap();
        if g.as_ref().is_some_and(|e| e.used.elapsed() >= idle) {
            if let (Some(e), Some(lib)) = (g.take(), lib) {
                unsafe { (lib.engine_delete)(e.ptr) };
            }
        }
    }

    /// One reply from `model` on the GPU. Holds the engine lock for the whole reply (one answer at a time).
    /// `status` names the model while it loads on the GPU (so that wait is not taken for the app's own), then clears.
    fn complete(&self, model: &Path, name: &str, req: &Request, cancel: &AtomicBool, status: &dyn Fn(&str)) -> Result<Value, ProviderError> {
        let fatal = |msg: String| ProviderError::Fatal { status: None, msg };
        let lib = self.lib().map_err(fatal)?;
        let mut g = self.engine.lock().unwrap();
        let vision = has_image(&req.messages);
        if g.as_ref().is_some_and(|e| e.model != model || (vision && !e.vision)) {
            if let Some(e) = g.take() {
                unsafe { (lib.engine_delete)(e.ptr) };
            }
        }
        if g.is_none() {
            status(&format!("Loading {name} on this phone's GPU, about 20 s"));
            let ptr = load_engine(&lib, model, &self.cache_dir, vision).map_err(fatal)?;
            *g = Some(Engine { ptr, model: model.to_path_buf(), vision, used: Instant::now() });
        }
        let engine = g.as_mut().unwrap();
        engine.used = Instant::now();
        // Loaded: the thinking line goes back to its usual words (user, 2026-11-15).
        status("");
        let (system, history, last) = split(&req.messages).map_err(fatal)?;
        let tools = tools_of(req.tools.iter().flatten());
        let reply = converse(&lib, engine.ptr, system.as_deref(), &tools, &history, &last, cancel);
        engine.used = Instant::now();
        reply
    }
}

fn load_engine(lib: &Lib, model: &Path, cache_dir: &Path, vision: bool) -> Result<P, String> {
    let _ = std::fs::create_dir_all(cache_dir);
    let path = cstr(&model.to_string_lossy())?;
    let (gpu, cache) = (cstr("gpu")?, cstr(&cache_dir.to_string_lossy())?);
    let seeing = if vision { gpu.as_ptr() } else { std::ptr::null() };
    let mut settings: P = std::ptr::null_mut();
    unsafe {
        lib.check((lib.settings_create)(path.as_ptr(), gpu.as_ptr(), seeing, std::ptr::null(), &mut settings), "Reading the model")?;
        let r = (|| {
            lib.check((lib.settings_max_tokens)(settings, MAX_TOKENS), "Setting the context size")?;
            lib.check((lib.settings_cache_dir)(settings, cache.as_ptr()), "Setting the cache folder")?;
            let mut engine: P = std::ptr::null_mut();
            lib.check((lib.engine_create)(settings, &mut engine), "Loading the model on the GPU")
                .map_err(|e| match vision {
                    true => format!("{e}. This phone's GPU could not load Gemma's photo reader: ask without the photo, or use a cloud provider in Ask settings"),
                    false => format!("{e}. This phone's GPU could not run it: use a cloud provider in Ask settings"),
                })?;
            Ok(engine)
        })();
        (lib.settings_delete)(settings);
        r
    }
}

/// Ask's tools as Gemma 4's chat template reads them: `{"type": "function", "function": {name, description, parameters}}`
/// (the template's `tool_data['function']['name']`). LiteRT-LM's docs show bare `{name, ...}`, but the template inside
/// the model file decides, and a bare one fails with "undefined value (in template:80)" (2026-11-15, on the phone).
fn tools_of<'a>(tools: impl Iterator<Item = &'a Value>) -> Vec<Value> {
    tools.map(|t| if t.get("function").is_some() { t.clone() } else { json!({"type": "function", "function": t}) }).collect()
}

/// One conversation: the system message, tools and earlier messages as its preface, then the last message sent.
fn converse(lib: &Lib, engine: P, system: Option<&str>, tools: &[Value], history: &[Value], last: &Value, cancel: &AtomicBool)
    -> Result<Value, ProviderError> {
    let fatal = |msg: String| ProviderError::Fatal { status: None, msg };
    let mut config: P = std::ptr::null_mut();
    let mut conv: P = std::ptr::null_mut();
    unsafe {
        lib.check((lib.conv_config_create)(&mut config), "Starting the conversation").map_err(fatal)?;
        let built = (|| {
            if let Some(s) = system {
                let s = cstr(&json!({"role": "system", "content": s}).to_string())?;
                lib.check((lib.conv_config_system)(config, s.as_ptr()), "Setting the instructions")?;
            }
            if !tools.is_empty() {
                let t = cstr(&Value::Array(tools.to_vec()).to_string())?;
                lib.check((lib.conv_config_tools)(config, t.as_ptr()), "Setting the tools")?;
            }
            if !history.is_empty() {
                let h = cstr(&Value::Array(history.to_vec()).to_string())?;
                lib.check((lib.conv_config_messages)(config, h.as_ptr()), "Setting the earlier messages")?;
            }
            lib.check((lib.conv_create)(engine, config, &mut conv), "Starting the conversation")
        })();
        (lib.conv_config_delete)(config);
        built.map_err(fatal)?;
    }
    // Stop the model from another thread when the user cancels (scoped: it ends before the conversation is freed).
    let done = AtomicBool::new(false);
    let conv_addr = conv as usize;
    let msg = cstr(&last.to_string());
    let mut resp: P = std::ptr::null_mut();
    let status = std::thread::scope(|sc| {
        sc.spawn(|| {
            while !done.load(Ordering::SeqCst) {
                if cancel.load(Ordering::SeqCst) {
                    unsafe { (lib.conv_cancel)(conv_addr as P) };
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        let status = match &msg {
            Ok(m) => unsafe { (lib.conv_send)(conv, m.as_ptr(), std::ptr::null(), std::ptr::null(), &mut resp) },
            Err(_) => 3,
        };
        done.store(true, Ordering::SeqCst);
        status
    });
    let out = if cancel.load(Ordering::SeqCst) {
        Err(ProviderError::Cancelled)
    } else if let Err(e) = msg {
        Err(fatal(e))
    } else if let Err(e) = lib.check(status, "The on-device model") {
        Err(fatal(e))
    } else {
        let mut s: *const c_char = std::ptr::null();
        unsafe { (lib.response_get)(resp, &mut s) };
        let text = if s.is_null() { String::new() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
        serde_json::from_str(&text).map_err(|e| fatal(format!("the on-device model's reply was not JSON: {e}")))
    };
    unsafe {
        if !resp.is_null() {
            (lib.response_delete)(resp);
        }
        (lib.conv_delete)(conv);
    }
    out
}

// ------------------------------------------------------------------------------------------------ message format

/// OpenAI-format messages → (system text, LiteRT preface messages, the message to send). Gemma 4's template
/// (chat_template.jinja of the model repo) wants tool calls as `{"function": {"name", "arguments": {..}}}` and tool
/// results as one `tool` message whose content lists `{"name", "response"}`; consecutive OpenAI tool messages are
/// merged into one.
pub fn split(messages: &[Value]) -> Result<(Option<String>, Vec<Value>, Value), String> {
    let mut system = None;
    let mut out: Vec<Value> = Vec::new();
    let mut names = std::collections::HashMap::new();
    for m in messages {
        match m["role"].as_str().unwrap_or("") {
            "system" if out.is_empty() && system.is_none() => system = Some(text_of(&m["content"])),
            "system" => out.push(json!({"role": "user", "content": text_of(&m["content"])})),
            "user" => out.push(json!({"role": "user", "content": user_content(&m["content"])})),
            "assistant" => {
                let mut a = json!({"role": "assistant", "content": text_of(&m["content"])});
                if let Some(calls) = m["tool_calls"].as_array().filter(|c| !c.is_empty()) {
                    let calls: Vec<Value> = calls.iter().map(|c| {
                        let name = c["function"]["name"].as_str().unwrap_or("").to_string();
                        names.insert(c["id"].as_str().unwrap_or("").to_string(), name.clone());
                        let args = c["function"]["arguments"].as_str().and_then(|s| serde_json::from_str::<Value>(s).ok())
                            .unwrap_or_else(|| c["function"]["arguments"].clone());
                        json!({"type": "function", "function": {"name": name, "arguments": args}})
                    }).collect();
                    a["tool_calls"] = Value::Array(calls);
                }
                out.push(a);
            }
            "tool" => {
                let name = names.get(m["tool_call_id"].as_str().unwrap_or("")).cloned().unwrap_or_else(|| "tool".into());
                let raw = text_of(&m["content"]);
                let response = serde_json::from_str::<Value>(&raw).unwrap_or(Value::String(raw));
                let item = json!({"name": name, "response": response});
                match out.last_mut() {
                    Some(prev) if prev["role"] == "tool" => prev["content"].as_array_mut().unwrap().push(item),
                    _ => out.push(json!({"role": "tool", "content": [item]})),
                }
            }
            other => return Err(format!("unknown message role {other:?}")),
        }
    }
    let last = out.pop().ok_or("no message to send")?;
    if last["role"] == "assistant" {
        return Err("the last message is the model's own".into());
    }
    Ok((system, out, last))
}

fn has_image(messages: &[Value]) -> bool {
    messages.iter().any(|m| m["content"].as_array().is_some_and(|p| p.iter().any(|i| i["type"] == "image_url")))
}

/// A question as LiteRT reads it: text, or text and image parts (an OpenAI `image_url` data URL becomes a `blob`).
fn user_content(c: &Value) -> Value {
    let Some(parts) = c.as_array().filter(|p| p.iter().any(|i| i["type"] == "image_url")) else {
        return json!(text_of(c));
    };
    let parts: Vec<Value> = parts
        .iter()
        .filter_map(|i| match i["type"].as_str()? {
            "text" => Some(json!({"type": "text", "text": i["text"]})),
            "image_url" => {
                let url = i["image_url"]["url"].as_str()?;
                Some(json!({"type": "image", "blob": url.split_once("base64,").map_or(url, |(_, d)| d)}))
            }
            _ => None,
        })
        .collect();
    json!(parts)
}

fn text_of(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(items) => items.iter().filter_map(|i| i["text"].as_str()).collect::<Vec<_>>().join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// LiteRT's reply message → the provider's Reply (tool call arguments become JSON text, ids are made up).
pub fn reply_of(msg: &Value, no_tool_calls: bool) -> Reply {
    let text = text_of(&msg["content"]);
    let tool_calls = if no_tool_calls {
        Vec::new()
    } else {
        msg["tool_calls"].as_array().into_iter().flatten().enumerate().map(|(i, c)| {
            let args = &c["function"]["arguments"];
            ToolCall {
                id: format!("call_{i}"),
                name: c["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: if let Some(s) = args.as_str() { s.to_string() } else if args.is_null() { "{}".into() } else { args.to_string() },
            }
        }).collect()
    };
    let finish = if tool_calls.is_empty() { "stop" } else { "tool_calls" }.to_string();
    Reply { text, reasoning: String::new(), tool_calls, usage: None, finish }
}

/// The provider for one installed model.
pub struct OnDevice {
    pub rt: Arc<Runtime>,
    pub path: PathBuf,
    pub name: String,
    /// Where the "Loading Gemma…" lines go (the chat's thinking line); none in tests.
    pub status: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

impl Provider for OnDevice {
    fn complete(&self, req: &Request, on_text: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<Reply, ProviderError> {
        let status = |t: &str| if let Some(f) = &self.status { f(t) };
        let msg = self.rt.complete(&self.path, &self.name, req, cancel, &status)?;
        let reply = reply_of(&msg, req.no_tool_calls);
        if !reply.text.is_empty() {
            on_text(&reply.text);
        }
        Ok(reply)
    }
}

// ------------------------------------------------------------------------------------------------ downloads

pub fn install_dir(root: &Path, m: &Model) -> PathBuf {
    root.join("models").join(m.id)
}

pub fn model_path(root: &Path, m: &Model) -> PathBuf {
    install_dir(root, m).join(m.file.path)
}

pub fn installed(root: &Path, m: &Model) -> bool {
    std::fs::metadata(model_path(root, m)).map(|x| x.len() == m.file.size).unwrap_or(false)
}

fn on_disk(root: &Path, m: &Model) -> u64 {
    let dir = install_dir(root, m);
    let part = std::fs::metadata(dir.join(format!("{}.part", m.file.path))).map(|x| x.len()).unwrap_or(0);
    let done = std::fs::metadata(dir.join(m.file.path)).map(|x| x.len()).unwrap_or(0);
    (part + done).min(m.file.size)
}

/// Downloads of on-device models: one at a time, resumable, hashed; waits while the phone is off Wi-Fi when the user
/// asked for Wi-Fi only (`cellular` is set from Android's network callback, as for the card art).
pub struct Downloads {
    root: PathBuf,
    current: Mutex<Option<(String, Arc<AtomicBool>)>>,
    error: Mutex<Option<(String, String)>>,
    waiting_wifi: AtomicBool,
    pub wifi_only: AtomicBool,
    pub cellular: AtomicBool,
}

impl Downloads {
    pub fn new(root: &Path) -> Arc<Downloads> {
        // Nothing downloads at start: a status or network file left by a killed process is stale.
        crate::bgjob::clear(root);
        let _ = std::fs::remove_file(root.join("downloads/network"));
        Arc::new(Downloads {
            root: root.to_path_buf(),
            current: Mutex::new(None),
            error: Mutex::new(None),
            waiting_wifi: AtomicBool::new(false),
            wifi_only: AtomicBool::new(true),
            cellular: AtomicBool::new(false),
        })
    }

    pub fn running(&self) -> Option<String> {
        self.current.lock().unwrap().as_ref().map(|(id, _)| id.clone())
    }

    /// Every model with its state, for the "On this device" tab and the download notification.
    pub fn status(&self) -> Value {
        let running = self.running();
        let error = self.error.lock().unwrap().clone();
        let models: Vec<Value> = MODELS.iter().map(|m| {
            let state = if installed(&self.root, m) {
                "installed"
            } else if running.as_deref() == Some(m.id) {
                if self.waiting_wifi.load(Ordering::SeqCst) { "waiting_wifi" } else { "downloading" }
            } else if on_disk(&self.root, m) > 0 {
                "paused"
            } else {
                "none"
            };
            json!({
                "id": m.id, "name": m.name, "note": m.note, "state": state,
                "bytes": on_disk(&self.root, m), "total": m.file.size,
                "error": error.as_ref().filter(|(id, _)| id == m.id).map(|(_, e)| e.clone()),
            })
        }).collect();
        json!({"t": "device_status", "models": models, "wifi_only": self.wifi_only.load(Ordering::SeqCst), "device": device_check(&self.root)})
    }

    pub fn start(self: &Arc<Self>, id: &str, sink: Arc<dyn Fn(Value) + Send + Sync>) -> Result<(), String> {
        let m = model(id).ok_or(format!("unknown on-device model {id:?}"))?;
        if installed(&self.root, m) {
            return Ok(());
        }
        let mut cur = self.current.lock().unwrap();
        if let Some((other, _)) = cur.as_ref() {
            return if other == id { Ok(()) } else { Err(format!("{other} is downloading: wait for it or pause it first")) };
        }
        let dir = install_dir(&self.root, m);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let remaining = m.file.size - on_disk(&self.root, m);
        if let Some(free) = free_bytes(&dir) {
            if free < remaining + FREE_MARGIN {
                return Err(format!("Not enough free space: {} needs {:.1} GB more. Free some space and try again",
                    m.name, (remaining + FREE_MARGIN) as f64 / 1e9));
            }
        }
        let cancel = Arc::new(AtomicBool::new(false));
        *cur = Some((id.to_string(), cancel.clone()));
        *self.error.lock().unwrap() = None;
        drop(cur);
        // Every status frame also goes to Android's background job (bgjob.rs): it keeps the download alive with the
        // app closed and shows it in one notification.
        let root = self.root.clone();
        let sink: Arc<dyn Fn(Value) + Send + Sync> = Arc::new(move |v: Value| {
            crate::bgjob::from_device_status(&root, &v);
            sink(v)
        });
        crate::bgjob::from_device_status(&self.root, &self.status());
        let me = self.clone();
        std::thread::Builder::new().name("model-download".into()).spawn(move || {
            let r = me.run(m, &cancel, &*sink);
            if let Err(e) = r {
                *me.error.lock().unwrap() = Some((m.id.to_string(), format!("{e}. Tap Download to try again")));
            }
            me.waiting_wifi.store(false, Ordering::SeqCst);
            *me.current.lock().unwrap() = None;
            sink(me.status());
        }).map_err(|_| "could not start the model download".to_string())?;
        Ok(())
    }

    fn run(&self, m: &Model, cancel: &AtomicBool, sink: &(dyn Fn(Value) + Send + Sync)) -> Result<(), String> {
        let dir = install_dir(&self.root, m);
        let url = format!("https://huggingface.co/{}/resolve/{}/{}", m.repo, m.rev, m.file.path);
        let mut last = Instant::now();
        let mut tries = 0;
        loop {
            if cancel.load(Ordering::SeqCst) {
                return Ok(());
            }
            if self.wifi_only.load(Ordering::SeqCst) && self.on_cellular() {
                if !self.waiting_wifi.swap(true, Ordering::SeqCst) {
                    sink(self.status());
                }
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
            if self.waiting_wifi.swap(false, Ordering::SeqCst) {
                sink(self.status());
            }
            // Stop between chunks when the user pauses, or when Wi-Fi drops under Wi-Fi only (then wait above).
            let stop = AtomicBool::new(false);
            let mut progress = || {
                if cancel.load(Ordering::SeqCst) || (self.wifi_only.load(Ordering::SeqCst) && self.on_cellular()) {
                    stop.store(true, Ordering::SeqCst);
                }
                if last.elapsed() >= Duration::from_millis(500) {
                    last = Instant::now();
                    sink(self.status());
                }
            };
            match describe::download_one(&dir, &m.file, &url, &describe::HttpDownloader, &stop, &mut progress) {
                Ok(true) => return Ok(()),
                Ok(false) => continue,
                // A dropped connection: resume after a pause, a few times, then tell the user.
                Err(e) if tries < 5 && !e.contains("sha256") => {
                    tries += 1;
                    std::thread::sleep(Duration::from_secs(5 * tries));
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// The background job's view of the network while it runs (the screen may be gone), else the web client's.
    fn on_cellular(&self) -> bool {
        crate::bgjob::cellular(&self.root).unwrap_or_else(|| self.cellular.load(Ordering::SeqCst))
    }

    pub fn pause(&self) {
        if let Some((_, c)) = self.current.lock().unwrap().as_ref() {
            c.store(true, Ordering::SeqCst);
        }
    }

    pub fn remove(&self, id: &str) -> Result<(), String> {
        let m = model(id).ok_or(format!("unknown on-device model {id:?}"))?;
        if self.running().as_deref() == Some(id) {
            self.pause();
            for _ in 0..50 {
                if self.running().is_none() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        let _ = std::fs::remove_dir_all(install_dir(&self.root, m));
        *self.error.lock().unwrap() = None;
        Ok(())
    }
}

fn free_bytes(path: &Path) -> Option<u64> {
    let c = CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

/// What the "On this device" tab shows before a download: GPU (an OpenCL driver the GPU accelerator can use),
/// memory and free storage.
pub fn device_check(root: &Path) -> Value {
    let gpu = ["/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "/system/lib64/libOpenCL.so"]
        .iter().any(|p| Path::new(p).exists());
    let mem_kb = std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
        s.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse::<u64>().ok()
    });
    let android = cfg!(target_os = "android");
    json!({
        "android": android,
        "gpu": gpu,
        "memory": mem_kb.map(|k| k * 1024),
        "free": free_bytes(root),
        "model": std::fs::read_to_string("/proc/device-tree/model").ok().map(|s| s.trim_end_matches('\0').trim().to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_keep_the_function_wrapper_the_template_reads() {
        let openai = json!({"type": "function", "function": {"name": "card", "parameters": {"type": "object"}}});
        let bare = json!({"name": "deck", "parameters": {"type": "object"}});
        let out = tools_of([&openai, &bare].into_iter());
        assert_eq!(out[0], openai);
        assert_eq!(out[1]["function"]["name"], "deck");
        assert_eq!(out[1]["type"], "function");
    }

    #[test]
    fn a_photo_becomes_an_image_blob_and_asks_for_vision() {
        let q = json!({"role": "user", "content": [{"type": "text", "text": "What card?"},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/AA"}}]});
        let (_, _, last) = split(&[json!({"role": "system", "content": "s"}), q.clone()]).unwrap();
        assert_eq!(last["content"], json!([{"type": "text", "text": "What card?"}, {"type": "image", "blob": "/9j/AA"}]));
        assert!(has_image(&[q]));
        assert!(!has_image(&[json!({"role": "user", "content": "text only"})]));
    }

    #[test]
    fn openai_messages_become_gemma_turns_with_merged_tool_results() {
        let msgs = vec![
            json!({"role": "system", "content": "You help with Yu-Gi-Oh."}),
            json!({"role": "user", "content": "What is Ash?"}),
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "card", "arguments": "{\"name\":\"Ash Blossom\"}"}},
                {"id": "b", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"ash\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "a", "content": "{\"type\":\"Monster\"}"}),
            json!({"role": "tool", "tool_call_id": "b", "content": "no rows"}),
        ];
        let (system, history, last) = split(&msgs).unwrap();
        assert_eq!(system.as_deref(), Some("You help with Yu-Gi-Oh."));
        assert_eq!(history.len(), 2);
        assert_eq!(history[1]["tool_calls"][0]["function"]["arguments"]["name"], "Ash Blossom");
        assert_eq!(last["role"], "tool");
        assert_eq!(last["content"][0], json!({"name": "card", "response": {"type": "Monster"}}));
        assert_eq!(last["content"][1], json!({"name": "search", "response": "no rows"}));
        assert!(split(&msgs[..3]).is_err(), "a history ending with the model's own message has nothing to send");
    }

    #[test]
    fn replies_carry_tool_calls_as_json_text_unless_calls_are_off() {
        let msg = json!({"role": "assistant", "content": [{"type": "text", "text": "Looking it up."}],
            "tool_calls": [{"type": "function", "function": {"name": "card", "arguments": {"name": "Ash"}}}]});
        let r = reply_of(&msg, false);
        assert_eq!(r.text, "Looking it up.");
        assert_eq!(r.tool_calls[0].name, "card");
        assert_eq!(serde_json::from_str::<Value>(&r.tool_calls[0].arguments).unwrap(), json!({"name": "Ash"}));
        assert_eq!(r.finish, "tool_calls");
        assert!(reply_of(&msg, true).tool_calls.is_empty());
    }

    #[test]
    fn status_lists_each_model_and_resumes_from_a_part_file() {
        let dir = std::env::temp_dir().join(format!("ondevice-{}", std::process::id()));
        let d = Downloads::new(&dir);
        assert_eq!(d.status()["models"][0]["state"], "none");
        let m = &MODELS[0];
        std::fs::create_dir_all(install_dir(&dir, m)).unwrap();
        std::fs::write(install_dir(&dir, m).join(format!("{}.part", m.file.path)), b"abc").unwrap();
        let s = d.status();
        assert_eq!(s["models"][0]["state"], "paused");
        assert_eq!(s["models"][0]["bytes"], 3);
        d.remove(m.id).unwrap();
        assert_eq!(d.status()["models"][0]["state"], "none");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_the_library_the_error_says_where_models_run() {
        let rt = Runtime::new(&std::env::temp_dir());
        let req = Request { messages: vec![json!({"role": "user", "content": "hi"})], ..Default::default() };
        // SAFETY of the test: only runs where liblitert-lm.so is absent (the laptop), so no model is loaded.
        if std::env::var("DECKLIGHT_LITERT_LIB").is_err() && !cfg!(target_os = "android") {
            let e = rt.complete(Path::new("/nonexistent.litertlm"), "Gemma", &req, &AtomicBool::new(false), &|_| {}).unwrap_err();
            assert!(matches!(&e, ProviderError::Fatal { msg, .. } if msg.contains("Android app")), "{e:?}");
        }
    }
}
