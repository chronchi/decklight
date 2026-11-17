//! The Ask ws messages (docs/plans/ask-tab.md §Wire protocol, step 5). Every reply echoes `req`; a failure is an
//! `ask_error` frame with the message and the action the user can take (§Errors the user sees).
//!
//!   ask_settings {op: get | set | key_set | key_delete | models}   → ask_settings (or ask_models)
//!   ask_list                                                       → ask_chats
//!   ask_open {chat}                                                → ask_chat (every row, the head, running)
//!   ask_send {chat?, parent?, text} | {chat, again: <user id>}     → ask_started, then ask_event … done | error
//!   ask_cancel                                                     → (the running turn ends with error "Stopped.")
//!   ask_fork {chat, message}                                       → ask_chat (the head moves; the next send goes there)
//!   ask_delete {chat}                                              → ask_chats
//!
//! One turn runs at a time, on its own thread; its events go to whichever connection is current (`set_out`).
//! Keys only ever leave as hints.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::chat::{Event, Limits};
use super::host::{self, AppTools};
use super::keys::ApiKeys;
use super::presets::{self, Preset};
use super::ondevice::{self, OnDevice};
use super::provider::{OpenAiCompatible, Provider};
use super::retry;
use super::store::{Chat, ChatStore, Policy, Row};
use super::turn::{self, Input, Setup};
use crate::server::App;

/// A provider ready to use: a remote endpoint, an installed on-device model, or the on-device preset with no model
/// chosen yet (it can list models but not answer).
enum Prov {
    Http(OpenAiCompatible),
    Device(OnDevice),
    DeviceList,
}

/// Where frames for the current connection go.
pub type Out = Box<dyn Fn(Value) + Send + Sync>;

/// The provider choice, saved as `settings.json` next to the chats.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Settings {
    pub preset: String,
    pub model: String,
    /// Only for the `custom` preset.
    #[serde(default)]
    pub base_url: String,
}

impl Default for Settings {
    fn default() -> Settings {
        let p = presets::preset("opencode-go").expect("the opencode-go preset exists");
        Settings { preset: p.id, model: p.default_model, base_url: String::new() }
    }
}

impl Settings {
    /// The checked settings: a known preset, a model, and an http(s) base URL where the preset has none.
    fn checked(self) -> Result<Settings, String> {
        let p = presets::preset(&self.preset).ok_or(format!("unknown provider {:?}", self.preset))?;
        let model = self.model.trim().to_string();
        if model.is_empty() || model.len() > 200 {
            return Err("pick a model (1 to 200 characters)".into());
        }
        if p.id == ondevice::PRESET {
            if ondevice::model(&model).is_none() && model != "-" {
                return Err(format!("{model:?} is not an on-device model: pick one under On this device"));
            }
            return Ok(Settings { preset: p.id, model, base_url: String::new() });
        }
        let base_url = self.base_url.trim().trim_end_matches('/').to_string();
        if p.base_url.is_empty() && !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(format!("{} needs a base URL that starts with http:// or https://", p.name));
        }
        Ok(Settings { preset: p.id, model, base_url: if p.base_url.is_empty() { base_url } else { String::new() } })
    }
}

struct Running {
    chat: String,
    cancel: Arc<AtomicBool>,
}

pub struct Ask {
    dir: PathBuf,
    store: Mutex<Option<Arc<ChatStore>>>,
    settings: Mutex<Option<Settings>>,
    running: Mutex<Option<Running>>,
    out: Mutex<Option<Out>>,
    pub policy: Policy,
    pub limits: Limits,
}

fn error_frame(message: impl Into<String>, action: &str) -> Value {
    json!({"t": "ask_error", "message": message.into(), "action": action})
}

/// The action for an error the handlers return as text (their messages already say what to do).
fn action_for(message: &str) -> &'static str {
    if message.contains("stop it first") {
        "cancel"
    } else if message.contains("may have been deleted") {
        "new_chat"
    } else if message.contains("Settings") || message.contains("base URL") || message.contains("provider") {
        "check_settings"
    } else if message.contains("could not be read or saved") || message.contains("cannot ") {
        "retry"
    } else {
        "none"
    }
}

fn with_req(mut frame: Value, req: &Option<Value>) -> Value {
    if let Some(r) = req {
        frame["req"] = r.clone();
    }
    frame
}

pub fn chat_json(c: &Chat) -> Value {
    json!({
        "id": c.id, "title": c.title, "created": c.created, "updated": c.updated, "head": c.head,
        "preset": c.preset, "model": c.model, "tokens": c.tokens(),
    })
}

fn row_json(r: &Row) -> Value {
    json!({"id": r.id, "parent": r.parent, "created": r.created, "role": r.role, "body": r.body, "meta": r.meta})
}

pub fn event_json(e: &Event) -> Value {
    match e {
        Event::Text(text) => json!({"kind": "text", "text": text}),
        Event::Retry(n) => json!({
            "kind": "retry", "attempt": n.attempt, "delay_ms": n.delay.as_millis() as u64, "reason": n.reason,
        }),
        Event::ToolStart { id, name, args } => json!({"kind": "tool_start", "id": id, "name": name, "args": args}),
        Event::ToolDone { id, name, ok, result } => {
            json!({"kind": "tool_done", "id": id, "name": name, "ok": ok, "result": result})
        }
        Event::LimitReached => json!({"kind": "limit"}),
    }
}

fn text_arg<'a>(msg: &'a Value, key: &str) -> Result<&'a str, String> {
    msg[key].as_str().filter(|s| !s.is_empty()).ok_or(format!("give `{key}`"))
}

impl Ask {
    /// Chats and settings live in `dir` (opened on first use, so a broken folder shows up as an Ask error only).
    pub fn new(dir: &Path) -> Ask {
        Ask {
            dir: dir.to_path_buf(),
            store: Mutex::new(None),
            settings: Mutex::new(None),
            running: Mutex::new(None),
            out: Mutex::new(None),
            policy: Policy::default(),
            limits: Limits::default(),
        }
    }

    pub fn set_out(&self, out: Option<Out>) {
        *self.out.lock().unwrap() = out;
    }

    pub fn emit(&self, frame: Value) {
        if let Some(out) = self.out.lock().unwrap().as_ref() {
            out(frame);
        }
    }

    /// The chat store, opened on first use; a failed open is tried again next time (the folder may be fixed by then).
    pub fn store(&self) -> Result<Arc<ChatStore>, String> {
        let mut store = self.store.lock().unwrap();
        if let Some(s) = store.as_ref() {
            return Ok(Arc::clone(s));
        }
        let s = Arc::new(ChatStore::open(&self.dir)?);
        *store = Some(Arc::clone(&s));
        Ok(s)
    }

    fn settings_path(&self) -> PathBuf {
        self.dir.join("settings.json")
    }

    /// The saved settings; a missing file is the defaults, a broken one is an error (never silently reset).
    pub fn settings(&self) -> Result<Settings, String> {
        let mut cached = self.settings.lock().unwrap();
        if let Some(s) = cached.as_ref() {
            return Ok(s.clone());
        }
        let path = self.settings_path();
        let s = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<Settings>(&text)
                .map_err(|e| format!("{} is broken ({e}): pick the provider again in Settings", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Settings::default(),
            Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
        };
        *cached = Some(s.clone());
        Ok(s)
    }

    pub fn save_settings(&self, s: Settings) -> Result<Settings, String> {
        let s = s.checked()?;
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("cannot create {}: {e}", self.dir.display()))?;
        let tmp = self.dir.join("settings.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&s).unwrap_or_default())
            .and_then(|_| std::fs::rename(&tmp, self.settings_path()))
            .map_err(|e| format!("cannot save the Ask settings: {e}"))?;
        *self.settings.lock().unwrap() = Some(s.clone());
        Ok(s)
    }

    /// The chat a turn is running on, if any.
    pub fn running_chat(&self) -> Option<String> {
        self.running.lock().unwrap().as_ref().map(|r| r.chat.clone())
    }

    fn settings_frame(&self, keys: &ApiKeys) -> Result<Value, String> {
        let s = self.settings()?;
        let store = keys.store();
        let mut list = Vec::new();
        for p in presets::presets() {
            list.push(json!({
                "id": p.id, "name": p.name, "note": p.note, "keys_url": p.keys_url, "needs_key": p.needs_key,
                "default_model": p.default_model, "base_url": p.base_url, "key": keys.status(&p.id)?,
            }));
        }
        Ok(json!({
            "t": "ask_settings", "presets": list, "preset": s.preset, "model": s.model, "base_url": s.base_url,
            "key_store": {"kind": store.kind(), "persistent": store.persistent()},
        }))
    }

    /// The provider for `s`, or the message and action when it cannot be used yet (no key, no base URL, the on-device
    /// model not downloaded).
    fn provider_for(&self, app: &App, s: &Settings) -> Result<(Preset, Prov), (String, &'static str)> {
        let p = presets::preset(&s.preset).ok_or((format!("unknown provider {:?}", s.preset), "check_settings"))?;
        if p.id == ondevice::PRESET {
            let Some(m) = ondevice::model(&s.model) else {
                return Ok((p, Prov::DeviceList));
            };
            if !ondevice::installed(&app.root, m) {
                return Err((format!("{} is not downloaded yet: download it under On this device", m.name), "check_settings"));
            }
            let path = ondevice::model_path(&app.root, m);
            return Ok((p, Prov::Device(OnDevice { rt: Arc::clone(&app.ondevice), path, name: m.name.to_string(), status: None })));
        }
        let key = app.api_keys.get(&p.id).map_err(|e| (e, "check_settings"))?.unwrap_or_default();
        if p.needs_key && key.is_empty() {
            let at = if p.keys_url.is_empty() { String::new() } else { format!(" (get one at {})", p.keys_url) };
            return Err((format!("No {} key yet: add it in Settings{at}", p.name), "add_key"));
        }
        let base = if p.base_url.is_empty() { s.base_url.clone() } else { p.base_url.clone() };
        if base.is_empty() {
            return Err((format!("{} needs a base URL: set it in Settings", p.name), "check_settings"));
        }
        Ok((p, Prov::Http(OpenAiCompatible::new(&base, &key, Duration::from_secs(15), Duration::from_secs(120)))))
    }

    /// The Ask messages; `None` for any other type.
    pub fn handle(&self, app: &Arc<App>, t: Option<&str>, msg: &Value) -> Option<Vec<Value>> {
        let req = msg.get("req").cloned();
        let result = match t? {
            "ask_settings" => self.on_settings(app, msg, &req),
            "ask_list" => self.chats_frame(),
            "ask_open" => text_arg(msg, "chat").and_then(|c| self.chat_frame(c)),
            "ask_send" => self.on_send(app, msg, &req),
            "ask_cancel" => {
                let running = self.running.lock().unwrap();
                if let Some(r) = running.as_ref() {
                    r.cancel.store(true, Ordering::SeqCst);
                }
                // The turn itself ends with an `ask_event` error "Stopped.".
                Ok(json!({"t": "ask_cancel", "running": running.as_ref().map(|r| r.chat.clone())}))
            }
            "ask_fork" => self.on_fork(msg),
            "ask_delete" => self.on_delete(msg),
            "ask_frames" => self.on_frames(app, msg, &req),
            "ask_data" => self.on_data(app, msg, &req),
            _ => return None,
        };
        Some(vec![with_req(result.unwrap_or_else(|e| { let a = action_for(&e); error_frame(e, a) }), &req)])
    }

    fn on_settings(&self, app: &Arc<App>, msg: &Value, req: &Option<Value>) -> Result<Value, String> {
        let keys = &app.api_keys;
        match msg["op"].as_str().unwrap_or("get") {
            "get" => {}
            "set" => {
                let cur = self.settings()?;
                let preset = msg["preset"].as_str().unwrap_or(&cur.preset).to_string();
                // A new provider without a model gets that provider's default.
                let model = match msg["model"].as_str() {
                    Some(m) => m.to_string(),
                    None if preset != cur.preset => presets::preset(&preset).map(|p| p.default_model).unwrap_or_default(),
                    None => cur.model,
                };
                let base_url = msg["base_url"].as_str().map(str::to_string).unwrap_or(cur.base_url);
                self.save_settings(Settings { preset, model, base_url })?;
            }
            "key_set" => {
                keys.set(text_arg(msg, "preset")?, msg["key"].as_str().unwrap_or(""))?;
            }
            "key_delete" => keys.delete(text_arg(msg, "preset")?)?,
            "models" => return self.on_models(app, msg, req),
            other => return Err(format!("unknown ask_settings op {other:?}: get, set, key_set, key_delete, models")),
        }
        self.settings_frame(keys)
    }

    /// The boards for a chat block (a turn's plays or one decision), replayed on a thread like the model list.
    fn on_frames(&self, app: &Arc<App>, msg: &Value, req: &Option<Value>) -> Result<Value, String> {
        let (app, req, args) = (Arc::clone(app), req.clone(), msg.clone());
        std::thread::spawn(move || {
            // A panic must still answer, or the block waits on "Replaying the game…" for good.
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::replay::frames(&app, &args, &AtomicBool::new(false))));
            let run = run.unwrap_or_else(|_| {
                eprintln!("ALERT ask: ask_frames panicked for {}", args);
                Err("the replay crashed (see the server log); try again, or report the game".into())
            });
            let frame = match run {
                Ok(mut v) => {
                    v["t"] = json!("ask_frames");
                    v["key"] = args["key"].clone();
                    v
                }
                Err(e) => json!({"t": "ask_frames", "key": args["key"], "error": e}),
            };
            app.ask.emit(with_req(frame, &req));
        });
        Ok(json!({"t": "ask_frames_pending", "key": msg["key"]}))
    }

    /// The Games screens' data (ask/games.rs): the same reads as the replay, bot_view and my_stats tools, on a
    /// thread. `what` is list, ids (every id under the list's chips and search), delete (to the records' trash), export
    /// (EDOPro replay or text files, ask/export.rs), game,
    /// check (every decision's bot view, with progress frames) or bot (one decision).
    fn on_data(&self, app: &Arc<App>, msg: &Value, req: &Option<Value>) -> Result<Value, String> {
        static CHECKING: AtomicBool = AtomicBool::new(false);
        let what = msg["what"].as_str().unwrap_or("").to_string();
        // Errors carry the key, so the screen that asked shows them.
        let refuse = |e: String| Ok(json!({"t": "ask_data", "key": msg["key"], "error": e}));
        if !["list", "ids", "delete", "export", "game", "check", "bot", "watch"].contains(&what.as_str()) {
            return refuse(format!("unknown ask_data {what:?}: list, ids, delete, export, game, check or bot"));
        }
        if what == "check" && CHECKING.swap(true, Ordering::SeqCst) {
            return refuse("a check of another game is still running: wait for it to finish, then try again".into());
        }
        let (app, req, key, args) = (Arc::clone(app), req.clone(), msg["key"].clone(), msg["args"].clone());
        let pending = json!({"t": "ask_data_pending", "key": key});
        std::thread::spawn(move || {
            let no = AtomicBool::new(false);
            let progress = |done: usize, total: usize| {
                app.ask.emit(with_req(json!({"t": "ask_data", "key": key, "progress": {"done": done, "total": total}}), &req));
            };
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match what.as_str() {
                "list" => super::games::list(&app, &args, &no),
                "ids" => super::games::ids(&app, &args),
                "delete" => super::games::delete(&app, &args),
                "export" => super::export::export(&app, &args, &no),
                "watch" => super::replay::watch(&app, &args, &no),
                "game" => super::games::page(&app, &args, &no),
                "check" => super::games::check(&app, &args, &no, &progress),
                _ => super::replay::bot_view(&app, &args, &no),
            }));
            if what == "check" {
                CHECKING.store(false, Ordering::SeqCst);
            }
            let run = run.unwrap_or_else(|_| {
                eprintln!("ALERT games: ask_data {what} panicked for {args}");
                Err("the replay crashed (see the server log); try again, or report the game".into())
            });
            let frame = match run {
                Ok(v) => json!({"t": "ask_data", "key": key, "result": v}),
                Err(e) => json!({"t": "ask_data", "key": key, "error": e}),
            };
            app.ask.emit(with_req(frame, &req));
        });
        Ok(pending)
    }

    /// The provider's model list, fetched on a thread (a network call never blocks the socket).
    fn on_models(&self, app: &Arc<App>, msg: &Value, req: &Option<Value>) -> Result<Value, String> {
        let mut s = self.settings()?;
        if let Some(p) = msg["preset"].as_str() {
            s.preset = p.to_string();
        }
        if let Some(b) = msg["base_url"].as_str() {
            s.base_url = b.to_string();
        }
        // The same checks as saving (a model name is not needed to list models).
        if s.model.trim().is_empty() {
            s.model = "-".into();
        }
        let s = s.checked()?;
        let (p, provider) = match self.provider_for(app, &s) {
            Ok(x) => x,
            Err((m, action)) => return Ok(error_frame(m, action)),
        };
        let (app, req) = (Arc::clone(app), req.clone());
        std::thread::spawn(move || {
            let listed = match &provider {
                Prov::Http(h) => h.models(),
                _ => Ok(ondevice::MODELS.iter().map(|m| m.id.to_string()).collect()),
            };
            let frame = match listed {
                Ok(models) if models.is_empty() => error_frame(
                    format!("{} listed no models: type the model name in Settings", p.name),
                    "check_settings",
                ),
                Ok(models) => json!({"t": "ask_models", "preset": p.id, "models": models}),
                Err(e) => {
                    let (m, action) = turn::explain(&e, &p.name, Duration::from_secs(20));
                    error_frame(m, action)
                }
            };
            app.ask.emit(with_req(frame, &req));
        });
        Ok(json!({"t": "ask_models_pending", "preset": s.preset}))
    }

    fn chats_frame(&self) -> Result<Value, String> {
        let chats: Vec<Value> = self.store()?.list()?.iter().map(chat_json).collect();
        Ok(json!({"t": "ask_chats", "chats": chats, "running": self.running_chat()}))
    }

    fn chat_frame(&self, chat: &str) -> Result<Value, String> {
        let store = self.store()?;
        let c = store.get(chat)?;
        let rows: Vec<Value> = store.rows(chat)?.iter().map(row_json).collect();
        let running = self.running_chat().as_deref() == Some(chat);
        Ok(json!({"t": "ask_chat", "chat": chat_json(&c), "rows": rows, "running": running}))
    }

    fn on_fork(&self, msg: &Value) -> Result<Value, String> {
        let chat = text_arg(msg, "chat")?;
        if self.running_chat().as_deref() == Some(chat) {
            return Err("An answer is still coming in this chat: stop it first".into());
        }
        self.store()?.set_head(chat, Some(text_arg(msg, "message")?))?;
        self.chat_frame(chat)
    }

    fn on_delete(&self, msg: &Value) -> Result<Value, String> {
        let chat = text_arg(msg, "chat")?;
        if self.running_chat().as_deref() == Some(chat) {
            return Err("An answer is still coming in this chat: stop it first".into());
        }
        let store = self.store()?;
        let photos: Vec<String> = store.rows(chat)?.iter()
            .flat_map(|r| super::attach::photos_of(r.body["attach"].as_array().map_or(&[][..], |a| a.as_slice()))).collect();
        store.delete(chat)?;
        // Its photos go with it (a failed removal only leaves a file behind).
        for f in &photos {
            let _ = std::fs::remove_file(super::attach::images_dir(&self.dir).join(f));
        }
        self.chats_frame()
    }

    fn on_send(&self, app: &Arc<App>, msg: &Value, req: &Option<Value>) -> Result<Value, String> {
        let store = self.store()?;
        let s = self.settings()?;
        let (p, provider) = match self.provider_for(app, &s) {
            Ok(x) => x,
            Err((m, action)) => return Ok(error_frame(m, action)),
        };
        // Before `running` is taken and a chat made: returning later left both behind (review 2026-11-15).
        if matches!(provider, Prov::DeviceList) {
            return Ok(error_frame("Pick a model under On this device first", "check_settings"));
        }
        let mut running = self.running.lock().unwrap();
        if running.is_some() {
            return Ok(error_frame("An answer is still coming: wait for it or stop it first", "cancel"));
        }
        let again = msg["again"].as_str().map(str::to_string);
        let text = msg["text"].as_str().unwrap_or("").to_string();
        let has_attach = msg["attach"].as_array().is_some_and(|a| !a.is_empty());
        if again.is_none() && text.trim().is_empty() && !has_attach {
            return Ok(error_frame("Type a question first.", "none"));
        }
        // Checked (and a photo saved) before a chat is made for it: a refused attachment leaves no empty chat.
        let images = super::attach::images_dir(&self.dir);
        let (attach, attach_meta) = match &again {
            Some(_) => (vec![], json!({})),
            None => match super::attach::accept(&msg["attach"], &images) {
                Ok(a) => a,
                Err(e) => return Ok(error_frame(e, "none")),
            },
        };
        // A photo saved for this send (not an edited question's, which another row holds): removed if no question is stored.
        let sent_photo = msg["attach"].as_array().is_some_and(|a| a.iter().any(|x| x["kind"] == "photo" && x.get("data").is_some()));
        let fresh: Vec<String> = if sent_photo { super::attach::photos_of(&attach) } else { vec![] };
        let chat = match msg["chat"].as_str() {
            Some(id) => {
                let c = store.get(id)?;
                if (c.preset.as_str(), c.model.as_str()) != (s.preset.as_str(), s.model.as_str()) {
                    store.set_model(id, &s.preset, &s.model)?;
                }
                id.to_string()
            }
            None if again.is_some() => return Err("give `chat` with `again`".into()),
            None => store.create(&s.preset, &s.model)?.id,
        };
        let parent = msg["parent"].as_str().map(str::to_string);
        let cancel = Arc::new(AtomicBool::new(false));
        *running = Some(Running { chat: chat.clone(), cancel: Arc::clone(&cancel) });
        drop(running);

        let provider: Box<dyn Provider> = match provider {
            Prov::Http(h) if p.session_header => Box::new(h.for_session(&chat)),
            Prov::Http(h) => Box::new(h),
            Prov::Device(d) => {
                // The thinking line names the on-device model while it loads and answers (user, 2026-11-15:
                // "otherwise people will think it's decklight").
                let (app, chat, req) = (Arc::clone(app), chat.clone(), req.clone());
                let status = move |t: &str| app.ask.emit(with_req(json!({"t": "ask_event", "chat": chat, "kind": "status", "text": t}), &req));
                Box::new(OnDevice { status: Some(Arc::new(status)), ..d })
            }
            Prov::DeviceList => unreachable!("refused before the turn starts"),
        };
        let (app, req2, chat2) = (Arc::clone(app), req.clone(), chat.clone());
        let title = if again.is_some() { store.get(&chat).map(|c| c.title).unwrap_or_default() } else { text.clone() };
        crate::bgjob::answer_start(&app.root, &title);
        std::thread::spawn(move || {
            let ask = &app.ask;
            let frame = |body: Value| {
                let mut f = json!({"t": "ask_event", "chat": chat2});
                if let (Some(f), Some(b)) = (f.as_object_mut(), body.as_object()) {
                    f.extend(b.clone());
                }
                with_req(f, &req2)
            };
            let setup = Setup {
                provider: provider.as_ref(),
                provider_name: &p.name,
                system_prompt: host::SYSTEM_PROMPT,
                versions: json!({"prompt": host::SYSTEM_PROMPT_VERSION, "tools": "tools.v1", "providers": "providers.v1"}),
                limits: ask.limits.clone(),
                policy: ask.policy.clone(),
                cards: &|t| host::card_refs(&app, t),
                images: &images,
            };
            let input = match &again {
                Some(user) => Input::Again { user },
                None => Input::New { parent: parent.as_deref(), text: &text, attach: &attach, meta: &attach_meta },
            };
            // A panic anywhere in the turn must not leave `running` set (every later send would be refused).
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut tools = AppTools::new(Arc::clone(&app));
                let mut on_event = |e: Event| ask.emit(frame(event_json(&e)));
                match ask.store() {
                    Ok(store) => turn::send(
                        &store, &chat2, input, &setup, &mut tools, &cancel, &mut on_event, &mut retry::sleep_cancellable,
                    ),
                    Err(e) => Err(turn::Failed { message: e, action: "retry", user: None }),
                }
            }))
            .unwrap_or_else(|_| {
                Err(turn::Failed {
                    message: "The answer stopped on an internal error (a bug): try again, and report it if it repeats".into(),
                    action: "retry",
                    user: None,
                })
            });
            *ask.running.lock().unwrap_or_else(|e| e.into_inner()) = None;
            crate::bgjob::answer_end(&app.root, result.is_ok());
            if matches!(&result, Err(f) if f.user.is_none()) {
                for f in &fresh {
                    let _ = std::fs::remove_file(images.join(f));
                }
            }
            ask.emit(frame(match result {
                Ok(sent) => json!({"kind": "done", "user": sent.user, "answer": sent.answer, "ids": sent.ids,
                                   "elided": sent.elided, "dropped": sent.dropped}),
                Err(f) => json!({"kind": "error", "message": f.message, "action": f.action, "user": f.user}),
            }));
        });
        Ok(json!({"t": "ask_started", "chat": chat, "preset": s.preset, "model": s.model}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ask-ws-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn settings_default_save_and_reload() {
        let d = dir("settings");
        let ask = Ask::new(&d);
        assert_eq!(ask.settings().unwrap(), Settings::default());
        let s = Settings { preset: "openrouter".into(), model: " x/y ".into(), base_url: "http://ignored".into() };
        let saved = ask.save_settings(s).unwrap();
        assert_eq!((saved.model.as_str(), saved.base_url.as_str()), ("x/y", ""));
        assert_eq!(Ask::new(&d).settings().unwrap(), saved);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn settings_are_checked() {
        let bad = |preset: &str, model: &str, base: &str| {
            Settings { preset: preset.into(), model: model.into(), base_url: base.into() }.checked().unwrap_err()
        };
        assert!(bad("nope", "m", "").contains("unknown provider"));
        assert!(bad("opencode-go", " ", "").contains("pick a model"));
        assert!(bad("custom", "m", "ftp://x").contains("base URL"));
        let ok = Settings { preset: "custom".into(), model: "m".into(), base_url: "http://127.0.0.1:11434/v1/".into() };
        assert_eq!(ok.checked().unwrap().base_url, "http://127.0.0.1:11434/v1");
    }

    #[test]
    fn broken_settings_fail_loud() {
        let d = dir("broken");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("settings.json"), "{").unwrap();
        assert!(Ask::new(&d).settings().unwrap_err().contains("pick the provider again"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn chats_open_fork_delete() {
        let d = dir("chats");
        let ask = Ask::new(&d);
        let store = ask.store().unwrap();
        let c = store.create("opencode-go", "m").unwrap();
        let ids = store
            .append(&c.id, None, &[json!({"role": "user", "content": "hi"}), json!({"role": "assistant", "content": "yo"})], &json!({}))
            .unwrap();
        let list = ask.chats_frame().unwrap();
        assert_eq!(list["chats"][0]["title"], "hi");
        let open = ask.chat_frame(&c.id).unwrap();
        assert_eq!(open["rows"].as_array().unwrap().len(), 2);
        assert_eq!(open["running"], false);
        let forked = ask.on_fork(&json!({"chat": c.id, "message": ids[0]})).unwrap();
        assert_eq!(forked["chat"]["head"], ids[0].as_str());
        // A turn running on the chat blocks fork and delete.
        *ask.running.lock().unwrap() = Some(Running { chat: c.id.clone(), cancel: Arc::default() });
        assert!(ask.on_delete(&json!({"chat": c.id})).unwrap_err().contains("stop it first"));
        *ask.running.lock().unwrap() = None;
        assert_eq!(ask.on_delete(&json!({"chat": c.id})).unwrap()["chats"], json!([]));
        assert!(ask.chat_frame(&c.id).unwrap_err().contains("deleted"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn text_errors_get_an_action() {
        assert_eq!(action_for("An answer is still coming in this chat: stop it first"), "cancel");
        assert_eq!(action_for("no chat \"x\"; it may have been deleted"), "new_chat");
        assert_eq!(action_for("x is broken (eof): pick the provider again in Settings"), "check_settings");
        assert_eq!(action_for("the chat history could not be read or saved (locked)"), "retry");
    }

    #[test]
    fn a_failed_open_is_tried_again() {
        let d = dir("reopen");
        std::fs::write(&d, "a file where the folder should be").unwrap();
        let ask = Ask::new(&d);
        assert!(ask.store().is_err());
        std::fs::remove_file(&d).unwrap();
        assert!(ask.store().is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn two_threads_write_two_chats_without_locking_errors() {
        let d = dir("writers");
        let store = Ask::new(&d).store().unwrap();
        let chats: Vec<String> = (0..2).map(|_| store.create("opencode-go", "m").unwrap().id).collect();
        let handles: Vec<_> = chats
            .into_iter()
            .map(|c| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    for i in 0..40 {
                        store.append(&c, None, &[json!({"role": "user", "content": format!("q{i}")})], &json!({})).unwrap();
                        store.add_usage(&c, 1, 1).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(store.append(&store.list().unwrap()[0].id, None, &[], &json!({})).is_err(), "empty append refused");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn events_as_frames() {
        assert_eq!(event_json(&Event::Text("a".into())), json!({"kind": "text", "text": "a"}));
        assert_eq!(event_json(&Event::LimitReached), json!({"kind": "limit"}));
    }
}
