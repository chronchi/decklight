//! One chat turn over the store (what `ask_send` runs, step 5): the cap check, the user message stored first, the
//! context built from the branch, the loop, and the outcome stored. A failure keeps the tool rounds that completed and
//! adds an error row that says what happened and what the user can do (docs/plans/ask-tab.md §Errors the user sees).

use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::chat::{self, Event, Limits, Tools};
use super::provider::{Provider, ProviderError};
use super::store::{ChatStore, Policy, build_context};

/// What the turn answers.
pub enum Input<'a> {
    /// A new user message under `parent` (`None` = the chat's head).
    /// `attach` are the checked references (`attach::accept`), `meta` the question row's meta (photo thumbnails).
    New { parent: Option<&'a str>, text: &'a str, attach: &'a [Value], meta: &'a Value },
    /// Answer an existing user message again (Retry after an error, or regenerate): a new branch under it.
    Again { user: &'a str },
}

/// The fixed parts of a turn: who to ask and with what.
pub struct Setup<'a> {
    pub provider: &'a dyn Provider,
    /// The preset's display name, for error messages ("OpenCode Go rejected the key").
    pub provider_name: &'a str,
    pub system_prompt: &'a str,
    /// Recorded with every answer (decision 6: prompts and tools are versioned files).
    pub versions: Value,
    pub limits: Limits,
    pub policy: Policy,
    /// The `[[Name]]` cards of an answer (`host::card_refs`: cards {name: code}, unresolved [name]), stored in its meta.
    pub cards: &'a dyn Fn(&str) -> Value,
    /// Where attached photos are saved (`attach::images_dir`).
    pub images: &'a std::path::Path,
}

#[derive(Debug)]
pub struct Sent {
    pub user: String,
    pub answer: String,
    /// The stored ids of this turn's assistant and tool messages.
    pub ids: Vec<String>,
    pub elided: usize,
    pub dropped: usize,
}

/// A failed turn: the message and the action for the user. `user` is the stored question when there is one (Retry
/// answers it again).
#[derive(Debug, PartialEq)]
pub struct Failed {
    pub message: String,
    /// retry | replace_key | pick_model | switch_provider | new_chat | check_settings | none; the ws layer adds
    /// add_key (no key saved yet) and cancel (another answer is still running).
    pub action: &'static str,
    pub user: Option<String>,
}

fn failed(message: String, action: &'static str, user: Option<String>) -> Failed {
    Failed { message, action, user }
}

/// "6 h", "25 min", "40 s".
fn span(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 5400 {
        format!("{} h", (s + 1800) / 3600)
    } else if s >= 90 {
        format!("{} min", (s + 30) / 60)
    } else {
        format!("{s} s")
    }
}

fn is_too_long(msg: &str) -> bool {
    let m = msg.to_lowercase();
    ["context length", "context_length", "maximum context", "context window", "too many tokens"].iter().any(|k| m.contains(k))
}

/// The phone could not reach the network at all (no DNS, no route), not a slow or failing provider.
fn is_offline(msg: &str) -> bool {
    let m = msg.to_lowercase();
    ["dns", "resolve", "network is unreachable", "no route to host", "failed to lookup"].iter().any(|k| m.contains(k))
}

fn is_about_images(msg: &str) -> bool {
    let m = msg.to_lowercase();
    ["image", "vision", "multimodal", "image_url"].iter().any(|k| m.contains(k))
}

/// A provider error as the message and action the user sees.
pub fn explain(e: &ProviderError, provider: &str, max_wait: Duration) -> (String, &'static str) {
    match e {
        ProviderError::Retryable { retry_after: Some(d), msg } if *d > max_wait => (
            format!("{provider} limit reached: try again in {}, or switch to another provider or model ({msg})", span(*d)),
            "switch_provider",
        ),
        ProviderError::Retryable { msg, .. } if is_offline(msg) => (
            format!("Could not reach {provider}: no internet connection. Check Wi-Fi or mobile data, then try again ({msg})"),
            "retry",
        ),
        ProviderError::Retryable { msg, .. } => (format!("{provider} did not answer after several tries ({msg}); try again"), "retry"),
        ProviderError::Fatal { status: Some(401 | 403), msg } => {
            (format!("{provider} rejected the key: replace it in Settings ({msg})"), "replace_key")
        }
        ProviderError::Fatal { status: Some(404), msg } => {
            (format!("{provider} does not know this model: pick another one ({msg})"), "pick_model")
        }
        // Before the model rule: "This model's maximum context length is …" is about the chat's length.
        ProviderError::Fatal { status: Some(_), msg } if is_too_long(msg) => (
            format!("This chat is too long for the model: start a new chat or pick a model with a longer context ({msg})"),
            "new_chat",
        ),
        // Before the model rule too: "this model does not support image input" is about the photo.
        ProviderError::Fatal { status: Some(_), msg } if is_about_images(msg) => (
            format!("{provider}'s model cannot read photos: pick a model that accepts images, or ask without the photo ({msg})"),
            "pick_model",
        ),
        ProviderError::Fatal { status: Some(_), msg } if msg.to_lowercase().contains("model") => {
            (format!("{provider} refused the model: pick another one ({msg})"), "pick_model")
        }
        ProviderError::Fatal { status: Some(_), msg } => {
            (format!("{provider} refused the request: check the provider settings ({msg})"), "check_settings")
        }
        ProviderError::Fatal { status: None, msg } => (format!("{msg}; try again"), "retry"),
        ProviderError::Cancelled => ("Stopped.".into(), "none"),
    }
}

/// Every assistant message with tool calls needs a result for each call right after it (OpenAI-format APIs reject the
/// history otherwise). A turn stopped between calls leaves some unanswered, and a fork or a send under such a message
/// puts a question right after it (review 2026-11-14), so the missing ones get a "not run" result.
pub fn complete_calls(msgs: &mut Vec<Value>) {
    let mut out = Vec::with_capacity(msgs.len());
    let mut it = std::mem::take(msgs).into_iter().peekable();
    while let Some(m) = it.next() {
        let calls: Vec<String> = m["tool_calls"]
            .as_array()
            .filter(|_| m["role"] == "assistant")
            .into_iter()
            .flatten()
            .filter_map(|c| c["id"].as_str().map(str::to_string))
            .collect();
        out.push(m);
        if calls.is_empty() {
            continue;
        }
        let mut answered = Vec::new();
        while let Some(t) = it.next_if(|t| t["role"] == "tool") {
            answered.extend(t["tool_call_id"].as_str().map(str::to_string));
            out.push(t);
        }
        for id in calls.into_iter().filter(|id| !answered.contains(id)) {
            let content = json!({"error": "not run: the turn stopped before this call"}).to_string();
            out.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
        }
    }
    *msgs = out;
}

fn question(text: &str, attach: &[Value]) -> Value {
    let mut q = json!({"role": "user", "content": text});
    if !attach.is_empty() {
        q["attach"] = json!(attach);
    }
    q
}

/// Run one turn of `chat`. Events stream as in `chat::run_turn`.
pub fn send(
    store: &ChatStore,
    chat_id: &str,
    input: Input,
    setup: &Setup,
    tools: &mut dyn Tools,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(Event),
    sleep: &mut dyn FnMut(Duration, &AtomicBool) -> bool,
) -> Result<Sent, Failed> {
    let chat = store.get(chat_id).map_err(|e| failed(e, "none", None))?;
    if chat.tokens() >= setup.policy.chat_token_cap {
        return Err(failed(
            format!(
                "This chat has used {} of its {} tokens: start a new chat (a fresh chat costs less per question)",
                chat.tokens(),
                setup.policy.chat_token_cap
            ),
            "new_chat",
            None,
        ));
    }
    let user = match input {
        Input::New { parent, text, attach, meta } => {
            let text = text.trim();
            if text.is_empty() && attach.is_empty() {
                return Err(failed("Type a question first.".into(), "none", None));
            }
            let parent = parent.map(str::to_string).or(chat.head.clone());
            let ids = store
                .append(chat_id, parent.as_deref(), &[question(text, attach)], meta)
                .map_err(|e| failed(e, "retry", None))?;
            ids[0].clone()
        }
        Input::Again { user } => {
            let b = store.branch(chat_id, Some(user)).map_err(|e| failed(e, "none", None))?;
            if b.last().is_none_or(|r| r.role != "user") {
                return Err(failed("Only a question can be answered again.".into(), "none", None));
            }
            store.set_head(chat_id, Some(user)).map_err(|e| failed(e, "retry", None))?;
            user.to_string()
        }
    };
    let branch = store.branch(chat_id, Some(&user)).map_err(|e| failed(e, "none", Some(user.clone())))?;
    let system = json!({"role": "system", "content": chat::system_prompt(setup.system_prompt, &setup.limits)});
    let mut ctx = build_context(&system, &branch, setup.policy.context_tokens);
    complete_calls(&mut ctx.messages);
    // A failed read (a photo gone from disk) is stored as the question's error row, like a provider error, so it stays
    // with its question and Retry or Edit can follow (review 2026-11-15).
    if let Err(message) = super::attach::expand(&mut ctx.messages, tools, setup.images, cancel) {
        store.append(chat_id, Some(&user), &[json!({"error": message, "action": "none"})], &json!({}))
            .map_err(|err| failed(err, "retry", Some(user.clone())))?;
        return Err(failed(message, "none", Some(user)));
    }
    let t = Instant::now();
    let result = chat::run_turn(setup.provider, tools, &chat.model, &ctx.messages, &setup.limits, cancel, on_event, sleep);
    let mut meta = json!({
        "preset": chat.preset, "model": chat.model, "versions": setup.versions,
        "latency_ms": t.elapsed().as_millis() as u64, "elided": ctx.elided, "dropped": ctx.dropped,
    });
    match result {
        Ok(out) => {
            meta["tokens"] = json!({"prompt": out.usage.prompt, "completion": out.usage.completion});
            meta["tool_calls"] = json!(out.tool_calls);
            let refs = (setup.cards)(&out.answer);
            for k in ["cards", "unresolved"] {
                if refs[k].as_object().is_some_and(|o| !o.is_empty()) || refs[k].as_array().is_some_and(|a| !a.is_empty()) {
                    meta[k] = refs[k].clone();
                }
            }
            // Usage first: the tokens were spent whatever happens next, and an answer stored before a failed usage
            // update would be shown as a failure and answered twice on Retry.
            store.add_usage(chat_id, out.usage.prompt, out.usage.completion).map_err(|e| failed(e, "retry", Some(user.clone())))?;
            let ids = store.append(chat_id, Some(&user), &out.messages, &meta).map_err(|e| failed(e, "retry", Some(user.clone())))?;
            Ok(Sent { user, answer: out.answer, ids, elided: ctx.elided, dropped: ctx.dropped })
        }
        Err(e) => {
            let (message, action) = explain(&e.error, setup.provider_name, setup.limits.retry.max_delay);
            meta["tokens"] = json!({"prompt": e.usage.prompt, "completion": e.usage.completion});
            let mut rows = e.partial.clone();
            complete_calls(&mut rows);
            rows.push(json!({"error": message, "action": action}));
            // The failure is shown even if saving it fails; the save error then wins (the chat is in doubt).
            store.add_usage(chat_id, e.usage.prompt, e.usage.completion).map_err(|err| failed(err, "retry", Some(user.clone())))?;
            store.append(chat_id, Some(&user), &rows, &meta).map_err(|err| failed(err, "retry", Some(user.clone())))?;
            Err(failed(message, action, Some(user)))
        }
    }
}
