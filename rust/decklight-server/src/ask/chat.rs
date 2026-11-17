//! The tool loop for one user turn (docs/plans/ask-tab.md §Core design; decisions 2026-11-14 §3). The model may call
//! at most `max_tool_calls` tools per answer and is told so: the system prompt states the limit, every tool result
//! carries `calls_left`, and at 0 the last result carries a note to answer with what it has and the answer request
//! forbids calls (`tool_choice: "none"`). Tool errors go back to the model as results so it can correct itself;
//! provider errors go through `retry`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use super::provider::{Provider, ProviderError, Reply, Request, Usage};
use super::retry::{self, Policy, RetryNote};

/// The `Tool` seam: the specs the model sees and the calls it makes. Results are compact JSON for the model.
///
/// `cancel` is the turn's Stop button (user 2026-11-14: "we just need to be 100% sure that if we are running compute
/// expensive tools we can cancel it when hitting the stop button without crashing the phone or conversation or leaving
/// anything stray"). A tool that can run longer than ~100 ms polls it at least every 100 ms, and on a stop returns
/// `Err` only after its own threads, files and engine handles are joined, closed or dropped. The turn then stores the
/// call as stopped and ends with "Stopped."; nothing of the tool outlives the call.
pub trait Tools {
    fn specs(&self) -> Vec<Value>;
    fn call(&mut self, name: &str, args: &Value, cancel: &AtomicBool) -> Result<Value, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Streamed answer text.
    Text(String),
    /// A retry: text streamed since the last tool round is void (the UI drops it) and the request runs again.
    Retry(RetryNote),
    ToolStart { id: String, name: String, args: Value },
    ToolDone { id: String, name: String, ok: bool, result: Value },
    /// The tool limit was reached; the final answer comes without tools.
    LimitReached,
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_tool_calls: u32,
    pub retry: Policy,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits { max_tool_calls: 8, retry: Policy::default() }
    }
}

#[derive(Debug, Default)]
pub struct Outcome {
    /// The messages this turn added after the user's (assistant and tool messages), in OpenAI format.
    pub messages: Vec<Value>,
    pub answer: String,
    pub usage: Usage,
    pub tool_calls: u32,
}

#[derive(Debug, PartialEq)]
pub struct TurnError {
    pub error: ProviderError,
    /// Messages added before the failure (tool rounds that completed), so the store keeps them and a retry goes on
    /// from there.
    pub partial: Vec<Value>,
    /// Tokens spent before the failure: they count toward the chat's cap.
    pub usage: Usage,
}

impl TurnError {
    fn at(error: ProviderError, out: &Outcome) -> TurnError {
        TurnError { error, partial: out.messages.clone(), usage: out.usage.clone() }
    }
}

/// The system prompt with its `{max_tool_calls}` filled in.
pub fn system_prompt(template: &str, limits: &Limits) -> String {
    template.replace("{max_tool_calls}", &limits.max_tool_calls.to_string())
}

/// Run one turn. `history` is the conversation so far, system prompt first and the new user message last.
pub fn run_turn(
    provider: &dyn Provider,
    tools: &mut dyn Tools,
    model: &str,
    history: &[Value],
    limits: &Limits,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(Event),
    sleep: &mut dyn FnMut(Duration, &AtomicBool) -> bool,
) -> Result<Outcome, TurnError> {
    let mut out = Outcome::default();
    let specs = tools.specs();
    // Shared by the streaming and retry callbacks (never borrowed by both at once).
    let events = std::cell::RefCell::new(on_event);
    let cancelled = |out: &Outcome| TurnError::at(ProviderError::Cancelled, out);
    // One extra answer request when a model still answers a no-tools request with calls and no text.
    let mut nudged = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(cancelled(&out));
        }
        let left = limits.max_tool_calls.saturating_sub(out.tool_calls);
        let mut messages = history.to_vec();
        messages.extend(out.messages.iter().cloned());
        if nudged {
            // Not stored: it only asks for the answer the stored conversation is missing.
            messages.push(json!({"role": "user", "content": "Please answer now in text, without calling tools."}));
        }
        let req = Request {
            model: model.to_string(),
            messages,
            tools: (!specs.is_empty()).then(|| specs.clone()),
            no_tool_calls: left == 0,
        };
        let reply = retry::run(
            &limits.retry,
            cancel,
            || provider.complete(&req, &mut |t| (events.borrow_mut())(Event::Text(t.to_string())), cancel),
            |note| (events.borrow_mut())(Event::Retry(note.clone())),
            |d, c| sleep(d, c),
        )
        .map_err(|error| TurnError::at(error, &out))?;
        if let Some(u) = &reply.usage {
            out.usage.prompt += u.prompt;
            out.usage.completion += u.completion;
        }
        // Calls made although calls were forbidden are ignored: that request asked for the answer.
        if reply.tool_calls.is_empty() || left == 0 || specs.is_empty() {
            if reply.text.trim().is_empty() {
                if !reply.tool_calls.is_empty() && !nudged {
                    nudged = true;
                    continue;
                }
                let msg = "the model returned an empty answer".to_string();
                return Err(TurnError::at(ProviderError::Fatal { status: None, msg }, &out));
            }
            out.messages.push(json!({"role": "assistant", "content": reply.text}));
            out.answer = reply.text;
            return Ok(out);
        }
        out.messages.push(assistant_with_calls(&reply));
        for tc in &reply.tool_calls {
            if cancel.load(Ordering::Relaxed) {
                return Err(cancelled(&out));
            }
            let args = match tc.arguments.trim() {
                "" => Ok(json!({})),
                a => serde_json::from_str::<Value>(a),
            };
            let shown = args.as_ref().cloned().unwrap_or_else(|_| json!(tc.arguments));
            (events.borrow_mut())(Event::ToolStart { id: tc.id.clone(), name: tc.name.clone(), args: shown });
            // Every call gets a result (the API requires one per id); calls beyond the limit get a refusal.
            let (ok, mut result) = if out.tool_calls >= limits.max_tool_calls {
                (false, json!({"error": "tool limit reached for this answer; answer with what you have"}))
            } else {
                out.tool_calls += 1;
                match args {
                    Err(e) => (false, json!({"error": format!("arguments are not valid JSON: {e}")})),
                    Ok(args) => match tools.call(&tc.name, &args, cancel) {
                        Ok(v) => (true, v),
                        Err(e) => (false, json!({"error": e})),
                    },
                }
            };
            (events.borrow_mut())(Event::ToolDone { id: tc.id.clone(), name: tc.name.clone(), ok, result: result.clone() });
            let calls_left = limits.max_tool_calls.saturating_sub(out.tool_calls);
            match result.as_object_mut() {
                Some(o) => {
                    o.insert("calls_left".into(), json!(calls_left));
                }
                None => result = json!({"result": result, "calls_left": calls_left}),
            }
            out.messages.push(json!({"role": "tool", "tool_call_id": tc.id, "content": result.to_string()}));
        }
        if out.tool_calls >= limits.max_tool_calls {
            (events.borrow_mut())(Event::LimitReached);
            // The note rides in the last tool result: a `system` message mid-conversation is rejected by strict
            // chat templates (llama.cpp, LM Studio, Ollama; some DeepSeek gateways), review 2026-11-14.
            let last = out.messages.last_mut().expect("a tool round adds tool messages");
            let mut content: Value = serde_json::from_str(last["content"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
            content["note"] = json!(format!(
                "Tool limit reached ({n} of {n} calls used). Answer now with what you have, and say briefly what \
                 you could not check.",
                n = limits.max_tool_calls
            ));
            last["content"] = json!(content.to_string());
        }
    }
}

fn assistant_with_calls(reply: &Reply) -> Value {
    let mut m = json!({
        "role": "assistant",
        "content": if reply.text.is_empty() { Value::Null } else { json!(reply.text) },
        "tool_calls": reply.tool_calls.iter().map(|c| json!({
            "id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments},
        })).collect::<Vec<_>>(),
    });
    if !reply.reasoning.is_empty() {
        m["reasoning_content"] = json!(reply.reasoning);
    }
    m
}
