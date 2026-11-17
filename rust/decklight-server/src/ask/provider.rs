//! The `Provider` seam (docs/plans/ask-tab.md §Core design): one request = one model reply, streamed. v1 has
//! `OpenAiCompatible` (OpenCode Go/Zen, OpenRouter, a local Ollama / LM Studio / llama-server); tests use a scripted
//! provider. Errors are classified here once: retryable (429, 5xx, 408, timeouts, a cut stream) or not (401/403, other
//! 4xx), so the retry module never guesses. The key never appears in an error message.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

/// One request: OpenAI-format messages and tool specs (`None` = the model may not call tools).
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    /// Send the tools but forbid calls (`tool_choice: "none"`): the answer request after the tool limit. History
    /// with tool calls stays valid on providers that require the tool specs next to it.
    pub no_tool_calls: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// The raw JSON text the model sent (parsed by the loop, which reports bad JSON back to the model).
    pub arguments: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    pub prompt: u64,
    pub completion: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reply {
    pub text: String,
    /// DeepSeek-style thinking text (`reasoning_content`): sent back with the assistant message of a tool round,
    /// which thinking-mode APIs require.
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub finish: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProviderError {
    /// Worth another try: 429, 5xx, 408, a timeout, a dropped connection, a stream cut before its end.
    Retryable { msg: String, retry_after: Option<Duration> },
    /// Not worth retrying: the key was rejected, the request is wrong, the model does not exist.
    Fatal { status: Option<u16>, msg: String },
    Cancelled,
}

pub trait Provider: Send + Sync {
    /// One model reply. `on_text` gets text as it streams; `cancel` stops the request between chunks.
    fn complete(
        &self,
        req: &Request,
        on_text: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<Reply, ProviderError>;
}

/// An OpenAI-compatible `/chat/completions` endpoint with a bearer key.
pub struct OpenAiCompatible {
    pub base_url: String,
    pub key: String,
    /// One stable id per conversation, sent as `x-opencode-session`: OpenCode Go refuses requests without it (HTTP 400)
    /// and uses it for routing and prompt caching; other providers ignore the header (NOTEBOOK 2026-11-14).
    pub session: Option<String>,
    agent: ureq::Agent,
}

impl OpenAiCompatible {
    /// `read_timeout` is the longest silence allowed between streamed chunks (not the whole reply).
    pub fn new(base_url: &str, key: &str, connect_timeout: Duration, read_timeout: Duration) -> OpenAiCompatible {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(connect_timeout)
            .timeout_read(read_timeout)
            .user_agent(concat!("decklight-ask/", env!("CARGO_PKG_VERSION")))
            .build();
        OpenAiCompatible { base_url: base_url.trim_end_matches('/').to_string(), key: key.to_string(), session: None, agent }
    }

    /// The same endpoint for one conversation (the agent and its connections are shared).
    pub fn for_session(&self, session: &str) -> OpenAiCompatible {
        OpenAiCompatible {
            base_url: self.base_url.clone(),
            key: self.key.clone(),
            session: Some(session.to_string()),
            agent: self.agent.clone(),
        }
    }

    /// The models this key can use (`GET /models`), for the model picker.
    pub fn models(&self) -> Result<Vec<String>, ProviderError> {
        let resp = self
            .agent
            .get(&format!("{}/models", self.base_url))
            .set("Authorization", &format!("Bearer {}", self.key))
            .call()
            .map_err(|e| classify(e, &self.key))?;
        let text = resp.into_string().map_err(|e| retryable(format!("network: {e}"), None))?;
        let body: Value = serde_json::from_str(&text).map_err(|e| retryable(format!("bad model list: {e}"), None))?;
        let mut ids: Vec<String> = body["data"]
            .as_array()
            .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        ids.sort();
        Ok(ids)
    }
}

fn retryable(msg: String, retry_after: Option<Duration>) -> ProviderError {
    ProviderError::Retryable { msg, retry_after }
}

/// `Retry-After` in seconds (the HTTP-date form is rare for APIs and treated as absent).
fn retry_after(resp: &ureq::Response) -> Option<Duration> {
    // Finite, capped at a day: `inf` or `1e300` would panic in from_secs_f64 (review 2026-11-14); a huge finite value
    // still means "not soon", so the retry gives up instead of backing off.
    let secs = resp.header("retry-after").and_then(|v| v.trim().parse::<f64>().ok())?;
    (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs.min(86_400.0)))
}

/// The provider's own error text, short (a body excerpt; never the request, which holds no key anyway).
fn excerpt(resp: ureq::Response) -> String {
    let body = resp.into_string().unwrap_or_default();
    let msg = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().or(v["error"].as_str()).map(str::to_string))
        .unwrap_or(body);
    msg.chars().take(300).collect()
}

/// Remove the key, and any bearer token, from text that came back from the provider (an error body may echo the
/// request's Authorization header).
pub fn redact(text: &str, key: &str) -> String {
    let mut out = if key.len() >= 4 { text.replace(key, "[key]") } else { text.to_string() };
    while let Some(i) = out.find("Bearer ").filter(|i| !out[i + 7..].starts_with("[key]")) {
        let end = out[i + 7..].find(|c: char| c.is_whitespace() || c == '"' || c == '\'').map_or(out.len(), |j| i + 7 + j);
        out.replace_range(i + 7..end, "[key]");
    }
    out
}

fn classify(e: ureq::Error, key: &str) -> ProviderError {
    let (code, resp) = match e {
        ureq::Error::Status(code, resp) => (code, resp),
        ureq::Error::Transport(t) => return retryable(redact(&format!("network: {t}"), key), None),
    };
    let after = retry_after(&resp);
    let msg = redact(&excerpt(resp), key);
    match code {
        408 | 429 | 500..=599 => retryable(format!("HTTP {code}: {msg}"), after),
        401 | 403 => ProviderError::Fatal { status: Some(code), msg: format!("key rejected (HTTP {code}): {msg}") },
        _ => ProviderError::Fatal { status: Some(code), msg: format!("HTTP {code}: {msg}") },
    }
}

/// More tool calls than this in one reply is a broken stream.
const MAX_CALLS_PER_REPLY: usize = 64;

/// Folds streamed chunks (`choices[0].delta`) into a reply; tool calls arrive in pieces keyed by `index`.
#[derive(Default)]
pub struct StreamFold {
    reply: Reply,
    done: bool,
}

impl StreamFold {
    /// One `data:` payload. Returns the text delta, if any.
    pub fn push(&mut self, data: &str) -> Result<Option<String>, ProviderError> {
        if data.trim() == "[DONE]" {
            self.done = true;
            return Ok(None);
        }
        let v: Value =
            serde_json::from_str(data).map_err(|e| retryable(format!("bad stream chunk: {e}"), None))?;
        if let Some(err) = v.get("error") {
            // Some providers report overload mid-stream as an error chunk.
            return Err(retryable(format!("provider error: {}", err["message"].as_str().unwrap_or("unknown")), None));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.reply.usage = Some(Usage {
                prompt: u["prompt_tokens"].as_u64().unwrap_or(0),
                completion: u["completion_tokens"].as_u64().unwrap_or(0),
            });
        }
        let Some(choice) = v["choices"].get(0) else { return Ok(None) };
        if let Some(f) = choice["finish_reason"].as_str() {
            self.reply.finish = f.to_string();
        }
        let delta = &choice["delta"];
        if let Some(r) = delta["reasoning_content"].as_str() {
            self.reply.reasoning.push_str(r);
        }
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let i = tc["index"].as_u64().unwrap_or(self.reply.tool_calls.len() as u64) as usize;
            // A bogus index would allocate placeholder calls without end (review 2026-11-14).
            if i >= MAX_CALLS_PER_REPLY {
                return Err(retryable(format!("bad stream chunk: tool call index {i}"), None));
            }
            while self.reply.tool_calls.len() <= i {
                self.reply.tool_calls.push(ToolCall { id: String::new(), name: String::new(), arguments: String::new() });
            }
            let slot = &mut self.reply.tool_calls[i];
            if let Some(id) = tc["id"].as_str() {
                slot.id = id.to_string();
            }
            // The name comes whole in its first piece; some providers repeat it in later pieces.
            if let Some(n) = tc["function"]["name"].as_str()
                && slot.name.is_empty()
            {
                slot.name = n.to_string();
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                slot.arguments.push_str(a);
            }
        }
        match delta["content"].as_str().filter(|t| !t.is_empty()) {
            Some(t) => {
                self.reply.text.push_str(t);
                Ok(Some(t.to_string()))
            }
            None => Ok(None),
        }
    }

    /// The reply, or a retryable error when the stream ended before `[DONE]` or a finish reason (a cut stream).
    pub fn finish(mut self) -> Result<Reply, ProviderError> {
        if !self.done && self.reply.finish.is_empty() {
            return Err(retryable("the reply stream was cut off".into(), None));
        }
        // Index gaps leave placeholders without a name: not calls, and they must not use the tool budget.
        self.reply.tool_calls.retain(|tc| !tc.name.is_empty());
        for (i, tc) in self.reply.tool_calls.iter_mut().enumerate() {
            if tc.id.is_empty() {
                tc.id = format!("call_{i}");
            }
        }
        Ok(self.reply)
    }
}

/// Server-sent events: an event's `data:` lines are joined with newlines and end at a blank line; comments
/// (`: ping`) and other fields are skipped.
#[derive(Default)]
pub struct SseLines {
    data: Vec<String>,
}

impl SseLines {
    /// One line (without its line ending; a trailing `\r` is dropped). Returns a finished event's data.
    pub fn line(&mut self, line: &str) -> Option<String> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            return self.end();
        }
        if let Some(d) = line.strip_prefix("data:") {
            self.data.push(d.strip_prefix(' ').unwrap_or(d).to_string());
        }
        None
    }

    /// The pending event at the end of the stream, if any.
    pub fn end(&mut self) -> Option<String> {
        (!self.data.is_empty()).then(|| std::mem::take(&mut self.data).join("\n"))
    }
}

impl Provider for OpenAiCompatible {
    fn complete(
        &self,
        req: &Request,
        on_text: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<Reply, ProviderError> {
        let mut body = json!({
            "model": req.model,
            "messages": req.messages,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if let Some(tools) = &req.tools {
            body["tools"] = json!(tools);
            if req.no_tool_calls {
                body["tool_choice"] = json!("none");
            }
        }
        // The request and its reads run on a helper thread so Stop answers within 100 ms even when the provider is
        // silent (a blocking read cannot see `cancel`). On Stop the helper is left to end by itself: it holds only its
        // socket, touches no chat state, and exits at its next line (its send fails) or read timeout.
        let (tx, rx) = std::sync::mpsc::channel::<Result<Option<String>, ProviderError>>();
        let (agent, url, key, session, body) =
            (self.agent.clone(), format!("{}/chat/completions", self.base_url), self.key.clone(), self.session.clone(), body.to_string());
        std::thread::spawn(move || {
            let mut post = agent
                .post(&url)
                .set("Authorization", &format!("Bearer {key}"))
                .set("Accept", "text/event-stream")
                .set("Content-Type", "application/json");
            if let Some(s) = &session {
                post = post.set("x-opencode-session", s);
            }
            let resp = match post.send_string(&body) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(classify(e, &key)));
                    return;
                }
            };
            for line in BufReader::new(resp.into_reader()).lines() {
                let item = line.map(Some).map_err(|e| retryable(redact(&format!("network: {e}"), &key), None));
                let stop = item.is_err();
                if tx.send(item).is_err() || stop {
                    return;
                }
            }
            let _ = tx.send(Ok(None));
        });
        let mut fold = StreamFold::default();
        let mut sse = SseLines::default();
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(ProviderError::Cancelled);
            }
            let line = match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(Some(line))) => line,
                Ok(Ok(None)) => break,
                Ok(Err(e)) => return Err(e),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(retryable("network: the reply ended without a result".into(), None));
                }
            };
            if let Some(data) = sse.line(&line)
                && let Some(t) = fold.push(&data)?
            {
                on_text(&t);
            }
            if fold.done {
                break;
            }
        }
        if let Some(data) = sse.end()
            && let Some(t) = fold.push(&data)?
        {
            on_text(&t);
        }
        fold.finish()
    }
}
