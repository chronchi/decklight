//! Offline tests of the Ask loop (docs/plans/ask-tab.md §Core design 7): a scripted provider drives `run_turn`, and a
//! fake HTTP server that fails on purpose (429 + Retry-After, 500, 401, 400, cut streams, silence, refused
//! connections) drives `OpenAiCompatible`. No tokens, no network beyond 127.0.0.1.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde_json::{Value, json};

use super::chat::{Event, Limits, Tools, run_turn, system_prompt};
use super::provider::{OpenAiCompatible, Provider, ProviderError, Reply, Request, ToolCall, Usage};

// ---- scripted provider and tools ----

struct Scripted {
    replies: Mutex<VecDeque<Result<Reply, ProviderError>>>,
    seen: Mutex<Vec<Request>>,
}

impl Scripted {
    fn new(replies: Vec<Result<Reply, ProviderError>>) -> Scripted {
        Scripted { replies: Mutex::new(replies.into()), seen: Mutex::new(Vec::new()) }
    }
    fn requests(&self) -> Vec<Request> {
        self.seen.lock().unwrap().clone()
    }
}

impl Provider for Scripted {
    fn complete(&self, req: &Request, on_text: &mut dyn FnMut(&str), _: &AtomicBool) -> Result<Reply, ProviderError> {
        self.seen.lock().unwrap().push(req.clone());
        let r = self.replies.lock().unwrap().pop_front().expect("the script ran out of replies");
        // Like a real stream: only non-empty text is streamed (a tool-call reply has none).
        if let Ok(reply) = r.as_ref() && !reply.text.is_empty() {
            on_text(&reply.text);
        }
        r
    }
}

fn text(t: &str) -> Result<Reply, ProviderError> {
    Ok(Reply { text: t.into(), finish: "stop".into(), usage: Some(Usage { prompt: 10, completion: 2 }), ..Reply::default() })
}

fn calls(names: &[(&str, &str)]) -> Result<Reply, ProviderError> {
    let tool_calls = names
        .iter()
        .enumerate()
        .map(|(i, (n, a))| ToolCall { id: format!("c{i}"), name: n.to_string(), arguments: a.to_string() })
        .collect();
    Ok(Reply { tool_calls, finish: "tool_calls".into(), usage: Some(Usage { prompt: 10, completion: 1 }), ..Reply::default() })
}

#[derive(Default)]
struct Fake {
    called: Vec<(String, Value)>,
}

impl Tools for Fake {
    fn specs(&self) -> Vec<Value> {
        vec![json!({"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}})]
    }
    fn call(&mut self, name: &str, args: &Value, _: &AtomicBool) -> Result<Value, String> {
        self.called.push((name.into(), args.clone()));
        match name {
            "echo" => Ok(json!({"echo": args})),
            _ => Err(format!("unknown tool {name}")),
        }
    }
}

fn turn(
    p: &dyn Provider,
    tools: &mut Fake,
    limits: &Limits,
) -> (Result<super::chat::Outcome, super::chat::TurnError>, Vec<Event>, Vec<Duration>) {
    let history = vec![json!({"role": "system", "content": "s"}), json!({"role": "user", "content": "q"})];
    let (mut events, mut slept) = (Vec::new(), Vec::new());
    let cancel = AtomicBool::new(false);
    let r = run_turn(p, tools, "m", &history, limits, &cancel, &mut |e| events.push(e), &mut |d, _| {
        slept.push(d);
        true
    });
    (r, events, slept)
}

fn content(m: &Value) -> Value {
    serde_json::from_str(m["content"].as_str().unwrap()).unwrap()
}

// ---- the loop ----

#[test]
fn plain_answer() {
    let p = Scripted::new(vec![text("Hello")]);
    let (r, events, _) = turn(&p, &mut Fake::default(), &Limits::default());
    let out = r.unwrap();
    assert_eq!(out.answer, "Hello");
    assert_eq!(events, vec![Event::Text("Hello".into())]);
    assert_eq!(out.usage, Usage { prompt: 10, completion: 2 });
    assert!(p.requests()[0].tools.is_some());
}

#[test]
fn tool_round_then_answer_with_calls_left() {
    let p = Scripted::new(vec![calls(&[("echo", r#"{"x":1}"#)]), text("Done")]);
    let mut tools = Fake::default();
    let (r, events, _) = turn(&p, &mut tools, &Limits::default());
    let out = r.unwrap();
    assert_eq!(tools.called, vec![("echo".into(), json!({"x": 1}))]);
    assert_eq!(out.tool_calls, 1);
    assert_eq!(out.messages[0]["tool_calls"][0]["function"]["name"], "echo");
    assert_eq!(out.messages[1]["role"], "tool");
    assert_eq!(content(&out.messages[1]), json!({"echo": {"x": 1}, "calls_left": 7}));
    assert_eq!(p.requests()[1].messages.len(), 4, "system, user, assistant call, tool result");
    assert!(matches!(&events[0], Event::ToolStart { name, .. } if name == "echo"));
    assert!(matches!(&events[1], Event::ToolDone { ok: true, .. }));
}

#[test]
fn the_limit_is_enforced_and_the_last_request_has_no_tools() {
    let mut script: Vec<_> = (0..3).map(|_| calls(&[("echo", "{}")])).collect();
    script.push(text("Here is what I found."));
    let p = Scripted::new(script);
    let mut tools = Fake::default();
    let limits = Limits { max_tool_calls: 3, ..Limits::default() };
    let (r, events, _) = turn(&p, &mut tools, &limits);
    let out = r.unwrap();
    assert_eq!((tools.called.len(), out.tool_calls), (3, 3));
    assert!(events.contains(&Event::LimitReached));
    let reqs = p.requests();
    assert_eq!(reqs.len(), 4);
    assert!(!reqs[2].no_tool_calls && reqs[3].no_tool_calls, "the answer request forbids calls");
    assert!(reqs[3].tools.is_some(), "the specs stay next to the tool-call history");
    let last = reqs[3].messages.last().unwrap();
    assert_eq!(last["role"], "tool", "no system message mid-conversation");
    assert!(content(last)["note"].as_str().unwrap().contains("Tool limit reached (3 of 3"));
    assert_eq!(content(&out.messages[5])["calls_left"], 0);
    assert!(out.messages.iter().all(|m| m["role"] != "system"));
}

#[test]
fn calls_past_the_limit_in_one_reply_are_refused_but_answered() {
    let five: Vec<(&str, &str)> = (0..5).map(|_| ("echo", "{}")).collect();
    let p = Scripted::new(vec![calls(&five), text("ok")]);
    let mut tools = Fake::default();
    let (r, _, _) = turn(&p, &mut tools, &Limits { max_tool_calls: 3, ..Limits::default() });
    let out = r.unwrap();
    assert_eq!(tools.called.len(), 3);
    let results: Vec<Value> = out.messages[1..6].iter().map(content).collect();
    assert_eq!(results.len(), 5, "every tool_call id gets a result");
    assert!(results[3]["error"].as_str().unwrap().contains("tool limit"));
    assert_eq!(out.messages[1..6].iter().map(|m| m["tool_call_id"].as_str().unwrap()).collect::<Vec<_>>(),
               vec!["c0", "c1", "c2", "c3", "c4"]);
}

#[test]
fn bad_arguments_and_tool_errors_go_back_to_the_model() {
    let p = Scripted::new(vec![calls(&[("echo", "{not json"), ("nope", "{}")]), text("sorry")]);
    let mut tools = Fake::default();
    let (r, events, _) = turn(&p, &mut tools, &Limits::default());
    let out = r.unwrap();
    assert!(content(&out.messages[1])["error"].as_str().unwrap().contains("not valid JSON"));
    assert!(matches!(&events[0], Event::ToolStart { args, .. } if args == "{not json"), "bad JSON still shows a start");
    assert_eq!(content(&out.messages[2])["error"], "unknown tool nope");
    assert_eq!(tools.called.len(), 1, "bad JSON never reaches the tool");
    assert_eq!(events.iter().filter(|e| matches!(e, Event::ToolDone { ok: false, .. })).count(), 2);
    assert_eq!(out.tool_calls, 2, "failed calls still count toward the limit");
}

#[test]
fn retryable_errors_are_retried_with_events() {
    let p = Scripted::new(vec![
        Err(ProviderError::Retryable { msg: "HTTP 429".into(), retry_after: Some(Duration::from_secs(3)) }),
        text("Hi"),
    ]);
    let (r, events, slept) = turn(&p, &mut Fake::default(), &Limits::default());
    assert_eq!(r.unwrap().answer, "Hi");
    assert_eq!(slept, vec![Duration::from_secs(3)]);
    assert!(matches!(&events[0], Event::Retry(n) if n.attempt == 2 && n.reason == "HTTP 429"));
}

#[test]
fn a_fatal_error_ends_the_turn_and_keeps_completed_rounds() {
    let fatal = ProviderError::Fatal { status: Some(401), msg: "key rejected (HTTP 401)".into() };
    let p = Scripted::new(vec![calls(&[("echo", "{}")]), Err(fatal.clone())]);
    let (r, _, _) = turn(&p, &mut Fake::default(), &Limits::default());
    let e = r.unwrap_err();
    assert_eq!((e.error, e.partial.len()), (fatal, 2));
}

#[test]
fn an_empty_answer_is_an_error() {
    let p = Scripted::new(vec![text("  ")]);
    let (r, _, _) = turn(&p, &mut Fake::default(), &Limits::default());
    assert!(matches!(r.unwrap_err().error, ProviderError::Fatal { status: None, .. }));
}

#[test]
fn the_shipped_prompt_states_the_limit() {
    let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/ask/prompts/system.v1.md")).unwrap();
    let s = system_prompt(&t, &Limits::default());
    assert!(s.contains("at most 8 tools") && !s.contains('{'), "{s}");
}

// ---- the fake failing server ----

/// Serve one canned response per connection, in order; returns the base URL and the requests it saw.
fn serve(responses: Vec<Box<dyn FnOnce(&mut TcpStream) + Send>>) -> (String, std::sync::Arc<Mutex<Vec<String>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", l.local_addr().unwrap());
    let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for respond in responses {
            let Ok((mut s, _)) = l.accept() else { return };
            seen2.lock().unwrap().push(read_request(&mut s));
            respond(&mut s);
        }
    });
    (base, seen)
}

fn read_request(s: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut b = [0u8; 4096];
    loop {
        let n = s.read(&mut b).unwrap_or(0);
        buf.extend_from_slice(&b[..n]);
        let text = String::from_utf8_lossy(&buf).to_string();
        if let Some(i) = text.find("\r\n\r\n") {
            let len = text[..i]
                .lines()
                .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
                .unwrap_or(0usize);
            if buf.len() >= i + 4 + len {
                return text;
            }
        }
        if n == 0 {
            return text;
        }
    }
}

fn status(code: u16, extra: &'static str, body: &'static str) -> Box<dyn FnOnce(&mut TcpStream) + Send> {
    Box::new(move |s| {
        let _ = write!(s, "HTTP/1.1 {code} X\r\n{extra}Content-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    })
}

fn sse(chunks: Vec<String>, done: bool) -> Box<dyn FnOnce(&mut TcpStream) + Send> {
    Box::new(move |s| {
        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
        for c in chunks {
            let _ = write!(s, "data: {c}\n\n");
            let _ = s.flush();
        }
        if done {
            let _ = write!(s, "data: [DONE]\n\n");
        }
    })
}

fn good_stream() -> Box<dyn FnOnce(&mut TcpStream) + Send> {
    sse(
        vec![
            json!({"choices": [{"delta": {"content": "Hel"}}]}).to_string(),
            json!({"choices": [{"delta": {"content": "lo"}}]}).to_string(),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t1", "function": {"name": "echo", "arguments": "{\"a\""}}]}}]}).to_string(),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":2}"}}]}, "finish_reason": "tool_calls"}]}).to_string(),
            json!({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 9}}).to_string(),
        ],
        true,
    )
}

const KEY: &str = "sk-test-SECRET-1a2b";

fn client(base: &str, read_ms: u64) -> OpenAiCompatible {
    OpenAiCompatible::new(base, KEY, Duration::from_millis(500), Duration::from_millis(read_ms))
}

fn ask(c: &OpenAiCompatible) -> (Result<Reply, ProviderError>, String) {
    let mut streamed = String::new();
    let req = Request {
        model: "deepseek-v4.1-flash".into(),
        messages: vec![json!({"role": "user", "content": "hi"})],
        ..Request::default()
    };
    let r = c.complete(&req, &mut |t| streamed.push_str(t), &AtomicBool::new(false));
    (r, streamed)
}

#[test]
fn streams_text_tool_calls_and_usage() {
    let (base, seen) = serve(vec![good_stream()]);
    let (r, streamed) = ask(&client(&base, 2000));
    let reply = r.unwrap();
    assert_eq!((streamed.as_str(), reply.text.as_str()), ("Hello", "Hello"));
    assert_eq!(reply.tool_calls, vec![ToolCall { id: "t1".into(), name: "echo".into(), arguments: "{\"a\":2}".into() }]);
    assert_eq!(reply.usage, Some(Usage { prompt: 120, completion: 9 }));
    let req = seen.lock().unwrap()[0].clone();
    assert!(req.starts_with("POST /v1/chat/completions"));
    assert!(req.contains(&format!("Bearer {KEY}")) && req.contains("\"stream\":true"));
}

#[test]
fn a_session_sends_its_id_and_our_user_agent() {
    let (base, seen) = serve(vec![good_stream(), good_stream()]);
    let c = client(&base, 2000);
    ask(&c).0.unwrap();
    ask(&c.for_session("chat-42")).0.unwrap();
    let seen = seen.lock().unwrap();
    let lower = |i: usize| seen[i].to_ascii_lowercase();
    assert!(!lower(0).contains("x-opencode-session") && lower(0).contains("user-agent: decklight-ask/"));
    assert!(lower(1).contains("x-opencode-session: chat-42"), "OpenCode Go refuses requests without it");
}

#[test]
fn http_errors_are_classified_and_never_show_the_key() {
    let (base, _) = serve(vec![
        status(429, "Retry-After: 3\r\n", r#"{"error":{"message":"rate limited"}}"#),
        status(503, "", "overloaded"),
        status(401, "", r#"{"error":{"message":"invalid api key"}}"#),
        status(400, "", r#"{"error":{"message":"unknown model"}}"#),
    ]);
    let c = client(&base, 2000);
    let (r, _) = ask(&c);
    assert_eq!(r, Err(ProviderError::Retryable { msg: "HTTP 429: rate limited".into(), retry_after: Some(Duration::from_secs(3)) }));
    assert!(matches!(ask(&c).0, Err(ProviderError::Retryable { retry_after: None, .. })));
    match ask(&c).0 {
        Err(ProviderError::Fatal { status: Some(401), msg }) => {
            assert!(msg.starts_with("key rejected") && !msg.contains("SECRET"), "{msg}");
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(ask(&c).0, Err(ProviderError::Fatal { status: Some(400), msg }) if msg.contains("unknown model")));
}

#[test]
fn cut_streams_silence_and_refused_connections_are_retryable() {
    let cut = sse(vec![json!({"choices": [{"delta": {"content": "Hal"}}]}).to_string()], false);
    let silent: Box<dyn FnOnce(&mut TcpStream) + Send> = Box::new(|s| {
        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
        let _ = s.flush();
        std::thread::sleep(Duration::from_millis(1500));
    });
    let midstream = sse(vec![json!({"error": {"message": "overloaded"}}).to_string()], false);
    // The silent server goes last: it keeps its thread asleep, so a response queued after it would arrive late.
    let (base, _) = serve(vec![cut, midstream, silent]);
    let c = client(&base, 300);
    let (r, streamed) = ask(&c);
    assert_eq!(streamed, "Hal", "partial text was streamed before the cut");
    assert!(matches!(r, Err(ProviderError::Retryable { msg, .. }) if msg.contains("cut off")));
    assert!(matches!(ask(&c).0, Err(ProviderError::Retryable { msg, .. }) if msg.contains("overloaded")));
    assert!(matches!(ask(&c).0, Err(ProviderError::Retryable { msg, .. }) if msg.starts_with("network")));
    // Nothing listens on this port any more.
    let closed = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    let (r, _) = ask(&client(&format!("http://{closed}/v1"), 300));
    assert!(matches!(r, Err(ProviderError::Retryable { .. })));
}

#[test]
fn a_whole_turn_survives_a_429_and_a_cut_stream() {
    let answer = sse(
        vec![json!({"choices": [{"delta": {"content": "Ash Blossom negates searches."}, "finish_reason": "stop"}]}).to_string()],
        true,
    );
    let cut = sse(vec![json!({"choices": [{"delta": {"content": "Ash"}}]}).to_string()], false);
    let (base, seen) = serve(vec![status(429, "Retry-After: 1\r\n", "{}"), cut, answer]);
    let c = client(&base, 2000);
    let history = vec![json!({"role": "user", "content": "ash?"})];
    let (mut events, mut slept) = (Vec::new(), Vec::new());
    let out = run_turn(&c, &mut Fake::default(), "m", &history, &Limits::default(), &AtomicBool::new(false),
        &mut |e| events.push(e), &mut |d, _| {
            slept.push(d);
            true
        })
    .unwrap();
    assert_eq!(out.answer, "Ash Blossom negates searches.");
    assert_eq!(slept, vec![Duration::from_secs(1), Duration::from_secs(2)]);
    let retries = events.iter().filter(|e| matches!(e, Event::Retry(_))).count();
    assert_eq!((retries, seen.lock().unwrap().len()), (2, 3));
}

#[test]
fn lists_models_for_the_picker() {
    let (base, seen) = serve(vec![status(200, "", r#"{"data":[{"id":"kimi-k2"},{"id":"deepseek-v4.1-flash"}]}"#)]);
    assert_eq!(client(&base, 2000).models().unwrap(), vec!["deepseek-v4.1-flash", "kimi-k2"]);
    assert!(seen.lock().unwrap()[0].starts_with("GET /v1/models"));
}

// ---- review 2026-11-14 cases ----

#[test]
fn refused_calls_show_start_and_done() {
    let p = Scripted::new(vec![calls(&[("echo", "{}"), ("echo", "{}")]), text("ok")]);
    let (r, events, _) = turn(&p, &mut Fake::default(), &Limits { max_tool_calls: 1, ..Limits::default() });
    r.unwrap();
    let starts = events.iter().filter(|e| matches!(e, Event::ToolStart { .. })).count();
    let dones = events.iter().filter(|e| matches!(e, Event::ToolDone { .. })).count();
    assert_eq!((starts, dones), (2, 2));
}

#[test]
fn a_model_that_keeps_calling_after_the_limit_is_asked_once_more() {
    let p = Scripted::new(vec![calls(&[("echo", "{}")]), calls(&[("echo", "{}")]), text("Answer.")]);
    let (r, _, _) = turn(&p, &mut Fake::default(), &Limits { max_tool_calls: 1, ..Limits::default() });
    assert_eq!(r.unwrap().answer, "Answer.");
    let reqs = p.requests();
    assert_eq!(reqs[2].messages.last().unwrap()["role"], "user", "the one-off nudge");
    let p = Scripted::new(vec![calls(&[("echo", "{}")]), calls(&[("echo", "{}")]), calls(&[("echo", "{}")])]);
    let (r, _, _) = turn(&p, &mut Fake::default(), &Limits { max_tool_calls: 1, ..Limits::default() });
    assert!(r.unwrap_err().error == ProviderError::Fatal { status: None, msg: "the model returned an empty answer".into() });
}

#[test]
fn reasoning_is_sent_back_with_the_tool_round() {
    let mut r = calls(&[("echo", "{}")]).unwrap();
    r.reasoning = "I should look it up.".into();
    let p = Scripted::new(vec![Ok(r), text("ok")]);
    let (out, _, _) = turn(&p, &mut Fake::default(), &Limits::default());
    assert_eq!(out.unwrap().messages[0]["reasoning_content"], "I should look it up.");
    assert_eq!(p.requests()[1].messages[2]["reasoning_content"], "I should look it up.");
}

#[test]
fn cancel_before_a_round_or_a_tool_stops() {
    struct CancelOnCall<'a>(&'a AtomicBool);
    impl Tools for CancelOnCall<'_> {
        fn specs(&self) -> Vec<Value> {
            Fake::default().specs()
        }
        fn call(&mut self, _: &str, _: &Value, _: &AtomicBool) -> Result<Value, String> {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(json!({}))
        }
    }
    let cancel = AtomicBool::new(false);
    let p = Scripted::new(vec![calls(&[("echo", "{}"), ("echo", "{}")])]);
    let history = vec![json!({"role": "user", "content": "q"})];
    let r = run_turn(&p, &mut CancelOnCall(&cancel), "m", &history, &Limits::default(), &cancel, &mut |_| {}, &mut |_, _| true);
    let e = r.unwrap_err();
    assert_eq!(e.error, ProviderError::Cancelled);
    assert_eq!(e.partial.len(), 2, "the assistant call and the one result before the cancel");
    assert_eq!(p.requests().len(), 1);
}

// A slow tool with its own worker threads: Stop mid-call ends the turn at once, the workers are joined before the
// call returns (nothing left running), the later calls never start, and the call is stored with a result.
#[test]
fn stop_during_a_slow_tool_joins_its_work_and_ends_the_turn() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Slow {
        live: Arc<AtomicUsize>,
        calls: usize,
    }
    impl Tools for Slow {
        fn specs(&self) -> Vec<Value> {
            Fake::default().specs()
        }
        fn call(&mut self, _: &str, _: &Value, cancel: &AtomicBool) -> Result<Value, String> {
            self.calls += 1;
            std::thread::scope(|s| {
                for _ in 0..4 {
                    let live = Arc::clone(&self.live);
                    s.spawn(move || {
                        live.fetch_add(1, Ordering::SeqCst);
                        let t0 = std::time::Instant::now();
                        while !cancel.load(Ordering::Relaxed) && t0.elapsed() < Duration::from_secs(30) {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        live.fetch_sub(1, Ordering::SeqCst);
                    });
                }
            });
            if cancel.load(Ordering::Relaxed) { Err("stopped".into()) } else { Ok(json!({})) }
        }
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let live = Arc::new(AtomicUsize::new(0));
    let mut tools = Slow { live: Arc::clone(&live), calls: 0 };
    let p = Scripted::new(vec![calls(&[("echo", "{}"), ("echo", "{}")])]);
    let history = vec![json!({"role": "user", "content": "q"})];
    let stopper = {
        let c = Arc::clone(&cancel);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            c.store(true, Ordering::SeqCst);
        })
    };
    let t0 = std::time::Instant::now();
    let mut events = Vec::new();
    let r = run_turn(&p, &mut tools, "m", &history, &Limits::default(), &cancel, &mut |e| events.push(e), &mut |_, _| true);
    stopper.join().unwrap();
    assert!(t0.elapsed() < Duration::from_secs(2), "stopped in {:?}", t0.elapsed());
    assert_eq!(r.unwrap_err().error, ProviderError::Cancelled);
    assert_eq!(live.load(Ordering::SeqCst), 0, "no worker outlives the call");
    assert_eq!(tools.calls, 1, "the second call never starts");
    assert!(events.iter().any(|e| matches!(e, Event::ToolDone { ok: false, .. })), "the stopped call is reported");
}

#[test]
fn the_default_limit_matches_the_tool_config() {
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/ask/tools.v1.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg["limits"]["tool_calls_per_answer"], Limits::default().max_tool_calls);
    assert_eq!(super::retry::Policy::default().delay(0, None), std::time::Duration::from_secs(1), "no underflow at 0");
}

#[test]
fn sse_events_span_lines_and_skip_comments() {
    use super::provider::SseLines;
    let mut sse = SseLines::default();
    let lines = [": ping", "data:{\"a\":", "data: 1}\r", "", "event: x", "data: [DONE]"];
    let got: Vec<String> = lines.iter().filter_map(|l| sse.line(l)).collect();
    assert_eq!(got, vec!["{\"a\":\n1}"]);
    assert_eq!(sse.end().as_deref(), Some("[DONE]"), "an event left open at the end of the stream");
}

#[test]
fn stream_folding_edge_cases() {
    use super::provider::StreamFold;
    let mut f = StreamFold::default();
    let chunk = |v: Value| v.to_string();
    f.push(&chunk(json!({"choices": [{"delta": {"tool_calls": [
        {"index": 1, "id": "b", "function": {"name": "echo", "arguments": "{}"}},
        {"index": 0, "id": "a", "function": {"name": "echo", "arguments": "{\"x\""}}]}}]}))).unwrap();
    // The name repeated in a later piece must not become "echoecho".
    f.push(&chunk(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "echo", "arguments": ":1}"}}]}}]}))).unwrap();
    f.push(&chunk(json!({"choices": [{"delta": {"tool_calls": [{"index": 3, "function": {"arguments": ""}}]}, "finish_reason": "tool_calls"}]}))).unwrap();
    let r = f.finish().unwrap();
    assert_eq!(r.tool_calls.iter().map(|c| (c.id.as_str(), c.name.as_str(), c.arguments.as_str())).collect::<Vec<_>>(),
               vec![("a", "echo", "{\"x\":1}"), ("b", "echo", "{}")], "index gaps leave no phantom calls");
    let mut f = StreamFold::default();
    let huge = chunk(json!({"choices": [{"delta": {"tool_calls": [{"index": 1_000_000_000u64, "function": {"name": "x"}}]}}]}));
    assert!(matches!(f.push(&huge), Err(ProviderError::Retryable { msg, .. }) if msg.contains("index")));
    // finish_reason without [DONE] ends a stream properly.
    let mut f = StreamFold::default();
    f.push(&chunk(json!({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}))).unwrap();
    assert_eq!(f.finish().unwrap().text, "hi");
}

#[test]
fn keys_and_bearer_tokens_are_redacted() {
    use super::provider::redact;
    assert_eq!(redact("echo: Authorization: Bearer sk-test-SECRET-1a2b", KEY), "echo: Authorization: Bearer [key]");
    assert_eq!(redact("{\"auth\":\"Bearer other-token\"} and sk-test-SECRET-1a2b", KEY), "{\"auth\":\"Bearer [key]\"} and [key]");
}

#[test]
fn absurd_retry_after_and_echoed_keys_from_a_server() {
    let (base, _) = serve(vec![
        status(429, "Retry-After: inf\r\n", "{}"),
        status(503, "Retry-After: 1e300\r\n", "{}"),
        status(500, "", r#"{"error":{"message":"bad header Authorization: Bearer sk-test-SECRET-1a2b"}}"#),
    ]);
    let c = client(&base, 2000);
    assert!(matches!(ask(&c).0, Err(ProviderError::Retryable { retry_after: None, .. })));
    assert!(matches!(ask(&c).0, Err(ProviderError::Retryable { retry_after: Some(d), .. }) if d == Duration::from_secs(86_400)));
    match ask(&c).0 {
        Err(ProviderError::Retryable { msg, .. }) => assert!(!msg.contains("SECRET") && msg.contains("[key]"), "{msg}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_answer_request_sends_tool_choice_none() {
    let answer = sse(vec![json!({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}).to_string()], true);
    let (base, seen) = serve(vec![answer]);
    let req = Request { model: "m".into(), messages: vec![], tools: Some(vec![json!({})]), no_tool_calls: true };
    client(&base, 2000).complete(&req, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert!(seen.lock().unwrap()[0].contains("\"tool_choice\":\"none\""));
}

// ---- turns over the store ----

mod turns {
    use super::*;
    use crate::ask::store::{ChatStore, Policy};
    use crate::ask::turn::{Failed, Input, Setup, complete_calls, explain, send};

    fn store() -> (ChatStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("decklight-turn-{}", decklight_record::uuid7()));
        (ChatStore::open(&dir).unwrap(), dir)
    }

    fn setup(p: &dyn Provider) -> Setup<'_> {
        Setup {
            provider: p,
            provider_name: "OpenCode Go",
            system_prompt: "Use tools. At most {max_tool_calls} tools.",
            versions: json!({"prompt": "system.v1", "tools": "tools.v1"}),
            limits: Limits::default(),
            policy: Policy::default(),
            cards: &|t| {
                if t.contains("[[Ash]]") {
                    json!({"cards": {"Ash": 14558127}, "unresolved": ["Nib"]})
                } else {
                    json!({"cards": {}, "unresolved": []})
                }
            },
            images: std::path::Path::new("/nonexistent"),
        }
    }

    fn go(s: &ChatStore, chat: &str, input: Input, setup: &Setup) -> Result<crate::ask::turn::Sent, Failed> {
        let mut tools = Fake::default();
        send(s, chat, input, setup, &mut tools, &AtomicBool::new(false), &mut |_| {}, &mut |_, _| true)
    }

    #[test]
    fn a_turn_stores_the_question_the_tool_round_and_the_answer() {
        let (s, dir) = store();
        let c = s.create("opencode-go", "deepseek-v4.1-flash").unwrap();
        let p = Scripted::new(vec![calls(&[("echo", "{\"a\":1}")]), text("Done.")]);
        let sent = go(&s, &c.id, Input::New { parent: None, text: "  echo this  ", attach: &[], meta: &json!({}) }, &setup(&p)).unwrap();
        assert_eq!(sent.answer, "Done.");
        let b = s.branch(&c.id, None).unwrap();
        assert_eq!(b.iter().map(|r| r.role.as_str()).collect::<Vec<_>>(), vec!["user", "assistant", "tool", "assistant"]);
        assert_eq!(b[0].body["content"], "echo this");
        assert_eq!(b[3].meta["versions"]["prompt"], "system.v1");
        assert_eq!(b[3].meta["tokens"], json!({"prompt": 20, "completion": 3}));
        assert_eq!(s.get(&c.id).unwrap().tokens(), 23);
        assert!(b[3].meta.get("cards").is_none() && b[3].meta.get("unresolved").is_none(), "no [[cards]], no map");
        let sys = &p.requests()[0].messages[0];
        assert_eq!(sys["content"], "Use tools. At most 8 tools.");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_cards_an_answer_names_are_stored_with_it() {
        let (s, dir) = store();
        let c = s.create("opencode-go", "deepseek-v4.1-flash").unwrap();
        let p = Scripted::new(vec![text("Run 3 [[Ash]].")]);
        go(&s, &c.id, Input::New { parent: None, text: "q", attach: &[], meta: &json!({}) }, &setup(&p)).unwrap();
        let b = s.branch(&c.id, None).unwrap();
        assert_eq!(b[1].meta["cards"], json!({"Ash": 14558127}));
        assert_eq!(b[1].meta["unresolved"], json!(["Nib"]));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_failure_keeps_the_round_says_what_to_do_and_retry_answers_again() {
        let (s, dir) = store();
        let c = s.create("opencode-go", "m").unwrap();
        let rejected = Err(ProviderError::Fatal { status: Some(401), msg: "key rejected (HTTP 401): expired".into() });
        let p = Scripted::new(vec![calls(&[("echo", "{}")]), rejected]);
        let f = go(&s, &c.id, Input::New { parent: None, text: "hi", attach: &[], meta: &json!({}) }, &setup(&p)).unwrap_err();
        assert_eq!(f.action, "replace_key");
        assert!(f.message.starts_with("OpenCode Go rejected the key"), "{}", f.message);
        let b = s.branch(&c.id, None).unwrap();
        assert_eq!(b.iter().map(|r| r.role.as_str()).collect::<Vec<_>>(), vec!["user", "assistant", "tool", "error"]);
        assert_eq!(s.get(&c.id).unwrap().tokens(), 11, "tokens of the failed turn count");
        // Retry: a new answer under the same question; the error row is never sent to the model.
        let p2 = Scripted::new(vec![text("Hello.")]);
        let user = f.user.clone().unwrap();
        go(&s, &c.id, Input::Again { user: &user }, &setup(&p2)).unwrap();
        let sent_roles: Vec<String> =
            p2.requests()[0].messages.iter().map(|m| m["role"].as_str().unwrap().to_string()).collect();
        assert_eq!(sent_roles, vec!["system", "user"]);
        let b = s.branch(&c.id, None).unwrap();
        assert_eq!(b.iter().map(|r| r.role.as_str()).collect::<Vec<_>>(), vec!["user", "assistant"]);
        // A follow-up after an error row: the error row is skipped, the history stays valid.
        s.set_head(&c.id, Some(&s.rows(&c.id).unwrap().iter().find(|r| r.role == "error").unwrap().id)).unwrap();
        let p3 = Scripted::new(vec![text("ok")]);
        go(&s, &c.id, Input::New { parent: None, text: "again?", attach: &[], meta: &json!({}) }, &setup(&p3)).unwrap();
        let roles: Vec<String> = p3.requests()[0].messages.iter().map(|m| m["role"].as_str().unwrap().to_string()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "user"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_token_cap_stops_a_chat_before_it_spends_more() {
        let (s, dir) = store();
        let c = s.create("opencode-go", "m").unwrap();
        s.add_usage(&c.id, 299_000, 1_000).unwrap();
        let p = Scripted::new(vec![]);
        let f = go(&s, &c.id, Input::New { parent: None, text: "hi", attach: &[], meta: &json!({}) }, &setup(&p)).unwrap_err();
        assert_eq!(f.action, "new_chat");
        assert!(f.message.contains("300000 of its 300000"), "{}", f.message);
        assert!(s.rows(&c.id).unwrap().is_empty() && p.requests().is_empty(), "nothing stored, nothing sent");
        assert_eq!(go(&s, &c.id, Input::Again { user: "nope" }, &setup(&p)).unwrap_err().action, "new_chat");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn empty_questions_and_answering_a_non_question_are_refused() {
        let (s, dir) = store();
        let c = s.create("opencode-go", "m").unwrap();
        let p = Scripted::new(vec![text("Hi.")]);
        assert_eq!(go(&s, &c.id, Input::New { parent: None, text: "   ", attach: &[], meta: &json!({}) }, &setup(&p)).unwrap_err().message, "Type a question first.");
        let sent = go(&s, &c.id, Input::New { parent: None, text: "hi", attach: &[], meta: &json!({}) }, &setup(&p)).unwrap();
        let f = go(&s, &c.id, Input::Again { user: &sent.ids[0] }, &setup(&p)).unwrap_err();
        assert!(f.message.contains("Only a question"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn errors_map_to_an_action() {
        let w = Duration::from_secs(20);
        let quota = ProviderError::Retryable { msg: "HTTP 429: Free usage limit reached".into(), retry_after: Some(Duration::from_secs(23713)) };
        let (m, a) = explain(&quota, "OpenCode Zen", w);
        assert_eq!(a, "switch_provider");
        assert!(m.contains("try again in 7 h"), "{m}");
        assert_eq!(explain(&ProviderError::Retryable { msg: "HTTP 503".into(), retry_after: None }, "X", w).1, "retry");
        let dns = "error sending request: Dns Failed: resolve dns name 'opencode.ai:443': failed to lookup address information";
        let off = explain(&ProviderError::Retryable { msg: dns.into(), retry_after: None }, "OpenCode Go", w);
        assert!(off.0.starts_with("Could not reach OpenCode Go: no internet connection"), "{}", off.0);
        assert_eq!(off.1, "retry");
        assert_eq!(explain(&ProviderError::Fatal { status: Some(404), msg: "no such model".into() }, "X", w).1, "pick_model");
        let img = explain(&ProviderError::Fatal { status: Some(400), msg: "this model does not support image input".into() }, "X", w);
        assert!(img.0.contains("cannot read photos") && img.1 == "pick_model", "{img:?}");
        assert_eq!(explain(&ProviderError::Fatal { status: Some(400), msg: "Model not found".into() }, "X", w).1, "pick_model");
        assert_eq!(explain(&ProviderError::Fatal { status: Some(400), msg: "missing header".into() }, "X", w).1, "check_settings");
        assert_eq!(explain(&ProviderError::Cancelled, "X", w).1, "none");
    }

    #[test]
    fn calls_cut_by_a_cancel_get_a_not_run_result() {
        let mut msgs = vec![
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "a"}, {"id": "b"}]}),
            json!({"role": "tool", "tool_call_id": "a", "content": "{}"}),
        ];
        complete_calls(&mut msgs);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2]["tool_call_id"], "b");
        assert!(msgs[2]["content"].as_str().unwrap().contains("not run"));
        complete_calls(&mut msgs);
        assert_eq!(msgs.len(), 3, "idempotent");
    }

    #[test]
    fn a_question_sent_under_a_tool_call_message_still_makes_a_valid_history() {
        // Fork to an assistant message with calls, then ask: the question would follow the calls directly.
        let mut msgs = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "a"}]}),
            json!({"role": "user", "content": "q2"}),
        ];
        complete_calls(&mut msgs);
        let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "user"]);
        assert_eq!(msgs[2]["tool_call_id"], "a");
    }

    #[test]
    fn a_too_long_chat_asks_for_a_new_chat_not_another_model() {
        let w = Duration::from_secs(20);
        let e = ProviderError::Fatal {
            status: Some(400),
            msg: "This model's maximum context length is 32768 tokens".into(),
        };
        assert_eq!(explain(&e, "X", w).1, "new_chat");
    }
}

// Stop reaches a provider that took the request and then went silent (no headers, then headers and no chunks):
// the turn ends in well under the 120 s read timeout.
#[test]
fn stop_answers_while_the_provider_is_silent() {
    for headers_first in [false, true] {
        let (base, _) = serve(vec![Box::new(move |s: &mut TcpStream| {
            if headers_first {
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
            }
            std::thread::sleep(Duration::from_secs(5));
        })]);
        let p = OpenAiCompatible::new(&base, "k", Duration::from_secs(5), Duration::from_secs(120));
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let c = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            c.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        let t0 = std::time::Instant::now();
        let req = Request { model: "m".into(), messages: vec![json!({"role": "user", "content": "q"})], ..Request::default() };
        let r = p.complete(&req, &mut |_| {}, &cancel);
        assert_eq!(r.unwrap_err(), ProviderError::Cancelled);
        assert!(t0.elapsed() < Duration::from_secs(2), "headers_first={headers_first}: stopped after {:?}", t0.elapsed());
    }
}
