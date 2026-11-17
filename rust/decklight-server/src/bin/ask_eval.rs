//! Ask eval runner (config/ask/README.md): each question of the eval set goes through the real chat loop with the
//! real tools, on fixture decks and a fixture game history, against a live OpenAI-compatible model. Writes one JSON
//! line per question and prints the tool and answer scores.
//!
//!   ask_eval [--preset opencode-go] [--model M] [--base-url URL] [--auth opencode | --key-env VAR]
//!            [--only s01,o01] [--eval config/ask/eval/v1.jsonl] [--out FILE]
//!
//! `--preset` (config/ask/providers.v1.json) gives the base URL, the model and the key from this machine's credential
//! store (`ask_key set <preset>`), as the app does; `--auth <name>` takes opencode's auth.json key instead, `--key-env`
//! a variable. Without `--preset`: OpenCode Zen, `deepseek-v4.1-flash-free`, `--auth opencode`. The key is never printed.
//! Exits 1 on a setup error (no question matches, no key), 2 when every answer is empty (a run that cannot be right),
//! 3 when the provider refuses for good (key rejected, unknown model, bad request): the run stops at the first one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use decklight_server::ask::chat::{self, Event, Limits};
use decklight_server::ask::eval::{self, Call};
use decklight_server::ask::host::{self, AppTools};
use decklight_server::ask::provider::{OpenAiCompatible, ProviderError};
use decklight_server::ask::retry;
use decklight_server::{Config, build_state};
use serde_json::{Value, json};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn key(args: &[String], preset: Option<&str>) -> Result<String, String> {
    if let Some(var) = arg(args, "--key-env") {
        return std::env::var(&var).map_err(|_| format!("{var} is not set"));
    }
    if let Some(p) = preset
        && arg(args, "--auth").is_none()
    {
        let saved = decklight_server::ask::keys::ApiKeys::default().get(p)?;
        // A local server (custom, needs_key false) runs without one.
        let needs = decklight_server::ask::presets::preset(p).is_none_or(|x| x.needs_key);
        return match saved {
            Some(k) => Ok(k),
            None if !needs => Ok(String::new()),
            None => Err(format!("no {p} key in this machine's credential store: run `ask_key set {p}`")),
        };
    }
    let provider = arg(args, "--auth").unwrap_or_else(|| "opencode".into());
    let path = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share/opencode/auth.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let auth: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    auth[&provider]["key"]
        .as_str()
        .filter(|k| !k.is_empty())
        .map(str::to_string)
        .ok_or(format!("no key for {provider:?} in {} (run `opencode auth login`)", path.display()))
}

/// A fresh fixture tree under `dir`: the starter decks and the records-small history index.
fn fixtures(root: &Path, dir: &Path) -> Result<(PathBuf, PathBuf), String> {
    let src = root.join("config/ask/eval/fixtures");
    let decks = dir.join("decks");
    let records = dir.join("records");
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(&decks).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&records).map_err(|e| e.to_string())?;
    for e in std::fs::read_dir(src.join("decks")).map_err(|e| e.to_string())? {
        let p = e.map_err(|e| e.to_string())?.path();
        std::fs::copy(&p, decks.join(p.file_name().unwrap())).map_err(|e| e.to_string())?;
    }
    let sql = std::fs::read_to_string(src.join("records-small.sql")).map_err(|e| e.to_string())?;
    let con = rusqlite::Connection::open(records.join("index.sqlite")).map_err(|e| e.to_string())?;
    con.execute_batch(&sql).map_err(|e| format!("records-small.sql: {e}"))?;
    Ok((decks, records))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("ask_eval: {e}");
            std::process::exit(1);
        }
    }
}

fn run(args: &[String]) -> Result<i32, String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    if !root.join("config/ask").is_dir() {
        return Err("run from ygo-zero/ (no config/ask here)".into());
    }
    let preset = match arg(args, "--preset") {
        Some(id) => Some(decklight_server::ask::presets::preset(&id).ok_or(format!("unknown preset {id:?}"))?),
        None => None,
    };
    let model = arg(args, "--model")
        .or(preset.as_ref().map(|p| p.default_model.clone()).filter(|m| !m.is_empty()))
        .unwrap_or_else(|| "deepseek-v4.1-flash-free".into());
    // A preset without a base URL (custom) needs --base-url: its key never goes to OpenCode's default (review 2026-11-14).
    let base = match (arg(args, "--base-url"), &preset) {
        (Some(b), _) => b,
        (None, Some(p)) if p.base_url.is_empty() => return Err(format!("--preset {} needs --base-url", p.id)),
        (None, Some(p)) => p.base_url.clone(),
        (None, None) => "https://opencode.ai/inference/openai/v1".into(),
    };
    let eval_path = root.join(arg(args, "--eval").unwrap_or_else(|| "config/ask/eval/v1.jsonl".into()));
    let only: Option<Vec<String>> =
        arg(args, "--only").map(|s| s.split(',').map(|id| id.trim().to_string()).filter(|id| !id.is_empty()).collect());
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let out_dir = root.join("data/cache/ask_eval");
    let out_path = arg(args, "--out").map(PathBuf::from).unwrap_or_else(|| out_dir.join(format!("{stamp}-{model}.jsonl")));

    let questions: Vec<Value> = std::fs::read_to_string(&eval_path)
        .map_err(|e| format!("{}: {e}", eval_path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("{}: {e}", eval_path.display())))
        .collect::<Result<_, _>>()?;
    let picked: Vec<&Value> =
        questions.iter().filter(|q| only.as_ref().is_none_or(|o| o.iter().any(|id| q["id"] == id.as_str()))).collect();
    if picked.is_empty() {
        return Err("no question matches --only".into());
    }

    let key = key(args, preset.as_ref().map(|p| p.id.as_str()))?;
    let (decks, records) = fixtures(&root, &out_dir.join(format!("fixture-{stamp}")))?;
    let mut config = Config::defaults(&root);
    config.user_decks_dir = decks;
    config.records_dir = records;
    // The app under test keeps keys in memory: the run reads its own key above and never writes the keychain.
    config.secrets = Some(Arc::new(decklight_server::ask::keys::SessionOnly::default()));
    eprintln!("ask_eval: building the app (cards, search index)");
    let app = build_state(&config)?;
    let names: BTreeMap<u32, String> = app.ctx.texts.names().clone();
    let index = app.records.root.join("index.sqlite");
    let stats = |a: &Value| decklight_server::ask::stats::my_stats(a, &index);

    let provider = OpenAiCompatible::new(&base, &key, Duration::from_secs(15), Duration::from_secs(120));
    let limits = Limits::default();
    let system = json!({"role": "system", "content": chat::system_prompt(host::SYSTEM_PROMPT, &limits)});
    let cancel = AtomicBool::new(false);
    std::fs::create_dir_all(out_path.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    let mut out = String::new();
    // Each question's conversation, for follow-ups (`context`).
    let mut chats: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut sessions: BTreeMap<String, String> = BTreeMap::new();
    let (mut tool_pass, mut answer_pass, mut empty) = (0, 0, 0);
    let (mut prompt_tok, mut completion_tok) = (0u64, 0u64);
    // `[[Name]]` marks (system.v2) and the ones that are not an exact card name (user 2026-11-14: measure first).
    let (mut marked, mut unmatched): (usize, Vec<String>) = (0, Vec::new());
    eprintln!("ask_eval: {} questions, model {model} at {base}", picked.len());

    for (i, q) in picked.iter().enumerate() {
        let id = q["id"].as_str().unwrap_or("?").to_string();
        let text = q["q"].as_str().unwrap_or("").to_string();
        let mut history = vec![system.clone()];
        let mut note = String::new();
        if let Some(ctx) = q["context"].as_str() {
            match chats.get(ctx) {
                Some(h) => history = h.clone(),
                None => note = format!("context {ctx} was not run; asked without it"),
            }
        }
        history.push(json!({"role": "user", "content": text}));
        // One session per conversation: a follow-up keeps the session of the question it follows.
        let session = q["context"]
            .as_str()
            .and_then(|c| chats.contains_key(c).then(|| sessions.get(c).cloned()).flatten())
            .unwrap_or_else(|| format!("decklight-eval-{stamp}-{id}"));
        sessions.insert(id.clone(), session.clone());
        // `answer.max_tool_calls` is the loop's limit for this question too.
        let mut limits = limits.clone();
        if let Some(n) = q["answer"]["max_tool_calls"].as_u64() {
            limits.max_tool_calls = n as u32;
        }
        // Earlier turns of this chat, for the invented-names check (the system prompt left out).
        let prior: String = history[1..history.len() - 1].iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        let provider = provider.for_session(&session);
        let mut tools = AppTools::new(Arc::clone(&app));
        let t = Instant::now();
        // From the events, so refusals and calls with bad JSON (which never reach the tools) are scored too.
        let mut calls: Vec<Call> = Vec::new();
        let mut started: BTreeMap<String, Value> = BTreeMap::new();
        let mut limit_reached = false;
        let mut retries = 0;
        let result = {
            let mut on_event = |e: Event| match e {
                Event::LimitReached => limit_reached = true,
                Event::ToolStart { id, args, .. } => {
                    started.insert(id, args);
                }
                Event::ToolDone { id, name, ok, result } => {
                    let args = started.remove(&id).unwrap_or(Value::Null);
                    // Calls past the limit are the ones refused.
                    let after_limit = calls.iter().filter(|c| !c.after_limit).count() >= limits.max_tool_calls as usize;
                    calls.push(Call { name, args, ok, result, after_limit });
                }
                Event::Retry(n) => {
                    retries += 1;
                    eprintln!("  {id}: retry {} in {:?}: {}", n.attempt, n.delay, n.reason);
                }
                _ => {}
            };
            chat::run_turn(&provider, &mut tools, &model, &history, &limits, &cancel, &mut on_event, &mut retry::sleep_cancellable)
        };
        let secs = t.elapsed().as_secs_f64();
        let (answer, error) = match &result {
            Ok(o) => {
                prompt_tok += o.usage.prompt;
                completion_tok += o.usage.completion;
                let mut h = history.clone();
                h.extend(o.messages.iter().cloned());
                chats.insert(id.clone(), h);
                (o.answer.clone(), None)
            }
            Err(e) => (String::new(), Some(format!("{:?}", e.error))),
        };
        if answer.trim().is_empty() {
            empty += 1;
        }
        let expected: Vec<Value> = q["tools"].as_array().cloned().unwrap_or_default();
        let (tp, tool_detail) = eval::tool_score(&expected, &calls);
        let fixture = eval::fixture_numbers(
            q["answer"]["numbers_from_fixture"].as_array().map(Vec::as_slice).unwrap_or(&[]),
            &expected,
            &stats,
        );
        let (ap, checks) = eval::answer_score(q, &text, &prior, &answer, &calls, &names, &fixture);
        let refs = host::card_refs(&app, &answer);
        let un: Vec<String> = refs["unresolved"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect();
        marked += refs["cards"].as_object().map_or(0, |o| o.len()) + un.len();
        unmatched.extend(un.iter().cloned());
        tool_pass += tp as usize;
        answer_pass += ap as usize;
        let failed: Vec<String> =
            checks.iter().filter(|c| c["pass"] != true).map(|c| c["check"].as_str().unwrap_or("").to_string()).collect();
        eprintln!(
            "[{}/{}] {id}: tools {} answer {} ({} calls, {secs:.1} s){}{}",
            i + 1,
            picked.len(),
            if tp { "ok" } else { "FAIL" },
            if ap { "ok" } else { "FAIL" },
            calls.len(),
            if tp { String::new() } else { format!(" | {tool_detail}") },
            if failed.is_empty() { String::new() } else { format!(" | failed: {}", failed.join("; ")) },
        );
        if let Some(e) = &error {
            eprintln!("  {id}: error: {}", e.chars().take(300).collect::<String>());
        }
        if !un.is_empty() {
            eprintln!("  {id}: card names that match no card: {un:?}");
        }
        let row = json!({
            "id": id, "q": text, "model": model, "base_url": base, "answer": answer, "error": error, "note": note,
            "calls": calls.iter().map(|c| json!({"name": c.name, "args": c.args, "ok": c.ok,
                "after_limit": c.after_limit, "result": c.result})).collect::<Vec<_>>(),
            "limit_reached": limit_reached, "retries": retries, "secs": secs,
            "tool_pass": tp, "tool_detail": tool_detail, "answer_pass": ap, "checks": checks, "fixture_numbers": fixture,
            "card_refs": refs,
            "usage": result.as_ref().map(|o| json!({"prompt": o.usage.prompt, "completion": o.usage.completion})).ok(),
        });
        out.push_str(&row.to_string());
        out.push('\n');
        std::fs::write(&out_path, &out).map_err(|e| format!("{}: {e}", out_path.display()))?;
        // Fail fast: a rejected key or unknown model fails every question the same way.
        if let Err(e) = &result
            && let ProviderError::Fatal { status: Some(_), msg } = &e.error
        {
            eprintln!("ALERT ask_eval: the provider refused ({msg}); stopping after {}/{}", i + 1, picked.len());
            return Ok(3);
        }
    }

    let n = picked.len();
    let pct = |k: usize| k as f64 * 100.0 / n as f64;
    println!(
        "ask_eval {model}: tools {tool_pass}/{n} ({:.0}%), answers {answer_pass}/{n} ({:.0}%), tokens {prompt_tok} in / {completion_tok} out -> {}",
        pct(tool_pass),
        pct(answer_pass),
        out_path.display()
    );
    println!("ask_eval {model}: card names marked {marked}, not an exact card name {} {unmatched:?}", unmatched.len());
    if empty == n {
        eprintln!("ALERT ask_eval: every answer is empty");
        return Ok(2);
    }
    Ok(0)
}
