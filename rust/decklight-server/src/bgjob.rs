//! The hand-off to Android's background download job (MainActivity's DownloadJob, AGENTS.md "Downloads keep going
//! without the app"). A download writes what it is doing to `<root>/downloads/status.json`; the job keeps the app's
//! process alive while the file exists, shows it in one system notification and ends when the file is removed. The
//! job also writes `<root>/downloads/network` ("wifi" | "cellular") from Android's network callback, so Wi-Fi only
//! keeps working after the app's screen (and with it the web client's network messages) is gone.
//!
//! An Ask answer uses the same job under `<root>/answer/` (user, 2026-11-16: "if we just ... go to another app while we
//! wait for the answer, we can come back and finish the answer just like we do with Claude, ChatGPT"): Android cuts a
//! backgrounded app's network, which failed an answer with a DNS error. `status.json` lives while the answer runs;
//! `done.json` says how it ended, so the job can post "Answer ready" when the app is not on screen.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

fn dir(root: &Path) -> PathBuf {
    root.join("downloads")
}

/// One running download: `state` is "downloading" or "waiting_wifi".
pub fn write(root: &Path, title: &str, bytes: u64, total: u64, state: &str) {
    let d = dir(root);
    let _ = std::fs::create_dir_all(&d);
    let tmp = d.join("status.json.tmp");
    let body = json!({"title": title, "bytes": bytes, "total": total, "state": state});
    // Written whole then renamed: the job never reads half a file.
    if std::fs::write(&tmp, body.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, d.join("status.json"));
    }
}

/// No download running: the job ends and its notification goes.
pub fn clear(root: &Path) {
    let _ = std::fs::remove_file(dir(root).join("status.json"));
}

/// An Ask answer started: `title` is the question, cut to a notification's line.
pub fn answer_start(root: &Path, title: &str) {
    let d = root.join("answer");
    let _ = std::fs::create_dir_all(&d);
    let _ = std::fs::remove_file(d.join("done.json"));
    let title: String = title.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(80).collect();
    let tmp = d.join("status.json.tmp");
    if std::fs::write(&tmp, json!({"title": title, "state": "answering"}).to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, d.join("status.json"));
    }
}

/// The answer ended: `ok` false for an error (the notification then says it failed and to open the app).
pub fn answer_end(root: &Path, ok: bool) {
    let d = root.join("answer");
    let title = std::fs::read_to_string(d.join("status.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v["title"].as_str().map(str::to_string))
        .unwrap_or_default();
    let _ = std::fs::write(d.join("done.json"), json!({"title": title, "ok": ok}).to_string());
    let _ = std::fs::remove_file(d.join("status.json"));
}

/// The job's view of the network: `Some(true)` on cellular (or a metered Wi-Fi), `None` when no job runs.
pub fn cellular(root: &Path) -> Option<bool> {
    std::fs::read_to_string(dir(root).join("network")).ok().map(|s| s.trim() == "cellular")
}

/// Mirrors a `device_status` frame: the model downloading (or waiting for Wi-Fi), else nothing.
pub fn from_device_status(root: &Path, status: &Value) {
    let busy = status["models"].as_array().and_then(|ms| {
        ms.iter().find(|m| matches!(m["state"].as_str(), Some("downloading" | "waiting_wifi")))
    });
    match busy {
        Some(m) => write(
            root,
            m["name"].as_str().unwrap_or("a model"),
            m["bytes"].as_u64().unwrap_or(0),
            m["total"].as_u64().unwrap_or(0),
            m["state"].as_str().unwrap_or("downloading"),
        ),
        None => clear(root),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_follows_the_busy_model_and_clears_when_none() {
        let root = std::env::temp_dir().join(format!("decklight-bgjob-{}", std::process::id()));
        let busy = json!({"models": [{"name": "Gemma 4 E4B", "state": "waiting_wifi", "bytes": 5, "total": 9}]});
        from_device_status(&root, &busy);
        let s: Value = serde_json::from_str(&std::fs::read_to_string(root.join("downloads/status.json")).unwrap()).unwrap();
        assert_eq!(s, json!({"title": "Gemma 4 E4B", "bytes": 5, "total": 9, "state": "waiting_wifi"}));
        assert_eq!(cellular(&root), None);
        std::fs::write(root.join("downloads/network"), "cellular\n").unwrap();
        assert_eq!(cellular(&root), Some(true));
        from_device_status(&root, &json!({"models": [{"name": "Gemma 4 E4B", "state": "installed"}]}));
        assert!(!root.join("downloads/status.json").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_answer_runs_then_leaves_how_it_ended() {
        let root = std::env::temp_dir().join(format!("decklight-bgjob-answer-{}", std::process::id()));
        answer_start(&root, "How do I   do going\nsecond?");
        let s: Value = serde_json::from_str(&std::fs::read_to_string(root.join("answer/status.json")).unwrap()).unwrap();
        assert_eq!(s, json!({"title": "How do I do going second?", "state": "answering"}));
        answer_end(&root, false);
        assert!(!root.join("answer/status.json").exists());
        let d: Value = serde_json::from_str(&std::fs::read_to_string(root.join("answer/done.json")).unwrap()).unwrap();
        assert_eq!(d, json!({"title": "How do I do going second?", "ok": false}));
        answer_start(&root, "next");
        assert!(!root.join("answer/done.json").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
