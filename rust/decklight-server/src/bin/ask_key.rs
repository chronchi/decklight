//! Ask API keys in this machine's credential store (src/ask/keys.rs), for the desktop server and the eval runner.
//! The key is never printed: `set` reads it with the terminal echo off (or from a pipe), the rest show hints only.
//!
//!   ask_key status
//!   ask_key set <preset>                       paste the key, then Enter
//!   ask_key import-opencode <preset>           copy opencode's key for opencode-go/-zen (auth.json) without showing it
//!   ask_key delete <preset>
//!
//! Presets: config/ask/providers.v1.json. Exits 1 on any failure, with what to do.

use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

use decklight_server::ask::keys::ApiKeys;
use decklight_server::ask::presets::presets;
use serde_json::Value;

/// One line from stdin, without echo when stdin is a terminal.
fn read_secret(prompt: &str) -> Result<String, String> {
    let stdin = std::io::stdin();
    let tty = stdin.is_terminal();
    if tty {
        eprint!("{prompt}");
        let _ = std::io::stderr().flush();
    }
    let mut line = String::new();
    #[cfg(unix)]
    let read = {
        let fd = 0;
        // SAFETY: plain termios calls on stdin; the old mode is restored before returning.
        let mut old: libc::termios = unsafe { std::mem::zeroed() };
        let echo_off = tty && unsafe { libc::tcgetattr(fd, &mut old) } == 0 && {
            let mut quiet = old;
            quiet.c_lflag &= !libc::ECHO;
            unsafe { libc::tcsetattr(fd, libc::TCSANOW, &quiet) == 0 }
        };
        let read = stdin.lock().read_line(&mut line);
        if echo_off {
            unsafe { libc::tcsetattr(fd, libc::TCSANOW, &old) };
            eprintln!();
        }
        read
    };
    // Elsewhere (Windows) the key shows while typed; the desktop app's Settings field is the usual way in.
    #[cfg(not(unix))]
    let read = stdin.lock().read_line(&mut line);
    read.map_err(|e| format!("could not read the key: {e}"))?;
    Ok(line)
}

fn opencode_key(name: &str) -> Result<String, String> {
    let path = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share/opencode/auth.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let auth: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    auth[name]["key"]
        .as_str()
        .filter(|k| !k.is_empty())
        .map(str::to_string)
        .ok_or(format!("no key {name:?} in {} (run `opencode auth login`)", path.display()))
}

/// Which opencode auth.json entry holds each OpenCode preset's key (a key is never moved to another provider).
fn opencode_name(preset: &str) -> Result<&'static str, String> {
    match preset {
        "opencode-zen" => Ok("opencode"),
        "opencode-go" => Ok("opencode-go"),
        other => Err(format!("import-opencode works for opencode-go and opencode-zen, not {other:?}")),
    }
}

/// A key saved where it is lost on exit would make the next command fail: say so and save nothing.
fn persistent(keys: &ApiKeys) -> Result<(), String> {
    let store = keys.store();
    if store.persistent() {
        Ok(())
    } else {
        Err(format!("{} keeps keys only while this command runs; start the desktop keyring (unlock it) and retry", store.kind()))
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let keys = ApiKeys::default();
    let preset_arg = |i: usize| args.get(i).cloned().ok_or("give a preset: opencode-go, opencode-zen, openrouter, custom");
    match args.get(1).map(String::as_str) {
        Some("status") | None => {
            let store = keys.store();
            println!("store: {}{}", store.kind(), if store.persistent() { "" } else { " (keys are lost when this stops)" });
            for p in presets() {
                let state = keys.status(&p.id)?.unwrap_or_else(|| "no key".into());
                println!("  {:<13} {state}", p.id);
            }
        }
        Some("set") => {
            let preset = preset_arg(2)?;
            persistent(&keys)?;
            let key = read_secret(&format!("{preset} key (not shown): "))?;
            println!("{preset}: saved {}", keys.set(&preset, &key)?);
        }
        Some("import-opencode") => {
            let preset = preset_arg(2)?;
            let name = opencode_name(&preset)?;
            persistent(&keys)?;
            println!("{preset}: saved {}", keys.set(&preset, &opencode_key(name)?)?);
        }
        Some("delete") => {
            let preset = preset_arg(2)?;
            keys.delete(&preset)?;
            println!("{preset}: deleted");
        }
        Some(other) => return Err(format!("unknown command {other:?}: status, set, import-opencode, delete")),
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args) {
        eprintln!("ask_key: {e}");
        std::process::exit(1);
    }
}
