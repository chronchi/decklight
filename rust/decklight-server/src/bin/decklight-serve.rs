//! `decklight-serve`: the web client's server (A2c).

use std::path::PathBuf;

use decklight_server::{Config, run};

fn die(msg: &str) -> ! {
    eprintln!("decklight-serve: {msg}");
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut c = Config::defaults(&root);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut decks: Vec<PathBuf> = Vec::new();
    let mut i = 0usize;
    while i < argv.len() {
        let a = argv[i].clone();
        i += 1;
        let mut val = || {
            let v = argv.get(i).cloned().unwrap_or_else(|| die(&format!("{a} needs a value")));
            i += 1;
            v
        };
        match a.as_str() {
            "--port" => c.port = val().parse().unwrap_or_else(|_| die("--port")),
            "--token" => c.token = val(),
            "--web-dir" => c.web_dir = PathBuf::from(val()),
            "--decks-dir" => decks.push(PathBuf::from(val())),
            "--onnx" => c.onnx_dir = PathBuf::from(val()),
            "--records" => c.records_dir = PathBuf::from(val()),
            "--pics-dir" => c.pics_dir = PathBuf::from(val()),
            "--ask-dir" => c.ask_dir = PathBuf::from(val()),
            "--fixed-seed" => c.fixed_seed = Some(val().parse().unwrap_or_else(|_| die("--fixed-seed"))),
            "--test-opponent" => c.test_opponent = Some(val()),
            "--default-opponent" => c.default_opponent = val(),
            "--ort-threads" => c.ort_threads = val().parse().unwrap_or_else(|_| die("--ort-threads")),
            "--replay-threads" => c.replay_threads = val().parse().unwrap_or_else(|_| die("--replay-threads")),
            "--packs-dir" => c.packs_dir = PathBuf::from(val()),
            "--pack-url" => c.pack_url = Some(val()),
            "--root" => c.root = PathBuf::from(val()),
            "--help" | "-h" => {
                println!(
                    "decklight-serve [--port N] [--token T] [--web-dir D] [--decks-dir D]... \
                     [--onnx D] [--records D] [--pics-dir D] [--ask-dir D] [--fixed-seed N] [--test-opponent stream] \
                     [--replay-threads N] [--ort-threads N] [--packs-dir D] [--pack-url URL]"
                );
                return;
            }
            _ => die(&format!("unknown argument {a}")),
        }
    }
    if !decks.is_empty() {
        c.decks_dirs = Some(decks);
    }
    if let Err(e) = run(c).await {
        eprintln!("ALERT decklight-serve: {e}");
        std::process::exit(1);
    }
}