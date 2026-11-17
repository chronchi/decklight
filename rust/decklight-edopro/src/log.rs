//! Packet log: one gzip-streamed JSONL line per frame sent or received, after framing.
//!
//! `{"dir":"s"|"r","p":proto,"x":"<payload hex>"}` (the same format `scripts/app/edopro_net_py.py` writes), so a
//! Rust process and a Python process can be compared line by line.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Serialize;

#[derive(Serialize)]
struct Line<'a> {
    dir: &'a str,
    p: u8,
    x: String,
}

pub struct PacketLog {
    enc: Option<GzEncoder<BufWriter<File>>>,
    n: u64,
}

impl PacketLog {
    /// No log (`--packet-log` not given).
    pub fn disabled() -> PacketLog {
        PacketLog { enc: None, n: 0 }
    }

    pub fn open(path: &Path) -> Result<PacketLog, String> {
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
            }
        }
        let f = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(PacketLog { enc: Some(GzEncoder::new(BufWriter::new(f), Compression::new(6))), n: 0 })
    }

    /// `dir` is "s" (sent) or "r" (received).
    pub fn log(&mut self, dir: &str, proto: u8, payload: &[u8]) {
        let Some(enc) = self.enc.as_mut() else { return };
        let line = Line { dir, p: proto, x: hex(payload) };
        // A log write must never take the duel down; failing to log is reported once by the caller.
        if serde_json::to_writer(&mut *enc, &line).is_ok() {
            let _ = enc.write_all(b"\n");
            self.n += 1;
            if self.n % 2048 == 0 {
                let _ = enc.flush();
            }
        }
    }

    pub fn finish(&mut self) -> Result<(), String> {
        match self.enc.take() {
            Some(enc) => {
                let mut w = enc.finish().map_err(|e| e.to_string())?;
                w.flush().map_err(|e| e.to_string())
            }
            None => Ok(()),
        }
    }
}

impl Drop for PacketLog {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

pub fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(char::from_digit((x >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((x & 15) as u32, 16).unwrap());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn writes_one_gzip_line_per_packet() {
        let dir = std::env::temp_dir().join(format!("decklight-plog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.jsonl.gz");
        {
            let mut log = PacketLog::open(&path).unwrap();
            log.log("s", 1, &[0xAA, 0xBB]);
            log.log("r", 0x12, &[]);
            log.finish().unwrap();
        }
        let mut text = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        assert_eq!(
            text,
            "{\"dir\":\"s\",\"p\":1,\"x\":\"aabb\"}\n{\"dir\":\"r\",\"p\":18,\"x\":\"\"}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
