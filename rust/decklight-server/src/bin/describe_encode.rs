//! Encode query texts with the fp16 Qwen3-Embedding model, for the parity check against the Python
//! reference (`scripts/search/describe_parity.py`).
//!
//!   describe_encode <model_dir> <text>...
//!
//! One JSON array (512 floats) per text on stdout. `model_dir` is the install dir holding `tokenizer.json`
//! and `onnx/model_fp16.onnx`; set `ORT_DYLIB_PATH` to the ONNX Runtime library.

use std::path::PathBuf;
use std::time::Instant;

use decklight_server::describe_enc::Encoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: describe_encode <model_dir> <text>...");
        std::process::exit(2);
    };
    let model_dir = PathBuf::from(dir);
    let t = Instant::now();
    let mut enc = match Encoder::load(&model_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ALERT describe_encode: {e}");
            std::process::exit(2);
        }
    };
    eprintln!("describe_encode: first load {} ms", t.elapsed().as_millis());
    for text in args {
        let t = Instant::now();
        match enc.encode(&text, 512) {
            Ok(v) => {
                eprintln!("describe_encode: encode {} ms", t.elapsed().as_millis());
                println!("{}", serde_json::to_string(&v).unwrap());
            }
            Err(e) => {
                eprintln!("ALERT describe_encode: {e}");
                std::process::exit(2);
            }
        }
    }
}
