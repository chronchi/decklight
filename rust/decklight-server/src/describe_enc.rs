//! The Qwen3-Embedding-0.6B query encoder (fp16 ONNX), a port of `Encoder` in
//! `scripts/search/semantic_eval.py`.
//!
//! Last-token pooling with an explicit EOS (151643), then the first `dim` values, L2-normalised to f32.
//! The model's KV-cache inputs are fed as empty f16 tensors `[1, 8, 0, 128]`; input names and the cache
//! element type are read from the loaded session.

use std::path::Path;

use half::f16;
use ort::session::Session;
use ort::value::{Tensor, ValueType};
use tokenizers::Tokenizer;

/// The query instruction from `semantic_eval.INSTRUCT` (line 28). The encoder expects it as a prefix.
pub const INSTRUCT: &str = "Instruct: Given a player's description of a Yu-Gi-Oh! card, retrieve the card whose text matches\nQuery:";

const EOS: u32 = 151643;
const OUTPUT: &str = "last_hidden_state";

struct PastInput {
    name: String,
    f16: bool,
    heads: usize,
    dim: usize,
}

pub struct Encoder {
    tok: Tokenizer,
    sess: Session,
    past: Vec<PastInput>,
}

impl Encoder {
    pub fn load(model_dir: &Path) -> Result<Encoder, String> {
        let tok_path = model_dir.join("tokenizer.json");
        let tok = Tokenizer::from_file(&tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
        let model = model_dir.join("onnx/model_fp16.onnx");
        let sess = decklight_onnx::session_from_file(&model, 4)?;
        let mut past = Vec::new();
        for input in sess.inputs() {
            let name = input.name().to_string();
            if !name.starts_with("past_key_values.") {
                continue;
            }
            let (f16, shape) = match input.dtype() {
                ValueType::Tensor { ty, shape, .. } => (
                    matches!(ty, ort::value::TensorElementType::Float16),
                    shape.to_vec(),
                ),
                _ => return Err(format!("{name}: expected a tensor input")),
            };
            let heads = shape.get(1).copied().filter(|d| *d > 0).unwrap_or(8) as usize;
            let dim = shape.get(3).copied().filter(|d| *d > 0).unwrap_or(128) as usize;
            past.push(PastInput { name, f16, heads, dim });
        }
        if past.is_empty() {
            return Err(format!("{}: no past_key_values inputs", model.display()));
        }
        Ok(Encoder { tok, sess, past })
    }

    /// Encode one query (already carrying `INSTRUCT`), keeping the first `dim` values.
    pub fn encode(&mut self, text: &str, dim: usize) -> Result<Vec<f32>, String> {
        let enc = self.tok.encode(text, true).map_err(|e| format!("tokenizer: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        if ids.last() != Some(&(EOS as i64)) {
            ids.push(EOS as i64);
        }
        let n = ids.len();
        if n == 0 {
            return Err("empty token sequence".into());
        }
        let ids_t = Tensor::from_array(([1usize, n], ids)).map_err(|e| e.to_string())?;
        let mask_t = Tensor::from_array(([1usize, n], vec![1i64; n])).map_err(|e| e.to_string())?;
        let pos_t =
            Tensor::from_array(([1usize, n], (0..n as i64).collect::<Vec<_>>())).map_err(|e| e.to_string())?;
        let mut feeds: Vec<(String, ort::value::DynValue)> = vec![
            ("input_ids".into(), ids_t.into_dyn()),
            ("attention_mask".into(), mask_t.into_dyn()),
            ("position_ids".into(), pos_t.into_dyn()),
        ];
        for p in &self.past {
            let v = if p.f16 {
                Tensor::from_array(([1usize, p.heads, 0usize, p.dim], Vec::<f16>::new()))
                    .map_err(|e| e.to_string())?
                    .into_dyn()
            } else {
                Tensor::from_array(([1usize, p.heads, 0usize, p.dim], Vec::<f32>::new()))
                    .map_err(|e| e.to_string())?
                    .into_dyn()
            };
            feeds.push((p.name.clone(), v));
        }
        let outs = self.sess.run(feeds).map_err(|e| e.to_string())?;
        let out = outs.get(OUTPUT).ok_or_else(|| format!("model has no {OUTPUT} output"))?;
        let (shape, data) = out.try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != n {
            return Err(format!("unexpected {OUTPUT} shape {shape:?} for {n} tokens"));
        }
        let width = shape[2];
        let take = dim.min(width);
        let start = (n - 1) * width;
        let mut v = data[start..start + take].to_vec();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err("non-finite or zero query vector".into());
        }
        for x in &mut v {
            *x /= norm;
        }
        Ok(v)
    }
}
