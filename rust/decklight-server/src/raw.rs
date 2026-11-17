//! The raw policy bot: ONNX argmax over the legal options (the exported product policy, `fast=True`).
//!
//! The graphs and their contract are `decklight.model.onnx_export`; the inference itself lives in
//! `decklight-onnx` (shared with the search). This module keeps the server's historical feeding contract: a single
//! observation padded to the full obs widths (untrimmed), as A2c did.

use std::path::Path;

use decklight_onnx::OnnxNets;
use serde_json::Value;
use ygoenv_sys::Obs;

pub use decklight_onnx::obs_columns;

pub struct RawPolicy {
    nets: OnnxNets,
    pub model_sha256: String,
}

impl RawPolicy {
    pub fn load(onnx_dir: &Path, repo_root: &Path) -> Result<RawPolicy, String> {
        let nets = OnnxNets::load(onnx_dir, repo_root)?;
        let model_sha256 = nets.policy_sha256.clone();
        Ok(RawPolicy { nets, model_sha256 })
    }

    /// Policy logits (64) for the pending decision in `obs`; code columns mapped to table rows.
    pub fn logits(&mut self, obs: &Obs) -> Result<Vec<f32>, String> {
        let out = self.nets.policy_logits_full(&[obs])?;
        out.into_iter().next().ok_or_else(|| "raw agent: no logits".to_string())
    }

    /// The raw agent's pick: argmax over legal options (`RawPolicyAgent.act`).
    pub fn choose(&mut self, obs: &Obs, n_opts: usize) -> Result<i32, String> {
        let z = self.logits(obs)?;
        if n_opts == 0 || n_opts > z.len() {
            return Err(format!("raw agent: {n_opts} options but {} logits", z.len()));
        }
        let mut best = f32::NEG_INFINITY;
        let mut arg = -1i32;
        for i in 0..n_opts {
            if obs.action_mask[i] == 0 {
                continue;
            }
            let v = z[i];
            if v.is_nan() {
                return Err("raw agent: NaN logits over legal options".into());
            }
            if v > best {
                best = v;
                arg = i as i32;
            }
        }
        if arg < 0 {
            return Err("raw agent: no legal option".into());
        }
        Ok(arg)
    }
}

pub fn manifest_label(onnx_dir: &Path) -> String {
    let man = std::fs::read_to_string(onnx_dir.join("manifest.json")).ok();
    let v: Option<Value> = man.and_then(|s| serde_json::from_str(&s).ok());
    v.and_then(|v| v.get("label").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| onnx_dir.file_name().and_then(|n| n.to_str()).unwrap_or("onnx").to_string())
}
