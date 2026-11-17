//! ONNX Runtime inference for the product policy/value graphs.
//!
//! This is the code that lived in `decklight-server/src/raw.rs` (A2c), moved here so the search and the server
//! share one feeding path. The contract is `decklight.model.onnx_export`: code columns become row indices into the
//! `card_proj` text table (row 0 = `proj(unknown)`), the token axes are trimmed to the batch's longest row (rounded
//! up to `TRIM_TO`), and the logits are padded back with `-inf` on the cut options.
//!
//! Import of the Python oracle is `onnx_parity.run_ort_full`; the search gate requires identical ORT inputs and
//! outputs (f32 bits) with the same batches, so this module must not reorder or re-batch them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use serde_json::Value;
use sha2::Digest;
use ygoenv_sys::Obs;

use ygoenv_sys::{ACTION_COLS, CARD_COLS, CHAIN_COLS, GLOBAL_COLS, HISTORY, HISTORY_COLS, MAX_CARDS, MAX_CHAIN, MAX_OPTIONS};

/// `policy.TRIM_TO`: trimmed lengths round up to a multiple of this.
pub const TRIM_TO: usize = 8;

/// Value returned by a graph is clipped to this range (as `ValueNet.values` does).
pub const VALUE_CLIP: f32 = 1.0;

/// Where a graph's `history` / `history_mask` inputs come from: the engine history (v1) or the
/// observable window (an `PolicyNet(observable=True)` checkpoint; ONNX `metadata_props.history_source`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HistorySource {
    History,
    ObsHistory,
}

impl HistorySource {
    fn from_session(sess: &Session) -> Self {
        match sess.metadata().ok().and_then(|m| m.custom("history_source")) {
            Some(s) if s == "obs_history" => Self::ObsHistory,
            _ => Self::History, // absent = an old export: history
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::History => "history",
            Self::ObsHistory => "obs_history",
        }
    }

    fn rows<'a>(self, o: &'a Obs) -> &'a [i32] {
        match self {
            Self::History => &o.history,
            Self::ObsHistory => &o.obs_history,
        }
    }

    fn mask<'a>(self, o: &'a Obs) -> &'a [u8] {
        match self {
            Self::History => &o.history_mask,
            Self::ObsHistory => &o.obs_history_mask,
        }
    }
}

/// The shared ORT environment (one per process; `ORT_DYLIB_PATH` selects the runtime library).
pub fn environment() -> Result<ort::environment::Environment, String> {
    ort_env()
}

/// Load one graph into a session with `intra_threads` ORT threads (the shared environment is reused).
pub fn session_from_file(path: &Path, intra_threads: usize) -> Result<Session, String> {
    session(path, intra_threads)
}

fn ort_env() -> Result<ort::environment::Environment, String> {
    static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e.clone());
    }
    let env = match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) if !p.is_empty() => ort::init_from(Path::new(&p))
            .map_err(|e| e.to_string())?
            .with_name("decklight-onnx")
            .build()
            .map_err(|e| e.to_string())?,
        _ => ort::init()
            .with_name("decklight-onnx")
            .build()
            .map_err(|e| e.to_string())?,
    };
    let _ = ENV.set(env.clone());
    Ok(env)
}

fn session(path: &Path, intra_threads: usize) -> Result<Session, String> {
    let env = ort_env()?;
    Session::builder(&env)
        .map_err(|e| e.to_string())?
        .with_intra_threads(intra_threads.max(1))
        .map_err(|e| e.to_string())?
        .commit_from_file(path)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn run_text(sess: &mut Session, table: ArrayD<f32>) -> Result<Array2<f32>, String> {
    let mut inputs = Vec::with_capacity(1);
    let t = Tensor::from_array(table).map_err(|e| e.to_string())?;
    inputs.push(("raw".to_string(), t.into_dyn()));
    let outs = sess.run(inputs).map_err(|e| e.to_string())?;
    let (shape, data) = outs["text"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec()).map_err(|e| e.to_string())?;
    arr.into_dimensionality::<ndarray::Ix2>().map_err(|e| e.to_string())
}

fn read_manifest(onnx_dir: &Path) -> Result<Value, String> {
    let p = onnx_dir.join("manifest.json");
    serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| e.to_string())
}

fn graph_inputs(manifest: &Value, graph: &str) -> Result<Vec<String>, String> {
    let g = manifest
        .get("graphs")
        .and_then(Value::as_object)
        .and_then(|g| g.get(graph))
        .ok_or_else(|| format!("manifest: no graph {graph}"))?;
    let ins = g
        .get("inputs")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("manifest: {graph} inputs"))?;
    Ok(ins.iter().filter_map(Value::as_str).map(str::to_string).collect())
}

fn graph_output(manifest: &Value, graph: &str, default: &str) -> String {
    manifest
        .get("graphs")
        .and_then(Value::as_object)
        .and_then(|g| g.get(graph))
        .and_then(|g| g.get("outputs"))
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

fn find_emb(manifest: &Value, repo_root: &Path) -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("YGO_EMB") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(inputs) = manifest.get("inputs").and_then(Value::as_object) {
        for k in inputs.keys() {
            if k.ends_with(".npz") && k.contains("emb") {
                let p = PathBuf::from(k);
                return Ok(if p.is_absolute() { p } else { repo_root.join(p) });
            }
        }
    }
    let p = repo_root.join("data/cards/emb/qwen3-0.6b-d512.npz");
    if p.is_file() { Ok(p) } else { Err("no embedding npz in manifest".into()) }
}

fn load_unknown(path: &Path) -> Result<Vec<f32>, String> {
    let a: ArrayD<f32> = ndarray_npy::read_npy(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(a.into_raw_vec_and_offset().0)
}

fn table_for(onnx_dir: &Path, emb_path: &Path, proj_graph: &str, unknown_file: &str) -> Result<(Vec<i64>, Array2<f32>), String> {
    let emb = decklight_cards::emb::Emb::load(emb_path)?;
    let codes: Vec<i64> = emb.codes().to_vec();
    let (raw, n, cols) = emb.raw_table();
    let raw2 = Array2::from_shape_vec((n, cols), raw).map_err(|e| e.to_string())?;
    let unknown = load_unknown(&onnx_dir.join(unknown_file))?;
    let mut table_raw = Array2::<f32>::zeros((n + 1, cols));
    for (j, v) in unknown.iter().enumerate().take(cols) {
        table_raw[[0, j]] = *v;
    }
    table_raw.slice_mut(ndarray::s![1.., ..]).assign(&raw2);
    let mut proj = session(&onnx_dir.join(format!("{proj_graph}.onnx")), 1)?;
    let text = run_text(&mut proj, table_raw.into_dyn())?;
    Ok((codes, text))
}

/// One ORT call's input/output hashes (the gate-1 trace).
#[derive(Clone, Debug)]
pub struct IoRec {
    pub kind: &'static str,
    pub batch: usize,
    pub input_sha256: [u8; 32],
    pub output_sha256: [u8; 32],
    pub output_shape: Vec<usize>,
}

impl IoRec {
    pub fn input_hex(&self) -> String {
        self.input_sha256.iter().map(|b| format!("{b:02x}")).collect()
    }
    pub fn output_hex(&self) -> String {
        self.output_sha256.iter().map(|b| format!("{b:02x}")).collect()
    }
}

fn hash_head(h: &mut sha2::Sha256, name: &str, dtype: u8, shape: &[usize]) {
    use sha2::Digest;
    h.update(name.as_bytes());
    h.update([0u8, dtype]);
    h.update((shape.len() as u64).to_le_bytes());
    for &d in shape {
        h.update((d as u64).to_le_bytes());
    }
}

fn hash_i32(h: &mut sha2::Sha256, name: &str, a: &ArrayD<i32>) {
    use sha2::Digest;
    hash_head(h, name, 1, a.shape());
    for &v in a.iter() {
        h.update(v.to_le_bytes());
    }
}

fn hash_bool(h: &mut sha2::Sha256, name: &str, a: &Array2<bool>) {
    use sha2::Digest;
    hash_head(h, name, 2, a.shape());
    for &v in a.iter() {
        h.update([v as u8]);
    }
}


fn sha_table(a: &Array2<f32>) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    for &v in a.iter() {
        h.update(v.to_le_bytes());
    }
    h.finalize().into()
}

/// The four product graphs with their text tables.
pub struct OnnxNets {
    codes: Vec<i64>,
    text: Array2<f32>,
    text_value: Array2<f32>,
    policy: Vec<Session>,
    policy_inputs: Vec<String>,
    policy_output: String,
    policy_history_source: HistorySource,
    value: Vec<Session>,
    value_inputs: Vec<String>,
    value_output: String,
    value_history_source: HistorySource,
    /// Extra single-thread policy sessions for `policy_logits_threads` (loaded on first use).
    policy_workers: Vec<Session>,
    policy_path: PathBuf,
    pub policy_sha256: String,
    pub value_sha256: String,
    text_sha: [u8; 32],
    text_value_sha: [u8; 32],
    io_log: Option<Vec<IoRec>>,
}

impl OnnxNets {
    pub fn load(onnx_dir: &Path, repo_root: &Path) -> Result<OnnxNets, String> {
        Self::load_with_threads(onnx_dir, repo_root, 1)
    }

    /// `intra_threads` = ORT intra-op threads for the policy and value sessions (1 = the parity setting). On x86-64
    /// 4 threads gave bit-identical games (NOTEBOOK A3b-3); check again on each new CPU family before relying on it.
    /// The card text table is always projected with 1 thread.
    pub fn load_with_threads(onnx_dir: &Path, repo_root: &Path, intra_threads: usize) -> Result<OnnxNets, String> {
        let manifest = read_manifest(onnx_dir)?;
        let emb = find_emb(&manifest, repo_root)?;
        let (codes, text) = table_for(onnx_dir, &emb, "card_proj", "unknown.npy")?;
        let (_, text_value) = table_for(onnx_dir, &emb, "card_proj_value", "unknown_value.npy")?;
        let policy_inputs = graph_inputs(&manifest, "policy")?;
        let value_inputs = graph_inputs(&manifest, "value")?;
        let policy_output = graph_output(&manifest, "policy", "logits");
        let value_output = graph_output(&manifest, "value", "value");
        let policy_sha256 = sha256_file(&onnx_dir.join("policy.onnx"));
        let value_sha256 = sha256_file(&onnx_dir.join("value.onnx"));
        let text_sha = sha_table(&text);
        let text_value_sha = sha_table(&text_value);
        let policy = session(&onnx_dir.join("policy.onnx"), intra_threads)?;
        let policy_history_source = HistorySource::from_session(&policy);
        let value = session(&onnx_dir.join("value.onnx"), intra_threads)?;
        let value_history_source = HistorySource::from_session(&value);
        let policy_path = onnx_dir.join("policy.onnx");
        eprintln!(
            "decklight-onnx: history source: policy={} value={}",
            policy_history_source.name(),
            value_history_source.name()
        );
        Ok(OnnxNets {
            codes,
            text,
            text_value,
            policy: vec![policy],
            policy_inputs,
            policy_output,
            policy_history_source,
            value: vec![value],
            value_inputs,
            value_output,
            value_history_source,
            policy_workers: Vec::new(),
            policy_path,
            policy_sha256,
            value_sha256,
            text_sha,
            text_value_sha,
            io_log: None,
        })
    }

    /// Start recording an `IoRec` for every ORT call (gate 1).
    pub fn enable_io_log(&mut self) {
        self.io_log = Some(Vec::new());
    }

    pub fn take_io_log(&mut self) -> Vec<IoRec> {
        self.io_log.as_mut().map(std::mem::take).unwrap_or_default()
    }

    fn row(&self, code: i32) -> i32 {
        if code <= 0 {
            return 0;
        }
        match self.codes.binary_search(&(code as i64)) {
            Ok(pos) => (pos + 1) as i32,
            Err(_) => 0,
        }
    }

    /// The `used` length of `trim_obs_np`: the prefix holding every set entry of the batch, rounded up to
    /// `TRIM_TO` and capped at the axis width (`1` when nothing is set).
    fn used(obs: &[&Obs], mask: impl Fn(&Obs) -> &[u8]) -> usize {
        let width = obs.first().map_or(1, |o| mask(o).len());
        let mut last = 0usize;
        let mut any = false;
        for o in obs {
            for (i, &v) in mask(o).iter().enumerate() {
                if v != 0 {
                    last = last.max(i + 1);
                    any = true;
                }
            }
        }
        let n = if any { last } else { 1 };
        width.min(n.div_ceil(TRIM_TO) * TRIM_TO)
    }

    fn trim_sizes(obs: &[&Obs], history: HistorySource) -> (usize, usize, usize, usize) {
        (
            Self::used(obs, |o| &o.card_mask),
            Self::used(obs, |o| &o.chain_mask),
            Self::used(obs, |o| history.mask(o)),
            Self::used(obs, |o| &o.action_mask),
        )
    }

    fn build_cards(&self, obs: &[&Obs], n_cards: usize) -> Array3<i32> {
        let b = obs.len();
        let mut out = Array3::<i32>::zeros((b, n_cards, CARD_COLS));
        for (i, o) in obs.iter().enumerate() {
            for r in 0..n_cards {
                for c in 0..CARD_COLS {
                    let v = o.cards[r * CARD_COLS + c];
                    out[[i, r, c]] = if c == 0 { self.row(v) } else { v };
                }
            }
        }
        out
    }

    fn build_chain(&self, obs: &[&Obs], n_chain: usize) -> Array3<i32> {
        let b = obs.len();
        let mut out = Array3::<i32>::zeros((b, n_chain, CHAIN_COLS));
        for (i, o) in obs.iter().enumerate() {
            for r in 0..n_chain {
                for c in 0..CHAIN_COLS {
                    let v = o.chain[r * CHAIN_COLS + c];
                    out[[i, r, c]] = if c == 1 { self.row(v) } else { v };
                }
            }
        }
        out
    }

    fn build_actions(&self, obs: &[&Obs], n_opt: usize) -> Result<Array3<i32>, String> {
        let b = obs.len();
        let mut out = Array3::<i32>::zeros((b, n_opt, ACTION_COLS));
        for (i, o) in obs.iter().enumerate() {
            for r in 0..n_opt {
                for c in 0..ACTION_COLS {
                    let v = o.actions[r * ACTION_COLS + c];
                    out[[i, r, c]] = if c == 2 || c == 3 { self.row(v) } else { v };
                }
            }
        }
        // `trim_obs_np`: a legal option may not point at a card row past the trimmed card axis.
        let n_cards = Self::used(obs, |o| &o.card_mask);
        for (i, o) in obs.iter().enumerate() {
            let _ = i;
            for r in 0..n_opt {
                if o.action_mask[r] == 0 {
                    continue;
                }
                let row = o.actions[r * ACTION_COLS + 1];
                if row >= n_cards as i32 {
                    return Err(format!("a legal option points at card row {row} >= {n_cards} used rows"));
                }
            }
        }
        Ok(out)
    }

    fn build_history(&self, obs: &[&Obs], n_hist: usize, source: HistorySource) -> Array3<i32> {
        let b = obs.len();
        let mut out = Array3::<i32>::zeros((b, n_hist, HISTORY_COLS));
        for (i, o) in obs.iter().enumerate() {
            let rows = source.rows(o);
            for r in 0..n_hist {
                for c in 0..HISTORY_COLS {
                    let v = rows[r * HISTORY_COLS + c];
                    out[[i, r, c]] = if c == 2 || c == 3 { self.row(v) } else { v };
                }
            }
        }
        out
    }

    fn masks(obs: &[&Obs], mask: impl Fn(&Obs) -> &[u8], n: usize) -> Array2<bool> {
        Array2::from_shape_fn((obs.len(), n), |(i, j)| mask(obs[i])[j] != 0)
    }

    fn run_values(
        &self,
        sess: &mut Session,
        is_policy: bool,
        obs: &[&Obs],
        trim: bool,
        text: &Array2<f32>,
        io_log: Option<&mut Vec<IoRec>>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let history_source = if is_policy {
            self.policy_history_source
        } else {
            self.value_history_source
        };
        let (n_cards, n_chain, n_hist, n_opt) = if trim {
            Self::trim_sizes(obs, history_source)
        } else {
            (MAX_CARDS, MAX_CHAIN, HISTORY, MAX_OPTIONS)
        };
        let cards = self.build_cards(obs, n_cards);
        let chain = self.build_chain(obs, n_chain);
        let history = self.build_history(obs, n_hist, history_source);
        let card_mask = Self::masks(obs, |o| &o.card_mask, n_cards);
        let chain_mask = Self::masks(obs, |o| &o.chain_mask, n_chain);
        let history_mask = Self::masks(obs, |o| history_source.mask(o), n_hist);
        let global = Array2::from_shape_fn((obs.len(), GLOBAL_COLS), |(i, j)| obs[i].global[j]);
        let (actions, action_mask) = if is_policy {
            (Some(self.build_actions(obs, n_opt)?), Some(Self::masks(obs, |o| &o.action_mask, n_opt)))
        } else {
            (None, None)
        };
        let inputs = if is_policy { &self.policy_inputs } else { &self.value_inputs };
        let logging = io_log.is_some();
        let mut hasher = sha2::Sha256::new();
        let mut all: Vec<(String, ort::value::DynValue)> = Vec::new();
        for name in inputs {
            let v: ort::value::DynValue = match name.as_str() {
                "cards" => {
                    let a = cards.clone().into_dyn();
                    if logging { hash_i32(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "global" => {
                    let a = global.clone().into_dyn();
                    if logging { hash_i32(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "chain" => {
                    let a = chain.clone().into_dyn();
                    if logging { hash_i32(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "actions" => {
                    let a = actions.as_ref().ok_or("actions input without actions")?.clone().into_dyn();
                    if logging { hash_i32(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "history" => {
                    let a = history.clone().into_dyn();
                    if logging { hash_i32(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "card_mask" => {
                    let a = card_mask.clone();
                    if logging { hash_bool(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "chain_mask" => {
                    let a = chain_mask.clone();
                    if logging { hash_bool(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "action_mask" => {
                    let a = action_mask.as_ref().ok_or("action_mask input without actions")?.clone();
                    if logging { hash_bool(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "history_mask" => {
                    let a = history_mask.clone();
                    if logging { hash_bool(&mut hasher, name, &a); }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                "text" => {
                    let a = text.clone().into_dyn();
                    if logging {
                        // The table is large and constant; bind its shape and precomputed digest instead of
                        // re-hashing ~15 MB per call (the Python oracle does the same).
                        hash_head(&mut hasher, name, 3, a.shape());
                        hasher.update(if is_policy { self.text_sha } else { self.text_value_sha });
                    }
                    Tensor::from_array(a).map_err(|e| e.to_string())?.into_dyn()
                }
                other => return Err(format!("{other} not supplied by the Rust obs")),
            };
            all.push((name.clone(), v));
        }
        let output = if is_policy { self.policy_output.clone() } else { self.value_output.clone() };
        let outs = sess.run(all).map_err(|e| e.to_string())?;
        let (shape, data) = outs[output.as_str()].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        if logging {
            use sha2::Digest;
            let mut oh = sha2::Sha256::new();
            for &x in data.iter() {
                oh.update(x.to_le_bytes());
            }
            let input_sha256: [u8; 32] = hasher.finalize().into();
            let output_sha256: [u8; 32] = oh.finalize().into();
            if let Some(log) = io_log {
                log.push(IoRec {
                    kind: if is_policy { "policy" } else { "value" },
                    batch: obs.len(),
                    input_sha256,
                    output_sha256,
                    output_shape: shape.clone(),
                });
            }
        }
        Ok((shape, data.to_vec()))
    }

    fn logits_from(shape: Vec<usize>, data: Vec<f32>) -> Result<Vec<Vec<f32>>, String> {
        if shape.len() != 2 || data.len() != shape[0] * shape[1] {
            return Err(format!("policy output shape {shape:?} ({} values)", data.len()));
        }
        let width = shape[1];
        let b = shape[0];
        let mut out = vec![vec![f32::NEG_INFINITY; MAX_OPTIONS]; b];
        for i in 0..b {
            let take = width.min(MAX_OPTIONS);
            out[i][..take].copy_from_slice(&data[i * width..i * width + take]);
        }
        Ok(out)
    }

    /// Policy logits for a batch of observations, trimmed like the app and padded back to `MAX_OPTIONS` with `-inf`.
    pub fn policy_logits(&mut self, obs: &[&Obs]) -> Result<Vec<Vec<f32>>, String> {
        if obs.is_empty() {
            return Ok(Vec::new());
        }
        let mut sessions = std::mem::take(&mut self.policy);
        let mut io = self.io_log.take();
        let out = self.run_values(&mut sessions[0], true, obs, true, &self.text, io.as_mut());
        self.policy = sessions;
        self.io_log = io;
        let (shape, data) = out?;
        Self::logits_from(shape, data)
    }

    /// Policy logits without trimming (the server's historical raw-bot path).
    pub fn policy_logits_full(&mut self, obs: &[&Obs]) -> Result<Vec<Vec<f32>>, String> {
        if obs.is_empty() {
            return Ok(Vec::new());
        }
        let mut sessions = std::mem::take(&mut self.policy);
        let mut io = self.io_log.take();
        let out = self.run_values(&mut sessions[0], true, obs, false, &self.text, io.as_mut());
        self.policy = sessions;
        self.io_log = io;
        let (shape, data) = out?;
        Self::logits_from(shape, data)
    }

    /// Policy logits with the observation batch split across `threads` single-thread sessions, row order kept.
    ///
    /// The per-row graph is batch-independent, so every row's logits are bit-identical to the sequential call; the
    /// chunks and their results are concatenated in the original observation order. With the gate-1 I/O trace on
    /// (or one thread) this is exactly `policy_logits`.
    pub fn policy_logits_threads(&mut self, obs: &[&Obs], threads: usize) -> Result<Vec<Vec<f32>>, String> {
        if obs.is_empty() {
            return Ok(Vec::new());
        }
        let threads = threads.max(1).min(obs.len());
        if threads <= 1 || self.io_log.is_some() {
            return self.policy_logits(obs);
        }
        self.ensure_policy_workers(threads)?;
        let n = obs.len();
        let chunk = n.div_ceil(threads);
        let mut workers = std::mem::take(&mut self.policy_workers);
        let chunks = {
            let me = &*self;
            std::thread::scope(|s| {
                let mut hs = Vec::new();
                for (k, sess) in workers.iter_mut().enumerate().take(threads) {
                    let lo = k * chunk;
                    if lo >= n {
                        break;
                    }
                    let hi = (lo + chunk).min(n);
                    let slice = &obs[lo..hi];
                    hs.push(s.spawn(move || {
                        let (shape, data) = me.run_values(sess, true, slice, true, &me.text, None)?;
                        Self::logits_from(shape, data)
                    }));
                }
                hs.into_iter()
                    .map(|h| h.join().map_err(|_| "policy worker panicked".to_string()))
                    .collect::<Result<Vec<_>, String>>()
            })
        };
        self.policy_workers = workers;
        let mut out = Vec::with_capacity(n);
        for part in chunks? {
            out.extend_from_slice(&part?);
        }
        Ok(out)
    }

    fn ensure_policy_workers(&mut self, n: usize) -> Result<(), String> {
        while self.policy_workers.len() < n {
            let sess = session(&self.policy_path, 1)?;
            self.policy_workers.push(sess);
        }
        Ok(())
    }

    /// The value net for a batch of observations, clipped to [-1, 1] in f32.
    pub fn values(&mut self, obs: &[&Obs]) -> Result<Vec<f32>, String> {
        if obs.is_empty() {
            return Ok(Vec::new());
        }
        let mut sessions = std::mem::take(&mut self.value);
        let mut io = self.io_log.take();
        let out = self.run_values(&mut sessions[0], false, obs, true, &self.text_value, io.as_mut());
        self.value = sessions;
        self.io_log = io;
        let (shape, data) = out?;
        if data.len() != obs.len() {
            return Err(format!("value output shape {shape:?} for {} observations", obs.len()));
        }
        Ok(data.into_iter().map(|x| x.clamp(-VALUE_CLIP, VALUE_CLIP)).collect())
    }

    pub fn codes(&self) -> &[i64] {
        &self.codes
    }

    /// The `history_source` chosen for the policy and value graphs (`history` or `obs_history`).
    pub fn history_sources(&self) -> (&'static str, &'static str) {
        (
            self.policy_history_source.name(),
            self.value_history_source.name(),
        )
    }
}

fn sha256_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    match std::fs::read(path) {
        Ok(buf) => h.update(&buf),
        Err(_) => return String::new(),
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// obs fields as the DAG / training batch sees them (the server's raw_dump hash).
pub fn obs_columns(obs: &Obs) -> HashMap<&'static str, Vec<i32>> {
    let mut m = HashMap::new();
    m.insert("cards", obs.cards.to_vec());
    m.insert("global", obs.global.to_vec());
    m.insert("chain", obs.chain.to_vec());
    m.insert("actions", obs.actions.to_vec());
    m.insert("history", obs.history.to_vec());
    m.insert("obs_history", obs.obs_history.to_vec());
    m.insert("obs_history_mask", obs.obs_history_mask.iter().map(|&v| v as i32).collect());
    m
}

/// Loaded `OnnxNets` kept for reuse: a server loads the graphs (and projects the card text table) once per
/// concurrent user instead of once per game. `take` hands out a free set or loads a new one; dropping the
/// `Pooled` handle gives it back. Each set is used by one thread at a time (ORT sessions run with `&mut`).
pub struct NetsPool {
    onnx_dir: PathBuf,
    repo_root: PathBuf,
    intra_threads: usize,
    free: std::sync::Mutex<Vec<OnnxNets>>,
}

impl NetsPool {
    pub fn new(onnx_dir: &Path, repo_root: &Path, intra_threads: usize) -> std::sync::Arc<NetsPool> {
        std::sync::Arc::new(NetsPool {
            onnx_dir: onnx_dir.to_path_buf(),
            repo_root: repo_root.to_path_buf(),
            intra_threads,
            free: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn take(self: &std::sync::Arc<Self>) -> Result<Pooled, String> {
        let got = self.free.lock().unwrap().pop();
        let nets = match got {
            Some(n) => n,
            // ort panics when its runtime library cannot be loaded: report that as an error, never kill the caller
            // (a panic on the server's async worker left the client "Waiting for the bot…" forever).
            None => std::panic::catch_unwind(|| {
                OnnxNets::load_with_threads(&self.onnx_dir, &self.repo_root, self.intra_threads)
            })
            .map_err(|p| format!("the bot's model could not be loaded: {}", panic_text(&*p)))??,
        };
        Ok(Pooled { nets: Some(nets), pool: self.clone() })
    }

    pub fn intra_threads(&self) -> usize {
        self.intra_threads
    }

    /// Sets currently idle in the pool.
    pub fn idle(&self) -> usize {
        self.free.lock().unwrap().len()
    }
}

pub struct Pooled {
    nets: Option<OnnxNets>,
    pool: std::sync::Arc<NetsPool>,
}

impl std::ops::Deref for Pooled {
    type Target = OnnxNets;
    fn deref(&self) -> &OnnxNets {
        self.nets.as_ref().expect("pooled nets taken")
    }
}

impl std::ops::DerefMut for Pooled {
    fn deref_mut(&mut self) -> &mut OnnxNets {
        self.nets.as_mut().expect("pooled nets taken")
    }
}

impl Drop for Pooled {
    fn drop(&mut self) {
        if let Some(mut n) = self.nets.take() {
            n.io_log = None;
            if let Ok(mut free) = self.pool.free.lock() {
                free.push(n);
            }
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn history_source_comes_from_metadata() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skip: no ORT_DYLIB_PATH");
            return;
        }
        // The old export predates the metadata: both graphs use the engine history.
        let old = root.join("data/cache/onnx/r1");
        if old.join("manifest.json").is_file() {
            let nets = OnnxNets::load(&old, &root).unwrap();
            assert_eq!(nets.history_sources(), ("history", "history"));
        }
        // The observable product policy (Y): policy reads obs_history, value (value_p2) still history.
        let y = root.join("data/cache/onnx/y");
        if y.join("manifest.json").is_file() {
            let nets = OnnxNets::load(&y, &root).unwrap();
            assert_eq!(nets.history_sources(), ("obs_history", "history"));
        }
    }

    #[test]
    fn pool_reuses_returned_sets() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = root.join("data/cache/onnx/r1");
        if !dir.join("manifest.json").is_file() || std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skip: no ONNX graphs or ORT_DYLIB_PATH");
            return;
        }
        let pool = NetsPool::new(&dir, &root, 1);
        let a = pool.take().unwrap();
        let b = pool.take().unwrap();
        assert_eq!(pool.idle(), 0);
        drop(a);
        drop(b);
        assert_eq!(pool.idle(), 2);
        let t0 = std::time::Instant::now();
        let c = pool.take().unwrap();
        assert!(t0.elapsed().as_millis() < 50, "take from a warm pool must not reload");
        assert_eq!(pool.idle(), 1);
        drop(c);
        assert!(NetsPool::new(&root.join("data/cache/onnx/none"), &root, 1).take().is_err());
    }
}

/// The message of a caught panic (`&str` or `String` payloads), for error text.
pub fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}
