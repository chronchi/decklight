//! ONNX parity spike (offline-app milestone A0, S2): run the exported product
//! graphs with `ort` (ONNX Runtime bindings) or `tract` (pure Rust), over the
//! `.npy` decision batch staged by `scripts/onnx_parity.py`.
//!
//! ```sh
//! onnx-parity --onnx data/cache/onnx/r1 --obs data/cache/onnx_parity/obs --out /tmp/rust_out [--limit N]
//! ```
//! Writes `$out/policy_logits.npy` (float32 [B, 64]) and `$out/value.npy`
//! (float32 [B]), and prints one JSON object on stdout with the backend name,
//! the per-graph input lists and a small latency bench (median of 20 runs for
//! batch 1 / 16 / 64 with 1 and 4 intra-op threads; tract is single-threaded
//! and reports the same session for both).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use serde_json::Value;

#[cfg(all(feature = "ort", feature = "tract"))]
compile_error!("enable exactly one of `ort`, `tract`");
#[cfg(not(any(feature = "ort", feature = "tract")))]
compile_error!("enable one of `ort`, `tract`");

#[cfg(feature = "ort")]
const BACKEND: &str = "ort";
#[cfg(all(feature = "tract", not(feature = "ort")))]
const BACKEND: &str = "tract";

fn err(s: impl ToString) -> String {
    s.to_string()
}

// ---------------------------------------------------------------------------
// .npy helpers
// ---------------------------------------------------------------------------

fn read_f32_2(path: &Path) -> Result<Array2<f32>, String> {
    ndarray_npy::read_npy(path).map_err(err)
}

fn read_i32(path: &Path) -> Result<ArrayD<i32>, String> {
    ndarray_npy::read_npy(path).map_err(err)
}

fn read_i64_1(path: &Path) -> Result<Array1<i64>, String> {
    ndarray_npy::read_npy(path).map_err(err)
}

fn read_bool(path: &Path) -> Result<ArrayD<bool>, String> {
    match ndarray_npy::read_npy(path) {
        Ok(a) => Ok(a),
        Err(_) => {
            // tolerate writers that store masks as u8
            let a: ArrayD<u8> = ndarray_npy::read_npy(path).map_err(err)?;
            Ok(a.mapv(|v| v != 0))
        }
    }
}

// ---------------------------------------------------------------------------
// codes -> row indices (mirrors decklight.model.onnx_export.codes_to_rows)
// ---------------------------------------------------------------------------

fn codes_to_rows(codes: &Array1<i64>, arr: &mut ArrayD<i32>, col: usize) {
    let n = arr.shape()[arr.ndim() - 1];
    assert!(col < n, "col {col} out of range (last dim {n})");
    let last = arr.ndim() - 1;
    let slice = codes.as_slice().expect("codes must be contiguous");
    for mut lane in arr.lanes_mut(Axis(last)) {
        let c = lane[col] as i64;
        let row = if c <= 0 {
            0
        } else {
            match slice.binary_search(&c) {
                Ok(pos) => (pos + 1) as i32,
                Err(_) => 0,
            }
        };
        lane[col] = row;
    }
}

// ---------------------------------------------------------------------------
// backend abstraction
// ---------------------------------------------------------------------------

enum Feed<'a> {
    F32(&'a ArrayD<f32>),
    I32(&'a ArrayD<i32>),
    Bool(&'a ArrayD<bool>),
}

struct Model {
    #[cfg(feature = "ort")]
    sessions: HashMap<usize, ort::session::Session>,
    #[cfg(feature = "ort")]
    env: ort::environment::Environment,
    #[cfg(feature = "ort")]
    path: PathBuf,
    #[cfg(feature = "tract")]
    runnable: tract_onnx::prelude::TypedSimplePlan<tract_onnx::prelude::TypedModel>,
    #[cfg(feature = "tract")]
    input_order: Vec<String>,
}

#[cfg(feature = "ort")]
fn ort_env() -> Result<ort::environment::Environment, String> {
    use std::sync::OnceLock;
    static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e.clone());
    }
    // load-dynamic: ORT_DYLIB_PATH points at libonnxruntime.so (see report);
    // otherwise ort falls back to the default library name.
    let env = match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) if !p.is_empty() => ort::init_from(std::path::Path::new(&p))
            .map_err(err)?
            .with_name("onnx-parity")
            .build()
            .map_err(err)?,
        _ => ort::init()
            .with_name("onnx-parity")
            .build()
            .map_err(err)?,
    };
    let _ = ENV.set(env.clone());
    Ok(env)
}

impl Model {
    fn load(path: &Path, intra_threads: usize) -> Result<Self, String> {
        #[cfg(feature = "ort")]
        {
            let env = ort_env()?;
            let session = ort::session::Session::builder(&env)
                .map_err(err)?
                .with_intra_threads(intra_threads)
                .map_err(err)?
                .commit_from_file(path)
                .map_err(err)?;
            let mut sessions = HashMap::new();
            sessions.insert(intra_threads, session);
            return Ok(Model { sessions, env, path: path.to_path_buf() });
        }
        #[cfg(all(feature = "tract", not(feature = "ort")))]
        {
            use tract_onnx::prelude::InferenceModelExt;
            use tract_onnx::tract_core::framework::Framework as _;
            let _ = intra_threads;
            let runnable = tract_onnx::onnx()
                .model_for_path(path)
                .map_err(|e| format!("model_for_path {path:?}: {e:?}"))?
                .into_optimized()
                .map_err(|e| format!("into_optimized {path:?}: {e:?}"))?
                .into_runnable()
                .map_err(|e| format!("into_runnable {path:?}: {e:?}"))?;
            let model = runnable.model();
            let mut input_order = Vec::new();
            for id in model.input_outlets().map_err(err)? {
                input_order.push(model.node(id.node).name.to_string());
            }
            return Ok(Model { runnable, input_order });
        }
    }

    fn with_threads(&mut self, intra_threads: usize) -> Result<(), String> {
        #[cfg(feature = "ort")]
        {
            if !self.sessions.contains_key(&intra_threads) {
                let session = ort::session::Session::builder(&self.env)
                    .map_err(err)?
                    .with_intra_threads(intra_threads)
                    .map_err(err)?
                    .commit_from_file(&self.path)
                    .map_err(err)?;
                self.sessions.insert(intra_threads, session);
            }
            return Ok(());
        }
        #[cfg(all(feature = "tract", not(feature = "ort")))]
        {
            let _ = (self, intra_threads);
            return Ok(()); // tract is single-threaded; the same session serves both settings
        }
    }

    fn run(&mut self, feeds: &HashMap<String, Feed>, intra_threads: usize) -> Result<HashMap<String, ArrayD<f32>>, String> {
        self.with_threads(intra_threads)?;
        #[cfg(feature = "ort")]
        {
            use ort::value::Tensor;
            let session = self.sessions.get_mut(&intra_threads).unwrap();
            // Build owned tensors; Tensor::from_array copies/clones into ORT-owned memory.
            let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::with_capacity(feeds.len());
            for (k, f) in feeds {
                let v = match f {
                    Feed::F32(a) => Tensor::from_array((*a).clone()).map_err(err)?.into_dyn(),
                    Feed::I32(a) => Tensor::from_array((*a).clone()).map_err(err)?.into_dyn(),
                    Feed::Bool(a) => Tensor::from_array((*a).clone()).map_err(err)?.into_dyn(),
                };
                inputs.push((k.clone(), v));
            }
            let outputs = session.run(inputs).map_err(err)?;
            let names: Vec<String> = {
                // outputs borrows session mutably; collect names first from a re-borrow
                // (names are static graph outputs, so re-query after run is fine once
                // the mutable borrow of `outputs` ends for the session handle itself;
                // instead we use the known output keys via keys()).
                outputs.keys().map(|k| k.to_string()).collect()
            };
            let mut out = HashMap::new();
            for name in names {
                let (shape, data) = outputs[name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(err)?;
                let shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec()).map_err(err)?;
                out.insert(name, arr);
            }
            return Ok(out);
        }
        #[cfg(all(feature = "tract", not(feature = "ort")))]
        {
            use tract_onnx::prelude::*;
            let mut tvec = TVec::new();
            for name in self.input_order.clone() {
                let f = feeds.get(&name).ok_or_else(|| format!("missing feed for input {name}"))?;
                // Build tract Tensors from shape+slice (avoids ndarray 0.16 vs 0.17 type clash:
                // tract bundles ndarray 0.16, this crate uses 0.17 for ort).
                let t: Tensor = match f {
                    Feed::F32(a) => {
                        let s: Vec<usize> = a.shape().to_vec();
                        let d = a.as_slice().ok_or("non-contiguous f32 input")?;
                        Tensor::from_shape(&s, d).map_err(err)?
                    }
                    Feed::I32(a) => {
                        let s: Vec<usize> = a.shape().to_vec();
                        let d = a.as_slice().ok_or("non-contiguous i32 input")?;
                        Tensor::from_shape(&s, d).map_err(err)?
                    }
                    Feed::Bool(a) => {
                        let s: Vec<usize> = a.shape().to_vec();
                        let d = a.as_slice().ok_or("non-contiguous bool input")?;
                        Tensor::from_shape(&s, d).map_err(err)?
                    }
                };
                tvec.push(t.into());
            }
            let result = self.runnable.run(tvec).map_err(err)?;
            let mut out = HashMap::new();
            let model = self.runnable.model();
            for (i, id) in model.output_outlets().map_err(err)?.iter().enumerate() {
                // The optimizer renames output nodes; the ONNX output name survives as the outlet label.
                let name = model.outlet_label(*id).map(str::to_string).unwrap_or_else(|| model.node(id.node).name.to_string());
                let view = result[i].to_array_view::<f32>().map_err(err)?;
                let shape: Vec<usize> = view.shape().to_vec();
                let data: Vec<f32> = view.iter().copied().collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(err)?;
                out.insert(name, arr);
            }
            let _ = intra_threads;
            return Ok(out);
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Args {
    onnx: PathBuf,
    obs: PathBuf,
    out: PathBuf,
    limit: Option<usize>,
    history_source: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--onnx" => a.onnx = PathBuf::from(it.next().ok_or("--onnx needs a value")?),
            "--obs" => a.obs = PathBuf::from(it.next().ok_or("--obs needs a value")?),
            "--out" => a.out = PathBuf::from(it.next().ok_or("--out needs a value")?),
            "--limit" => {
                a.limit = Some(it.next().ok_or("--limit needs a value")?.parse::<usize>().map_err(err)?)
            }
            "--history-source" => {
                a.history_source = Some(it.next().ok_or("--history-source needs a value")?)
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    if a.onnx.as_os_str().is_empty() || a.obs.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err("usage: onnx-parity --onnx DIR --obs DIR --out DIR [--limit N]".to_string());
    }
    Ok(a)
}

fn code_columns(codes_json: &Value) -> Result<Vec<(String, usize)>, String> {
    let cols = codes_json.get("columns").and_then(Value::as_array).ok_or("codes.json: no columns")?;
    cols.iter()
        .map(|c| {
            Ok((
                c.get("key").and_then(Value::as_str).ok_or("column without key")?.to_string(),
                c.get("col").and_then(Value::as_u64).ok_or("column without col")? as usize,
            ))
        })
        .collect()
}

/// Copies of the obs maps with `history` / `history_mask` taken from the graph's history source (the ONNX
/// `history_source`: `obs_history` for an observable net, else `history`).
fn routed(
    ints: &HashMap<String, ArrayD<i32>>,
    masks: &HashMap<String, ArrayD<bool>>,
    source: &str,
) -> Result<(HashMap<String, ArrayD<i32>>, HashMap<String, ArrayD<bool>>), String> {
    let mut i = ints.clone();
    let mut m = masks.clone();
    if source == "obs_history" {
        let h = ints.get("obs_history").ok_or("history_source=obs_history but no obs_history.npy")?;
        let hm = masks
            .get("obs_history_mask")
            .ok_or("history_source=obs_history but no obs_history_mask.npy")?;
        i.insert("history".to_string(), h.clone());
        m.insert("history_mask".to_string(), hm.clone());
    } else if source != "history" {
        return Err(format!("unknown history source {source}"));
    }
    Ok((i, m))
}

fn main() {
    let rc = real_main();
    if let Err(e) = rc {
        eprintln!("onnx-parity [{BACKEND}] error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out).map_err(err)?;

    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(args.onnx.join("manifest.json")).map_err(err)?)
            .map_err(err)?;
    let graphs = manifest.get("graphs").ok_or("manifest: no graphs")?;
    let policy_inputs: Vec<String> = graphs
        .get("policy")
        .and_then(|g| g.get("inputs"))
        .and_then(Value::as_array)
        .ok_or("manifest: no policy inputs")?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    let value_inputs: Vec<String> = graphs
        .get("value")
        .and_then(|g| g.get("inputs"))
        .and_then(Value::as_array)
        .ok_or("manifest: no value inputs")?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    let codes_json: Value =
        serde_json::from_str(&fs::read_to_string(args.onnx.join("codes.json")).map_err(err)?)
            .map_err(err)?;
    let columns = code_columns(&codes_json)?;
    let codes = read_i64_1(&args.obs.join("codes.npy"))?;
    let raw = read_f32_2(&args.obs.join("raw.npy"))?;
    let unknown1: Array1<f32> = ndarray_npy::read_npy(args.onnx.join("unknown.npy")).map_err(err)?;
    let unknown_v1: Array1<f32> =
        ndarray_npy::read_npy(args.onnx.join("unknown_value.npy")).map_err(err)?;
    let unknown = unknown1.insert_axis(Axis(0));
    let unknown_v = unknown_v1.insert_axis(Axis(0));

    // raw_full = concat([unknown, raw]) -> card_proj -> text table
    let raw_full =
        ndarray::concatenate(Axis(0), &[unknown.view(), raw.view()]).map_err(err)?;
    let raw_full_v =
        ndarray::concatenate(Axis(0), &[unknown_v.view(), raw.view()]).map_err(err)?;

    let mut proj = Model::load(&args.onnx.join("card_proj.onnx"), 1)?;
    let mut proj_v = Model::load(&args.onnx.join("card_proj_value.onnx"), 1)?;
    let mut policy = Model::load(&args.onnx.join("policy.onnx"), 1)?;
    let mut value = Model::load(&args.onnx.join("value.onnx"), 1)?;

    let mut feeds = HashMap::new();
    let raw_full_dyn = raw_full.clone().into_dyn();
    feeds.insert("raw".to_string(), Feed::F32(&raw_full_dyn));
    let text = proj.run(&feeds, 1)?.remove("text").ok_or("card_proj: no text output")?;
    let mut feeds_v = HashMap::new();
    let raw_full_v_dyn = raw_full_v.clone().into_dyn();
    feeds_v.insert("raw".to_string(), Feed::F32(&raw_full_v_dyn));
    let text_v = proj_v.run(&feeds_v, 1)?.remove("text").ok_or("card_proj_value: no text output")?;

    // obs (optionally truncated), codes -> rows. The observable window is optional: old obs dirs lack it.
    let mut ints = HashMap::new();
    for k in ["cards", "global", "chain", "actions", "history", "obs_history"] {
        let p = args.obs.join(format!("{k}.npy"));
        if !p.is_file() {
            continue;
        }
        let mut a = read_i32(&p)?;
        if let Some(n) = args.limit {
            let take = n.min(a.shape()[0]);
            a = a.slice_axis(Axis(0), (0..take).into()).to_owned();
        }
        ints.insert(k.to_string(), a);
    }
    let mut masks = HashMap::new();
    for k in ["card_mask", "chain_mask", "action_mask", "history_mask", "obs_history_mask"] {
        let p = args.obs.join(format!("{k}.npy"));
        if !p.is_file() {
            continue;
        }
        let mut a = read_bool(&p)?;
        if let Some(n) = args.limit {
            let take = n.min(a.shape()[0]);
            a = a.slice_axis(Axis(0), (0..take).into()).to_owned();
        }
        masks.insert(k.to_string(), a);
    }
    let nrows = ints["cards"].shape()[0];
    for (key, col) in &columns {
        if let Some(a) = ints.get_mut(key) {
            codes_to_rows(&codes, a, *col);
        }
    }
    // The manifest names each graph's history source (absent = history); `--history-source` overrides the
    // policy's, for the wrong-source check.
    let policy_source = args
        .history_source
        .clone()
        .or_else(|| manifest.get("history_source").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| "history".to_string());
    let value_source = manifest
        .get("value_history_source")
        .and_then(Value::as_str)
        .unwrap_or("history")
        .to_string();
    let (ints_p, masks_p) = routed(&ints, &masks, &policy_source)?;
    let (ints_v, masks_v) = routed(&ints, &masks, &value_source)?;
    eprintln!("onnx-parity [{BACKEND}]: history source policy={policy_source} value={value_source}");

    // chunked inference over all rows (batch 256 keeps RSS small; matches scripts/onnx_parity.py)
    let chunk = 256;
    let mut all_logits: Vec<f32> = Vec::new();
    let mut all_value: Vec<f32> = Vec::new();
    let mut n_opt = 0usize;
    let mut k = 0;
    while k < nrows {
        let j = (k + chunk).min(nrows);
        let cut_i = |m: &HashMap<String, ArrayD<i32>>| -> HashMap<String, ArrayD<i32>> {
            m.iter()
                .map(|(name, a)| (name.clone(), a.slice_axis(Axis(0), (k..j).into()).to_owned()))
                .collect()
        };
        let cut_m = |m: &HashMap<String, ArrayD<bool>>| -> HashMap<String, ArrayD<bool>> {
            m.iter()
                .map(|(name, a)| (name.clone(), a.slice_axis(Axis(0), (k..j).into()).to_owned()))
                .collect()
        };
        let sl_ip = cut_i(&ints_p);
        let sl_mp = cut_m(&masks_p);
        let sl_iv = cut_i(&ints_v);
        let sl_mv = cut_m(&masks_v);
        let mut fp = HashMap::new();
        for name in &policy_inputs {
            if name == "text" {
                fp.insert(name.clone(), Feed::F32(&text));
            } else if let Some(a) = sl_ip.get(name) {
                fp.insert(name.clone(), Feed::I32(a));
            } else if let Some(m) = sl_mp.get(name) {
                fp.insert(name.clone(), Feed::Bool(m));
            }
        }
        let mut fv = HashMap::new();
        for name in &value_inputs {
            if name == "text" {
                fv.insert(name.clone(), Feed::F32(&text_v));
            } else if let Some(a) = sl_iv.get(name) {
                fv.insert(name.clone(), Feed::I32(a));
            } else if let Some(m) = sl_mv.get(name) {
                fv.insert(name.clone(), Feed::Bool(m));
            }
        }
        let lo = policy.run(&fp, 1)?.remove("logits").ok_or("policy: no logits")?;
        let va = value.run(&fv, 1)?.remove("value").ok_or("value: no value")?;
        n_opt = lo.shape()[1];
        all_logits.extend(lo.iter().copied());
        all_value.extend(va.iter().copied());
        k = j;
    }
    let logits = ArrayD::from_shape_vec(IxDyn(&[nrows, n_opt]), all_logits).map_err(err)?;
    let values = ArrayD::from_shape_vec(IxDyn(&[nrows]), all_value).map_err(err)?;
    ndarray_npy::write_npy(args.out.join("policy_logits.npy"), &logits).map_err(err)?;
    ndarray_npy::write_npy(args.out.join("value.npy"), &values).map_err(err)?;

    // bench: median of 20 for batch 1 / 16 / 64 at 1 and 4 threads
    let mut bench = serde_json::Map::new();
    for &b in &[1usize, 16, 64] {
        let bb = b.min(nrows);
        let sl_ip: HashMap<String, ArrayD<i32>> = ints_p
            .iter()
            .map(|(name, a)| (name.clone(), a.slice_axis(Axis(0), (0..bb).into()).to_owned()))
            .collect();
        let sl_mp: HashMap<String, ArrayD<bool>> = masks_p
            .iter()
            .map(|(name, a)| (name.clone(), a.slice_axis(Axis(0), (0..bb).into()).to_owned()))
            .collect();
        let sl_iv: HashMap<String, ArrayD<i32>> = ints_v
            .iter()
            .map(|(name, a)| (name.clone(), a.slice_axis(Axis(0), (0..bb).into()).to_owned()))
            .collect();
        let sl_mv: HashMap<String, ArrayD<bool>> = masks_v
            .iter()
            .map(|(name, a)| (name.clone(), a.slice_axis(Axis(0), (0..bb).into()).to_owned()))
            .collect();
        for &th in &[1usize, 4] {
            let mut fp = HashMap::new();
            for name in &policy_inputs {
                if name == "text" {
                    fp.insert(name.clone(), Feed::F32(&text));
                } else if let Some(a) = sl_ip.get(name) {
                    fp.insert(name.clone(), Feed::I32(a));
                } else if let Some(m) = sl_mp.get(name) {
                    fp.insert(name.clone(), Feed::Bool(m));
                }
            }
            let mut fv = HashMap::new();
            for name in &value_inputs {
                if name == "text" {
                    fv.insert(name.clone(), Feed::F32(&text_v));
                } else if let Some(a) = sl_iv.get(name) {
                    fv.insert(name.clone(), Feed::I32(a));
                } else if let Some(m) = sl_mv.get(name) {
                    fv.insert(name.clone(), Feed::Bool(m));
                }
            }
            // warmup
            policy.run(&fp, th)?;
            value.run(&fv, th)?;
            let mut dt: Vec<f64> = Vec::new();
            for _ in 0..20 {
                let t0 = Instant::now();
                policy.run(&fp, th)?;
                value.run(&fv, th)?;
                dt.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            dt.sort_by(|x, y| x.partial_cmp(y).unwrap());
            bench.insert(
                format!("batch{bb}_t{th}"),
                Value::from(dt[dt.len() / 2]),
            );
        }
    }
    let report = serde_json::json!({
        "backend": BACKEND,
        "n": nrows,
        "n_opt": n_opt,
        "policy_inputs": policy_inputs,
        "value_inputs": value_inputs,
        "history_source": {"policy": policy_source, "value": value_source},
        "bench_ms_median20": bench,
        "threads_note": if BACKEND == "tract" {
            "tract is single-threaded; t1 and t4 use the same session"
        } else {
            "ort intra-op threads set per session"
        },
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}
