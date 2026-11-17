//! Times one Qwen3-Embedding ONNX query encode on this device, with the ONNX Runtime the app ships.
//!   embed-bench <model.onnx> <threads> <len,len,...> [reps] [f32|f16]
//! Inputs are input_ids (fixed ids), attention_mask (ones), position_ids (0..len) and an empty KV cache
//! (past_key_values.{0..27}.{key,value}, [1, 8, 0, 128], in the model's float type). Prints load time, median and
//! p90 per length, and peak RSS (VmHWM).
use ndarray::Array2;
use ort::value::Tensor;
use std::time::Instant;

fn vm_hwm_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(-1.0)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: embed-bench <model.onnx> <threads> <len,len,...> [reps]");
        std::process::exit(2);
    }
    let threads: usize = a[2].parse().expect("threads");
    let lens: Vec<usize> = a[3].split(',').map(|x| x.parse().expect("len")).collect();
    let reps: usize = a.get(4).map(|x| x.parse().expect("reps")).unwrap_or(10);
    let f16 = a.get(5).map(|x| x == "f16").unwrap_or(false);
    let dylib = std::env::var("ORT_DYLIB_PATH").expect("ORT_DYLIB_PATH");
    let env = ort::init_from(std::path::Path::new(&dylib)).unwrap().with_name("embed-bench").build().unwrap();
    let t0 = Instant::now();
    let mut sess = ort::session::Session::builder(&env)
        .unwrap()
        .with_intra_threads(threads)
        .unwrap()
        .commit_from_file(&a[1])
        .unwrap();
    println!("load_s {:.2} rss_mb {:.0}", t0.elapsed().as_secs_f64(), vm_hwm_mb());
    for &len in &lens {
        let mut times = Vec::with_capacity(reps);
        for r in 0..reps + 1 {
            let ids = Array2::<i64>::from_shape_fn((1, len), |(_, j)| 1000 + (j as i64 * 37) % 50000);
            let mask = Array2::<i64>::ones((1, len));
            let pos = Array2::<i64>::from_shape_fn((1, len), |(_, j)| j as i64);
            let t = Instant::now();
            let mut feeds: Vec<(String, ort::value::DynValue)> = vec![
                ("input_ids".into(), Tensor::from_array(ids).unwrap().into_dyn()),
                ("attention_mask".into(), Tensor::from_array(mask).unwrap().into_dyn()),
                ("position_ids".into(), Tensor::from_array(pos).unwrap().into_dyn()),
            ];
            for layer in 0..28 {
                for kv in ["key", "value"] {
                    let v = if f16 {
                        Tensor::from_array(ndarray::Array4::<half::f16>::from_elem((1, 8, 0, 128), half::f16::ZERO)).unwrap().into_dyn()
                    } else {
                        Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 8, 0, 128))).unwrap().into_dyn()
                    };
                    feeds.push((format!("past_key_values.{layer}.{kv}"), v));
                }
            }
            let out = sess.run(feeds).unwrap();
            let shape = out[0].shape().to_vec();
            assert!(shape.iter().product::<i64>() > 0, "empty output");
            if r > 0 {
                times.push(t.elapsed().as_secs_f64() * 1000.0); // first run is warm-up
            }
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!(
            "len {len} threads {threads} median_ms {:.0} p90_ms {:.0} rss_mb {:.0}",
            times[times.len() / 2],
            times[(times.len() * 9 / 10).min(times.len() - 1)],
            vm_hwm_mb()
        );
    }
}
