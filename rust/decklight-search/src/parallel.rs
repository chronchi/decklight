//! Speed plan E6: run independent engine replays on scoped threads.
//!
//! Every job owns its `Env` (one engine, separate envs, exactly the Python pool) and the result is written back by
//! index, so the output does not depend on scheduling: it is bit-identical to the sequential loop.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `f(0..n)` on up to `threads` scoped threads, results in index order. `threads <= 1` is the sequential loop.
pub fn map<T, F>(threads: usize, n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    if threads <= 1 || n <= 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    let workers = threads.min(n);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let v = f(i);
                    out.lock().unwrap()[i] = Some(v);
                }
            });
        }
    });
    out.into_inner()
        .unwrap()
        .into_iter()
        .map(|x| x.expect("parallel::map: a job produced no result"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn keeps_order_and_runs_all_jobs() {
        let calls = AtomicUsize::new(0);
        let out = map(4, 10, |i| {
            calls.fetch_add(1, Ordering::SeqCst);
            i * i
        });
        assert_eq!(out, (0..10).map(|i| i * i).collect::<Vec<_>>());
        assert_eq!(calls.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn sequential_path_matches() {
        assert_eq!(map(1, 5, |i| i + 1), vec![1, 2, 3, 4, 5]);
        assert_eq!(map(4, 0, |i| i), Vec::<usize>::new());
    }
}
