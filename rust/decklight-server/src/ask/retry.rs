//! Retries in one place (docs/plans/ask-tab.md §Core design 4): retryable provider errors are tried again with
//! exponential backoff, honouring `Retry-After`; fatal errors and cancels return at once. Sleeping and the clock are
//! passed in, so tests run without waiting.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::provider::ProviderError;

#[derive(Clone, Debug)]
pub struct Policy {
    /// Tries in all, the first included.
    pub attempts: u32,
    pub base: Duration,
    pub max_delay: Duration,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy { attempts: 4, base: Duration::from_secs(1), max_delay: Duration::from_secs(20) }
    }
}

impl Policy {
    /// The wait before try `n + 1` after try `n` failed (n from 1): the server's `Retry-After` when given, else
    /// base · 2^(n-1); both capped at `max_delay` so a huge Retry-After does not freeze the chat.
    pub fn delay(&self, n: u32, retry_after: Option<Duration>) -> Duration {
        let backoff = self.base.saturating_mul(1u32 << n.saturating_sub(1).min(16));
        retry_after.unwrap_or(backoff).min(self.max_delay)
    }
}

/// What the UI is told before each wait ("Retrying in 4 s").
#[derive(Clone, Debug, PartialEq)]
pub struct RetryNote {
    /// The try that comes next (2 = the first retry).
    pub attempt: u32,
    pub delay: Duration,
    pub reason: String,
}

/// Run `f` until it succeeds, fails for good, is cancelled, or runs out of tries (the last error is returned).
/// A `Retry-After` longer than `max_delay` returns the error at once: the wait is beyond what a chat sits through
/// (OpenCode Zen answers a spent free quota with 429 and a long Retry-After, NOTEBOOK 2026-11-14).
/// `sleep` waits in steps and returns false when cancelled during the wait.
pub fn run<T>(
    policy: &Policy,
    cancel: &AtomicBool,
    mut f: impl FnMut() -> Result<T, ProviderError>,
    mut on_retry: impl FnMut(&RetryNote),
    mut sleep: impl FnMut(Duration, &AtomicBool) -> bool,
) -> Result<T, ProviderError> {
    let mut n = 1;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(ProviderError::Cancelled);
        }
        match f() {
            Ok(v) => return Ok(v),
            Err(ProviderError::Retryable { msg, retry_after })
                if n < policy.attempts && retry_after.is_none_or(|d| d <= policy.max_delay) =>
            {
                let delay = policy.delay(n, retry_after);
                on_retry(&RetryNote { attempt: n + 1, delay, reason: msg });
                if !sleep(delay, cancel) {
                    return Err(ProviderError::Cancelled);
                }
                n += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// The real sleep: 100 ms steps, so a cancel lands quickly.
pub fn sleep_cancellable(d: Duration, cancel: &AtomicBool) -> bool {
    let step = Duration::from_millis(100);
    let mut left = d;
    while !left.is_zero() {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        let s = left.min(step);
        std::thread::sleep(s);
        left -= s;
    }
    !cancel.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retry(msg: &str, after: Option<u64>) -> ProviderError {
        ProviderError::Retryable { msg: msg.into(), retry_after: after.map(Duration::from_secs) }
    }

    #[test]
    fn backoff_doubles_honours_retry_after_and_caps() {
        let p = Policy::default();
        assert_eq!(p.delay(1, None), Duration::from_secs(1));
        assert_eq!(p.delay(3, None), Duration::from_secs(4));
        assert_eq!(p.delay(1, Some(Duration::from_secs(7))), Duration::from_secs(7));
        assert_eq!(p.delay(1, Some(Duration::from_secs(3600))), Duration::from_secs(20));
        assert_eq!(p.delay(40, None), Duration::from_secs(20), "no overflow on many tries");
    }

    #[test]
    fn retries_then_succeeds_and_reports_each_wait() {
        let cancel = AtomicBool::new(false);
        let mut script = vec![Err(retry("HTTP 429", Some(2))), Err(retry("HTTP 503", None)), Ok(42)].into_iter();
        let (mut notes, mut slept) = (Vec::new(), Vec::new());
        let out = run(&Policy::default(), &cancel, || script.next().unwrap(), |n| notes.push(n.clone()), |d, _| {
            slept.push(d);
            true
        });
        assert_eq!(out, Ok(42));
        assert_eq!(slept, vec![Duration::from_secs(2), Duration::from_secs(2)]);
        assert_eq!((notes[0].attempt, notes[1].attempt), (2, 3));
        assert_eq!(notes[1].reason, "HTTP 503");
    }

    #[test]
    fn fatal_gives_up_at_once_and_tries_are_bounded() {
        let cancel = AtomicBool::new(false);
        let mut calls = 0;
        let fatal = ProviderError::Fatal { status: Some(401), msg: "key rejected".into() };
        let out: Result<(), _> = run(&Policy::default(), &cancel, || {
            calls += 1;
            Err(fatal.clone())
        }, |_| {}, |_, _| true);
        assert_eq!((out, calls), (Err(fatal), 1));
        let mut calls = 0;
        let out: Result<(), _> = run(&Policy::default(), &cancel, || {
            calls += 1;
            Err(retry("HTTP 500", None))
        }, |_| {}, |_, _| true);
        assert!(matches!(out, Err(ProviderError::Retryable { .. })));
        assert_eq!(calls, 4);
    }

    #[test]
    fn a_retry_after_beyond_the_cap_gives_up_at_once() {
        let cancel = AtomicBool::new(false);
        let mut calls = 0;
        let out: Result<(), _> = run(&Policy::default(), &cancel, || {
            calls += 1;
            Err(retry("HTTP 429: Free usage limit reached", Some(3600)))
        }, |_| panic!("no wait"), |_, _| true);
        assert_eq!(calls, 1);
        assert!(matches!(out, Err(ProviderError::Retryable { .. })));
    }

    #[test]
    fn cancel_during_a_wait_stops() {
        let cancel = AtomicBool::new(false);
        let out: Result<(), _> = run(&Policy::default(), &cancel, || Err(retry("HTTP 500", None)), |_| {}, |_, c| {
            c.store(true, Ordering::Relaxed);
            false
        });
        assert_eq!(out, Err(ProviderError::Cancelled));
    }
}
