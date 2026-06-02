//! Structured, span-based telemetry.
//!
//! Directive compliance: **no `log`/`println!`** for operational events. We use
//! `tracing` so latency can be measured across `async` boundaries down to the
//! microsecond. Spans nest, so you can see exactly *where* a 2 ms stall happened
//! (e.g. inside the dark-pool fetcher vs. the Markov transition-matrix solve).
//!
//! # Quick start
//! ```no_run
//! sovereign_core::telemetry::init();          // human-readable, RUST_LOG-driven
//! // or: sovereign_core::telemetry::init_json();  // machine-ingestible
//! ```

use std::time::Instant;

use tracing::Level;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Build an [`EnvFilter`] from `RUST_LOG`, defaulting to `info` for the whole
/// stack (and `debug` for our own crates) when the variable is unset.
fn default_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info,sovereign_core=debug,sovereign_quant=debug,sovereign_engine=debug")
    })
}

/// Initialize human-readable telemetry with microsecond timestamps and span
/// open/close events. Idempotent: a second call is a no-op.
pub fn init() {
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_level(true)
        .with_span_events(fmt::format::FmtSpan::CLOSE) // emit span duration on close
        .with_timer(fmt::time::uptime());

    let _ = tracing_subscriber::registry()
        .with(default_filter())
        .with(fmt_layer)
        .try_init();
}

/// Initialize JSON telemetry (one event per line) for log shippers / dashboards.
pub fn init_json() {
    let fmt_layer = fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_span_events(fmt::format::FmtSpan::CLOSE);

    let _ = tracing_subscriber::registry()
        .with(default_filter())
        .with(fmt_layer)
        .try_init();
}

/// RAII latency probe: logs the elapsed time in **microseconds** when dropped.
///
/// Use it for hot paths where a full `#[tracing::instrument]` span is overkill
/// but you still want precise timing:
/// ```
/// # fn fetch() {}
/// {
///     let _g = sovereign_core::telemetry::LatencyGuard::new("dark_pool_fetch");
///     fetch();
/// } // <- elapsed µs logged here
/// ```
#[derive(Debug)]
#[must_use = "the guard measures until it is dropped; bind it to a variable"]
pub struct LatencyGuard {
    label: &'static str,
    start: Instant,
    level: Level,
}

impl LatencyGuard {
    /// Start a probe at `DEBUG` level.
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            start: Instant::now(),
            level: Level::DEBUG,
        }
    }

    /// Start a probe at `INFO` level (for coarser, always-on measurements).
    pub fn info(label: &'static str) -> Self {
        Self {
            label,
            start: Instant::now(),
            level: Level::INFO,
        }
    }

    /// Read the elapsed microseconds without dropping the guard.
    pub fn elapsed_us(&self) -> u128 {
        self.start.elapsed().as_micros()
    }
}

impl Drop for LatencyGuard {
    fn drop(&mut self) {
        let us = self.start.elapsed().as_micros();
        match self.level {
            Level::INFO => tracing::info!(probe = self.label, latency_us = us, "latency"),
            _ => tracing::debug!(probe = self.label, latency_us = us, "latency"),
        }
    }
}
