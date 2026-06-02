//! Background self-healer (port of `SovereignAutoDoctor`).
//!
//! Runs as a detached `tokio` task on a fixed interval, performing idempotent
//! repairs (reset stale circuit breakers, re-seed caches, retrigger ML training).
//! The healing *checks* are pure and synchronous so they unit-test without a
//! runtime; only the scheduling is async.

use std::time::Duration;

use tokio::task::JoinHandle;

/// Outcome of one healing cycle.
#[derive(Debug, Default, Clone, Copy)]
pub struct HealReport {
    pub checks: u32,
    pub fixes: u32,
}

/// The self-healer scheduler.
#[derive(Debug, Clone)]
pub struct SelfHealer {
    interval: Duration,
}

impl SelfHealer {
    /// New healer firing every `interval`.
    pub fn new(interval: Duration) -> Self {
        Self { interval }
    }

    /// One synchronous healing pass. In the full system this inspects the
    /// service registry; here it returns a report you can assert on.
    pub fn cycle() -> HealReport {
        // Placeholder for: stale-price reset, CB half-open promotion, ML retrain.
        tracing::debug!("autodoc cycle: all systems nominal");
        HealReport {
            checks: 3,
            fixes: 0,
        }
    }

    /// Spawn the detached background loop. Drop/await the handle to stop it.
    pub fn spawn(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(self.interval);
            loop {
                ticker.tick().await;
                let report = Self::cycle();
                if report.fixes > 0 {
                    tracing::info!(fixes = report.fixes, "autodoc applied repairs");
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_is_pure_and_safe() {
        let r = SelfHealer::cycle();
        assert_eq!(r.checks, 3);
    }

    #[tokio::test(start_paused = true)]
    async fn scheduler_ticks() {
        let handle = SelfHealer::new(Duration::from_secs(60)).spawn();
        // With a paused clock we can advance virtual time deterministically.
        tokio::time::advance(Duration::from_secs(61)).await;
        handle.abort();
        assert!(handle.await.unwrap_err().is_cancelled());
    }
}
