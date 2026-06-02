//! The `DataProvider` abstraction.
//!
//! Every price/news/macro source (IBKR, Stooq, CBOE, yfinance, FRED, SEC-EDGAR)
//! implements this one async trait, so the [`crate::fallback::FallbackChain`] can
//! treat them uniformly — the structural replacement for the Python web of
//! per-source `_score_*` functions and monkey-patched fallbacks.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use sovereign_core::domain::Instrument;
use sovereign_core::error::Result;

/// A single OHLCV bar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PriceBar {
    /// Unix timestamp (seconds).
    pub ts: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl PriceBar {
    /// Basic semantic validity (port of `DataSanityGuard`): finite, positive,
    /// and `low <= {open,close} <= high`.
    pub fn is_sane(&self) -> bool {
        let fields = [self.open, self.high, self.low, self.close];
        fields.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.low <= self.high
            && self.low <= self.open
            && self.open <= self.high
            && self.low <= self.close
            && self.close <= self.high
            && self.volume.is_finite()
            && self.volume >= 0.0
    }
}

/// Health of a provider, surfaced to the dashboard (`SourceHealth`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Ok,
    Degraded,
    Down,
}

/// An async source of price bars for an [`Instrument`].
#[async_trait]
pub trait DataProvider: Send + Sync {
    /// Stable identifier, e.g. `"stooq"`, `"ibkr"`, `"cboe"`.
    fn name(&self) -> &str;

    /// Fetch up to `lookback_days` of daily bars, oldest-first.
    async fn fetch_series(
        &self,
        instrument: &Instrument,
        lookback_days: u32,
    ) -> Result<Vec<PriceBar>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sane_bar_passes_corrupt_bar_fails() {
        let good = PriceBar {
            ts: 0,
            open: 10.0,
            high: 11.0,
            low: 9.5,
            close: 10.5,
            volume: 1e6,
        };
        assert!(good.is_sane());
        let bad = PriceBar {
            ts: 0,
            open: 10.0,
            high: 9.0,
            low: 11.0,
            close: 0.0,
            volume: -1.0,
        };
        assert!(!bad.is_sane());
    }
}
