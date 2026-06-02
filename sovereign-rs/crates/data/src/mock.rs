//! A deterministic in-memory [`DataProvider`] for tests and offline runs.

use async_trait::async_trait;

use sovereign_core::domain::Instrument;
use sovereign_core::error::{Result, SovereignError};

use crate::provider::{DataProvider, PriceBar};

/// Serves a fixed set of bars (or always fails, for fallback testing).
#[derive(Debug, Clone)]
pub struct MockProvider {
    name: String,
    bars: Vec<PriceBar>,
    fail: bool,
}

impl MockProvider {
    /// A provider that returns `bars`.
    pub fn ok(name: impl Into<String>, bars: Vec<PriceBar>) -> Self {
        Self {
            name: name.into(),
            bars,
            fail: false,
        }
    }

    /// A provider that always errors (to exercise fallback chains).
    pub fn failing(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bars: Vec::new(),
            fail: true,
        }
    }

    /// A synthetic sine-wave series of `n` daily bars around `base`.
    pub fn synthetic(name: impl Into<String>, n: usize, base: f64) -> Self {
        let bars = (0..n)
            .map(|i| {
                let c = base + (i as f64 * 0.2).sin() * base * 0.02;
                PriceBar {
                    ts: i as i64 * 86_400,
                    open: c,
                    high: c * 1.01,
                    low: c * 0.99,
                    close: c,
                    volume: 1.0e6,
                }
            })
            .collect();
        Self::ok(name, bars)
    }
}

#[async_trait]
impl DataProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn fetch_series(
        &self,
        _instrument: &Instrument,
        _lookback_days: u32,
    ) -> Result<Vec<PriceBar>> {
        if self.fail {
            Err(SovereignError::data(self.name.clone(), "mock failure"))
        } else {
            Ok(self.bars.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn synthetic_returns_sane_bars() {
        let p = MockProvider::synthetic("mock", 50, 100.0);
        let bars = p
            .fetch_series(&Instrument::parse("AAPL"), 30)
            .await
            .unwrap();
        assert_eq!(bars.len(), 50);
        assert!(bars.iter().all(|b| b.is_sane()));
    }
}
