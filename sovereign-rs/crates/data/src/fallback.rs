//! Multi-source fallback waterfall.
//!
//! Port of the `MacroOracle` / `PriceEngine` layered-fallback pattern: try
//! providers in priority order, return the first success, and **degrade the
//! conviction multiplier** when a non-primary source had to be used (so the BFT
//! layer trusts proxy data less — the Python `CONVICTION_DEGRADED_MULTIPLIER`).

use std::sync::Arc;

use sovereign_core::domain::Instrument;
use sovereign_core::error::{Result, SovereignError};

use crate::provider::{DataProvider, PriceBar};

/// Multiplier applied to downstream conviction when a fallback source was used.
pub const DEGRADED_MULTIPLIER: f64 = 0.70;

/// The outcome of a successful waterfall fetch.
#[derive(Debug, Clone)]
pub struct FetchOutcome {
    /// The bars returned.
    pub bars: Vec<PriceBar>,
    /// Which provider supplied them.
    pub source: String,
    /// True if a non-primary (fallback) provider was used.
    pub degraded: bool,
    /// Conviction multiplier to pass downstream (1.0 primary, 0.7 degraded).
    pub conviction_mult: f64,
}

/// An ordered chain of providers tried newest-best-first.
pub struct FallbackChain {
    providers: Vec<Arc<dyn DataProvider>>,
}

impl FallbackChain {
    /// Build from an ordered list (index 0 = primary).
    pub fn new(providers: Vec<Arc<dyn DataProvider>>) -> Self {
        Self { providers }
    }

    /// Try each provider in order; return the first that yields ≥1 sane bar.
    pub async fn fetch(&self, instrument: &Instrument, lookback_days: u32) -> Result<FetchOutcome> {
        for (idx, provider) in self.providers.iter().enumerate() {
            match provider.fetch_series(instrument, lookback_days).await {
                Ok(bars) if bars.iter().any(|b| b.is_sane()) => {
                    let degraded = idx > 0;
                    return Ok(FetchOutcome {
                        bars,
                        source: provider.name().to_string(),
                        degraded,
                        conviction_mult: if degraded { DEGRADED_MULTIPLIER } else { 1.0 },
                    });
                }
                Ok(_) => {
                    tracing::debug!(provider = provider.name(), "empty/insane bars, trying next");
                }
                Err(e) => {
                    tracing::debug!(provider = provider.name(), error = %e, "provider failed, trying next");
                }
            }
        }
        Err(SovereignError::AllProvidersFailed {
            key: instrument.yf_ticker(),
            tried: self.providers.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct Always {
        name: String,
        bars: Vec<PriceBar>,
    }
    #[async_trait]
    impl DataProvider for Always {
        fn name(&self) -> &str {
            &self.name
        }
        async fn fetch_series(&self, _i: &Instrument, _d: u32) -> Result<Vec<PriceBar>> {
            Ok(self.bars.clone())
        }
    }

    struct Fails(String);
    #[async_trait]
    impl DataProvider for Fails {
        fn name(&self) -> &str {
            &self.0
        }
        async fn fetch_series(&self, _i: &Instrument, _d: u32) -> Result<Vec<PriceBar>> {
            Err(SovereignError::data(self.0.clone(), "boom"))
        }
    }

    fn bar() -> PriceBar {
        PriceBar {
            ts: 0,
            open: 10.0,
            high: 11.0,
            low: 9.5,
            close: 10.5,
            volume: 1e6,
        }
    }

    #[tokio::test]
    async fn falls_through_to_secondary_and_degrades() {
        let chain = FallbackChain::new(vec![
            Arc::new(Fails("primary".into())),
            Arc::new(Always {
                name: "secondary".into(),
                bars: vec![bar()],
            }),
        ]);
        let out = chain.fetch(&Instrument::parse("AAPL"), 30).await.unwrap();
        assert_eq!(out.source, "secondary");
        assert!(out.degraded);
        assert_eq!(out.conviction_mult, DEGRADED_MULTIPLIER);
    }

    #[tokio::test]
    async fn primary_success_is_not_degraded() {
        let chain = FallbackChain::new(vec![Arc::new(Always {
            name: "primary".into(),
            bars: vec![bar()],
        })]);
        let out = chain.fetch(&Instrument::parse("SPY"), 30).await.unwrap();
        assert!(!out.degraded);
        assert_eq!(out.conviction_mult, 1.0);
    }

    #[tokio::test]
    async fn all_fail_is_error() {
        let chain = FallbackChain::new(vec![
            Arc::new(Fails("a".into())),
            Arc::new(Fails("b".into())),
        ]);
        let res = chain.fetch(&Instrument::parse("SPY"), 30).await;
        assert!(res.is_err());
    }
}
