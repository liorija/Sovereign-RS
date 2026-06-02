//! Stooq CSV price provider (the Python `StooqEngine`).
//!
//! The CSV parser is a pure function tested offline; only the live fetch needs
//! the network.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::NaiveDate;

use sovereign_core::domain::Instrument;
use sovereign_core::error::{Result, SovereignError};

use crate::http::HttpClient;
use crate::provider::{DataProvider, PriceBar};

/// Parse a Stooq daily CSV (`Date,Open,High,Low,Close,Volume`) into bars.
/// Rows with non-numeric fields (e.g. `N/D`) are skipped.
pub fn parse_stooq_csv(csv: &str) -> Result<Vec<PriceBar>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv.as_bytes());
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| SovereignError::data("stooq", e.to_string()))?;
        if rec.len() < 5 {
            continue;
        }
        let num = |idx: usize| rec.get(idx).and_then(|s| s.parse::<f64>().ok());
        if let (Some(o), Some(h), Some(l), Some(c)) = (num(1), num(2), num(3), num(4)) {
            let volume = num(5).unwrap_or(0.0);
            let ts = rec
                .get(0)
                .and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|dt| dt.and_utc().timestamp())
                .unwrap_or(0);
            out.push(PriceBar {
                ts,
                open: o,
                high: h,
                low: l,
                close: c,
                volume,
            });
        }
    }
    Ok(out)
}

/// Live Stooq provider.
#[derive(Debug, Clone)]
pub struct StooqProvider {
    http: Arc<HttpClient>,
}

impl StooqProvider {
    pub fn new(http: Arc<HttpClient>) -> Self {
        Self { http }
    }

    fn url(symbol: &str) -> String {
        format!("https://stooq.com/q/d/l/?s={}&i=d", symbol.to_lowercase())
    }
}

#[async_trait]
impl DataProvider for StooqProvider {
    fn name(&self) -> &str {
        "stooq"
    }

    async fn fetch_series(
        &self,
        instrument: &Instrument,
        _lookback_days: u32,
    ) -> Result<Vec<PriceBar>> {
        let csv = self
            .http
            .get_text(&Self::url(&instrument.yf_ticker()))
            .await?;
        parse_stooq_csv(&csv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "Date,Open,High,Low,Close,Volume\n\
        2026-05-28,100.0,101.5,99.5,101.0,1000000\n\
        2026-05-29,101.0,102.0,100.5,N/D,0\n\
        2026-05-30,101.0,103.0,100.0,102.5,1200000\n";

    #[test]
    fn parses_and_skips_bad_rows() {
        let bars = parse_stooq_csv(SAMPLE).unwrap();
        assert_eq!(bars.len(), 2); // the N/D row is dropped
        assert_eq!(bars[0].close, 101.0);
        assert!(bars[0].is_sane());
        assert!(bars[0].ts > 0);
    }

    #[test]
    fn empty_csv_is_ok_empty() {
        let bars = parse_stooq_csv("Date,Open,High,Low,Close,Volume\n").unwrap();
        assert!(bars.is_empty());
    }
}
