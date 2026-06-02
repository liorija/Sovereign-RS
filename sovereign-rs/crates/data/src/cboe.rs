//! CBOE daily-history JSON provider (the Python VIX/VVIX `_cboe_fetch`).
//!
//! CBOE serves `{"data": [["YYYY-MM-DD", open, high, low, close], ...]}` with no
//! auth and no rate-limiting — the most reliable VIX source.

use chrono::NaiveDate;
use serde::Deserialize;

use sovereign_core::error::{Result, SovereignError};

use crate::provider::PriceBar;

#[derive(Debug, Deserialize)]
struct CboeResponse {
    data: Vec<(String, f64, f64, f64, f64)>,
}

/// Parse a CBOE history JSON payload into bars (volume is 0 for an index).
pub fn parse_cboe_json(body: &str) -> Result<Vec<PriceBar>> {
    let resp: CboeResponse = serde_json::from_str(body).map_err(|e| SovereignError::Serde {
        context: "cboe".into(),
        source: e,
    })?;
    Ok(resp
        .data
        .into_iter()
        .map(|(date, o, h, l, c)| {
            let ts = NaiveDate::parse_from_str(&date, "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|dt| dt.and_utc().timestamp())
                .unwrap_or(0);
            PriceBar {
                ts,
                open: o,
                high: h,
                low: l,
                close: c,
                volume: 0.0,
            }
        })
        .collect())
}

/// The latest close (used as the spot VIX/VVIX value).
pub fn latest_close(bars: &[PriceBar]) -> Option<f64> {
    bars.last().map(|b| b.close)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str =
        r#"{"data":[["2026-05-28",17.1,18.0,16.9,17.5],["2026-05-29",17.5,17.9,16.8,16.95]]}"#;

    #[test]
    fn parses_cboe_history() {
        let bars = parse_cboe_json(SAMPLE).unwrap();
        assert_eq!(bars.len(), 2);
        assert!((latest_close(&bars).unwrap() - 16.95).abs() < 1e-9);
    }

    #[test]
    fn malformed_json_errors() {
        assert!(parse_cboe_json("{not json}").is_err());
    }
}
