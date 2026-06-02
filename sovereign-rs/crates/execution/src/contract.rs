//! Maps a typed [`Instrument`] to a broker contract specification.
//!
//! Port of `v329_universe_expansion.IBKR_CONTRACT_MAP` + the futures spec and
//! front-month logic in `sovereign_offhours_fix.py`, but type-driven: an
//! [`Instrument::Index`] simply has no contract (data-only), enforced by the
//! `Option` return rather than a scattered `DATA_ONLY_TICKERS` set.

use chrono::{Datelike, NaiveDate};
use serde::Serialize;

use sovereign_core::domain::Instrument;

/// Broker security type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SecType {
    Stock,
    Future,
    Forex,
    Crypto,
}

/// A fully-qualified contract ready for the order router.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContractSpec {
    pub sec_type: SecType,
    pub symbol: String,
    pub exchange: String,
    pub currency: String,
    pub multiplier: Option<f64>,
    /// Front-month `YYYYMM` for futures.
    pub last_trade_ym: Option<String>,
}

/// IBKR futures spec: `root → (ib_symbol, exchange, currency, multiplier)`.
fn futures_spec(root: &str) -> Option<(&'static str, &'static str, &'static str, f64)> {
    Some(match root {
        "ES" => ("ES", "CME", "USD", 50.0),
        "NQ" => ("NQ", "CME", "USD", 20.0),
        "RTY" => ("RTY", "CME", "USD", 50.0),
        "YM" => ("YM", "CBOT", "USD", 5.0),
        "MES" => ("MES", "CME", "USD", 5.0),
        "MNQ" => ("MNQ", "CME", "USD", 2.0),
        "GC" => ("GC", "COMEX", "USD", 100.0),
        "SI" => ("SI", "COMEX", "USD", 5000.0),
        "HG" => ("HG", "COMEX", "USD", 25000.0),
        "CL" => ("CL", "NYMEX", "USD", 1000.0),
        "NG" => ("NG", "NYMEX", "USD", 10000.0),
        "ZB" => ("ZB", "CBOT", "USD", 1000.0),
        "ZN" => ("ZN", "CBOT", "USD", 1000.0),
        "6E" => ("EUR", "CME", "USD", 125000.0),
        "6J" => ("JPY", "CME", "USD", 12500000.0),
        "BTC" => ("BTC", "CME", "USD", 5.0),
        _ => return None,
    })
}

/// Contract months (1-indexed) for a futures root.
fn expiry_months(root: &str) -> Vec<u32> {
    match root {
        "GC" => vec![2, 4, 6, 8, 10, 12],
        "SI" | "HG" => vec![3, 5, 7, 9, 12],
        "CL" | "NG" => (1..=12).collect(),
        // Equity index, bonds, FX futures: quarterly.
        "ES" | "NQ" | "RTY" | "YM" | "MES" | "MNQ" | "ZB" | "ZN" | "6E" | "6J" => {
            vec![3, 6, 9, 12]
        }
        "BTC" => (1..=12).collect(),
        _ => vec![3, 6, 9, 12],
    }
}

/// Compute the front-month `YYYYMM`, skipping any contract within
/// `lookahead_days` of expiry. Returns `None` for unknown roots.
pub fn front_month(root: &str, today: NaiveDate, lookahead_days: i64) -> Option<String> {
    let months = expiry_months(root);
    if months.is_empty() {
        return None;
    }
    let cutoff = today + chrono::Duration::days(lookahead_days);
    let cutoff_ym = cutoff.year() * 100 + cutoff.month() as i32;
    for year in today.year()..=today.year() + 2 {
        for &m in &months {
            let ym = year * 100 + m as i32;
            if ym >= cutoff_ym {
                return Some(format!("{ym:06}"));
            }
        }
    }
    None
}

/// Route an instrument to a broker contract. `Index` instruments are data-only
/// and return `None`.
pub fn route(instrument: &Instrument, today: NaiveDate) -> Option<ContractSpec> {
    match instrument {
        Instrument::Index { .. } => None,
        Instrument::Equity { symbol } => Some(ContractSpec {
            sec_type: SecType::Stock,
            symbol: symbol.clone(),
            exchange: "SMART".into(),
            currency: "USD".into(),
            multiplier: None,
            last_trade_ym: None,
        }),
        Instrument::Forex { base, quote } => Some(ContractSpec {
            sec_type: SecType::Forex,
            symbol: base.clone(),
            exchange: "IDEALPRO".into(),
            currency: quote.clone(),
            multiplier: None,
            last_trade_ym: None,
        }),
        Instrument::Crypto { symbol } => Some(ContractSpec {
            sec_type: SecType::Crypto,
            symbol: symbol.trim_end_matches("-USD").to_string(),
            exchange: "PAXOS".into(),
            currency: "USD".into(),
            multiplier: None,
            last_trade_ym: None,
        }),
        Instrument::Future { root, .. } => {
            let (sym, exch, curr, mult) = futures_spec(root)?;
            Some(ContractSpec {
                sec_type: SecType::Future,
                symbol: sym.into(),
                exchange: exch.into(),
                currency: curr.into(),
                multiplier: Some(mult),
                last_trade_ym: front_month(root, today, 15),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 6, 2).unwrap()
    }

    #[test]
    fn equity_routes_to_smart_stock() {
        let c = route(&Instrument::parse("AAPL"), today()).unwrap();
        assert_eq!(c.sec_type, SecType::Stock);
        assert_eq!(c.exchange, "SMART");
    }

    #[test]
    fn future_gets_multiplier_and_front_month() {
        let c = route(&Instrument::parse("ES=F"), today()).unwrap();
        assert_eq!(c.sec_type, SecType::Future);
        assert_eq!(c.exchange, "CME");
        assert_eq!(c.multiplier, Some(50.0));
        // ES is quarterly (Mar/Jun/Sep/Dec). On 2026-06-02 the front month is
        // still June (202606) — it only rolls once within 15 days of expiry.
        assert_eq!(c.last_trade_ym.as_deref(), Some("202606"));
    }

    #[test]
    fn forex_splits_pair() {
        let c = route(&Instrument::parse("EURUSD=X"), today()).unwrap();
        assert_eq!(c.sec_type, SecType::Forex);
        assert_eq!(c.symbol, "EUR");
        assert_eq!(c.currency, "USD");
    }

    #[test]
    fn index_is_data_only() {
        assert!(route(&Instrument::parse("^VIX"), today()).is_none());
    }
}
