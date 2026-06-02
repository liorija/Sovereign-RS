//! Core domain model.
//!
//! This is where the Python system's stringly-typed universe ("ES=F", "EURUSD=X",
//! "BTC-USD", "^VIX", ...) and its 8 regime labels become **type-safe** Rust
//! enums. Illegal states (e.g. routing an index as a stock order) become
//! unrepresentable instead of being caught by a runtime `if sym.endswith(...)`.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Broad asset class, used by the execution router to pick a contract type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetClass {
    Equity,
    Future,
    Forex,
    Crypto,
    Index,
}

/// A tradeable (or data-only) instrument, parsed from yfinance-style tickers.
///
/// Mirrors `v329_universe_expansion.IBKR_CONTRACT_MAP` and the `_is_nonequity`
/// helpers from `sovereign_offhours_fix.py`, but enforced by the compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Instrument {
    /// Common stock or ETF, e.g. `AAPL`, `SPY`, `IBIT`.
    Equity { symbol: String },
    /// CME/COMEX/NYMEX/CBOT future. `root` is the IBKR root (`ES`), `yf` the
    /// yfinance ticker (`ES=F`).
    Future { root: String, yf: String },
    /// Spot FX pair, e.g. `EUR/USD` from `EURUSD=X`.
    Forex { base: String, quote: String },
    /// Crypto spot, e.g. `BTC-USD`.
    Crypto { symbol: String },
    /// Non-tradeable index / yield used for signals only, e.g. `^VIX`, `^TNX`.
    Index { symbol: String },
}

impl Instrument {
    /// Parse a yfinance-style ticker into a typed instrument.
    ///
    /// Rules (in priority order):
    /// `^...` → Index · `...=F` → Future · `...=X` → Forex ·
    /// `...-USD`/`-USDT`/`-BTC` → Crypto · otherwise → Equity.
    pub fn parse(raw: &str) -> Self {
        let t = raw.trim();
        let upper = t.to_ascii_uppercase();

        if let Some(sym) = upper.strip_prefix('^') {
            return Instrument::Index {
                symbol: format!("^{sym}"),
            };
        }
        if let Some(root) = upper.strip_suffix("=F") {
            return Instrument::Future {
                root: root.to_string(),
                yf: upper.clone(),
            };
        }
        if let Some(pair) = upper.strip_suffix("=X") {
            // A 6-char pair like EURUSD splits 3/3; otherwise quote defaults USD.
            let (base, quote) = if pair.len() == 6 {
                (pair[..3].to_string(), pair[3..].to_string())
            } else {
                (pair.to_string(), "USD".to_string())
            };
            return Instrument::Forex { base, quote };
        }
        for suf in ["-USD", "-USDT", "-BTC"] {
            if upper.ends_with(suf) {
                return Instrument::Crypto {
                    symbol: upper.clone(),
                };
            }
        }
        Instrument::Equity { symbol: upper }
    }

    /// The broad asset class.
    pub fn asset_class(&self) -> AssetClass {
        match self {
            Instrument::Equity { .. } => AssetClass::Equity,
            Instrument::Future { .. } => AssetClass::Future,
            Instrument::Forex { .. } => AssetClass::Forex,
            Instrument::Crypto { .. } => AssetClass::Crypto,
            Instrument::Index { .. } => AssetClass::Index,
        }
    }

    /// Reconstruct the canonical yfinance ticker.
    pub fn yf_ticker(&self) -> String {
        match self {
            Instrument::Equity { symbol } | Instrument::Crypto { symbol } => symbol.clone(),
            Instrument::Index { symbol } => symbol.clone(),
            Instrument::Future { yf, .. } => yf.clone(),
            Instrument::Forex { base, quote } => format!("{base}{quote}=X"),
        }
    }

    /// Index/yield instruments are signal-only and can never be ordered.
    /// (Replaces the `DATA_ONLY_TICKERS` set + scattered guards in Python.)
    pub fn is_orderable(&self) -> bool {
        !matches!(self, Instrument::Index { .. })
    }

    /// Non-equity instruments follow CME/forex/crypto hours, not equity hours.
    /// (Replaces `_is_nonequity` from `sovereign_offhours_fix.py`.)
    pub fn is_non_equity(&self) -> bool {
        !matches!(self, Instrument::Equity { .. })
    }
}

impl fmt::Display for Instrument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.yf_ticker())
    }
}

/// The 8 market regimes from V319's regime engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    Bull,
    Bear,
    Sideways,
    Crisis,
    Recovery,
    Goldilocks,
    Reflation,
    Stagflation,
}

impl Regime {
    /// All regimes, in canonical order (handy for Markov state indexing).
    pub const ALL: [Regime; 8] = [
        Regime::Bull,
        Regime::Bear,
        Regime::Sideways,
        Regime::Crisis,
        Regime::Recovery,
        Regime::Goldilocks,
        Regime::Reflation,
        Regime::Stagflation,
    ];

    /// Stable string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::Bull => "BULL",
            Regime::Bear => "BEAR",
            Regime::Sideways => "SIDEWAYS",
            Regime::Crisis => "CRISIS",
            Regime::Recovery => "RECOVERY",
            Regime::Goldilocks => "GOLDILOCKS",
            Regime::Reflation => "REFLATION",
            Regime::Stagflation => "STAGFLATION",
        }
    }

    /// Index into [`Regime::ALL`] — the Markov chain state id.
    pub fn index(&self) -> usize {
        Regime::ALL.iter().position(|r| r == self).unwrap_or(0)
    }

    /// Map a state id back to a regime, saturating at the last variant.
    pub fn from_index(i: usize) -> Regime {
        Regime::ALL.get(i).copied().unwrap_or(Regime::Sideways)
    }

    /// Risk-on allocation multiplier (port of the `_alloc()` / megafix mapping:
    /// GOLDILOCKS/RECOVERY/REFLATION ≈ BULL, STAGFLATION ≈ defensive, CRISIS small).
    pub fn allocation_multiplier(&self) -> f64 {
        match self {
            Regime::Bull | Regime::Goldilocks | Regime::Recovery => 1.0,
            Regime::Reflation => 0.9,
            Regime::Sideways => 0.6,
            Regime::Bear => 0.4,
            Regime::Stagflation => 0.3,
            Regime::Crisis => 0.2,
        }
    }
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// US equity trading session (from `SessionDetector`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Session {
    Premarket,
    Regular,
    Postmarket,
    Closed,
}

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
    Short,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_asset_class() {
        assert_eq!(Instrument::parse("AAPL").asset_class(), AssetClass::Equity);
        assert_eq!(Instrument::parse("ES=F").asset_class(), AssetClass::Future);
        assert_eq!(
            Instrument::parse("EURUSD=X").asset_class(),
            AssetClass::Forex
        );
        assert_eq!(
            Instrument::parse("BTC-USD").asset_class(),
            AssetClass::Crypto
        );
        assert_eq!(Instrument::parse("^VIX").asset_class(), AssetClass::Index);
    }

    #[test]
    fn forex_pair_splits_correctly() {
        match Instrument::parse("eurusd=x") {
            Instrument::Forex { base, quote } => {
                assert_eq!(base, "EUR");
                assert_eq!(quote, "USD");
            }
            other => panic!("expected forex, got {other:?}"),
        }
    }

    #[test]
    fn index_is_not_orderable() {
        assert!(!Instrument::parse("^TNX").is_orderable());
        assert!(Instrument::parse("AAPL").is_orderable());
    }

    #[test]
    fn regime_index_roundtrip() {
        for r in Regime::ALL {
            assert_eq!(Regime::from_index(r.index()), r);
        }
    }
}
