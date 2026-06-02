//! Curated multi-asset ticker lists (port of `v329_universe_expansion`).
//!
//! The full equity universe (~11k US listings) is loaded at runtime from a feed
//! or file; these constants are the always-present multi-asset overlay plus the
//! liquid core, and a synthetic generator is provided so the round-robin
//! scanner can be exercised at full 11k scale offline.

/// Sector / broad-market ETFs.
pub const SECTOR_ETFS: &[&str] = &[
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU",
    "XLRE", "XLC", "SMH", "IBB", "XRT", "IYT",
];

/// Equity-index, metals, energy, bond and crypto futures (yfinance form).
pub const FUTURES: &[&str] = &[
    "ES=F", "NQ=F", "YM=F", "RTY=F", "MES=F", "MNQ=F", "GC=F", "SI=F", "HG=F", "CL=F", "NG=F",
    "ZB=F", "ZN=F", "6E=F", "6J=F", "BTC=F", "ETH=F",
];

/// Major spot FX pairs.
pub const FOREX: &[&str] = &[
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURJPY=X",
    "USDIDR=X",
];

/// Spot crypto + crypto-equity proxies.
pub const CRYPTO: &[&str] = &[
    "BTC-USD", "ETH-USD", "IBIT", "FBTC", "ETHA", "COIN", "MSTR", "MARA", "RIOT",
];

/// International / regional ETFs.
pub const INTERNATIONAL: &[&str] = &[
    "EEM", "VEA", "VWO", "EFA", "EWJ", "FXI", "MCHI", "INDA", "EWZ", "EWG", "EWY", "EWT", "EWC",
    "EIDO",
];

/// Volatility & macro instruments (some data-only).
pub const VOLATILITY_MACRO: &[&str] = &[
    "VXX", "UVXY", "SVXY", "^VIX", "^VVIX", "^TNX", "UUP", "TLT", "HYG", "LQD", "GLD", "SLV",
];

/// The always-present multi-asset overlay (deduplicated by the [`crate::Universe`]).
pub fn multi_asset() -> Vec<String> {
    let mut out = Vec::new();
    for list in [
        SECTOR_ETFS,
        FUTURES,
        FOREX,
        CRYPTO,
        INTERNATIONAL,
        VOLATILITY_MACRO,
    ] {
        out.extend(list.iter().map(|s| s.to_string()));
    }
    out
}
