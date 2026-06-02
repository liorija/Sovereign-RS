//! Strongly-typed configuration, loaded from TOML + environment.
//!
//! Replaces the constellation of module-level constants (`KH_MIN_BACKTEST_WR`,
//! `MIN_CONVICTION`, tier allocations, API keys read via `os.getenv`) scattered
//! across the Python files with one validated struct.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SovereignError};

/// Top-level engine configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub risk: RiskConfig,
    pub killhouse: KillHouseConfig,
    pub montecarlo: MonteCarloConfig,
    pub data: DataConfig,
}

/// Position/risk sizing knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    /// Constant per-trade dollar risk (InverseVolSizer).
    pub dollar_risk_per_trade: f64,
    /// Hard cap: max fraction of capital in a single name.
    pub max_position_pct: f64,
    /// Drawdown ladder trip levels (e.g. -0.05, -0.10, -0.15, -0.20).
    pub drawdown_ladder: Vec<f64>,
}

/// Kill-House gate thresholds (regime-adaptive defaults live in code).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KillHouseConfig {
    /// Minimum conviction to even enter the gate funnel.
    pub min_conviction: i32,
    /// Monte-Carlo CVaR floor (Gate 3); a position must have CVaR >= this.
    pub max_monte_carlo_es: f64,
    /// VPIN ceiling (Gate 5); informed-flow above this blocks entry.
    pub vpin_ceiling: f64,
}

/// Monte-Carlo simulation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MonteCarloConfig {
    pub paths: usize,
    pub horizon_days: usize,
    pub confidence: f64,
    pub seed: u64,
}

/// Data layer settings; secrets come from env, never the TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataConfig {
    /// Polite User-Agent for SEC-EDGAR (their fair-access policy requires one).
    pub sec_user_agent: String,
    /// Per-request timeout (seconds) for HTTP fetchers.
    pub http_timeout_s: u64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            dollar_risk_per_trade: 250.0,
            max_position_pct: 0.15,
            drawdown_ladder: vec![-0.05, -0.10, -0.15, -0.20],
        }
    }
}

impl Default for KillHouseConfig {
    fn default() -> Self {
        Self {
            min_conviction: 8,
            max_monte_carlo_es: -0.08,
            vpin_ceiling: 0.65,
        }
    }
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            paths: 50_000,
            horizon_days: 7,
            confidence: 0.95,
            seed: 0xC0FFEE,
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            sec_user_agent: "sovereign-rs/0.1 (contact: ops@example.com)".to_string(),
            http_timeout_s: 10,
        }
    }
}

impl Config {
    /// Parse from a TOML string. Missing fields fall back to [`Default`].
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).map_err(|e| SovereignError::Config(e.to_string()))
    }

    /// Load from a TOML file path, or return defaults if the file is absent.
    pub fn load_or_default(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => Self::from_toml_str(&s).unwrap_or_else(|e| {
                tracing::warn!(error = %e, path, "invalid config, using defaults");
                Config::default()
            }),
            Err(_) => Config::default(),
        }
    }

    /// Validate cross-field invariants. Returns the config back on success.
    pub fn validate(self) -> Result<Self> {
        if !(0.0..=1.0).contains(&self.risk.max_position_pct) {
            return Err(SovereignError::Config(
                "risk.max_position_pct must be in [0,1]".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.montecarlo.confidence) {
            return Err(SovereignError::Config(
                "montecarlo.confidence must be in [0,1)".into(),
            ));
        }
        if self.montecarlo.paths == 0 {
            return Err(SovereignError::Config(
                "montecarlo.paths must be > 0".into(),
            ));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_toml_merges_with_defaults() {
        let cfg = Config::from_toml_str("[risk]\ndollar_risk_per_trade = 500.0\n").unwrap();
        assert_eq!(cfg.risk.dollar_risk_per_trade, 500.0);
        // untouched field keeps its default
        assert_eq!(cfg.risk.max_position_pct, 0.15);
    }

    #[test]
    fn validation_rejects_bad_confidence() {
        let mut cfg = Config::default();
        cfg.montecarlo.confidence = 1.5;
        assert!(cfg.validate().is_err());
    }
}
