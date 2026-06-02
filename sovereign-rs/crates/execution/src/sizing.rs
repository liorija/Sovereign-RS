//! Inverse-volatility (risk-parity) position sizing (port of
//! `v322_inverse_vol_sizer.InverseVolSizer`).
//!
//! Keeps the per-trade dollar risk constant: a 2× more volatile name gets ~½ the
//! shares, so every position has the same expected loss on an adverse 1-vol move.

use serde::Serialize;

use sovereign_quant::garch::Garch11;

/// Fallback daily-vol assumption (2% of price) when no estimate is available.
const FALLBACK_VOL_PCT: f64 = 0.02;

/// Result of a sizing decision.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SizeResult {
    pub shares: u64,
    pub vol_used: f64,
    pub method: &'static str,
    pub conviction_delta: i32,
    pub capped: bool,
}

/// The sizer.
#[derive(Debug, Clone, Copy)]
pub struct InverseVolSizer {
    pub dollar_risk_per_trade: f64,
    pub max_position_pct: f64,
    pub use_garch: bool,
}

impl Default for InverseVolSizer {
    fn default() -> Self {
        Self {
            dollar_risk_per_trade: 250.0,
            max_position_pct: 0.15,
            use_garch: true,
        }
    }
}

impl InverseVolSizer {
    /// Size a position. `atr` is the dollar ATR if known; `returns` feeds the
    /// GARCH estimate. Either may be empty/`None` — the sizer degrades safely.
    pub fn size(&self, price: f64, capital: f64, returns: &[f64], atr: Option<f64>) -> SizeResult {
        if !(price.is_finite() && price > 0.0 && capital.is_finite() && capital > 0.0) {
            return SizeResult {
                shares: 1,
                vol_used: 0.0,
                method: "INVALID",
                conviction_delta: -4,
                capped: false,
            };
        }

        let garch_dollar = if self.use_garch && returns.len() >= 20 {
            let v = Garch11::estimate(returns) * price; // fraction → dollars
            if v.is_finite() && v > 0.0 {
                Some(v)
            } else {
                None
            }
        } else {
            None
        };

        let (vol_used, method) = match (atr.filter(|a| a.is_finite() && *a > 0.0), garch_dollar) {
            (Some(a), Some(g)) => (0.6 * a + 0.4 * g, "ATR+GARCH"),
            (Some(a), None) => (a, "ATR"),
            (None, Some(g)) => (g, "GARCH"),
            (None, None) => (FALLBACK_VOL_PCT * price, "FALLBACK"),
        };
        let vol_used = vol_used.max(1e-6);

        let raw = self.dollar_risk_per_trade / vol_used;
        let max_by_capital = (capital * self.max_position_pct) / price;
        let capped = raw > max_by_capital;
        let shares = raw.min(max_by_capital).floor().max(1.0) as u64;

        let conviction_delta = if method == "FALLBACK" {
            -4
        } else if capped {
            -2
        } else {
            4
        };

        SizeResult {
            shares,
            vol_used,
            method,
            conviction_delta,
            capped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn higher_vol_gets_fewer_shares() {
        let s = InverseVolSizer::default();
        let low = s.size(100.0, 1_000_000.0, &[], Some(1.0)); // $1 ATR
        let high = s.size(100.0, 1_000_000.0, &[], Some(5.0)); // $5 ATR
        assert!(low.shares > high.shares);
    }

    #[test]
    fn capital_cap_enforced() {
        let s = InverseVolSizer {
            dollar_risk_per_trade: 1e9,
            ..Default::default()
        };
        let r = s.size(100.0, 10_000.0, &[], Some(0.01));
        // max 15% of $10k / $100 = 15 shares
        assert!(r.shares <= 15);
        assert!(r.capped);
    }

    proptest! {
        #[test]
        fn shares_at_least_one_and_finite_vol(
            price in 0.01f64..10_000.0,
            capital in 1.0f64..1e7,
            atr in 0.0f64..1000.0,
        ) {
            let s = InverseVolSizer::default();
            let r = s.size(price, capital, &[], Some(atr));
            prop_assert!(r.shares >= 1);
            prop_assert!(r.vol_used.is_finite() && r.vol_used > 0.0);
        }
    }
}
