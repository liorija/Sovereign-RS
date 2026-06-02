//! GARCH(1,1) volatility estimator via moment-matching (port of
//! `v322_inverse_vol_sizer.GARCHEstimator`).
//!
//! Model: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}` with `α + β < 1` (stationarity).
//! Moment-matching avoids an optimizer, so it's fast enough for the live loop.

/// A GARCH(1,1) conditional-volatility estimator.
#[derive(Debug, Clone, Copy)]
pub struct Garch11 {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Default for Garch11 {
    fn default() -> Self {
        // Typical equity parameters (Engle 2001).
        Self {
            omega: 1e-6,
            alpha: 0.09,
            beta: 0.90,
        }
    }
}

impl Garch11 {
    const OMEGA_FLOOR: f64 = 1e-12;

    /// Construct, enforcing covariance-stationarity (`α + β < 1`).
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Self {
        let mut alpha = alpha.max(0.0);
        let mut beta = beta.max(0.0);
        let total = alpha + beta;
        if total >= 1.0 {
            let scale = 0.98 / total;
            alpha *= scale;
            beta *= scale;
        }
        Self {
            omega: omega.max(Self::OMEGA_FLOOR),
            alpha,
            beta,
        }
    }

    /// Calibrate `ω` from the long-run variance: `ω = var·(1 − α − β)`.
    pub fn fit(&mut self, returns: &[f64]) -> &mut Self {
        let clean: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
        if clean.len() < 20 {
            return self;
        }
        let mean = clean.iter().sum::<f64>() / clean.len() as f64;
        let var = clean.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / clean.len() as f64;
        self.omega = (var * (1.0 - self.alpha - self.beta)).max(Self::OMEGA_FLOOR);
        self
    }

    /// 1-step-ahead conditional volatility (standard deviation).
    pub fn forecast(&self, returns: &[f64]) -> f64 {
        let clean: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
        if clean.is_empty() {
            return self.omega.sqrt();
        }
        let mean = clean.iter().sum::<f64>() / clean.len() as f64;
        let mut sigma2 =
            clean.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / clean.len() as f64;
        sigma2 = sigma2.max(Self::OMEGA_FLOOR);
        for &r in &clean {
            sigma2 = self.omega + self.alpha * r * r + self.beta * sigma2;
            sigma2 = sigma2.max(Self::OMEGA_FLOOR);
        }
        sigma2.sqrt()
    }

    /// One-shot: fit + forecast on the same series.
    pub fn estimate(returns: &[f64]) -> f64 {
        let mut g = Garch11::default();
        g.fit(returns);
        g.forecast(returns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn stationarity_enforced() {
        let g = Garch11::new(1e-6, 0.5, 0.7); // sums to 1.2 → rescaled
        assert!(g.alpha + g.beta < 1.0);
    }

    #[test]
    fn higher_vol_series_forecasts_higher() {
        let calm: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin() * 0.001).collect();
        let wild: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin() * 0.05).collect();
        assert!(Garch11::estimate(&wild) > Garch11::estimate(&calm));
    }

    proptest! {
        #[test]
        fn forecast_finite_and_nonneg(
            rets in proptest::collection::vec(-0.5f64..0.5, 0..300)
        ) {
            let v = Garch11::estimate(&rets);
            prop_assert!(v.is_finite() && v >= 0.0);
        }
    }
}
