//! Cointegration tooling: OLS, augmented-Dickey-Fuller t-statistic, Engle-Granger
//! two-step test, and Ornstein-Uhlenbeck half-life (port of the StatArb pipeline
//! in `v322_stat_arb_engine`).
//!
//! The ADF here uses the no-lag specification with a constant; its t-statistic
//! is compared against MacKinnon's asymptotic 5% critical value (≈ −2.86).

/// 5% ADF critical value (constant, no trend, large sample).
pub const ADF_CRIT_5PCT: f64 = -2.86;

/// Ordinary least squares for `y = intercept + slope·x`. Returns `(slope, intercept)`.
pub fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len());
    if n < 2 {
        return (0.0, 0.0);
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..n {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
    }
    if sxx.abs() < 1e-300 {
        return (0.0, my);
    }
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// Augmented Dickey-Fuller t-statistic (no lag, with constant).
/// More negative ⇒ stronger evidence of stationarity (mean reversion).
pub fn adf_tstat(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 10 {
        return 0.0;
    }
    // Regress Δy_t on y_{t-1} (with intercept).
    let y_lag = &series[..n - 1];
    let dy: Vec<f64> = (1..n).map(|i| series[i] - series[i - 1]).collect();
    let m = dy.len();
    let (slope, intercept) = ols(y_lag, &dy);

    // Residual variance and standard error of the slope.
    let mx = y_lag.iter().sum::<f64>() / m as f64;
    let sxx: f64 = y_lag.iter().map(|v| (v - mx) * (v - mx)).sum();
    if sxx.abs() < 1e-300 {
        return 0.0;
    }
    let ssr: f64 = (0..m)
        .map(|i| {
            let pred = intercept + slope * y_lag[i];
            (dy[i] - pred).powi(2)
        })
        .sum();
    let dof = (m as f64 - 2.0).max(1.0);
    let s2 = ssr / dof;
    let se = (s2 / sxx).sqrt();
    if se < 1e-300 {
        return 0.0;
    }
    slope / se
}

/// Whether a series is (weakly) stationary at the 5% level.
pub fn is_stationary(series: &[f64]) -> bool {
    adf_tstat(series) < ADF_CRIT_5PCT
}

/// Ornstein-Uhlenbeck half-life of mean reversion from an AR(1) fit on the
/// spread's first differences. Returns `f64::INFINITY` if not mean-reverting.
pub fn half_life(spread: &[f64]) -> f64 {
    let n = spread.len();
    if n < 20 {
        return f64::INFINITY;
    }
    let lag = &spread[..n - 1];
    let dspread: Vec<f64> = (1..n).map(|i| spread[i] - spread[i - 1]).collect();
    let (slope, _) = ols(lag, &dspread);
    if slope >= 0.0 || !slope.is_finite() {
        return f64::INFINITY;
    }
    -std::f64::consts::LN_2 / slope
}

/// Engle-Granger two-step result for a pair.
#[derive(Debug, Clone, Copy)]
pub struct EngleGranger {
    /// Hedge ratio β from regressing `a` on `b`.
    pub beta: f64,
    pub intercept: f64,
    /// ADF t-statistic on the residual spread.
    pub adf_tstat: f64,
    /// OU half-life of the residual spread (days/bars).
    pub half_life: f64,
    /// True if residual is stationary at 5% and half-life is sane (1.5–30).
    pub cointegrated: bool,
}

/// Run the Engle-Granger two-step cointegration test for `a` vs `b`.
pub fn engle_granger(a: &[f64], b: &[f64]) -> EngleGranger {
    let (beta, intercept) = ols(b, a); // a = intercept + beta·b
    let n = a.len().min(b.len());
    let resid: Vec<f64> = (0..n).map(|i| a[i] - (intercept + beta * b[i])).collect();
    let t = adf_tstat(&resid);
    let hl = half_life(&resid);
    let cointegrated = t < ADF_CRIT_5PCT && (1.5..=30.0).contains(&hl);
    EngleGranger {
        beta,
        intercept,
        adf_tstat: t,
        half_life: hl,
        cointegrated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic mean-reverting AR(1): x_t = 0.5·x_{t-1} + drive.
    fn mean_reverting(n: usize) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in 1..n {
            let drive = ((i as f64) * 0.7).sin();
            x[i] = 0.5 * x[i - 1] + drive;
        }
        x
    }

    /// Random-walk-like trending series (unit root).
    fn trending(n: usize) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in 1..n {
            x[i] = x[i - 1] + 1.0 + ((i as f64) * 0.3).sin() * 0.01;
        }
        x
    }

    #[test]
    fn stationary_series_flagged() {
        assert!(is_stationary(&mean_reverting(300)));
    }

    #[test]
    fn trending_series_not_stationary() {
        assert!(!is_stationary(&trending(300)));
    }

    #[test]
    fn half_life_positive_for_mean_reverting() {
        let hl = half_life(&mean_reverting(300));
        assert!(hl.is_finite() && hl > 0.0, "hl={hl}");
    }

    #[test]
    fn ols_recovers_line() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.0 + 2.0 * v).collect();
        let (slope, intercept) = ols(&x, &y);
        assert!((slope - 2.0).abs() < 1e-9 && (intercept - 3.0).abs() < 1e-9);
    }
}
