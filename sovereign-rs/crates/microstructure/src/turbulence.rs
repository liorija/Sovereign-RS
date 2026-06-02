//! Navier-Stokes-inspired market microstructure: detect when **laminar**
//! (orderly, directional) flow breaks into **turbulence** (panic, chaotic),
//! so the engine can switch from directional trading to market-making.
//!
//! Two diagnostics:
//! * the **Hurst exponent** `H` (via rescaled-range R/S) — persistence of flow
//!   (`H>0.5` trending/laminar, `H≈0.5` diffusive, `H<0.5` mean-reverting), and
//! * a **microstructure Reynolds number** — volatility-of-volatility intensity;
//!   high values mark the onset of turbulence.

use serde::Serialize;

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    (xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Rescaled range `R/S` of one chunk (mean-adjusted cumulative range / std).
fn rescaled_range(chunk: &[f64]) -> Option<f64> {
    if chunk.len() < 4 {
        return None;
    }
    let m = mean(chunk);
    let mut cum = 0.0;
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;
    for &v in chunk {
        cum += v - m;
        max = max.max(cum);
        min = min.min(cum);
    }
    let r = max - min;
    let s = std(chunk);
    if s < 1e-12 || !r.is_finite() {
        None
    } else {
        Some(r / s)
    }
}

/// Estimate the Hurst exponent of a series via R/S analysis over dyadic window
/// sizes (log-log regression of mean R/S on window size). Returns `0.5`
/// (random walk) when there isn't enough data.
pub fn hurst_exponent(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 16 {
        return 0.5;
    }
    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();
    let mut size = n;
    while size >= 8 {
        let chunks = n / size;
        let mut rs = Vec::new();
        for c in 0..chunks {
            if let Some(v) = rescaled_range(&series[c * size..(c + 1) * size]) {
                rs.push(v);
            }
        }
        if !rs.is_empty() {
            let avg = mean(&rs);
            if avg > 0.0 {
                log_n.push((size as f64).ln());
                log_rs.push(avg.ln());
            }
        }
        size /= 2;
    }
    if log_n.len() < 2 {
        return 0.5;
    }
    // slope of log(R/S) vs log(n) = H
    let mx = mean(&log_n);
    let my = mean(&log_rs);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..log_n.len() {
        num += (log_n[i] - mx) * (log_rs[i] - my);
        den += (log_n[i] - mx).powi(2);
    }
    if den.abs() < 1e-12 {
        0.5
    } else {
        (num / den).clamp(0.0, 1.0)
    }
}

/// Microstructure "Reynolds number": the coefficient of variation of rolling
/// volatility, scaled by √(#windows). High ⇒ volatility clustering / turbulence.
pub fn microstructure_reynolds(returns: &[f64], sub_window: usize) -> f64 {
    let sub = sub_window.max(2);
    if returns.len() < sub * 2 {
        return 0.0;
    }
    let vols: Vec<f64> = returns.windows(sub).map(std).collect();
    let m = mean(&vols);
    if m < 1e-12 {
        return 0.0;
    }
    (std(&vols) / m) * (vols.len() as f64).sqrt()
}

/// Flow regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FlowRegime {
    /// Orderly directional flow — trade with the trend.
    Laminar,
    /// Mixed.
    Transitional,
    /// Chaotic / panic — switch to market-making, widen risk controls.
    Turbulent,
}

/// Combined turbulence diagnostic.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Turbulence {
    pub hurst: f64,
    pub reynolds: f64,
    pub regime: FlowRegime,
}

/// Reynolds threshold above which flow is considered turbulent.
pub const RE_TURBULENT: f64 = 4.0;
/// Reynolds threshold below which (with persistent Hurst) flow is laminar.
pub const RE_LAMINAR: f64 = 2.0;

/// Classify a return series into a [`FlowRegime`].
pub fn classify(returns: &[f64]) -> Turbulence {
    let hurst = hurst_exponent(returns);
    let sub = (returns.len() / 8).max(5);
    let reynolds = microstructure_reynolds(returns, sub);
    let regime = if reynolds >= RE_TURBULENT {
        FlowRegime::Turbulent
    } else if hurst > 0.55 && reynolds < RE_LAMINAR {
        FlowRegime::Laminar
    } else {
        FlowRegime::Transitional
    };
    Turbulence {
        hurst,
        reynolds,
        regime,
    }
}

impl FlowRegime {
    /// Whether the engine should prefer market-making over directional bets.
    pub fn prefer_market_making(&self) -> bool {
        matches!(self, FlowRegime::Turbulent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn trending_more_persistent_than_mean_reverting() {
        // Persistent: cumulative random-ish walk with drift.
        let trend: Vec<f64> = (0..512)
            .map(|i| 0.1 + (i as f64 * 0.05).sin() * 0.3)
            .collect();
        // Anti-persistent: alternating signs.
        let revert: Vec<f64> = (0..512)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        assert!(hurst_exponent(&trend) > hurst_exponent(&revert));
    }

    #[test]
    fn volatility_clustering_raises_reynolds() {
        // Smooth constant-vol returns.
        let smooth: Vec<f64> = (0..400).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        // Clustered: calm then violent.
        let mut clustered = vec![0.0001; 200];
        clustered.extend((0..200).map(|i| if i % 2 == 0 { 0.05 } else { -0.05 }));
        let re_smooth = microstructure_reynolds(&smooth, 20);
        let re_cluster = microstructure_reynolds(&clustered, 20);
        assert!(
            re_cluster > re_smooth,
            "cluster {re_cluster} smooth {re_smooth}"
        );
    }

    #[test]
    fn classify_returns_a_regime() {
        let r: Vec<f64> = (0..300).map(|i| (i as f64 * 0.3).sin() * 0.01).collect();
        let t = classify(&r);
        assert!(t.hurst.is_finite() && t.reynolds.is_finite());
    }

    proptest! {
        #[test]
        fn diagnostics_finite_and_bounded(
            r in proptest::collection::vec(-0.2f64..0.2, 0..400)
        ) {
            let t = classify(&r);
            prop_assert!((0.0..=1.0).contains(&t.hurst));
            prop_assert!(t.reynolds.is_finite() && t.reynolds >= 0.0);
        }
    }
}
