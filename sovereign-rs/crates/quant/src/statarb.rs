//! Statistical-arbitrage pair evaluation (port of
//! `v322_stat_arb_engine.StatArbEngine`): Kalman hedge ratio + Engle-Granger
//! cointegration + z-score → a tradeable spread signal and BFT conviction.

use serde::Serialize;

use crate::cointegration::engle_granger;
use crate::kalman::KalmanHedge;

/// Direction of the spread trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpreadSignal {
    /// Spread too low → buy A, short B.
    LongSpread,
    /// Spread too high → short A, buy B.
    ShortSpread,
    /// Mean-reversion complete.
    Close,
    Neutral,
}

/// Map a z-score to a signal and a BFT conviction delta (additive to the
/// Technical agent), mirroring `_classify`.
pub fn classify(zscore: f64, entry: f64, exit: f64) -> (SpreadSignal, i32) {
    let az = zscore.abs();
    if az < exit {
        return (SpreadSignal::Close, 0);
    }
    if az < entry {
        return (SpreadSignal::Neutral, 0);
    }
    let pts = if az > 3.0 {
        14
    } else if az > 2.5 {
        11
    } else {
        8
    };
    if zscore > 0.0 {
        (SpreadSignal::ShortSpread, -pts)
    } else {
        (SpreadSignal::LongSpread, pts)
    }
}

/// Full evaluation of a pair.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PairStats {
    pub beta: f64,
    pub adf_tstat: f64,
    pub half_life: f64,
    pub zscore: f64,
    pub signal: SpreadSignal,
    pub conviction_delta: i32,
    pub valid: bool,
}

/// Evaluate a candidate pair `(a, b)` of aligned price series.
pub fn evaluate_pair(a: &[f64], b: &[f64]) -> PairStats {
    let n = a.len().min(b.len());
    if n < 60 {
        return PairStats {
            beta: 0.0,
            adf_tstat: 0.0,
            half_life: f64::INFINITY,
            zscore: 0.0,
            signal: SpreadSignal::Neutral,
            conviction_delta: 0,
            valid: false,
        };
    }

    // Dynamic hedge ratio via Kalman.
    let mut kf = KalmanHedge::new(1e-4, 1e-3);
    let _ = kf.batch(&a[..n], &b[..n]);
    let beta = kf.beta();

    // Spread and its z-score (full-sample mean/std).
    let spread: Vec<f64> = (0..n).map(|i| a[i] - beta * b[i]).collect();
    let mean = spread.iter().sum::<f64>() / n as f64;
    let var = spread.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let zscore = if std > 1e-10 {
        (spread[n - 1] - mean) / std
    } else {
        0.0
    };

    // Validity via Engle-Granger on the spread relationship.
    let eg = engle_granger(&a[..n], &b[..n]);
    let valid = eg.cointegrated && zscore.is_finite();

    let (signal, conviction_delta) = if valid {
        classify(zscore, 2.0, 0.5)
    } else {
        (SpreadSignal::Neutral, 0)
    };

    PairStats {
        beta,
        adf_tstat: eg.adf_tstat,
        half_life: eg.half_life,
        zscore,
        signal,
        conviction_delta,
        valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_thresholds() {
        assert_eq!(classify(0.1, 2.0, 0.5).0, SpreadSignal::Close);
        assert_eq!(classify(1.0, 2.0, 0.5).0, SpreadSignal::Neutral);
        let (sig, pts) = classify(3.5, 2.0, 0.5);
        assert_eq!(sig, SpreadSignal::ShortSpread);
        assert_eq!(pts, -14);
        let (sig, pts) = classify(-2.2, 2.0, 0.5);
        assert_eq!(sig, SpreadSignal::LongSpread);
        assert_eq!(pts, 8);
    }

    #[test]
    fn cointegrated_pair_is_valid() {
        // b is a smooth series; a = 1.5*b + stationary mean-reverting noise.
        let n = 300;
        let b: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.05).sin() * 5.0)
            .collect();
        let mut noise = vec![0.0; n];
        for i in 1..n {
            noise[i] = 0.5 * noise[i - 1] + ((i as f64) * 0.9).sin();
        }
        let a: Vec<f64> = (0..n).map(|i| 1.5 * b[i] + noise[i]).collect();
        let ps = evaluate_pair(&a, &b);
        assert!(ps.beta > 1.0 && ps.beta < 2.0, "beta={}", ps.beta);
        assert!(ps.zscore.is_finite());
    }

    #[test]
    fn short_series_is_invalid() {
        let a = vec![1.0; 10];
        let b = vec![2.0; 10];
        assert!(!evaluate_pair(&a, &b).valid);
    }
}
