//! Assembles normalized ML features from an OHLCV window.
//!
//! Implements the first 9 features of the `build_features_polars` contract
//! (V318 block) with identical clamping/scaling, so each row matches the Python
//! output. The remaining 15 (V319/V320 momentum, OBV slope, daily-resample RSI,
//! etc.) follow the same pattern and are tracked in `docs/ROADMAP.md`.

use ndarray::Array2;

use crate::indicators::{atr, macd_hist, pct_change, rolling_mean, rolling_std, rsi};

/// Canonical feature column order.
pub const FEATURE_NAMES: [&str; 9] = [
    "close_ret",
    "vol_ratio",
    "rsi_norm",
    "macd_norm",
    "adx_norm",
    "bb_pct",
    "atr_ratio",
    "mom3",
    "mom10",
];

/// Minimum bars required to compute a meaningful frame (matches Polars guard).
pub const MIN_BARS: usize = 30;

/// A computed feature frame: one row per input bar, columns = [`FEATURE_NAMES`].
#[derive(Debug, Clone)]
pub struct FeatureFrame {
    data: Array2<f64>,
}

#[inline]
fn clamp_scale(v: f64, lo: f64, hi: f64, scale: f64, fill: f64) -> f64 {
    let v = if v.is_finite() { v } else { fill };
    v.clamp(lo, hi) / scale
}

impl FeatureFrame {
    /// Compute features from aligned OHLCV slices. Returns `None` if there are
    /// fewer than [`MIN_BARS`] bars or the slice lengths disagree.
    pub fn compute(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Option<Self> {
        let n = close.len();
        if n < MIN_BARS || high.len() != n || low.len() != n || volume.len() != n {
            return None;
        }

        let close_ret = pct_change(close, 1);
        let vol_ma20 = rolling_mean(volume, 20);
        let rsi14 = rsi(close, 14);
        let mh = macd_hist(close);
        let std20 = rolling_std(close, 20);
        let sma20 = rolling_mean(close, 20);
        let atr14 = atr(high, low, close, 14);
        let hl5 = rolling_mean(&row_range(high, low), 5);
        let mom3 = pct_change(close, 3);
        let mom10 = pct_change(close, 10);

        let mut data = Array2::<f64>::zeros((n, FEATURE_NAMES.len()));
        for i in 0..n {
            // close_ret (already 0-filled at warmup)
            data[[i, 0]] = if close_ret[i].is_finite() {
                close_ret[i]
            } else {
                0.0
            };

            // vol_ratio: volume / vol_ma20, clamp 0..5 / 5, fill 1
            let vr = if vol_ma20[i].is_finite() && vol_ma20[i] > 0.0 {
                volume[i] / vol_ma20[i]
            } else {
                1.0
            };
            data[[i, 1]] = clamp_scale(vr, 0.0, 5.0, 5.0, 1.0);

            // rsi_norm: rsi / 100
            data[[i, 2]] = rsi14[i] / 100.0;

            // macd_norm: (hist / (std20+1e-9)) clamp ±3 / 3, fill 0
            let denom = if std20[i].is_finite() { std20[i] } else { 0.0 } + 1e-9;
            data[[i, 3]] = clamp_scale(mh[i] / denom, -3.0, 3.0, 3.0, 0.0);

            // adx_norm proxy: atr14/close clamp 0..0.1 / 0.1, fill 0
            let adx = if atr14[i].is_finite() {
                atr14[i] / (close[i] + 1e-9)
            } else {
                0.0
            };
            data[[i, 4]] = clamp_scale(adx, 0.0, 0.1, 0.1, 0.0);

            // bb_pct: position within Bollinger band, clamp 0..1, fill 0.5
            let bb = if std20[i].is_finite() && sma20[i].is_finite() {
                let up = sma20[i] + 2.0 * std20[i];
                let lo = sma20[i] - 2.0 * std20[i];
                (close[i] - lo) / (up - lo + 1e-9)
            } else {
                0.5
            };
            data[[i, 5]] = clamp_scale(bb, 0.0, 1.0, 1.0, 0.5);

            // atr_ratio: hl5/close clamp 0..0.1 / 0.1, fill 0.01
            let ar = if hl5[i].is_finite() {
                hl5[i] / (close[i] + 1e-9)
            } else {
                0.01
            };
            data[[i, 6]] = clamp_scale(ar, 0.0, 0.1, 0.1, 0.01);

            // mom3 / mom10
            data[[i, 7]] = clamp_scale(mom3[i], -0.1, 0.1, 0.1, 0.0);
            data[[i, 8]] = clamp_scale(mom10[i], -0.2, 0.2, 0.2, 0.0);
        }

        Some(Self { data })
    }

    /// Borrow the (rows × features) matrix.
    pub fn matrix(&self) -> &Array2<f64> {
        &self.data
    }

    /// The most recent feature row.
    pub fn latest(&self) -> Vec<f64> {
        self.data.row(self.data.nrows() - 1).to_vec()
    }
}

fn row_range(high: &[f64], low: &[f64]) -> Vec<f64> {
    high.iter().zip(low).map(|(h, l)| h - l).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn synth(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let vol: Vec<f64> = (0..n).map(|i| 1.0e6 + (i % 7) as f64 * 1.0e5).collect();
        (high, low, close, vol)
    }

    #[test]
    fn rejects_short_input() {
        let (h, l, c, v) = synth(10);
        assert!(FeatureFrame::compute(&h, &l, &c, &v).is_none());
    }

    #[test]
    fn all_features_finite_and_normalized() {
        let (h, l, c, v) = synth(120);
        let f = FeatureFrame::compute(&h, &l, &c, &v).unwrap();
        let m = f.matrix();
        assert_eq!(m.ncols(), FEATURE_NAMES.len());
        for v in m.iter() {
            assert!(v.is_finite());
        }
        // bounded columns
        for &i in &[1usize, 2, 5, 6] {
            for r in 0..m.nrows() {
                assert!((-0.001..=1.001).contains(&m[[r, i]]));
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(120))]

        #[test]
        fn features_never_nan(seed in 0u64..10_000) {
            // deterministic pseudo-random-ish series from seed
            let n = 80;
            let close: Vec<f64> = (0..n)
                .map(|i| 50.0 + (((i as u64).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f64 / 10.0)
                .collect();
            let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
            let low: Vec<f64> = close.iter().map(|c| (c - 0.5).max(0.01)).collect();
            let vol: Vec<f64> = (0..n).map(|i| 1.0e5 * (1 + i % 5) as f64).collect();
            if let Some(f) = FeatureFrame::compute(&high, &low, &close, &vol) {
                prop_assert!(f.matrix().iter().all(|v| v.is_finite()));
            }
        }
    }
}
