//! Technical indicators, vectorized over `&[f64]` slices.
//!
//! Faithful to `build_features_polars.py`: the **null-propagation contract** is
//! preserved by neutral-filling the warmup region of each *final* feature
//! exactly as the Polars `.fill_null(...)` calls do, so a Rust feature row is
//! numerically identical to the Python one.

/// Simple rolling mean. Positions `< window-1` are `f64::NAN` (warmup).
pub fn rolling_mean(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || window > n {
        return out;
    }
    let mut sum = 0.0;
    for i in 0..n {
        sum += x[i];
        if i >= window {
            sum -= x[i - window];
        }
        if i + 1 >= window {
            out[i] = sum / window as f64;
        }
    }
    out
}

/// Rolling (population) standard deviation; warmup is `NAN`.
pub fn rolling_std(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || window > n {
        return out;
    }
    for i in (window - 1)..n {
        let slice = &x[i + 1 - window..=i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / window as f64;
        out[i] = var.sqrt();
    }
    out
}

/// Exponential moving average (`adjust=false`), matching pandas/polars.
pub fn ema(x: &[f64], span: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 || span == 0 {
        return out;
    }
    let k = 2.0 / (span as f64 + 1.0);
    let mut prev = x[0];
    out[0] = prev;
    for i in 1..n {
        prev = k * x[i] + (1.0 - k) * prev;
        out[i] = prev;
    }
    out
}

/// `pct_change(n)`: `x[t] / x[t-n] - 1`. Warmup is `0.0` (as Polars `.fill_null(0)`).
pub fn pct_change(x: &[f64], n: usize) -> Vec<f64> {
    let len = x.len();
    let mut out = vec![0.0; len];
    if n == 0 {
        return out;
    }
    for i in n..len {
        let prev = x[i - n];
        out[i] = if prev != 0.0 { x[i] / prev - 1.0 } else { 0.0 };
    }
    out
}

/// RSI(period); neutral warmup filled with `50.0` (Polars `.fill_null(50)`).
pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![50.0; n];
    if n < 2 || period == 0 {
        return out;
    }
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];
    for i in 1..n {
        let d = close[i] - close[i - 1];
        gains[i] = d.max(0.0);
        losses[i] = (-d).max(0.0);
    }
    let avg_gain = rolling_mean(&gains, period);
    let avg_loss = rolling_mean(&losses, period);
    for i in 0..n {
        if avg_gain[i].is_finite() && avg_loss[i].is_finite() {
            let rs = avg_gain[i] / (avg_loss[i] + 1e-9);
            out[i] = 100.0 - 100.0 / (1.0 + rs);
        }
    }
    out
}

/// MACD histogram = EMA12 − EMA26 − signal(EMA9). No warmup fill (caller normalizes).
pub fn macd_hist(close: &[f64]) -> Vec<f64> {
    let e12 = ema(close, 12);
    let e26 = ema(close, 26);
    let macd: Vec<f64> = e12.iter().zip(&e26).map(|(a, b)| a - b).collect();
    let sig = ema(&macd, 9);
    macd.iter().zip(&sig).map(|(m, s)| m - s).collect()
}

/// True Range, then rolling-mean → ATR(period). Warmup is `NAN`.
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0; n];
    for i in 0..n {
        if i == 0 {
            tr[i] = high[i] - low[i];
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }
    }
    rolling_mean(&tr, period)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rsi_in_unit_scale_range() {
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.3).sin()).collect();
        let r = rsi(&close, 14);
        assert!(r.iter().all(|v| (0.0..=100.0).contains(v)));
    }

    #[test]
    fn ema_tracks_constant_series() {
        let x = vec![5.0; 50];
        let e = ema(&x, 10);
        assert!(e.iter().all(|v| (*v - 5.0).abs() < 1e-9));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn rsi_always_bounded(close in proptest::collection::vec(1.0f64..1000.0, 2..300)) {
            let r = rsi(&close, 14);
            prop_assert!(r.iter().all(|v| v.is_finite() && (0.0..=100.0).contains(v)));
        }

        #[test]
        fn pct_change_warmup_is_zero(close in proptest::collection::vec(1.0f64..1000.0, 1..200), n in 1usize..20) {
            let pc = pct_change(&close, n);
            for v in pc.iter().take(n) {
                prop_assert_eq!(*v, 0.0);
            }
            prop_assert!(pc.iter().all(|v| v.is_finite()));
        }
    }
}
