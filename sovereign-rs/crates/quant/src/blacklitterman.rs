//! Black-Litterman portfolio optimizer (port of
//! `v325_upgrades.BlackLittermanOptimizer`).
//!
//! Blends the CAPM market-equilibrium prior with the BFT agents' views to
//! produce robust weights that don't deviate wildly from market-cap weights —
//! more stable than raw Kelly. Linear algebra via `nalgebra`; any numerical
//! failure falls back to equal weight.

use nalgebra::{DMatrix, DVector};
use ndarray::Array2;

/// Risk-aversion coefficient (market average).
const DELTA: f64 = 2.5;
/// Prior uncertainty scalar.
const TAU: f64 = 0.05;

/// Compute Black-Litterman posterior weights.
///
/// * `returns` — `T × N` matrix of per-period asset returns.
/// * `views`   — length-`N` vector of expected excess returns (alpha) per asset.
/// * `view_conf` — confidence in the views, `[0,1]`.
/// * `max_weight` — per-asset cap.
///
/// Returns `N` non-negative weights summing to 1.
pub fn optimize(returns: &Array2<f64>, views: &[f64], view_conf: f64, max_weight: f64) -> Vec<f64> {
    let n = returns.ncols();
    let t = returns.nrows();
    let equal = || vec![1.0 / n.max(1) as f64; n];
    if n < 2 || t < 5 || views.len() != n {
        return equal();
    }

    // Sample covariance (N×N).
    let means: Vec<f64> = (0..n).map(|j| returns.column(j).sum() / t as f64).collect();
    let mut cov = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..t {
                s += (returns[[k, i]] - means[i]) * (returns[[k, j]] - means[j]);
            }
            let c = s / (t as f64 - 1.0);
            cov[(i, j)] = c;
            cov[(j, i)] = c;
        }
    }

    let w_mkt = DVector::from_element(n, 1.0 / n as f64);
    let pi = cov.scale(DELTA) * &w_mkt; // equilibrium excess returns
    let q = DVector::from_iterator(
        n,
        views.iter().map(|v| if v.is_finite() { *v } else { 0.0 }),
    );

    // Ω diagonal (uncertainty of each absolute view) and its inverse.
    let conf = view_conf.clamp(0.0, 1.0);
    let mut omega_inv = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let oii = ((1.0 - conf) * TAU * cov[(i, i)]).max(1e-8);
        omega_inv[(i, i)] = 1.0 / oii;
    }

    let tau_cov = cov.scale(TAU);
    let tau_cov_inv = match tau_cov.try_inverse() {
        Some(m) => m,
        None => return equal(),
    };

    // Posterior mean: μ = (τΣ)⁻¹ + Ω⁻¹)⁻¹ ((τΣ)⁻¹ π + Ω⁻¹ Q)   (P = I).
    let a = &tau_cov_inv + &omega_inv;
    let a_inv = match a.try_inverse() {
        Some(m) => m,
        None => return equal(),
    };
    let rhs = &tau_cov_inv * &pi + &omega_inv * &q;
    let mu_bl = a_inv * rhs;

    // Optimal weights: w = (δΣ)⁻¹ μ.
    let cov_inv = match cov.scale(DELTA).try_inverse() {
        Some(m) => m,
        None => return equal(),
    };
    let w_raw = cov_inv * mu_bl;

    // Long-only, normalized, then capped via iterative water-filling so the cap
    // is respected exactly (a single cap+renormalize would not be).
    let w: Vec<f64> = w_raw
        .iter()
        .map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
        .collect();
    let total: f64 = w.iter().sum();
    if total < 1e-9 {
        return equal();
    }
    cap_simplex(w, max_weight)
}

/// Normalize `w` to sum 1, then enforce a per-element `cap` by repeatedly
/// clipping over-cap elements and redistributing their excess proportionally
/// to the others. If `cap·n < 1` the cap is infeasible and the (uncapped)
/// normalized weights are returned.
fn cap_simplex(mut w: Vec<f64>, cap: f64) -> Vec<f64> {
    let n = w.len();
    if n == 0 {
        return w;
    }
    let total: f64 = w.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / n as f64; n];
    }
    for x in &mut w {
        *x /= total;
    }
    if cap * n as f64 <= 1.0 + 1e-12 {
        return w; // cap infeasible — return normalized weights as-is
    }
    for _ in 0..=n {
        let mut excess = 0.0;
        let mut uncapped = 0.0;
        for &x in &w {
            if x > cap {
                excess += x - cap;
            } else {
                uncapped += x;
            }
        }
        if excess <= 1e-12 || uncapped <= 1e-12 {
            break;
        }
        for x in &mut w {
            if *x > cap {
                *x = cap;
            } else {
                *x += excess * (*x / uncapped);
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn weights_form_a_simplex() {
        // 3 assets, 60 periods of mild noise.
        let mut data = Array2::<f64>::zeros((60, 3));
        for k in 0..60 {
            data[[k, 0]] = ((k as f64 * 0.3).sin()) * 0.01;
            data[[k, 1]] = ((k as f64 * 0.2).cos()) * 0.012;
            data[[k, 2]] = ((k as f64 * 0.5).sin()) * 0.008;
        }
        let views = [0.02, 0.01, -0.005];
        let w = optimize(&data, &views, 0.5, 0.6);
        assert_eq!(w.len(), 3);
        let s: f64 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum={s}");
        assert!(w.iter().all(|x| *x >= 0.0 && *x <= 0.6 + 1e-9));
    }

    #[test]
    fn degenerate_input_is_equal_weight() {
        let data = array![[0.0, 0.0]];
        let w = optimize(&data, &[0.0, 0.0], 0.5, 1.0);
        assert_eq!(w, vec![0.5, 0.5]);
    }

    proptest! {
        #[test]
        fn never_panics(seed in 0u64..2000) {
            let n = 3usize;
            let t = 40usize;
            let mut data = Array2::<f64>::zeros((t, n));
            for k in 0..t {
                for j in 0..n {
                    let v = (((k as u64 * 31 + j as u64 * 17 + seed) % 200) as f64 - 100.0) / 1000.0;
                    data[[k, j]] = v;
                }
            }
            let w = optimize(&data, &[0.01, 0.0, -0.01], 0.5, 0.5);
            prop_assert_eq!(w.len(), n);
            prop_assert!(w.iter().all(|x| x.is_finite()));
        }
    }
}
