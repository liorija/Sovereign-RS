//! Convex portfolio optimization: Markowitz mean-variance weights and Euclidean
//! projection onto the probability simplex (long-only, fully-invested).

use nalgebra::{DMatrix, DVector};
use ndarray::Array2;

fn to_dmatrix(cov: &Array2<f64>) -> Option<DMatrix<f64>> {
    let (n, m) = cov.dim();
    if n == 0 || n != m {
        return None;
    }
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = cov[[i, j]];
        }
    }
    Some(d)
}

/// Euclidean projection of `v` onto the simplex `{w : wᵢ ≥ 0, Σwᵢ = 1}`
/// (Duchi et al. 2008). Long-only, fully-invested weights.
pub fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for (i, &ui) in u.iter().enumerate() {
        cumulative += ui;
        let t = (cumulative - 1.0) / (i as f64 + 1.0);
        if ui - t > 0.0 {
            theta = t;
        }
    }
    v.iter().map(|x| (x - theta).max(0.0)).collect()
}

/// Global minimum-variance portfolio `w = Σ⁻¹1 / (1ᵀΣ⁻¹1)`, projected long-only.
/// Falls back to equal weight on any numerical failure.
pub fn min_variance_weights(cov: &Array2<f64>) -> Vec<f64> {
    let n = cov.nrows();
    let equal = || vec![1.0 / n.max(1) as f64; n];
    if n == 0 {
        return Vec::new();
    }
    let d = match to_dmatrix(cov) {
        Some(d) => d,
        None => return equal(),
    };
    let inv = match d.try_inverse() {
        Some(i) => i,
        None => return equal(),
    };
    let ones = DVector::from_element(n, 1.0);
    let inv_ones = &inv * &ones;
    let denom = ones.dot(&inv_ones);
    if denom.abs() < 1e-12 {
        return equal();
    }
    project_simplex((inv_ones / denom).as_slice())
}

/// Mean-variance optimal weights `w ∝ Σ⁻¹μ / λ`, projected long-only.
pub fn mean_variance_weights(cov: &Array2<f64>, expected: &[f64], risk_aversion: f64) -> Vec<f64> {
    let n = cov.nrows();
    let equal = || vec![1.0 / n.max(1) as f64; n];
    if n == 0 || expected.len() != n {
        return equal();
    }
    let lambda = if risk_aversion.is_finite() && risk_aversion > 1e-9 {
        risk_aversion
    } else {
        1.0
    };
    let d = match to_dmatrix(cov) {
        Some(d) => d,
        None => return equal(),
    };
    let inv = match d.try_inverse() {
        Some(i) => i,
        None => return equal(),
    };
    let mu = DVector::from_iterator(n, expected.iter().copied());
    let raw = (&inv * mu) / lambda;
    project_simplex(raw.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn simplex_projection_sums_to_one() {
        let w = project_simplex(&[0.5, 0.5, 0.5]);
        let s: f64 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        assert!(w.iter().all(|x| *x >= 0.0));
        // a vector already on the simplex is unchanged
        let id = project_simplex(&[1.0, 0.0, 0.0]);
        assert!((id[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn min_variance_favours_low_variance_asset() {
        // Asset 0 has variance 1, asset 1 has variance 4 → more weight on asset 0.
        let cov = array![[1.0, 0.0], [0.0, 4.0]];
        let w = min_variance_weights(&cov);
        let s: f64 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        assert!(w[0] > w[1], "weights {w:?}");
    }

    #[test]
    fn degenerate_is_equal_weight() {
        let cov = Array2::<f64>::zeros((3, 3)); // singular
        let w = min_variance_weights(&cov);
        assert_eq!(w.len(), 3);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}
