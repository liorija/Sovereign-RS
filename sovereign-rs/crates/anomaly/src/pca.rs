//! Principal Component Analysis via covariance eigendecomposition (nalgebra).
//!
//! Extracts the latent "factors" driving a panel of asset returns — the basis
//! for factor hedging and the Random-Matrix-Theory noise filter in [`crate::rmt`].

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::Array2;

/// Result of a PCA fit, sorted by descending eigenvalue.
#[derive(Debug, Clone)]
pub struct Pca {
    /// Eigenvalues (variance per component), descending.
    pub eigenvalues: Vec<f64>,
    /// Fraction of total variance per component.
    pub explained_ratio: Vec<f64>,
    /// Eigenvectors (loadings) as columns aligned with `eigenvalues`.
    pub components: Vec<Vec<f64>>,
}

/// Fit PCA on a `T × N` matrix (rows = samples, cols = features).
/// Returns `None` if there are fewer than 2 samples or no features.
pub fn pca(data: &Array2<f64>) -> Option<Pca> {
    let (t, n) = data.dim();
    if t < 2 || n == 0 {
        return None;
    }
    // Column means.
    let means: Vec<f64> = (0..n).map(|j| data.column(j).sum() / t as f64).collect();
    // Covariance (N × N), unbiased.
    let mut cov = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..t {
                s += (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]);
            }
            let c = s / (t as f64 - 1.0);
            cov[(i, j)] = c;
            cov[(j, i)] = c;
        }
    }

    let eig = SymmetricEigen::new(cov);
    // Pair eigenvalues with their eigenvectors and sort descending.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total: f64 = eig
        .eigenvalues
        .iter()
        .map(|v| v.max(0.0))
        .sum::<f64>()
        .max(1e-12);
    let eigenvalues: Vec<f64> = idx.iter().map(|&i| eig.eigenvalues[i]).collect();
    let explained_ratio: Vec<f64> = eigenvalues.iter().map(|v| v.max(0.0) / total).collect();
    let components: Vec<Vec<f64>> = idx
        .iter()
        .map(|&i| eig.eigenvectors.column(i).iter().copied().collect())
        .collect();

    Some(Pca {
        eigenvalues,
        explained_ratio,
        components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn first_component_dominates_correlated_data() {
        // Two highly correlated features: y ≈ 2x.
        let mut data = Array2::<f64>::zeros((100, 2));
        for k in 0..100 {
            let x = (k as f64 * 0.1).sin();
            data[[k, 0]] = x;
            data[[k, 1]] = 2.0 * x + 0.001 * (k as f64).cos();
        }
        let p = pca(&data).unwrap();
        assert_eq!(p.eigenvalues.len(), 2);
        // PC1 should explain the overwhelming majority of variance.
        assert!(p.explained_ratio[0] > 0.98, "ratio {:?}", p.explained_ratio);
        // Descending order.
        assert!(p.eigenvalues[0] >= p.eigenvalues[1]);
    }

    #[test]
    fn rejects_degenerate() {
        assert!(pca(&Array2::<f64>::zeros((1, 3))).is_none());
    }
}
