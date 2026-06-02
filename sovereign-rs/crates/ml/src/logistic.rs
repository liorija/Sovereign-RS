//! L2-regularized logistic regression trained by gradient descent, with
//! built-in feature standardization.
//!
//! This is the native, dependency-free baseline for the V319 ML ensemble. The
//! production RF/LightGBM/XGBoost stack is loaded as ONNX via `tract` in a later
//! milestone; both implement [`crate::Model`], so callers don't change.

use serde::{Deserialize, Serialize};

use crate::Model;

/// A trained logistic-regression classifier (predicts P(up-move)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    mean: Vec<f64>,
    std: Vec<f64>,
}

#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl LogisticRegression {
    /// Train on `x` (rows = samples, inner = features) and binary labels `y`.
    ///
    /// Features are standardized internally. `epochs`, `lr` (learning rate) and
    /// `l2` (ridge penalty) control fitting. Returns `None` if the data is empty
    /// or ragged.
    pub fn train(x: &[Vec<f64>], y: &[u8], epochs: usize, lr: f64, l2: f64) -> Option<Self> {
        let n = x.len();
        if n == 0 || y.len() != n {
            return None;
        }
        let d = x[0].len();
        if d == 0 || x.iter().any(|row| row.len() != d) {
            return None;
        }

        // Standardization parameters.
        let mut mean = vec![0.0; d];
        for row in x {
            for (j, v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut std = vec![0.0; d];
        for row in x {
            for (j, v) in row.iter().enumerate() {
                std[j] += (v - mean[j]).powi(2);
            }
        }
        for s in &mut std {
            *s = (*s / n as f64).sqrt().max(1e-9);
        }

        // Standardized design matrix.
        let xs: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (v - mean[j]) / std[j])
                    .collect()
            })
            .collect();

        let mut weights = vec![0.0; d];
        let mut bias = 0.0;
        let inv_n = 1.0 / n as f64;

        for _ in 0..epochs {
            let mut grad_w = vec![0.0; d];
            let mut grad_b = 0.0;
            for (row, &label) in xs.iter().zip(y) {
                let z = bias + row.iter().zip(&weights).map(|(a, b)| a * b).sum::<f64>();
                let err = sigmoid(z) - label as f64;
                grad_b += err;
                for (g, v) in grad_w.iter_mut().zip(row) {
                    *g += err * v;
                }
            }
            bias -= lr * grad_b * inv_n;
            for (w, g) in weights.iter_mut().zip(&grad_w) {
                *w -= lr * (g * inv_n + l2 * *w);
            }
        }

        Some(Self {
            weights,
            bias,
            mean,
            std,
        })
    }

    /// Number of input features.
    pub fn n_features(&self) -> usize {
        self.weights.len()
    }
}

impl Model for LogisticRegression {
    fn predict_proba(&self, features: &[f64]) -> f64 {
        if features.len() != self.weights.len() {
            return 0.5; // shape mismatch → neutral
        }
        let z = self.bias
            + features
                .iter()
                .enumerate()
                .map(|(j, v)| ((v - self.mean[j]) / self.std[j]) * self.weights[j])
                .sum::<f64>();
        let p = sigmoid(z);
        if p.is_finite() {
            p
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Linearly separable: label = 1 iff x0 + x1 > 0.
    fn dataset(n: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = (i as f64 * 0.7).sin();
            let b = (i as f64 * 0.3).cos();
            x.push(vec![a, b]);
            y.push(if a + b > 0.0 { 1 } else { 0 });
        }
        (x, y)
    }

    #[test]
    fn learns_separable_pattern() {
        let (x, y) = dataset(400);
        let model = LogisticRegression::train(&x, &y, 500, 0.5, 1e-4).unwrap();
        let correct = x
            .iter()
            .zip(&y)
            .filter(|(row, &label)| (model.predict_proba(row) >= 0.5) == (label == 1))
            .count();
        let acc = correct as f64 / x.len() as f64;
        assert!(acc > 0.85, "accuracy {acc}");
    }

    #[test]
    fn rejects_ragged_input() {
        let x = vec![vec![1.0, 2.0], vec![1.0]];
        assert!(LogisticRegression::train(&x, &[0, 1], 10, 0.1, 0.0).is_none());
    }

    proptest! {
        #[test]
        fn proba_always_in_unit_interval(
            feats in proptest::collection::vec(-100.0f64..100.0, 2..=2),
        ) {
            let (x, y) = dataset(100);
            let model = LogisticRegression::train(&x, &y, 50, 0.3, 1e-4).unwrap();
            let p = model.predict_proba(&feats);
            prop_assert!((0.0..=1.0).contains(&p));
        }
    }
}
