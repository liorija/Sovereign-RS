//! # sovereign-ml
//!
//! ML inference behind one trait. A native [`logistic::LogisticRegression`]
//! provides a dependency-free baseline today; the production RF/LightGBM/XGBoost
//! ensemble (trained with purged CV in Python) loads as ONNX via `tract` in a
//! later milestone and implements the same [`Model`] trait — so the signal layer
//! never needs to know which backend it's talking to.
#![forbid(unsafe_code)]

pub mod logistic;

pub use logistic::LogisticRegression;

/// A probabilistic classifier: returns `P(up-move)` in `[0, 1]`.
pub trait Model: Send + Sync {
    /// Probability of a positive outcome given a feature row.
    fn predict_proba(&self, features: &[f64]) -> f64;

    /// Convenience: hard label at a 0.5 threshold.
    fn predict(&self, features: &[f64]) -> bool {
        self.predict_proba(features) >= 0.5
    }
}
