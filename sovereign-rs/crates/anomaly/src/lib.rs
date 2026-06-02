//! # sovereign-anomaly
//!
//! Structural-anomaly mathematics — the "exploit" toolkit:
//! * [`kelly`] — growth-optimal bet sizing.
//! * [`pca`] / [`rmt`] — latent factor extraction + Marchenko-Pastur noise filtering.
//! * [`hawkes`] — self-exciting order-flow aftershocks.
//! * [`copula`] — Kendall's τ and crisis tail-dependence (Sklar).
//! * [`kl`] — Kullback-Leibler / Jensen-Shannon model-vs-reality drift.
//! * [`spectral`] — dependency-free FFT + dominant-cycle detection.
//! * [`lyapunov`] — largest Lyapunov exponent (chaos horizon).
//!
//! All pure, dependency-light (ndarray + nalgebra), and property-tested.
#![forbid(unsafe_code)]

pub mod copula;
pub mod hawkes;
pub mod kelly;
pub mod kl;
pub mod lyapunov;
pub mod pca;
pub mod rmt;
pub mod spectral;

pub use copula::{kendall_tau, tail_dependence};
pub use hawkes::Hawkes;
pub use kelly::{fractional_kelly, kelly_fraction, kelly_from_returns};
pub use kl::{js_divergence, kl_divergence};
pub use lyapunov::largest_lyapunov;
pub use pca::{pca, Pca};
pub use rmt::{clean_eigenvalues, marchenko_pastur_bounds, signal_count};
pub use spectral::{dominant_period, power_spectrum};
