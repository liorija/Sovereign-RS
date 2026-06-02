//! # sovereign-quant
//!
//! The quantitative core: Markov chains over market regimes, regime-switching
//! Monte-Carlo (with a stable-Rust SIMD kernel), and a Gaussian HMM regime
//! detector. Every public entry point is hardened against adversarial input
//! (negative prices, infinite volatility, NaN payloads) and proven panic-free
//! by `proptest`.
#![forbid(unsafe_code)]

pub mod blacklitterman;
pub mod cointegration;
pub mod cvar;
pub mod garch;
pub mod hmm;
pub mod kalman;
pub mod markov;
pub mod montecarlo;
pub mod regime;
pub mod statarb;

pub use cointegration::{engle_granger, half_life, is_stationary, EngleGranger};
pub use cvar::{historical_cvar, parametric_cvar};
pub use garch::Garch11;
pub use hmm::{GaussianEmission, GaussianHmm};
pub use kalman::KalmanHedge;
pub use markov::{MarkovChain, TransitionMatrix};
pub use montecarlo::{MonteCarloResult, RegimeParams, RegimeSwitchingMonteCarlo};
pub use statarb::{evaluate_pair, PairStats, SpreadSignal};
