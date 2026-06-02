//! # sovereign-physics
//!
//! Physics & complex-systems models applied to markets:
//! * [`mechanics`] — Newtonian price kinematics (velocity/acceleration/jerk),
//!   Le Chatelier equilibrium pressure, financial Heisenberg uncertainty.
//! * [`stochastic`] — Ornstein-Uhlenbeck fit, Itô / Euler-Maruyama, Fokker-Planck.
//! * [`optimization`] — Markowitz mean-variance + simplex projection (convex).
//! * [`complex_systems`] — Lotka-Volterra (MM vs HFT), Rule-30 automata, Arrhenius/catalyst.
//! * [`wavefunction`] — Schrödinger-inspired price probability cloud.
//!
//! Pure Rust (ndarray + nalgebra), fully unit/property-tested.
#![forbid(unsafe_code)]

pub mod complex_systems;
pub mod mechanics;
pub mod optimization;
pub mod stochastic;
pub mod wavefunction;

pub use complex_systems::{
    arrhenius_rate, breakout_ready, liquidity_exhausted, lotka_volterra, rule30_step, LvParams,
};
pub use mechanics::{kinematics, Equilibrium, Kinematics};
pub use optimization::{mean_variance_weights, min_variance_weights, project_simplex};
pub use stochastic::{fit_ou, fokker_planck_evolve, ito_log_drift, OuParams};
pub use wavefunction::WaveFunction;
