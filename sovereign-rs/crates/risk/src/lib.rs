//! # sovereign-risk
//!
//! The Kill-House as a composable [`gate::Gate`] pipeline (Monte-Carlo CVaR,
//! tick-rule VPIN, conviction), plus intraday [`guards`] (drawdown ladder,
//! black-swan freeze). Settlement/commission gates and the liquidity gate layer
//! on top using the execution crate; see `docs/ROADMAP.md`.
#![forbid(unsafe_code)]

pub mod gate;
pub mod gates;
pub mod guards;

pub use gate::{
    DayOfBounds, Dim, Gate, GateContext, GateOutcome, KillHouse, StringTensor, TernaryState,
    Verdict, TENSOR_DIMS,
};
pub use gates::{ConvictionGate, HyperLayeredVpinGate, MonteCarloGate, VpinGate};
pub use guards::{BlackSwanGuard, DrawdownLadder, SwanState};
