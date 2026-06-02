//! # sovereign-engine
//!
//! Orchestration layer. Composes [`sovereign_features`], [`sovereign_signals`]
//! and [`sovereign_risk`] into a single per-candidate [`pipeline::Pipeline`],
//! and runs the [`selfheal::SelfHealer`] background task. Wiring is explicit
//! (constructor injection / [`sovereign_core::registry::ServiceRegistry`]) — no
//! global mutable state.
#![forbid(unsafe_code)]

pub mod agents;
pub mod pipeline;
pub mod selfheal;

pub use pipeline::{pin_to_last_core, CoreIsolation, Decision, IsolatedExecutor, Pipeline};
pub use selfheal::{HealReport, SelfHealer};
