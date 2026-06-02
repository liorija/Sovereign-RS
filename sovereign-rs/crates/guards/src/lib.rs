//! # sovereign-guards
//!
//! The defensive "Soul" of Sovereign V331, translated to Rust with fidelity:
//!
//! * [`killswitch::GlobalKillSwitch`] — lock-free global halt (no `Arc<Mutex>`).
//! * [`fault::FaultBoundary`] — `catch_unwind` isolation; fault storms escalate.
//! * [`thermal::ThermodynamicGuard`] — degrade sim depth as the CPU heats.
//! * [`axiom::AxiomBreaker`] — Gödel/Turing entropy halting → hedge on max divergence.
//!
//! These are cross-cutting safety primitives; the engine wires them via message
//! passing (`tokio::sync::mpsc`) rather than shared locks.
#![forbid(unsafe_code)]

pub mod axiom;
pub mod fault;
pub mod killswitch;
pub mod thermal;

pub use axiom::{shannon_entropy, AxiomBreaker};
pub use fault::FaultBoundary;
pub use killswitch::{GlobalKillSwitch, KillReason};
pub use thermal::{read_cpu_temp_celsius, ThermodynamicGuard};
