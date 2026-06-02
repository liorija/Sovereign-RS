//! # sovereign-signals
//!
//! Byzantine-fault-tolerant consensus over a panel of voting agents
//! (Technical / Fundamental / Sentiment / Risk), with a regime-adaptive quorum
//! and a hard Risk veto. Alternative-data conviction deltas (the 12-engine
//! `AlternativeDataEngine`) feed in as additional agents; see `docs/ROADMAP.md`.
#![forbid(unsafe_code)]

pub mod agent;
pub mod altdata;
pub mod consensus;

pub use agent::{Action, Agent, SignalContext, Vote};
pub use altdata::{aggregate, AltInputs, AltScore};
pub use consensus::{adaptive_quorum, base_quorum, BftConsensus, ConsensusResult};
