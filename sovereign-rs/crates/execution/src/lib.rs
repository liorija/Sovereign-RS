//! # sovereign-execution
//!
//! The order side of the engine: routing a typed [`sovereign_core::domain::Instrument`]
//! to a broker [`contract::ContractSpec`], building [`order::Order`]s and
//! [`order::BracketOrder`]s, inverse-volatility [`sizing`], the T+1
//! [`settlement::SettlementCarousel`] and the [`settlement::CommissionGate`].
//!
//! The live IBKR adapter (actually placing orders over TWS/Gateway) is the one
//! environment-dependent piece, tracked in `docs/ROADMAP.md`; everything here is
//! pure and unit-tested.
#![forbid(unsafe_code)]

pub mod contract;
pub mod order;
pub mod settlement;
pub mod sizing;

pub use contract::{front_month, route, ContractSpec, SecType};
pub use order::{BracketOrder, Order, OrderType};
pub use settlement::{CommissionGate, SettlementCarousel};
pub use sizing::{InverseVolSizer, SizeResult};
