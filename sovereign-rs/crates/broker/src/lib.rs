//! # sovereign-broker
//!
//! Order routing behind one [`Broker`] trait: a deterministic in-memory
//! [`PaperBroker`] for backtests/dry-runs, and an [`IbkrBroker`] whose account
//! and connection settings are read from the environment (`IBKR_HOST`,
//! `IBKR_PORT`, `IBKR_CLIENT_ID`, `IBKR_ACCOUNT`). The live TWS/Gateway socket
//! is gated — until connected, the IBKR broker returns a typed error rather
//! than pretending to trade.
#![forbid(unsafe_code)]

pub mod broker;

pub use broker::{AccountSummary, Broker, IbkrBroker, IbkrConfig, OrderId, PaperBroker, Position};
