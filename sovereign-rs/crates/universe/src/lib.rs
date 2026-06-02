//! # sovereign-universe
//!
//! The tradeable universe and its scanner. [`Universe`] holds the deduplicated
//! master symbol list (multi-asset overlay + a runtime-loaded ~11k equity list);
//! [`RoundRobin`] guarantees every symbol is eventually evaluated; [`CapitalTier`]
//! makes scan breadth, position count and allocation adapt from micro to large.
#![forbid(unsafe_code)]

pub mod lists;
pub mod scan;

pub use scan::{CapitalTier, RoundRobin, Universe};
