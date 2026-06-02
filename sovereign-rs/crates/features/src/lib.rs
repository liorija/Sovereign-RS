//! # sovereign-features
//!
//! Vectorized technical-indicator feature engineering, a faithful `ndarray` port
//! of `build_features_polars.py`. Polars/Parquet I/O (zero-copy load of the
//! `sovereign_hive_cache.parquet`) layers on top via the `polars` crate; see
//! `docs/ROADMAP.md`.
#![forbid(unsafe_code)]

pub mod frame;
pub mod indicators;

pub use frame::{FeatureFrame, FEATURE_NAMES, MIN_BARS};
