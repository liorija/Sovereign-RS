//! # sovereign-data
//!
//! Resilient data ingestion: the [`provider::DataProvider`] trait, an
//! error-aware [`circuit_breaker::CircuitBreaker`], a [`fallback::FallbackChain`]
//! waterfall that degrades conviction when a proxy source is used, and concrete
//! providers ([`stooq`], [`cboe`], [`sec_edgar`]) over a hardened
//! [`http::HttpClient`] (rustls, polite User-Agent).
//!
//! All parsers are pure and unit-tested offline; only live fetches touch the
//! network (free-tier, rate-limit-friendly).
#![forbid(unsafe_code)]

pub mod cboe;
pub mod circuit_breaker;
pub mod fallback;
pub mod http;
pub mod mock;
pub mod provider;
pub mod sec_edgar;
pub mod stooq;

pub use circuit_breaker::{BreakerRegistry, CircuitBreaker, ErrorClass};
pub use fallback::{FallbackChain, FetchOutcome};
pub use http::HttpClient;
pub use mock::MockProvider;
pub use provider::{DataProvider, Health, PriceBar};
pub use sec_edgar::{count_form4, parse_recent_filings, Filing};
pub use stooq::{parse_stooq_csv, StooqProvider};
