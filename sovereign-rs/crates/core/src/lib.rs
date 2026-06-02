//! # sovereign-core
//!
//! Foundational types shared by every other crate in Sovereign-RS:
//! errors, telemetry, config, the domain model, and the service registry.
//!
//! Design rules enforced workspace-wide:
//! * **No `unwrap`/`expect` on fallible runtime paths** — return [`error::Result`].
//! * **No `log`/`println!`** for events — use [`telemetry`] (`tracing`).
//! * **No global mutable state** — wire dependencies through [`registry`].
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod config;
pub mod domain;
pub mod error;
pub mod registry;
pub mod telemetry;

/// Common imports: `use sovereign_core::prelude::*;`
pub mod prelude {
    pub use crate::config::Config;
    pub use crate::domain::{AssetClass, Instrument, Regime, Session, Side};
    pub use crate::error::{Result, SovereignError};
    pub use crate::registry::ServiceRegistry;
}
