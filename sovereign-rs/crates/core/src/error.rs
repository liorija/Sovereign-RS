//! Central error type for the whole engine.
//!
//! Directive compliance: we never `.unwrap()`/`.expect()` on fallible runtime
//! paths. Every fallible function returns [`Result`], and errors carry enough
//! context to be actionable from a `tracing` log line.

use thiserror::Error;

/// The one error type to rule them all. Library crates return this; the binary
/// may wrap it in `anyhow::Error` at the very top for ergonomic reporting.
#[derive(Debug, Error)]
pub enum SovereignError {
    /// A data provider failed (network, rate-limit, malformed upstream payload).
    #[error("data provider `{provider}` failed: {reason}")]
    Data { provider: String, reason: String },

    /// All providers in a fallback chain were exhausted.
    #[error("all {tried} providers failed for `{key}`")]
    AllProvidersFailed { key: String, tried: usize },

    /// A circuit breaker is open; the call was short-circuited.
    #[error("circuit `{source_name}` is OPEN (cooldown {cooldown_s:.0}s remaining)")]
    CircuitOpen {
        source_name: String,
        cooldown_s: f64,
    },

    /// Serialization / deserialization of an API or SEC-EDGAR payload failed.
    #[error("serde error in `{context}`: {source}")]
    Serde {
        context: String,
        #[source]
        source: serde_json::Error,
    },

    /// A quantitative model received numerically invalid input or diverged.
    #[error("quant model `{model}`: {reason}")]
    Quant { model: String, reason: String },

    /// A risk gate vetoed an action (this is a *normal* control-flow outcome,
    /// surfaced as an error only when a caller treats a veto as fatal).
    #[error("risk veto from `{gate}`: {reason}")]
    RiskVeto { gate: String, reason: String },

    /// Configuration could not be loaded or validated.
    #[error("config error: {0}")]
    Config(String),

    /// A requested service was not present in the [`crate::registry::ServiceRegistry`].
    #[error("service `{0}` not registered")]
    ServiceMissing(&'static str),

    /// Wrapped I/O error.
    #[error("io error in `{context}`: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    /// A sub-system panicked and was caught by a `FaultBoundary`.
    #[error("component `{component}` panicked: {message}")]
    Panic { component: String, message: String },

    /// The engine halted itself (kill switch / axiom breaker / thermal guard).
    #[error("engine halted: {0}")]
    Halted(String),
}

impl SovereignError {
    /// Convenience constructor for data-provider failures.
    pub fn data(provider: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Data {
            provider: provider.into(),
            reason: reason.into(),
        }
    }

    /// Convenience constructor for quant-model failures.
    pub fn quant(model: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Quant {
            model: model.into(),
            reason: reason.into(),
        }
    }
}

/// Crate-wide `Result` alias.
pub type Result<T> = std::result::Result<T, SovereignError>;
