//! BFT agent abstraction (Technical / Fundamental / Sentiment / Risk).

use sovereign_core::domain::Regime;

/// What an agent recommends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Buy,
    Hold,
    Reject,
}

/// One agent's vote.
#[derive(Debug, Clone)]
pub struct Vote {
    pub action: Action,
    /// Confidence in `[0,1]`.
    pub confidence: f64,
    /// A hard veto (Risk agent) short-circuits consensus regardless of others.
    pub veto: bool,
    pub reason: String,
}

impl Vote {
    /// Helper for a plain buy/hold/reject vote.
    pub fn new(action: Action, confidence: f64, reason: impl Into<String>) -> Self {
        Self {
            action,
            confidence: confidence.clamp(0.0, 1.0),
            veto: false,
            reason: reason.into(),
        }
    }

    /// A vetoing vote (Risk agent).
    pub fn veto(reason: impl Into<String>) -> Self {
        Self {
            action: Action::Reject,
            confidence: 1.0,
            veto: true,
            reason: reason.into(),
        }
    }
}

/// Everything an agent needs to vote on one candidate.
#[derive(Debug, Clone)]
pub struct SignalContext {
    pub ticker: String,
    pub regime: Regime,
    /// Composite conviction score accumulated so far.
    pub conviction: f64,
    /// VIX-derived risk score in roughly `[-1, 1]`.
    pub vix_score: f64,
    /// Current portfolio heat in `[0, 1]`.
    pub heat: f64,
}

/// A voting agent. Implementations are pure (no I/O) so they're trivially
/// testable and `Send + Sync` for parallel evaluation.
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    /// Relative weight in the weighted-confidence tally.
    fn weight(&self) -> f64 {
        1.0
    }
    fn vote(&self, ctx: &SignalContext) -> Vote;
}
