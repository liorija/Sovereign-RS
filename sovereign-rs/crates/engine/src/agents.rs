//! The concrete 4-agent BFT panel (Technical / Fundamental / Sentiment / Risk).
//!
//! These are intentionally simple, pure functions of the [`SignalContext`]; the
//! richer scoring (alt-data, factor model, ML ensemble) plugs in as additional
//! agents without touching the consensus core.

use sovereign_signals::agent::{Action, Agent, SignalContext, Vote};

/// Technical agent — buys on positive conviction, scaled by magnitude.
pub struct TechnicalAgent;
impl Agent for TechnicalAgent {
    fn name(&self) -> &str {
        "TECH"
    }
    fn weight(&self) -> f64 {
        1.2
    }
    fn vote(&self, ctx: &SignalContext) -> Vote {
        if ctx.conviction > 6.0 {
            Vote::new(Action::Buy, (ctx.conviction / 20.0).min(0.95), "momentum+")
        } else if ctx.conviction < -2.0 {
            Vote::new(Action::Reject, 0.6, "momentum-")
        } else {
            Vote::new(Action::Hold, 0.4, "neutral")
        }
    }
}

/// Fundamental agent — mild buy bias outside defensive regimes.
pub struct FundamentalAgent;
impl Agent for FundamentalAgent {
    fn name(&self) -> &str {
        "FUND"
    }
    fn vote(&self, ctx: &SignalContext) -> Vote {
        let mult = ctx.regime.allocation_multiplier();
        if mult >= 0.6 && ctx.conviction > 4.0 {
            Vote::new(Action::Buy, 0.55 * mult, "constructive")
        } else {
            Vote::new(Action::Hold, 0.4, "wait")
        }
    }
}

/// Sentiment agent — leans on the VIX score.
pub struct SentimentAgent;
impl Agent for SentimentAgent {
    fn name(&self) -> &str {
        "SENT"
    }
    fn vote(&self, ctx: &SignalContext) -> Vote {
        if ctx.vix_score > 0.2 {
            Vote::new(Action::Buy, 0.6, "calm tape")
        } else if ctx.vix_score < -0.5 {
            Vote::new(Action::Reject, 0.6, "fear spike")
        } else {
            Vote::new(Action::Hold, 0.4, "mixed")
        }
    }
}

/// Risk agent — hard-vetoes on overheated portfolio, else neutral-to-buy.
pub struct RiskAgent;
impl Agent for RiskAgent {
    fn name(&self) -> &str {
        "RISK"
    }
    fn vote(&self, ctx: &SignalContext) -> Vote {
        if ctx.heat > 0.9 {
            Vote::veto(format!("portfolio heat {:.0}% > 90%", ctx.heat * 100.0))
        } else if ctx.heat < 0.6 {
            Vote::new(Action::Buy, 0.6, "risk budget available")
        } else {
            Vote::new(Action::Hold, 0.5, "heat elevated")
        }
    }
}

/// The default panel used by the demo pipeline.
pub fn default_panel() -> Vec<Box<dyn Agent>> {
    vec![
        Box::new(TechnicalAgent),
        Box::new(FundamentalAgent),
        Box::new(SentimentAgent),
        Box::new(RiskAgent),
    ]
}
