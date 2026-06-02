//! Byzantine-fault-tolerant consensus over agent votes.
//!
//! Port of `sovereign_v328_mega_patch._BFT_QUORUM_MAP` + `BFTConsensus.evaluate`
//! (FIX-5/FIX-8): a regime-adaptive quorum, a weighted-confidence tally, and a
//! hard Risk veto that short-circuits everything.

use sovereign_core::domain::Regime;

use crate::agent::{Action, Agent, SignalContext, Vote};

/// Base quorum (number of BUY votes required out of all agents) per regime.
pub fn base_quorum(regime: Regime) -> u32 {
    match regime {
        Regime::Bull | Regime::Goldilocks | Regime::Recovery => 3,
        Regime::Sideways | Regime::Reflation => 2, // ambiguous market → lower bar
        Regime::Bear | Regime::Stagflation => 3,   // defensive → stay strict
        Regime::Crisis => 4,                       // extreme → unanimity
    }
}

/// Adjusted quorum: NANO/MICRO tiers and core tickers each relax it by one
/// (floored at 2), mirroring `_get_bft_quorum`.
pub fn adaptive_quorum(regime: Regime, is_small_tier: bool, is_core: bool) -> u32 {
    let mut q = base_quorum(regime);
    if is_small_tier {
        q = q.saturating_sub(1).max(2);
    }
    if is_core {
        q = q.saturating_sub(1).max(2);
    }
    q
}

/// The outcome of a consensus round.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub approved: bool,
    pub action: Action,
    pub buy_count: u32,
    pub quorum: u32,
    /// Weighted fraction of confidence that voted BUY, in `[0,1]`.
    pub weighted_buy_pct: f64,
    pub reason: String,
    pub votes: Vec<(String, Vote)>,
}

/// The consensus engine over a fixed panel of agents.
pub struct BftConsensus {
    agents: Vec<Box<dyn Agent>>,
}

impl BftConsensus {
    /// Build from a panel of agents (typically Tech/Fund/Sent/Risk).
    pub fn new(agents: Vec<Box<dyn Agent>>) -> Self {
        Self { agents }
    }

    /// Run a consensus round for one candidate.
    pub fn evaluate(
        &self,
        ctx: &SignalContext,
        is_small_tier: bool,
        is_core: bool,
    ) -> ConsensusResult {
        let mut votes: Vec<(String, Vote)> = Vec::with_capacity(self.agents.len());
        for agent in &self.agents {
            votes.push((agent.name().to_string(), agent.vote(ctx)));
        }

        // Risk veto short-circuits (FIX-8 semantics).
        if let Some((name, v)) = votes.iter().find(|(_, v)| v.veto) {
            return ConsensusResult {
                approved: false,
                action: Action::Reject,
                buy_count: 0,
                quorum: adaptive_quorum(ctx.regime, is_small_tier, is_core),
                weighted_buy_pct: 0.0,
                reason: format!("🛑 VETO by {name}: {}", v.reason),
                votes,
            };
        }

        let quorum = adaptive_quorum(ctx.regime, is_small_tier, is_core);
        let total_weight: f64 = self
            .agents
            .iter()
            .map(|a| a.weight())
            .sum::<f64>()
            .max(1e-9);
        let mut buy_weight = 0.0;
        let mut buy_count = 0u32;
        for (agent, (_, v)) in self.agents.iter().zip(&votes) {
            if v.action == Action::Buy {
                buy_count += 1;
                buy_weight += agent.weight() * v.confidence;
            }
        }
        let weighted_buy_pct = buy_weight / total_weight;
        let approved = buy_count >= quorum;

        let (action, reason) = if approved {
            (
                Action::Buy,
                format!(
                    "✅ BFT {buy_count}/{} APPROVE (q={quorum}, w={:.0}%)",
                    self.agents.len(),
                    weighted_buy_pct * 100.0
                ),
            )
        } else if buy_count >= 2 {
            (
                Action::Hold,
                format!(
                    "⚠️ BFT {buy_count}/{} — quorum {quorum} not met",
                    self.agents.len()
                ),
            )
        } else {
            (
                Action::Reject,
                format!(
                    "❌ BFT {buy_count}/{} REJECT (q={quorum})",
                    self.agents.len()
                ),
            )
        };

        ConsensusResult {
            approved,
            action,
            buy_count,
            quorum,
            weighted_buy_pct,
            reason,
            votes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// An agent that always casts the same vote (for deterministic tests).
    struct Fixed(&'static str, Action, f64, bool);
    impl Agent for Fixed {
        fn name(&self) -> &str {
            self.0
        }
        fn vote(&self, _ctx: &SignalContext) -> Vote {
            if self.3 {
                Vote::veto("risk")
            } else {
                Vote::new(self.1, self.2, "fixed")
            }
        }
    }

    fn ctx(regime: Regime) -> SignalContext {
        SignalContext {
            ticker: "AAPL".into(),
            regime,
            conviction: 10.0,
            vix_score: 0.0,
            heat: 0.1,
        }
    }

    #[test]
    fn veto_short_circuits() {
        let panel: Vec<Box<dyn Agent>> = vec![
            Box::new(Fixed("tech", Action::Buy, 0.9, false)),
            Box::new(Fixed("risk", Action::Reject, 1.0, true)),
        ];
        let c = BftConsensus::new(panel);
        let r = c.evaluate(&ctx(Regime::Bull), false, false);
        assert!(!r.approved);
        assert!(r.reason.contains("VETO"));
    }

    #[test]
    fn sideways_needs_only_two() {
        let panel: Vec<Box<dyn Agent>> = vec![
            Box::new(Fixed("tech", Action::Buy, 0.8, false)),
            Box::new(Fixed("fund", Action::Buy, 0.7, false)),
            Box::new(Fixed("sent", Action::Hold, 0.5, false)),
            Box::new(Fixed("risk", Action::Hold, 0.5, false)),
        ];
        let c = BftConsensus::new(panel);
        let r = c.evaluate(&ctx(Regime::Sideways), false, false);
        assert_eq!(r.quorum, 2);
        assert!(r.approved);
    }

    proptest! {
        #[test]
        fn crisis_quorum_is_strictest(is_small in any::<bool>(), is_core in any::<bool>()) {
            // Even with both relaxations, crisis stays >= 2 and >= other regimes' relaxed quorum.
            let q = adaptive_quorum(Regime::Crisis, is_small, is_core);
            prop_assert!(q >= 2);
            prop_assert!(q >= adaptive_quorum(Regime::Sideways, is_small, is_core));
        }
    }
}
