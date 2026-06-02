//! T+1 settlement carousel + commission gate (ports of `SettlementCarousel`
//! and `CommissionGate`).
//!
//! Cash from a sale is not immediately re-deployable: it settles on T+1. The
//! carousel tracks pending vs. settled cash so the buy loop never spends
//! unsettled funds (avoiding good-faith-violation risk).

/// Tracks unsettled sale proceeds and releases them on their settlement day.
#[derive(Debug, Default, Clone)]
pub struct SettlementCarousel {
    /// `(settlement_day, amount)` still pending.
    pending: Vec<(i64, f64)>,
    settled: f64,
}

impl SettlementCarousel {
    /// New carousel seeded with `settled_cash` already available.
    pub fn new(settled_cash: f64) -> Self {
        Self {
            pending: Vec::new(),
            settled: settled_cash.max(0.0),
        }
    }

    /// Record a sale on `day`; proceeds settle on `day + 1`.
    pub fn record_sale(&mut self, day: i64, amount: f64) {
        if amount.is_finite() && amount > 0.0 {
            self.pending.push((day + 1, amount));
        }
    }

    /// Advance the clock to `today`, releasing everything settling on or before.
    pub fn settle(&mut self, today: i64) {
        let mut still_pending = Vec::with_capacity(self.pending.len());
        for (settle_day, amount) in self.pending.drain(..) {
            if settle_day <= today {
                self.settled += amount;
            } else {
                still_pending.push((settle_day, amount));
            }
        }
        self.pending = still_pending;
    }

    /// Cash available to deploy right now.
    pub fn available(&self) -> f64 {
        self.settled
    }

    /// Cash still settling.
    pub fn pending_total(&self) -> f64 {
        self.pending.iter().map(|(_, a)| a).sum()
    }

    /// Spend settled cash; returns false (no-op) if insufficient.
    pub fn spend(&mut self, amount: f64) -> bool {
        if amount.is_finite() && amount > 0.0 && amount <= self.settled {
            self.settled -= amount;
            true
        } else {
            false
        }
    }
}

/// Blocks trades whose expected edge can't clear the commission barrier
/// (the Python `$0.35` alpha gate). Requires expected gross profit to exceed a
/// multiple of the fee so post-cost expectancy stays positive.
#[derive(Debug, Clone, Copy)]
pub struct CommissionGate {
    pub fee: f64,
    /// Required profit-to-fee ratio (e.g. 3× the round-trip cost).
    pub min_ratio: f64,
}

impl Default for CommissionGate {
    fn default() -> Self {
        Self {
            fee: 0.35,
            min_ratio: 3.0,
        }
    }
}

impl CommissionGate {
    /// Whether an expected gross dollar profit clears the barrier.
    pub fn passes(&self, expected_gross_profit: f64) -> bool {
        expected_gross_profit > self.fee * self.min_ratio
    }

    /// Convenience: estimate gross profit from size × price × expected move.
    pub fn passes_move(&self, shares: u64, price: f64, expected_move_pct: f64) -> bool {
        self.passes(shares as f64 * price * expected_move_pct.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_plus_one_settlement() {
        let mut c = SettlementCarousel::new(1000.0);
        c.record_sale(10, 500.0); // settles day 11
        c.settle(10);
        assert_eq!(c.available(), 1000.0); // not yet
        assert_eq!(c.pending_total(), 500.0);
        c.settle(11);
        assert_eq!(c.available(), 1500.0);
        assert_eq!(c.pending_total(), 0.0);
    }

    #[test]
    fn cannot_overspend() {
        let mut c = SettlementCarousel::new(100.0);
        assert!(!c.spend(200.0));
        assert!(c.spend(60.0));
        assert_eq!(c.available(), 40.0);
    }

    #[test]
    fn commission_barrier() {
        let g = CommissionGate::default(); // fee 0.35, ratio 3 → need > 1.05
        assert!(!g.passes(1.0));
        assert!(g.passes(2.0));
        assert!(g.passes_move(100, 50.0, 0.01)); // $50 expected
    }
}
