//! Capital-tier-adaptive round-robin scanner that guarantees the *entire*
//! 11k+ universe is eventually evaluated (port of `UniverseRoundRobin`), with
//! scan breadth that scales from micro to large capital.

use std::collections::BTreeSet;

/// Capital tier — drives scan breadth, position count and allocation. Adaptive
/// across the whole range from micro to large.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapitalTier {
    Nano,
    Micro,
    Small,
    Medium,
    Large,
}

impl CapitalTier {
    /// Classify deployable capital ($) into a tier.
    pub fn from_capital(usd: f64) -> Self {
        match usd {
            x if x < 2_000.0 => CapitalTier::Nano,
            x if x < 25_000.0 => CapitalTier::Micro,
            x if x < 100_000.0 => CapitalTier::Small,
            x if x < 1_000_000.0 => CapitalTier::Medium,
            _ => CapitalTier::Large,
        }
    }

    /// How many symbols to evaluate per scan cycle (breadth grows with capital:
    /// micro concentrates, large diversifies).
    pub fn scan_depth(&self) -> usize {
        match self {
            CapitalTier::Nano => 120,
            CapitalTier::Micro => 250,
            CapitalTier::Small => 500,
            CapitalTier::Medium => 900,
            CapitalTier::Large => 1_600,
        }
    }

    /// Maximum concurrent positions.
    pub fn max_positions(&self) -> usize {
        match self {
            CapitalTier::Nano => 3,
            CapitalTier::Micro => 6,
            CapitalTier::Small => 12,
            CapitalTier::Medium => 25,
            CapitalTier::Large => 60,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CapitalTier::Nano => "NANO",
            CapitalTier::Micro => "MICRO",
            CapitalTier::Small => "SMALL",
            CapitalTier::Medium => "MEDIUM",
            CapitalTier::Large => "LARGE",
        }
    }
}

/// The tradeable universe: a deduplicated master symbol list.
#[derive(Debug, Clone, Default)]
pub struct Universe {
    master: Vec<String>,
}

impl Universe {
    /// Start from the always-present multi-asset overlay.
    pub fn multi_asset() -> Self {
        let mut seen = BTreeSet::new();
        let mut master = Vec::new();
        for s in crate::lists::multi_asset() {
            if seen.insert(s.clone()) {
                master.push(s);
            }
        }
        Self { master }
    }

    /// Extend with a runtime-loaded equity list (deduplicated).
    pub fn with_equities(mut self, equities: impl IntoIterator<Item = String>) -> Self {
        let mut seen: BTreeSet<String> = self.master.iter().cloned().collect();
        for s in equities {
            if seen.insert(s.clone()) {
                self.master.push(s);
            }
        }
        self
    }

    /// Add `n` synthetic equity tickers (for full-scale offline scan testing).
    pub fn with_synthetic_equities(self, n: usize) -> Self {
        self.with_equities((0..n).map(|i| format!("SYN{i:05}")))
    }

    pub fn len(&self) -> usize {
        self.master.len()
    }
    pub fn is_empty(&self) -> bool {
        self.master.is_empty()
    }
    pub fn master(&self) -> &[String] {
        &self.master
    }
}

/// Round-robin cursor over the master universe — every cycle serves the next
/// `scan_depth` symbols so all 11k+ are eventually covered.
#[derive(Debug, Default)]
pub struct RoundRobin {
    ptr: usize,
    served: u64,
    cycles: u64,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the next `n` symbols (wraps around the end of `master`).
    pub fn next_batch<'a>(&mut self, n: usize, master: &'a [String]) -> Vec<&'a str> {
        let len = master.len();
        if len == 0 || n == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n.min(len));
        for _ in 0..n {
            out.push(master[self.ptr % len].as_str());
            self.ptr = (self.ptr + 1) % len;
        }
        self.served += n as u64;
        self.cycles += 1;
        out
    }

    /// Fraction of the universe served since startup (can exceed 1.0 over time).
    pub fn coverage(&self, master_len: usize) -> f64 {
        if master_len == 0 {
            0.0
        } else {
            self.served as f64 / master_len as f64
        }
    }

    pub fn cycles(&self) -> u64 {
        self.cycles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn tiers_scale_with_capital() {
        assert_eq!(CapitalTier::from_capital(500.0), CapitalTier::Nano);
        assert_eq!(CapitalTier::from_capital(50_000.0), CapitalTier::Small);
        assert_eq!(CapitalTier::from_capital(5_000_000.0), CapitalTier::Large);
        // breadth is monotonic micro→large
        assert!(CapitalTier::Nano.scan_depth() < CapitalTier::Large.scan_depth());
    }

    #[test]
    fn round_robin_covers_full_11k_universe() {
        let uni = Universe::multi_asset().with_synthetic_equities(11_000);
        assert!(uni.len() > 11_000);
        let mut rr = RoundRobin::new();
        let depth = CapitalTier::Large.scan_depth();
        let mut seen = std::collections::HashSet::new();
        // Enough cycles to wrap the whole universe at least once.
        let cycles = uni.len() / depth + 2;
        for _ in 0..cycles {
            for s in rr.next_batch(depth, uni.master()) {
                seen.insert(s.to_string());
            }
        }
        assert_eq!(seen.len(), uni.len(), "every symbol must be scanned");
        assert!(rr.coverage(uni.len()) >= 1.0);
    }

    proptest! {
        #[test]
        fn batch_size_and_wraparound(n in 1usize..50, depth in 1usize..20) {
            let uni = Universe::default().with_synthetic_equities(n);
            let mut rr = RoundRobin::new();
            let batch = rr.next_batch(depth, uni.master());
            prop_assert_eq!(batch.len(), depth); // always returns exactly `depth` (wrapping)
        }
    }
}
