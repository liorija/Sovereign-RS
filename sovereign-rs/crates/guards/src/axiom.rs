//! Gödel/Turing uncertainty guard — the "Axiom Breaker".
//!
//! A predictor that keeps computing on an undecidable input is the trading
//! analogue of the halting problem: when the BFT panel is maximally divided,
//! more computation buys nothing and trusting the near-coin-flip is reckless.
//!
//! We quantify "undecidability" as the **Shannon entropy** of the decision
//! distribution. When normalized entropy approaches its maximum (uniform
//! disagreement), the Axiom Breaker short-circuits prediction and instructs the
//! engine to default to volatility hedging.

/// Normalized Shannon entropy of a (possibly unnormalized) distribution, in
/// `[0, 1]`. `0` = certain (one outcome), `1` = maximally uncertain (uniform).
pub fn shannon_entropy(weights: &[f64]) -> f64 {
    let clean: Vec<f64> = weights
        .iter()
        .copied()
        .filter(|p| p.is_finite() && *p > 0.0)
        .collect();
    let n = clean.len();
    if n < 2 {
        return 0.0;
    }
    let sum: f64 = clean.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for p in &clean {
        let q = p / sum;
        h -= q * q.ln();
    }
    (h / (n as f64).ln()).clamp(0.0, 1.0)
}

/// The halting guard.
#[derive(Debug, Clone, Copy)]
pub struct AxiomBreaker {
    /// Normalized-entropy threshold above which prediction halts.
    pub entropy_threshold: f64,
}

impl Default for AxiomBreaker {
    fn default() -> Self {
        Self {
            entropy_threshold: 0.95,
        }
    }
}

impl AxiomBreaker {
    pub fn new(entropy_threshold: f64) -> Self {
        Self {
            entropy_threshold: entropy_threshold.clamp(0.0, 1.0),
        }
    }

    /// Divergence = normalized entropy of the decision distribution.
    pub fn divergence(&self, distribution: &[f64]) -> f64 {
        shannon_entropy(distribution)
    }

    /// Whether to short-circuit prediction and hedge volatility instead.
    pub fn should_halt(&self, distribution: &[f64]) -> bool {
        self.divergence(distribution) >= self.entropy_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn uniform_is_maximal_entropy() {
        assert!((shannon_entropy(&[0.25, 0.25, 0.25, 0.25]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn peaked_is_low_entropy() {
        assert!(shannon_entropy(&[0.97, 0.01, 0.01, 0.01]) < 0.3);
    }

    #[test]
    fn breaker_halts_on_division() {
        let ab = AxiomBreaker::default();
        assert!(ab.should_halt(&[1.0, 1.0, 1.0, 1.0])); // perfectly split
        assert!(!ab.should_halt(&[0.9, 0.05, 0.05])); // decisive
    }

    proptest! {
        #[test]
        fn entropy_in_unit_interval(
            w in proptest::collection::vec(0.0f64..1000.0, 0..20)
        ) {
            let h = shannon_entropy(&w);
            prop_assert!((0.0..=1.0).contains(&h));
        }
    }
}
