//! Schrödinger-inspired price "probability cloud": the price is modelled as a
//! wave function over discrete levels whose `|ψ|²` is the probability of being
//! observed there. The cloud "collapses" to a concrete price when an institution
//! executes (observes) — a metaphor for reflexive, observation-driven pricing.

use rand::Rng;

/// A discretized 1-D wave function over price levels.
#[derive(Debug, Clone)]
pub struct WaveFunction {
    levels: Vec<f64>,
    amplitudes: Vec<f64>,
}

impl WaveFunction {
    /// A Gaussian wave packet centred at `center` with width `spread`, sampled on
    /// `n` levels across `[lo, hi]`. `|ψ|²` is then a discretized Gaussian.
    pub fn gaussian_packet(center: f64, spread: f64, lo: f64, hi: f64, n: usize) -> Self {
        let n = n.max(1);
        let s = spread.abs().max(1e-9);
        let levels: Vec<f64> = if n == 1 || (hi - lo).abs() < 1e-12 {
            vec![center; n]
        } else {
            (0..n)
                .map(|i| lo + (hi - lo) * i as f64 / (n as f64 - 1.0))
                .collect()
        };
        // amplitude ∝ exp(−(x−c)² / (4σ²)) ⇒ |ψ|² ∝ exp(−(x−c)² / (2σ²)).
        let amplitudes = levels
            .iter()
            .map(|x| (-((x - center).powi(2)) / (4.0 * s * s)).exp())
            .collect();
        Self { levels, amplitudes }
    }

    /// Born-rule probabilities `|ψ|²`, normalized to sum to 1.
    pub fn probabilities(&self) -> Vec<f64> {
        let sq: Vec<f64> = self.amplitudes.iter().map(|a| a * a).collect();
        let total: f64 = sq.iter().sum();
        if total <= 1e-300 {
            let n = sq.len().max(1);
            return vec![1.0 / n as f64; sq.len()];
        }
        sq.iter().map(|p| p / total).collect()
    }

    /// Expected (mean) price `Σ xᵢ |ψᵢ|²`.
    pub fn expected_price(&self) -> f64 {
        self.levels
            .iter()
            .zip(self.probabilities())
            .map(|(x, p)| x * p)
            .sum()
    }

    /// Collapse the wave function to a single observed price (inverse-CDF sample).
    pub fn collapse<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let probs = self.probabilities();
        let u: f64 = rng.gen::<f64>();
        let mut acc = 0.0;
        for (x, p) in self.levels.iter().zip(&probs) {
            acc += p;
            if u <= acc {
                return *x;
            }
        }
        self.levels.last().copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn probabilities_normalize() {
        let wf = WaveFunction::gaussian_packet(100.0, 2.0, 90.0, 110.0, 201);
        let p = wf.probabilities();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(p.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn expected_price_near_center() {
        let wf = WaveFunction::gaussian_packet(100.0, 2.0, 90.0, 110.0, 201);
        assert!((wf.expected_price() - 100.0).abs() < 0.1);
    }

    #[test]
    fn collapse_lands_in_range() {
        let wf = WaveFunction::gaussian_packet(100.0, 3.0, 90.0, 110.0, 101);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..1000 {
            let x = wf.collapse(&mut rng);
            assert!((90.0..=110.0).contains(&x));
        }
    }
}
