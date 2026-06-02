//! Hawkes self-exciting point process with an exponential kernel.
//!
//! Models how one large order triggers aftershocks of follow-on orders:
//! `λ(t) = μ + Σ_{tᵢ<t} α·e^(−β(t−tᵢ))`. The branching ratio `α/β` is the
//! expected number of children per event — `< 1` ⇒ stable, `≥ 1` ⇒ explosive.

/// Exponential-kernel Hawkes process.
#[derive(Debug, Clone, Copy)]
pub struct Hawkes {
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Hawkes {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        Self {
            mu: mu.max(0.0),
            alpha: alpha.max(0.0),
            beta: beta.max(1e-9),
        }
    }

    /// Expected children per event. `< 1` ⇒ stationary/stable.
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    pub fn is_stable(&self) -> bool {
        self.branching_ratio() < 1.0
    }

    /// Conditional intensity at time `t` given past event times (`< t`).
    pub fn intensity(&self, t: f64, events: &[f64]) -> f64 {
        let excite: f64 = events
            .iter()
            .filter(|&&ti| ti < t)
            .map(|&ti| self.alpha * (-self.beta * (t - ti)).exp())
            .sum();
        self.mu + excite
    }

    /// Log-likelihood of an ordered event sequence over `[0, horizon]`
    /// (exponential-kernel recursion, O(n)).
    pub fn log_likelihood(&self, events: &[f64], horizon: f64) -> f64 {
        if events.is_empty() {
            return -self.mu * horizon.max(0.0);
        }
        // Compensator: ∫λ = μT + (α/β) Σ (1 − e^{−β(T−tᵢ)})
        let comp_sum: f64 = events
            .iter()
            .map(|&ti| 1.0 - (-self.beta * (horizon - ti)).exp())
            .sum();
        let compensator = self.mu * horizon + (self.alpha / self.beta) * comp_sum;
        // Sum of log-intensities via the recursion A_i = e^{−β Δ}(1 + A_{i−1}).
        let mut a = 0.0;
        let mut log_sum = (self.mu).max(1e-300).ln(); // first event has no history
        for w in events.windows(2) {
            a = (-self.beta * (w[1] - w[0])).exp() * (1.0 + a);
            log_sum += (self.mu + self.alpha * a).max(1e-300).ln();
        }
        log_sum - compensator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intensity_jumps_after_events() {
        let h = Hawkes::new(0.1, 0.8, 1.0);
        let events = [1.0, 2.0];
        let before = h.intensity(0.5, &events); // = mu only
        let after = h.intensity(2.001, &events); // mu + two excitations
        assert!((before - 0.1).abs() < 1e-9);
        assert!(after > before);
    }

    #[test]
    fn branching_ratio_and_stability() {
        assert!(Hawkes::new(0.1, 0.5, 1.0).is_stable());
        assert!(!Hawkes::new(0.1, 1.5, 1.0).is_stable());
    }

    #[test]
    fn log_likelihood_is_finite() {
        let h = Hawkes::new(0.2, 0.5, 1.2);
        let events = [0.3, 0.7, 0.75, 1.5, 3.0];
        let ll = h.log_likelihood(&events, 4.0);
        assert!(ll.is_finite());
    }
}
