//! Kelly criterion — optimal growth-maximizing bet fraction.

/// Discrete Kelly: `f* = p − (1−p)/b` for win prob `p` and win/loss payoff `b`.
/// Clamped to `[0, 1]` (long-only, no leverage).
pub fn kelly_fraction(win_prob: f64, win_loss_ratio: f64) -> f64 {
    let p = win_prob.clamp(0.0, 1.0);
    if !(win_loss_ratio.is_finite() && win_loss_ratio > 0.0) {
        return 0.0;
    }
    (p - (1.0 - p) / win_loss_ratio).clamp(0.0, 1.0)
}

/// Fractional Kelly (e.g. half-Kelly) — the practitioner's variance-tamed bet.
pub fn fractional_kelly(full_kelly: f64, fraction: f64) -> f64 {
    (full_kelly.max(0.0) * fraction.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Continuous Kelly from a return series: `f* ≈ μ / σ²`, clamped to `[0, 1]`.
pub fn kelly_from_returns(returns: &[f64]) -> f64 {
    let clean: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.len() < 2 {
        return 0.0;
    }
    let mean = clean.iter().sum::<f64>() / clean.len() as f64;
    let var = clean.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / clean.len() as f64;
    if var < 1e-12 {
        return 0.0;
    }
    (mean / var).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn even_money_edge() {
        // p=0.6, b=1 → 0.6 - 0.4 = 0.2
        assert!((kelly_fraction(0.6, 1.0) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn no_edge_is_zero() {
        assert_eq!(kelly_fraction(0.5, 1.0), 0.0);
        assert_eq!(kelly_fraction(0.3, 1.0), 0.0); // negative → clamped
    }

    #[test]
    fn half_kelly_halves() {
        assert!((fractional_kelly(0.4, 0.5) - 0.2).abs() < 1e-12);
    }

    proptest! {
        #[test]
        fn fraction_always_unit_interval(p in 0.0f64..1.0, b in 0.01f64..10.0) {
            let f = kelly_fraction(p, b);
            prop_assert!((0.0..=1.0).contains(&f));
        }
    }
}
