//! Conditional Value-at-Risk / Expected Shortfall (port of
//! `v325_upgrades.CVaRRiskManager`).
//!
//! Complements the Monte-Carlo CVaR in [`crate::montecarlo`] with closed-form /
//! historical estimates used for fast position sizing.

/// Historical (non-parametric) CVaR of a return series at `confidence`.
/// Returns a negative number = expected loss in the tail.
pub fn historical_cvar(returns: &[f64], confidence: f64) -> f64 {
    let mut r: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
    if r.len() < 5 {
        return -0.05; // conservative default
    }
    r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = (1.0 - confidence).clamp(1e-6, 1.0);
    let idx = ((alpha * r.len() as f64).floor() as usize).min(r.len() - 1);
    let tail = &r[..=idx];
    tail.iter().sum::<f64>() / tail.len() as f64
}

/// Parametric (Gaussian) CVaR: `μ − σ·φ(zα)/α`.
pub fn parametric_cvar(returns: &[f64], confidence: f64) -> f64 {
    let clean: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.is_empty() {
        return -0.05;
    }
    let n = clean.len() as f64;
    let mu = clean.iter().sum::<f64>() / n;
    let var = clean.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / n;
    let sigma = var.sqrt() + 1e-12;
    let alpha = (1.0 - confidence).clamp(1e-6, 0.999_999);
    let z = inv_norm_cdf(alpha);
    let phi = std_normal_pdf(z);
    mu - sigma * phi / alpha
}

/// Scale a budget so the expected tail loss stays within `max_loss·equity`.
/// (Port of `position_size_from_cvar`.)
pub fn position_size_from_cvar(budget: f64, cvar: f64, max_loss: f64, equity: f64) -> f64 {
    if cvar >= 0.0 || equity <= 0.0 {
        return budget;
    }
    let cvar_limit = (equity * max_loss) / cvar.abs();
    budget.min(cvar_limit).max(0.0)
}

/// Standard normal PDF.
fn std_normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Inverse standard-normal CDF (Acklam's rational approximation).
fn inv_norm_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn cvar_is_negative_for_risky_series() {
        let rets: Vec<f64> = (0..300)
            .map(|i| ((i * 7 % 13) as f64 - 6.0) * 0.01)
            .collect();
        assert!(historical_cvar(&rets, 0.95) < 0.0);
    }

    #[test]
    fn sizing_caps_budget() {
        // CVaR -10%, max loss 2% of $10k = $200 → budget capped at 200/0.1 = 2000
        let sized = position_size_from_cvar(5000.0, -0.10, 0.02, 10_000.0);
        assert!((sized - 2000.0).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn cvar_never_panics(
            rets in proptest::collection::vec(-1.0f64..1.0, 0..300),
            conf in 0.5f64..0.999,
        ) {
            prop_assert!(historical_cvar(&rets, conf).is_finite());
            prop_assert!(parametric_cvar(&rets, conf).is_finite());
        }
    }
}
