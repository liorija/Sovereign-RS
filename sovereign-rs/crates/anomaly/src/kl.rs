//! Kullback-Leibler & Jensen-Shannon divergence — how far the model's predicted
//! distribution has drifted from the realized market distribution.

const FLOOR: f64 = 1e-12;

fn normalize(p: &[f64]) -> Vec<f64> {
    let clean: Vec<f64> = p
        .iter()
        .map(|v| if v.is_finite() && *v > 0.0 { *v } else { 0.0 })
        .collect();
    let sum: f64 = clean.iter().sum();
    if sum <= 0.0 {
        let n = clean.len().max(1);
        return vec![1.0 / n as f64; clean.len()];
    }
    clean.iter().map(|v| v / sum).collect()
}

/// `D_KL(p ‖ q) = Σ pᵢ ln(pᵢ/qᵢ)` (nats), `≥ 0`, `0` iff equal. Inputs are
/// normalized internally; `q` is floored to avoid division by zero.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len().min(q.len());
    if n == 0 {
        return 0.0;
    }
    let pn = normalize(&p[..n]);
    let qn = normalize(&q[..n]);
    let mut d = 0.0;
    for i in 0..n {
        if pn[i] > 0.0 {
            d += pn[i] * (pn[i] / qn[i].max(FLOOR)).ln();
        }
    }
    d.max(0.0)
}

/// Symmetric Jensen-Shannon divergence in `[0, ln 2]`.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len().min(q.len());
    if n == 0 {
        return 0.0;
    }
    let pn = normalize(&p[..n]);
    let qn = normalize(&q[..n]);
    let m: Vec<f64> = (0..n).map(|i| 0.5 * (pn[i] + qn[i])).collect();
    0.5 * kl_divergence(&pn, &m) + 0.5 * kl_divergence(&qn, &m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn identical_is_zero() {
        let p = [0.2, 0.3, 0.5];
        assert!(kl_divergence(&p, &p) < 1e-12);
        assert!(js_divergence(&p, &p) < 1e-12);
    }

    #[test]
    fn divergence_is_positive_for_different() {
        let p = [0.9, 0.05, 0.05];
        let q = [0.1, 0.45, 0.45];
        assert!(kl_divergence(&p, &q) > 0.0);
        assert!(js_divergence(&p, &q) > 0.0);
    }

    proptest! {
        #[test]
        fn kl_nonnegative(
            p in proptest::collection::vec(0.0f64..10.0, 2..8),
            q in proptest::collection::vec(0.0f64..10.0, 2..8),
        ) {
            prop_assert!(kl_divergence(&p, &q) >= -1e-9);
            let js = js_divergence(&p, &q);
            prop_assert!((0.0..=0.7).contains(&js)); // ln 2 ≈ 0.693
        }
    }
}
