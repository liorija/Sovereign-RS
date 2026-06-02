//! Dependence measures — Kendall's τ and empirical tail dependence (Sklar).
//!
//! Tail dependence answers the crisis question: *given asset A is crashing in
//! its worst `q`-tail, what's the probability B is too?* High lower-tail
//! dependence means the pair will "die together" in a meltdown.

fn ranks_to_uniform(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut u = vec![0.0; n];
    for (rank, &i) in idx.iter().enumerate() {
        u[i] = (rank as f64 + 1.0) / (n as f64 + 1.0); // (0,1)
    }
    u
}

/// Kendall's τ rank correlation in `[-1, 1]` (O(n²)).
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let s = (x[i] - x[j]) * (y[i] - y[j]);
            if s > 0.0 {
                concordant += 1;
            } else if s < 0.0 {
                discordant += 1;
            }
        }
    }
    let total = (concordant + discordant) as f64;
    if total < 1.0 {
        0.0
    } else {
        (concordant - discordant) as f64 / total
    }
}

/// Empirical `(lower, upper)` tail-dependence coefficients at quantile `q`.
/// `lower ≈ P(V ≤ q | U ≤ q)`, `upper ≈ P(V > 1−q | U > 1−q)`.
pub fn tail_dependence(x: &[f64], y: &[f64], q: f64) -> (f64, f64) {
    let n = x.len().min(y.len());
    if n < 5 {
        return (0.0, 0.0);
    }
    let q = q.clamp(0.01, 0.49);
    let u = ranks_to_uniform(&x[..n]);
    let v = ranks_to_uniform(&y[..n]);

    let (mut lo_u, mut lo_joint, mut hi_u, mut hi_joint) = (0u64, 0u64, 0u64, 0u64);
    for i in 0..n {
        if u[i] <= q {
            lo_u += 1;
            if v[i] <= q {
                lo_joint += 1;
            }
        }
        if u[i] > 1.0 - q {
            hi_u += 1;
            if v[i] > 1.0 - q {
                hi_joint += 1;
            }
        }
    }
    let lower = if lo_u == 0 {
        0.0
    } else {
        lo_joint as f64 / lo_u as f64
    };
    let upper = if hi_u == 0 {
        0.0
    } else {
        hi_joint as f64 / hi_u as f64
    };
    (lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comonotonic_has_full_tail_dependence() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let y = x.clone(); // identical → perfect dependence
        assert!((kendall_tau(&x, &y) - 1.0).abs() < 1e-9);
        let (lo, hi) = tail_dependence(&x, &y, 0.1);
        assert!(lo > 0.9 && hi > 0.9);
    }

    #[test]
    fn anti_monotonic_negative_tau() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| -(i as f64)).collect();
        assert!((kendall_tau(&x, &y) + 1.0).abs() < 1e-9);
    }
}
