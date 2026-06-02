//! Largest Lyapunov exponent (simplified Rosenstein) — how fast nearby price
//! trajectories diverge. A positive exponent means sensitive dependence
//! (chaos): the horizon beyond which prediction is futile and the AI should
//! defer to the Axiom Breaker.

/// Estimate the largest Lyapunov exponent of a scalar series via 1-step nearest-
/// neighbour divergence in an `emb_dim`-delay embedding. Returns the mean log
/// divergence rate; `> 0` ⇒ chaotic expansion, `≤ 0` ⇒ stable/periodic.
pub fn largest_lyapunov(series: &[f64], emb_dim: usize, lag: usize) -> f64 {
    let m = emb_dim.max(2);
    let lag = lag.max(1);
    let span = (m - 1) * lag;
    let n = series.len();
    if n < span + 3 {
        return 0.0;
    }
    // Embedding vectors X_i (one fewer at the end so X_{i+1} exists).
    let count = n - span - 1;
    if count < 3 {
        return 0.0;
    }
    let embed = |i: usize| -> Vec<f64> { (0..m).map(|k| series[i + k * lag]).collect() };
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };
    // Minimum temporal separation so neighbours aren't trivially adjacent.
    let theiler = lag.max(1);

    let mut log_sum = 0.0;
    let mut used = 0u64;
    for i in 0..count {
        let xi = embed(i);
        // nearest neighbour j (Euclidean), excluding the Theiler window.
        let mut best_j = usize::MAX;
        let mut best_d = f64::INFINITY;
        for j in 0..count {
            if (i as isize - j as isize).unsigned_abs() <= theiler {
                continue;
            }
            let d = dist(&xi, &embed(j));
            if d > 0.0 && d < best_d {
                best_d = d;
                best_j = j;
            }
        }
        if best_j == usize::MAX || !best_d.is_finite() || best_d <= 0.0 {
            continue;
        }
        // Divergence one step later.
        let d1 = dist(&embed(i + 1), &embed(best_j + 1));
        if d1 > 0.0 && d1.is_finite() {
            log_sum += (d1 / best_d).ln();
            used += 1;
        }
    }
    if used == 0 {
        0.0
    } else {
        log_sum / used as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_is_not_strongly_chaotic() {
        let sig: Vec<f64> = (0..400).map(|i| (i as f64 * 0.2).sin()).collect();
        let l = largest_lyapunov(&sig, 3, 1);
        assert!(l.is_finite());
        assert!(l < 0.5, "periodic should not be strongly positive, got {l}");
    }

    #[test]
    fn logistic_map_is_chaotic() {
        // x_{n+1} = 4 x (1-x): classic chaos, true λ = ln 2 ≈ 0.693.
        let mut x = 0.2;
        let mut sig = Vec::with_capacity(500);
        for _ in 0..500 {
            x = 4.0 * x * (1.0 - x);
            sig.push(x);
        }
        let l = largest_lyapunov(&sig, 3, 1);
        assert!(
            l > 0.0,
            "logistic map should have a positive exponent, got {l}"
        );
    }
}
