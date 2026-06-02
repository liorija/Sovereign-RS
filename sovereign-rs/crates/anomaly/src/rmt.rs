//! Random Matrix Theory — Marchenko-Pastur noise filtering of correlation
//! eigenvalues. Eigenvalues inside the MP band are statistical noise; those
//! above `λ₊` carry genuine cross-sectional signal.

/// Marchenko-Pastur support `[λ₋, λ₊]` for ratio `q = N/T` (assets/observations)
/// and noise std `sigma`.
pub fn marchenko_pastur_bounds(q: f64, sigma: f64) -> (f64, f64) {
    let q = q.max(0.0);
    let s2 = sigma * sigma;
    let lo = s2 * (1.0 - q.sqrt()).powi(2);
    let hi = s2 * (1.0 + q.sqrt()).powi(2);
    (lo.max(0.0), hi)
}

/// Number of eigenvalues that lie **above** the MP upper edge (signal modes).
pub fn signal_count(eigenvalues: &[f64], q: f64, sigma: f64) -> usize {
    let (_, hi) = marchenko_pastur_bounds(q, sigma);
    eigenvalues.iter().filter(|&&l| l > hi).count()
}

/// "Clean" eigenvalues: every eigenvalue at/below the MP edge (noise) is
/// replaced by the average of the noise bulk, preserving the total trace.
/// (Bouchaud-Potters style clipping.)
pub fn clean_eigenvalues(eigenvalues: &[f64], q: f64, sigma: f64) -> Vec<f64> {
    let (_, hi) = marchenko_pastur_bounds(q, sigma);
    let noise: Vec<f64> = eigenvalues.iter().copied().filter(|&l| l <= hi).collect();
    if noise.is_empty() {
        return eigenvalues.to_vec();
    }
    let avg = noise.iter().sum::<f64>() / noise.len() as f64;
    eigenvalues
        .iter()
        .map(|&l| if l <= hi { avg } else { l })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_widen_with_q() {
        let (lo1, hi1) = marchenko_pastur_bounds(0.1, 1.0);
        let (lo2, hi2) = marchenko_pastur_bounds(0.5, 1.0);
        assert!(hi2 > hi1);
        assert!(lo2 < lo1);
    }

    #[test]
    fn separates_signal_from_noise() {
        // Noise eigenvalues near 1, one big "market mode" at 8.
        let eigs = [0.5, 0.8, 1.0, 1.2, 1.4, 8.0];
        let n = signal_count(&eigs, 0.2, 1.0);
        assert_eq!(n, 1, "only the 8.0 mode is signal");
        let cleaned = clean_eigenvalues(&eigs, 0.2, 1.0);
        assert!((cleaned[5] - 8.0).abs() < 1e-9); // signal preserved
        assert!(cleaned[0] > 0.5); // noise lifted to bulk average
    }

    #[test]
    fn trace_is_approximately_preserved() {
        let eigs = [0.5, 0.8, 1.0, 1.2, 1.4, 8.0];
        let before: f64 = eigs.iter().sum();
        let after: f64 = clean_eigenvalues(&eigs, 0.2, 1.0).iter().sum();
        assert!((before - after).abs() < 1e-9);
    }
}
