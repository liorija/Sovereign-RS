//! Discrete-time Markov chains over market regimes.
//!
//! This is the typed, validated replacement for the ad-hoc transition logic in
//! the Python `HMMRegime` / `RegimeMapper`. A [`TransitionMatrix`] is guaranteed
//! by construction to be square, finite, non-negative and row-stochastic, so
//! downstream code (the regime-switching Monte Carlo) can never be fed a
//! malformed matrix.
//!
//! Numerical methods used:
//! * **MLE estimation** from an observed regime sequence (transition counting
//!   with optional Laplace smoothing),
//! * **n-step** matrices via exponentiation-by-squaring,
//! * **stationary distribution** via power iteration.

use ndarray::{Array1, Array2};
use rand::Rng;

use sovereign_core::error::{Result, SovereignError};

/// A row-stochastic transition matrix `P` where `P[i][j] = Pr(next = j | cur = i)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionMatrix {
    p: Array2<f64>,
}

impl TransitionMatrix {
    /// Numerical tolerance for the "rows sum to 1" invariant.
    pub const ROW_SUM_TOL: f64 = 1e-9;

    /// Build from a raw matrix, validating the stochastic invariants.
    pub fn from_array(p: Array2<f64>) -> Result<Self> {
        let (r, c) = p.dim();
        if r == 0 || r != c {
            return Err(SovereignError::quant(
                "markov",
                "transition matrix must be square, non-empty",
            ));
        }
        for v in p.iter() {
            if !v.is_finite() || *v < 0.0 {
                return Err(SovereignError::quant(
                    "markov",
                    "entries must be finite and non-negative",
                ));
            }
        }
        for (i, row) in p.rows().into_iter().enumerate() {
            let s: f64 = row.sum();
            if (s - 1.0).abs() > Self::ROW_SUM_TOL {
                return Err(SovereignError::quant(
                    "markov",
                    format!("row {i} sums to {s}, expected 1.0"),
                ));
            }
        }
        Ok(Self { p })
    }

    /// Build from nested rows, normalizing each row defensively.
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self> {
        let n = rows.len();
        if n == 0 || rows.iter().any(|r| r.len() != n) {
            return Err(SovereignError::quant(
                "markov",
                "rows must form a non-empty square matrix",
            ));
        }
        let mut p = Array2::<f64>::zeros((n, n));
        for (i, row) in rows.into_iter().enumerate() {
            let s: f64 = row.iter().filter(|v| v.is_finite() && **v >= 0.0).sum();
            for (j, v) in row.into_iter().enumerate() {
                let v = if v.is_finite() && v >= 0.0 { v } else { 0.0 };
                p[[i, j]] = if s > 0.0 { v / s } else { 1.0 / n as f64 };
            }
        }
        Self::from_array(p)
    }

    /// The all-equal `1/n` matrix.
    pub fn uniform(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(SovereignError::quant("markov", "n must be > 0"));
        }
        Ok(Self {
            p: Array2::from_elem((n, n), 1.0 / n as f64),
        })
    }

    /// The identity (absorbing) matrix.
    pub fn identity(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(SovereignError::quant("markov", "n must be > 0"));
        }
        Ok(Self { p: Array2::eye(n) })
    }

    /// **Maximum-likelihood estimate** from an observed state sequence.
    ///
    /// `states` are state ids in `0..n_states`; out-of-range ids are skipped.
    /// `smoothing` is a Laplace pseudo-count added to every transition (use a
    /// small value like `1.0` to avoid zero rows for unobserved source states).
    pub fn from_observations(states: &[usize], n_states: usize, smoothing: f64) -> Result<Self> {
        if n_states == 0 {
            return Err(SovereignError::quant("markov", "n_states must be > 0"));
        }
        let smoothing = smoothing.max(0.0);
        let mut counts = Array2::<f64>::from_elem((n_states, n_states), smoothing);
        for w in states.windows(2) {
            let (i, j) = (w[0], w[1]);
            if i < n_states && j < n_states {
                counts[[i, j]] += 1.0;
            }
        }
        // Normalize rows; a row whose total is 0 (only possible when smoothing==0
        // and the state was never a source) becomes uniform.
        let mut p = Array2::<f64>::zeros((n_states, n_states));
        for i in 0..n_states {
            let total: f64 = counts.row(i).sum();
            for j in 0..n_states {
                p[[i, j]] = if total > 0.0 {
                    counts[[i, j]] / total
                } else {
                    1.0 / n_states as f64
                };
            }
        }
        Self::from_array(p)
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.p.nrows()
    }

    /// Borrow the underlying matrix.
    pub fn as_array(&self) -> &Array2<f64> {
        &self.p
    }

    /// `Pr(next = j | cur = i)`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.p[[i, j]]
    }

    /// Matrix product of two transition matrices (still stochastic).
    pub fn matmul(&self, other: &TransitionMatrix) -> Result<TransitionMatrix> {
        if self.n_states() != other.n_states() {
            return Err(SovereignError::quant(
                "markov",
                "dimension mismatch in matmul",
            ));
        }
        // Re-validate: floating error can drift row sums; from_array tolerates ROW_SUM_TOL.
        Self::from_array(self.p.dot(&other.p))
    }

    /// The `k`-step transition matrix `P^k` via exponentiation by squaring.
    /// `power(0)` is the identity.
    pub fn power(&self, k: usize) -> Result<TransitionMatrix> {
        let n = self.n_states();
        let mut result = Self::identity(n)?;
        if k == 0 {
            return Ok(result);
        }
        let mut base = self.clone();
        let mut e = k;
        while e > 0 {
            if e & 1 == 1 {
                result = result.matmul(&base)?;
            }
            e >>= 1;
            if e > 0 {
                base = base.matmul(&base)?;
            }
        }
        Ok(result)
    }

    /// The stationary distribution `π` such that `π P = π`, via power iteration.
    /// Returns a probability vector (non-negative, sums to 1).
    pub fn stationary_distribution(&self, max_iter: usize, tol: f64) -> Vec<f64> {
        let n = self.n_states();
        let mut pi = Array1::from_elem(n, 1.0 / n as f64);
        for _ in 0..max_iter.max(1) {
            let next = pi.dot(&self.p);
            let diff: f64 = (&next - &pi).iter().map(|v| v.abs()).sum();
            pi = next;
            // Renormalize to fight floating drift.
            let s: f64 = pi.sum();
            if s > 0.0 {
                pi.mapv_inplace(|v| v / s);
            }
            if diff < tol {
                break;
            }
        }
        pi.to_vec()
    }

    /// Whether every row sums to 1 within `tol`.
    pub fn is_stochastic(&self, tol: f64) -> bool {
        self.p
            .rows()
            .into_iter()
            .all(|row| (row.sum() - 1.0).abs() <= tol)
    }
}

/// A Markov chain you can sample paths from.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    matrix: TransitionMatrix,
}

impl MarkovChain {
    /// Wrap a validated transition matrix.
    pub fn new(matrix: TransitionMatrix) -> Self {
        Self { matrix }
    }

    /// Borrow the transition matrix.
    pub fn matrix(&self) -> &TransitionMatrix {
        &self.matrix
    }

    /// Sample the next state given the current one, via inverse-CDF on the row.
    /// Out-of-range `state` saturates to the last state (never panics).
    pub fn sample_next<R: Rng + ?Sized>(&self, state: usize, rng: &mut R) -> usize {
        let n = self.matrix.n_states();
        let i = state.min(n - 1);
        let u: f64 = rng.gen::<f64>();
        let mut acc = 0.0;
        for j in 0..n {
            acc += self.matrix.get(i, j);
            if u <= acc {
                return j;
            }
        }
        n - 1 // floating residue: fall through to last state
    }

    /// Generate a path of `steps` states starting from `start`.
    pub fn simulate<R: Rng + ?Sized>(&self, start: usize, steps: usize, rng: &mut R) -> Vec<usize> {
        let mut path = Vec::with_capacity(steps + 1);
        let mut s = start.min(self.matrix.n_states() - 1);
        path.push(s);
        for _ in 0..steps {
            s = self.sample_next(s, rng);
            path.push(s);
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn from_observations_is_stochastic() {
        let seq = [0usize, 1, 1, 2, 0, 1, 2, 2, 0];
        let tm = TransitionMatrix::from_observations(&seq, 3, 1.0).unwrap();
        assert!(tm.is_stochastic(TransitionMatrix::ROW_SUM_TOL));
    }

    #[test]
    fn power_zero_is_identity() {
        let tm = TransitionMatrix::uniform(4).unwrap();
        let id = tm.power(0).unwrap();
        assert_eq!(id, TransitionMatrix::identity(4).unwrap());
    }

    #[test]
    fn stationary_of_uniform_is_uniform() {
        let tm = TransitionMatrix::uniform(5).unwrap();
        let pi = tm.stationary_distribution(1000, 1e-12);
        for v in pi {
            assert!((v - 0.2).abs() < 1e-9);
        }
    }

    // ── Property-based: invariants must hold for ANY observed sequence ────────
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        #[test]
        fn estimated_matrix_always_stochastic(
            seq in proptest::collection::vec(0usize..6, 0..200),
            n in 1usize..6,
            smoothing in 0.0f64..3.0,
        ) {
            let tm = TransitionMatrix::from_observations(&seq, n, smoothing).unwrap();
            prop_assert!(tm.is_stochastic(1e-6));
            // every entry is a valid probability
            for v in tm.as_array().iter() {
                prop_assert!(v.is_finite() && *v >= 0.0 && *v <= 1.0 + 1e-9);
            }
        }

        #[test]
        fn n_step_preserves_stochasticity(
            seq in proptest::collection::vec(0usize..5, 2..150),
            k in 0usize..8,
        ) {
            let tm = TransitionMatrix::from_observations(&seq, 5, 1.0).unwrap();
            let pk = tm.power(k).unwrap();
            prop_assert!(pk.is_stochastic(1e-6));
        }

        #[test]
        fn stationary_is_a_probability_vector(
            seq in proptest::collection::vec(0usize..5, 5..150),
        ) {
            let tm = TransitionMatrix::from_observations(&seq, 5, 0.5).unwrap();
            let pi = tm.stationary_distribution(2000, 1e-12);
            let s: f64 = pi.iter().sum();
            prop_assert!((s - 1.0).abs() < 1e-6);
            prop_assert!(pi.iter().all(|v| *v >= -1e-12));
        }

        #[test]
        fn simulated_paths_stay_in_range(
            n in 1usize..6,
            start in 0usize..10,
            steps in 0usize..50,
            seed in any::<u64>(),
        ) {
            let chain = MarkovChain::new(TransitionMatrix::uniform(n).unwrap());
            let mut rng = StdRng::seed_from_u64(seed);
            let path = chain.simulate(start, steps, &mut rng);
            prop_assert_eq!(path.len(), steps + 1);
            prop_assert!(path.iter().all(|s| *s < n));
        }
    }
}
