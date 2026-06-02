//! Gaussian Hidden Markov Model for regime *detection*.
//!
//! Where [`crate::markov`] *generates* regime paths, this *infers* the most
//! likely hidden regime sequence behind an observed return series — the port of
//! the Python `HMMRegime` + Viterbi decoder. All recursions run in **log space**
//! for numerical stability over long histories.

use sovereign_core::error::{Result, SovereignError};

use crate::markov::TransitionMatrix;

const LN_2PI: f64 = 1.837_877_066_409_345_6; // ln(2π)

/// A univariate Gaussian emission for one hidden state.
#[derive(Debug, Clone, Copy)]
pub struct GaussianEmission {
    pub mean: f64,
    pub std: f64,
}

impl GaussianEmission {
    /// Construct, sanitizing a non-positive/NaN std to a tiny floor.
    pub fn new(mean: f64, std: f64) -> Self {
        let mean = if mean.is_finite() { mean } else { 0.0 };
        let std = if std.is_finite() {
            std.abs().max(1e-9)
        } else {
            1e-9
        };
        Self { mean, std }
    }

    /// Log-density `ln N(x; μ, σ)`.
    #[inline]
    fn log_pdf(&self, x: f64) -> f64 {
        let x = if x.is_finite() { x } else { self.mean };
        let z = (x - self.mean) / self.std;
        -0.5 * LN_2PI - self.std.ln() - 0.5 * z * z
    }
}

/// A Gaussian-emission HMM.
#[derive(Debug, Clone)]
pub struct GaussianHmm {
    initial: Vec<f64>,
    trans: TransitionMatrix,
    emit: Vec<GaussianEmission>,
}

impl GaussianHmm {
    /// Build an HMM, validating that all three dimensions agree.
    pub fn new(
        initial: Vec<f64>,
        trans: TransitionMatrix,
        emit: Vec<GaussianEmission>,
    ) -> Result<Self> {
        let n = trans.n_states();
        if initial.len() != n || emit.len() != n {
            return Err(SovereignError::quant(
                "hmm",
                "initial/emit length must equal n_states",
            ));
        }
        // Normalize initial distribution defensively.
        let s: f64 = initial.iter().filter(|v| v.is_finite() && **v >= 0.0).sum();
        let initial = if s > 0.0 {
            initial
                .iter()
                .map(|v| {
                    if v.is_finite() && *v >= 0.0 {
                        v / s
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            vec![1.0 / n as f64; n]
        };
        Ok(Self {
            initial,
            trans,
            emit,
        })
    }

    /// Number of hidden states.
    pub fn n_states(&self) -> usize {
        self.emit.len()
    }

    #[inline]
    fn ln_trans(&self, i: usize, j: usize) -> f64 {
        // Floor probabilities so ln() never returns -inf and poisons the max.
        self.trans.get(i, j).max(1e-300).ln()
    }

    /// **Viterbi**: most likely hidden-state path for `obs`.
    /// Returns one state id per observation; never panics (obs are sanitized).
    pub fn viterbi(&self, obs: &[f64]) -> Vec<usize> {
        let n = self.n_states();
        if obs.is_empty() || n == 0 {
            return Vec::new();
        }
        let t = obs.len();
        let mut delta = vec![f64::NEG_INFINITY; n]; // best log-prob ending in state s
        let mut psi = vec![vec![0usize; n]; t]; // backpointers

        // Initialization (t = 0).
        for (s, d) in delta.iter_mut().enumerate() {
            *d = self.initial[s].max(1e-300).ln() + self.emit[s].log_pdf(obs[0]);
        }

        // Recursion.
        let mut next = vec![f64::NEG_INFINITY; n];
        for (ti, &o) in obs.iter().enumerate().skip(1) {
            for s in 0..n {
                let mut best = f64::NEG_INFINITY;
                let mut best_prev = 0;
                for (sp, &dprev) in delta.iter().enumerate() {
                    let val = dprev + self.ln_trans(sp, s);
                    if val > best {
                        best = val;
                        best_prev = sp;
                    }
                }
                next[s] = best + self.emit[s].log_pdf(o);
                psi[ti][s] = best_prev;
            }
            std::mem::swap(&mut delta, &mut next);
        }

        // Termination + backtrack.
        let mut last = 0;
        let mut best = f64::NEG_INFINITY;
        for (s, &d) in delta.iter().enumerate() {
            if d > best {
                best = d;
                last = s;
            }
        }
        let mut path = vec![0usize; t];
        path[t - 1] = last;
        for ti in (1..t).rev() {
            path[ti - 1] = psi[ti][path[ti]];
        }
        path
    }

    /// Forward log-likelihood `ln P(obs | model)` via the scaled forward pass.
    pub fn log_likelihood(&self, obs: &[f64]) -> f64 {
        let n = self.n_states();
        if obs.is_empty() || n == 0 {
            return 0.0;
        }
        // alpha in log space using log-sum-exp.
        let mut alpha: Vec<f64> = (0..n)
            .map(|s| self.initial[s].max(1e-300).ln() + self.emit[s].log_pdf(obs[0]))
            .collect();

        for &o in obs.iter().skip(1) {
            let mut next = vec![0.0f64; n];
            for (s, ns) in next.iter_mut().enumerate() {
                let terms: Vec<f64> = (0..n).map(|sp| alpha[sp] + self.ln_trans(sp, s)).collect();
                *ns = log_sum_exp(&terms) + self.emit[s].log_pdf(o);
            }
            alpha = next;
        }
        log_sum_exp(&alpha)
    }
}

/// Numerically stable `ln(Σ exp(x_i))`.
fn log_sum_exp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let sum: f64 = xs.iter().map(|x| (x - m).exp()).sum();
    m + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A 2-state HMM: state 0 = "bear" (neg mean), state 1 = "bull" (pos mean).
    fn two_state() -> GaussianHmm {
        let trans = TransitionMatrix::from_rows(vec![vec![0.9, 0.1], vec![0.1, 0.9]]).unwrap();
        let emit = vec![
            GaussianEmission::new(-0.02, 0.01),
            GaussianEmission::new(0.02, 0.01),
        ];
        GaussianHmm::new(vec![0.5, 0.5], trans, emit).unwrap()
    }

    #[test]
    fn viterbi_recovers_obvious_regimes() {
        let hmm = two_state();
        // Clearly bearish then clearly bullish observations.
        let obs = [-0.02, -0.021, -0.019, 0.02, 0.021, 0.019];
        let path = hmm.viterbi(&obs);
        assert_eq!(path.len(), obs.len());
        assert_eq!(path[0], 0); // bear
        assert_eq!(*path.last().unwrap(), 1); // bull
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn viterbi_never_panics_and_stays_in_range(
            obs in proptest::collection::vec(
                prop_oneof![Just(f64::NAN), Just(f64::INFINITY), -1.0f64..1.0],
                0..200,
            ),
        ) {
            let hmm = two_state();
            let path = hmm.viterbi(&obs);
            prop_assert_eq!(path.len(), obs.len());
            prop_assert!(path.iter().all(|s| *s < hmm.n_states()));
        }

        #[test]
        fn log_likelihood_is_finite_or_neg_inf(
            obs in proptest::collection::vec(-0.5f64..0.5, 1..100),
        ) {
            let hmm = two_state();
            let ll = hmm.log_likelihood(&obs);
            prop_assert!(ll.is_finite() || ll == f64::NEG_INFINITY);
        }
    }
}
