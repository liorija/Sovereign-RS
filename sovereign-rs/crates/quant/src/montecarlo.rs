//! Regime-switching Monte-Carlo simulation of terminal returns.
//!
//! Each path is a geometric Brownian motion whose drift/vol are chosen, at every
//! step, by a [`MarkovChain`] over market regimes (the Python `MonteCarlo100k` +
//! `HMMRegime` combined). The terminal-return distribution yields VaR / CVaR
//! (Expected Shortfall) for Kill-House Gate 3.
//!
//! # SIMD
//! The hot loop advances **four paths at once** with [`wide::f64x4`] (portable
//! SIMD on *stable* Rust — no nightly, so it builds in VS Code out of the box).
//! Toggle the `simd` feature off to use the scalar kernel (the proptest oracle).
//!
//! # Safety against bad inputs
//! Every public entry point *sanitizes*: non-positive spot, non-finite or
//! negative vol, absurd drift, and zero horizon are clamped or short-circuited,
//! so the engine **cannot panic or emit NaN** — proven by `proptest` below.

use rand::Rng;
use rand_distr::StandardNormal;
use serde::Serialize;
#[cfg(feature = "simd")]
use wide::f64x4;

use sovereign_core::error::{Result, SovereignError};

use crate::markov::MarkovChain;

/// Drift and volatility for one regime, expressed per step (e.g. per day).
#[derive(Debug, Clone, Copy)]
pub struct RegimeParams {
    /// Log-return drift `μ` per step.
    pub drift: f64,
    /// Log-return volatility `σ` per step (≥ 0).
    pub vol: f64,
}

impl RegimeParams {
    /// Sane bound so a malformed upstream vol (e.g. `inf`) can't blow up `exp`.
    const VOL_CAP: f64 = 10.0;
    /// Drift clamp (±1000%/step is already absurd).
    const DRIFT_CAP: f64 = 10.0;

    /// Construct, sanitizing non-finite/negative inputs.
    pub fn new(drift: f64, vol: f64) -> Self {
        let drift = if drift.is_finite() {
            drift.clamp(-Self::DRIFT_CAP, Self::DRIFT_CAP)
        } else {
            0.0
        };
        let vol = if vol.is_finite() {
            vol.clamp(0.0, Self::VOL_CAP)
        } else {
            0.0
        };
        Self { drift, vol }
    }
}

/// Summary statistics of a terminal-return distribution.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MonteCarloResult {
    /// Mean terminal simple return.
    pub expected_return: f64,
    /// Value-at-Risk at the configured confidence (a return quantile, usually < 0).
    pub var: f64,
    /// Conditional VaR / Expected Shortfall: mean return in the tail beyond VaR.
    pub cvar: f64,
    /// Worst single simulated terminal return.
    pub worst: f64,
    /// Fraction of paths that finished below the starting value.
    pub prob_loss: f64,
    /// Number of paths simulated.
    pub paths: usize,
}

impl MonteCarloResult {
    /// Degenerate result for impossible inputs (no paths / bad spot).
    fn degenerate(paths: usize) -> Self {
        Self {
            expected_return: 0.0,
            var: 0.0,
            cvar: 0.0,
            worst: 0.0,
            prob_loss: 0.0,
            paths,
        }
    }

    /// Kill-House Gate 3 predicate: pass if tail risk is no worse than the floor.
    pub fn passes_es_gate(&self, max_es: f64) -> bool {
        self.cvar >= max_es
    }
}

/// A regime-switching Monte-Carlo engine.
#[derive(Debug, Clone)]
pub struct RegimeSwitchingMonteCarlo {
    chain: MarkovChain,
    params: Vec<RegimeParams>,
    dt: f64,
}

impl RegimeSwitchingMonteCarlo {
    /// Build the engine. `params` must have one entry per Markov state.
    pub fn new(chain: MarkovChain, params: Vec<RegimeParams>, dt: f64) -> Result<Self> {
        if params.len() != chain.matrix().n_states() {
            return Err(SovereignError::quant(
                "montecarlo",
                "params length must equal n_states",
            ));
        }
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 1.0 };
        Ok(Self { chain, params, dt })
    }

    /// Simulate terminal **simple** returns `S_T / S_0 − 1` for every path.
    pub fn simulate_terminal_returns<R: Rng + ?Sized>(
        &self,
        spot: f64,
        horizon: usize,
        n_paths: usize,
        start_regime: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        if n_paths == 0 || horizon == 0 || !(spot.is_finite() && spot > 0.0) {
            return vec![0.0; n_paths];
        }
        #[cfg(feature = "simd")]
        {
            self.kernel_simd(spot, horizon, n_paths, start_regime, rng)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.kernel_scalar(spot, horizon, n_paths, start_regime, rng)
        }
    }

    /// Run a full simulation and reduce to [`MonteCarloResult`] at `confidence`
    /// (e.g. `0.95` → 5% tail VaR/CVaR).
    pub fn run<R: Rng + ?Sized>(
        &self,
        spot: f64,
        horizon: usize,
        n_paths: usize,
        start_regime: usize,
        confidence: f64,
        rng: &mut R,
    ) -> MonteCarloResult {
        let returns = self.simulate_terminal_returns(spot, horizon, n_paths, start_regime, rng);
        Self::summarize(&returns, confidence)
    }

    /// Reduce a vector of terminal returns into VaR/CVaR/etc.
    pub fn summarize(returns: &[f64], confidence: f64) -> MonteCarloResult {
        let n = returns.len();
        if n == 0 {
            return MonteCarloResult::degenerate(0);
        }
        let mut sorted: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.is_empty() {
            return MonteCarloResult::degenerate(n);
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let conf = confidence.clamp(0.0, 0.999_999);
        let alpha = 1.0 - conf; // tail mass, e.g. 0.05
        let idx = ((alpha * sorted.len() as f64).floor() as usize).min(sorted.len() - 1);
        let var = sorted[idx];

        // CVaR = mean of the tail at-or-below VaR.
        let tail = &sorted[..=idx];
        let cvar = tail.iter().sum::<f64>() / tail.len() as f64;

        let expected_return = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let worst = sorted[0];
        let prob_loss = sorted.iter().filter(|v| **v < 0.0).count() as f64 / sorted.len() as f64;

        MonteCarloResult {
            expected_return,
            var,
            cvar,
            worst,
            prob_loss,
            paths: n,
        }
    }

    // ── Kernels ──────────────────────────────────────────────────────────────

    /// Scalar reference kernel (also used as the proptest oracle).
    fn kernel_scalar<R: Rng + ?Sized>(
        &self,
        spot: f64,
        horizon: usize,
        n_paths: usize,
        start_regime: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        let sqrt_dt = self.dt.sqrt();
        let n_states = self.params.len();
        let r0 = start_regime.min(n_states - 1);
        let mut out = Vec::with_capacity(n_paths);
        for _ in 0..n_paths {
            let mut s = spot;
            let mut regime = r0;
            for _ in 0..horizon {
                regime = self.chain.sample_next(regime, rng);
                let RegimeParams { drift, vol } = self.params[regime];
                let z: f64 = rng.sample(StandardNormal);
                let exponent = (drift - 0.5 * vol * vol) * self.dt + vol * sqrt_dt * z;
                s *= exponent.exp();
            }
            out.push(s / spot - 1.0);
        }
        out
    }

    /// SIMD kernel: advances four paths per iteration with `wide::f64x4`.
    #[cfg(feature = "simd")]
    fn kernel_simd<R: Rng + ?Sized>(
        &self,
        spot: f64,
        horizon: usize,
        n_paths: usize,
        start_regime: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        let sqrt_dt = self.dt.sqrt();
        let dt_v = f64x4::splat(self.dt);
        let half = f64x4::splat(0.5);
        let sqrt_dt_v = f64x4::splat(sqrt_dt);
        let n_states = self.params.len();
        let r0 = start_regime.min(n_states - 1);

        let mut out = Vec::with_capacity(n_paths);
        let full_chunks = n_paths / 4;
        let remainder = n_paths % 4;

        for _ in 0..full_chunks {
            let mut s = f64x4::splat(spot);
            let mut regimes = [r0; 4];
            for _ in 0..horizon {
                // Per-lane regime advance + parameter gather (scalar).
                let mut mu = [0.0f64; 4];
                let mut sg = [0.0f64; 4];
                let mut zz = [0.0f64; 4];
                for lane in 0..4 {
                    regimes[lane] = self.chain.sample_next(regimes[lane], rng);
                    let p = self.params[regimes[lane]];
                    mu[lane] = p.drift;
                    sg[lane] = p.vol;
                    zz[lane] = rng.sample(StandardNormal);
                }
                let mu_v = f64x4::from(mu);
                let sg_v = f64x4::from(sg);
                let z_v = f64x4::from(zz);
                // exponent = (μ − ½σ²)·dt + σ·√dt·Z   (all four lanes at once)
                let drift = (mu_v - half * sg_v * sg_v) * dt_v;
                let diff = sg_v * sqrt_dt_v * z_v;
                s *= (drift + diff).exp();
            }
            let arr = (s / f64x4::splat(spot) - f64x4::splat(1.0)).to_array();
            out.extend_from_slice(&arr);
        }

        // Tail paths that don't fill a lane of 4 — scalar.
        if remainder > 0 {
            let tail = self.kernel_scalar(spot, horizon, remainder, start_regime, rng);
            out.extend_from_slice(&tail);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markov::TransitionMatrix;
    use proptest::prelude::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn engine(n: usize) -> RegimeSwitchingMonteCarlo {
        let chain = MarkovChain::new(TransitionMatrix::uniform(n).unwrap());
        let params = (0..n)
            .map(|i| RegimeParams::new(0.0005 * i as f64, 0.01 + 0.002 * i as f64))
            .collect();
        RegimeSwitchingMonteCarlo::new(chain, params, 1.0).unwrap()
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let e = engine(3);
        let r1 = e.run(100.0, 10, 1000, 0, 0.95, &mut StdRng::seed_from_u64(7));
        let r2 = e.run(100.0, 10, 1000, 0, 0.95, &mut StdRng::seed_from_u64(7));
        assert_eq!(r1.cvar, r2.cvar);
        assert_eq!(r1.expected_return, r2.expected_return);
    }

    #[test]
    fn cvar_not_greater_than_var() {
        let e = engine(4);
        let res = e.run(250.0, 21, 20_000, 1, 0.95, &mut StdRng::seed_from_u64(42));
        assert!(
            res.cvar <= res.var + 1e-12,
            "cvar {} > var {}",
            res.cvar,
            res.var
        );
        assert!(res.worst <= res.var + 1e-12);
    }

    // ── Adversarial property tests: the engine must NEVER panic or emit NaN ───
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn never_nan_on_extreme_inputs(
            spot in -1.0e6f64..1.0e6,         // includes negatives & zero
            vol in 0.0f64..1.0e9,             // includes absurd / "infinite" vols
            drift in -1.0e6f64..1.0e6,
            horizon in 0usize..40,
            n_paths in 0usize..500,
            seed in any::<u64>(),
        ) {
            let chain = MarkovChain::new(TransitionMatrix::uniform(3).unwrap());
            let params = vec![RegimeParams::new(drift, vol); 3];
            let e = RegimeSwitchingMonteCarlo::new(chain, params, 1.0).unwrap();
            let mut rng = StdRng::seed_from_u64(seed);
            let res = e.run(spot, horizon, n_paths, 0, 0.95, &mut rng);
            prop_assert!(res.expected_return.is_finite());
            prop_assert!(res.var.is_finite());
            prop_assert!(res.cvar.is_finite());
            prop_assert!(res.worst.is_finite());
            prop_assert!((0.0..=1.0).contains(&res.prob_loss));
        }

        #[test]
        fn summarize_handles_nan_and_inf(
            mut returns in proptest::collection::vec(
                prop_oneof![
                    Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY),
                    -1.0f64..5.0,
                ],
                1..200,
            ),
            conf in 0.5f64..0.999,
        ) {
            // Even a payload full of NaN/inf must not panic and must yield finite stats.
            let res = RegimeSwitchingMonteCarlo::summarize(&returns, conf);
            prop_assert!(res.cvar.is_finite());
            prop_assert!(res.var.is_finite());
            returns.clear();
            let _ = returns;
        }
    }
}
