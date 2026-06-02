//! Glue between the core [`Regime`] enum and the Markov / Monte-Carlo machinery.
//!
//! These helpers let the rest of the system speak in `Regime`s while the math
//! layer works in state indices.

use sovereign_core::domain::Regime;

use crate::markov::{MarkovChain, TransitionMatrix};
use crate::montecarlo::{RegimeParams, RegimeSwitchingMonteCarlo};
use sovereign_core::error::Result;

/// Estimate a transition matrix over the 8 regimes from an observed sequence.
pub fn transition_from_regimes(regimes: &[Regime], smoothing: f64) -> Result<TransitionMatrix> {
    let ids: Vec<usize> = regimes.iter().map(|r| r.index()).collect();
    TransitionMatrix::from_observations(&ids, Regime::ALL.len(), smoothing)
}

/// Default per-regime daily drift/vol (illustrative; tune from history).
pub fn default_regime_params() -> Vec<RegimeParams> {
    Regime::ALL
        .iter()
        .map(|r| match r {
            Regime::Bull => RegimeParams::new(0.0008, 0.009),
            Regime::Bear => RegimeParams::new(-0.0010, 0.020),
            Regime::Sideways => RegimeParams::new(0.0000, 0.010),
            Regime::Crisis => RegimeParams::new(-0.0030, 0.045),
            Regime::Recovery => RegimeParams::new(0.0012, 0.018),
            Regime::Goldilocks => RegimeParams::new(0.0010, 0.007),
            Regime::Reflation => RegimeParams::new(0.0006, 0.013),
            Regime::Stagflation => RegimeParams::new(-0.0004, 0.022),
        })
        .collect()
}

/// Build a regime-switching Monte-Carlo engine from an observed regime history.
pub fn build_engine(history: &[Regime], smoothing: f64) -> Result<RegimeSwitchingMonteCarlo> {
    let trans = transition_from_regimes(history, smoothing)?;
    let chain = MarkovChain::new(trans);
    RegimeSwitchingMonteCarlo::new(chain, default_regime_params(), 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_engine_from_history() {
        let hist = [
            Regime::Bull,
            Regime::Bull,
            Regime::Sideways,
            Regime::Bear,
            Regime::Recovery,
        ];
        let engine = build_engine(&hist, 1.0).unwrap();
        use rand::{rngs::StdRng, SeedableRng};
        let res = engine.run(
            100.0,
            7,
            5000,
            Regime::Bull.index(),
            0.95,
            &mut StdRng::seed_from_u64(1),
        );
        assert!(res.cvar.is_finite());
        assert_eq!(res.paths, 5000);
    }
}
