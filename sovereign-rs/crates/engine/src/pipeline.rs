//! The per-candidate decision pipeline: Kill-House gates **and** BFT consensus.
//!
//! The Kill-House now runs the 6-stage [`HyperLayeredVpinGate`] over an 11-D
//! [`StringTensor`] built from the candidate's data, calibrated by an **Adaptive
//! Capital Protocol**: micro capital tightens the dead-zone (hyper-aggressive),
//! massive capital relaxes it and flags order-splitting to protect market topology.

use sovereign_core::domain::Regime;
use sovereign_core::error::Result;
use sovereign_quant::regime::build_engine;
use sovereign_risk::gate::{
    DayOfBounds, GateContext, KillHouse, StringTensor, TernaryState, Verdict, TENSOR_DIMS,
};
use sovereign_risk::gates::{ConvictionGate, HyperLayeredVpinGate, MonteCarloGate, VpinGate};
use sovereign_signals::agent::SignalContext;
use sovereign_signals::consensus::{BftConsensus, ConsensusResult};

use crate::agents::default_panel;

/// Reference capital ($) at which the capital-mass multiplier is ~1.0.
const CAPITAL_PIVOT: f64 = 100_000.0;
/// Above this notional ($) the engine breaks orders down to protect topology.
const CAPITAL_SPLIT_THRESHOLD: f64 = 1_000_000.0;
/// Session entropy ceiling (axiom-breaker) for the cascade.
const MAX_ENTROPY: f64 = 0.97;

/// The combined verdict for one candidate.
#[derive(Debug)]
pub struct Decision {
    pub ticker: String,
    /// Approved only if the Kill-House *and* BFT consensus both pass.
    pub approved: bool,
    pub killhouse: Verdict,
    pub consensus: ConsensusResult,
    /// Net cascade direction across the gates (Short / Flat / Long).
    pub cascade_state: TernaryState,
    /// The adaptive capital-mass multiplier used this evaluation.
    pub capital_mass: f64,
    /// Whether the order should be sliced (massive capital → protect topology).
    pub split_orders: bool,
}

/// A reusable evaluation pipeline.
pub struct Pipeline {
    killhouse: KillHouse,
    consensus: BftConsensus,
    capital_size: f64,
}

impl Pipeline {
    /// Construct from the parts (defaults to pivot capital).
    pub fn new(killhouse: KillHouse, consensus: BftConsensus) -> Self {
        Self {
            killhouse,
            consensus,
            capital_size: CAPITAL_PIVOT,
        }
    }

    /// Inject the deployable capital size ($) that drives the Adaptive Capital Protocol.
    pub fn with_capital_size(mut self, capital_size: f64) -> Self {
        self.capital_size = if capital_size.is_finite() && capital_size > 0.0 {
            capital_size
        } else {
            CAPITAL_PIVOT
        };
        self
    }

    /// Build the default pipeline: three gates (conviction, Monte-Carlo CVaR,
    /// **HyperLayered cascade**) plus the 4-agent BFT panel, with the Monte-Carlo
    /// engine estimated from an observed regime `history`.
    pub fn default_from_history(history: &[Regime]) -> Result<Self> {
        let mc_engine = build_engine(history, 1.0)?;
        let killhouse = KillHouse::new(vec![
            Box::new(ConvictionGate { min: 8.0 }),
            Box::new(MonteCarloGate {
                engine: mc_engine,
                max_es: -0.08,
                horizon: 7,
                paths: 20_000,
                confidence: 0.95,
                seed: 0xC0FFEE,
            }),
            Box::new(HyperLayeredVpinGate),
        ]);
        Ok(Self {
            killhouse,
            consensus: BftConsensus::new(default_panel()),
            capital_size: CAPITAL_PIVOT,
        })
    }

    /// Adaptive Capital Protocol: map deployable capital to a mass multiplier.
    /// Micro (≪ pivot) ⇒ `< 1` (tightens the dead-zone → aggressive);
    /// massive (≫ pivot) ⇒ `> 1` (relaxes it → conservative + order-splitting).
    fn capital_mass(&self) -> f64 {
        (self.capital_size / CAPITAL_PIVOT)
            .powf(0.25)
            .clamp(0.25, 4.0)
    }

    /// Evaluate one candidate end-to-end.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        ticker: &str,
        regime: Regime,
        closes: Vec<f64>,
        volumes: Vec<f64>,
        conviction: f64,
        vix_score: f64,
        heat: f64,
        is_small_tier: bool,
        is_core: bool,
    ) -> Decision {
        let capital_mass = self.capital_mass();
        let split_orders = self.capital_size > CAPITAL_SPLIT_THRESHOLD;

        // Build the 11-D state tensor from the candidate data (borrows the slices
        // before they're moved into the GateContext — zero extra copies).
        let tensor = build_tensor(&closes, &volumes, conviction, vix_score, regime);

        let gate_ctx = GateContext {
            ticker: ticker.to_string(),
            regime,
            closes,
            volumes,
            conviction,
            tensor,
            capital_mass,
        };
        let killhouse = self.killhouse.run(&gate_ctx);
        let cascade_state = killhouse.net_cascade();

        let sig_ctx = SignalContext {
            ticker: ticker.to_string(),
            regime,
            conviction,
            vix_score,
            heat,
        };
        let consensus = self.consensus.evaluate(&sig_ctx, is_small_tier, is_core);

        Decision {
            ticker: ticker.to_string(),
            approved: killhouse.approved && consensus.approved,
            killhouse,
            consensus,
            cascade_state,
            capital_mass,
            split_orders,
        }
    }
}

// ── tensor construction helpers (pure, adaptive, panic-free) ────────────────

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    (xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Binary (direction) entropy in `[0,1]` from the fraction of up-moves.
fn direction_entropy(rets: &[f64]) -> f64 {
    if rets.is_empty() {
        return 0.0;
    }
    let up = rets.iter().filter(|r| **r > 0.0).count() as f64 / rets.len() as f64;
    if up <= 0.0 || up >= 1.0 {
        0.0
    } else {
        -(up * up.log2() + (1.0 - up) * (1.0 - up).log2())
    }
}

/// Assemble the 11-D [`StringTensor`] with adaptive [`DayOfBounds`].
fn build_tensor(
    closes: &[f64],
    volumes: &[f64],
    conviction: f64,
    vix_score: f64,
    regime: Regime,
) -> StringTensor {
    let rets: Vec<f64> = closes
        .windows(2)
        .map(|w| if w[0] != 0.0 { w[1] / w[0] - 1.0 } else { 0.0 })
        .collect();

    let price_mom = if rets.len() >= 5 {
        mean(&rets[rets.len() - 5..])
    } else {
        mean(&rets)
    };
    let realized_vol = stddev(&rets);
    let mean_vol = mean(volumes);
    let last_vol = volumes.last().copied().unwrap_or(mean_vol);
    let vol_ratio = if mean_vol > 0.0 {
        last_vol / mean_vol
    } else {
        1.0
    };
    let vpin = VpinGate::compute(closes, volumes, 50); // [0,1] order-flow imbalance
    let dir = ((price_mom > 0.0) as i32 - (price_mom < 0.0) as i32) as f64;
    let net_entropy = direction_entropy(&rets);

    // Canonical Dim order: Price, Volume, Time, Spread, Order-Flow, Sentiment,
    // Liquidity, Volatility, Network-Entropy, Regime-Vector, Cointegration-Delta.
    let values: [f64; TENSOR_DIMS] = [
        (price_mom * 50.0).tanh(),                 // Price (directional, [-1,1])
        (vol_ratio / 2.0).clamp(0.0, 1.0),         // Volume
        0.5,                                       // Time (no intraday clock here)
        0.1,                                       // Spread (no L2 book → small const)
        vpin * dir,                                // Order-Flow (signed imbalance)
        ((vix_score + 1.0) / 2.0).clamp(0.0, 1.0), // Sentiment
        1.0 / (1.0 + realized_vol * 50.0),         // Liquidity (inverse of vol)
        (realized_vol * 50.0).clamp(0.0, 1.0),     // Volatility
        net_entropy,                               // Network-Entropy
        regime.allocation_multiplier(),            // Regime-Vector
        (conviction / 20.0).tanh(),                // Cointegration-Delta (proxy)
    ];

    let bounds = DayOfBounds::new(
        MAX_ENTROPY,
        (realized_vol * 50.0).clamp(0.05, 1.0), // adaptive dead-zone = today's vol
        (conviction / 20.0).clamp(0.0, 1.0),    // adaptive fitness rank
    );
    StringTensor::new(values, bounds)
}

// ═══════════════════════════════════════════════════════════════════════════
//  CPU core pinning & thread isolation (Guerilla Protocol)
// ═══════════════════════════════════════════════════════════════════════════

/// Outcome of an attempt to pin the current thread to an isolated core.
#[derive(Debug, Clone, Copy)]
pub struct CoreIsolation {
    pub pinned: bool,
    pub core: Option<usize>,
    pub total_cores: usize,
}

/// Pin the **current** thread to the *last* available core (e.g. core 15 on a
/// 16-thread Ryzen 7), away from the OS's default scheduling on core 0.
///
/// Never panics: if the OS can't enumerate cores or rejects the affinity lock,
/// it logs and returns `pinned = false` (graceful fallback to normal scheduling).
pub fn pin_to_last_core() -> CoreIsolation {
    match core_affinity::get_core_ids() {
        Some(ids) if !ids.is_empty() => {
            let total = ids.len();
            let last = ids[total - 1];
            let pinned = core_affinity::set_for_current(last);
            if pinned {
                tracing::info!(
                    core = last.id,
                    total_cores = total,
                    "🔒 hot thread pinned to isolated core"
                );
            } else {
                tracing::warn!(
                    core = last.id,
                    "OS rejected core pin — falling back to normal scheduling"
                );
            }
            CoreIsolation {
                pinned,
                core: Some(last.id),
                total_cores: total,
            }
        }
        _ => {
            tracing::warn!("core affinity unavailable — normal scheduling");
            CoreIsolation {
                pinned: false,
                core: None,
                total_cores: 0,
            }
        }
    }
}

/// A request shipped to the isolated worker (message passing — no shared locks).
struct EvalRequest {
    ticker: String,
    regime: Regime,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    conviction: f64,
    vix_score: f64,
    heat: f64,
    is_small_tier: bool,
    is_core: bool,
    reply: std::sync::mpsc::Sender<Decision>,
}

/// Runs a [`Pipeline`] on a dedicated OS thread pinned to the isolated core, so
/// the latency-critical cascade is shielded from the Tokio scheduler and OS
/// background tasks. Work is dispatched over an `mpsc` channel (zero shared
/// mutable state) and the result returned over a one-shot reply channel.
pub struct IsolatedExecutor {
    tx: Option<std::sync::mpsc::Sender<EvalRequest>>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// The isolation status established on the worker thread.
    pub isolation: CoreIsolation,
}

impl IsolatedExecutor {
    /// Spawn the pinned worker that owns `pipeline`. Returns an `io::Error` only
    /// if the OS refuses to create the thread (never panics).
    pub fn spawn(pipeline: Pipeline) -> std::io::Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<EvalRequest>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<CoreIsolation>();
        let handle = std::thread::Builder::new()
            .name("sovereign-iso".into())
            .spawn(move || {
                let _ = ready_tx.send(pin_to_last_core());
                while let Ok(req) = rx.recv() {
                    let decision = pipeline.evaluate(
                        &req.ticker,
                        req.regime,
                        req.closes,
                        req.volumes,
                        req.conviction,
                        req.vix_score,
                        req.heat,
                        req.is_small_tier,
                        req.is_core,
                    );
                    let _ = req.reply.send(decision);
                }
            })?;
        let isolation = ready_rx.recv().unwrap_or(CoreIsolation {
            pinned: false,
            core: None,
            total_cores: 0,
        });
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            isolation,
        })
    }

    /// Evaluate a candidate on the isolated core. Returns `None` only if the
    /// worker has shut down.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        ticker: &str,
        regime: Regime,
        closes: Vec<f64>,
        volumes: Vec<f64>,
        conviction: f64,
        vix_score: f64,
        heat: f64,
        is_small_tier: bool,
        is_core: bool,
    ) -> Option<Decision> {
        let (reply, rx) = std::sync::mpsc::channel();
        let req = EvalRequest {
            ticker: ticker.to_string(),
            regime,
            closes,
            volumes,
            conviction,
            vix_score,
            heat,
            is_small_tier,
            is_core,
            reply,
        };
        self.tx.as_ref()?.send(req).ok()?;
        rx.recv().ok()
    }
}

impl Drop for IsolatedExecutor {
    fn drop(&mut self) {
        // Dropping the sender ends the worker's recv loop; then join cleanly.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_bars(n: usize, up: bool) -> (Vec<f64>, Vec<f64>) {
        let closes: Vec<f64> = (0..n)
            .map(|i| {
                if up {
                    100.0 + i as f64 * 0.1
                } else {
                    100.0 - (i as f64 * 0.05).sin()
                }
            })
            .collect();
        let vols = vec![1.0e6; n];
        (closes, vols)
    }

    #[test]
    fn pipeline_runs_end_to_end() {
        let history = [
            Regime::Bull,
            Regime::Bull,
            Regime::Sideways,
            Regime::Recovery,
            Regime::Bull,
        ];
        let pipe = Pipeline::default_from_history(&history).unwrap();
        let (closes, vols) = synth_bars(80, true);
        let d = pipe.evaluate(
            "AAPL",
            Regime::Bull,
            closes,
            vols,
            12.0,
            0.3,
            0.2,
            false,
            true,
        );
        assert_eq!(d.killhouse.total, 3); // conviction + monte-carlo + cascade
        assert_eq!(d.consensus.votes.len(), 4);
        assert!(d.capital_mass.is_finite() && d.capital_mass > 0.0);
    }

    #[test]
    fn overheated_portfolio_is_vetoed() {
        let history = [Regime::Bull, Regime::Sideways, Regime::Bull];
        let pipe = Pipeline::default_from_history(&history).unwrap();
        let (closes, vols) = synth_bars(80, true);
        let d = pipe.evaluate(
            "NVDA",
            Regime::Bull,
            closes,
            vols,
            15.0,
            0.3,
            0.95,
            false,
            false,
        );
        assert!(!d.approved);
        assert!(d.consensus.reason.contains("VETO"));
    }

    #[test]
    fn adaptive_capital_protocol_scales_mass() {
        let history = [Regime::Bull, Regime::Sideways, Regime::Bull];
        let micro = Pipeline::default_from_history(&history)
            .unwrap()
            .with_capital_size(1_000.0);
        let massive = Pipeline::default_from_history(&history)
            .unwrap()
            .with_capital_size(50_000_000.0);
        // Micro tightens (mass < 1), massive relaxes (mass > 1) and splits orders.
        assert!(micro.capital_mass() < 1.0);
        assert!(massive.capital_mass() > 1.0);

        let (c, v) = synth_bars(80, true);
        let d = massive.evaluate("ES", Regime::Bull, c, v, 12.0, 0.3, 0.2, false, true);
        assert!(d.split_orders);
        assert!(d.capital_mass > 1.0);
    }

    #[test]
    fn core_pin_is_graceful() {
        // Must never panic, regardless of whether the sandbox allows pinning.
        let iso = pin_to_last_core();
        if iso.pinned {
            assert!(iso.core.is_some());
        }
    }

    #[test]
    fn isolated_executor_runs_the_cascade() {
        let history = [Regime::Bull, Regime::Sideways, Regime::Bull];
        let pipe = Pipeline::default_from_history(&history).unwrap();
        let exec = IsolatedExecutor::spawn(pipe).expect("worker thread spawns");
        let (c, v) = synth_bars(80, true);
        let d = exec
            .evaluate("AAPL", Regime::Bull, c, v, 12.0, 0.3, 0.2, false, true)
            .expect("worker returns a decision");
        assert_eq!(d.killhouse.total, 3);
    }
}
