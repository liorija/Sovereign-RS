//! Concrete Kill-House gates.
//!
//! Besides the classic gates (Conviction, Monte-Carlo CVaR, VPIN) this module
//! hosts the [`HyperLayeredVpinGate`] — a branchless 6-stage multi-paradigm
//! compression cascade (Octonion → Quaternion → Fuzzy → Quaternary →
//! Balanced-Ternary → Binary) over the 11-D [`StringTensor`].

use rand::{rngs::StdRng, SeedableRng};

use sovereign_quant::montecarlo::RegimeSwitchingMonteCarlo;

use crate::gate::{Dim, Gate, GateContext, GateOutcome, TernaryState};

/// Gate 1 — minimum conviction.
pub struct ConvictionGate {
    pub min: f64,
}

impl Gate for ConvictionGate {
    fn name(&self) -> &str {
        "conviction"
    }
    fn evaluate(&self, ctx: &GateContext) -> GateOutcome {
        if ctx.conviction >= self.min {
            GateOutcome::pass(
                ctx.conviction,
                format!("conv {:.1} ≥ {:.1}", ctx.conviction, self.min),
            )
        } else {
            GateOutcome::block(
                ctx.conviction,
                format!("conv {:.1} < {:.1}", ctx.conviction, self.min),
            )
        }
    }
}

/// Gate 3 — Monte-Carlo CVaR / Expected-Shortfall floor.
pub struct MonteCarloGate {
    pub engine: RegimeSwitchingMonteCarlo,
    pub max_es: f64,
    pub horizon: usize,
    pub paths: usize,
    pub confidence: f64,
    pub seed: u64,
}

impl Gate for MonteCarloGate {
    fn name(&self) -> &str {
        "monte_carlo_es"
    }
    fn evaluate(&self, ctx: &GateContext) -> GateOutcome {
        let spot = ctx.closes.last().copied().unwrap_or(0.0);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let res = self.engine.run(
            spot,
            self.horizon,
            self.paths,
            ctx.regime.index(),
            self.confidence,
            &mut rng,
        );
        if res.passes_es_gate(self.max_es) {
            GateOutcome::pass(
                res.cvar,
                format!("CVaR {:.4} ≥ {:.4}", res.cvar, self.max_es),
            )
        } else {
            GateOutcome::block(
                res.cvar,
                format!("CVaR {:.4} < {:.4}", res.cvar, self.max_es),
            )
        }
    }
}

/// Gate 5 — informed-flow VPIN ceiling (tick-rule). Retained for direct use and
/// as the order-flow feed into the [`HyperLayeredVpinGate`].
pub struct VpinGate {
    pub ceiling: f64,
    pub window: usize,
}

impl VpinGate {
    /// Tick-rule VPIN over the last `window` bars (port of `_compute_true_vpin`).
    /// Returns a neutral `0.5` when data is insufficient.
    pub fn compute(closes: &[f64], volumes: &[f64], window: usize) -> f64 {
        let n = closes.len().min(volumes.len());
        if n < window.max(2) + 1 {
            return 0.5;
        }
        let mut buy = 0.0;
        let mut sell = 0.0;
        let start = n.saturating_sub(window);
        for i in start.max(1)..n {
            let dp = closes[i] - closes[i - 1];
            let v = volumes[i].max(0.0);
            let b = if dp > 0.0 {
                v
            } else if dp < 0.0 {
                0.0
            } else {
                v * 0.5
            };
            buy += b;
            sell += v - b;
        }
        let total = buy + sell;
        if total < 1.0 {
            return 0.5;
        }
        ((buy - sell).abs() / total).clamp(0.0, 1.0)
    }
}

impl Gate for VpinGate {
    fn name(&self) -> &str {
        "vpin"
    }
    fn evaluate(&self, ctx: &GateContext) -> GateOutcome {
        let vpin = Self::compute(&ctx.closes, &ctx.volumes, self.window);
        if vpin < self.ceiling {
            GateOutcome::pass(vpin, format!("VPIN {vpin:.2} < {:.2}", self.ceiling))
        } else {
            GateOutcome::block(
                vpin,
                format!("VPIN {vpin:.2} ≥ {:.2} (informed flow)", self.ceiling),
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Branchless math primitives for the cascade
// ═══════════════════════════════════════════════════════════════════════════

/// Branchless `x >= t` as `1.0 / 0.0`.
#[inline(always)]
fn ge(x: f64, t: f64) -> f64 {
    (x >= t) as i32 as f64
}

#[inline(always)]
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

#[inline(always)]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ── Quaternion / octonion algebra (Cayley-Dickson) ──────────────────────────
type Quat = [f64; 4];
type Oct = [f64; 8];

#[inline(always)]
fn q_mul(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[inline(always)]
fn q_conj(a: Quat) -> Quat {
    [a[0], -a[1], -a[2], -a[3]]
}

#[inline(always)]
fn q_unit(a: Quat) -> Quat {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
    if n < 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n, a[3] / n]
    }
}

/// Octonion product via Cayley-Dickson over quaternions: this *is* the Fano-plane
/// non-associative multiplication, without a hand-written sign table.
#[inline(always)]
fn o_mul(p: Oct, q: Oct) -> Oct {
    let a: Quat = [p[0], p[1], p[2], p[3]];
    let b: Quat = [p[4], p[5], p[6], p[7]];
    let c: Quat = [q[0], q[1], q[2], q[3]];
    let d: Quat = [q[4], q[5], q[6], q[7]];
    // (a,b)(c,d) = (a·c − d*·b ,  d·a + b·c*)
    let dcb = q_mul(q_conj(d), b);
    let bcc = q_mul(b, q_conj(c));
    let ac = q_mul(a, c);
    let da = q_mul(d, a);
    [
        ac[0] - dcb[0],
        ac[1] - dcb[1],
        ac[2] - dcb[2],
        ac[3] - dcb[3],
        da[0] + bcc[0],
        da[1] + bcc[1],
        da[2] + bcc[2],
        da[3] + bcc[3],
    ]
}

/// Rotate a 3-vector by a quaternion (gimbal-lock-free): `v' = q·v·q⁻¹`.
#[inline(always)]
fn q_rotate(q: Quat, v: [f64; 3]) -> [f64; 3] {
    let u = q_unit(q);
    let qv: Quat = [0.0, v[0], v[1], v[2]];
    let r = q_mul(q_mul(u, qv), q_conj(u));
    [r[1], r[2], r[3]]
}

/// **Layer 5** kernel (branchless): sift a quaternary actor category through the
/// adaptive dead-zone `tau` and the entropy axiom-break. Whale categories (3/0)
/// always carry their direction; retail categories (2/1) survive only if their
/// signal strength clears `tau`; `entropy_kill = 1.0` crushes everything to Flat.
#[inline(always)]
fn sift_ternary(cat: u8, fuzzy: f64, tau: f64, entropy_kill: f64) -> TernaryState {
    let strength = (fuzzy - 0.5).abs() * 2.0; // [0,1]
    let is_whale_up = (cat == 3) as i32 as f64;
    let is_whale_dn = (cat == 0) as i32 as f64;
    let is_retail = (cat == 1 || cat == 2) as i32 as f64;
    let retail_sign = (fuzzy - 0.5).signum();
    let retail_live = ge(strength, tau);
    let raw = is_whale_up - is_whale_dn + is_retail * retail_sign * retail_live;
    TernaryState::from_value(raw * (1.0 - entropy_kill))
}

// ═══════════════════════════════════════════════════════════════════════════
//  HyperLayeredVpinGate — the 6-stage cascade
// ═══════════════════════════════════════════════════════════════════════════

/// Branchless 6-stage multi-paradigm compression cascade. Every evaluation
/// participates in a downward squeeze from continuous 8-D hypercomplex space to
/// a single boolean hardware trigger, calibrated by the day's [`DayOfBounds`]
/// and the adaptive `capital_mass`.
pub struct HyperLayeredVpinGate;

/// Intermediate cascade telemetry (for tests / dashboards).
#[derive(Debug, Clone, Copy)]
pub struct CascadeTrace {
    pub fuzzy: f64,
    pub quaternary: u8,
    pub ternary: TernaryState,
}

impl HyperLayeredVpinGate {
    /// Run all six layers; returns the fuzzy score, the quaternary actor
    /// category and the balanced-ternary verdict.
    pub fn cascade(&self, ctx: &GateContext) -> CascadeTrace {
        let t = &ctx.tensor;
        let b = t.bounds;

        // ── Layer 1 — Octonion (Base-8): non-associative dimensional mix ────
        // Price/Volume/Spread/Liquidity (the named non-associative axes) plus
        // the hypercomplex tail, rotated by a contextual rotor (Sentiment/Regime/Time).
        let a: Oct = [
            t.dim(Dim::Price),
            t.dim(Dim::Volume),
            t.dim(Dim::Spread),
            t.dim(Dim::Liquidity),
            t.dim(Dim::OrderFlow),
            t.dim(Dim::Volatility),
            t.dim(Dim::NetworkEntropy),
            t.dim(Dim::CointegrationDelta),
        ];
        let rotor: Oct = [
            1.0,
            t.dim(Dim::Sentiment),
            t.dim(Dim::RegimeVector),
            t.dim(Dim::Time),
            t.dim(Dim::OrderFlow),
            t.dim(Dim::Sentiment),
            t.dim(Dim::RegimeVector),
            0.0,
        ];
        let oct = o_mul(a, rotor);

        // ── Layer 2 — Quaternion (Base-4): spatial trend rotation, no gimbal lock ──
        let q: Quat = [oct[0], oct[1], oct[2], oct[3]];
        let trend: [f64; 3] = [oct[4], oct[5], oct[6]];
        let rotated = q_rotate(q, trend);
        let theta = rotated[0] + rotated[1] + rotated[2];

        // ── Layer 3 — Fuzzy [0,1]: elastic normalization vs today's fitness ──
        let scale = (0.5 + b.relative_fitness_rank).max(1e-6); // adaptive elasticity
        let fuzzy = clamp01(logistic(theta / scale));

        // ── Layer 4 — Quaternary (Base-4): 4 actor categories via linear rounding ──
        // 3 = Whale Accumulation · 2 = Retail FOMO · 1 = Retail Panic · 0 = Whale Distribution
        let cat = (fuzzy * 4.0).floor().clamp(0.0, 3.0) as u8;

        // ── Layer 5 — Balanced Ternary (Base-3): sift through vol × capital, crush noise ──
        let tau = b.volatility_threshold * ctx.capital_mass.max(1e-9); // adaptive dead-zone
        let entropy_kill = ge(t.dim(Dim::NetworkEntropy), b.max_entropy); // Gödel axiom-break
        let ternary = sift_ternary(cat, fuzzy, tau, entropy_kill);

        CascadeTrace {
            fuzzy,
            quaternary: cat,
            ternary,
        }
    }
}

impl Gate for HyperLayeredVpinGate {
    fn name(&self) -> &str {
        "hyper_vpin_cascade"
    }
    fn evaluate(&self, ctx: &GateContext) -> GateOutcome {
        let tr = self.cascade(ctx);
        // ── Layer 6 — Binary (Base-2): physical execution trigger ──
        let label = format!(
            "cat{} fuzzy={:.3} → {}",
            tr.quaternary,
            tr.fuzzy,
            tr.ternary.as_str()
        );
        GateOutcome::cascade(tr.fuzzy, label, tr.ternary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{DayOfBounds, StringTensor, TENSOR_DIMS};
    use proptest::prelude::*;
    use sovereign_core::domain::Regime;

    fn ctx_with(tensor: StringTensor, capital_mass: f64) -> GateContext {
        GateContext {
            ticker: "X".into(),
            regime: Regime::Bull,
            closes: Vec::new(),
            volumes: Vec::new(),
            conviction: 0.0,
            tensor,
            capital_mass,
        }
    }

    #[test]
    fn vpin_extremes() {
        let closes: Vec<f64> = (0..60).map(|i| 100.0 + i as f64).collect();
        let vols = vec![1000.0; 60];
        let v = VpinGate::compute(&closes, &vols, 50);
        assert!(v > 0.9, "got {v}");
    }

    #[test]
    fn conviction_gate_blocks_low() {
        let g = ConvictionGate { min: 8.0 };
        let ctx = GateContext::classic("X", Regime::Bull, vec![10.0; 60], vec![1.0; 60], 5.0);
        assert!(!g.evaluate(&ctx).passed);
    }

    #[test]
    fn cascade_is_deterministic() {
        let mut v = [0.6; TENSOR_DIMS];
        v[Dim::Price as usize] = 0.9;
        let t = StringTensor::new(v, DayOfBounds::new(0.95, 0.2, 0.5));
        let g = HyperLayeredVpinGate;
        let a = g.evaluate(&ctx_with(t.clone(), 1.0));
        let b = g.evaluate(&ctx_with(t, 1.0));
        assert_eq!(a.cascade_state, b.cascade_state);
        assert!(a.score.is_finite());
    }

    #[test]
    fn maximal_entropy_crushes_to_flat() {
        let mut v = [0.8; TENSOR_DIMS];
        v[Dim::NetworkEntropy as usize] = 1.0; // ≥ max_entropy below
        let t = StringTensor::new(v, DayOfBounds::new(0.5, 0.2, 0.5));
        let out = HyperLayeredVpinGate.evaluate(&ctx_with(t, 1.0));
        assert_eq!(out.cascade_state, TernaryState::Flat);
        assert!(!out.passed); // Flat = noise dead-zone = no execution trigger
    }

    #[test]
    fn deadzone_widens_with_capital_mass() {
        // Retail category (2 = FOMO), modest strength: a tight dead-zone lets it
        // through, a wide one (huge capital × vol) crushes it to Flat.
        assert!(sift_ternary(2, 0.62, 0.10, 0.0).is_active());
        assert_eq!(sift_ternary(2, 0.62, 5.00, 0.0), TernaryState::Flat);
    }

    #[test]
    fn whales_ignore_the_deadzone() {
        // Whale accumulation/distribution always carry direction regardless of τ.
        assert_eq!(sift_ternary(3, 0.99, 100.0, 0.0), TernaryState::Long);
        assert_eq!(sift_ternary(0, 0.01, 100.0, 0.0), TernaryState::Short);
    }

    #[test]
    fn entropy_axiom_break_forces_flat() {
        assert_eq!(sift_ternary(3, 0.99, 0.0, 1.0), TernaryState::Flat);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn cascade_never_panics(
            vals in proptest::collection::vec(-5.0f64..5.0, TENSOR_DIMS),
            vt in 0.0f64..2.0,
            rfr in 0.0f64..1.0,
            cm in 0.01f64..100.0,
        ) {
            let mut arr = [0.0f64; TENSOR_DIMS];
            for (i, v) in vals.iter().enumerate() { arr[i] = *v; }
            let t = StringTensor::new(arr, DayOfBounds::new(0.95, vt, rfr));
            let out = HyperLayeredVpinGate.evaluate(&ctx_with(t, cm));
            prop_assert!(out.score.is_finite() && (0.0..=1.0).contains(&out.score));
            // passed must agree with the tri-state's activeness
            prop_assert_eq!(out.passed, out.cascade_state.is_active());
        }
    }
}
