//! The Kill-House as a composable pipeline of [`Gate`]s, upgraded with a
//! **continuous dimensional wrapper** for the 6-stage multi-paradigm cascade.
//!
//! Each gate is independently testable; a candidate is approved only if **every**
//! gate passes. The cascade-aware gates additionally return a [`TernaryState`]
//! (Short / Flat / Long) where `Flat (0)` is the definitive noise dead-zone.

use ndarray::Array1;

use sovereign_core::domain::Regime;

/// Number of dimensions in the state tensor.
pub const TENSOR_DIMS: usize = 11;

/// Named axes of the 11-dimensional state tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Dim {
    Price = 0,
    Volume = 1,
    Time = 2,
    Spread = 3,
    OrderFlow = 4,
    Sentiment = 5,
    Liquidity = 6,
    Volatility = 7,
    NetworkEntropy = 8,
    RegimeVector = 9,
    CointegrationDelta = 10,
}

/// Tri-state signal. `Flat (0)` is the noise filter / dead-zone.
///
/// **Cache-line isolated:** forced to a full 64-byte L1 line via
/// `#[repr(C, align(64))]` so two cores never false-share the same line when
/// reading/updating adjacent cascade verdicts on the Ryzen. (The discriminant
/// values `-1/0/1` are preserved; only the alignment/footprint changes.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(64))]
pub enum TernaryState {
    Short = -1,
    Flat = 0,
    Long = 1,
}

impl TernaryState {
    /// Branchless construction from a balanced-ternary value (`-1 / 0 / +1`-ish):
    /// positive ⇒ Long, negative ⇒ Short, zero ⇒ Flat.
    #[inline(always)]
    pub fn from_value(v: f64) -> Self {
        // (v > 0) - (v < 0)  →  +1 / 0 / -1   (no branches)
        let s = (v > 0.0) as i8 - (v < 0.0) as i8;
        match s {
            1 => TernaryState::Long,
            -1 => TernaryState::Short,
            _ => TernaryState::Flat,
        }
    }

    #[inline(always)]
    pub fn as_i8(self) -> i8 {
        self as i8
    }

    /// Whether the state carries a tradeable direction (non-Flat).
    #[inline(always)]
    pub fn is_active(self) -> bool {
        !matches!(self, TernaryState::Flat)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TernaryState::Short => "SHORT",
            TernaryState::Flat => "FLAT",
            TernaryState::Long => "LONG",
        }
    }
}

/// Per-session calibration that isolates "day-of" behaviour and kills historical
/// overfitting: every threshold downstream is expressed *relative* to these.
/// Cache-line aligned (`#[repr(C, align(64))]`) with explicit tail padding so
/// the struct is *exactly* one 64-byte L1 line — the CPU fetches all calibration
/// in a single cache transaction with no false sharing.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct DayOfBounds {
    /// Entropy ceiling — above this the cascade collapses to Flat (axiom break).
    pub max_entropy: f64,
    /// Today's volatility dead-zone width (adaptive noise floor).
    pub volatility_threshold: f64,
    /// Cross-sectional fitness rank in `[0,1]` used to elastically normalize.
    pub relative_fitness_rank: f64,
    /// Explicit padding: 24 bytes of data + 40 = exactly 64.
    _pad: [u8; 40],
}

impl Default for DayOfBounds {
    fn default() -> Self {
        Self {
            max_entropy: 0.95,
            volatility_threshold: 0.5,
            relative_fitness_rank: 0.5,
            _pad: [0; 40],
        }
    }
}

impl DayOfBounds {
    pub fn new(max_entropy: f64, volatility_threshold: f64, relative_fitness_rank: f64) -> Self {
        Self {
            max_entropy: max_entropy.clamp(0.0, 1.0),
            volatility_threshold: volatility_threshold.max(0.0),
            relative_fitness_rank: relative_fitness_rank.clamp(0.0, 1.0),
            _pad: [0; 40],
        }
    }
}

/// The continuous dimensional wrapper: the 11-D state tensor plus the day's
/// calibration bounds. Held by [`GateContext`] and read zero-copy by the gates.
///
/// `#[repr(C, align(64))]` pins the hot metadata to the L1 cache line; with
/// 64-byte alignment the compiler guarantees the footprint is a whole multiple
/// of a cache line (no straddling fetch for the tensor header).
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub struct StringTensor {
    /// 11-D state: Price, Volume, Time, Spread, Order-Flow, Sentiment,
    /// Liquidity, Volatility, Network-Entropy, Regime-Vector, Cointegration-Delta.
    pub state: Array1<f64>,
    pub bounds: DayOfBounds,
}

impl StringTensor {
    /// Build from a fixed-size array (guarantees the 11-D invariant).
    pub fn new(values: [f64; TENSOR_DIMS], bounds: DayOfBounds) -> Self {
        Self {
            state: Array1::from_vec(values.to_vec()),
            bounds,
        }
    }

    /// A neutral tensor (all axes mid-scale) for gates that don't use it.
    pub fn neutral() -> Self {
        Self::new([0.5; TENSOR_DIMS], DayOfBounds::default())
    }

    /// Safe axis accessor (never panics; out-of-range ⇒ 0.0).
    #[inline(always)]
    pub fn dim(&self, d: Dim) -> f64 {
        self.state.get(d as usize).copied().unwrap_or(0.0)
    }

    /// Safe index accessor.
    #[inline(always)]
    pub fn at(&self, i: usize) -> f64 {
        self.state.get(i).copied().unwrap_or(0.0)
    }
}

/// Inputs available to every gate for one candidate.
#[derive(Debug, Clone)]
pub struct GateContext {
    pub ticker: String,
    pub regime: Regime,
    /// Recent closes, oldest-first.
    pub closes: Vec<f64>,
    /// Recent volumes, aligned with `closes`.
    pub volumes: Vec<f64>,
    /// Composite conviction accumulated by the signal layer.
    pub conviction: f64,
    /// The 11-D dimensional wrapper + day-of calibration.
    pub tensor: StringTensor,
    /// Adaptive capital-mass multiplier (set by the Adaptive Capital Protocol).
    pub capital_mass: f64,
}

impl GateContext {
    /// Convenience builder for the classic (pre-cascade) gates: a neutral tensor
    /// and unit capital mass are supplied so existing gates keep working.
    pub fn classic(
        ticker: impl Into<String>,
        regime: Regime,
        closes: Vec<f64>,
        volumes: Vec<f64>,
        conviction: f64,
    ) -> Self {
        Self {
            ticker: ticker.into(),
            regime,
            closes,
            volumes,
            conviction,
            tensor: StringTensor::neutral(),
            capital_mass: 1.0,
        }
    }
}

/// The result of one gate.
#[derive(Debug, Clone)]
pub struct GateOutcome {
    pub passed: bool,
    /// A diagnostic numeric (e.g. the CVaR or VPIN value).
    pub score: f64,
    pub label: String,
    /// The cascade's tri-state verdict (`Flat` for gates that don't cascade).
    pub cascade_state: TernaryState,
}

impl GateOutcome {
    pub fn pass(score: f64, label: impl Into<String>) -> Self {
        Self {
            passed: true,
            score,
            label: label.into(),
            cascade_state: TernaryState::Flat,
        }
    }
    pub fn block(score: f64, label: impl Into<String>) -> Self {
        Self {
            passed: false,
            score,
            label: label.into(),
            cascade_state: TernaryState::Flat,
        }
    }
    /// Full constructor used by the cascade gate to carry the tri-state.
    pub fn cascade(score: f64, label: impl Into<String>, state: TernaryState) -> Self {
        Self {
            passed: state.is_active(),
            score,
            label: label.into(),
            cascade_state: state,
        }
    }
    /// Attach/override the cascade state (builder style).
    pub fn with_cascade(mut self, state: TernaryState) -> Self {
        self.cascade_state = state;
        self
    }
}

/// A single Kill-House gate.
pub trait Gate: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, ctx: &GateContext) -> GateOutcome;
}

/// The final verdict after running all gates.
#[derive(Debug, Clone)]
pub struct Verdict {
    pub approved: bool,
    pub passed: usize,
    pub total: usize,
    pub outcomes: Vec<(String, GateOutcome)>,
}

impl Verdict {
    /// The dominant cascade direction across gates that produced one
    /// (sum of tri-states → net Short/Flat/Long).
    pub fn net_cascade(&self) -> TernaryState {
        let sum: i32 = self
            .outcomes
            .iter()
            .map(|(_, o)| o.cascade_state.as_i8() as i32)
            .sum();
        TernaryState::from_value(sum as f64)
    }
}

/// An ordered set of gates.
pub struct KillHouse {
    gates: Vec<Box<dyn Gate>>,
}

impl KillHouse {
    pub fn new(gates: Vec<Box<dyn Gate>>) -> Self {
        Self { gates }
    }

    /// Run every gate; approve iff all pass.
    pub fn run(&self, ctx: &GateContext) -> Verdict {
        let mut outcomes = Vec::with_capacity(self.gates.len());
        let mut passed = 0usize;
        for gate in &self.gates {
            let o = gate.evaluate(ctx);
            if o.passed {
                passed += 1;
            } else {
                tracing::debug!(gate = gate.name(), score = o.score, label = %o.label, "gate blocked");
            }
            outcomes.push((gate.name().to_string(), o));
        }
        Verdict {
            approved: passed == self.gates.len(),
            passed,
            total: self.gates.len(),
            outcomes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_from_value_is_branchless_correct() {
        assert_eq!(TernaryState::from_value(2.0), TernaryState::Long);
        assert_eq!(TernaryState::from_value(-0.1), TernaryState::Short);
        assert_eq!(TernaryState::from_value(0.0), TernaryState::Flat);
        assert!(!TernaryState::Flat.is_active());
        assert!(TernaryState::Long.is_active());
    }

    #[test]
    fn tensor_accessors_never_panic() {
        let t = StringTensor::neutral();
        assert_eq!(t.dim(Dim::Price), 0.5);
        assert_eq!(t.at(999), 0.0); // out of range → 0
    }

    #[test]
    fn outcome_cascade_links_passed_to_active() {
        let o = GateOutcome::cascade(0.7, "x", TernaryState::Long);
        assert!(o.passed);
        let f = GateOutcome::cascade(0.0, "x", TernaryState::Flat);
        assert!(!f.passed);
    }

    #[test]
    fn hot_structs_are_cache_line_aligned() {
        use std::mem::{align_of, size_of};
        assert_eq!(align_of::<DayOfBounds>(), 64);
        assert_eq!(size_of::<DayOfBounds>(), 64); // exactly one line
        assert_eq!(align_of::<StringTensor>(), 64);
        assert_eq!(size_of::<StringTensor>() % 64, 0); // whole multiple of a line
        assert_eq!(align_of::<TernaryState>(), 64);
        // discriminant values survive the repr change
        assert_eq!(TernaryState::Short.as_i8(), -1);
        assert_eq!(TernaryState::Long.as_i8(), 1);
    }
}
