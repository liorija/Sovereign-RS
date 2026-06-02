//! Global kill switch (the Python `GlobalKillSwitch` / `MultiLayerKillSwitch`).
//!
//! **No-Lazy-Rust note:** this is deliberately *lock-free*. A kill switch is
//! read on every hot-path iteration, so we use `AtomicBool` + `AtomicU8`
//! (the reason code) rather than `Arc<Mutex<_>>`. Cloning shares the same atomics.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

/// Why the engine was halted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KillReason {
    None = 0,
    Manual = 1,
    Drawdown = 2,
    BlackSwan = 3,
    Thermal = 4,
    FaultStorm = 5,
    AxiomBreak = 6,
}

impl KillReason {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => KillReason::Manual,
            2 => KillReason::Drawdown,
            3 => KillReason::BlackSwan,
            4 => KillReason::Thermal,
            5 => KillReason::FaultStorm,
            6 => KillReason::AxiomBreak,
            _ => KillReason::None,
        }
    }

    /// Human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            KillReason::None => "NONE",
            KillReason::Manual => "MANUAL",
            KillReason::Drawdown => "DRAWDOWN",
            KillReason::BlackSwan => "BLACK_SWAN",
            KillReason::Thermal => "THERMAL",
            KillReason::FaultStorm => "FAULT_STORM",
            KillReason::AxiomBreak => "AXIOM_BREAK",
        }
    }
}

/// A cheaply-clonable, lock-free global kill switch shared across all tasks.
#[derive(Debug, Clone, Default)]
pub struct GlobalKillSwitch {
    tripped: Arc<AtomicBool>,
    reason: Arc<AtomicU8>,
}

impl GlobalKillSwitch {
    /// A fresh, un-tripped switch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Trip the switch (idempotent — the first reason wins).
    pub fn trip(&self, reason: KillReason) {
        // Set the reason only on the first trip so we keep the root cause.
        if !self.tripped.swap(true, Ordering::SeqCst) {
            self.reason.store(reason as u8, Ordering::SeqCst);
            tracing::error!(reason = reason.as_str(), "🛑 GLOBAL KILL SWITCH TRIPPED");
        }
    }

    /// Whether trading is halted.
    #[inline]
    pub fn is_tripped(&self) -> bool {
        self.tripped.load(Ordering::SeqCst)
    }

    /// The reason for the halt.
    pub fn reason(&self) -> KillReason {
        KillReason::from_u8(self.reason.load(Ordering::SeqCst))
    }

    /// Reset (manual recovery only).
    pub fn reset(&self) {
        self.reason.store(0, Ordering::SeqCst);
        self.tripped.store(false, Ordering::SeqCst);
        tracing::warn!("kill switch reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trips_and_keeps_first_reason() {
        let ks = GlobalKillSwitch::new();
        assert!(!ks.is_tripped());
        ks.trip(KillReason::Drawdown);
        ks.trip(KillReason::Manual); // ignored — first reason wins
        assert!(ks.is_tripped());
        assert_eq!(ks.reason(), KillReason::Drawdown);
    }

    #[test]
    fn clone_shares_state() {
        let a = GlobalKillSwitch::new();
        let b = a.clone();
        a.trip(KillReason::Thermal);
        assert!(b.is_tripped());
        assert_eq!(b.reason(), KillReason::Thermal);
        b.reset();
        assert!(!a.is_tripped());
    }
}
