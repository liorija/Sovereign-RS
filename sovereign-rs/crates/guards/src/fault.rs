//! Fault boundary (the Python `FaultBoundary`): isolate a panicking sub-system
//! so one bad component can't take the whole engine down.
//!
//! Uses `std::panic::catch_unwind` (which is why the release profile keeps
//! `panic = "unwind"`). Repeated faults ("a fault storm") escalate to the
//! [`GlobalKillSwitch`].

use std::any::Any;
use std::panic::{catch_unwind, UnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};

use sovereign_core::error::{Result, SovereignError};

use crate::killswitch::{GlobalKillSwitch, KillReason};

/// Wraps fallible, possibly-panicking work in an isolation boundary.
#[derive(Debug)]
pub struct FaultBoundary {
    name: String,
    faults: AtomicU32,
    max_faults: u32,
    kill_switch: Option<GlobalKillSwitch>,
}

impl FaultBoundary {
    /// New boundary that tolerates up to `max_faults` panics before it's a storm.
    pub fn new(name: impl Into<String>, max_faults: u32) -> Self {
        Self {
            name: name.into(),
            faults: AtomicU32::new(0),
            max_faults: max_faults.max(1),
            kill_switch: None,
        }
    }

    /// Escalate fault storms to a shared kill switch.
    pub fn with_kill_switch(mut self, ks: GlobalKillSwitch) -> Self {
        self.kill_switch = Some(ks);
        self
    }

    /// Run `f`, catching any panic and converting it into a typed error.
    /// On the `max_faults`-th panic, trips the kill switch (if attached).
    pub fn run<T>(&self, f: impl FnOnce() -> T + UnwindSafe) -> Result<T> {
        match catch_unwind(f) {
            Ok(v) => Ok(v),
            Err(payload) => {
                let n = self.faults.fetch_add(1, Ordering::SeqCst) + 1;
                let message = panic_message(payload.as_ref());
                tracing::error!(component = %self.name, fault = n, message = %message, "fault isolated");
                if n >= self.max_faults {
                    if let Some(ks) = &self.kill_switch {
                        ks.trip(KillReason::FaultStorm);
                    }
                }
                Err(SovereignError::Panic {
                    component: self.name.clone(),
                    message,
                })
            }
        }
    }

    /// Total faults seen.
    pub fn fault_count(&self) -> u32 {
        self.faults.load(Ordering::SeqCst)
    }

    /// Whether the fault threshold has been reached.
    pub fn is_storm(&self) -> bool {
        self.fault_count() >= self.max_faults
    }

    /// Clear the fault counter (after a recovery).
    pub fn reset(&self) {
        self.faults.store(0, Ordering::SeqCst);
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_path_passes_through() {
        let fb = FaultBoundary::new("calc", 3);
        let v = fb.run(|| 2 + 2).unwrap();
        assert_eq!(v, 4);
        assert_eq!(fb.fault_count(), 0);
    }

    #[test]
    fn panic_is_isolated_and_counted() {
        let fb = FaultBoundary::new("risky", 3);
        let r: Result<i32> = fb.run(|| panic!("boom in component"));
        assert!(matches!(r, Err(SovereignError::Panic { .. })));
        assert_eq!(fb.fault_count(), 1);
        assert!(!fb.is_storm());
    }

    #[test]
    fn fault_storm_trips_kill_switch() {
        let ks = GlobalKillSwitch::new();
        let fb = FaultBoundary::new("flaky", 2).with_kill_switch(ks.clone());
        let _: Result<()> = fb.run(|| panic!("1"));
        assert!(!ks.is_tripped());
        let _: Result<()> = fb.run(|| panic!("2"));
        assert!(ks.is_tripped());
        assert_eq!(ks.reason(), KillReason::FaultStorm);
    }
}
