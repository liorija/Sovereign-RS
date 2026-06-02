//! Thermodynamic halting guard — the Guerilla-Protocol thermal throttle.
//!
//! On a fanless ThinkPad, a deep Monte-Carlo / ML burst will thermally throttle
//! the CPU, silently destroying latency. This guard reads the die temperature
//! and **degrades simulation depth** (e.g. 10 000 → 1 000 paths) as the chip
//! heats, trading accuracy for thermal headroom *before* the hardware does it
//! for us.

use std::fs;
use std::path::Path;

/// Read the hottest CPU thermal zone in °C (Linux `sysfs`), or `None` if
/// unavailable (other OS / container without thermal zones).
pub fn read_cpu_temp_celsius() -> Option<f64> {
    let mut max: Option<f64> = None;
    let entries = fs::read_dir(Path::new("/sys/class/thermal")).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let is_zone = path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.starts_with("thermal_zone"));
        if !is_zone {
            continue;
        }
        if let Ok(raw) = fs::read_to_string(path.join("temp")) {
            if let Ok(milli) = raw.trim().parse::<f64>() {
                let c = milli / 1000.0;
                if c.is_finite() && (1.0..200.0).contains(&c) {
                    max = Some(max.map_or(c, |m| m.max(c)));
                }
            }
        }
    }
    max
}

/// Degrades workload depth as temperature rises.
#[derive(Debug, Clone, Copy)]
pub struct ThermodynamicGuard {
    /// Temperature at which throttling begins.
    pub warn_c: f64,
    /// Temperature at which depth is clamped to `min_depth`.
    pub crit_c: f64,
    /// Minimum allowed depth (never simulate fewer than this).
    pub min_depth: usize,
}

impl Default for ThermodynamicGuard {
    fn default() -> Self {
        Self {
            warn_c: 80.0,
            crit_c: 95.0,
            min_depth: 1_000,
        }
    }
}

impl ThermodynamicGuard {
    pub fn new(warn_c: f64, crit_c: f64, min_depth: usize) -> Self {
        Self {
            warn_c,
            crit_c,
            min_depth,
        }
    }

    /// Linearly degrade `base_depth` toward `min_depth` across `[warn_c, crit_c]`.
    pub fn adaptive_depth(&self, base_depth: usize, temp_c: f64) -> usize {
        let floor = self.min_depth.min(base_depth);
        if !temp_c.is_finite() || temp_c < self.warn_c {
            return base_depth;
        }
        if temp_c >= self.crit_c {
            return floor;
        }
        let denom = (self.crit_c - self.warn_c).max(1e-6);
        let frac = ((temp_c - self.warn_c) / denom).clamp(0.0, 1.0);
        let depth = base_depth as f64 - frac * (base_depth as f64 - floor as f64);
        (depth.round() as usize).clamp(floor, base_depth)
    }

    /// Read the live temperature and pick a depth (falls back to `base_depth`
    /// when no sensor is available).
    pub fn current_depth(&self, base_depth: usize) -> usize {
        match read_cpu_temp_celsius() {
            Some(t) => self.adaptive_depth(base_depth, t),
            None => base_depth,
        }
    }

    /// Whether we're at or past the warning temperature.
    pub fn should_throttle(&self, temp_c: f64) -> bool {
        temp_c.is_finite() && temp_c >= self.warn_c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn cool_keeps_full_depth() {
        let g = ThermodynamicGuard::default();
        assert_eq!(g.adaptive_depth(10_000, 50.0), 10_000);
    }

    #[test]
    fn hot_clamps_to_min() {
        let g = ThermodynamicGuard::default();
        assert_eq!(g.adaptive_depth(10_000, 99.0), 1_000);
    }

    #[test]
    fn midrange_degrades_monotonically() {
        let g = ThermodynamicGuard::default();
        let warm = g.adaptive_depth(10_000, 85.0);
        let hotter = g.adaptive_depth(10_000, 90.0);
        assert!((1_000..=10_000).contains(&warm));
        assert!(hotter <= warm);
    }

    #[test]
    fn live_read_never_panics() {
        // May be Some or None depending on the host; just must not panic.
        let _ = read_cpu_temp_celsius();
        let g = ThermodynamicGuard::default();
        assert!(g.current_depth(5_000) >= 1_000);
    }

    proptest! {
        #[test]
        fn depth_always_within_bounds(base in 1usize..100_000, temp in -10.0f64..150.0) {
            let g = ThermodynamicGuard::default();
            let d = g.adaptive_depth(base, temp);
            prop_assert!(d <= base);
            prop_assert!(d >= g.min_depth.min(base));
        }
    }
}
