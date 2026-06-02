//! Intraday guards: drawdown ladder + black-swan circuit breaker.
//! Ports of `DrawdownLadder` and `risk_patch.BlackSwanGuard`.

use std::collections::VecDeque;

/// Tiered drawdown circuit breaker. Trips progressively as equity falls from
/// its peak through the configured levels (e.g. -5/-10/-15/-20%).
#[derive(Debug, Clone)]
pub struct DrawdownLadder {
    levels: Vec<f64>, // negative fractions, ascending in magnitude
    peak: f64,
}

impl DrawdownLadder {
    /// `levels` are negative fractions like `[-0.05, -0.10, -0.15, -0.20]`.
    pub fn new(levels: Vec<f64>) -> Self {
        Self {
            levels,
            peak: f64::MIN,
        }
    }

    /// Update with the latest equity and return the deepest tripped tier index
    /// (`None` = no breach). Tier 0 is the shallowest level.
    pub fn update(&mut self, equity: f64) -> Option<usize> {
        if equity.is_finite() {
            self.peak = self.peak.max(equity);
        }
        if self.peak <= 0.0 || !self.peak.is_finite() {
            return None;
        }
        let dd = equity / self.peak - 1.0;
        let mut tripped = None;
        for (i, &lvl) in self.levels.iter().enumerate() {
            if dd <= lvl {
                tripped = Some(i);
            }
        }
        tripped
    }

    /// Current drawdown from peak (≤ 0).
    pub fn drawdown(&self, equity: f64) -> f64 {
        if self.peak > 0.0 && self.peak.is_finite() {
            equity / self.peak - 1.0
        } else {
            0.0
        }
    }
}

/// Black-swan freeze state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwanState {
    Normal,
    Freeze,
    Recover,
}

/// Real-time SPY-based intraday circuit breaker (port of `BlackSwanGuard`).
/// Time is injected (`now_secs`) so transitions are deterministic in tests.
#[derive(Debug, Clone)]
pub struct BlackSwanGuard {
    prices: VecDeque<(f64, f64)>, // (now_secs, price)
    spy_open: f64,
    state: SwanState,
    freeze_ts: f64,
    reason: String,
}

impl Default for BlackSwanGuard {
    fn default() -> Self {
        Self {
            prices: VecDeque::with_capacity(Self::HISTORY_LEN),
            spy_open: 0.0,
            state: SwanState::Normal,
            freeze_ts: 0.0,
            reason: String::new(),
        }
    }
}

impl BlackSwanGuard {
    const DROP_DAY: f64 = 0.030; // -3% from the open
    const DROP_15M: f64 = 0.015; // -1.5% in 15 minutes
    const VELOCITY: f64 = 0.010; // -1%/bar over the last few bars
    const RECOVER_S: f64 = 1800.0; // 30 min stable before unfreezing
    const HISTORY_LEN: usize = 20;

    /// Feed the latest SPY price; returns the (possibly updated) state.
    pub fn update(&mut self, now_secs: f64, spy: f64) -> SwanState {
        if !(spy.is_finite() && spy > 0.0) {
            return self.state;
        }
        if self.prices.len() == Self::HISTORY_LEN {
            self.prices.pop_front();
        }
        self.prices.push_back((now_secs, spy));
        if self.spy_open <= 0.0 {
            self.spy_open = spy;
        }

        // 1) Drop from today's open.
        if (self.spy_open - spy) / self.spy_open >= Self::DROP_DAY {
            return self.activate(now_secs, "day_drop");
        }
        // 2) Drop over the last 15 minutes.
        let window: Vec<f64> = self
            .prices
            .iter()
            .filter(|(t, _)| now_secs - t <= 900.0)
            .map(|(_, p)| *p)
            .collect();
        if window.len() >= 3 {
            let first = window[0];
            if first > 0.0 && (first - spy) / first >= Self::DROP_15M {
                return self.activate(now_secs, "15m_drop");
            }
        }
        // 3) Velocity over the last 4 samples.
        if self.prices.len() >= 4 {
            let v: Vec<f64> = self.prices.iter().rev().take(4).map(|(_, p)| *p).collect();
            // v is newest-first; compute consecutive drops oldest→newest
            let drops = [
                (v[3] - v[2]) / v[3],
                (v[2] - v[1]) / v[2],
                (v[1] - v[0]) / v[1],
            ];
            if drops.iter().sum::<f64>() / 3.0 >= Self::VELOCITY {
                return self.activate(now_secs, "velocity");
            }
        }

        // Recovery path.
        match self.state {
            SwanState::Freeze => {
                self.state = SwanState::Recover;
            }
            SwanState::Recover => {
                if now_secs - self.freeze_ts >= Self::RECOVER_S {
                    self.state = SwanState::Normal;
                    self.spy_open = spy;
                }
            }
            SwanState::Normal => {}
        }
        self.state
    }

    fn activate(&mut self, now_secs: f64, reason: &str) -> SwanState {
        if self.state == SwanState::Normal {
            self.freeze_ts = now_secs;
            self.reason = reason.to_string();
            tracing::warn!(reason, "BLACK SWAN — new buys frozen");
        }
        self.state = SwanState::Freeze;
        self.state
    }

    /// Whether new buys are currently allowed.
    pub fn is_buy_allowed(&self) -> bool {
        self.state == SwanState::Normal
    }

    /// Why the guard last tripped.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_shallow_drawdown_does_not_trip() {
        let mut l = DrawdownLadder::new(vec![-0.05, -0.10, -0.15, -0.20]);
        assert_eq!(l.update(100.0), None);
        assert_eq!(l.update(96.0), None); // -4% does not breach the -5% tier
    }

    #[test]
    fn ladder_levels() {
        let mut l = DrawdownLadder::new(vec![-0.05, -0.10, -0.15, -0.20]);
        l.update(100.0);
        assert_eq!(l.update(94.0), Some(0)); // -6% → tier 0
        assert_eq!(l.update(88.0), Some(1)); // -12% → tier 1
        assert_eq!(l.update(79.0), Some(3)); // -21% → tier 3
    }

    #[test]
    fn flash_crash_freezes_buys() {
        let mut g = BlackSwanGuard::default();
        // gradual then sharp drop within the same minute window
        let base = 1000.0;
        g.update(0.0, 500.0); // open
        for (i, px) in [497.0, 494.0, 491.0, 488.0].iter().enumerate() {
            g.update((i + 1) as f64, *px);
        }
        let _ = base;
        assert!(!g.is_buy_allowed());
    }
}
