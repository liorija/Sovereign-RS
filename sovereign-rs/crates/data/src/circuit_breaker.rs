//! Error-type-aware circuit breaker with exponential backoff.
//!
//! Direct port of `pratati_patch.CircuitBreakerV322`, but as a proper state
//! machine instead of a dict of dicts. Time is **injected** (`now_secs`) so the
//! transitions are unit-testable without sleeping.

use std::collections::HashMap;

/// What kind of failure occurred — drives the cooldown length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// DNS resolution failed — slow to recover (1h base).
    Dns,
    /// Connection actively refused / rate-limited (10m base).
    Refused,
    /// Timed out — likely transient (5m base).
    Timeout,
    /// Anything else (5m base).
    Other,
}

impl ErrorClass {
    /// Classify from an error string, mirroring the Python substring matching.
    pub fn classify(msg: &str) -> Self {
        let m = msg.to_ascii_lowercase();
        if m.contains("getaddrinfo") || m.contains("name resolution") || m.contains("dns") {
            ErrorClass::Dns
        } else if m.contains("refused") || m.contains("10061") {
            ErrorClass::Refused
        } else if m.contains("timeout") || m.contains("timed out") {
            ErrorClass::Timeout
        } else {
            ErrorClass::Other
        }
    }

    fn base_ttl(self) -> f64 {
        match self {
            ErrorClass::Dns => 3600.0,
            ErrorClass::Refused => 600.0,
            ErrorClass::Timeout | ErrorClass::Other => 300.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Closed,
    Open,
    HalfOpen,
}

/// Per-source circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: State,
    fails: u32,
    opened_at: f64,
    ttl: f64,
    backoff: u32,
    max_fails: u32,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            state: State::Closed,
            fails: 0,
            opened_at: 0.0,
            ttl: 300.0,
            backoff: 0,
            max_fails: 3,
        }
    }
}

impl CircuitBreaker {
    /// Absolute ceiling on any cooldown (2h), matching `TTL_MAX`.
    pub const TTL_MAX: f64 = 7200.0;

    /// New breaker with a custom failure threshold.
    pub fn with_threshold(max_fails: u32) -> Self {
        Self {
            max_fails: max_fails.max(1),
            ..Default::default()
        }
    }

    /// Whether the circuit is currently blocking calls. Transitions `Open` →
    /// `HalfOpen` automatically once the cooldown elapses (allowing one probe).
    pub fn is_open(&mut self, now_secs: f64) -> bool {
        if self.state == State::Open {
            if now_secs - self.opened_at > self.ttl {
                self.state = State::HalfOpen;
                return false;
            }
            return true;
        }
        false
    }

    /// Record a success — fully closes the circuit and clears backoff.
    pub fn record_success(&mut self) {
        self.state = State::Closed;
        self.fails = 0;
        self.backoff = 0;
        self.ttl = 300.0;
    }

    /// Record a failure of a given class. Opens the circuit (with exponential
    /// backoff) once `max_fails` is reached.
    pub fn record_failure(&mut self, class: ErrorClass, now_secs: f64) {
        self.fails += 1;
        if self.fails >= self.max_fails {
            let ttl = (class.base_ttl() * 2f64.powi(self.backoff as i32)).min(Self::TTL_MAX);
            self.ttl = ttl;
            self.state = State::Open;
            self.opened_at = now_secs;
            self.backoff += 1;
            tracing::warn!(class = ?class, ttl_s = ttl, backoff = self.backoff, "circuit OPEN");
        }
    }

    /// Remaining cooldown in seconds (0 if not open).
    pub fn cooldown_remaining(&self, now_secs: f64) -> f64 {
        if self.state == State::Open {
            (self.ttl - (now_secs - self.opened_at)).max(0.0)
        } else {
            0.0
        }
    }

    /// Force the breaker closed (manual dashboard reset).
    pub fn force_reset(&mut self) {
        *self = Self {
            max_fails: self.max_fails,
            ..Default::default()
        };
    }
}

/// A registry of breakers keyed by source name (e.g. `"fred"`, `"openinsider"`).
#[derive(Debug, Default)]
pub struct BreakerRegistry {
    breakers: HashMap<String, CircuitBreaker>,
}

impl BreakerRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get-or-create the breaker for `source`.
    pub fn entry(&mut self, source: &str) -> &mut CircuitBreaker {
        self.breakers.entry(source.to_string()).or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn opens_after_threshold_then_half_opens_after_ttl() {
        let mut cb = CircuitBreaker::with_threshold(3);
        let t = 1000.0;
        assert!(!cb.is_open(t));
        cb.record_failure(ErrorClass::Timeout, t);
        cb.record_failure(ErrorClass::Timeout, t);
        assert!(!cb.is_open(t)); // 2 < 3 → still closed
        cb.record_failure(ErrorClass::Timeout, t);
        assert!(cb.is_open(t)); // now open
                                // before ttl (300s) elapses, still open
        assert!(cb.is_open(t + 299.0));
        // after ttl, becomes half-open (probe allowed)
        assert!(!cb.is_open(t + 301.0));
    }

    #[test]
    fn dns_cooldown_longer_than_timeout() {
        let mut dns = CircuitBreaker::with_threshold(1);
        let mut to = CircuitBreaker::with_threshold(1);
        dns.record_failure(ErrorClass::Dns, 0.0);
        to.record_failure(ErrorClass::Timeout, 0.0);
        assert!(dns.cooldown_remaining(0.0) > to.cooldown_remaining(0.0));
    }

    #[test]
    fn success_resets_backoff() {
        let mut cb = CircuitBreaker::with_threshold(1);
        cb.record_failure(ErrorClass::Refused, 0.0);
        cb.record_success();
        assert!(!cb.is_open(0.0));
        assert_eq!(cb.cooldown_remaining(0.0), 0.0);
    }

    proptest! {
        #[test]
        fn backoff_never_exceeds_ttl_max(n in 1u32..20) {
            let mut cb = CircuitBreaker::with_threshold(1);
            for _ in 0..n {
                cb.record_failure(ErrorClass::Dns, 0.0);
            }
            prop_assert!(cb.cooldown_remaining(0.0) <= CircuitBreaker::TTL_MAX);
        }
    }
}
