//! Price "mechanics": Newtonian kinematics, Le Chatelier equilibrium pressure,
//! and a financial Heisenberg uncertainty bound.

fn std(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Discrete 1st/2nd/3rd derivatives of a price path.
#[derive(Debug, Clone, Copy)]
pub struct Kinematics {
    /// Velocity — trend speed (last first-difference).
    pub velocity: f64,
    /// Acceleration — is the trend speeding up or slowing down.
    pub acceleration: f64,
    /// Jerk — change in acceleration (early exhaustion signal).
    pub jerk: f64,
}

/// Compute kinematics from the tail of a price series (needs ≥ 4 points).
pub fn kinematics(prices: &[f64]) -> Kinematics {
    let n = prices.len();
    if n < 4 {
        return Kinematics {
            velocity: 0.0,
            acceleration: 0.0,
            jerk: 0.0,
        };
    }
    let v = |i: usize| prices[i] - prices[i - 1]; // velocity at i
    let velocity = v(n - 1);
    let acceleration = v(n - 1) - v(n - 2);
    let jerk = (v(n - 1) - v(n - 2)) - (v(n - 2) - v(n - 3));
    Kinematics {
        velocity,
        acceleration,
        jerk,
    }
}

impl Kinematics {
    /// True when the trend is decelerating (velocity and acceleration oppose) —
    /// the trend is "running out of fuel" before a reversal.
    pub fn losing_momentum(&self) -> bool {
        self.velocity * self.acceleration < 0.0
    }
}

/// Le Chatelier dynamic equilibrium: a disturbance (whale dump) creates a
/// restoring pressure proportional to the displacement from equilibrium.
#[derive(Debug, Clone, Copy)]
pub struct Equilibrium {
    pub level: f64,
    /// Restoring stiffness `k ≥ 0`.
    pub stiffness: f64,
}

impl Equilibrium {
    pub fn new(level: f64, stiffness: f64) -> Self {
        Self {
            level,
            stiffness: stiffness.max(0.0),
        }
    }
    /// Restoring force `−k·(price − level)` (negative above equilibrium).
    pub fn restoring_force(&self, price: f64) -> f64 {
        -self.stiffness * (price - self.level)
    }
    /// One relaxation step back toward equilibrium.
    pub fn relax(&self, price: f64, dt: f64) -> f64 {
        price + self.restoring_force(price) * dt
    }
}

/// Financial Heisenberg bound: the product of price-level uncertainty and
/// momentum (return) uncertainty over a window. Measuring one precisely blurs
/// the other; a collapsing product warns the tape is being "pinned".
pub fn uncertainty_product(prices: &[f64], window: usize) -> f64 {
    let w = window.max(2);
    if prices.len() < w + 1 {
        return 0.0;
    }
    let tail = &prices[prices.len() - w..];
    let rets: Vec<f64> = tail
        .windows(2)
        .map(|p| if p[0] != 0.0 { p[1] / p[0] - 1.0 } else { 0.0 })
        .collect();
    std(tail) * std(&rets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_path_has_zero_acceleration() {
        let prices: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        let k = kinematics(&prices);
        assert!((k.velocity - 1.0).abs() < 1e-9);
        assert!(k.acceleration.abs() < 1e-9);
    }

    #[test]
    fn decelerating_trend_loses_momentum() {
        // Rising but with negative acceleration (concave): 0, 3, 5, 6, 6.5...
        let prices = [0.0, 3.0, 5.0, 6.0, 6.5];
        let k = kinematics(&prices);
        assert!(k.velocity > 0.0 && k.acceleration < 0.0);
        assert!(k.losing_momentum());
    }

    #[test]
    fn equilibrium_pulls_back() {
        let eq = Equilibrium::new(100.0, 0.5);
        assert!(eq.restoring_force(110.0) < 0.0); // above → pushed down
        let relaxed = eq.relax(110.0, 1.0);
        assert!((100.0..110.0).contains(&relaxed));
    }

    #[test]
    fn uncertainty_is_finite_nonneg() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.3).sin()).collect();
        let u = uncertainty_product(&prices, 20);
        assert!(u.is_finite() && u >= 0.0);
    }
}
