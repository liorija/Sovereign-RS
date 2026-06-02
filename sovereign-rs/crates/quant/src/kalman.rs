//! Online Kalman filter for a time-varying hedge ratio (port of
//! `v322_stat_arb_engine.KalmanHedgeTracker`).
//!
//! State `x = [intercept, β]`; observation `price_a = intercept + β·price_b + ε`.
//! Tracks β as the linear relationship between two assets drifts — the basis of
//! dynamic pairs hedging.

/// A 2-state Kalman filter estimating an intercept and a slope (β).
#[derive(Debug, Clone)]
pub struct KalmanHedge {
    q: f64, // state transition noise (delta)
    r: f64, // observation noise
    p: [[f64; 2]; 2],
    x: [f64; 2], // [intercept, beta]
}

impl KalmanHedge {
    /// `delta` controls how fast β adapts (1e-4 typical); `noise_obs` is the
    /// measurement variance (1e-3 typical).
    pub fn new(delta: f64, noise_obs: f64) -> Self {
        Self {
            q: delta.max(0.0),
            r: noise_obs.max(1e-12),
            p: [[10.0, 0.0], [0.0, 10.0]],
            x: [0.0, 0.0],
        }
    }

    /// Feed one observation; returns the updated β. Non-finite inputs are
    /// ignored (β unchanged), so the filter never produces NaN.
    pub fn update(&mut self, price_a: f64, price_b: f64) -> f64 {
        if !price_a.is_finite() || !price_b.is_finite() {
            return self.x[1];
        }
        // H = [1, price_b]
        let h = [1.0, price_b];
        // P_pred = P + Q   (Q = delta * I)
        let pp = [
            [self.p[0][0] + self.q, self.p[0][1]],
            [self.p[1][0], self.p[1][1] + self.q],
        ];
        // S = H P_pred H^T + R   (scalar)
        let php = [
            h[0] * pp[0][0] + h[1] * pp[1][0],
            h[0] * pp[0][1] + h[1] * pp[1][1],
        ];
        let s = php[0] * h[0] + php[1] * h[1] + self.r;
        if !s.is_finite() || s.abs() < 1e-300 {
            return self.x[1];
        }
        // K = P_pred H^T / S   (2-vector)
        let pht = [
            pp[0][0] * h[0] + pp[0][1] * h[1],
            pp[1][0] * h[0] + pp[1][1] * h[1],
        ];
        let k = [pht[0] / s, pht[1] / s];
        // innovation
        let y = price_a - (h[0] * self.x[0] + h[1] * self.x[1]);
        self.x[0] += k[0] * y;
        self.x[1] += k[1] * y;
        // P = (I - K H) P_pred
        let kh = [[k[0] * h[0], k[0] * h[1]], [k[1] * h[0], k[1] * h[1]]];
        let imkh = [[1.0 - kh[0][0], -kh[0][1]], [-kh[1][0], 1.0 - kh[1][1]]];
        self.p = mat2_mul(imkh, pp);
        self.x[1]
    }

    /// Run the filter over aligned series, returning the β path.
    pub fn batch(&mut self, pa: &[f64], pb: &[f64]) -> Vec<f64> {
        let n = pa.len().min(pb.len());
        (0..n).map(|i| self.update(pa[i], pb[i])).collect()
    }

    /// Current β estimate.
    pub fn beta(&self) -> f64 {
        self.x[1]
    }

    /// Current intercept estimate.
    pub fn intercept(&self) -> f64 {
        self.x[0]
    }
}

fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn recovers_known_beta() {
        // a = 2 + 1.5 b + small noise
        let mut k = KalmanHedge::new(1e-3, 1e-2);
        let mut beta = 0.0;
        for i in 0..2000 {
            let b = 100.0 + (i as f64 * 0.01).sin() * 10.0;
            let a = 2.0 + 1.5 * b;
            beta = k.update(a, b);
        }
        assert!((beta - 1.5).abs() < 0.1, "beta={beta}");
    }

    proptest! {
        #[test]
        fn never_nan(seq in proptest::collection::vec((-1e3f64..1e3, -1e3f64..1e3), 1..200)) {
            let mut k = KalmanHedge::new(1e-4, 1e-3);
            for (a, b) in seq {
                let beta = k.update(a, b);
                prop_assert!(beta.is_finite());
            }
        }
    }
}
