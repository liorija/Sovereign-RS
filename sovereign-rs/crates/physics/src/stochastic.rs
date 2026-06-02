//! Stochastic calculus: Ornstein-Uhlenbeck estimation, Euler-Maruyama (Itô) SDE
//! integration, and a 1-D Fokker-Planck forward density solver.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Fitted Ornstein-Uhlenbeck parameters: `dX = θ(μ − X)dt + σ dW`.
#[derive(Debug, Clone, Copy)]
pub struct OuParams {
    /// Mean-reversion speed (`> 0` ⇒ reverting).
    pub theta: f64,
    /// Long-run mean.
    pub mu: f64,
    /// Instantaneous volatility.
    pub sigma: f64,
    /// Half-life of mean reversion (bars).
    pub half_life: f64,
}

/// Estimate OU parameters from a series via the AR(1) discretization
/// `X_{t+1} = a + b·X_t + ε`.
pub fn fit_ou(series: &[f64], dt: f64) -> OuParams {
    let n = series.len();
    let dt = if dt > 0.0 { dt } else { 1.0 };
    if n < 3 {
        return OuParams {
            theta: 0.0,
            mu: series.first().copied().unwrap_or(0.0),
            sigma: 0.0,
            half_life: f64::INFINITY,
        };
    }
    let x = &series[..n - 1];
    let y = &series[1..];
    let m = x.len() as f64;
    let mx = x.iter().sum::<f64>() / m;
    let my = y.iter().sum::<f64>() / m;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..x.len() {
        sxx += (x[i] - mx) * (x[i] - mx);
        sxy += (x[i] - mx) * (y[i] - my);
    }
    let b = if sxx > 1e-12 { sxy / sxx } else { 0.0 };
    let a = my - b * mx;
    // b must be in (0,1) for a valid reverting fit.
    let b_clamped = b.clamp(1e-6, 0.999_999);
    let theta = -b_clamped.ln() / dt;
    let mu = a / (1.0 - b_clamped);
    // Residual std → sigma.
    let resid_var = {
        let mut s = 0.0;
        for i in 0..x.len() {
            let e = y[i] - (a + b * x[i]);
            s += e * e;
        }
        s / m
    };
    let sigma = (resid_var * 2.0 * theta / (1.0 - b_clamped * b_clamped).max(1e-9))
        .max(0.0)
        .sqrt();
    let half_life = if theta > 1e-12 {
        std::f64::consts::LN_2 / theta
    } else {
        f64::INFINITY
    };
    OuParams {
        theta,
        mu,
        sigma,
        half_life,
    }
}

/// Itô drift correction for log-prices: the GBM log-drift is `μ − ½σ²`.
pub fn ito_log_drift(mu: f64, sigma: f64) -> f64 {
    mu - 0.5 * sigma * sigma
}

/// Euler-Maruyama simulation of an OU path.
pub fn euler_maruyama_ou<R: Rng + ?Sized>(
    x0: f64,
    p: &OuParams,
    dt: f64,
    steps: usize,
    rng: &mut R,
) -> Vec<f64> {
    let mut x = x0;
    let sqrt_dt = dt.max(0.0).sqrt();
    let mut out = Vec::with_capacity(steps + 1);
    out.push(x);
    for _ in 0..steps {
        let z: f64 = StandardNormal.sample(rng);
        x += p.theta * (p.mu - x) * dt + p.sigma * sqrt_dt * z;
        out.push(x);
    }
    out
}

/// One explicit finite-difference step of the Fokker-Planck equation
/// `∂p/∂t = −μ ∂p/∂x + ½σ² ∂²p/∂x²` with zero-flux boundaries, renormalized.
pub fn fokker_planck_step(
    density: &[f64],
    dx: f64,
    dt: f64,
    drift: f64,
    diffusion: f64,
) -> Vec<f64> {
    let n = density.len();
    if n < 3 || dx <= 0.0 {
        return density.to_vec();
    }
    let mut next = density.to_vec();
    let inv_dx = 1.0 / dx;
    let inv_dx2 = inv_dx * inv_dx;
    for i in 1..n - 1 {
        let advection = -drift * (density[i + 1] - density[i - 1]) * 0.5 * inv_dx;
        let diff = 0.5 * diffusion * (density[i + 1] - 2.0 * density[i] + density[i - 1]) * inv_dx2;
        next[i] = (density[i] + dt * (advection + diff)).max(0.0);
    }
    // Renormalize to keep ∫p dx = 1.
    let mass: f64 = next.iter().sum::<f64>() * dx;
    if mass > 1e-12 {
        for v in &mut next {
            *v /= mass;
        }
    }
    next
}

/// Evolve a density `steps` times.
pub fn fokker_planck_evolve(
    density: &[f64],
    dx: f64,
    dt: f64,
    drift: f64,
    diffusion: f64,
    steps: usize,
) -> Vec<f64> {
    let mut p = density.to_vec();
    for _ in 0..steps {
        p = fokker_planck_step(&p, dx, dt, drift, diffusion);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn ou_fit_recovers_reversion() {
        // Synthetic OU around mu=5, fast reversion.
        let mut x = 0.0;
        let mut series = Vec::new();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..2000 {
            let z: f64 = StandardNormal.sample(&mut rng);
            x += 0.3 * (5.0 - x) * 1.0 + 0.2 * z;
            series.push(x);
        }
        let p = fit_ou(&series, 1.0);
        assert!(p.theta > 0.0, "theta {}", p.theta);
        assert!(p.half_life.is_finite() && p.half_life > 0.0);
        assert!((p.mu - 5.0).abs() < 1.5, "mu {}", p.mu);
    }

    #[test]
    fn ito_drift() {
        assert!((ito_log_drift(0.1, 0.2) - (0.1 - 0.02)).abs() < 1e-12);
    }

    #[test]
    fn fokker_planck_conserves_mass_and_spreads() {
        // Start as a narrow spike; pure diffusion should preserve mass & widen it.
        let n = 101;
        let dx = 0.1;
        let mut density = vec![0.0; n];
        density[n / 2] = 1.0 / dx; // unit mass concentrated
        let evolved = fokker_planck_evolve(&density, dx, 0.01, 0.0, 1.0, 50);
        let mass: f64 = evolved.iter().sum::<f64>() * dx;
        assert!((mass - 1.0).abs() < 1e-6, "mass {mass}");
        // It spread out: the central bin is lower than the initial spike.
        assert!(evolved[n / 2] < density[n / 2]);
    }
}
