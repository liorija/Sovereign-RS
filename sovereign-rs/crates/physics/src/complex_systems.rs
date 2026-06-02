//! Complex-systems models: Lotka-Volterra (market-makers vs HFT takers),
//! the Rule-30 cellular automaton (emergent randomness from simple rules), and
//! Arrhenius activation-energy / catalyst breakout dynamics.

// ── Lotka-Volterra predator-prey ────────────────────────────────────────────

/// LV coefficients. Prey = liquidity providers (market-makers), Predator = HFT
/// takers. `dPrey = a·Prey − b·Prey·Pred`, `dPred = d·Prey·Pred − c·Pred`.
#[derive(Debug, Clone, Copy)]
pub struct LvParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

fn lv_deriv(prey: f64, pred: f64, p: &LvParams) -> (f64, f64) {
    (
        p.a * prey - p.b * prey * pred,
        p.d * prey * pred - p.c * pred,
    )
}

/// Integrate the LV system with RK4. Populations are floored at 0.
pub fn lotka_volterra(
    prey0: f64,
    pred0: f64,
    p: &LvParams,
    dt: f64,
    steps: usize,
) -> Vec<(f64, f64)> {
    let mut prey = prey0.max(0.0);
    let mut pred = pred0.max(0.0);
    let mut out = Vec::with_capacity(steps + 1);
    out.push((prey, pred));
    for _ in 0..steps {
        let (k1p, k1q) = lv_deriv(prey, pred, p);
        let (k2p, k2q) = lv_deriv(prey + 0.5 * dt * k1p, pred + 0.5 * dt * k1q, p);
        let (k3p, k3q) = lv_deriv(prey + 0.5 * dt * k2p, pred + 0.5 * dt * k2q, p);
        let (k4p, k4q) = lv_deriv(prey + dt * k3p, pred + dt * k3q, p);
        prey = (prey + dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)).max(0.0);
        pred = (pred + dt / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)).max(0.0);
        out.push((prey, pred));
    }
    out
}

/// Liquidity is exhausted when the market-maker (prey) population collapses.
pub fn liquidity_exhausted(trajectory: &[(f64, f64)], floor: f64) -> bool {
    trajectory.iter().any(|(prey, _)| *prey < floor)
}

// ── Elementary cellular automaton (Rule 30) ─────────────────────────────────

/// One step of an elementary CA with the given Wolfram `rule` (wrap-around).
pub fn rule_step(row: &[bool], rule: u8) -> Vec<bool> {
    let n = row.len();
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| {
            let l = row[(i + n - 1) % n] as u8;
            let c = row[i] as u8;
            let r = row[(i + 1) % n] as u8;
            let idx = (l << 2) | (c << 1) | r;
            (rule >> idx) & 1 == 1
        })
        .collect()
}

/// Rule 30 — chaotic, used as a pseudo-randomness / complexity reference.
pub fn rule30_step(row: &[bool]) -> Vec<bool> {
    rule_step(row, 30)
}

/// Binary (Shannon) entropy in `[0,1]` of the fraction of "on" cells.
pub fn row_entropy(row: &[bool]) -> f64 {
    let n = row.len();
    if n == 0 {
        return 0.0;
    }
    let p = row.iter().filter(|b| **b).count() as f64 / n as f64;
    if p <= 0.0 || p >= 1.0 {
        0.0
    } else {
        -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
    }
}

// ── Arrhenius activation energy / catalyst ──────────────────────────────────

/// Arrhenius-style reaction rate `exp(−Eₐ / T)` (k = 1). Higher "temperature"
/// (market energy) or lower activation energy ⇒ faster reaction (breakout).
pub fn arrhenius_rate(temperature: f64, activation_energy: f64) -> f64 {
    let t = temperature.max(1e-9);
    (-activation_energy.max(0.0) / t).exp()
}

/// A breakout needs activation energy (volume). News acts as a catalyst that
/// lowers the required threshold (`catalyst` in `[0, 0.95]`).
pub fn breakout_threshold(base_threshold: f64, catalyst: f64) -> f64 {
    base_threshold * (1.0 - catalyst.clamp(0.0, 0.95))
}

/// Whether the current volume clears the (catalyst-adjusted) breakout threshold.
pub fn breakout_ready(volume: f64, base_threshold: f64, catalyst: f64) -> bool {
    volume >= breakout_threshold(base_threshold, catalyst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lotka_volterra_oscillates_and_stays_finite() {
        let p = LvParams {
            a: 1.1,
            b: 0.4,
            c: 0.4,
            d: 0.1,
        };
        let traj = lotka_volterra(10.0, 5.0, &p, 0.01, 3000);
        assert!(traj
            .iter()
            .all(|(a, b)| a.is_finite() && b.is_finite() && *a >= 0.0 && *b >= 0.0));
        let max_prey = traj.iter().map(|(a, _)| *a).fold(f64::MIN, f64::max);
        let min_prey = traj.iter().map(|(a, _)| *a).fold(f64::MAX, f64::min);
        assert!(
            max_prey > 10.0 && min_prey < 10.0,
            "should oscillate around the start"
        );
    }

    #[test]
    fn rule30_preserves_width_and_bounds_entropy() {
        let mut row = vec![false; 21];
        row[10] = true;
        for _ in 0..10 {
            row = rule30_step(&row);
            assert_eq!(row.len(), 21);
            assert!((0.0..=1.0).contains(&row_entropy(&row)));
        }
    }

    #[test]
    fn catalyst_lowers_threshold_and_rate_rises_with_temp() {
        assert!(breakout_threshold(100.0, 0.3) < 100.0);
        assert!(breakout_ready(80.0, 100.0, 0.3)); // catalyst makes 80 enough
        assert!(arrhenius_rate(2.0, 1.0) > arrhenius_rate(1.0, 1.0));
    }
}
