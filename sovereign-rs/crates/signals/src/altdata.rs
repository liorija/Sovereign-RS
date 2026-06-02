//! Alternative-data conviction scoring (port of
//! `v322_alternative_data_engine.AlternativeDataEngine`).
//!
//! Each "engine" is an independent weak signal mapping an observation to a
//! bounded conviction delta; the aggregator combines them with the same weights
//! as the Python (`credit 1.5`, `crypto 1.3`, `supply_chain 1.2`, `gtrends 1.1`).
//! Here we port the **scoring** — the network fetchers (Open-Meteo, OpenSky,
//! FRED, HIBP, …) live in the data plane (see `docs/ROADMAP.md`).

use serde::Serialize;

/// Observations fed to the alt-data engines for one ticker.
#[derive(Debug, Clone, Copy, Default)]
pub struct AltInputs {
    /// Bitcoin 7-day momentum (fraction), the global risk-on/off canary.
    pub btc_mom_7d: f64,
    pub btc_mom_30d: f64,
    /// High-yield credit spread (bps) and its 30-day average.
    pub hy_spread: f64,
    pub hy_spread_ma: f64,
    /// Baltic Dry Index and the copper/gold ratio (growth proxies).
    pub bdi: f64,
    pub copper_gold: f64,
    /// Google-Trends search-interest z-score.
    pub trend_z: f64,
    /// Whether the ticker is a risk-on / risk-off name.
    pub is_risk_on: bool,
    pub is_risk_off: bool,
}

fn clip(d: i32, lo: i32, hi: i32) -> i32 {
    d.clamp(lo, hi)
}

/// Crypto risk-appetite engine.
pub fn crypto_risk(i: &AltInputs) -> i32 {
    let (m7, m30) = (i.btc_mom_7d, i.btc_mom_30d);
    let d = if i.is_risk_on {
        if m7 > 0.10 && m30 > 0.05 {
            7
        } else if m7 > 0.05 {
            4
        } else if m7 < -0.10 {
            -7
        } else if m7 < -0.05 {
            -4
        } else {
            0
        }
    } else if i.is_risk_off {
        if m7 < -0.10 {
            5
        } else if m7 > 0.10 {
            -3
        } else {
            0
        }
    } else {
        0
    };
    clip(d, -10, 10)
}

/// Credit-spread engine (market-wide).
pub fn credit_spread(i: &AltInputs) -> i32 {
    if i.hy_spread <= 0.0 || i.hy_spread_ma <= 0.0 {
        return 0;
    }
    let d = if i.hy_spread > i.hy_spread_ma * 1.2 {
        -4
    } else if i.hy_spread < i.hy_spread_ma * 0.85 {
        3
    } else {
        0
    };
    clip(d, -10, 10)
}

/// Supply-chain engine (BDI + Dr. Copper).
pub fn supply_chain(i: &AltInputs) -> i32 {
    let mut d = 0;
    if i.copper_gold > 0.0 {
        if i.copper_gold > 0.22 {
            d += 3;
        } else if i.copper_gold < 0.15 {
            d -= 4;
        }
    }
    if i.bdi > 0.0 {
        if i.bdi > 2500.0 {
            d += 4;
        } else if i.bdi < 900.0 {
            d -= 4;
        }
    }
    clip(d, -10, 10)
}

/// Google-Trends attention engine (Da et al. 2011).
pub fn google_trends(i: &AltInputs) -> i32 {
    let z = i.trend_z;
    let d = if z > 2.5 {
        8
    } else if z > 1.5 {
        5
    } else if z > 0.8 {
        2
    } else if z < -1.5 {
        -4
    } else {
        0
    };
    clip(d, -10, 10)
}

/// The weighted composite, clipped to ±20 (matches `AlternativeDataEngine`).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AltScore {
    pub crypto: i32,
    pub credit: i32,
    pub supply_chain: i32,
    pub gtrends: i32,
    pub total: i32,
}

/// Run all engines and combine with the production weights.
pub fn aggregate(i: &AltInputs) -> AltScore {
    let crypto = crypto_risk(i);
    let credit = credit_spread(i);
    let supply = supply_chain(i);
    let gt = google_trends(i);
    let raw = 1.3 * crypto as f64 + 1.5 * credit as f64 + 1.2 * supply as f64 + 1.1 * gt as f64;
    let total = (raw.round() as i32).clamp(-20, 20);
    AltScore {
        crypto,
        credit,
        supply_chain: supply,
        gtrends: gt,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn risk_on_rallies_with_btc() {
        let i = AltInputs {
            btc_mom_7d: 0.12,
            btc_mom_30d: 0.06,
            is_risk_on: true,
            ..Default::default()
        };
        assert!(crypto_risk(&i) > 0);
    }

    #[test]
    fn wide_credit_spread_is_bearish() {
        let i = AltInputs {
            hy_spread: 500.0,
            hy_spread_ma: 400.0,
            ..Default::default()
        };
        assert!(credit_spread(&i) < 0);
    }

    #[test]
    fn composite_is_bounded() {
        let i = AltInputs {
            btc_mom_7d: 1.0,
            btc_mom_30d: 1.0,
            is_risk_on: true,
            copper_gold: 0.3,
            bdi: 5000.0,
            trend_z: 3.0,
            hy_spread: 100.0,
            hy_spread_ma: 400.0,
            ..Default::default()
        };
        assert!(aggregate(&i).total <= 20);
    }

    proptest! {
        #[test]
        fn aggregate_always_in_range(
            btc in -2.0f64..2.0, hy in 0.0f64..1000.0, hyma in 1.0f64..1000.0,
            bdi in 0.0f64..6000.0, cg in 0.0f64..0.5, z in -4.0f64..4.0,
        ) {
            let i = AltInputs {
                btc_mom_7d: btc, btc_mom_30d: btc, hy_spread: hy, hy_spread_ma: hyma,
                bdi, copper_gold: cg, trend_z: z, is_risk_on: true, is_risk_off: false,
            };
            let s = aggregate(&i);
            prop_assert!((-20..=20).contains(&s.total));
        }
    }
}
