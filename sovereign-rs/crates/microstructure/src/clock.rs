//! Information-driven clocks — sampling by *information*, not chronological time.
//!
//! Standard OHLCV samples on wall-clock time, which under-samples bursts and
//! over-samples quiet periods. Following López de Prado, we sample on
//! **volume** and **tick-imbalance**: a bar closes when a fixed amount of volume
//! (or signed order-flow imbalance) has accumulated. The bot's perception of
//! time therefore *dilates* during crashes — many information-bars form per
//! chronological minute when flow is violent, few when the tape is calm.

use serde::Serialize;

/// A raw trade tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tick {
    /// Unix timestamp (seconds, fractional allowed).
    pub ts: f64,
    pub price: f64,
    pub volume: f64,
}

/// An information bar (the unit of "dilated" time).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct InfoBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub n_ticks: u32,
    pub start_ts: f64,
    pub end_ts: f64,
}

impl InfoBar {
    fn start(t: &Tick) -> Self {
        Self {
            open: t.price,
            high: t.price,
            low: t.price,
            close: t.price,
            volume: t.volume.max(0.0),
            n_ticks: 1,
            start_ts: t.ts,
            end_ts: t.ts,
        }
    }
    fn push(&mut self, t: &Tick) {
        self.high = self.high.max(t.price);
        self.low = self.low.min(t.price);
        self.close = t.price;
        self.volume += t.volume.max(0.0);
        self.n_ticks += 1;
        self.end_ts = t.ts;
    }
    /// Chronological span of the bar in seconds.
    pub fn duration_s(&self) -> f64 {
        (self.end_ts - self.start_ts).max(0.0)
    }
}

/// Build **volume bars**: a new bar closes once cumulative volume reaches
/// `volume_per_bar`. Ticks with non-finite/negative fields are skipped.
pub fn volume_bars(ticks: &[Tick], volume_per_bar: f64) -> Vec<InfoBar> {
    let threshold = if volume_per_bar.is_finite() && volume_per_bar > 0.0 {
        volume_per_bar
    } else {
        return Vec::new();
    };
    let mut bars = Vec::new();
    let mut cur: Option<InfoBar> = None;
    for t in ticks {
        if !(t.price.is_finite() && t.volume.is_finite()) {
            continue;
        }
        match &mut cur {
            None => cur = Some(InfoBar::start(t)),
            Some(bar) => bar.push(t),
        }
        if let Some(bar) = &cur {
            if bar.volume >= threshold {
                bars.push(*bar);
                cur = None;
            }
        }
    }
    if let Some(bar) = cur {
        bars.push(bar); // flush the partial final bar
    }
    bars
}

/// Build **tick-imbalance bars**: a new bar closes once the absolute signed-volume
/// imbalance `θ = Σ bₜ·vₜ` (with the tick rule `bₜ = sign(Δprice)`) reaches
/// `imbalance_threshold`. This concentrates sampling exactly where informed
/// trading drives price.
pub fn tick_imbalance_bars(ticks: &[Tick], imbalance_threshold: f64) -> Vec<InfoBar> {
    let threshold = if imbalance_threshold.is_finite() && imbalance_threshold > 0.0 {
        imbalance_threshold
    } else {
        return Vec::new();
    };
    let mut bars = Vec::new();
    let mut cur: Option<InfoBar> = None;
    let mut theta = 0.0;
    let mut last_sign = 1.0f64;
    let mut prev_price: Option<f64> = None;
    for t in ticks {
        if !(t.price.is_finite() && t.volume.is_finite()) {
            continue;
        }
        let b = match prev_price {
            Some(p) if t.price > p => 1.0,
            Some(p) if t.price < p => -1.0,
            _ => last_sign, // tie → carry previous sign (López de Prado)
        };
        last_sign = b;
        prev_price = Some(t.price);
        theta += b * t.volume.max(0.0);

        match &mut cur {
            None => cur = Some(InfoBar::start(t)),
            Some(bar) => bar.push(t),
        }
        if theta.abs() >= threshold {
            if let Some(bar) = cur.take() {
                bars.push(bar);
            }
            theta = 0.0;
        }
    }
    if let Some(bar) = cur {
        bars.push(bar);
    }
    bars
}

/// "Time-dilation" intensity: information-bars formed **per chronological second**.
/// Rises sharply during crashes (volume floods in) — the engine then runs more
/// decision cycles per wall-clock second.
pub fn clock_intensity(bars: &[InfoBar]) -> f64 {
    if bars.len() < 2 {
        return 0.0;
    }
    let span = (bars.last().unwrap().end_ts - bars[0].start_ts).max(1e-9);
    bars.len() as f64 / span
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn ticks(prices: &[f64], vol: f64) -> Vec<Tick> {
        prices
            .iter()
            .enumerate()
            .map(|(i, p)| Tick {
                ts: i as f64,
                price: *p,
                volume: vol,
            })
            .collect()
    }

    #[test]
    fn volume_bars_close_at_threshold() {
        // 10 ticks × 100 volume = 1000; bars of 300 → 3 full + 1 partial(100).
        let t = ticks(
            &[10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0],
            100.0,
        );
        let bars = volume_bars(&t, 300.0);
        assert_eq!(bars.len(), 4);
        assert!(bars[0].volume >= 300.0);
        assert_eq!(bars[3].volume, 100.0); // partial flush
    }

    #[test]
    fn imbalance_bars_react_to_one_sided_flow() {
        // strictly rising → all +1 ticks → imbalance accrues fast
        let t = ticks(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 100.0);
        let bars = tick_imbalance_bars(&t, 250.0);
        assert!(!bars.is_empty());
    }

    #[test]
    fn crash_dilates_clock() {
        // Calm: low volume per tick → few bars/sec. Crash: high volume → many.
        let calm = volume_bars(&ticks(&[10.0; 20], 10.0), 100.0);
        let crash = volume_bars(&ticks(&[10.0; 20], 1000.0), 100.0);
        assert!(clock_intensity(&crash) > clock_intensity(&calm));
    }

    proptest! {
        #[test]
        fn never_panics(
            prices in proptest::collection::vec(prop_oneof![Just(f64::NAN), 0.1f64..1000.0], 0..200),
            vpb in 0.0f64..1e6,
        ) {
            let t: Vec<Tick> = prices.iter().enumerate().map(|(i,p)| Tick{ts:i as f64, price:*p, volume:100.0}).collect();
            let _ = volume_bars(&t, vpb);
            let _ = tick_imbalance_bars(&t, vpb);
        }
    }
}
