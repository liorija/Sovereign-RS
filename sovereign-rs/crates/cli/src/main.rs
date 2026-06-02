//! Sovereign-RS command-line entry point.
//!
//! Directive compliance, all visible here:
//! * **Custom global allocator** — `mimalloc` is installed below to keep
//!   allocation fast and fragmentation-free under heavy streaming.
//! * **`tracing`, not `println!`** — every line of output is a structured span/event.
//! * Exercises the **Markov chain**, the **SIMD regime-switching Monte-Carlo**,
//!   and the full **decision pipeline**.

// ── Custom global allocator (mimalloc) ───────────────────────────────────────
// One line, but it changes every allocation in the process. Swap to jemalloc on
// Linux by gating this behind a feature if you prefer.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rand::{rngs::StdRng, SeedableRng};
use tracing::info;

use sovereign_core::domain::Regime;
use sovereign_core::telemetry::{self, LatencyGuard};
use sovereign_engine::Pipeline;
use sovereign_quant::regime::{build_engine, transition_from_regimes};

#[derive(Parser, Debug)]
#[command(name = "sovereign", version, about = "Sovereign-RS HFT quant engine")]
struct Cli {
    /// Emit JSON telemetry instead of human-readable.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the end-to-end demo (Markov → Monte-Carlo → decision pipeline).
    Demo,
    /// Run a standalone regime-switching Monte-Carlo and print VaR/CVaR.
    Mc {
        #[arg(long, default_value_t = 100.0)]
        spot: f64,
        #[arg(long, default_value_t = 21)]
        horizon: usize,
        #[arg(long, default_value_t = 50_000)]
        paths: usize,
    },
    /// Showcase the V331 physics layer: turbulence, information clock, axiom
    /// breaker, thermodynamic guard, and the global kill switch.
    Physics,
    /// Show capital-tier-adaptive scanning over the full 11k+ multi-asset universe.
    Scan {
        #[arg(long, default_value_t = 100_000.0)]
        capital: f64,
    },
}

/// A representative observed regime history (would come from the HMM in prod).
fn sample_history() -> Vec<Regime> {
    use Regime::*;
    vec![
        Bull, Bull, Goldilocks, Bull, Sideways, Sideways, Bear, Crisis, Recovery, Bull, Bull,
        Reflation, Sideways, Bull, Goldilocks,
    ]
}

fn run_mc(spot: f64, horizon: usize, paths: usize) -> Result<()> {
    let history = sample_history();
    let engine = build_engine(&history, 1.0)?;

    let _g = LatencyGuard::info("monte_carlo_run");
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let res = engine.run(spot, horizon, paths, Regime::Bull.index(), 0.95, &mut rng);

    info!(
        spot,
        horizon,
        paths,
        expected_return = format_args!("{:.4}", res.expected_return),
        var_95 = format_args!("{:.4}", res.var),
        cvar_95 = format_args!("{:.4}", res.cvar),
        worst = format_args!("{:.4}", res.worst),
        prob_loss = format_args!("{:.3}", res.prob_loss),
        "regime-switching Monte-Carlo complete"
    );
    Ok(())
}

fn run_demo() -> Result<()> {
    let history = sample_history();

    // 1) Markov chain: estimate transitions and the stationary regime mix.
    let tm = transition_from_regimes(&history, 1.0)?;
    let stationary = tm.stationary_distribution(2000, 1e-12);
    for (r, p) in Regime::ALL.iter().zip(&stationary) {
        info!(
            regime = r.as_str(),
            prob = format_args!("{:.3}", p),
            "stationary regime weight"
        );
    }

    // 2) Monte-Carlo tail risk.
    run_mc(100.0, 7, 50_000)?;

    // 3) Full decision pipeline on a synthetic bullish candidate.
    let pipe = Pipeline::default_from_history(&history)?;
    let closes: Vec<f64> = (0..80).map(|i| 100.0 + i as f64 * 0.15).collect();
    let volumes = vec![1.0e6; 80];
    let decision = pipe.evaluate(
        "NVDA",
        Regime::Bull,
        closes,
        volumes,
        13.0,
        0.35,
        0.25,
        false,
        true,
    );

    for (name, outcome) in &decision.killhouse.outcomes {
        info!(gate = name, passed = outcome.passed, detail = %outcome.label, "kill-house gate");
    }
    info!(detail = %decision.consensus.reason, "bft consensus");
    info!(
        ticker = %decision.ticker,
        approved = decision.approved,
        gates = format_args!("{}/{}", decision.killhouse.passed, decision.killhouse.total),
        cascade = decision.cascade_state.as_str(),
        capital_mass = format_args!("{:.3}", decision.capital_mass),
        "DECISION"
    );

    // Adaptive Capital Protocol — same candidate, three capital sizes.
    for cap in [1_000.0, 100_000.0, 50_000_000.0] {
        let p = Pipeline::default_from_history(&history)?.with_capital_size(cap);
        let closes: Vec<f64> = (0..80).map(|i| 100.0 + i as f64 * 0.15).collect();
        let volumes = vec![1.0e6; 80];
        let d = p.evaluate(
            "NVDA",
            Regime::Bull,
            closes,
            volumes,
            13.0,
            0.35,
            0.25,
            false,
            true,
        );
        info!(
            capital = format_args!("${:.0}", cap),
            capital_mass = format_args!("{:.3}", d.capital_mass),
            split_orders = d.split_orders,
            cascade = d.cascade_state.as_str(),
            "adaptive capital protocol"
        );
    }
    Ok(())
}

/// Show that scanning covers the whole 11k+ universe and adapts to capital tier.
fn run_scan(capital: f64) -> Result<()> {
    use sovereign_universe::{CapitalTier, RoundRobin, Universe};

    let tier = CapitalTier::from_capital(capital);
    let universe = Universe::multi_asset().with_synthetic_equities(11_000);
    info!(
        capital = format_args!("${:.0}", capital),
        tier = tier.as_str(),
        scan_depth = tier.scan_depth(),
        max_positions = tier.max_positions(),
        universe_size = universe.len(),
        "scan configuration (adaptive to capital tier)"
    );

    let mut rr = RoundRobin::new();
    let cycles_to_cover = universe.len() / tier.scan_depth() + 1;
    for cycle in 0..3 {
        let batch = rr.next_batch(tier.scan_depth(), universe.master());
        info!(
            cycle,
            batch = batch.len(),
            first = batch.first().copied().unwrap_or(""),
            last = batch.last().copied().unwrap_or(""),
            "scan cycle"
        );
    }
    info!(
        cycles_for_full_coverage = cycles_to_cover,
        "every one of the {} symbols is evaluated within {} cycles",
        universe.len(),
        cycles_to_cover
    );
    Ok(())
}

/// Demonstrate the V331 "Gray-Area" physics/math layer end-to-end.
fn run_physics() -> Result<()> {
    use sovereign_guards::{
        read_cpu_temp_celsius, AxiomBreaker, GlobalKillSwitch, KillReason, ThermodynamicGuard,
    };
    use sovereign_microstructure::{classify, clock_intensity, volume_bars, Tick};

    // 1) Navier-Stokes turbulence — calm (laminar) vs panic (turbulent) flow.
    let calm: Vec<f64> = (0..400).map(|i| (i as f64 * 0.1).sin() * 0.003).collect();
    let mut panic_flow = vec![0.001f64; 200];
    panic_flow.extend((0..200).map(|i| if i % 2 == 0 { 0.05 } else { -0.05 }));
    for (name, series) in [("calm", &calm), ("panic", &panic_flow)] {
        let t = classify(series);
        info!(
            tape = name,
            hurst = format_args!("{:.3}", t.hurst),
            reynolds = format_args!("{:.2}", t.reynolds),
            regime = ?t.regime,
            market_making = t.regime.prefer_market_making(),
            "microstructure turbulence"
        );
    }

    // 2) Information clock — time dilates when volume floods in (crash).
    let calm_ticks: Vec<Tick> = (0..100)
        .map(|i| Tick {
            ts: i as f64,
            price: 100.0,
            volume: 10.0,
        })
        .collect();
    let crash_ticks: Vec<Tick> = (0..100)
        .map(|i| Tick {
            ts: i as f64,
            price: 100.0,
            volume: 2000.0,
        })
        .collect();
    info!(
        calm = format_args!("{:.3}", clock_intensity(&volume_bars(&calm_ticks, 100.0))),
        crash = format_args!("{:.3}", clock_intensity(&volume_bars(&crash_ticks, 100.0))),
        "information clock (info-bars/sec — perception of time dilates in the crash)"
    );

    // 3) Gödel/Turing axiom breaker — a maximally split panel halts → hedge.
    let ab = AxiomBreaker::default();
    info!(
        entropy = format_args!("{:.3}", ab.divergence(&[1.0, 1.0, 1.0, 1.0])),
        halt = ab.should_halt(&[1.0, 1.0, 1.0, 1.0]),
        "axiom breaker (split panel)"
    );
    info!(
        entropy = format_args!("{:.3}", ab.divergence(&[0.9, 0.05, 0.05])),
        halt = ab.should_halt(&[0.9, 0.05, 0.05]),
        "axiom breaker (decisive panel)"
    );

    // 4) Thermodynamic guard — degrade Monte-Carlo depth as the die heats.
    let tg = ThermodynamicGuard::default();
    for temp in [60.0, 85.0, 99.0] {
        info!(
            cpu_c = temp,
            sim_depth = tg.adaptive_depth(10_000, temp),
            "thermodynamic guard"
        );
    }
    info!(live_cpu_c = ?read_cpu_temp_celsius(), live_depth = tg.current_depth(10_000), "thermodynamic guard (live sensor)");

    // 5) Lock-free global kill switch.
    let ks = GlobalKillSwitch::new();
    ks.trip(KillReason::Drawdown);
    info!(
        tripped = ks.is_tripped(),
        reason = ks.reason().as_str(),
        "global kill switch"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.json {
        telemetry::init_json();
    } else {
        telemetry::init();
    }

    info!(
        version = env!("CARGO_PKG_VERSION"),
        allocator = "mimalloc",
        "sovereign-rs starting"
    );

    // Guerilla Protocol: pin this thread to the isolated core (graceful fallback).
    let iso = sovereign_engine::pin_to_last_core();
    info!(pinned = iso.pinned, core = ?iso.core, total_cores = iso.total_cores, "core isolation");

    match cli.cmd {
        Cmd::Demo => run_demo()?,
        Cmd::Mc {
            spot,
            horizon,
            paths,
        } => run_mc(spot, horizon, paths)?,
        Cmd::Physics => run_physics()?,
        Cmd::Scan { capital } => run_scan(capital)?,
    }
    Ok(())
}
