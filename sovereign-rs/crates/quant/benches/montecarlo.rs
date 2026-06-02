//! Throughput benchmark for the regime-switching Monte-Carlo.
//! Run with: `cargo bench -p sovereign-quant`
//! Compare kernels: `cargo bench -p sovereign-quant --no-default-features` (scalar).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, SeedableRng};

use sovereign_core::domain::Regime;
use sovereign_quant::regime::build_engine;

fn mc_throughput(c: &mut Criterion) {
    let history = vec![
        Regime::Bull,
        Regime::Bear,
        Regime::Sideways,
        Regime::Crisis,
        Regime::Recovery,
        Regime::Bull,
    ];
    let engine = build_engine(&history, 1.0).expect("engine builds");

    let mut group = c.benchmark_group("monte_carlo");
    group.sample_size(20);
    group.bench_function("50k_paths_21d", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(1);
            let res = engine.run(black_box(100.0), 21, 50_000, 0, 0.95, &mut rng);
            black_box(res.cvar)
        })
    });
    group.finish();
}

criterion_group!(benches, mc_throughput);
criterion_main!(benches);
