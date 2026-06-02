# Sovereign-RS

> Ultra-low-latency, memory-safe **Rust** migration of the Python "Sovereign
> Aeternum V331 Citadel Edition" quant trading system — re-architected to
> eliminate the GIL, GC pauses and RAM fragmentation, and upgraded with a
> "Gray-Area" physics/math layer (information-driven clocks, microstructure
> turbulence, Gödel/Turing halting, thermodynamic throttling).

**Status:** 17-crate Cargo workspace · **~10,600 lines** · **181 tests passing**
(unit + adversarial proptest) · `clippy -D warnings` clean · cache-aligned +
core-pinned · runnable.

```bash
cargo run -p sovereign-cli -- demo        # Markov → Monte-Carlo → 6-stage cascade → decision
cargo run -p sovereign-cli -- physics     # turbulence · info-clock · axiom breaker · thermal · kill switch
cargo run -p sovereign-cli -- scan --capital 500   # adaptive scan over the 11k+ universe
cargo test --workspace                    # 165 tests
```

👉 **New to it? Read [`docs/USAGE.md`](docs/USAGE.md)** (step-by-step, in Bahasa Indonesia):
install, env config (Gemini/Grok/IBKR keys), capital tiers, scanning, extending.

---

## Directives — and where each lives

| Directive | Status | Where |
|-----------|--------|-------|
| No "Pythonic"/"Lazy" Rust; ownership, **no `Arc<Mutex>` overuse** | ✅ | lock-free `GlobalKillSwitch` (atomics), enums, trait objects, DI registry |
| **tokio** async, no blocking in runtime | ✅ | `data` (async providers, fallback), `engine::SelfHealer` |
| **polars** + **ndarray** + SIMD | ◑ | ndarray indicators + `wide::f64x4` Monte-Carlo; Polars/Parquet I/O = M3 |
| No `unwrap`/`expect`; `thiserror`/`anyhow` | ✅ | `core::error::SovereignError` |
| **serde / serde_json** (SEC-EDGAR, APIs) | ✅ | `data::{sec_edgar,cboe}`, config, DTOs |
| **mimalloc** global allocator | ✅ | `cli/src/main.rs` |
| **tracing** (µs spans), not `println!` | ✅ | `core::telemetry` (`LatencyGuard`) |
| **SIMD** Monte-Carlo / Markov | ✅ | `quant::montecarlo` (`wide::f64x4`, stable Rust) |
| **proptest** adversarial testing | ✅ | every math/state module |
| **Markov chain** + HMM Viterbi | ✅ | `quant::{markov,hmm}` |
| **Information-Driven Clocks** (relativity) | ✅ | `microstructure::clock` (volume / tick-imbalance bars; time dilation) |
| **Navier-Stokes turbulence** (Hurst/Reynolds) | ✅ | `microstructure::turbulence` |
| **Gödel/Turing halting** (Axiom Breaker) | ✅ | `guards::axiom` (entropy → hedge) |
| **Thermodynamic halting** (thermal throttle) | ✅ | `guards::thermal` (degrade sim depth) |
| Preserve the **Soul** (KillSwitch, FaultBoundary, BFT) | ✅ | `guards::{killswitch,fault}`, `signals::consensus` |
| Free-tier feeds, rate-limit arbitrage | ✅ | `data::{circuit_breaker,fallback}` (error-aware backoff + waterfall) |

`◑` partial — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Workspace (16 crates, dependencies point inward)

```
sovereign-core            errors · telemetry(µs) · config · domain enums · ServiceRegistry(DI)
   ▲
   ├── sovereign-quant         Markov · regime-switch Monte-Carlo(SIMD) · HMM · Kalman · GARCH
   │                           · cointegration · CVaR · Black-Litterman · StatArb
   ├── sovereign-anomaly       Kelly · PCA · RMT(Marchenko-Pastur) · Hawkes · copula · KL · FFT · Lyapunov
   ├── sovereign-physics       Newton kinematics · OU · Itô · Fokker-Planck · Markowitz · Lotka-Volterra · Rule-30 · Schrödinger cloud
   ├── sovereign-microstructure  ⚛ info-clocks (volume/imbalance bars) · Hurst · Reynolds · flow regime
   ├── sovereign-guards          ⚛ GlobalKillSwitch · FaultBoundary · AxiomBreaker · ThermodynamicGuard
   ├── sovereign-data          DataProvider · CircuitBreaker · FallbackChain · Stooq/CBOE/SEC-EDGAR · HTTP
   ├── sovereign-features      vectorized indicators (ndarray)
   ├── sovereign-universe      11k+ multi-asset universe · capital-tier-adaptive round-robin scan
   ├── sovereign-signals       BFT consensus · agents · 12-engine alt-data scoring
   ├── sovereign-risk          6-stage HyperLayered cascade · Kill-House · guards · cache-line aligned
   ├── sovereign-ml            Model trait · native logistic regression (ONNX ensemble later)
   ├── sovereign-llm           Gemini / Grok / Ollama clients · key-rotation pool (keys from env)
   ├── sovereign-execution     contract router · orders/brackets · inverse-vol sizing · T+1 settlement
   ├── sovereign-broker        Broker trait · PaperBroker · IbkrBroker (account from env)
   ├── sovereign-engine        Adaptive Capital Protocol · core-pinned IsolatedExecutor · SelfHealer
   └── sovereign-cli           binary `sovereign`: mimalloc + tracing + demos (demo/physics/scan/mc)
```

## The V331 "Gray-Area" physics layer (`cargo run -p sovereign-cli -- physics`)

* **Information-Driven Clocks** (`microstructure::clock`) — sample by *information*
  not wall-clock: volume bars & tick-imbalance bars (López de Prado). `clock_intensity`
  rises ~10× during a volume flood, so the engine runs more decision cycles per
  second exactly when it matters — *time dilates in the crash*.
* **Navier-Stokes turbulence** (`microstructure::turbulence`) — the **Hurst
  exponent** (R/S analysis) and a **microstructure Reynolds number** classify flow
  as Laminar / Transitional / **Turbulent**; turbulence flips the engine from
  directional trading to market-making.
* **Gödel/Turing Axiom Breaker** (`guards::axiom`) — when the BFT panel's decision
  **entropy** approaches maximum (undecidable, like the halting problem), it
  short-circuits prediction and defaults to volatility hedging instead of trusting
  a coin-flip.
* **Thermodynamic Guard** (`guards::thermal`) — reads the CPU die temperature and
  degrades Monte-Carlo depth (10 000 → 1 000 paths) before the hardware throttles —
  the Guerilla Protocol for a fanless ThinkPad.

## How the Python design maps to Rust

| Python (Sovereign V3xx) | Rust |
|--------------------------|------|
| `service_registry.py` + `apply_*()` `__main__` monkey-patching | `core::registry::ServiceRegistry` (typed DI) |
| stringly-typed tickers (`ES=F`,`EURUSD=X`,`^VIX`) | `core::domain::Instrument` enum |
| `HMMRegime`, Viterbi, `MonteCarlo100k` | `quant::{hmm,markov,montecarlo}` |
| V322 StatArb/Kalman/GARCH/Black-Litterman/CVaR | `quant::{statarb,kalman,garch,blacklitterman,cvar,cointegration}` |
| `CircuitBreakerV322`, `MacroOracle` waterfall | `data::{circuit_breaker,fallback}` |
| `StooqEngine`,`_cboe_fetch`,`SECEdgarPipeline` | `data::{stooq,cboe,sec_edgar}` |
| `BFTConsensus`+`_BFT_QUORUM_MAP`, 12-engine `AlternativeDataEngine` | `signals::{consensus,altdata}` |
| `run_kill_house_v312`, `BlackSwanGuard`, `DrawdownLadder` | `risk::{gate,gates,guards}` |
| `IBKR_CONTRACT_MAP`, `InverseVolSizer`, `SettlementCarousel`, `CommissionGate` | `execution::*` |
| `GlobalKillSwitch`, `FaultBoundary`, `SovereignAutoDoctor` | `guards::{killswitch,fault}`, `engine::SelfHealer` |

## Using it in VS Code

Install the recommended extensions (auto-prompt): **rust-analyzer**, **CodeLLDB**,
**Even Better TOML**. Open the repo root *or* `sovereign-rs/` — the root
`.vscode/settings.json` points rust-analyzer at the workspace. `Ctrl/Cmd+Shift+B`
builds; the Test Explorer runs the suite; `F5` debugs the demo. Toolchain is pinned
in `rust-toolchain.toml`.

## Commands

| Task | Command |
|------|---------|
| End-to-end demo | `cargo run -p sovereign-cli -- demo` |
| Physics layer demo | `cargo run -p sovereign-cli -- physics` |
| Monte-Carlo only | `cargo run -p sovereign-cli -- mc --paths 100000 --horizon 21` |
| Test (incl. proptest) | `cargo test --workspace` |
| Lint (strict) | `cargo clippy --workspace --all-targets -- -D warnings` |
| Benchmark | `cargo bench -p sovereign-quant` |
| Scalar (non-SIMD) | `cargo build -p sovereign-quant --no-default-features` |

## License

MIT OR Apache-2.0.

