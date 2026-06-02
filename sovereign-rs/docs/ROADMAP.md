# Migration Roadmap

Iterative migration of "Sovereign Aeternum V331" to Rust. Status as of this build:
**12 crates, ~7,100 lines, 120 tests green, clippy `-D warnings` clean.**

## ✅ Done

### Foundation & quant core
- `core`: errors, telemetry (µs spans), config, domain enums, `ServiceRegistry` (DI).
- `quant`: Markov chain (MLE, n-step, stationary), regime-switching Monte-Carlo
  (SIMD `wide::f64x4` + scalar oracle, VaR/CVaR), Gaussian HMM (Viterbi).

### Market-neutral (V322) — `quant`
- Kalman hedge ratio, GARCH(1,1), Engle-Granger cointegration + ADF + OU half-life,
  historical/parametric CVaR, Black-Litterman optimizer (iterative water-filling cap),
  StatArb pair evaluation.

### Data plane — `data`
- `DataProvider` trait, error-aware `CircuitBreaker` (DNS/refused/timeout backoff),
  `FallbackChain` waterfall (degraded conviction), hardened `HttpClient` (reqwest+rustls),
  parsers for **Stooq** (CSV), **CBOE** (VIX/VVIX JSON), **SEC-EDGAR** (filings, serde),
  reusable `MockProvider`. *(Parsers tested offline; live fetch needs the network.)*

### Signals / risk / execution / ml
- `signals`: BFT consensus (regime-adaptive quorum + veto), 12-engine alt-data scoring.
- `risk`: Kill-House gate pipeline (Conviction / Monte-Carlo CVaR / VPIN), BlackSwanGuard, DrawdownLadder.
- `execution`: contract router (Stock/Future/Forex/Crypto + front-month), orders/brackets,
  inverse-vol sizing, T+1 settlement carousel, commission gate.
- `ml`: `Model` trait + native logistic regression (GD + standardization).

### ⚛ V331 "Gray-Area" physics layer
- `microstructure`: information-driven clocks (volume / tick-imbalance bars, time dilation),
  Hurst exponent + microstructure Reynolds number + laminar/turbulent flow regime.
- `guards`: lock-free `GlobalKillSwitch`, `FaultBoundary` (`catch_unwind`), `AxiomBreaker`
  (Gödel/Turing entropy halting), `ThermodynamicGuard` (thermal-aware sim-depth degradation).

### Orchestration
- `engine`: per-candidate decision pipeline, `SelfHealer` tokio task.
- `cli`: mimalloc allocator, tracing, `demo` + `physics` + `mc` subcommands.

## ☐ Remaining (mostly environment-dependent)

- **M3 Columnar I/O**: Polars `LazyFrame` load of `sovereign_hive_cache.parquet`;
  extend features 9 → 24 (V319/V320 block: `mom21/63`, OBV slope, daily-resample `rsi14_d`/`macd_d`).
- **Live data**: DNS-hardened fetch wired into providers; FRED / Open-Meteo / Adzuna
  free-tier engines (rate-limit arbitrage already modeled by the circuit breaker).
- **M5 live execution**: IBKR TWS/Gateway adapter (needs credentials + a running gateway).
- **M7 ensemble**: load the purged-CV RF/LightGBM/XGBoost models as ONNX via `tract`
  (implements the same `Model` trait).
- **M8 live loop**: full async orchestration with `tokio::sync::mpsc` message passing,
  scheduler, JSON-telemetry dashboard, deterministic replay.
- **CI/ops**: `cargo deny`, coverage, containerization, GitHub Actions.

These need your machine (IBKR creds, live network, trained model files) — which is why
they're staged rather than stubbed-as-done.
