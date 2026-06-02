# Panduan Pakai Sovereign-RS

Panduan praktis (Bahasa Indonesia) untuk menjalankan, mengonfigurasi, dan
mengembangkan engine. Perintah & kode tetap dalam bahasa aslinya.

---

## 1. Prasyarat (sekali saja)

1. Install Rust → <https://rustup.rs> (toolchain sudah ke-pin lewat `rust-toolchain.toml`).
2. Extract `sovereign-rs.tar.gz`, lalu:
   ```bash
   cd sovereign-rs
   cargo build              # build semua crate (perlu internet sekali untuk dependency)
   cargo test --workspace   # 165 test harus lulus
   ```
3. Buka folder `sovereign-rs/` di **VS Code** → install ekstensi yang ditawarkan
   (rust-analyzer, CodeLLDB, Even Better TOML). Tombol Run/Debug & Test Explorer langsung jalan.

---

## 2. Perintah CLI (binary `sovereign`)

```bash
cargo run -p sovereign-cli -- demo        # Markov → Monte-Carlo → 6-stage cascade → keputusan
cargo run -p sovereign-cli -- physics     # turbulensi, info-clock, axiom breaker, thermal, kill switch
cargo run -p sovereign-cli -- scan --capital 500       # scan 11k+ universe (tier NANO)
cargo run -p sovereign-cli -- scan --capital 50000000  # (tier LARGE)
cargo run -p sovereign-cli -- mc --paths 100000 --horizon 21   # Monte-Carlo VaR/CVaR

# Telemetri JSON (untuk dashboard / log shipper):
cargo run -p sovereign-cli -- --json demo
```

Saat start, engine otomatis **pin ke core terisolasi** (lihat baris `core isolation`)
dan memakai **mimalloc** allocator. Semua output adalah event `tracing` ber-struktur
(bukan `println!`), lengkap dengan `latency_us` di jalur kritis.

---

## 3. Konfigurasi rahasia (API key & akun) — lewat ENV, tidak pernah di-hardcode

Set environment variable sebelum menjalankan. **Tidak ada key di dalam kode.**

```bash
# ── LLM (WarRoom brains) ────────────────────────────────────────────────
export GEMINI_API_KEY="AIza..."             # atau beberapa key dipisah koma:
export GEMINI_API_KEYS="key1,key2,key3"     # → rotasi otomatis (KeyPool) saat kena rate-limit
export GROK_API_KEY="xai-..."               # (atau XAI_API_KEY)
export OLLAMA_HOST="http://localhost:11434" # brain offline (default)
export GEMINI_MODEL="gemini-2.5-flash"      # opsional override
export GROK_MODEL="grok-2"

# ── Akun IBKR ───────────────────────────────────────────────────────────
export IBKR_HOST="127.0.0.1"
export IBKR_PORT="7497"        # 7497=TWS paper, 7496=TWS live, 4002/4001=IB Gateway paper/live
export IBKR_CLIENT_ID="1"
export IBKR_ACCOUNT="DU1234567"
```

Cara baca di kode:
```rust
let llm_cfg = sovereign_llm::LlmConfig::from_env();       // resolve semua key dari env
let primary  = llm_cfg.build_primary()?;                  // Gemini → Grok → Ollama (fallback)
let ibkr_cfg = sovereign_broker::IbkrConfig::from_env();  // host/port/client_id/account
```

> Catatan jujur: panggilan LLM live & order IBKR live butuh jaringan + kredensialmu +
> TWS/IB Gateway yang berjalan. Tanpa itu, `IbkrBroker.connect()` mengembalikan error
> bertipe (bukan panic), dan kamu pakai `PaperBroker` untuk dry-run.

Tuning non-rahasia ada di `config.toml` (lihat `config.example.toml`).

---

## 4. Modal Adaptif (micro → large) — semuanya jalan & menyesuaikan

Tier ditentukan otomatis dari besar modal (`CapitalTier::from_capital`):

| Tier | Modal | scan_depth | max_positions | capital_mass (cascade) |
|------|-------|-----------|---------------|------------------------|
| NANO | < $2k | 120 | 3 | < 1 → dead-zone sempit (hyper-agresif) |
| MICRO | < $25k | 250 | 6 | < 1 |
| SMALL | < $100k | 500 | 12 | ≈ 1 |
| MEDIUM | < $1M | 900 | 25 | > 1 |
| LARGE | ≥ $1M | 1.600 | 60 | > 1 → dead-zone lebar + **order di-split** |

Adaptif berarti **tidak ada angka kaku**: ambang gate dihitung relatif terhadap
volatilitas & fitness hari itu (`DayOfBounds`) dikali `capital_mass`. Coba sendiri:
```bash
cargo run -p sovereign-cli -- scan --capital 500       # NANO
cargo run -p sovereign-cli -- scan --capital 5000000   # LARGE → split_orders=true
```

---

## 5. Scan seluruh 11.000+ saham & ETF

`Universe` = overlay multi-asset (ETF sektor, futures, forex, crypto, internasional)
**+** daftar ~11k ekuitas yang dimuat saat runtime. `RoundRobin` menjamin **setiap**
simbol kebagian giliran dievaluasi (tidak ada yang terlewat).

```rust
use sovereign_universe::{Universe, RoundRobin, CapitalTier};

let tier = CapitalTier::from_capital(100_000.0);
// Di produksi: muat daftar ekuitas nyata dari file/feed:
let universe = Universe::multi_asset().with_equities(load_my_11k_tickers());
let mut rr = RoundRobin::new();
loop {
    let batch = rr.next_batch(tier.scan_depth(), universe.master()); // borrow, zero-copy
    for symbol in batch { /* fetch → features → signals → kill-house → decision */ }
    // rr.coverage(universe.len()) ≥ 1.0 artinya satu putaran penuh sudah selesai
}
```
Demo memakai `with_synthetic_equities(11_000)` untuk membuktikan cakupan penuh offline.

---

## 6. Memakai modul matematika (sebagai library)

```rust
use sovereign_anomaly::{kelly_fraction, largest_lyapunov, dominant_period, tail_dependence};
use sovereign_anomaly::{pca, marchenko_pastur_bounds, kl_divergence, Hawkes};

let f      = kelly_fraction(0.6, 1.5);              // bet fraction optimal
let lam    = largest_lyapunov(&returns, 3, 1);      // eksponen chaos (>0 = tak terprediksi)
let period = dominant_period(&prices);              // siklus dominan (FFT)
let (lo,_) = tail_dependence(&a, &b, 0.05);         // probabilitas crash bareng
let factors= pca(&returns_matrix);                  // faktor laten pasar
let (_,hi) = marchenko_pastur_bounds(0.2, 1.0);     // batas noise RMT
let drift  = kl_divergence(&pred_dist, &real_dist); // model vs realita

use sovereign_quant::{TransitionMatrix, RegimeSwitchingMonteCarlo, evaluate_pair};
use sovereign_microstructure::{classify, volume_bars};
use sovereign_guards::{AxiomBreaker, ThermodynamicGuard, GlobalKillSwitch};
```

---

## 7. Eksekusi order (paper dulu, IBKR nanti)

```rust
use sovereign_broker::{Broker, PaperBroker, IbkrBroker};
use sovereign_execution::{Order, BracketOrder};
use sovereign_core::domain::Side;

let paper = PaperBroker::new(10_000.0);
paper.set_mark("NVDA", 120.0);
let id = paper.submit(&Order::market("NVDA", Side::Buy, 10)).await?;
let acct = paper.account().await?;   // cash, equity, buying_power

// Saat siap live (TWS/Gateway jalan):
let mut ibkr = IbkrBroker::from_env();
ibkr.connect().await?;               // sekarang masih return error sampai adapter live diaktifkan
```

---

## 8. Cara mengembangkan (extend)

- **Tambah gate Kill-House**: implement trait `sovereign_risk::Gate` → masukkan ke `KillHouse::new(vec![...])`.
- **Tambah agen BFT**: implement `sovereign_signals::Agent` → masukkan ke `BftConsensus`.
- **Tambah data provider**: implement `sovereign_data::DataProvider` → daftarkan di `FallbackChain`.
- **Tambah brain LLM**: implement `sovereign_llm::LlmClient`.
- **Injeksi dependency**: pakai `sovereign_core::ServiceRegistry` (bukan global state).

---

## 9. Peta crate (16)

```
core · quant · anomaly · microstructure · guards          ← matematika & primitives
data · features · universe                                 ← data plane & scanning
signals · risk · ml · llm                                  ← sinyal & keputusan
execution · broker                                         ← eksekusi & order routing
engine · cli                                               ← orkestrasi & binary
```

## 10. Perintah harian

| Tujuan | Perintah |
|--------|----------|
| Build | `cargo build` |
| Test | `cargo test --workspace` |
| Lint ketat | `cargo clippy --workspace --all-targets -- -D warnings` |
| Format | `cargo fmt --all` |
| Benchmark MC | `cargo bench -p sovereign-quant` |
| Jalankan demo | `cargo run -p sovereign-cli -- demo` |

Selamat memakai. Mulai dari **paper trading** dulu sebelum modal nyata. 🚦
