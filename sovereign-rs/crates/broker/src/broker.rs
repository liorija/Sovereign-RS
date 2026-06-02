//! Broker abstraction + paper and IBKR implementations.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::Serialize;

use sovereign_core::domain::Side;
use sovereign_core::error::{Result, SovereignError};
use sovereign_execution::order::{Order, OrderType};

/// A broker-assigned order id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OrderId(pub u64);

/// Account snapshot.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AccountSummary {
    pub cash: f64,
    pub equity: f64,
    pub buying_power: f64,
}

/// An open position.
#[derive(Debug, Clone, Serialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
}

/// A broker the engine can route orders through.
#[async_trait]
pub trait Broker: Send + Sync {
    fn name(&self) -> &str;
    fn is_connected(&self) -> bool;
    async fn submit(&self, order: &Order) -> Result<OrderId>;
    async fn account(&self) -> Result<AccountSummary>;
    async fn positions(&self) -> Result<Vec<Position>>;
}

// ── Paper broker (in-memory, deterministic) ─────────────────────────────────

#[derive(Debug)]
struct PaperState {
    cash: f64,
    next_id: u64,
    positions: HashMap<String, Position>,
    marks: HashMap<String, f64>,
}

/// A fully in-memory paper-trading broker for backtests / dry-runs.
/// (Uses one `Mutex` for the account ledger — a legitimate, non-hot-path use.)
#[derive(Debug)]
pub struct PaperBroker {
    state: Mutex<PaperState>,
}

impl PaperBroker {
    /// New paper account seeded with `starting_cash`.
    pub fn new(starting_cash: f64) -> Self {
        Self {
            state: Mutex::new(PaperState {
                cash: starting_cash.max(0.0),
                next_id: 1,
                positions: HashMap::new(),
                marks: HashMap::new(),
            }),
        }
    }

    /// Set the mark price used to fill market orders.
    pub fn set_mark(&self, symbol: &str, price: f64) {
        if let Ok(mut s) = self.state.lock() {
            s.marks.insert(symbol.to_string(), price);
        }
    }

    fn fill_price(state: &PaperState, order: &Order) -> Option<f64> {
        match order.order_type {
            OrderType::Limit(p) if p > 0.0 => Some(p),
            _ => state.marks.get(&order.symbol).copied().filter(|p| *p > 0.0),
        }
    }
}

#[async_trait]
impl Broker for PaperBroker {
    fn name(&self) -> &str {
        "paper"
    }
    fn is_connected(&self) -> bool {
        true
    }

    async fn submit(&self, order: &Order) -> Result<OrderId> {
        let mut s = self
            .state
            .lock()
            .map_err(|_| SovereignError::data("paper", "ledger lock poisoned"))?;
        let price = Self::fill_price(&s, order).ok_or_else(|| {
            SovereignError::data("paper", "no fill price (set a mark or use a limit)")
        })?;
        let signed = match order.side {
            Side::Buy => order.quantity as f64,
            Side::Sell | Side::Short => -(order.quantity as f64),
        };
        let notional = signed * price;
        s.cash -= notional;
        let entry = s
            .positions
            .entry(order.symbol.clone())
            .or_insert_with(|| Position {
                symbol: order.symbol.clone(),
                quantity: 0.0,
                avg_price: price,
            });
        // Weighted-average entry on adds (sign-aware; no float-equality compares).
        let new_qty = entry.quantity + signed;
        let same_direction = signed * entry.quantity > 0.0;
        if same_direction {
            let cost = entry.avg_price * entry.quantity + price * signed;
            entry.avg_price = if new_qty.abs() > 1e-12 {
                cost / new_qty
            } else {
                price
            };
        } else if entry.quantity.abs() < 1e-12 {
            entry.avg_price = price;
        }
        entry.quantity = new_qty;
        let id = s.next_id;
        s.next_id += 1;
        tracing::info!(broker = "paper", id, symbol = %order.symbol, qty = signed, price, "filled");
        Ok(OrderId(id))
    }

    async fn account(&self) -> Result<AccountSummary> {
        let s = self
            .state
            .lock()
            .map_err(|_| SovereignError::data("paper", "ledger lock poisoned"))?;
        let positions_value: f64 = s
            .positions
            .values()
            .map(|p| p.quantity * s.marks.get(&p.symbol).copied().unwrap_or(p.avg_price))
            .sum();
        let equity = s.cash + positions_value;
        Ok(AccountSummary {
            cash: s.cash,
            equity,
            buying_power: s.cash.max(0.0),
        })
    }

    async fn positions(&self) -> Result<Vec<Position>> {
        let s = self
            .state
            .lock()
            .map_err(|_| SovereignError::data("paper", "ledger lock poisoned"))?;
        Ok(s.positions
            .values()
            .filter(|p| p.quantity.abs() > 1e-9)
            .cloned()
            .collect())
    }
}

// ── IBKR adapter (config preserved; live socket gated) ──────────────────────

/// IBKR connection config — read from the environment, never hardcoded.
#[derive(Debug, Clone)]
pub struct IbkrConfig {
    pub host: String,
    pub port: u16,
    pub client_id: i32,
    pub account: Option<String>,
}

impl Default for IbkrConfig {
    fn default() -> Self {
        // 7497 = TWS paper, 7496 = TWS live, 4002/4001 = IB Gateway paper/live.
        Self {
            host: "127.0.0.1".into(),
            port: 7497,
            client_id: 1,
            account: None,
        }
    }
}

impl IbkrConfig {
    /// Resolve from `IBKR_HOST` / `IBKR_PORT` / `IBKR_CLIENT_ID` / `IBKR_ACCOUNT`.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            host: std::env::var("IBKR_HOST")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or(d.host),
            port: std::env::var("IBKR_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d.port),
            client_id: std::env::var("IBKR_CLIENT_ID")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d.client_id),
            account: std::env::var("IBKR_ACCOUNT").ok().filter(|s| !s.is_empty()),
        }
    }
}

/// IBKR broker adapter. The config (your account / host / port) is preserved;
/// the live TWS/Gateway socket implementation is gated behind the `ibkr-live`
/// feature and a running gateway, so this build never silently "pretends" to
/// trade — it returns a typed `Halted` error until connected.
#[derive(Debug)]
pub struct IbkrBroker {
    pub config: IbkrConfig,
    connected: bool,
}

impl IbkrBroker {
    pub fn new(config: IbkrConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }
    pub fn from_env() -> Self {
        Self::new(IbkrConfig::from_env())
    }
    /// Attempt to connect. The real socket adapter is a future milestone; until
    /// then this returns a clear, non-panicking error.
    pub async fn connect(&mut self) -> Result<()> {
        Err(SovereignError::Halted(format!(
            "IBKR live adapter not enabled (host={}:{}, client_id={}, account={:?}). \
             Run TWS/IB Gateway and build with the `ibkr-live` feature.",
            self.config.host, self.config.port, self.config.client_id, self.config.account
        )))
    }
    fn not_connected(&self) -> SovereignError {
        SovereignError::Halted(
            "IBKR broker not connected — call connect() with a live gateway".into(),
        )
    }
}

#[async_trait]
impl Broker for IbkrBroker {
    fn name(&self) -> &str {
        "ibkr"
    }
    fn is_connected(&self) -> bool {
        self.connected
    }
    async fn submit(&self, _order: &Order) -> Result<OrderId> {
        Err(self.not_connected())
    }
    async fn account(&self) -> Result<AccountSummary> {
        Err(self.not_connected())
    }
    async fn positions(&self) -> Result<Vec<Position>> {
        Err(self.not_connected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn paper_fills_and_tracks_position() {
        let b = PaperBroker::new(10_000.0);
        let order = Order::limit("AAPL", Side::Buy, 10, 100.0);
        let id = b.submit(&order).await.unwrap();
        assert_eq!(id, OrderId(1));
        let acct = b.account().await.unwrap();
        assert!((acct.cash - 9_000.0).abs() < 1e-6); // 10 * $100 spent
        let pos = b.positions().await.unwrap();
        assert_eq!(pos.len(), 1);
        assert_eq!(pos[0].quantity, 10.0);
    }

    #[tokio::test]
    async fn paper_needs_a_price_for_market() {
        let b = PaperBroker::new(1_000.0);
        let res = b.submit(&Order::market("XYZ", Side::Buy, 5)).await;
        assert!(res.is_err());
        b.set_mark("XYZ", 20.0);
        assert!(b.submit(&Order::market("XYZ", Side::Buy, 5)).await.is_ok());
    }

    #[tokio::test]
    async fn ibkr_is_safe_until_connected() {
        let mut ib = IbkrBroker::new(IbkrConfig::default());
        assert!(!ib.is_connected());
        assert!(ib.connect().await.is_err()); // typed error, no panic
        assert!(ib
            .submit(&Order::market("AAPL", Side::Buy, 1))
            .await
            .is_err());
    }
}
