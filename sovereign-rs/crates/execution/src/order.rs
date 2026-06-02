//! Order and bracket-order construction.

use serde::Serialize;

use sovereign_core::domain::Side;

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum OrderType {
    Market,
    Limit(f64),
}

/// A single order leg.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Order {
    pub symbol: String,
    pub side: Side,
    pub quantity: u64,
    pub order_type: OrderType,
    pub tag: String,
}

impl Order {
    /// Market order helper.
    pub fn market(symbol: impl Into<String>, side: Side, quantity: u64) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            quantity,
            order_type: OrderType::Market,
            tag: String::new(),
        }
    }

    /// Limit order helper.
    pub fn limit(symbol: impl Into<String>, side: Side, quantity: u64, price: f64) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            quantity,
            order_type: OrderType::Limit(price),
            tag: String::new(),
        }
    }

    /// Attach an order tag (e.g. `"BETA_NEUTRAL_LEG_A"`).
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = tag.into();
        self
    }
}

/// A bracket: an entry plus protective take-profit and stop-loss legs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BracketOrder {
    pub entry: Order,
    pub take_profit: f64,
    pub stop_loss: f64,
}

impl BracketOrder {
    /// Build a long bracket from an entry price and TP/SL percentages.
    /// E.g. `tp_pct = 0.09`, `sl_pct = 0.04`.
    pub fn long(
        symbol: impl Into<String> + Clone,
        quantity: u64,
        entry_price: f64,
        tp_pct: f64,
        sl_pct: f64,
    ) -> Self {
        Self {
            entry: Order::limit(symbol, Side::Buy, quantity, entry_price).with_tag("BRACKET_ENTRY"),
            take_profit: entry_price * (1.0 + tp_pct.max(0.0)),
            stop_loss: entry_price * (1.0 - sl_pct.max(0.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_levels_are_correct() {
        let b = BracketOrder::long("NVDA", 10, 100.0, 0.09, 0.04);
        assert_eq!(b.entry.side, Side::Buy);
        assert!((b.take_profit - 109.0).abs() < 1e-9);
        assert!((b.stop_loss - 96.0).abs() < 1e-9);
    }

    #[test]
    fn tagging_works() {
        let o = Order::market("SPY", Side::Sell, 5).with_tag("HEDGE");
        assert_eq!(o.tag, "HEDGE");
    }
}
