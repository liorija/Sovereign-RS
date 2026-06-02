//! Round-robin API-key pool (port of `_GEMINI_KEY_POOL` rotation) — rotate
//! across free-tier keys and ban exhausted ones to dodge rate limits.

use std::collections::HashSet;

/// A rotating pool of API keys.
#[derive(Debug, Clone, Default)]
pub struct KeyPool {
    keys: Vec<String>,
    banned: HashSet<usize>,
    cursor: usize,
    last: Option<usize>,
}

impl KeyPool {
    pub fn new(keys: Vec<String>) -> Self {
        Self {
            keys,
            ..Default::default()
        }
    }

    /// Parse a comma-separated key list (e.g. the `GEMINI_API_KEYS` env var).
    pub fn from_csv(s: &str) -> Self {
        Self::new(
            s.split(',')
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect(),
        )
    }

    /// Next available key (round-robin, skipping banned). Remembers it so it can
    /// be banned on a 429/quota error.
    pub fn next_key(&mut self) -> Option<String> {
        let n = self.keys.len();
        if n == 0 {
            return None;
        }
        for _ in 0..n {
            let i = self.cursor % n;
            self.cursor = (self.cursor + 1) % n;
            if !self.banned.contains(&i) {
                self.last = Some(i);
                return Some(self.keys[i].clone());
            }
        }
        None
    }

    /// Ban the key most recently returned by [`next_key`](Self::next_key).
    pub fn ban_last(&mut self) {
        if let Some(i) = self.last.take() {
            self.banned.insert(i);
            tracing::warn!(index = i, "API key banned (rate-limited/exhausted)");
        }
    }

    /// Number of keys still usable.
    pub fn available(&self) -> usize {
        self.keys.len().saturating_sub(self.banned.len())
    }

    /// Whether the pool is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.available() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_and_bans() {
        let mut p = KeyPool::from_csv("a, b , c");
        assert_eq!(p.available(), 3);
        let first = p.next_key().unwrap();
        p.ban_last();
        assert_eq!(p.available(), 2);
        // banned key never returned again
        for _ in 0..10 {
            assert_ne!(p.next_key().unwrap(), first);
        }
    }

    #[test]
    fn empty_pool_is_exhausted() {
        let mut p = KeyPool::from_csv("");
        assert!(p.is_exhausted());
        assert_eq!(p.next_key(), None);
    }
}
