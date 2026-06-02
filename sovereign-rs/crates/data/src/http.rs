//! Async HTTP client wrapper around `reqwest` (rustls — no system OpenSSL).
//!
//! Centralizes timeout, User-Agent (SEC-EDGAR fair-access requires one) and
//! error mapping so every provider shares the same hardened transport.

use serde::de::DeserializeOwned;

use sovereign_core::error::{Result, SovereignError};

/// A shared HTTP client.
#[derive(Debug, Clone)]
pub struct HttpClient {
    client: reqwest::Client,
}

impl HttpClient {
    /// Build a client with a polite User-Agent and per-request timeout.
    pub fn new(user_agent: &str, timeout_s: u64) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(user_agent)
            .timeout(std::time::Duration::from_secs(timeout_s.max(1)))
            .gzip(true)
            .build()
            .map_err(|e| SovereignError::data("http", e.to_string()))?;
        Ok(Self { client })
    }

    /// GET the URL and return the body as text.
    pub async fn get_text(&self, url: &str) -> Result<String> {
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| SovereignError::data("http", e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(SovereignError::data(
                "http",
                format!("{url} → HTTP {status}"),
            ));
        }
        resp.text()
            .await
            .map_err(|e| SovereignError::data("http", e.to_string()))
    }

    /// GET the URL and deserialize the JSON body into `T`.
    pub async fn get_json<T: DeserializeOwned>(&self, url: &str) -> Result<T> {
        let text = self.get_text(url).await?;
        serde_json::from_str(&text).map_err(|e| SovereignError::Serde {
            context: url.to_string(),
            source: e,
        })
    }
}
