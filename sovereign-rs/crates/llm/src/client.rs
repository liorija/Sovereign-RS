//! LLM client trait + Gemini / Grok (x.ai) / Ollama implementations.
//!
//! API keys are **always** read from the environment (`GEMINI_API_KEY`,
//! `GROK_API_KEY`/`XAI_API_KEY`) — never hardcoded, never removed. The request
//! body builders are pure functions (unit-tested offline); only `complete()`
//! touches the network.

use async_trait::async_trait;
use serde_json::{json, Value};

use sovereign_core::error::{Result, SovereignError};

/// An async text-completion backend (the WarRoom "brains").
#[async_trait]
pub trait LlmClient: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(&self, prompt: &str) -> Result<String>;
}

fn http(timeout_s: u64) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_s.max(1)))
        .build()
        .map_err(|e| SovereignError::data("llm", e.to_string()))
}

// ── Google Gemini ───────────────────────────────────────────────────────────

/// Google Gemini (`generativelanguage.googleapis.com`).
#[derive(Debug, Clone)]
pub struct GeminiClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
}

impl GeminiClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            http: http(20)?,
            api_key: api_key.into(),
            model: model.into(),
        })
    }

    fn url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        )
    }

    /// Pure request-body builder (testable without a key/network).
    pub fn request_body(prompt: &str) -> Value {
        json!({ "contents": [ { "parts": [ { "text": prompt } ] } ] })
    }
}

#[async_trait]
impl LlmClient for GeminiClient {
    fn name(&self) -> &str {
        "gemini"
    }
    async fn complete(&self, prompt: &str) -> Result<String> {
        let resp = self
            .http
            .post(self.url())
            .json(&Self::request_body(prompt))
            .send()
            .await
            .map_err(|e| SovereignError::data("gemini", e.to_string()))?;
        let v: Value = resp
            .json()
            .await
            .map_err(|e| SovereignError::data("gemini", e.to_string()))?;
        v["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| SovereignError::data("gemini", "no text in response"))
    }
}

// ── xAI Grok (OpenAI-compatible) ─────────────────────────────────────────────

/// xAI Grok chat-completions (`api.x.ai`).
#[derive(Debug, Clone)]
pub struct GrokClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
}

impl GrokClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            http: http(20)?,
            api_key: api_key.into(),
            model: model.into(),
        })
    }

    /// Pure request-body builder.
    pub fn request_body(model: &str, prompt: &str) -> Value {
        json!({ "model": model, "messages": [ { "role": "user", "content": prompt } ] })
    }
}

#[async_trait]
impl LlmClient for GrokClient {
    fn name(&self) -> &str {
        "grok"
    }
    async fn complete(&self, prompt: &str) -> Result<String> {
        let resp = self
            .http
            .post("https://api.x.ai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&Self::request_body(&self.model, prompt))
            .send()
            .await
            .map_err(|e| SovereignError::data("grok", e.to_string()))?;
        let v: Value = resp
            .json()
            .await
            .map_err(|e| SovereignError::data("grok", e.to_string()))?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| SovereignError::data("grok", "no content in response"))
    }
}

// ── Local Ollama ──────────────────────────────────────────────────────────────

/// Local Ollama server (offline brain — `llama3.1` etc.).
#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    host: String,
    model: String,
}

impl OllamaClient {
    pub fn new(host: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Ok(Self {
            http: http(60)?,
            host: host.into(),
            model: model.into(),
        })
    }
}

#[async_trait]
impl LlmClient for OllamaClient {
    fn name(&self) -> &str {
        "ollama"
    }
    async fn complete(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.host.trim_end_matches('/'));
        let body = json!({ "model": self.model, "prompt": prompt, "stream": false });
        let resp = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SovereignError::data("ollama", e.to_string()))?;
        let v: Value = resp
            .json()
            .await
            .map_err(|e| SovereignError::data("ollama", e.to_string()))?;
        v["response"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| SovereignError::data("ollama", "no response field"))
    }
}

// ── Offline mock ──────────────────────────────────────────────────────────────

/// Deterministic mock for tests / dry-runs.
#[derive(Debug, Clone)]
pub struct MockLlm {
    pub reply: String,
}

#[async_trait]
impl LlmClient for MockLlm {
    fn name(&self) -> &str {
        "mock"
    }
    async fn complete(&self, _prompt: &str) -> Result<String> {
        Ok(self.reply.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_body_shape() {
        let b = GeminiClient::request_body("hello");
        assert_eq!(b["contents"][0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn grok_body_shape() {
        let b = GrokClient::request_body("grok-2", "hi");
        assert_eq!(b["model"], "grok-2");
        assert_eq!(b["messages"][0]["content"], "hi");
    }

    #[tokio::test]
    async fn mock_completes() {
        let m = MockLlm {
            reply: "BUY".into(),
        };
        assert_eq!(m.complete("anything").await.unwrap(), "BUY");
    }
}
