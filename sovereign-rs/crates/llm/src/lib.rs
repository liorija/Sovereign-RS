//! # sovereign-llm
//!
//! The WarRoom's language-model backends: [`GeminiClient`], [`GrokClient`],
//! [`OllamaClient`], a [`KeyPool`] for free-tier rotation, and an offline
//! [`MockLlm`]. **Secrets come from the environment only** — they are never
//! hardcoded and never stripped from the config:
//!
//! | Env var | Purpose |
//! |---------|---------|
//! | `GEMINI_API_KEY` / `GEMINI_API_KEYS` (csv) | Gemini key(s) |
//! | `GROK_API_KEY` / `XAI_API_KEY` | Grok key |
//! | `OLLAMA_HOST` | Ollama server (default `http://localhost:11434`) |
//! | `GEMINI_MODEL` / `GROK_MODEL` / `OLLAMA_MODEL` | model overrides |
#![forbid(unsafe_code)]

pub mod client;
pub mod pool;

pub use client::{GeminiClient, GrokClient, LlmClient, MockLlm, OllamaClient};
pub use pool::KeyPool;

/// LLM configuration resolved from the environment.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub gemini_keys: Vec<String>,
    pub grok_key: Option<String>,
    pub ollama_host: String,
    pub gemini_model: String,
    pub grok_model: String,
    pub ollama_model: String,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

impl LlmConfig {
    /// Resolve from environment variables (keys never hardcoded).
    pub fn from_env() -> Self {
        let gemini_keys: Vec<String> = match env("GEMINI_API_KEYS") {
            Some(csv) => csv
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            None => env("GEMINI_API_KEY").into_iter().collect(),
        };
        Self {
            gemini_keys,
            grok_key: env("GROK_API_KEY").or_else(|| env("XAI_API_KEY")),
            ollama_host: env("OLLAMA_HOST").unwrap_or_else(|| "http://localhost:11434".into()),
            gemini_model: env("GEMINI_MODEL").unwrap_or_else(|| "gemini-2.5-flash".into()),
            grok_model: env("GROK_MODEL").unwrap_or_else(|| "grok-2".into()),
            ollama_model: env("OLLAMA_MODEL").unwrap_or_else(|| "llama3.1".into()),
        }
    }

    pub fn has_gemini(&self) -> bool {
        !self.gemini_keys.is_empty()
    }
    pub fn has_grok(&self) -> bool {
        self.grok_key.is_some()
    }

    /// Build the preferred available client: Gemini → Grok → Ollama (always
    /// available locally). Returns a boxed trait object for the WarRoom.
    pub fn build_primary(&self) -> sovereign_core::error::Result<Box<dyn LlmClient>> {
        if let Some(key) = self.gemini_keys.first() {
            return Ok(Box::new(GeminiClient::new(
                key.clone(),
                self.gemini_model.clone(),
            )?));
        }
        if let Some(key) = &self.grok_key {
            return Ok(Box::new(GrokClient::new(
                key.clone(),
                self.grok_model.clone(),
            )?));
        }
        Ok(Box::new(OllamaClient::new(
            self.ollama_host.clone(),
            self.ollama_model.clone(),
        )?))
    }

    /// A pool over the configured Gemini keys.
    pub fn gemini_pool(&self) -> KeyPool {
        KeyPool::new(self.gemini_keys.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_has_safe_defaults() {
        // Without env vars set, no keys but Ollama defaults are present.
        let cfg = LlmConfig::from_env();
        assert!(cfg.ollama_host.starts_with("http"));
        assert!(!cfg.gemini_model.is_empty());
        // build_primary always yields *some* client (falls back to Ollama).
        assert!(cfg.build_primary().is_ok());
    }
}
