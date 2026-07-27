pub mod anthropic;
pub mod format;
pub mod memory_queue;
pub mod openai;
pub mod openrouter;
pub mod sqlite;
pub mod worker_queue;

/// Provider HTTP client. Bounds connection establishment so a dead provider
/// fails retryable instead of hanging the call; request duration is bounded by
/// the call's retry policy, not the client.
pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}
