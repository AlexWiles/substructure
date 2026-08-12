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

/// The payload of an SSE `data:` line, or nothing for any other line. The space
/// after the colon is optional in the spec, and some providers omit it.
pub(crate) fn sse_data(line: &str) -> Option<&str> {
    line.strip_prefix("data:")
        .map(|d| d.strip_prefix(' ').unwrap_or(d))
}

#[cfg(test)]
mod tests {
    use super::sse_data;

    #[test]
    fn a_data_line_reads_the_same_with_or_without_the_space() {
        assert_eq!(sse_data("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(sse_data("data:{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(sse_data("data: [DONE]"), Some("[DONE]"));
        assert_eq!(sse_data("data:[DONE]"), Some("[DONE]"));
        assert_eq!(sse_data("event: message"), None);
        assert_eq!(sse_data(": keep-alive"), None);
    }

    #[test]
    fn only_the_first_space_goes() {
        assert_eq!(sse_data("data:  x"), Some(" x"));
    }
}
