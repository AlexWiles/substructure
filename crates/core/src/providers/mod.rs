pub mod anthropic;
pub mod format;
pub mod memory_queue;
pub mod openai;
pub mod openrouter;
pub mod sqlite;
pub mod worker_queue;

pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

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
