//! Centralized defaults for user-configurable runtime behavior.

/// LLM call timeout in seconds.
pub const LLM_TIMEOUT_SECS: u32 = 60;
/// Tool call timeout in seconds.
pub const TOOL_TIMEOUT_SECS: u32 = 120;
/// Maximum retry attempts for failed LLM calls.
pub const MAX_RETRIES: u32 = 3;
/// Base delay for exponential backoff in seconds.
pub const BACKOFF_BASE_SECS: u32 = 2;
/// Maximum backoff delay in seconds.
pub const BACKOFF_MAX_SECS: u32 = 60;
/// Maximum tool result size in bytes.
pub const MAX_TOOL_RESULT_BYTES: usize = 100_000;
