//! Why a JSON document did not parse.
//!
//! Every seam where untrusted JSON enters the engine — a worker's decision
//! response, an SSE frame, a provider's reply — parses through [`from_slice`]
//! rather than `serde_json::from_slice`, because the plain serde error names
//! only a line and a column, and `reqwest`'s decode error hides even that
//! (`Display` for it is the constant "error decoding response body"). What the
//! reader needs is which field was wrong.
//!
//! The result is shaped like a Stripe API error: one sentence naming the
//! offending value, and a [`param`](JsonParseError::param) naming the field to
//! go and fix. The raw document is kept for the log line and never travels with
//! the error — it is unbounded, and it holds whatever the document held.
//!
//! [`serde_path_to_error`] supplies the path; the excerpt is cut from the raw
//! bytes, which is possible only because callers read the body themselves
//! instead of handing it to a `.json()` that consumes and discards it.

use std::fmt;

use serde::de::DeserializeOwned;

/// How much of the document to quote around the failure.
const WINDOW: usize = 160;

/// A document that could not be deserialized into its target type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonParseError {
    /// What the document was meant to be, for the reader who sees only this.
    pub what: &'static str,
    /// Dotted path to the offending value; empty when it is the root.
    pub path: String,
    /// Serde's own message, including its line and column.
    pub message: String,
    /// The raw document around the failure, whitespace collapsed. For the log
    /// line only: it goes to the operator, never into the error.
    pub excerpt: String,
}

impl JsonParseError {
    /// The offending field, in the sense of Stripe's `param`: the one input a
    /// caller has to go and fix. `None` when the document failed as a whole.
    pub fn param(&self) -> Option<&str> {
        (!self.path.is_empty()).then_some(self.path.as_str())
    }
}

/// One sentence: the field, what was wrong, and the value that was wrong —
/// Stripe's `No such customer: 'cus_123'`. The excerpt is deliberately absent:
/// a quoted body is noise to whoever reads the error, and it carries whatever
/// the document held (a system prompt, worker state) to wherever the error is
/// shown.
impl fmt::Display for JsonParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid {}", self.what)?;
        if self.path.is_empty() {
            // No field to name, so serde's line and column are the only
            // locator there is — a syntax error has no path.
            return write!(f, ": {}", self.message);
        }
        write!(
            f,
            " at `{}`: {}",
            self.path,
            without_position(&self.message)
        )
    }
}

/// Drop serde's trailing " at line N column M". It is the weaker locator once a
/// field is named, and on the single-line JSON that crosses a wire it says
/// nothing a reader can use.
fn without_position(message: &str) -> &str {
    match message.rfind(" at line ") {
        Some(cut) => &message[..cut],
        None => message,
    }
}

impl std::error::Error for JsonParseError {}

/// A document the engine could not read is always the same kind of failure,
/// and always names the field it choked on.
impl From<JsonParseError> for crate::protocol::ErrorInfo {
    fn from(e: JsonParseError) -> Self {
        let info = crate::protocol::ErrorInfo::new(
            crate::protocol::ErrorCode::InvalidResponse,
            e.to_string(),
        );
        match e.param() {
            Some(param) => info.with_param(param),
            None => info,
        }
    }
}

/// Deserialize `bytes`, naming the field that failed.
pub fn from_slice<T: DeserializeOwned>(
    what: &'static str,
    bytes: &[u8],
) -> Result<T, JsonParseError> {
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let value = serde_path_to_error::deserialize(&mut de).map_err(|e| {
        let path = e.path().to_string();
        let inner = e.into_inner();
        JsonParseError {
            what,
            // The crate spells the root as ".".
            path: if path == "." { String::new() } else { path },
            message: inner.to_string(),
            excerpt: excerpt_at(bytes, inner.line(), inner.column()),
        }
    })?;
    // Trailing content means the body is not the single document it claims to
    // be — two frames concatenated, or JSON followed by a stack trace.
    de.end().map_err(|e| JsonParseError {
        what,
        path: String::new(),
        message: e.to_string(),
        excerpt: excerpt_at(bytes, e.line(), e.column()),
    })?;
    Ok(value)
}

/// [`from_slice`] for text already in hand, such as an SSE frame's data.
pub fn from_str<T: DeserializeOwned>(what: &'static str, text: &str) -> Result<T, JsonParseError> {
    from_slice(what, text.as_bytes())
}

/// [`from_slice`] for a value already parsed as JSON — a field the outer
/// document kept opaque. A `Value` has no line or column, so the excerpt is the
/// head of the value instead of the failing site.
pub fn from_value<T: DeserializeOwned>(
    what: &'static str,
    value: serde_json::Value,
) -> Result<T, JsonParseError> {
    serde_path_to_error::deserialize(&value).map_err(|e| {
        let path = e.path().to_string();
        JsonParseError {
            what,
            path: if path == "." { String::new() } else { path },
            message: e.into_inner().to_string(),
            excerpt: excerpt(value.to_string().as_bytes()),
        }
    })
}

/// A bounded, single-line quote of `bytes`, for a body that is not a parse
/// failure — an error page, an unexpected shape.
pub fn excerpt(bytes: &[u8]) -> String {
    excerpt_at(bytes, 0, 0)
}

/// Where a sentence stops being a message and starts being a body. A worker's
/// `{"error": …}` is quoted the way Stripe quotes an offending value, so it has
/// to stay quotable; a framework that puts a whole stack trace there has sent a
/// body, and that belongs in the log.
const MAX_MESSAGE: usize = 300;

/// A human message hiding in an arbitrary error body: `{"error": "..."}`,
/// `{"message": "..."}`, or either nested one level (`{"error": {"message": …}}`),
/// which covers what web frameworks and API gateways emit. Anything else is not
/// a message and is better left to the excerpt in `detail`.
pub fn error_message(bytes: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let field = |v: &serde_json::Value, key: &str| -> Option<String> {
        match v.get(key)? {
            serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
            nested => nested
                .get("message")
                .or_else(|| nested.get("error"))
                .and_then(|m| m.as_str())
                .map(str::to_string),
        }
    };
    field(&value, "error")
        .or_else(|| field(&value, "message"))
        .or_else(|| field(&value, "detail"))
        .map(|m| one_line(&m, MAX_MESSAGE))
}

/// Collapse to a single bounded line.
fn one_line(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max) {
        Some((cut, _)) => format!("{}…", &flat[..cut]),
        None => flat,
    }
}

/// Quote the document around `line`/`column`, or its head when serde gave no
/// position (both zero).
fn excerpt_at(bytes: &[u8], line: usize, column: usize) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    let at = offset_of(bytes, line, column);
    let start = at.saturating_sub(WINDOW / 2);
    let end = (at + WINDOW / 2).min(bytes.len());

    let mut text = String::from_utf8_lossy(&bytes[start..end])
        .trim_matches(char::REPLACEMENT_CHARACTER)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if start > 0 {
        text.insert(0, '…');
    }
    if end < bytes.len() {
        text.push('…');
    }
    text
}

/// The byte offset of a 1-based line and column; zero when serde reported none.
fn offset_of(bytes: &[u8], line: usize, column: usize) -> usize {
    let mut offset = 0;
    for _ in 1..line.max(1) {
        match bytes[offset..].iter().position(|b| *b == b'\n') {
            Some(i) => offset += i + 1,
            None => break,
        }
    }
    (offset + column.saturating_sub(1)).min(bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Outer {
        #[allow(dead_code)]
        items: Vec<Inner>,
    }

    #[derive(Debug, Deserialize)]
    struct Inner {
        #[allow(dead_code)]
        name: String,
    }

    /// The point of the module: the failing field is named, not just a column.
    #[test]
    fn the_error_names_the_path_and_quotes_the_value() {
        let body = br#"{"items":[{"name":"ok"},{"name":42}]}"#;
        let err = from_slice::<Outer>("decision response", body).expect_err("42 is not a string");
        assert_eq!(err.path, "items[1].name");
        assert!(err.message.contains("invalid type"), "{}", err.message);
        assert!(err.excerpt.contains("42"), "{}", err.excerpt);
        let rendered = err.to_string();
        assert!(rendered.contains("invalid decision response at `items[1].name`"));
    }

    #[test]
    fn a_root_failure_has_no_path() {
        let err = from_slice::<Outer>("decision response", b"[]").expect_err("not an object");
        assert!(err.path.is_empty(), "{}", err.path);
    }

    #[test]
    fn trailing_content_is_a_failure_not_a_silent_prefix() {
        let body = br#"{"items":[]} <html>oops</html>"#;
        let err = from_slice::<Outer>("decision response", body).expect_err("trailing junk");
        assert!(err.excerpt.contains("html"), "{}", err.excerpt);
    }

    #[test]
    fn an_excerpt_is_bounded_and_single_line() {
        let long = format!(r#"{{"items":[{}]}}"#, "{\"name\":\n1},".repeat(200));
        let err =
            from_slice::<Outer>("decision response", long.as_bytes()).expect_err("not a string");
        assert!(err.excerpt.chars().count() <= WINDOW + 2, "{}", err.excerpt);
        assert!(!err.excerpt.contains('\n'), "{}", err.excerpt);
    }

    #[test]
    fn a_worker_error_body_gives_up_its_message() {
        assert_eq!(
            error_message(br#"{"error":"ANTHROPIC_API_KEY is not set"}"#).as_deref(),
            Some("ANTHROPIC_API_KEY is not set")
        );
        assert_eq!(
            error_message(br#"{"error":{"message":"model not found","type":"invalid_request"}}"#)
                .as_deref(),
            Some("model not found")
        );
        assert_eq!(
            error_message(br#"{"message":"Internal Server Error"}"#).as_deref(),
            Some("Internal Server Error")
        );
        assert_eq!(error_message(b"<html>500</html>"), None);
        assert_eq!(error_message(br#"{"actions":[]}"#), None);
    }
}
