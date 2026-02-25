//! JSON-RPC 2.0 types per <https://www.jsonrpc.org/specification>.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Id — request/response identifier (String or Number per spec)
// ---------------------------------------------------------------------------

/// JSON-RPC request identifier. The spec allows String, Number, or Null.
/// Null is only valid in error responses with unknown id, so we model it
/// separately (see [`JsonRpcResponse::id`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Id {
    Number(u64),
    String(String),
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request object.
///
/// Per spec: `jsonrpc` MUST be `"2.0"`, `method` is a String, `params` is an
/// optional structured value (Object or Array), and `id` identifies the request.
#[derive(Debug, Serialize)]
pub struct Request {
    jsonrpc: &'static str,
    pub id: Id,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl Request {
    pub fn new(
        id: impl Into<Id>,
        method: impl Into<String>,
        params: Option<serde_json::Value>,
    ) -> Self {
        Self {
            jsonrpc: "2.0",
            id: id.into(),
            method: method.into(),
            params,
        }
    }
}

// ---------------------------------------------------------------------------
// Notification — a request without an id
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 notification (request without an `id`).
///
/// Per spec: the server MUST NOT reply to notifications.
#[derive(Debug, Serialize)]
pub struct Notification {
    jsonrpc: &'static str,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl Notification {
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            method: method.into(),
            params,
        }
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 response object.
///
/// Per spec: exactly one of `result` or `error` MUST be present (but not both).
/// The `id` is Null when the request id could not be determined.
#[derive(Debug, Deserialize)]
pub struct Response {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub error: Option<Error>,
}

impl Response {
    /// Check whether the response `id` matches an expected [`Id`].
    pub fn id_matches(&self, expected: &Id) -> bool {
        match (&self.id, expected) {
            (serde_json::Value::Number(n), Id::Number(e)) => n.as_u64() == Some(*e),
            (serde_json::Value::String(s), Id::String(e)) => s == e,
            // Spec allows servers to return id as either type, so handle
            // a numeric id serialized as a string or vice-versa.
            (serde_json::Value::String(s), Id::Number(e)) => s == &e.to_string(),
            (serde_json::Value::Number(n), Id::String(e)) => {
                n.as_u64().map(|n| n.to_string()).as_deref() == Some(e.as_str())
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 error object.
///
/// Per spec: `code` is an integer indicating the error type, `message` is a
/// short description, and `data` may carry additional information.
#[derive(Debug, Deserialize)]
pub struct Error {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Reserved error codes (spec §5.1)
// ---------------------------------------------------------------------------

/// Invalid JSON was received by the server.
pub const PARSE_ERROR: i32 = -32700;
/// The JSON sent is not a valid Request object.
pub const INVALID_REQUEST: i32 = -32600;
/// The method does not exist / is not available.
pub const METHOD_NOT_FOUND: i32 = -32601;
/// Invalid method parameter(s).
pub const INVALID_PARAMS: i32 = -32602;
/// Internal JSON-RPC error.
pub const INTERNAL_ERROR: i32 = -32603;

// ---------------------------------------------------------------------------
// Convenience conversions
// ---------------------------------------------------------------------------

impl From<u64> for Id {
    fn from(n: u64) -> Self {
        Id::Number(n)
    }
}

impl From<String> for Id {
    fn from(s: String) -> Self {
        Id::String(s)
    }
}

impl From<&str> for Id {
    fn from(s: &str) -> Self {
        Id::String(s.to_owned())
    }
}
