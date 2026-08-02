use std::error::Error as _;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{header, Method, Response, StatusCode};
use serde::{de::DeserializeOwned, Serialize};

use crate::api::v1::ApiError;

use super::telemetry;

/// Nothing answered: DNS, connect, TLS or timeout. A type rather than a string
/// so callers can tell "the server said no" from "there was no server" —
/// nothing about the deployment may be inferred from this.
#[derive(Debug)]
pub struct Unreachable {
    pub url: String,
    kind: &'static str,
    detail: String,
}

impl fmt::Display for Unreachable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}: {}", self.kind, self.url, self.detail)
    }
}

impl std::error::Error for Unreachable {}

/// The server answered, with a status that is not a success.
#[derive(Debug)]
pub struct HttpStatus {
    pub status: StatusCode,
    pub url: String,
    /// The API's own error code, absent when the body did not come from one.
    pub code: Option<String>,
    message: String,
    suffix: &'static str,
}

impl fmt::Display for HttpStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = match self.code.as_deref() {
            Some(c) if !c.is_empty() => format!("HTTP {} {}", self.status.as_u16(), c),
            _ => format!("HTTP {}", self.status.as_u16()),
        };
        write!(
            f,
            "{prefix} from {}: {}.{}",
            self.url, self.message, self.suffix
        )
    }
}

impl std::error::Error for HttpStatus {}

/// The status a server answered with, or None when none did.
pub fn status_of(err: &anyhow::Error) -> Option<StatusCode> {
    err.downcast_ref::<HttpStatus>().map(|e| e.status)
}

#[derive(Debug)]
pub struct CloudClient {
    base_url: String,
    token: Option<String>,
    http: reqwest::Client,
    default_org: Mutex<Option<String>>,
    default_project: Mutex<Option<String>>,
    defaults_probed: AtomicBool,
}

impl CloudClient {
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Self {
        let user_agent = format!(
            "subs/{} ({}; {})",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
        );
        let mut default_headers = HeaderMap::new();
        if let Some(t) = telemetry::get() {
            if let Ok(v) = HeaderValue::from_str(&t.invocation_id) {
                default_headers.insert(HeaderName::from_static("x-subs-invocation-id"), v);
            }
            if let Ok(v) = HeaderValue::from_str(t.command) {
                default_headers.insert(HeaderName::from_static("x-subs-command"), v);
            }
        }
        let http = reqwest::Client::builder()
            .user_agent(user_agent)
            .default_headers(default_headers)
            .build()
            .expect("reqwest client");
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token,
            http,
            default_org: Mutex::new(None),
            default_project: Mutex::new(None),
            defaults_probed: AtomicBool::new(false),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Org/project a single-tenant server advertised via response headers (None
    /// against the cloud). Populated as a side effect of any request.
    pub fn default_org(&self) -> Option<String> {
        self.default_org.lock().unwrap().clone()
    }

    pub fn default_project(&self) -> Option<String> {
        self.default_project.lock().unwrap().clone()
    }

    /// True the first time only, so callers probe the server for defaults once.
    pub fn needs_default_probe(&self) -> bool {
        !self.defaults_probed.swap(true, Ordering::Relaxed)
    }

    fn capture_defaults(&self, headers: &HeaderMap) {
        if let Some(v) = headers
            .get("x-substructure-org")
            .and_then(|v| v.to_str().ok())
        {
            *self.default_org.lock().unwrap() = Some(v.to_string());
        }
        if let Some(v) = headers
            .get("x-substructure-project")
            .and_then(|v| v.to_str().ok())
        {
            *self.default_project.lock().unwrap() = Some(v.to_string());
        }
    }

    fn url(&self, path: &str) -> String {
        if path.starts_with("http://") || path.starts_with("https://") {
            path.to_string()
        } else {
            format!("{}{}", self.base_url, path)
        }
    }

    fn request(&self, method: Method, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.http.request(method, self.url(path));
        if let Some(t) = &self.token {
            req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        req
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> Result<Response> {
        let res = req.send().await.map_err(|e| self.transport_error(e))?;
        self.capture_defaults(res.headers());
        Ok(res)
    }

    // Turn reqwest's nested connect/TLS/timeout chain into one actionable
    // line that names the URL the CLI is trying to reach. Status-code
    // errors are handled separately in check_status.
    fn transport_error(&self, e: reqwest::Error) -> anyhow::Error {
        let kind = if e.is_timeout() {
            "timed out reaching"
        } else if e.is_connect() {
            "could not connect to"
        } else if e.is_request() {
            "could not reach"
        } else {
            return anyhow::Error::new(e);
        };
        // The innermost source (e.g. "tls handshake eof", "dns error") is
        // far more useful than the outer "error sending request for url …".
        let mut deepest: Option<&dyn std::error::Error> = e.source();
        while let Some(next) = deepest.and_then(|s| s.source()) {
            deepest = Some(next);
        }
        let detail = deepest
            .map(|s| s.to_string())
            .unwrap_or_else(|| e.to_string());
        anyhow::Error::new(Unreachable {
            url: self.base_url.clone(),
            kind,
            detail,
        })
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self.send(self.request(Method::GET, path)).await?;
        decode(&self.base_url, res).await
    }

    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .send(self.request(Method::POST, path).json(body))
            .await?;
        decode(&self.base_url, res).await
    }

    pub async fn patch_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .send(self.request(Method::PATCH, path).json(body))
            .await?;
        decode(&self.base_url, res).await
    }

    pub async fn put_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .send(self.request(Method::PUT, path).json(body))
            .await?;
        decode(&self.base_url, res).await
    }

    /// A PUT whose success is a 204 — a write-only surface has nothing to
    /// return, and decoding an empty body as JSON would fail.
    pub async fn put_discard<B: Serialize>(&self, path: &str, body: &B) -> Result<()> {
        let res = self
            .send(self.request(Method::PUT, path).json(body))
            .await?;
        check_status(&self.base_url, res).await?;
        Ok(())
    }

    /// A POST whose success is a 204: nothing to decode, and decoding an empty
    /// body as JSON would fail.
    pub async fn post_json_discard<B: Serialize>(&self, path: &str, body: &B) -> Result<()> {
        let res = self
            .send(self.request(Method::POST, path).json(body))
            .await?;
        check_status(&self.base_url, res).await?;
        Ok(())
    }

    pub async fn delete_discard(&self, path: &str) -> Result<()> {
        let res = self.send(self.request(Method::DELETE, path)).await?;
        check_status(&self.base_url, res).await?;
        Ok(())
    }

    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self
            .send(
                self.request(Method::POST, path)
                    .header(header::CONTENT_LENGTH, "0"),
            )
            .await?;
        decode(&self.base_url, res).await
    }

    // Raw Response: OAuth device-flow polling treats 400 `authorization_pending` as "keep polling".
    // Better Auth's device endpoints require JSON (form-encoded yields UNSUPPORTED_MEDIA_TYPE).
    pub async fn post_json_raw<B: Serialize>(&self, path: &str, body: &B) -> Result<Response> {
        self.send(self.request(Method::POST, path).json(body)).await
    }

    /// Open an SSE stream and invoke `on_line` for each non-empty line
    /// (caller decides which lines to keep — e.g. `data:` payloads).
    /// Runs until the server closes the stream or the caller hits Ctrl-C.
    pub async fn stream_sse<F>(&self, path: &str, mut on_line: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        let res = self
            .send(
                self.request(Method::GET, path)
                    .header(header::ACCEPT, "text/event-stream"),
            )
            .await?;
        let res = check_status(&self.base_url, res).await?;

        let mut stream = res.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| {
                format!("the connection to {} dropped mid-stream", self.base_url)
            })?;
            buf.extend_from_slice(&chunk);
            // Lines are terminated by \n; \r is tolerated and stripped.
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let s = std::str::from_utf8(&line[..line.len() - 1])
                    .context("non-UTF8 in SSE stream")?
                    .trim_end_matches('\r');
                on_line(s);
            }
        }
        Ok(())
    }
}

/// One line of a body that is not the API's own error shape, so an HTML page
/// from a proxy is readable rather than pasted whole.
fn snippet(body: &str) -> String {
    let line = body.trim().lines().next().unwrap_or("").trim();
    match line.char_indices().nth(80) {
        Some((cut, _)) => format!("{}…", &line[..cut]),
        None => line.to_string(),
    }
}

async fn decode<T: DeserializeOwned>(url: &str, res: Response) -> Result<T> {
    let res = check_status(url, res).await?;
    res.json::<T>().await.context("decoding response JSON")
}

async fn check_status(url: &str, res: Response) -> Result<Response> {
    let status = res.status();
    if status.is_success() {
        return Ok(res);
    }

    let path = res.url().path().to_string();
    let body_text = res.text().await.unwrap_or_default();
    let parsed: Option<ApiError> = serde_json::from_str(&body_text).ok();
    let from_api = parsed.is_some();
    let (code, message) = parsed
        .map(|b| (b.error.code, b.error.message))
        .unwrap_or((None, None));

    // A body the API did not write means a proxy or the wrong URL answered, so
    // its status describes that, not the account.
    let message = match message {
        Some(m) => m.trim_end_matches('.').to_string(),
        None if !from_api && !body_text.trim().is_empty() => {
            format!("not an API response: \"{}\"", snippet(&body_text))
        }
        None if status == StatusCode::NOT_FOUND => format!("no such endpoint: {path}"),
        None => format!("no detail for {path}"),
    };
    let suffix = match status {
        StatusCode::UNAUTHORIZED if from_api => " Run `subs login`.",
        _ => "",
    };

    Err(anyhow::Error::new(HttpStatus {
        status,
        url: url.to_string(),
        code,
        message,
        suffix,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unreachable_server_names_the_url_and_the_reason() {
        let e = Unreachable {
            url: "http://127.0.0.1:1".to_string(),
            kind: "could not connect to",
            detail: "Connection refused (os error 61)".to_string(),
        };
        assert_eq!(
            e.to_string(),
            "could not connect to http://127.0.0.1:1: Connection refused (os error 61)"
        );
    }

    #[test]
    fn a_status_carries_the_api_code_and_only_the_api_gets_the_login_hint() {
        let from_api = HttpStatus {
            status: StatusCode::UNAUTHORIZED,
            url: "https://api.example".to_string(),
            code: Some("unauthenticated".to_string()),
            message: "Not authenticated".to_string(),
            suffix: " Run `subs login`.",
        };
        assert_eq!(
            from_api.to_string(),
            "HTTP 401 unauthenticated from https://api.example: Not authenticated. \
             Run `subs login`."
        );

        let from_proxy = HttpStatus {
            status: StatusCode::UNAUTHORIZED,
            url: "https://api.example".to_string(),
            code: None,
            message: "not an API response: \"<html>\"".to_string(),
            suffix: "",
        };
        assert_eq!(
            from_proxy.to_string(),
            "HTTP 401 from https://api.example: not an API response: \"<html>\"."
        );
    }

    #[test]
    fn a_body_that_is_not_the_api_is_cut_to_one_readable_line() {
        assert_eq!(snippet("  <html>\n<body>\n"), "<html>");
        assert_eq!(snippet(""), "");
        let long = "x".repeat(300);
        let cut = snippet(&long);
        assert_eq!(cut.chars().count(), 81);
        assert!(cut.ends_with('…'));
    }
}
