use std::error::Error as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{header, Method, Response, StatusCode};
use serde::{de::DeserializeOwned, Serialize};

use crate::api::v1::ApiError;

use super::telemetry;

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
        anyhow!("{kind} {}: {detail}", self.base_url)
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self.send(self.request(Method::GET, path)).await?;
        decode(res).await
    }

    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .send(self.request(Method::POST, path).json(body))
            .await?;
        decode(res).await
    }

    pub async fn patch_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .send(self.request(Method::PATCH, path).json(body))
            .await?;
        decode(res).await
    }

    pub async fn put_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let res = self
            .send(self.request(Method::PUT, path).json(body))
            .await?;
        decode(res).await
    }

    /// A PUT whose success is a 204 — a write-only surface has nothing to
    /// return, and decoding an empty body as JSON would fail.
    pub async fn put_discard<B: Serialize>(&self, path: &str, body: &B) -> Result<()> {
        let res = self
            .send(self.request(Method::PUT, path).json(body))
            .await?;
        check_status(res).await?;
        Ok(())
    }

    /// A POST whose success is a 204: nothing to decode, and decoding an empty
    /// body as JSON would fail.
    pub async fn post_json_discard<B: Serialize>(&self, path: &str, body: &B) -> Result<()> {
        let res = self
            .send(self.request(Method::POST, path).json(body))
            .await?;
        check_status(res).await?;
        Ok(())
    }

    pub async fn delete_discard(&self, path: &str) -> Result<()> {
        let res = self.send(self.request(Method::DELETE, path)).await?;
        check_status(res).await?;
        Ok(())
    }

    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let res = self
            .send(
                self.request(Method::POST, path)
                    .header(header::CONTENT_LENGTH, "0"),
            )
            .await?;
        decode(res).await
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
        let res = check_status(res).await?;

        let mut stream = res.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.context("reading SSE chunk")?);
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

async fn decode<T: DeserializeOwned>(res: Response) -> Result<T> {
    let res = check_status(res).await?;
    res.json::<T>().await.context("decoding response JSON")
}

async fn check_status(res: Response) -> Result<Response> {
    let status = res.status();
    if status.is_success() {
        return Ok(res);
    }

    let body_text = res.text().await.unwrap_or_default();
    let parsed: Option<ApiError> = serde_json::from_str(&body_text).ok();
    let (code, message) = parsed
        .map(|b| (b.error.code, b.error.message))
        .unwrap_or((None, None));

    let raw_msg = message.unwrap_or_else(|| body_text.lines().next().unwrap_or("").to_string());
    let msg = raw_msg.trim_end_matches('.');
    let prefix = match code.as_deref() {
        Some(c) if !c.is_empty() => format!("HTTP {} {}", status.as_u16(), c),
        _ => format!("HTTP {}", status.as_u16()),
    };
    let suffix = match status {
        StatusCode::UNAUTHORIZED => " Run `subs login` to authenticate.",
        StatusCode::FORBIDDEN => " Your account does not have access to this resource.",
        _ => "",
    };

    bail!("{prefix}: {msg}.{suffix}")
}
