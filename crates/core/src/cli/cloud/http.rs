use std::error::Error as _;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};

use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
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
    /// The body was the API's error shape, so the API itself answered.
    from_api: bool,
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

/// The status the API itself answered with, or None when something else did.
/// A 404 from a proxy, a dev server, or the wrong port says nothing about the
/// deployment, so nothing about it may be inferred from one.
pub fn api_status_of(err: &anyhow::Error) -> Option<StatusCode> {
    err.downcast_ref::<HttpStatus>()
        .filter(|e| e.from_api)
        .map(|e| e.status)
}

/// How to get a credential the deployment will take, once it has refused the
/// one this machine sent. `None` when there is nobody to ask, which leaves the
/// 401 to the caller. The client knows nothing of how a login works — see
/// [`super::context`], which installs it.
pub type Reauth = Box<dyn Fn() -> BoxFuture<'static, Result<Option<String>>> + Send + Sync>;

pub struct CloudClient {
    base_url: String,
    token: RwLock<Option<String>>,
    http: reqwest::Client,
    reauth: Option<Reauth>,
    /// One login at a time, however many requests are refused at once.
    login: tokio::sync::Mutex<()>,
    default_org: Mutex<Option<String>>,
    default_project: Mutex<Option<String>>,
    defaults_probed: AtomicBool,
}

impl fmt::Debug for CloudClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudClient")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
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
            token: RwLock::new(token),
            http,
            reauth: None,
            login: tokio::sync::Mutex::new(()),
            default_org: Mutex::new(None),
            default_project: Mutex::new(None),
            defaults_probed: AtomicBool::new(false),
        }
    }

    /// Let this client answer a refusal with a login instead of an error.
    pub fn with_reauth(mut self, reauth: Reauth) -> Self {
        self.reauth = Some(reauth);
        self
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

    /// The request without its credential: [`send`](Self::send) carries that,
    /// since which one it carries can change between the two attempts.
    fn request(&self, method: Method, path: &str) -> reqwest::RequestBuilder {
        self.http.request(method, self.url(path))
    }

    fn token(&self) -> Option<String> {
        self.token.read().unwrap().clone()
    }

    /// A refused credential is the one failure this CLI can fix by itself, so
    /// it logs in and sends the request again rather than handing the reader a
    /// 401 that only names the command to run. Once: a second refusal is the
    /// deployment's answer about the credential it just issued.
    async fn send(&self, req: reqwest::RequestBuilder) -> Result<Response> {
        let again = req.try_clone();
        let sent_with = self.token();
        let res = self.send_once(req, sent_with.as_deref()).await?;
        if res.status() != StatusCode::UNAUTHORIZED {
            return Ok(res);
        }
        let Some(again) = again else { return Ok(res) };
        match self.login(sent_with.as_deref()).await? {
            Some(token) => self.send_once(again, Some(&token)).await,
            None => Ok(res),
        }
    }

    async fn send_once(
        &self,
        req: reqwest::RequestBuilder,
        token: Option<&str>,
    ) -> Result<Response> {
        let req = match token {
            Some(t) => req.header(header::AUTHORIZATION, format!("Bearer {t}")),
            None => req,
        };
        let res = req.send().await.map_err(|e| self.transport_error(e))?;
        self.capture_defaults(res.headers());
        Ok(res)
    }

    /// The credential to try again with, or None when nothing can be done
    /// about the refusal. A caller that waited here while another request ran
    /// the login takes what that one won, rather than opening a second browser.
    async fn login(&self, refused: Option<&str>) -> Result<Option<String>> {
        let Some(reauth) = &self.reauth else {
            return Ok(None);
        };
        let _guard = self.login.lock().await;
        let current = self.token();
        if current.is_some() && current.as_deref() != refused {
            return Ok(current);
        }
        let fresh = reauth().await?;
        if fresh.is_some() {
            *self.token.write().unwrap() = fresh.clone();
        }
        Ok(fresh)
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
    pub async fn stream_sse<F>(&self, path: &str, on_line: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        self.stream(self.request(Method::GET, path), on_line).await
    }

    /// The same, for a stream a request body opens: a run is submitted and
    /// watched by one call, so nothing can happen between the two.
    pub async fn post_sse<B: Serialize, F>(&self, path: &str, body: &B, on_line: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        self.stream(self.request(Method::POST, path).json(body), on_line)
            .await
    }

    async fn stream<F>(&self, req: reqwest::RequestBuilder, mut on_line: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        let res = self
            .send(req.header(header::ACCEPT, "text/event-stream"))
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
        from_api,
        message,
        suffix,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use axum::extract::State;
    use axum::http::HeaderMap as AxumHeaders;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use axum::Router;

    use super::*;

    #[derive(Default)]
    struct Calls {
        seen: Mutex<Vec<Option<String>>>,
        logins: AtomicUsize,
    }

    /// Takes `Bearer fresh` and refuses everything else, recording what each
    /// request carried.
    async fn guarded(State(calls): State<Arc<Calls>>, headers: AxumHeaders) -> impl IntoResponse {
        let auth = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        calls.seen.lock().unwrap().push(auth.clone());
        match auth.as_deref() {
            Some("Bearer fresh") => (StatusCode::OK, "{\"ok\":true}").into_response(),
            _ => (
                StatusCode::UNAUTHORIZED,
                "{\"error\":{\"code\":\"UNAUTHORIZED\",\"message\":\"Not authenticated\"}}",
            )
                .into_response(),
        }
    }

    async fn serve(calls: Arc<Calls>) -> String {
        let app = Router::new()
            .route("/guarded", get(guarded))
            .with_state(calls);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    fn reauth(calls: Arc<Calls>, issues: Option<&'static str>) -> Reauth {
        Box::new(move || {
            let calls = calls.clone();
            Box::pin(async move {
                calls.logins.fetch_add(1, Ordering::Relaxed);
                Ok(issues.map(str::to_string))
            })
        })
    }

    #[derive(Debug, serde::Deserialize)]
    struct Ok_ {
        ok: bool,
    }

    #[tokio::test]
    async fn a_refused_credential_is_replaced_and_the_request_sent_again() {
        let calls = Arc::new(Calls::default());
        let url = serve(calls.clone()).await;
        let client = CloudClient::new(url, Some("stale".to_string()))
            .with_reauth(reauth(calls.clone(), Some("fresh")));

        let res: Ok_ = client.get("/guarded").await.expect("the second attempt");
        assert!(res.ok);
        assert_eq!(
            *calls.seen.lock().unwrap(),
            vec![
                Some("Bearer stale".to_string()),
                Some("Bearer fresh".to_string())
            ]
        );

        // The credential the login won is the one every later request carries.
        let res: Ok_ = client.get("/guarded").await.expect("no second login");
        assert!(res.ok);
        assert_eq!(calls.logins.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn nobody_to_log_in_leaves_the_refusal_to_the_caller() {
        let calls = Arc::new(Calls::default());
        let url = serve(calls.clone()).await;
        let client = CloudClient::new(url, Some("stale".to_string()))
            .with_reauth(reauth(calls.clone(), None));

        let err = client.get::<Ok_>("/guarded").await.expect_err("401");
        assert_eq!(api_status_of(&err), Some(StatusCode::UNAUTHORIZED));
        assert!(err.to_string().ends_with("Run `subs login`."));
        assert_eq!(calls.seen.lock().unwrap().len(), 1, "no second attempt");
    }

    #[tokio::test]
    async fn a_client_with_no_login_to_run_sends_once() {
        let calls = Arc::new(Calls::default());
        let url = serve(calls.clone()).await;
        let client = CloudClient::new(url, Some("stale".to_string()));

        client.get::<Ok_>("/guarded").await.expect_err("401");
        assert_eq!(calls.seen.lock().unwrap().len(), 1);
        assert_eq!(calls.logins.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn one_login_serves_the_requests_refused_at_the_same_time() {
        let calls = Arc::new(Calls::default());
        let url = serve(calls.clone()).await;
        let client = CloudClient::new(url, Some("stale".to_string()))
            .with_reauth(reauth(calls.clone(), Some("fresh")));

        let (a, b) = tokio::join!(client.get::<Ok_>("/guarded"), client.get::<Ok_>("/guarded"));
        assert!(a.is_ok() && b.is_ok());
        assert_eq!(
            calls.logins.load(Ordering::Relaxed),
            1,
            "the second waits and takes what the first won"
        );
    }

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
            from_api: true,
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
            from_api: false,
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
