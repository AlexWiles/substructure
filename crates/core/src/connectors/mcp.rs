//! A Streamable HTTP MCP client (spec revision 2025-11-25).
//!
//! One `McpClient` fronts one connection. It initializes lazily, carries the
//! server's `MCP-Session-Id` for the life of that session, and re-initializes
//! when the server drops it (404), so a long-lived agent survives a server
//! restart without the caller knowing.
//!
//! Only the calls the engine needs are implemented: `initialize`, `tools/list`,
//! `tools/call`. The server-to-client direction (the optional GET stream) is not
//! opened — nothing in the engine consumes server-initiated requests.

use std::sync::atomic::{AtomicU64, Ordering};

use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use super::{AuthNeed, ConnectorError, RemoteTool, ToolAnnotations, ToolOutcome};

/// The revision we negotiate. The 2026-07-28 revision drops the handshake and
/// the session id entirely, so moving to it deletes `SessionState` rather than
/// changing the call sites.
const PROTOCOL_VERSION: &str = "2025-11-25";

const SESSION_HEADER: &str = "mcp-session-id";
const VERSION_HEADER: &str = "mcp-protocol-version";

/// Guards against a server that pages `tools/list` forever.
const MAX_TOOL_PAGES: usize = 50;

pub struct McpClient {
    http: reqwest::Client,
    endpoint: String,
    auth: HeaderMap,
    next_id: AtomicU64,
    session: Mutex<SessionState>,
}

#[derive(Default)]
struct SessionState {
    id: Option<String>,
    /// The version the server agreed to, echoed on every later request.
    version: Option<String>,
    ready: bool,
}

impl McpClient {
    /// `auth` carries the credential headers for this connection, already
    /// resolved. The client never reads a token from anywhere else.
    pub fn new(http: reqwest::Client, endpoint: impl Into<String>, auth: HeaderMap) -> Self {
        Self {
            http,
            endpoint: endpoint.into(),
            auth,
            next_id: AtomicU64::new(1),
            session: Mutex::new(SessionState::default()),
        }
    }

    /// Every tool the connection offers, following `nextCursor` to the end.
    pub async fn list_tools(&self) -> Result<Vec<RemoteTool>, ConnectorError> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_TOOL_PAGES {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let result = self.request("tools/list", params).await?;
            let page: ToolsList = parse_result(result)?;
            tools.extend(page.tools.into_iter().map(RemoteTool::from));
            match page.next_cursor {
                Some(next) if !next.is_empty() => cursor = Some(next),
                _ => return Ok(tools),
            }
        }
        Err(ConnectorError::permanent(format!(
            "connection paged past {MAX_TOOL_PAGES} tool pages"
        )))
    }

    /// Call `name` with `arguments`. A tool that fails on the far side comes
    /// back as `ToolOutcome { is_error: true }`, not an `Err` — only reaching
    /// the connection at all is this function's concern.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<ToolOutcome, ConnectorError> {
        let params = json!({ "name": name, "arguments": arguments });
        let result = self.request("tools/call", params).await?;
        let call: CallResult = parse_result(result)?;
        Ok(ToolOutcome {
            content: render_content(&call.content),
            structured: call.structured_content,
            is_error: call.is_error,
        })
    }

    /// One JSON-RPC request, initializing first if needed. A server that has
    /// forgotten our session (404) gets one fresh handshake and one retry;
    /// beyond that the failure is real.
    async fn request(&self, method: &str, params: Value) -> Result<Value, ConnectorError> {
        self.ensure_ready().await?;
        match self.request_once(method, &params).await {
            Err(err) if err.session_expired => {
                self.session.lock().await.ready = false;
                self.ensure_ready().await?;
                self.request_once(method, &params)
                    .await
                    .map_err(|e| e.error)
            }
            Err(err) => Err(err.error),
            Ok(value) => Ok(value),
        }
    }

    async fn request_once(&self, method: &str, params: &Value) -> Result<Value, RequestFailure> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let (session, version) = {
            let state = self.session.lock().await;
            (state.id.clone(), state.version.clone())
        };
        let response = self
            .post(&body, session.as_deref(), version.as_deref())
            .await?;
        let (status, _, payload) = response;
        read_rpc_result(status, payload, id).map_err(RequestFailure::from)
    }

    /// Handshake: `initialize`, then the `notifications/initialized` the server
    /// waits for before accepting anything else.
    async fn ensure_ready(&self) -> Result<(), ConnectorError> {
        let mut state = self.session.lock().await;
        if state.ready {
            return Ok(());
        }
        state.id = None;
        state.version = None;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "substructure", "version": env!("CARGO_PKG_VERSION") },
            }
        });
        let (status, headers, payload) = self.post(&body, None, None).await.map_err(|f| f.error)?;
        let result = read_rpc_result(status, payload, id)?;
        let init: InitializeResult = parse_result(result)?;

        state.id = headers
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        state.version = Some(init.protocol_version);

        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        self.post(&note, state.id.as_deref(), state.version.as_deref())
            .await
            .map_err(|f| f.error)?;

        state.ready = true;
        Ok(())
    }

    async fn post(
        &self,
        body: &Value,
        session: Option<&str>,
        version: Option<&str>,
    ) -> Result<(StatusCode, HeaderMap, Payload), RequestFailure> {
        let mut req = self
            .http
            .post(&self.endpoint)
            .headers(self.auth.clone())
            // The spec requires both: the server picks one to answer with.
            .header(ACCEPT, "application/json, text/event-stream")
            .json(body);
        if let Some(session) = session {
            req = req.header(SESSION_HEADER, session);
        }
        if let Some(version) = version {
            req = req.header(VERSION_HEADER, version);
        }

        let response = req
            .send()
            .await
            .map_err(|e| ConnectorError::retryable(format!("connection unreachable: {e}")))?;

        let status = response.status();
        let headers = response.headers().clone();

        if status == StatusCode::NOT_FOUND && session.is_some() {
            return Err(RequestFailure {
                error: ConnectorError::retryable("connection session expired"),
                session_expired: true,
            });
        }
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            // The registry knows which credential went out. It corrects this.
            return Err(ConnectorError::unauthorized(
                AuthNeed::Reauthorize,
                format!("connection rejected the credential ({status})"),
            )
            .into());
        }
        // A notification is answered with 202 and no body.
        if status == StatusCode::ACCEPTED {
            return Ok((status, headers, Payload::Empty));
        }
        if status.is_server_error() {
            return Err(ConnectorError::retryable(format!("connection failed ({status})")).into());
        }

        let is_sse = headers
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));

        let payload = if is_sse {
            Payload::Stream(collect_sse(response).await?)
        } else {
            let text = response
                .text()
                .await
                .map_err(|e| ConnectorError::retryable(format!("unreadable response: {e}")))?;
            Payload::Json(text)
        };

        if !status.is_success() {
            return Err(ConnectorError::permanent(format!(
                "connection rejected the request ({status})"
            ))
            .into());
        }
        Ok((status, headers, payload))
    }
}

enum Payload {
    Json(String),
    /// The `data` field of every SSE event, in order.
    Stream(Vec<String>),
    Empty,
}

/// A failure plus whether it was specifically a dropped session, which the
/// caller answers by re-initializing rather than by giving up.
struct RequestFailure {
    error: ConnectorError,
    session_expired: bool,
}

impl From<ConnectorError> for RequestFailure {
    fn from(error: ConnectorError) -> Self {
        Self {
            error,
            session_expired: false,
        }
    }
}

/// Read the SSE stream to its end, keeping each event's data. The response to
/// our request is one of them; the spec allows unrelated server messages first.
async fn collect_sse(response: reqwest::Response) -> Result<Vec<String>, ConnectorError> {
    use eventsource_stream::Eventsource;

    let mut events = response.bytes_stream().eventsource();
    let mut data = Vec::new();
    while let Some(event) = events.next().await {
        match event {
            Ok(event) if !event.data.is_empty() => data.push(event.data),
            Ok(_) => {}
            Err(e) => return Err(ConnectorError::retryable(format!("stream broke: {e}"))),
        }
    }
    Ok(data)
}

/// Pull the JSON-RPC result for `want_id` out of whichever shape the server
/// answered with.
fn read_rpc_result(
    status: StatusCode,
    payload: Payload,
    want_id: u64,
) -> Result<Value, ConnectorError> {
    let candidates = match payload {
        Payload::Json(text) => vec![text],
        Payload::Stream(events) => events,
        Payload::Empty => {
            return Err(ConnectorError::permanent(format!(
                "connection answered {status} with no body"
            )))
        }
    };

    let mut rpc_error = None;
    for raw in &candidates {
        let Ok(message) = serde_json::from_str::<RpcResponse>(raw) else {
            continue;
        };
        if message.id.as_ref().and_then(Value::as_u64) != Some(want_id) {
            continue;
        }
        if let Some(err) = message.error {
            rpc_error = Some(err);
            break;
        }
        return Ok(message.result.unwrap_or(Value::Null));
    }

    match rpc_error {
        Some(err) => Err(ConnectorError::permanent(format!(
            "connection returned an error: {} ({})",
            err.message, err.code
        ))),
        None => Err(ConnectorError::retryable(
            "connection sent no response to the request",
        )),
    }
}

fn parse_result<T: for<'de> Deserialize<'de>>(result: Value) -> Result<T, ConnectorError> {
    serde_json::from_value(result).map_err(|e| {
        ConnectorError::permanent(format!("connection sent an unreadable result: {e}"))
    })
}

/// Content blocks flattened for the model. Text passes through; anything else
/// is named rather than dropped, so a tool answering with an image doesn't look
/// like it answered with nothing.
fn render_content(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(|b| match (&b.text, b.kind.as_str()) {
            (Some(text), "text") => text.clone(),
            (_, kind) => format!("[{kind} content]"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the credential headers for a connection. `Authorization: Bearer` is the
/// default; a connection may name its own header instead (Sentry uses one).
pub fn auth_headers(header: Option<&str>, token: &str) -> Result<HeaderMap, ConnectorError> {
    let name = header.unwrap_or("authorization");
    let value = if header.is_none() {
        format!("Bearer {token}")
    } else {
        token.to_string()
    };
    let name = HeaderName::try_from(name)
        .map_err(|_| ConnectorError::permanent(format!("`{name}` is not a valid header name")))?;
    let mut value = HeaderValue::try_from(value)
        .map_err(|_| ConnectorError::permanent("credential is not a valid header value"))?;
    value.set_sensitive(true);
    let mut headers = HeaderMap::new();
    headers.insert(name, value);
    Ok(headers)
}

// ── Wire types ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RpcResponse {
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<RpcError>,
}

#[derive(Deserialize)]
struct RpcError {
    code: i64,
    message: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeResult {
    protocol_version: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolsList {
    #[serde(default)]
    tools: Vec<WireTool>,
    #[serde(default)]
    next_cursor: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireTool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    input_schema: Option<Value>,
    #[serde(default)]
    output_schema: Option<Value>,
    #[serde(default)]
    annotations: Option<WireAnnotations>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireAnnotations {
    #[serde(default)]
    read_only_hint: Option<bool>,
    #[serde(default)]
    destructive_hint: Option<bool>,
    #[serde(default)]
    idempotent_hint: Option<bool>,
    #[serde(default)]
    open_world_hint: Option<bool>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CallResult {
    #[serde(default)]
    content: Vec<ContentBlock>,
    #[serde(default)]
    structured_content: Option<Value>,
    #[serde(default)]
    is_error: bool,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

impl From<WireTool> for RemoteTool {
    fn from(t: WireTool) -> Self {
        RemoteTool {
            name: t.name,
            description: t.description,
            input: t.input_schema,
            output: t.output_schema,
            annotations: t.annotations.map(ToolAnnotations::from).unwrap_or_default(),
        }
    }
}

impl From<WireAnnotations> for ToolAnnotations {
    fn from(a: WireAnnotations) -> Self {
        ToolAnnotations {
            read_only: a.read_only_hint,
            destructive: a.destructive_hint,
            idempotent: a.idempotent_hint,
            open_world: a.open_world_hint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use axum::Router;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Mutex as StdMutex};

    #[derive(Default)]
    struct Mock {
        /// Answer with SSE instead of a single JSON body.
        sse: bool,
        /// Drop the session once, so the client has to re-handshake.
        expire_once: AtomicBool,
        reject_credential: bool,
        /// Page `tools/list` once before returning the last page.
        paginate: bool,
        initializes: AtomicUsize,
        seen: StdMutex<Vec<SeenRequest>>,
    }

    /// method, session header, protocol-version header
    type SeenRequest = (String, Option<String>, Option<String>);

    impl Mock {
        fn record(&self, method: &str, headers: &axum::http::HeaderMap) {
            let get = |n: &str| {
                headers
                    .get(n)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string)
            };
            self.seen.lock().unwrap().push((
                method.to_string(),
                get(SESSION_HEADER),
                get(VERSION_HEADER),
            ));
        }

        fn methods(&self) -> Vec<String> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|(m, _, _)| m.clone())
                .collect()
        }
    }

    fn ok_body(id: &Value, result: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id, "result": result })
    }

    async fn handle(
        State(mock): State<Arc<Mock>>,
        headers: axum::http::HeaderMap,
        body: String,
    ) -> Response {
        let req: Value = serde_json::from_str(&body).expect("test sends json");
        let method = req["method"].as_str().unwrap_or_default().to_string();
        mock.record(&method, &headers);

        if mock.reject_credential {
            return (StatusCode::UNAUTHORIZED, "nope").into_response();
        }
        // Notifications get 202 and no body.
        let Some(id) = req.get("id") else {
            return StatusCode::ACCEPTED.into_response();
        };

        if method == "initialize" {
            mock.initializes.fetch_add(1, Ordering::Relaxed);
            let n = mock.initializes.load(Ordering::Relaxed);
            let mut resp = ok_body(
                id,
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "serverInfo": { "name": "mock", "version": "1" }
                }),
            )
            .to_string()
            .into_response();
            resp.headers_mut().insert(
                HeaderName::from_static(SESSION_HEADER),
                HeaderValue::try_from(format!("session-{n}")).unwrap(),
            );
            return resp;
        }

        if mock.expire_once.swap(false, Ordering::Relaxed) {
            return (StatusCode::NOT_FOUND, "gone").into_response();
        }

        let result = match method.as_str() {
            "tools/list" => {
                let first = req["params"].get("cursor").is_none();
                if mock.paginate && first {
                    json!({
                        "tools": [{ "name": "page_one", "inputSchema": { "type": "object" } }],
                        "nextCursor": "c1"
                    })
                } else {
                    json!({ "tools": [{
                        "name": "search_issues",
                        "description": "Search issues.",
                        "inputSchema": { "type": "object", "properties": { "q": { "type": "string" } } },
                        "annotations": { "readOnlyHint": true, "destructiveHint": false }
                    }] })
                }
            }
            "tools/call" => {
                if req["params"]["name"] == json!("boom") {
                    json!({ "content": [{ "type": "text", "text": "it failed" }], "isError": true })
                } else {
                    json!({
                        "content": [
                            { "type": "text", "text": "found 2" },
                            { "type": "image", "data": "..." }
                        ],
                        "structuredContent": { "count": 2 }
                    })
                }
            }
            other => {
                return serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32601, "message": format!("no method {other}") }
                })
                .to_string()
                .into_response()
            }
        };

        if mock.sse {
            // A server message before the response, as the spec permits.
            let body = format!(
                "event: message\ndata: {}\n\nevent: message\ndata: {}\n\n",
                json!({ "jsonrpc": "2.0", "method": "notifications/progress" }),
                ok_body(id, result)
            );
            ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
        } else {
            ok_body(id, result).to_string().into_response()
        }
    }

    async fn serve(mock: Arc<Mock>) -> String {
        let app = Router::new().route("/mcp", post(handle)).with_state(mock);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}/mcp")
    }

    async fn client_for(mock: Arc<Mock>) -> McpClient {
        let url = serve(mock).await;
        McpClient::new(
            reqwest::Client::new(),
            url,
            auth_headers(None, "tok").unwrap(),
        )
    }

    #[tokio::test]
    async fn a_json_server_lists_tools_with_their_annotations() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock.clone()).await;
        let tools = client.list_tools().await.expect("lists");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search_issues");
        assert_eq!(tools[0].annotations.read_only, Some(true));
        assert_eq!(tools[0].annotations.destructive, Some(false));
        assert_eq!(
            tools[0].annotations.idempotent, None,
            "an unstated hint stays unknown rather than becoming false"
        );
        assert_eq!(
            mock.methods(),
            vec!["initialize", "notifications/initialized", "tools/list"],
            "the handshake precedes the first call"
        );
    }

    #[tokio::test]
    async fn an_sse_server_is_read_the_same_way() {
        let mock = Arc::new(Mock {
            sse: true,
            ..Default::default()
        });
        let client = client_for(mock).await;
        let tools = client.list_tools().await.expect("lists");
        assert_eq!(
            tools[0].name, "search_issues",
            "the response is found past unrelated server messages"
        );
    }

    #[tokio::test]
    async fn the_session_id_is_carried_after_the_handshake() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock.clone()).await;
        client.list_tools().await.expect("lists");

        let seen = mock.seen.lock().unwrap().clone();
        let (_, init_session, init_version) = &seen[0];
        assert_eq!(*init_session, None, "initialize carries no session yet");
        assert_eq!(*init_version, None);
        for (method, session, version) in &seen[1..] {
            assert_eq!(
                session.as_deref(),
                Some("session-1"),
                "{method} must echo the session"
            );
            assert_eq!(version.as_deref(), Some(PROTOCOL_VERSION));
        }
    }

    #[tokio::test]
    async fn a_dropped_session_is_re_established_once() {
        let mock = Arc::new(Mock::default());
        mock.expire_once.store(true, Ordering::Relaxed);
        let client = client_for(mock.clone()).await;

        let tools = client.list_tools().await.expect("recovers from a 404");
        assert_eq!(tools[0].name, "search_issues");
        assert_eq!(
            mock.initializes.load(Ordering::Relaxed),
            2,
            "the client re-handshakes rather than failing the call"
        );
    }

    #[tokio::test]
    async fn pages_are_followed_to_the_end() {
        let mock = Arc::new(Mock {
            paginate: true,
            ..Default::default()
        });
        let client = client_for(mock).await;
        let tools = client.list_tools().await.expect("lists");
        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["page_one", "search_issues"]);
    }

    #[tokio::test]
    async fn a_call_renders_content_and_keeps_structured_output() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock).await;
        let outcome = client
            .call_tool("search_issues", &json!({ "q": "crash" }))
            .await
            .expect("calls");

        assert_eq!(outcome.content, "found 2\n[image content]");
        assert_eq!(outcome.structured, Some(json!({ "count": 2 })));
        assert!(!outcome.is_error);
    }

    #[tokio::test]
    async fn a_tool_that_fails_is_an_outcome_not_a_transport_error() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock).await;
        let outcome = client
            .call_tool("boom", &json!({}))
            .await
            .expect("reaches the server");
        assert!(
            outcome.is_error,
            "the far side failed, the connection did not"
        );
        assert_eq!(outcome.content, "it failed");
    }

    #[tokio::test]
    async fn a_rejected_credential_asks_for_re_auth() {
        let mock = Arc::new(Mock {
            reject_credential: true,
            ..Default::default()
        });
        let client = client_for(mock).await;
        let err = client.list_tools().await.unwrap_err();
        assert_eq!(
            err.auth,
            Some(AuthNeed::Reauthorize),
            "401 is a re-auth signal, not a plain failure; the registry refines it \
             against what the file declares"
        );
        assert!(
            !err.retryable,
            "retrying with the same rejected credential is pointless"
        );
    }

    #[tokio::test]
    async fn an_unreachable_connection_is_retryable() {
        let client = McpClient::new(
            reqwest::Client::new(),
            "http://127.0.0.1:1/mcp",
            HeaderMap::new(),
        );
        let err = client.list_tools().await.unwrap_err();
        assert!(err.retryable);
    }

    #[test]
    fn a_named_header_carries_the_token_verbatim() {
        let bearer = auth_headers(None, "tok").unwrap();
        assert_eq!(bearer.get("authorization").unwrap(), "Bearer tok");

        let named = auth_headers(Some("sentry-bearer"), "tok").unwrap();
        assert_eq!(
            named.get("sentry-bearer").unwrap(),
            "tok",
            "a connection naming its own header gets the raw token"
        );
        assert!(
            named.get("sentry-bearer").unwrap().is_sensitive(),
            "credentials are not logged"
        );
    }
}
