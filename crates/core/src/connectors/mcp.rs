use std::collections::HashMap;
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientInfo, ContentBlock,
    Implementation, ProtocolVersion, ResourceContents, Tool,
};
use rmcp::service::{ClientLifecycleMode, ClientServiceExt, RoleClient, RunningService};
use rmcp::transport::streamable_http_client::{
    SseError, StreamableHttpClient, StreamableHttpClientTransport,
    StreamableHttpClientTransportConfig, StreamableHttpError, StreamableHttpPostResponse,
};
use rmcp::ServiceError;
use serde_json::Value;
use sse_stream::Sse;
use tokio::sync::Mutex;

use crate::runtime::blob::{text_like, BlobStore, NewBlob};

use crate::protocol::{StoredContent, StoredResult};

use super::{AuthNeed, ConnectorError, CredentialSource, RemoteTool, ToolAnnotations};

const PREFERRED: ProtocolVersion = ProtocolVersion::V_2026_07_28;
const LEGACY: ProtocolVersion = ProtocolVersion::V_2025_11_25;

pub struct McpClient {
    endpoint: String,
    http: Credentialed,
    blobs: Arc<dyn BlobStore>,
    tenant_id: String,
    service: Mutex<Option<Arc<Running>>>,
}

type Running = RunningService<RoleClient, ClientInfo>;

impl McpClient {
    pub fn new(
        http: reqwest::Client,
        endpoint: impl Into<String>,
        auth: Arc<dyn CredentialSource>,
        blobs: Arc<dyn BlobStore>,
        tenant_id: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            http: Credentialed { http, auth },
            blobs,
            tenant_id: tenant_id.into(),
            service: Mutex::new(None),
        }
    }

    pub async fn instructions(&self) -> Option<String> {
        let service = self.service.lock().await;
        service
            .as_ref()
            .and_then(|s| s.peer_info())
            .and_then(|info| info.instructions.clone())
    }

    pub async fn server_title(&self) -> Option<String> {
        let service = self.service.lock().await;
        service
            .as_ref()
            .and_then(|s| s.peer_info())
            .and_then(|info| info.server_info.as_ref()?.title.clone())
    }

    pub async fn list_tools(&self) -> Result<Vec<RemoteTool>, ConnectorError> {
        let service = self.connect().await?;
        match service.list_all_tools().await {
            Ok(tools) => Ok(tools.into_iter().map(remote_tool).collect()),
            Err(e) => Err(self.lower(e).await),
        }
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<StoredResult, ConnectorError> {
        let service = self.connect().await?;
        let mut params = CallToolRequestParams::new(name.to_string());
        params.arguments = arguments.as_object().cloned();
        match service.call_tool(params).await {
            Ok(result) => Ok(self.outcome(result).await),
            Err(e) => Err(self.lower(e).await),
        }
    }

    async fn outcome(&self, result: CallToolResult) -> StoredResult {
        let mut content = Vec::new();
        for block in &result.content {
            match block {
                ContentBlock::Text(t) => content.push(StoredContent::Text {
                    text: t.text.clone(),
                }),
                ContentBlock::Image(i) => {
                    self.keep(&i.data, &i.mime_type, None, &mut content).await
                }
                ContentBlock::Audio(a) => {
                    self.keep(&a.data, &a.mime_type, None, &mut content).await
                }
                ContentBlock::Resource(r) => match &r.resource {
                    ResourceContents::TextResourceContents { text, .. } => {
                        content.push(StoredContent::Text { text: text.clone() })
                    }
                    ResourceContents::BlobResourceContents {
                        blob,
                        mime_type,
                        uri,
                        ..
                    } => {
                        let mime = mime_type.as_deref().unwrap_or("application/octet-stream");
                        self.keep(blob, mime, Some(uri), &mut content).await
                    }
                    _ => content.push(StoredContent::Text {
                        text: "[resource content]".to_string(),
                    }),
                },
                ContentBlock::ResourceLink(r) => content.push(StoredContent::Link {
                    uri: r.uri.clone(),
                    name: Some(r.name.clone()),
                    mime_type: r.mime_type.clone(),
                }),
                _ => content.push(StoredContent::Text {
                    text: "[content]".to_string(),
                }),
            }
        }
        StoredResult {
            content,
            structured_content: result.structured_content,
            is_error: result.is_error.unwrap_or(false),
        }
    }

    async fn keep(&self, data: &str, mime: &str, uri: Option<&str>, out: &mut Vec<StoredContent>) {
        let named = mime.split('/').next().unwrap_or("file").to_string();
        let mut note = |text: String| out.push(StoredContent::Text { text });
        let Ok(bytes) = BASE64.decode(data) else {
            note(format!("[unreadable {named} content]"));
            return;
        };
        if text_like(mime) {
            match String::from_utf8(bytes) {
                Ok(text) => note(text),
                Err(_) => note(format!("[unreadable {named} content]")),
            }
            return;
        }
        let name = uri.map(|u| u.rsplit('/').next().unwrap_or(u).to_string());
        let blob = NewBlob {
            tenant_id: self.tenant_id.clone(),
            mime: mime.to_string(),
            name: name.clone(),
            bytes,
        };
        match self.blobs.put(blob).await {
            Ok(r) => out.push(StoredContent::Blob { uri: r.uri() }),
            Err(e) => {
                tracing::warn!("keeping {named} content from a tool failed: {e}");
                note(format!("[{named} content]"));
            }
        }
    }

    async fn connect(&self) -> Result<Arc<Running>, ConnectorError> {
        let mut slot = self.service.lock().await;
        if let Some(service) = slot.as_ref() {
            return Ok(service.clone());
        }
        let transport = StreamableHttpClientTransport::with_client(
            self.http.clone(),
            StreamableHttpClientTransportConfig::with_uri(self.endpoint.clone()),
        );
        let mut info = ClientInfo::default();
        info.protocol_version = PREFERRED;
        info.capabilities = ClientCapabilities::default();
        info.client_info = Implementation::new("substructure", env!("CARGO_PKG_VERSION"));
        let service = info
            .serve_with_lifecycle(
                transport,
                ClientLifecycleMode::Auto {
                    preferred_versions: vec![PREFERRED, LEGACY],
                    legacy_version: Some(LEGACY),
                },
            )
            .await
            .map_err(initialize_error)?;
        let service = Arc::new(service);
        *slot = Some(service.clone());
        Ok(service)
    }

    async fn lower(&self, error: ServiceError) -> ConnectorError {
        if matches!(
            error,
            ServiceError::TransportClosed | ServiceError::TransportSend(_)
        ) {
            *self.service.lock().await = None;
        }
        service_error(error)
    }
}

#[derive(Clone)]
struct Credentialed {
    http: reqwest::Client,
    auth: Arc<dyn CredentialSource>,
}

impl Credentialed {
    async fn headers(
        &self,
        mut custom: Headers,
    ) -> Result<(Option<String>, Headers), StreamableHttpError<WireError>> {
        let headers = self
            .auth
            .headers()
            .await
            .map_err(|e| StreamableHttpError::Client(WireError::Credential(e)))?;
        let mut bearer = None;
        for (name, value) in headers.iter() {
            let token = (name == AUTHORIZATION)
                .then(|| value.to_str().ok())
                .flatten()
                .and_then(|v| v.strip_prefix("Bearer "));
            match token {
                Some(token) => bearer = Some(token.to_string()),
                None => {
                    custom.insert(name.clone(), value.clone());
                }
            }
        }
        Ok((bearer, custom))
    }
}

type Headers = HashMap<HeaderName, HeaderValue>;

impl StreamableHttpClient for Credentialed {
    type Error = WireError;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: rmcp::model::ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        _auth_header: Option<String>,
        custom_headers: Headers,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let (auth, custom) = self.headers(custom_headers).await?;
        self.http
            .post_message(uri, message, session_id, auth, custom)
            .await
            .map_err(lift)
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: rmcp::model::ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        _auth_header: Option<String>,
        custom_headers: Headers,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let (auth, custom) = self.headers(custom_headers).await?;
        self.http
            .post_message_with_max_sse_event_size(
                uri,
                message,
                session_id,
                auth,
                custom,
                max_sse_event_size,
            )
            .await
            .map_err(lift)
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        _auth_header: Option<String>,
        custom_headers: Headers,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        let (auth, custom) = self.headers(custom_headers).await?;
        self.http
            .delete_session(uri, session_id, auth, custom)
            .await
            .map_err(lift)
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        _auth_header: Option<String>,
        custom_headers: Headers,
    ) -> Result<
        futures_util::stream::BoxStream<'static, Result<Sse, SseError>>,
        StreamableHttpError<Self::Error>,
    > {
        let (auth, custom) = self.headers(custom_headers).await?;
        self.http
            .get_stream(uri, session_id, last_event_id, auth, custom)
            .await
            .map_err(lift)
    }
}

#[derive(Debug, thiserror::Error)]
enum WireError {
    #[error("{0}")]
    Credential(ConnectorError),
    #[error("{0}")]
    Http(reqwest::Error),
}

fn lift(error: StreamableHttpError<reqwest::Error>) -> StreamableHttpError<WireError> {
    use StreamableHttpError as E;
    match error {
        E::Client(e) => E::Client(WireError::Http(e)),
        E::Sse(e) => E::Sse(e),
        E::Io(e) => E::Io(e),
        E::Deserialize(e) => E::Deserialize(e),
        E::TokioJoinError(e) => E::TokioJoinError(e),
        E::UnexpectedEndOfStream => E::UnexpectedEndOfStream,
        E::UnexpectedServerResponse(s) => E::UnexpectedServerResponse(s),
        E::UnexpectedContentType(s) => E::UnexpectedContentType(s),
        E::ServerDoesNotSupportSse => E::ServerDoesNotSupportSse,
        E::ServerDoesNotSupportDeleteSession => E::ServerDoesNotSupportDeleteSession,
        E::TransportChannelClosed => E::TransportChannelClosed,
        E::MissingSessionIdInResponse => E::MissingSessionIdInResponse,
        E::AuthRequired(e) => E::AuthRequired(e),
        E::InsufficientScope(e) => E::InsufficientScope(e),
        E::ReservedHeaderConflict(s) => E::ReservedHeaderConflict(s),
        E::SessionExpired => E::SessionExpired,
        other => E::UnexpectedServerResponse(other.to_string().into()),
    }
}

fn initialize_error(error: rmcp::service::ClientInitializeError) -> ConnectorError {
    use rmcp::service::ClientInitializeError;
    match error {
        ClientInitializeError::TransportError { error, .. } => {
            transport_error(error.error.as_ref())
        }
        ClientInitializeError::ConnectionClosed(_) => {
            ConnectorError::retryable("connection closed before it answered")
        }
        other => ConnectorError::permanent(format!("connection could not be opened: {other}")),
    }
}

fn service_error(error: ServiceError) -> ConnectorError {
    match error {
        ServiceError::McpError(e) => {
            ConnectorError::permanent(format!("connection returned an error: {e}"))
        }
        ServiceError::TransportSend(e) => transport_error(e.error.as_ref()),
        ServiceError::TransportClosed => ConnectorError::retryable("connection closed"),
        ServiceError::Timeout { timeout } => {
            ConnectorError::retryable(format!("connection did not answer within {timeout:?}"))
        }
        ServiceError::InputRequiredRoundsExceeded { .. } => ConnectorError::permanent(
            "connection asked for input this client does not provide \
             (sampling, elicitation or roots)",
        ),
        other => ConnectorError::permanent(format!("connection failed: {other}")),
    }
}

fn transport_error(error: &(dyn std::error::Error + 'static)) -> ConnectorError {
    if let Some(wire) = error.downcast_ref::<StreamableHttpError<WireError>>() {
        return match wire {
            StreamableHttpError::AuthRequired(_) | StreamableHttpError::InsufficientScope(_) => {
                ConnectorError::unauthorized(
                    AuthNeed::Reauthorize,
                    "connection rejected the credential",
                )
            }
            StreamableHttpError::Client(WireError::Credential(e)) => e.clone(),
            StreamableHttpError::Client(WireError::Http(e)) => match e.status() {
                Some(status) if status.as_u16() == 401 || status.as_u16() == 403 => {
                    ConnectorError::unauthorized(
                        AuthNeed::Reauthorize,
                        format!("connection rejected the credential ({status})"),
                    )
                }
                _ => ConnectorError::retryable(format!("connection unreachable: {e}")),
            },
            StreamableHttpError::SessionExpired => {
                ConnectorError::retryable("connection session expired")
            }
            StreamableHttpError::Deserialize(e) => {
                ConnectorError::permanent(format!("connection sent an unreadable result: {e}"))
            }
            StreamableHttpError::UnexpectedServerResponse(message) => match status_of(message) {
                Some(401 | 403) => ConnectorError::unauthorized(
                    AuthNeed::Reauthorize,
                    format!("connection rejected the credential ({message})"),
                ),
                Some(code) if code < 500 => {
                    ConnectorError::permanent(format!("connection rejected the request: {message}"))
                }
                _ => ConnectorError::retryable(format!("connection failed: {message}")),
            },
            other => ConnectorError::retryable(format!("connection failed: {other}")),
        };
    }
    ConnectorError::retryable(format!("connection failed: {error}"))
}

fn status_of(message: &str) -> Option<u16> {
    message
        .strip_prefix("HTTP ")?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn remote_tool(tool: Tool) -> RemoteTool {
    RemoteTool {
        name: tool.name.into_owned(),
        title: tool.title,
        description: tool.description.map(|d| d.into_owned()).unwrap_or_default(),
        input: Some(Value::Object((*tool.input_schema).clone())),
        output: tool
            .output_schema
            .map(|schema| Value::Object((*schema).clone())),
        annotations: tool
            .annotations
            .map(|a| ToolAnnotations {
                read_only: a.read_only_hint,
                destructive: a.destructive_hint,
                idempotent: a.idempotent_hint,
                open_world: a.open_world_hint,
            })
            .unwrap_or_default(),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::blob::MemoryBlobStore;
    use axum::extract::State;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use axum::Router;
    use reqwest::header::CONTENT_TYPE;
    use reqwest::StatusCode;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};

    const MODERN: &str = "2026-07-28";
    const OLD: &str = "2025-11-25";
    const SESSION_HEADER: &str = "mcp-session-id";
    const VERSION_HEADER: &str = "mcp-protocol-version";
    const METHOD_HEADER: &str = "mcp-method";
    const NAME_HEADER: &str = "mcp-name";
    const METHOD_NOT_FOUND: i64 = -32601;

    #[derive(Default)]
    struct Mock {
        legacy: bool,
        offers: Option<Vec<String>>,
        sse: bool,
        no_session: bool,
        expire_once: AtomicBool,
        reject_credential: bool,
        paginate: bool,
        wants_input: bool,
        discovers: AtomicUsize,
        initializes: AtomicUsize,
        seen: StdMutex<Vec<Seen>>,
    }

    #[derive(Clone)]
    struct Seen {
        method: String,
        headers: axum::http::HeaderMap,
        body: Value,
    }

    impl Seen {
        fn header(&self, name: &str) -> Option<String> {
            self.headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        }
    }

    impl Mock {
        fn methods(&self) -> Vec<String> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|s| s.method.clone())
                .collect()
        }

        fn last(&self, method: &str) -> Seen {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .rev()
                .find(|s| s.method == method)
                .unwrap_or_else(|| panic!("no {method} was sent, saw {:?}", self.methods()))
                .clone()
        }
    }

    fn ok_body(id: &Value, mut result: Value) -> Value {
        result["resultType"] = json!("complete");
        if result.get("tools").is_some() {
            result["ttlMs"] = json!(0);
            result["cacheScope"] = json!("private");
        }
        json!({ "jsonrpc": "2.0", "id": id, "result": result })
    }

    fn json_ok(body: Value) -> Response {
        ([(CONTENT_TYPE, "application/json")], body.to_string()).into_response()
    }

    fn json_status(status: StatusCode, body: Value) -> Response {
        (
            status,
            [(CONTENT_TYPE, "application/json")],
            body.to_string(),
        )
            .into_response()
    }

    fn rpc_error(id: Option<&Value>, code: i64, data: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": id.cloned().unwrap_or(Value::Null),
            "error": { "code": code, "message": "no", "data": data }
        })
    }

    async fn handle(
        State(mock): State<Arc<Mock>>,
        headers: axum::http::HeaderMap,
        body: String,
    ) -> Response {
        let req: Value = serde_json::from_str(&body).expect("test sends json");
        let method = req["method"].as_str().unwrap_or_default().to_string();
        mock.seen.lock().unwrap().push(Seen {
            method: method.clone(),
            headers: headers.clone(),
            body: req.clone(),
        });

        if mock.reject_credential {
            return (StatusCode::UNAUTHORIZED, "nope").into_response();
        }
        let Some(id) = req.get("id") else {
            return StatusCode::ACCEPTED.into_response();
        };

        if method == "server/discover" {
            mock.discovers.fetch_add(1, Ordering::Relaxed);
            if mock.legacy {
                return json_status(
                    StatusCode::NOT_FOUND,
                    rpc_error(Some(id), METHOD_NOT_FOUND, json!("server/discover")),
                );
            }
            let offers = mock
                .offers
                .clone()
                .unwrap_or_else(|| vec![MODERN.to_string()]);
            return json_ok(ok_body(
                id,
                json!({
                    "supportedVersions": offers,
                    "capabilities": { "tools": {} },
                    "instructions": "A mock.",
                    "ttlMs": 0,
                    "cacheScope": "private",
                }),
            ));
        }

        if method == "initialize" {
            mock.initializes.fetch_add(1, Ordering::Relaxed);
            let n = mock.initializes.load(Ordering::Relaxed);
            let mut resp = json_ok(json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "protocolVersion": OLD,
                    "capabilities": {},
                    "serverInfo": { "name": "mock", "version": "1" },
                    "instructions": "A mock.",
                }
            }));
            if !mock.no_session {
                resp.headers_mut().insert(
                    reqwest::header::HeaderName::from_static(SESSION_HEADER),
                    reqwest::header::HeaderValue::try_from(format!("session-{n}")).unwrap(),
                );
            }
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
                    json!({ "tools": [
                        {
                            "name": "search_issues",
                            "description": "Search issues.",
                            "inputSchema": {
                                "type": "object",
                                "properties": { "q": { "type": "string" } }
                            },
                            "annotations": { "readOnlyHint": true, "destructiveHint": false }
                        },
                        {
                            "name": "execute_sql",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "region": { "type": "string", "x-mcp-header": "Region" },
                                    "query": { "type": "string" }
                                }
                            }
                        }
                    ] })
                }
            }
            "tools/call" => {
                if mock.wants_input {
                    return json_ok(json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {
                            "resultType": "input_required",
                            "inputRequests": [{
                                "method": "elicitation/create",
                                "params": { "message": "who?", "requestedSchema": {
                                    "type": "object", "properties": {}
                                } }
                            }]
                        }
                    }));
                }
                if req["params"]["name"] == json!("boom") {
                    json!({ "content": [{ "type": "text", "text": "it failed" }], "isError": true })
                } else {
                    json!({
                        "content": [
                            { "type": "text", "text": "found 2" },
                            { "type": "image", "data": "aGVsbG8=", "mimeType": "image/png" }
                        ],
                        "structuredContent": { "count": 2 }
                    })
                }
            }
            other => {
                return json_status(
                    StatusCode::NOT_FOUND,
                    rpc_error(Some(id), METHOD_NOT_FOUND, json!(other)),
                )
            }
        };

        if mock.sse {
            let body = format!("event: message\ndata: {}\n\n", ok_body(id, result));
            ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
        } else {
            json_ok(ok_body(id, result))
        }
    }

    struct Fixed(HeaderMap);

    #[async_trait::async_trait]
    impl CredentialSource for Fixed {
        async fn headers(&self) -> Result<HeaderMap, ConnectorError> {
            Ok(self.0.clone())
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
        client_at(url, Fixed(auth_headers(None, "tok").unwrap()))
    }

    fn client_at(url: String, auth: Fixed) -> McpClient {
        McpClient::new(
            reqwest::Client::new(),
            url,
            Arc::new(auth),
            Arc::new(MemoryBlobStore::new()),
            "t1",
        )
    }

    #[tokio::test]
    async fn a_modern_server_is_used_without_a_handshake() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock.clone()).await;
        let tools = client.list_tools().await.expect("lists");

        assert_eq!(tools[0].name, "search_issues");
        assert!(
            !mock.methods().contains(&"initialize".to_string()),
            "a stateless server must not be handshaken with, saw {:?}",
            mock.methods()
        );
        assert_eq!(client.instructions().await.as_deref(), Some("A mock."));
    }

    #[tokio::test]
    async fn a_legacy_server_falls_back_to_the_handshake() {
        let mock = Arc::new(Mock {
            legacy: true,
            ..Default::default()
        });
        let client = client_for(mock.clone()).await;
        let tools = client.list_tools().await.expect("lists");

        assert_eq!(tools[0].name, "search_issues");
        assert_eq!(
            mock.initializes.load(Ordering::Relaxed),
            1,
            "a refused probe costs one round trip, then the old handshake"
        );
        assert_eq!(client.instructions().await.as_deref(), Some("A mock."));

        let list = mock.last("tools/list");
        assert_eq!(list.header(SESSION_HEADER).as_deref(), Some("session-1"));
    }

    #[tokio::test]
    async fn the_era_is_settled_once_and_kept() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock.clone()).await;
        client.list_tools().await.expect("lists");
        client.list_tools().await.expect("lists again");

        assert_eq!(
            mock.discovers.load(Ordering::Relaxed),
            1,
            "the era is a property of the server, not of a request"
        );
    }

    #[tokio::test]
    async fn every_modern_request_carries_its_version_and_identity() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock.clone()).await;
        client.list_tools().await.expect("lists");

        let list = mock.last("tools/list");
        let meta = &list.body["params"]["_meta"];
        assert_eq!(
            meta["io.modelcontextprotocol/protocolVersion"],
            json!(MODERN)
        );
        assert_eq!(
            meta["io.modelcontextprotocol/clientInfo"]["name"],
            json!("substructure")
        );
        assert!(
            meta.get("io.modelcontextprotocol/clientCapabilities")
                .is_some(),
            "capabilities are required even when we declare none"
        );
        assert_eq!(list.header(VERSION_HEADER).as_deref(), Some(MODERN));
        assert_eq!(list.header(METHOD_HEADER).as_deref(), Some("tools/list"));
        assert_eq!(
            list.header(SESSION_HEADER),
            None,
            "sessions are not part of this revision"
        );
    }

    #[tokio::test]
    async fn a_call_names_its_tool_in_a_header() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock.clone()).await;
        client
            .call_tool("search_issues", &json!({ "q": "crash" }))
            .await
            .expect("calls");

        let call = mock.last("tools/call");
        assert_eq!(call.header(METHOD_HEADER).as_deref(), Some("tools/call"));
        assert_eq!(call.header(NAME_HEADER).as_deref(), Some("search_issues"));
    }

    #[tokio::test]
    async fn annotated_arguments_are_mirrored_into_headers() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock.clone()).await;
        client.list_tools().await.expect("lists");
        client
            .call_tool(
                "execute_sql",
                &json!({ "region": "us-west1", "query": "SELECT 1" }),
            )
            .await
            .expect("calls");

        let call = mock.last("tools/call");
        assert_eq!(
            call.header("mcp-param-Region").as_deref(),
            Some("us-west1"),
            "a server routing on the header sees what the body says"
        );
    }

    #[tokio::test]
    async fn a_result_asking_for_input_is_an_error_not_an_empty_answer() {
        let mock = Arc::new(Mock {
            wants_input: true,
            ..Default::default()
        });
        let client = client_for(mock).await;
        let err = client
            .call_tool("search_issues", &json!({}))
            .await
            .unwrap_err();
        assert!(
            !err.retryable,
            "the model must not be handed a blank result: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn a_call_renders_content_and_keeps_structured_output() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock).await;
        let outcome = client
            .call_tool("search_issues", &json!({ "q": "crash" }))
            .await
            .expect("calls");

        assert_eq!(
            outcome.as_text(),
            "found 2",
            "an image is not described in the text; it rides as its own block"
        );
        assert_eq!(outcome.structured_content, Some(json!({ "count": 2 })));
        assert!(!outcome.is_error);
    }

    #[tokio::test]
    async fn an_image_is_kept_rather_than_described() {
        let mock = Arc::new(Mock::default());
        let url = serve(mock).await;
        let blobs = Arc::new(MemoryBlobStore::new());
        let client = McpClient::new(
            reqwest::Client::new(),
            url,
            Arc::new(Fixed(auth_headers(None, "tok").unwrap())),
            blobs.clone(),
            "t1",
        );
        let outcome = client
            .call_tool("search_issues", &json!({}))
            .await
            .expect("calls");

        let StoredContent::Blob { uri } = &outcome.content[1] else {
            panic!(
                "the stored bytes become a blob block, got {:?}",
                outcome.content
            );
        };
        let r = crate::runtime::blob::BlobRef::parse(uri).expect("a ref");
        assert_eq!(r.mime, "image/png", "the ref carries what it is");
        assert_eq!(r.tenant_id, "t1");
        assert_eq!(
            blobs.get(&r).await.expect("stored"),
            b"hello".to_vec(),
            "the bytes reached the store, not the event log"
        );
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
        assert_eq!(outcome.as_text(), "it failed");
    }

    #[tokio::test]
    async fn annotations_survive_the_listing() {
        let mock = Arc::new(Mock::default());
        let client = client_for(mock).await;
        let tools = client.list_tools().await.expect("lists");
        assert_eq!(tools[0].annotations.read_only, Some(true));
        assert_eq!(tools[0].annotations.destructive, Some(false));
        assert_eq!(
            tools[0].annotations.idempotent, None,
            "an unstated hint stays unknown rather than becoming false"
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
        assert_eq!(tools[0].name, "search_issues");
    }

    #[tokio::test]
    async fn a_legacy_server_without_sessions_works() {
        let mock = Arc::new(Mock {
            legacy: true,
            no_session: true,
            ..Default::default()
        });
        let client = client_for(mock.clone()).await;
        client.list_tools().await.expect("lists");
        client.list_tools().await.expect("lists again");

        assert_eq!(
            mock.initializes.load(Ordering::Relaxed),
            1,
            "a server that mints no session has not forgotten one"
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
        assert_eq!(names, vec!["page_one", "search_issues", "execute_sql"]);
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
             against what the file declares. saw: {}",
            err.message
        );
        assert!(
            !err.retryable,
            "retrying with the same rejected credential is pointless"
        );
    }

    #[tokio::test]
    async fn an_unreachable_connection_is_retryable() {
        let client = client_at(
            "http://127.0.0.1:1/mcp".to_string(),
            Fixed(HeaderMap::new()),
        );
        let err = client.list_tools().await.unwrap_err();
        assert!(err.retryable, "{}", err.message);
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
        assert!(named.get("sentry-bearer").unwrap().is_sensitive());
    }

    #[tokio::test]
    async fn a_named_credential_header_reaches_the_server() {
        let mock = Arc::new(Mock::default());
        let url = serve(mock.clone()).await;
        let client = client_at(
            url,
            Fixed(auth_headers(Some("sentry-bearer"), "raw").unwrap()),
        );
        client.list_tools().await.expect("lists");
        assert_eq!(
            mock.last("tools/list").header("sentry-bearer").as_deref(),
            Some("raw"),
            "a connection that names its own header is not forced onto Bearer"
        );
    }
}
