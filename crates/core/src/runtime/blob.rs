use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use percent_encoding::{percent_decode_str, utf8_percent_encode, NON_ALPHANUMERIC};

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmResolver};
use crate::mime::essence;
use crate::protocol::{
    ClientPayload, Content, ContentPart, DraftMessage, ErrorCode, LlmRequest, LlmResponse,
    PromptContent, PromptMessage, PromptPart, PromptRequest, SessionOwner, StoredContent,
    StoredResult, ToolContent, ToolResult, OCTET_STREAM,
};

pub const BLOB_SCHEME: &str = "blob://";

/// One stored blob. Keyed by `(tenant_id, id)`: nothing resolves across
/// tenants, and a random id tells nothing about the bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobRef {
    pub tenant_id: String,
    /// A UUID the store mints at `put`; never derived from the bytes.
    pub id: String,
    pub mime: String,
    pub name: Option<String>,
    pub size: u64,
}

/// The `?…` half of a ref uri.
#[derive(serde::Serialize, serde::Deserialize)]
struct Query {
    mime: String,
    size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl BlobRef {
    /// `blob://{tenant}/{id}?mime=…&size=…&name=…`
    pub fn uri(&self) -> String {
        let query = serde_urlencoded::to_string(Query {
            mime: self.mime.clone(),
            size: self.size,
            name: self.name.clone(),
        })
        .expect("flat struct");
        format!("{BLOB_SCHEME}{}/{}?{query}", enc(&self.tenant_id), self.id)
    }

    pub fn parse(uri: &str) -> Option<Self> {
        let rest = uri.strip_prefix(BLOB_SCHEME)?;
        let (path, query) = rest.split_once('?')?;
        let (tenant, id) = path.split_once('/')?;
        uuid::Uuid::try_parse(id).ok()?;
        let q: Query = serde_urlencoded::from_str(query).ok()?;
        Some(Self {
            tenant_id: dec(tenant)?,
            id: id.to_string(),
            mime: q.mime,
            name: q.name,
            size: q.size,
        })
    }
}

fn enc(s: &str) -> String {
    utf8_percent_encode(s, NON_ALPHANUMERIC).to_string()
}

fn dec(s: &str) -> Option<String> {
    percent_decode_str(s).decode_utf8().ok().map(Into::into)
}

#[derive(Debug)]
pub enum BlobError {
    NotFound,
    Invalid(String),
    Io(String),
}

impl std::fmt::Display for BlobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlobError::NotFound => write!(f, "blob not found"),
            BlobError::Invalid(m) => write!(f, "invalid blob: {m}"),
            BlobError::Io(m) => write!(f, "blob io: {m}"),
        }
    }
}

impl std::error::Error for BlobError {}

/// Bytes going in; `put` mints the id.
pub struct NewBlob {
    pub tenant_id: String,
    pub mime: String,
    pub name: Option<String>,
    pub bytes: Vec<u8>,
}

#[async_trait]
pub trait BlobStore: Send + Sync {
    async fn put(&self, blob: NewBlob) -> Result<BlobRef, BlobError>;
    async fn get(&self, r: &BlobRef) -> Result<Vec<u8>, BlobError>;
}

#[cfg(test)]
pub(crate) struct MemoryBlobStore(std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>);

#[cfg(test)]
impl MemoryBlobStore {
    pub(crate) fn new() -> Self {
        Self(std::sync::Mutex::new(std::collections::HashMap::new()))
    }
}

#[cfg(test)]
#[async_trait]
impl BlobStore for MemoryBlobStore {
    async fn put(&self, blob: NewBlob) -> Result<BlobRef, BlobError> {
        let id = uuid::Uuid::now_v7().to_string();
        let size = blob.bytes.len() as u64;
        self.0.lock().unwrap().insert(id.clone(), blob.bytes);
        Ok(BlobRef {
            tenant_id: blob.tenant_id,
            id,
            mime: blob.mime,
            name: blob.name,
            size,
        })
    }

    async fn get(&self, r: &BlobRef) -> Result<Vec<u8>, BlobError> {
        self.0
            .lock()
            .unwrap()
            .get(&r.id)
            .cloned()
            .ok_or(BlobError::NotFound)
    }
}

impl From<ContentPart> for StoredContent {
    fn from(part: ContentPart) -> Self {
        let link = |uri: String, name: Option<String>, mime_type: Option<&str>| Self::Link {
            uri,
            name,
            mime_type: mime_type.map(str::to_string),
        };
        match part {
            ContentPart::Text { text } => Self::Text { text },
            ContentPart::ImageUrl { image_url } => link(image_url.url, None, Some("image/*")),
            ContentPart::VideoUrl { video_url } => link(video_url.url, None, Some("video/*")),
            ContentPart::File { file } => link(file.file_data, Some(file.filename), None),
            ContentPart::InputAudio { input_audio } => {
                let mime = format!("audio/{}", input_audio.format);
                let uri = format!("data:{mime};base64,{}", input_audio.data);
                link(uri, None, Some(&mime))
            }
        }
    }
}

pub async fn store(result: ToolResult, blobs: &dyn BlobStore, tenant_id: &str) -> StoredResult {
    let mut content = Vec::with_capacity(result.content.len());
    for block in result.content {
        content.push(match block {
            ToolContent::Text { text } => StoredContent::Text { text },
            ToolContent::ResourceLink {
                uri,
                name,
                mime_type,
            } => StoredContent::Link {
                uri,
                name,
                mime_type,
            },
            ToolContent::Image { data, mime_type } => {
                keep(data, mime_type, None, blobs, tenant_id).await
            }
            ToolContent::Audio { data, mime_type } => {
                keep(data, mime_type, None, blobs, tenant_id).await
            }
            ToolContent::Resource { resource } => match resource.blob {
                None => StoredContent::Text {
                    text: resource.text.unwrap_or_default(),
                },
                Some(data) => {
                    let name = resource
                        .uri
                        .rsplit('/')
                        .next()
                        .unwrap_or(&resource.uri)
                        .to_string();
                    let mime = resource
                        .mime_type
                        .unwrap_or_else(|| OCTET_STREAM.to_string());
                    keep(data, mime, Some(name), blobs, tenant_id).await
                }
            },
        });
    }
    StoredResult {
        content,
        structured_content: result.structured_content,
        is_error: result.is_error,
    }
}

async fn keep(
    data: String,
    mime: String,
    name: Option<String>,
    blobs: &dyn BlobStore,
    tenant_id: &str,
) -> StoredContent {
    match base64::engine::general_purpose::STANDARD.decode(&data) {
        Ok(bytes) => put(bytes, mime, name, blobs, tenant_id).await,
        Err(_) => unkept("unreadable", &mime),
    }
}

async fn put(
    bytes: Vec<u8>,
    mime: String,
    name: Option<String>,
    blobs: &dyn BlobStore,
    tenant_id: &str,
) -> StoredContent {
    match stash(bytes, mime.clone(), name, blobs, tenant_id).await {
        Ok(r) => StoredContent::Blob { uri: r.uri() },
        Err(_) => unkept("unstored", &mime),
    }
}

async fn stash(
    bytes: Vec<u8>,
    mime: String,
    name: Option<String>,
    blobs: &dyn BlobStore,
    tenant_id: &str,
) -> Result<BlobRef, BlobError> {
    blobs
        .put(NewBlob {
            tenant_id: tenant_id.to_string(),
            mime,
            name,
            bytes,
        })
        .await
        .inspect_err(|e| tracing::warn!("storing inline content failed: {e}"))
}

fn unkept(what: &str, mime: &str) -> StoredContent {
    StoredContent::Text {
        text: format!("[{what} {} content]", essence(mime)),
    }
}

pub async fn intern_payload(payload: &mut ClientPayload, blobs: &dyn BlobStore, tenant_id: &str) {
    match payload {
        ClientPayload::Message(m) => {
            intern(std::slice::from_mut(&mut m.message), blobs, tenant_id).await
        }
        ClientPayload::Messages(m) => intern(&mut m.messages, blobs, tenant_id).await,
        ClientPayload::Append(a) => intern(&mut a.messages, blobs, tenant_id).await,
        ClientPayload::Action(_) => {}
    }
}

pub async fn intern(messages: &mut [DraftMessage], blobs: &dyn BlobStore, tenant_id: &str) {
    for message in messages {
        let Some(Content::Parts(parts)) = &mut message.content else {
            continue;
        };
        for part in parts {
            let (uri, name) = match part {
                StoredContent::Blob { uri } => (uri, None),
                StoredContent::Link { uri, name, .. } => (uri, name.clone()),
                StoredContent::Text { .. } | StoredContent::Attachment(_) => continue,
            };
            let Some((mime, bytes)) = parse_data_uri(uri) else {
                continue;
            };
            *part = put(bytes, mime, name, blobs, tenant_id).await;
        }
    }
}

pub struct Nowhere;

pub static NOWHERE: Nowhere = Nowhere;

#[async_trait]
impl BlobStore for Nowhere {
    async fn put(&self, _: NewBlob) -> Result<BlobRef, BlobError> {
        Err(BlobError::NotFound)
    }
    async fn get(&self, _: &BlobRef) -> Result<Vec<u8>, BlobError> {
        Err(BlobError::NotFound)
    }
}

pub struct BlobResolvingLlm {
    inner: Arc<dyn LlmResolver>,
    blobs: Arc<dyn BlobStore>,
}

impl BlobResolvingLlm {
    pub fn new(inner: Arc<dyn LlmResolver>, blobs: Arc<dyn BlobStore>) -> Self {
        Self { inner, blobs }
    }
}

#[async_trait]
impl LlmResolver for BlobResolvingLlm {
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    async fn resolve(
        &self,
        llm: &str,
        owner: &SessionOwner,
    ) -> Result<Arc<dyn LlmCallable>, String> {
        let inner = self.inner.resolve(llm, owner).await?;
        Ok(Arc::new(BlobResolvingCallable {
            inner,
            blobs: self.blobs.clone(),
        }))
    }
}

struct BlobResolvingCallable {
    inner: Arc<dyn LlmCallable>,
    blobs: Arc<dyn BlobStore>,
}

pub async fn resolve(
    request: &LlmRequest,
    blobs: &dyn BlobStore,
    tenant_id: &str,
) -> Result<PromptRequest, LlmCallError> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        let content = match &message.content {
            None => None,
            Some(Content::Text(text)) => Some(PromptContent::Text(text.clone())),
            Some(Content::Parts(parts)) => {
                let mut out = Vec::with_capacity(parts.len());
                for part in parts {
                    out.push(prompt_part(part, blobs, tenant_id).await?);
                }
                Some(PromptContent::Parts(out))
            }
        };
        messages.push(PromptMessage {
            role: message.role.clone(),
            content,
            tool_calls: message.tool_calls.clone(),
            tool_call_id: message.tool_call_id.clone(),
            name: message.name.clone(),
            reasoning: message.reasoning.clone(),
        });
    }
    Ok(PromptRequest {
        model: request.model.clone(),
        messages,
        tools: request.tools.clone(),
        temperature: request.temperature,
        max_completion_tokens: request.max_completion_tokens,
        reasoning: request.reasoning.clone(),
    })
}

async fn prompt_part(
    part: &StoredContent,
    blobs: &dyn BlobStore,
    tenant_id: &str,
) -> Result<PromptPart, LlmCallError> {
    match part {
        StoredContent::Text { text } => Ok(PromptPart::Text { text: text.clone() }),
        StoredContent::Attachment(attachment) => Ok(PromptPart::Text {
            text: attachment.line(),
        }),
        StoredContent::Link {
            uri,
            mime_type,
            name,
        } => Ok(PromptPart::Link {
            uri: uri.clone(),
            name: name.clone(),
            mime_type: mime_type.clone(),
        }),
        StoredContent::Blob { uri } => {
            let r =
                BlobRef::parse(uri).ok_or_else(|| blob_call_error("unparsable blob ref", false))?;
            if r.tenant_id != tenant_id {
                return Err(blob_call_error("blob ref for another tenant", false));
            }
            let bytes = blobs
                .get(&r)
                .await
                .map_err(|e| blob_call_error(&e.to_string(), matches!(e, BlobError::Io(_))))?;
            Ok(PromptPart::Media {
                mime: r.mime,
                name: r.name,
                bytes,
            })
        }
    }
}

impl BlobResolvingCallable {
    /// Swap each generated image's `data:` URI for a stored ref. A store
    /// failure keeps the URI: the call succeeded, and inline bytes still work.
    async fn store_images(&self, response: &mut LlmResponse, ctx: &CallContext<'_>) {
        for img in &mut response.images {
            let Some((mime, bytes)) = parse_data_uri(&img.url) else {
                continue;
            };
            if let Ok(r) = stash(bytes, mime, None, self.blobs.as_ref(), ctx.tenant_id).await {
                img.url = r.uri();
            }
        }
    }
}

/// Decodes without requiring canonical padding; providers differ.
const B64_LENIENT: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    base64::engine::GeneralPurposeConfig::new()
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent),
);

/// `data:[<mime>][;base64],<data>` (RFC 2397). Only base64 bodies with an
/// explicit media type are taken; anything else stays inline.
fn parse_data_uri(url: &str) -> Option<(String, Vec<u8>)> {
    let rest = url.strip_prefix("data:")?;
    let (header, body) = rest.split_once(',')?;
    let marker = header.len().checked_sub(7)?;
    if !header[marker..].eq_ignore_ascii_case(";base64") {
        return None;
    }
    let mime = &header[..marker];
    if mime.is_empty() {
        return None;
    }
    let body: String = body.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let bytes = B64_LENIENT.decode(body).ok()?;
    Some((mime.to_string(), bytes))
}

fn blob_call_error(message: &str, retryable: bool) -> LlmCallError {
    LlmCallError::new(ErrorCode::Internal, message, retryable)
}

#[async_trait]
impl LlmCallable for BlobResolvingCallable {
    async fn call(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        let mut response = self.inner.call(request, ctx).await?;
        self.store_images(&mut response, ctx).await;
        Ok(response)
    }

    async fn call_streaming(
        &self,
        request: &PromptRequest,
        ctx: &CallContext<'_>,
        tx: tokio::sync::mpsc::UnboundedSender<crate::protocol::StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let mut response = self.inner.call_streaming(request, ctx, tx).await?;
        self.store_images(&mut response, ctx).await;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{DraftMessage, Role};

    const ID: &str = "0198b2a0-3c5d-7f00-8000-0123456789ab";

    #[tokio::test]
    async fn every_payload_that_carries_messages_stores_its_data_uris() {
        use crate::protocol::{ClientAppend, ClientMessage, ClientMessages};
        let file = || DraftMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Parts(vec![StoredContent::Link {
                uri: "data:text/csv;base64,YQ==".to_string(),
                name: Some("a.csv".to_string()),
                mime_type: None,
            }])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        };
        let payloads = [
            (
                "client.message",
                ClientPayload::Message(ClientMessage {
                    message: file(),
                    stream: false,
                }),
            ),
            (
                "client.messages",
                ClientPayload::Messages(ClientMessages {
                    messages: vec![file()],
                    stream: false,
                    client: Default::default(),
                }),
            ),
            (
                "client.append",
                ClientPayload::Append(ClientAppend {
                    messages: vec![file()],
                    stream: false,
                    client: Default::default(),
                }),
            ),
        ];
        for (kind, mut payload) in payloads {
            let blobs = MemoryBlobStore::new();
            intern_payload(&mut payload, &blobs, "t1").await;

            let messages = match &payload {
                ClientPayload::Message(m) => std::slice::from_ref(&m.message),
                ClientPayload::Messages(m) => m.messages.as_slice(),
                ClientPayload::Append(a) => a.messages.as_slice(),
                ClientPayload::Action(_) => panic!("{kind} carries no message"),
            };
            let Some(Content::Parts(parts)) = &messages[0].content else {
                panic!("{kind} lost its parts");
            };
            let [StoredContent::Blob { uri }] = parts.as_slice() else {
                panic!("{kind} kept {:?} instead of a blob", parts[0]);
            };
            let stored = BlobRef::parse(uri).expect("a parseable ref");
            assert_eq!(stored.tenant_id, "t1");
            assert_eq!(stored.mime, "text/csv");
            assert_eq!(stored.name.as_deref(), Some("a.csv"), "{kind}");
            assert_eq!(blobs.get(&stored).await.expect("stored"), b"a");
        }
    }

    fn blob_ref(tenant: &str) -> BlobRef {
        BlobRef {
            tenant_id: tenant.to_string(),
            id: ID.to_string(),
            mime: "image/png".to_string(),
            name: Some("a photo & more.png".to_string()),
            size: 4,
        }
    }

    #[test]
    fn uri_round_trips_with_escaped_fields() {
        let r = blob_ref("acme co/1");
        assert_eq!(BlobRef::parse(&r.uri()), Some(r));
        let bare = BlobRef {
            name: None,
            ..blob_ref("t1")
        };
        assert_eq!(BlobRef::parse(&bare.uri()), Some(bare));
    }

    #[test]
    fn parse_rejects_malformed_refs() {
        assert_eq!(BlobRef::parse("https://x/y"), None);
        assert_eq!(BlobRef::parse("blob://t1/not-a-uuid?mime=a&size=1"), None);
        assert_eq!(
            BlobRef::parse(&format!("blob://t1/{ID}?mime=image%2Fpng")),
            None
        );
        assert_eq!(BlobRef::parse("blob://t1?mime=a&size=1"), None);
    }

    #[tokio::test]
    async fn blobs_resolve_to_media_with_their_mime_bytes_and_name() {
        let blobs = MemoryBlobStore::new();
        let mut refs = Vec::new();
        for (mime, bytes) in [
            ("audio/mpeg", b"aaa".to_vec()),
            ("video/mp4", b"vvv".to_vec()),
            ("application/zip", b"zzz".to_vec()),
        ] {
            refs.push(
                blobs
                    .put(NewBlob {
                        tenant_id: "t1".into(),
                        mime: mime.into(),
                        name: Some("f".into()),
                        bytes,
                    })
                    .await
                    .expect("stored"),
            );
        }
        let mut request = request("unused");
        request.messages[0].content = Some(Content::Parts(
            refs.iter()
                .map(|r| StoredContent::Blob { uri: r.uri() })
                .collect(),
        ));

        let out = resolve(&request, &blobs, "t1").await.expect("resolves");
        let Some(PromptContent::Parts(parts)) = &out.messages[0].content else {
            panic!("parts");
        };
        let media = |mime: &str, bytes: &[u8]| PromptPart::Media {
            mime: mime.into(),
            name: Some("f".into()),
            bytes: bytes.to_vec(),
        };
        assert_eq!(
            parts,
            &[
                media("audio/mpeg", b"aaa"),
                media("video/mp4", b"vvv"),
                media("application/zip", b"zzz"),
            ],
            "nothing is decided here; the adapter sees the mime and the bytes"
        );
    }

    #[tokio::test]
    async fn links_reach_the_adapter_as_they_were_recorded() {
        let mut request = request("https://x/y.png");
        request.messages[0].content = Some(Content::Parts(vec![
            StoredContent::Link {
                uri: "https://cdn.example/a.png".into(),
                name: None,
                mime_type: Some("image/png".into()),
            },
            StoredContent::Link {
                uri: "https://docs.example/spec".into(),
                name: Some("spec".into()),
                mime_type: None,
            },
        ]));
        let out = resolve(&request, &FakeStore, "t1").await.expect("resolves");
        let Some(PromptContent::Parts(parts)) = &out.messages[0].content else {
            panic!("parts");
        };
        assert_eq!(
            parts,
            &[
                PromptPart::Link {
                    uri: "https://cdn.example/a.png".into(),
                    name: None,
                    mime_type: Some("image/png".into()),
                },
                PromptPart::Link {
                    uri: "https://docs.example/spec".into(),
                    name: Some("spec".into()),
                    mime_type: None,
                },
            ]
        );
    }

    #[tokio::test]
    async fn inline_bytes_are_stored_and_replaced_by_a_ref() {
        use crate::protocol::{ResourceContents, ToolContent, ToolResult};

        let blobs = MemoryBlobStore::new();
        let result = ToolResult {
            content: vec![
                ToolContent::Text {
                    text: "the chart".into(),
                },
                ToolContent::Image {
                    data: "aGVsbG8=".into(),
                    mime_type: "image/png".into(),
                },
                ToolContent::Resource {
                    resource: ResourceContents {
                        uri: "file:///a/report.pdf".into(),
                        mime_type: Some("application/pdf".into()),
                        text: None,
                        blob: Some("d29ybGQ=".into()),
                    },
                },
            ],
            ..Default::default()
        };
        let result = store(result, &blobs, "t1").await;

        let StoredContent::Blob { uri } = &result.content[1] else {
            panic!(
                "stored bytes become a blob block, got {:?}",
                result.content[1]
            );
        };
        let r = BlobRef::parse(uri).expect("a ref");
        assert_eq!(r.mime, "image/png", "the ref carries what it is");
        assert_eq!(r.tenant_id, "t1");
        assert_eq!(blobs.get(&r).await.expect("stored"), b"hello".to_vec());

        let StoredContent::Blob { uri } = &result.content[2] else {
            panic!("a resource with bytes becomes a blob block too");
        };
        let r = BlobRef::parse(uri).expect("a ref");
        assert_eq!(blobs.get(&r).await.expect("stored"), b"world".to_vec());
        assert_eq!(
            r.name.as_deref(),
            Some("report.pdf"),
            "the resource uri names the file"
        );

        assert!(
            matches!(&result.content[0], StoredContent::Text { text } if text == "the chart"),
            "text is left alone"
        );
    }

    #[tokio::test]
    async fn a_store_that_refuses_never_leaves_bytes_in_the_result() {
        use crate::protocol::{ToolContent, ToolResult};

        struct Refuses;
        #[async_trait]
        impl BlobStore for Refuses {
            async fn put(&self, _: NewBlob) -> Result<BlobRef, BlobError> {
                Err(BlobError::NotFound)
            }
            async fn get(&self, _: &BlobRef) -> Result<Vec<u8>, BlobError> {
                Err(BlobError::NotFound)
            }
        }

        let result = ToolResult {
            content: vec![ToolContent::Image {
                data: "aGVsbG8=".into(),
                mime_type: "image/png".into(),
            }],
            ..Default::default()
        };
        let result = store(result, &Refuses, "t1").await;

        assert!(
            matches!(&result.content[0], StoredContent::Text { text } if text == "[unstored image content]"),
            "a refused store must not persist the bytes: {:?}",
            result.content[0]
        );
    }

    #[tokio::test]
    async fn a_ref_that_is_already_stored_is_left_alone() {
        use crate::protocol::{ToolContent, ToolResult};

        let blobs = MemoryBlobStore::new();
        let stored = blobs
            .put(NewBlob {
                tenant_id: "t1".into(),
                mime: "image/png".into(),
                name: None,
                bytes: b"hi".to_vec(),
            })
            .await
            .expect("stored");
        let uri = stored.uri();
        let result = ToolResult {
            content: vec![ToolContent::Image {
                data: "aGk=".into(),
                mime_type: "image/png".into(),
            }],
            ..Default::default()
        };
        let result = store(result, &blobs, "t1").await;

        let StoredContent::Blob { uri: after } = &result.content[0] else {
            panic!("a blob block");
        };
        assert_ne!(*after, uri, "each put mints its own id");
        assert_eq!(
            blobs
                .get(&BlobRef::parse(after).expect("a ref"))
                .await
                .expect("stored"),
            b"hi".to_vec()
        );
    }

    struct FakeStore;

    #[async_trait]
    impl BlobStore for FakeStore {
        async fn put(&self, blob: NewBlob) -> Result<BlobRef, BlobError> {
            Ok(BlobRef {
                id: ID.to_string(),
                tenant_id: blob.tenant_id,
                mime: blob.mime,
                name: blob.name,
                size: blob.bytes.len() as u64,
            })
        }

        async fn get(&self, r: &BlobRef) -> Result<Vec<u8>, BlobError> {
            match r.id.as_str() {
                ID => Ok(vec![1, 2, 3, 4]),
                _ => Err(BlobError::NotFound),
            }
        }
    }

    struct Echo;

    #[async_trait]
    impl LlmCallable for Echo {
        async fn call(
            &self,
            request: &PromptRequest,
            _ctx: &CallContext<'_>,
        ) -> Result<LlmResponse, LlmCallError> {
            Ok(LlmResponse {
                model: request.model.clone(),
                content: serde_json::to_string(&request.messages).ok(),
                reasoning: None,
                tool_calls: Vec::new(),
                finish_reason: None,
                usage: None,
                cost: None,
                images: Vec::new(),
            })
        }
    }

    fn request(url: &str) -> LlmRequest {
        LlmRequest {
            model: "m".into(),
            messages: vec![DraftMessage {
                id: None,
                role: Role::User,
                content: Some(Content::Parts(vec![
                    StoredContent::Text { text: "hi".into() },
                    match BlobRef::parse(url) {
                        Some(_) => StoredContent::Blob { uri: url.into() },
                        None => StoredContent::Link {
                            uri: url.into(),
                            name: None,
                            mime_type: Some("image/png".into()),
                        },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning: None,
            }],
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    fn ctx<'a>(tenant: &'a str, owner: &'a SessionOwner) -> CallContext<'a> {
        CallContext {
            session_id: "s1",
            tenant_id: tenant,
            agent_id: "a1",
            call_id: "c1",
            attempt: 0,
            owner,
            ancestry: &[],
            defer_tools_strategy: Default::default(),
        }
    }

    #[tokio::test]
    async fn blob_refs_resolve_to_media_when_resolved() {
        let request = request(&blob_ref("t1").uri());
        let prompt = resolve(&request, &FakeStore, "t1").await.expect("resolves");
        let Some(PromptContent::Parts(parts)) = &prompt.messages[0].content else {
            panic!("parts");
        };
        assert_eq!(
            parts[1],
            PromptPart::Media {
                mime: "image/png".into(),
                name: Some("a photo & more.png".into()),
                bytes: vec![1, 2, 3, 4],
            }
        );
        match &request.messages[0].content {
            Some(Content::Parts(parts)) => {
                assert!(matches!(&parts[1], StoredContent::Blob { .. }))
            }
            _ => panic!("recorded parts"),
        }
    }

    #[tokio::test]
    async fn cross_tenant_and_missing_blobs_fail_the_call() {
        let err = resolve(&request(&blob_ref("t2").uri()), &FakeStore, "t1")
            .await
            .unwrap_err();
        assert!(err.error.message.contains("another tenant"));

        let missing = BlobRef {
            id: "0198b2a0-3c5d-7f00-8000-ffffffffffff".into(),
            ..blob_ref("t1")
        };
        let err = resolve(&request(&missing.uri()), &FakeStore, "t1")
            .await
            .unwrap_err();
        assert!(!err.retryable);
    }

    struct Generates;

    #[async_trait]
    impl LlmCallable for Generates {
        async fn call(
            &self,
            request: &PromptRequest,
            _ctx: &CallContext<'_>,
        ) -> Result<LlmResponse, LlmCallError> {
            let b64 = base64::engine::general_purpose::STANDARD.encode([9u8, 9, 9]);
            Ok(LlmResponse {
                model: request.model.clone(),
                content: Some("drew it".into()),
                reasoning: None,
                tool_calls: Vec::new(),
                finish_reason: None,
                usage: None,
                cost: None,
                images: vec![
                    crate::protocol::ResponseImage {
                        url: format!("data:image/png;base64,{b64}"),
                    },
                    crate::protocol::ResponseImage {
                        url: "https://provider.example/i.png".into(),
                    },
                ],
            })
        }
    }

    fn client_message(parts: Vec<StoredContent>) -> DraftMessage {
        DraftMessage {
            id: None,
            role: Role::User,
            content: Some(Content::Parts(parts)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    #[tokio::test]
    async fn client_data_uris_become_refs() {
        let blobs = MemoryBlobStore::new();
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3]);
        let mut messages = vec![client_message(vec![
            StoredContent::Text {
                text: "look".into(),
            },
            StoredContent::Blob {
                uri: format!("data:image/png;base64,{b64}"),
            },
            StoredContent::Blob {
                uri: "blob://t1/not-ours".into(),
            },
        ])];

        intern(&mut messages, &blobs, "t1").await;

        let Some(Content::Parts(parts)) = &messages[0].content else {
            panic!("parts");
        };
        let StoredContent::Blob { uri } = &parts[1] else {
            panic!("expected a ref, got {:?}", parts[1]);
        };
        let r = BlobRef::parse(uri).expect("a stored ref");
        assert_eq!(r.mime, "image/png");
        assert_eq!(blobs.get(&r).await.expect("bytes"), vec![1, 2, 3]);
        assert_eq!(
            parts[2],
            StoredContent::Blob {
                uri: "blob://t1/not-ours".into()
            },
            "non-data uris are left alone"
        );
    }

    #[tokio::test]
    async fn model_message_parts_are_taken_and_interned() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3]);
        let json = serde_json::json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "look" },
                { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{b64}") } },
                { "type": "image_url", "image_url": { "url": "https://x.example/p.png" } },
                { "type": "file", "file": { "filename": "q.pdf", "file_data": format!("data:application/pdf;base64,{b64}") } },
                { "type": "input_audio", "input_audio": { "data": b64, "format": "mp3" } },
                { "type": "video_url", "video_url": { "url": "https://x.example/v.mp4" } },
                { "type": "blob", "uri": format!("data:image/jpeg;base64,{b64}") },
            ]
        });
        let mut messages = vec![serde_json::from_value::<DraftMessage>(json).expect("parses")];
        let blobs = MemoryBlobStore::new();

        intern(&mut messages, &blobs, "t1").await;

        let Some(Content::Parts(parts)) = &messages[0].content else {
            panic!("parts");
        };
        let stored = |part: &StoredContent| match part {
            StoredContent::Blob { uri } => BlobRef::parse(uri).expect("a stored ref"),
            other => panic!("expected a ref, got {other:?}"),
        };
        assert_eq!(
            parts[0],
            StoredContent::Text {
                text: "look".into()
            }
        );
        assert_eq!(stored(&parts[1]).mime, "image/png");
        assert_eq!(
            parts[2],
            StoredContent::Link {
                uri: "https://x.example/p.png".into(),
                name: None,
                mime_type: Some("image/*".into()),
            }
        );
        let pdf = stored(&parts[3]);
        assert_eq!(
            (pdf.mime.as_str(), pdf.name.as_deref()),
            ("application/pdf", Some("q.pdf"))
        );
        let audio = stored(&parts[4]);
        assert_eq!(audio.mime, "audio/mp3", "the format is kept as given");
        assert_eq!(blobs.get(&audio).await.expect("bytes"), vec![1, 2, 3]);
        assert_eq!(
            parts[5],
            StoredContent::Link {
                uri: "https://x.example/v.mp4".into(),
                name: None,
                mime_type: Some("video/*".into()),
            }
        );
        assert_eq!(stored(&parts[6]).mime, "image/jpeg");

        let request = LlmRequest {
            model: "m".into(),
            messages: messages.clone(),
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        };
        let out = resolve(&request, &blobs, "t1").await.expect("resolves");
        let Some(PromptContent::Parts(sent)) = &out.messages[0].content else {
            panic!("parts");
        };
        let kinds: Vec<&str> = sent
            .iter()
            .map(|p| match ContentPart::from(p) {
                ContentPart::Text { .. } => "text",
                ContentPart::ImageUrl { .. } => "image_url",
                ContentPart::File { .. } => "file",
                ContentPart::InputAudio { .. } => "input_audio",
                ContentPart::VideoUrl { .. } => "video_url",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "text",
                "image_url",
                "image_url",
                "file",
                "input_audio",
                "video_url",
                "image_url"
            ],
            "on the wire each part keeps the shape the client sent"
        );
    }

    #[tokio::test]
    async fn unstorable_client_bytes_leave_a_note() {
        let mut messages = vec![client_message(vec![StoredContent::Blob {
            uri: "data:image/jpeg;base64,AQID".into(),
        }])];
        intern(&mut messages, &NOWHERE, "t1").await;
        let Some(Content::Parts(parts)) = &messages[0].content else {
            panic!("parts");
        };
        assert_eq!(
            parts[0],
            StoredContent::Text {
                text: "[unstored image content]".into()
            }
        );
    }

    #[test]
    fn data_uri_parse_covers_rfc_2397_variants() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3]);
        let ok = |uri: &str| parse_data_uri(uri).unwrap();
        assert_eq!(ok(&format!("data:image/png;base64,{b64}")).1, vec![1, 2, 3]);
        // Case-insensitive marker, unpadded body, whitespace in body.
        assert_eq!(ok("data:image/png;BASE64,AQID").1, vec![1, 2, 3]);
        assert_eq!(ok("data:image/png;base64,AQI").1, vec![1, 2]);
        assert_eq!(ok("data:image/png;base64,AQ ID").1, vec![1, 2, 3]);
        // No mime, no base64 marker, or no comma: left inline.
        assert_eq!(parse_data_uri("data:;base64,AQID"), None);
        assert_eq!(parse_data_uri("data:text/plain,hello"), None);
        assert_eq!(parse_data_uri("data:image/png;base64"), None);
        assert_eq!(parse_data_uri("https://x/y.png"), None);
    }

    #[tokio::test]
    async fn the_worker_wire_carries_each_part_by_kind() {
        let callable = BlobResolvingCallable {
            inner: Arc::new(Echo),
            blobs: Arc::new(FakeStore),
        };
        let owner = SessionOwner::default();
        let file_part = |mime: &str, filename: &str| StoredContent::Blob {
            uri: BlobRef {
                mime: mime.into(),
                name: Some(filename.into()),
                ..blob_ref("t1")
            }
            .uri(),
        };
        let mut request = request("https://x/y.png");
        request.messages[0].content = Some(Content::Parts(vec![
            file_part("application/pdf", "q3.pdf"),
            file_part("text/csv", "sales.csv"),
        ]));
        let prompt = resolve(&request, &FakeStore, "t1").await.expect("resolves");
        let resp = callable.call(&prompt, &ctx("t1", &owner)).await.unwrap();
        let sent = resp.content.unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3, 4]);
        assert!(sent.contains(&format!("data:application/pdf;base64,{b64}")));
        assert!(sent.contains("q3.pdf"));
        assert!(sent.contains(&format!("data:text/csv;base64,{b64}")));
        assert!(sent.contains("sales.csv"));
    }

    #[tokio::test]
    async fn generated_images_are_stored_as_refs() {
        let callable = BlobResolvingCallable {
            inner: Arc::new(Generates),
            blobs: Arc::new(FakeStore),
        };
        let owner = SessionOwner::default();
        let prompt = resolve(&request("https://x/y.png"), &FakeStore, "t1")
            .await
            .expect("resolves");
        let resp = callable.call(&prompt, &ctx("t1", &owner)).await.unwrap();
        let stored = BlobRef::parse(&resp.images[0].url).unwrap();
        assert_eq!(stored.tenant_id, "t1");
        assert_eq!(stored.mime, "image/png");
        assert_eq!(stored.size, 3);
        // A provider-hosted url is not ours to store.
        assert_eq!(resp.images[1].url, "https://provider.example/i.png");
    }

    #[tokio::test]
    async fn non_blob_urls_pass_through_untouched() {
        let callable = BlobResolvingCallable {
            inner: Arc::new(Echo),
            blobs: Arc::new(FakeStore),
        };
        let owner = SessionOwner::default();
        let req = resolve(&request("https://example.com/a.png"), &FakeStore, "t1")
            .await
            .expect("resolves");
        let resp = callable.call(&req, &ctx("t1", &owner)).await.unwrap();
        assert!(resp.content.unwrap().contains("https://example.com/a.png"));
    }
}
