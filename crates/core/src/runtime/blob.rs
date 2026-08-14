//! Blob storage for message attachments and model-generated media.
//!
//! Messages and prompts persist only a `blob://` reference; bytes stay in a
//! [`BlobStore`] and are inlined as `data:` URIs at the provider call. On the
//! way back, a generated image's `data:` URI is stored and the response keeps
//! the ref. A ref never carries a fetchable URL, so nothing persisted can
//! leak the bytes.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use percent_encoding::{percent_decode_str, utf8_percent_encode, NON_ALPHANUMERIC};

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmResolver};
use crate::protocol::{
    AudioData, Content, ContentPart, ErrorCode, FileData, ImageUrl, LlmRequest, LlmResponse,
    SessionOwner, VideoUrl,
};
use crate::protocol::{StoredContent, StoredResult, ToolContent, ToolResult};

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

/// An in-memory store for tests in other modules, keyed by minted id.
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

/// The name a provider knows an audio encoding by, from its mime.
fn audio_format(mime: &str) -> &str {
    match mime.split('/').nth(1).unwrap_or_default() {
        "mpeg" | "mp3" => "mp3",
        "wav" | "x-wav" | "wave" => "wav",
        other => other,
    }
}

/// Store the bytes a tool sent inline, leaving a `blob://` ref in their place,
/// so a result carrying an image is small enough to persist.
///
/// A block the store refuses keeps its bytes: the call succeeded, and inline
/// bytes still reach the model.
pub async fn store(result: ToolResult, blobs: &dyn BlobStore, tenant_id: &str) -> StoredResult {
    let mut content = Vec::with_capacity(result.content.len());
    for block in result.content {
        let Some((data, mime)) = block
            .inline()
            .map(|(data, mime)| (data.to_string(), mime.to_string()))
        else {
            content.push(match &block {
                ToolContent::ResourceLink {
                    uri,
                    name,
                    mime_type,
                } => StoredContent::Link {
                    uri: uri.clone(),
                    name: name.clone(),
                    mime_type: mime_type.clone(),
                },
                ToolContent::Resource { resource } => StoredContent::Text {
                    text: resource.text.clone().unwrap_or_default(),
                },
                ToolContent::Text { text } => StoredContent::Text { text: text.clone() },
                // Carried bytes; `inline` would have returned them.
                ToolContent::Image { .. } | ToolContent::Audio { .. } => StoredContent::Text {
                    text: String::new(),
                },
            });
            continue;
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&data) else {
            content.push(StoredContent::Text {
                text: format!("[unreadable {} content]", essence(&mime)),
            });
            continue;
        };
        let name = block.name().map(str::to_string);
        let put = blobs
            .put(NewBlob {
                tenant_id: tenant_id.to_string(),
                mime: mime.clone(),
                name: name.clone(),
                bytes,
            })
            .await;
        content.push(match put {
            Ok(r) => StoredContent::Blob { uri: r.uri() },
            // Naming the content costs the model an image. The result has
            // nowhere to keep bytes, which is the point: a store that fails
            // cannot put them in the event log instead.
            Err(e) => {
                tracing::warn!("storing a tool's inline content failed: {e}");
                StoredContent::Text {
                    text: format!("[{} content]", essence(&mime)),
                }
            }
        });
    }
    StoredResult {
        content,
        structured_content: result.structured_content,
        is_error: result.is_error,
    }
}

/// Stands in where a deployment stores nothing. Every block is named rather
/// than kept, which is the same answer a broken store gives.
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

fn essence(mime: &str) -> &str {
    mime.split('/').next().unwrap_or("file")
}

/// A stored attachment as the message part its kind rides in.
pub fn attachment_part(r: &BlobRef) -> ContentPart {
    if r.mime.starts_with("image/") {
        ContentPart::ImageUrl {
            image_url: ImageUrl { url: r.uri() },
        }
    } else {
        ContentPart::File {
            file: FileData {
                filename: r.name.clone().unwrap_or_else(|| "file".to_string()),
                file_data: r.uri(),
            },
        }
    }
}

/// Mimes that read as text: the file inlines into the prompt as a text part,
/// which every provider takes.
pub fn text_like(mime: &str) -> bool {
    let essence = mime.split(';').next().unwrap_or_default().trim();
    essence.starts_with("text/")
        || matches!(
            essence,
            "application/json"
                | "application/xml"
                | "application/yaml"
                | "application/x-yaml"
                | "application/toml"
                | "application/csv"
                | "application/x-ndjson"
                | "application/javascript"
                | "application/typescript"
                | "application/x-sh"
                | "application/sql"
        )
}

/// Wraps a resolver so every callable inlines `blob://` images as `data:`
/// URIs just before the provider call, and stores a response's generated
/// images as blobs on the way back. The recorded prompt and message keep the
/// refs; the bytes exist only for the duration of the request.
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

/// Replace every `blob://` ref in `request` with the bytes it names. `None`
/// when the request holds none; the original then goes through untouched.
///
/// Every path that hands a request to a provider calls this — the engine's own
/// callable, and the dispatch that sends a call out to a worker — so the two
/// cannot inline differently.
pub async fn inline_refs(
    request: &LlmRequest,
    blobs: &dyn BlobStore,
    tenant_id: &str,
) -> Result<Option<LlmRequest>, LlmCallError> {
    {
        let has_blob = request.messages.iter().any(
            |m| matches!(&m.content, Some(Content::Parts(parts)) if parts.iter().any(is_blob_part)),
        );
        if !has_blob {
            return Ok(None);
        }
        let mut resolved = request.clone();
        for message in &mut resolved.messages {
            let Some(Content::Parts(parts)) = &mut message.content else {
                continue;
            };
            for part in parts {
                let url = match part {
                    ContentPart::ImageUrl { image_url } => &image_url.url,
                    ContentPart::File { file } => &file.file_data,
                    _ => continue,
                };
                let Some(r) = BlobRef::parse(url) else {
                    if url.starts_with(BLOB_SCHEME) {
                        return Err(blob_call_error("unparsable blob ref", false));
                    }
                    continue;
                };
                if r.tenant_id != tenant_id {
                    return Err(blob_call_error("blob ref for another tenant", false));
                }
                let bytes = blobs
                    .get(&r)
                    .await
                    .map_err(|e| blob_call_error(&e.to_string(), matches!(e, BlobError::Io(_))))?;
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
                        image_url.url = format!("data:{};base64,{b64}", r.mime);
                    }
                    // A ref travels as a file; what it becomes is its mime's
                    // to say. Text inlines as text, which every provider takes,
                    // and sound and moving pictures take the parts named for
                    // them. Anything else rides as a data url.
                    ContentPart::File { file } => {
                        let b64 = || base64::engine::general_purpose::STANDARD.encode(&bytes);
                        let essence = r.mime.split('/').next().unwrap_or_default();
                        match essence {
                            _ if text_like(&r.mime) => {
                                let name = file.filename.clone();
                                let text = String::from_utf8_lossy(&bytes);
                                *part = ContentPart::Text {
                                    text: format!("<file name={name:?}>\n{text}\n</file>"),
                                };
                            }
                            "audio" => {
                                *part = ContentPart::InputAudio {
                                    input_audio: AudioData {
                                        data: b64(),
                                        format: audio_format(&r.mime).to_string(),
                                    },
                                }
                            }
                            "video" => {
                                *part = ContentPart::VideoUrl {
                                    video_url: VideoUrl {
                                        url: format!("data:{};base64,{}", r.mime, b64()),
                                    },
                                }
                            }
                            _ => file.file_data = format!("data:{};base64,{}", r.mime, b64()),
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(Some(resolved))
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
            let stored = self
                .blobs
                .put(NewBlob {
                    tenant_id: ctx.tenant_id.to_string(),
                    mime,
                    name: None,
                    bytes,
                })
                .await;
            match stored {
                Ok(r) => img.url = r.uri(),
                Err(e) => {
                    tracing::warn!(error = %e, call = %ctx.call_id, "generated image not stored; kept inline")
                }
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

fn is_blob_part(part: &ContentPart) -> bool {
    match part {
        ContentPart::ImageUrl { image_url } => image_url.url.starts_with(BLOB_SCHEME),
        ContentPart::File { file } => file.file_data.starts_with(BLOB_SCHEME),
        _ => false,
    }
}

fn blob_call_error(message: &str, retryable: bool) -> LlmCallError {
    LlmCallError::new(ErrorCode::Internal, message, retryable)
}

#[async_trait]
impl LlmCallable for BlobResolvingCallable {
    async fn call(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        let mut response = match inline_refs(request, self.blobs.as_ref(), ctx.tenant_id).await? {
            Some(resolved) => self.inner.call(&resolved, ctx).await?,
            None => self.inner.call(request, ctx).await?,
        };
        self.store_images(&mut response, ctx).await;
        Ok(response)
    }

    async fn call_streaming(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
        tx: tokio::sync::mpsc::UnboundedSender<crate::protocol::StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let mut response = match inline_refs(request, self.blobs.as_ref(), ctx.tenant_id).await? {
            Some(resolved) => self.inner.call_streaming(&resolved, ctx, tx).await?,
            None => self.inner.call_streaming(request, ctx, tx).await?,
        };
        self.store_images(&mut response, ctx).await;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{DraftMessage, ImageUrl, Role};

    const ID: &str = "0198b2a0-3c5d-7f00-8000-0123456789ab";

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
    async fn sound_and_moving_pictures_take_the_parts_named_for_them() {
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
                .map(|r| ContentPart::File {
                    file: FileData {
                        filename: "f".into(),
                        file_data: r.uri(),
                    },
                })
                .collect(),
        ));

        let out = inline_refs(&request, &blobs, "t1")
            .await
            .expect("resolves")
            .expect("held refs");
        let Some(Content::Parts(parts)) = &out.messages[0].content else {
            panic!("parts");
        };
        match &parts[0] {
            ContentPart::InputAudio { input_audio } => {
                assert_eq!(input_audio.format, "mp3", "the mime names the encoding");
                assert_eq!(
                    base64::engine::general_purpose::STANDARD
                        .decode(&input_audio.data)
                        .expect("base64"),
                    b"aaa".to_vec()
                );
            }
            other => panic!("expected audio, got {other:?}"),
        }
        assert!(
            matches!(&parts[1], ContentPart::VideoUrl { video_url }
                if video_url.url.starts_with("data:video/mp4;base64,")),
            "video takes its own part, not a file: {:?}",
            parts[1]
        );
        assert!(
            matches!(&parts[2], ContentPart::File { file }
                if file.file_data.starts_with("data:application/zip;base64,")),
            "anything else still rides as a file"
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
            matches!(&result.content[0], StoredContent::Text { text } if text == "[image content]"),
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
            request: &LlmRequest,
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
                    ContentPart::Text { text: "hi".into() },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl { url: url.into() },
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
    async fn blob_refs_inline_as_data_uris_at_the_call() {
        let callable = BlobResolvingCallable {
            inner: Arc::new(Echo),
            blobs: Arc::new(FakeStore),
        };
        let owner = SessionOwner::default();
        let request = request(&blob_ref("t1").uri());
        let resp = callable.call(&request, &ctx("t1", &owner)).await.unwrap();
        let expected = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3, 4])
        );
        assert!(resp.content.unwrap().contains(&expected));
        // The caller's request still holds the ref.
        match &request.messages[0].content {
            Some(Content::Parts(parts)) => assert!(is_blob_part(&parts[1])),
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn cross_tenant_and_missing_blobs_fail_the_call() {
        let callable = BlobResolvingCallable {
            inner: Arc::new(Echo),
            blobs: Arc::new(FakeStore),
        };
        let owner = SessionOwner::default();
        let err = callable
            .call(&request(&blob_ref("t2").uri()), &ctx("t1", &owner))
            .await
            .unwrap_err();
        assert!(err.error.message.contains("another tenant"));
        let missing = BlobRef {
            id: "0198b2a0-3c5d-7f00-8000-ffffffffffff".into(),
            ..blob_ref("t1")
        };
        let err = callable
            .call(&request(&missing.uri()), &ctx("t1", &owner))
            .await
            .unwrap_err();
        assert!(!err.retryable);
    }

    struct Generates;

    #[async_trait]
    impl LlmCallable for Generates {
        async fn call(
            &self,
            request: &LlmRequest,
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
    async fn file_refs_resolve_by_kind_at_the_call() {
        let callable = BlobResolvingCallable {
            inner: Arc::new(Echo),
            blobs: Arc::new(FakeStore),
        };
        let owner = SessionOwner::default();
        let file_part = |mime: &str, filename: &str| ContentPart::File {
            file: crate::protocol::FileData {
                filename: filename.into(),
                file_data: BlobRef {
                    mime: mime.into(),
                    ..blob_ref("t1")
                }
                .uri(),
            },
        };
        let mut request = request("https://x/y.png");
        request.messages[0].content = Some(Content::Parts(vec![
            file_part("application/pdf", "q3.pdf"),
            file_part("text/csv", "sales.csv"),
        ]));
        let resp = callable.call(&request, &ctx("t1", &owner)).await.unwrap();
        let sent = resp.content.unwrap();
        // The pdf keeps its file shape with the bytes inlined…
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3, 4]);
        assert!(sent.contains(&format!("data:application/pdf;base64,{b64}")));
        assert!(sent.contains("q3.pdf"));
        // …while the text file becomes a text part any provider takes.
        assert!(sent.contains("<file name=\\\"sales.csv\\\">"));
        assert!(!sent.contains("data:text/csv"));
    }

    #[test]
    fn text_like_covers_the_business_formats() {
        for mime in [
            "text/plain",
            "text/markdown",
            "text/csv; charset=utf-8",
            "application/json",
            "application/x-yaml",
        ] {
            assert!(text_like(mime), "{mime}");
        }
        for mime in ["application/pdf", "image/png", "application/zip"] {
            assert!(!text_like(mime), "{mime}");
        }
    }

    #[tokio::test]
    async fn generated_images_are_stored_as_refs() {
        let callable = BlobResolvingCallable {
            inner: Arc::new(Generates),
            blobs: Arc::new(FakeStore),
        };
        let owner = SessionOwner::default();
        let resp = callable
            .call(&request("https://x/y.png"), &ctx("t1", &owner))
            .await
            .unwrap();
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
        let req = request("https://example.com/a.png");
        let resp = callable.call(&req, &ctx("t1", &owner)).await.unwrap();
        assert!(resp.content.unwrap().contains("https://example.com/a.png"));
    }
}
