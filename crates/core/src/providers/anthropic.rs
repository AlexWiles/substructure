//! Anthropic Messages API (`POST /v1/messages`) provider.
//!
//! The engine's internal request/response types are OpenAI-Chat-shaped, so this
//! module translates in both directions: the flat `messages` array with a
//! `system` role and OpenAI-style `tool_calls`/`tool` messages is mapped onto
//! Anthropic's top-level `system`, content-block messages, and
//! `tool_use`/`tool_result` blocks — and the block-structured response and SSE
//! event stream are mapped back.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::StreamExt;

use crate::llm::{CallContext, LlmCallError, LlmCallable, LlmProviderTrait};
use crate::protocol::{
    Content, ContentPart, DeferToolsStrategy, ErrorCode, LlmRequest, LlmResponse, Reasoning,
    ReasoningEffort, ReasoningProvider, Role, SessionOwner, StreamDelta, ToolCall, ToolCallChunk,
    ToolCallFunction, Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u64 = 4096;

// ── Request wire format ──────────────────────────────────────────────────

/// A cache breakpoint. Anthropic caches nothing without one, so a request that
/// carries none is re-read in full on every turn.
#[derive(Serialize, Clone, Copy)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
    /// Absent is the default life, five minutes.
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<&'static str>,
}

impl CacheControl {
    const EPHEMERAL: Self = Self {
        kind: "ephemeral",
        ttl: None,
    };

    /// One life for every breakpoint in the request: the API reads a longer one
    /// after a shorter one as an error, and a single value cannot order wrong.
    fn with_ttl(ttl: Option<&str>) -> Self {
        match ttl {
            Some("1h") => Self {
                kind: "ephemeral",
                ttl: Some("1h"),
            },
            _ => Self::EPHEMERAL,
        }
    }
}

#[derive(Serialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct AnthropicBody {
    model: String,
    max_tokens: u64,
    /// Only for the models that still read it: Opus 4.7 and later refuse it.
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    /// Never sent: a strategy the engine answers drops each deferred tool
    /// before this point. The field is what a strategy the provider answers
    /// serializes as its own flag.
    #[serde(skip)]
    defer: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: Vec<Block>,
}

/// One block of a turn: built from the transcript, or held verbatim. Anthropic
/// signs a thinking block and rejects one that comes back edited, so a thinking
/// block goes out as the bytes it arrived as.
#[derive(Serialize)]
#[serde(untagged)]
enum Block {
    Raw(serde_json::Value),
    Built(RequestBlock),
}

impl Block {
    fn cache(&mut self, control: CacheControl) {
        match self {
            Block::Built(b) => b.cache(control),
            // A mark would edit a signed block.
            Block::Raw(_) => {}
        }
    }
}

impl From<RequestBlock> for Block {
    fn from(b: RequestBlock) -> Self {
        Block::Built(b)
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RequestBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl RequestBlock {
    fn cache(&mut self, control: CacheControl) {
        match self {
            RequestBlock::Text { cache_control, .. }
            | RequestBlock::Image { cache_control, .. }
            | RequestBlock::ToolUse { cache_control, .. }
            | RequestBlock::ToolResult { cache_control, .. }
            | RequestBlock::Document { cache_control, .. } => *cache_control = Some(control),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ImageSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

#[derive(Serialize)]
struct Thinking {
    #[serde(rename = "type")]
    thinking_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u64>,
}

#[derive(Serialize)]
struct OutputConfig {
    effort: &'static str,
}

/// What one model reads of the reasoning surface. Each field is a 400 when it
/// is wrong: an effort level the model does not know, an adaptive `thinking`
/// block it predates, an explicit `disabled` it refuses.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Capabilities {
    /// `temperature` and the other sampling parameters. Newer models refuse
    /// them outright.
    sampling: bool,
    /// `output_config.effort`.
    effort: bool,
    /// `xhigh` among the effort levels.
    xhigh: bool,
    /// `thinking: {"type": "adaptive"}`. Models before it take a token budget.
    adaptive: bool,
    /// Thinks when the request omits `thinking`, so not thinking is a request.
    thinks_unasked: bool,
    /// Always thinks: `disabled` comes back a 400.
    thinks_always: bool,
}

impl Capabilities {
    //                      sampling  effort  xhigh  adaptive  unasked  always
    /// Fable and Mythos, which think whatever the request says.
    const ALWAYS: Self = Self::new(false, true, true, true, true, true);
    /// Opus 5 and Sonnet 5, which think until told not to.
    const DEFAULT_ON: Self = Self::new(false, true, true, true, true, false);
    /// The current surface: every reasoning field, and no sampling.
    const CURRENT: Self = Self::new(false, true, true, true, false, false);
    /// Before `xhigh`, which arrived with Opus 4.7. Sampling went at the same
    /// time.
    const NO_XHIGH: Self = Self::new(true, true, false, true, false, false);
    /// Before adaptive thinking, which arrived with Opus 4.6.
    const EFFORT_ONLY: Self = Self::new(true, true, false, false, false, false);
    /// Before the reasoning fields: a token budget, and nothing else.
    const LEGACY: Self = Self::new(true, false, false, false, false, false);

    const fn new(
        sampling: bool,
        effort: bool,
        xhigh: bool,
        adaptive: bool,
        thinks_unasked: bool,
        thinks_always: bool,
    ) -> Self {
        Self {
            sampling,
            effort,
            xhigh,
            adaptive,
            thinks_unasked,
            thinks_always,
        }
    }
}

/// One generation: every model in `family` at or above `since` reads `caps`
/// and writes at most `output` tokens.
struct Generation {
    family: &'static str,
    since: (u32, u32),
    output: u64,
    caps: Capabilities,
}

/// What each generation reads and writes. The first row a model matches wins,
/// so a family's rows run newest first and end at `(0, 0)`, which is everything
/// older. A name in no family at all is not in the table: see `generation`.
#[rustfmt::skip]
const GENERATIONS: &[Generation] = &[
    Generation { family: "fable",  since: (0, 0), output: 128_000, caps: Capabilities::ALWAYS },
    Generation { family: "mythos", since: (0, 0), output: 128_000, caps: Capabilities::ALWAYS },
    Generation { family: "opus",   since: (5, 0), output: 128_000, caps: Capabilities::DEFAULT_ON },
    Generation { family: "opus",   since: (4, 7), output: 128_000, caps: Capabilities::CURRENT },
    Generation { family: "opus",   since: (4, 6), output: 128_000, caps: Capabilities::NO_XHIGH },
    Generation { family: "opus",   since: (4, 5), output:  64_000, caps: Capabilities::EFFORT_ONLY },
    Generation { family: "opus",   since: (4, 1), output:  32_000, caps: Capabilities::LEGACY },
    Generation { family: "opus",   since: (0, 0), output:  32_000, caps: Capabilities::LEGACY },
    Generation { family: "sonnet", since: (5, 0), output: 128_000, caps: Capabilities::DEFAULT_ON },
    Generation { family: "sonnet", since: (4, 6), output: 128_000, caps: Capabilities::NO_XHIGH },
    Generation { family: "sonnet", since: (0, 0), output:  64_000, caps: Capabilities::LEGACY },
    Generation { family: "haiku",  since: (4, 5), output:  64_000, caps: Capabilities::LEGACY },
    Generation { family: "haiku",  since: (0, 0), output:   4_096, caps: Capabilities::LEGACY },
];

/// The row a model sits on, or nothing for a name the table does not place: a
/// new family, an alias, a gateway's own spelling.
fn generation(model: &str) -> Option<&'static Generation> {
    GENERATIONS
        .iter()
        .find(|g| version(model, g.family).is_some_and(|v| v >= g.since))
}

/// The reasoning fields a model reads.
///
/// An unplaced name reads as the current generation: the fields a newer model
/// would refuse are the ones it is too new to refuse. `thinks_unasked` stays
/// off there, because sending `disabled` to a model that rejects the word is a
/// 400 and an unplaced name gives no way to tell.
fn capabilities(model: &str) -> Capabilities {
    generation(model).map_or(Capabilities::CURRENT, |g| g.caps)
}

/// What the request asks to write, held to what the model can. A ceiling the
/// model would refuse is a 400, and a session that switches model should not
/// have to carry the smallest one everywhere. An unplaced name has no known
/// ceiling and keeps what it asked for.
fn output_tokens(model: &str, asked: u64) -> u64 {
    match generation(model) {
        Some(g) => asked.min(g.output),
        None => asked,
    }
}

/// The `<major>, <minor>` of a `claude-<family>-<major>[-<minor>]` name, or
/// nothing for another family. A run of more than two digits is a date stamp,
/// not a minor: `claude-opus-4-20250514` is 4.0, and reading it as 4.20250514
/// would hand an Opus 4.0 the whole current surface.
fn version(model: &str, family: &str) -> Option<(u32, u32)> {
    const MAX_MINOR_DIGITS: usize = 2;

    let prefix = format!("claude-{family}-");
    let at = model.find(&prefix)?;
    let mut parts = model[at + prefix.len()..].split('-');
    let major = parts.next()?.parse().ok()?;
    let minor = match parts.next() {
        Some(p) if p.len() <= MAX_MINOR_DIGITS => p.parse().unwrap_or(0),
        _ => 0,
    };
    Some((major, minor))
}

fn anthropic_effort(e: ReasoningEffort, xhigh: bool) -> &'static str {
    match e {
        ReasoningEffort::Xhigh if xhigh => "xhigh",
        // A model that predates `xhigh` reads the level below it.
        ReasoningEffort::Xhigh | ReasoningEffort::High => "high",
        ReasoningEffort::Medium => "medium",
        // Anthropic has no `minimal`/`none`; `low` is the nearest floor.
        ReasoningEffort::Low | ReasoningEffort::Minimal | ReasoningEffort::None => "low",
    }
}

/// Legacy thinking asks for a token budget where effort asks for a word. The
/// budget comes out of `max_tokens` and the answer needs what is left, so a
/// budget that would starve the answer is no thinking at all.
fn legacy_budget(e: ReasoningEffort, max_tokens: u64) -> Option<u64> {
    /// The floor the API accepts, and the room the answer keeps.
    const MIN_BUDGET: u64 = 1024;

    let want = match e {
        ReasoningEffort::Xhigh | ReasoningEffort::High => 24_000,
        ReasoningEffort::Medium => 8_000,
        ReasoningEffort::Low | ReasoningEffort::Minimal | ReasoningEffort::None => MIN_BUDGET,
    };
    let ceiling = max_tokens.saturating_sub(MIN_BUDGET);
    (ceiling >= MIN_BUDGET).then(|| want.min(ceiling))
}

/// The `thinking` and `output_config` a request carries, for what this model
/// reads.
fn reasoning_fields(
    model: &str,
    effort: Option<ReasoningEffort>,
    max_tokens: u64,
) -> (Option<Thinking>, Option<OutputConfig>) {
    let caps = capabilities(model);
    let effort = match effort {
        // Off is the default everywhere but the models that think unasked, and
        // asking a model that always thinks to stop is a 400.
        None | Some(ReasoningEffort::None) => {
            let thinking = (caps.thinks_unasked && !caps.thinks_always).then_some(Thinking {
                thinking_type: "disabled",
                budget_tokens: None,
            });
            return (thinking, None);
        }
        Some(e) => e,
    };

    let output_config = caps.effort.then(|| OutputConfig {
        effort: anthropic_effort(effort, caps.xhigh),
    });
    let thinking = if caps.adaptive {
        Some(Thinking {
            thinking_type: "adaptive",
            budget_tokens: None,
        })
    } else {
        legacy_budget(effort, max_tokens).map(|budget| Thinking {
            thinking_type: "enabled",
            budget_tokens: Some(budget),
        })
    };
    (thinking, output_config)
}

/// Map an OpenAI-style role message's content into Anthropic content blocks.
fn content_to_blocks(content: Option<&Content>) -> Vec<RequestBlock> {
    match content {
        None => Vec::new(),
        Some(Content::Text(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![RequestBlock::Text {
                    text: s.clone(),
                    cache_control: None,
                }]
            }
        }
        Some(Content::Parts(parts)) => parts.iter().filter_map(part_to_block).collect(),
    }
}

fn part_to_block(part: &ContentPart) -> Option<RequestBlock> {
    match part {
        ContentPart::Text { text } => Some(RequestBlock::Text {
            text: text.clone(),
            cache_control: None,
        }),
        ContentPart::ImageUrl { image_url } => Some(image_block(&image_url.url)),
        ContentPart::File { file } => document_block(&file.file_data),
        // Audio/video parts have no Messages API equivalent here.
        _ => None,
    }
}

/// A PDF as a `document` block. Other file payloads have no Messages API
/// shape and are dropped; text files become text parts before this layer.
fn document_block(file_data: &str) -> Option<RequestBlock> {
    let rest = file_data.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media_type = meta.split(';').next().unwrap_or_default();
    if media_type != "application/pdf" {
        return None;
    }
    Some(RequestBlock::Document {
        source: ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: data.to_string(),
        },
        cache_control: None,
    })
}

fn image_block(url: &str) -> RequestBlock {
    if let Some(rest) = url.strip_prefix("data:") {
        if let Some((meta, data)) = rest.split_once(',') {
            let media_type = meta.split(';').next().unwrap_or("image/png").to_string();
            return RequestBlock::Image {
                source: ImageSource::Base64 {
                    media_type,
                    data: data.to_string(),
                },
                cache_control: None,
            };
        }
    }
    RequestBlock::Image {
        source: ImageSource::Url {
            url: url.to_string(),
        },
        cache_control: None,
    }
}

/// Append blocks to the last turn if it shares the role, else open a new turn.
/// Anthropic requires strictly alternating user/assistant turns.
fn push_turn(turns: &mut Vec<AnthropicMessage>, role: &'static str, blocks: Vec<Block>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = turns.last_mut() {
        if last.role == role {
            last.content.extend(blocks);
            return;
        }
    }
    turns.push(AnthropicMessage {
        role,
        content: blocks,
    });
}

/// `stream: None` omits the field so the body is valid input for both the
/// create and stream calls a worker might make.
fn build_body(
    request: &LlmRequest,
    default_max_tokens: u64,
    search: DeferToolsStrategy,
    stream: Option<bool>,
    cache: CacheControl,
) -> AnthropicBody {
    let mut system_parts: Vec<String> = Vec::new();
    let mut turns: Vec<AnthropicMessage> = Vec::new();

    for msg in &request.messages {
        match msg.role {
            Role::System => {
                if let Some(c) = &msg.content {
                    let text = c.text_owned();
                    if !text.is_empty() {
                        system_parts.push(text);
                    }
                }
            }
            Role::User => {
                push_turn(
                    &mut turns,
                    "user",
                    content_to_blocks(msg.content.as_ref())
                        .into_iter()
                        .map(Block::Built)
                        .collect(),
                );
            }
            Role::Assistant => {
                // Thinking leads the turn: the API requires the blocks that
                // preceded a tool call back with it, ahead of what they led to.
                let mut blocks: Vec<Block> = msg
                    .reasoning
                    .iter()
                    .flat_map(|r| r.blocks_for(ReasoningProvider::Anthropic))
                    .cloned()
                    .map(Block::Raw)
                    .collect();
                if let Some(c) = &msg.content {
                    let text = c.text_owned();
                    if !text.is_empty() {
                        blocks.push(
                            RequestBlock::Text {
                                text,
                                cache_control: None,
                            }
                            .into(),
                        );
                    }
                }
                if let Some(tcs) = &msg.tool_calls {
                    for tc in tcs {
                        let input = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        blocks.push(
                            RequestBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                input,
                                cache_control: None,
                            }
                            .into(),
                        );
                    }
                }
                push_turn(&mut turns, "assistant", blocks);
            }
            Role::Tool => {
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                let content = msg
                    .content
                    .as_ref()
                    .map(|c| c.text_owned())
                    .unwrap_or_default();
                push_turn(
                    &mut turns,
                    "user",
                    vec![RequestBlock::ToolResult {
                        tool_use_id,
                        content,
                        cache_control: None,
                    }
                    .into()],
                );
            }
        }
    }

    // Four breakpoints: the tools and the system prompt, which hold for the
    // session, and the last two user turns, which follow the transcript.
    let system = (!system_parts.is_empty()).then(|| {
        vec![SystemBlock {
            kind: "text",
            text: system_parts.join("\n\n"),
            cache_control: Some(cache),
        }]
    });

    let tools = request.offered_tools(search).map(|ts| {
        let mut ts: Vec<AnthropicTool> = ts
            .into_iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema(),
                defer: t.defer,
                cache_control: None,
            })
            .collect();
        // The last tool the provider reads for itself. A deferred one cannot
        // carry a breakpoint, so the mark goes on the last that is not.
        if let Some(last) = ts.iter_mut().rev().find(|t| !t.defer) {
            last.cache_control = Some(cache);
        }
        ts
    });

    // Two marks: this transcript's end, and the previous user turn, where the
    // last request ended and left an entry. One will not do — a mark reaches
    // twenty blocks back, and a turn of parallel tool calls adds more.
    let last = turns.len().checked_sub(1);
    let previous = turns[..turns.len().saturating_sub(1)]
        .iter()
        .rposition(|t| t.role == "user");
    for turn in [previous, last].into_iter().flatten() {
        if let Some(block) = turns[turn].content.last_mut() {
            block.cache(cache);
        }
    }

    let max_tokens = output_tokens(
        &request.model,
        request.max_completion_tokens.unwrap_or(default_max_tokens),
    );
    let (thinking, output_config) = reasoning_fields(
        &request.model,
        request.reasoning.as_ref().and_then(|r| r.effort),
        max_tokens,
    );

    AnthropicBody {
        model: request.model.clone(),
        max_tokens,
        temperature: capabilities(&request.model)
            .sampling
            .then_some(request.temperature)
            .flatten(),
        system,
        messages: turns,
        tools,
        thinking,
        output_config,
        stream,
    }
}

/// Map an Anthropic `stop_reason` onto the OpenAI-style `finish_reason` vocab
/// the rest of the engine uses.
fn map_stop_reason(reason: &str) -> String {
    match reason {
        "tool_use" => "tool_calls",
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        other => other,
    }
    .to_string()
}

// ── Non-streaming response ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    model: String,
    /// Raw, because a thinking block has to go back exactly as it arrived and
    /// a typed round-trip would drop whatever the API adds to one next.
    #[serde(default)]
    content: Vec<serde_json::Value>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

/// The counts the API reports, where `input_tokens` is the part of the prompt
/// it did not read from the cache.
#[derive(Debug, Default, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

impl AnthropicUsage {
    fn normalize(&self, raw: serde_json::Value) -> Usage {
        Usage::new(
            self.input_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
            self.output_tokens,
        )
        .with_provider(raw)
    }
}

/// The counts of one response, or nothing where the provider reported none.
fn usage_from_value(raw: Option<serde_json::Value>) -> Option<Usage> {
    let raw = raw?;
    let counts: AnthropicUsage = serde_json::from_value(raw.clone()).unwrap_or_default();
    Some(counts.normalize(raw))
}

/// Whether a response block is one the API signs and wants back untouched.
fn is_thinking(kind: &str) -> bool {
    matches!(kind, "thinking" | "redacted_thinking")
}

impl MessagesResponse {
    fn into_llm_response(self) -> LlmResponse {
        let mut content = String::new();
        let mut thought = String::new();
        let mut tool_calls = Vec::new();
        let mut blocks = Vec::new();
        for block in self.content {
            match block
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
            {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        content.push_str(text);
                    }
                }
                "tool_use" => {
                    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                    tool_calls.push(ToolCall {
                        id: str_field(&block, "id"),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: str_field(&block, "name"),
                            arguments: input.to_string(),
                        },
                    });
                }
                kind if is_thinking(kind) => {
                    if let Some(text) = block.get("thinking").and_then(|t| t.as_str()) {
                        thought.push_str(text);
                    }
                    blocks.push(block);
                }
                _ => {}
            }
        }
        LlmResponse {
            model: self.model,
            content: (!content.is_empty()).then_some(content),
            reasoning: Reasoning::new(
                ReasoningProvider::Anthropic,
                (!thought.is_empty()).then_some(thought),
                blocks,
            ),
            tool_calls,
            finish_reason: self.stop_reason.as_deref().map(map_stop_reason),
            usage: usage_from_value(self.usage),
            cost: None,
            images: Vec::new(),
        }
    }
}

fn str_field(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

// ── Worker-format seam ───────────────────────────────────────────────────

/// The Messages API body for `request`, `stream` omitted.
pub(crate) fn request_to_wire(
    request: &LlmRequest,
    search: DeferToolsStrategy,
) -> serde_json::Value {
    serde_json::to_value(build_body(
        request,
        DEFAULT_MAX_TOKENS,
        search,
        None,
        CacheControl::EPHEMERAL,
    ))
    .unwrap_or_default()
}

/// A raw Messages API response → the neutral `LlmResponse`.
pub(crate) fn response_from_wire(value: serde_json::Value) -> Result<LlmResponse, String> {
    crate::json::from_value::<MessagesResponse>("anthropic response", value)
        .map(MessagesResponse::into_llm_response)
        .map_err(|e| e.to_string())
}

// ── Streaming events ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        message: StreamMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: StreamContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: StreamBlockDelta,
    },
    ContentBlockStop {},
    MessageDelta {
        delta: StreamMessageDelta,
        #[serde(default)]
        usage: Option<serde_json::Value>,
    },
    MessageStop {},
    Ping {},
    Error {
        error: StreamError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct StreamMessageStart {
    #[serde(default)]
    model: String,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamContentBlock {
    ToolUse {
        id: String,
        name: String,
    },
    Thinking {},
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamBlockDelta {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct StreamMessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

/// What the stream said about tokens. `message_start` reports the input and
/// what the cache did with it, `message_delta` the output, so the counts of one
/// call arrive in two events. The later event wins each field it names.
#[derive(Default)]
struct UsageAccum {
    raw: serde_json::Map<String, serde_json::Value>,
}

impl UsageAccum {
    fn merge(&mut self, reported: serde_json::Value) {
        if let serde_json::Value::Object(obj) = reported {
            self.raw.extend(obj);
        }
    }

    fn into_usage(self) -> Option<Usage> {
        match self.raw.is_empty() {
            true => None,
            false => usage_from_value(Some(serde_json::Value::Object(self.raw))),
        }
    }
}

#[derive(Debug, Deserialize)]
struct StreamError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// Per-content-block streaming accumulator.
enum BlockAccum {
    /// Text — streamed out, not retained past the delta.
    Passthrough,
    ToolUse {
        id: String,
        name: String,
        arguments: String,
    },
    /// Reassembled because the turn that follows it has to carry it back.
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

/// Folds Messages API stream events into `StreamDelta`s and the final
/// response. Shared by the server-side SSE loop and the worker-format delta
/// seam (`providers::format`).
#[derive(Default)]
pub(crate) struct StreamParser {
    content: String,
    blocks: Vec<BlockAccum>,
    finish_reason: Option<String>,
    model: Option<String>,
    usage: UsageAccum,
}

impl StreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fold one raw event payload; unknown or non-streaming events yield nothing.
    /// Errors are the worker's to surface, so an error event yields nothing too.
    pub(crate) fn parse_data(&mut self, data: &str) -> Vec<StreamDelta> {
        serde_json::from_str(data)
            .ok()
            .and_then(|event| self.on_event(event).ok().flatten())
            .into_iter()
            .collect()
    }

    fn on_event(&mut self, event: StreamEvent) -> Result<Option<StreamDelta>, LlmCallError> {
        match event {
            StreamEvent::MessageStart { message } => {
                if !message.model.is_empty() {
                    self.model = Some(message.model);
                }
                if let Some(u) = message.usage {
                    self.usage.merge(u);
                }
                Ok(None)
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                while self.blocks.len() <= index {
                    self.blocks.push(BlockAccum::Passthrough);
                }
                self.blocks[index] = match content_block {
                    StreamContentBlock::ToolUse { id, name } => BlockAccum::ToolUse {
                        id,
                        name,
                        arguments: String::new(),
                    },
                    StreamContentBlock::Thinking {} => BlockAccum::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    },
                    StreamContentBlock::RedactedThinking { data } => {
                        BlockAccum::RedactedThinking { data }
                    }
                    StreamContentBlock::Other => BlockAccum::Passthrough,
                };
                Ok(None)
            }
            StreamEvent::ContentBlockDelta { index, delta } => Ok(match delta {
                StreamBlockDelta::TextDelta { text } => {
                    self.content.push_str(&text);
                    Some(StreamDelta {
                        text: Some(text),
                        ..Default::default()
                    })
                }
                StreamBlockDelta::ThinkingDelta { thinking } => {
                    if let Some(BlockAccum::Thinking { thinking: acc, .. }) =
                        self.blocks.get_mut(index)
                    {
                        acc.push_str(&thinking);
                    }
                    (!thinking.is_empty()).then(|| StreamDelta {
                        reasoning: Some(thinking),
                        ..Default::default()
                    })
                }
                StreamBlockDelta::SignatureDelta { signature: sig } => {
                    if let Some(BlockAccum::Thinking { signature, .. }) = self.blocks.get_mut(index)
                    {
                        signature.push_str(&sig);
                    }
                    None
                }
                StreamBlockDelta::InputJsonDelta { partial_json } => {
                    if let Some(BlockAccum::ToolUse {
                        id,
                        name,
                        arguments,
                    }) = self.blocks.get_mut(index)
                    {
                        arguments.push_str(&partial_json);
                        Some(StreamDelta {
                            tool_calls: vec![ToolCallChunk {
                                id: id.clone(),
                                name: (!name.is_empty()).then(|| name.clone()),
                                arguments: Some(partial_json),
                            }],
                            ..Default::default()
                        })
                    } else {
                        None
                    }
                }
                StreamBlockDelta::Other => None,
            }),
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(u) = usage {
                    self.usage.merge(u);
                }
                Ok(delta.stop_reason.map(|reason| {
                    let mapped = map_stop_reason(&reason);
                    self.finish_reason = Some(mapped.clone());
                    StreamDelta {
                        finish_reason: Some(mapped),
                        ..Default::default()
                    }
                }))
            }
            StreamEvent::Error { error } => Err(LlmCallError::new(
                ErrorCode::ProviderError,
                format!(
                    "Anthropic stream error ({}): {}",
                    error.error_type, error.message
                ),
                matches!(error.error_type.as_str(), "overloaded_error" | "api_error"),
            )),
            StreamEvent::ContentBlockStop {}
            | StreamEvent::MessageStop {}
            | StreamEvent::Ping {}
            | StreamEvent::Unknown => Ok(None),
        }
    }

    fn into_response(self, fallback_model: &str) -> LlmResponse {
        let mut tool_calls = Vec::new();
        let mut thought = String::new();
        let mut thinking_blocks = Vec::new();
        // In index order, which is the order the API sent them and the order it
        // wants them back.
        for block in self.blocks {
            match block {
                BlockAccum::ToolUse {
                    id,
                    name,
                    arguments,
                } if !id.is_empty() => tool_calls.push(ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: ToolCallFunction { name, arguments },
                }),
                BlockAccum::Thinking {
                    thinking,
                    signature,
                } => {
                    thought.push_str(&thinking);
                    thinking_blocks.push(json!({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": signature,
                    }));
                }
                BlockAccum::RedactedThinking { data } => {
                    thinking_blocks.push(json!({ "type": "redacted_thinking", "data": data }));
                }
                _ => {}
            }
        }
        LlmResponse {
            model: self.model.unwrap_or_else(|| fallback_model.to_string()),
            content: (!self.content.is_empty()).then_some(self.content),
            reasoning: Reasoning::new(
                ReasoningProvider::Anthropic,
                (!thought.is_empty()).then_some(thought),
                thinking_blocks,
            ),
            tool_calls,
            finish_reason: self.finish_reason,
            usage: self.usage.into_usage(),
            cost: None,
            images: Vec::new(),
        }
    }
}

// ── Config / client ──────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct AnthropicConfig {
    pub base_url: String,
    pub api_key: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
    /// How long a cached prefix lives: `1h`, or the default five minutes.
    #[serde(default)]
    pub cache_ttl: Option<String>,
}

fn default_version() -> String {
    DEFAULT_VERSION.to_string()
}

fn default_max_tokens() -> u64 {
    DEFAULT_MAX_TOKENS
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            version: DEFAULT_VERSION.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            cache_ttl: None,
        }
    }
}

pub struct AnthropicClient {
    http: Client,
    base_url: String,
    headers: HeaderMap,
    default_max_tokens: u64,
    cache: CacheControl,
}

impl AnthropicClient {
    pub fn from_config(config: AnthropicConfig) -> Self {
        let mut headers = HeaderMap::new();
        if let Ok(v) = HeaderValue::from_str(&config.api_key) {
            headers.insert(HeaderName::from_static("x-api-key"), v);
        }
        if let Ok(v) = HeaderValue::from_str(&config.version) {
            headers.insert(HeaderName::from_static("anthropic-version"), v);
        }
        Self {
            http: crate::providers::http_client(),
            base_url: config.base_url,
            headers,
            default_max_tokens: config.max_tokens,
            cache: CacheControl::with_ttl(config.cache_ttl.as_deref()),
        }
    }

    async fn post_messages(
        &self,
        request: &LlmRequest,
        search: DeferToolsStrategy,
        stream: bool,
    ) -> Result<reqwest::Response, LlmCallError> {
        let body = build_body(
            request,
            self.default_max_tokens,
            search,
            Some(stream),
            self.cache,
        );
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        self.http
            .post(&url)
            .headers(self.headers.clone())
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                LlmCallError::new(
                    ErrorCode::ProviderError,
                    format!("HTTP request failed: {e}"),
                    e.is_timeout() || e.is_connect(),
                )
            })
    }
}

fn classify_error(status: reqwest::StatusCode, body: &str) -> LlmCallError {
    let status_code = status.as_u16();
    let retryable = status.is_server_error() || status_code == 408 || status_code == 429;
    let code = if status_code == 429 {
        ErrorCode::RateLimited
    } else {
        ErrorCode::ProviderError
    };
    // The provider says why in `error.message`; the rest of the body is a
    // request id and echoed request, which the log keeps and the reader does
    // not need.
    let message = match crate::json::error_message(body.as_bytes()) {
        Some(reported) => format!("Anthropic API error {status}: {reported}"),
        None => format!("Anthropic API error {status}"),
    };
    LlmCallError::new(code, message, retryable)
}

#[async_trait]
impl LlmCallable for AnthropicClient {
    async fn call(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_messages(request, ctx.defer_tools_strategy, false)
            .await?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            LlmCallError::new(ErrorCode::ProviderError, format!("read body: {e}"), true)
        })?;

        if !status.is_success() {
            tracing::warn!(status = status.as_u16(), body = %crate::json::excerpt(body.as_bytes()), "anthropic api call failed");
            return Err(classify_error(status, &body));
        }

        let parsed: MessagesResponse =
            crate::json::from_str("anthropic response", &body).map_err(LlmCallError::from)?;

        Ok(parsed.into_llm_response())
    }

    async fn call_streaming(
        &self,
        request: &LlmRequest,
        ctx: &CallContext<'_>,
        chunk_tx: UnboundedSender<StreamDelta>,
    ) -> Result<LlmResponse, LlmCallError> {
        let resp = self
            .post_messages(request, ctx.defer_tools_strategy, true)
            .await?;
        let status = resp.status();

        if !status.is_success() {
            let body = resp.text().await.map_err(|e| {
                LlmCallError::new(ErrorCode::ProviderError, format!("read body: {e}"), true)
            })?;
            return Err(classify_error(status, &body));
        }

        let mut parser = StreamParser::new();

        let byte_stream = resp.bytes_stream();
        let mut stream =
            tokio_stream::StreamExt::map(byte_stream, |chunk| chunk.map_err(std::io::Error::other));
        let mut line_buf = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| {
                LlmCallError::new(ErrorCode::ProviderError, format!("stream read: {e}"), true)
            })?;
            line_buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();

                // Anthropic sends both `event:` and `data:` lines; the `data:`
                // JSON carries its own `type`, so we only parse those.
                let data = match crate::providers::sse_data(&line) {
                    Some(d) => d,
                    None => continue,
                };

                let event: StreamEvent = match serde_json::from_str(data) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                if let Some(delta) = parser.on_event(event)? {
                    let _ = chunk_tx.send(delta);
                }
            }
        }

        Ok(parser.into_response(&request.model))
    }
}

pub struct AnthropicProvider {
    client: Arc<AnthropicClient>,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            client: Arc::new(AnthropicClient::from_config(config)),
        }
    }
}

#[async_trait]
impl LlmProviderTrait for AnthropicProvider {
    async fn resolve(&self, _owner: &SessionOwner) -> Result<Arc<dyn LlmCallable>, String> {
        Ok(self.client.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{DraftMessage, LlmTool, ReasoningConfig};
    use serde_json::json;

    fn msg(role: Role, content: Option<&str>) -> DraftMessage {
        DraftMessage {
            id: None,
            role,
            content: content.map(|c| Content::Text(c.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning: None,
        }
    }

    fn tool_call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    fn req(messages: Vec<DraftMessage>) -> LlmRequest {
        LlmRequest {
            model: "claude-opus-4-8".to_string(),
            messages,
            tools: None,
            temperature: None,
            max_completion_tokens: None,
            reasoning: None,
        }
    }

    fn thinking_block() -> serde_json::Value {
        json!({ "type": "thinking", "thinking": "let me check", "signature": "sig-abc" })
    }

    #[test]
    fn a_pdf_file_part_becomes_a_document_block() {
        let parts = vec![
            ContentPart::File {
                file: crate::protocol::FileData {
                    filename: "q3.pdf".into(),
                    file_data: "data:application/pdf;base64,AQID".into(),
                },
            },
            // Not a Messages API shape: dropped rather than sent broken.
            ContentPart::File {
                file: crate::protocol::FileData {
                    filename: "deck.pptx".into(),
                    file_data: "data:application/vnd.ms-powerpoint;base64,AQID".into(),
                },
            },
        ];
        let blocks = content_to_blocks(Some(&Content::Parts(parts)));
        assert_eq!(blocks.len(), 1);
        let v = serde_json::to_value(&blocks[0]).unwrap();
        assert_eq!(v["type"], "document");
        assert_eq!(v["source"]["type"], "base64");
        assert_eq!(v["source"]["media_type"], "application/pdf");
        assert_eq!(v["source"]["data"], "AQID");
    }

    #[test]
    fn a_thought_response_keeps_the_blocks_and_reads_the_text() {
        let parsed = response_from_wire(json!({
            "model": "claude-opus-5",
            "content": [
                thinking_block(),
                { "type": "redacted_thinking", "data": "opaque" },
                { "type": "text", "text": "done" },
            ],
            "stop_reason": "end_turn",
        }))
        .expect("parses");

        let reasoning = parsed.reasoning.expect("reasoning");
        assert_eq!(reasoning.provider, ReasoningProvider::Anthropic);
        assert_eq!(reasoning.text.as_deref(), Some("let me check"));
        assert_eq!(
            reasoning.blocks,
            vec![
                thinking_block(),
                json!({ "type": "redacted_thinking", "data": "opaque" }),
            ]
        );
        assert_eq!(parsed.content.as_deref(), Some("done"));
    }

    #[test]
    fn a_response_that_did_not_think_carries_no_reasoning() {
        let parsed = response_from_wire(json!({
            "model": "claude-opus-5",
            "content": [{ "type": "text", "text": "hi" }],
        }))
        .expect("parses");
        assert!(parsed.reasoning.is_none());
    }

    #[test]
    fn thinking_leads_the_turn_it_thought_for() {
        let mut assistant = msg(Role::Assistant, Some("checking"));
        assistant.tool_calls = Some(vec![tool_call("t1", "lookup", "{}")]);
        assistant.reasoning = Reasoning::new(
            ReasoningProvider::Anthropic,
            Some("let me check".to_string()),
            vec![thinking_block()],
        );
        let body = request_to_wire(
            &req(vec![msg(Role::User, Some("go")), assistant]),
            DeferToolsStrategy::Search,
        );

        let blocks = &body["messages"][1]["content"];
        assert_eq!(blocks[0], thinking_block(), "thinking comes first");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[2]["type"], "tool_use");
    }

    #[test]
    fn another_providers_thinking_does_not_ride_along() {
        let mut assistant = msg(Role::Assistant, Some("hi"));
        assistant.reasoning = Reasoning::new(
            ReasoningProvider::Openrouter,
            Some("thought".to_string()),
            vec![json!({ "type": "reasoning.text", "text": "thought" })],
        );
        let body = request_to_wire(
            &req(vec![msg(Role::User, Some("go")), assistant]),
            DeferToolsStrategy::Search,
        );

        let blocks = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
    }

    #[test]
    fn a_streamed_thought_is_reassembled_with_its_signature() {
        let mut parser = StreamParser::new();
        for event in [
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "let me "}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "check"}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": "sig-abc"}}),
            json!({"type": "content_block_stop"}),
        ] {
            parser.parse_data(&event.to_string());
        }

        let response = parser.into_response("claude-opus-5");
        let reasoning = response.reasoning.expect("reasoning");
        assert_eq!(reasoning.text.as_deref(), Some("let me check"));
        assert_eq!(reasoning.blocks, vec![thinking_block()]);
    }

    #[test]
    fn extracts_system_and_maps_user_text() {
        let body = build_body(
            &req(vec![
                msg(Role::System, Some("be nice")),
                msg(Role::User, Some("hi")),
            ]),
            4096,
            DeferToolsStrategy::Search,
            Some(false),
            CacheControl::EPHEMERAL,
        );
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["system"][0]["text"], "be nice");
        assert_eq!(v["max_tokens"], 4096);
        assert_eq!(v["stream"], false);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"][0]["type"], "text");
        assert_eq!(v["messages"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn folds_tool_results_into_one_user_turn_and_maps_tool_calls() {
        let mut assistant = msg(Role::Assistant, None);
        assistant.tool_calls = Some(vec![
            tool_call("call_1", "get_weather", r#"{"city":"NYC"}"#),
            tool_call("call_2", "get_time", "{}"),
        ]);
        let mut result_1 = msg(Role::Tool, Some("72F"));
        result_1.tool_call_id = Some("call_1".to_string());
        let mut result_2 = msg(Role::Tool, Some("noon"));
        result_2.tool_call_id = Some("call_2".to_string());

        let body = build_body(
            &req(vec![
                msg(Role::User, Some("weather?")),
                assistant,
                result_1,
                result_2,
            ]),
            4096,
            DeferToolsStrategy::Search,
            Some(false),
            CacheControl::EPHEMERAL,
        );
        let v = serde_json::to_value(&body).unwrap();

        // user(weather?), assistant(2 tool_use), user(2 tool_result) — coalesced
        assert_eq!(v["messages"].as_array().unwrap().len(), 3);
        assert_eq!(v["messages"][1]["role"], "assistant");
        assert_eq!(v["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(v["messages"][1]["content"][0]["id"], "call_1");
        assert_eq!(v["messages"][1]["content"][0]["name"], "get_weather");
        assert_eq!(v["messages"][1]["content"][0]["input"]["city"], "NYC");
        assert_eq!(v["messages"][2]["role"], "user");
        assert_eq!(v["messages"][2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(v["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(v["messages"][2]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(v["messages"][2]["content"][0]["content"], "72F");
        assert_eq!(v["messages"][2]["content"][1]["tool_use_id"], "call_2");
    }

    #[test]
    fn every_cacheable_part_of_the_request_carries_a_breakpoint() {
        let mut r = req(vec![
            msg(Role::System, Some("be nice")),
            msg(Role::User, Some("hello")),
        ]);
        r.tools = Some(vec![
            LlmTool {
                name: "a".into(),
                description: String::new(),
                input: None,
                output: None,
                defer: false,
            },
            LlmTool {
                name: "b".into(),
                description: String::new(),
                input: None,
                output: None,
                defer: false,
            },
        ]);
        let v = request_to_wire(&r, DeferToolsStrategy::Search);
        let mark = serde_json::json!({ "type": "ephemeral" });

        assert_eq!(v["tools"][1]["cache_control"], mark, "after the tools");
        assert!(
            v["tools"][0].get("cache_control").is_none(),
            "one breakpoint for the block, not one for each tool"
        );
        assert_eq!(v["system"][0]["cache_control"], mark, "after the system");
        let turns = v["messages"].as_array().expect("messages");
        let last = turns.last().expect("a turn");
        let block = last["content"].as_array().unwrap().last().unwrap();
        assert_eq!(
            block["cache_control"], mark,
            "after the last block, so the transcript is cached as it grows"
        );
    }

    /// One user message of many blocks, as a transcript reaches the engine
    /// after a run of parallel tool calls.
    fn wide_message(blocks: usize) -> DraftMessage {
        let parts = (0..blocks)
            .map(|i| ContentPart::Text {
                text: format!("block {i}"),
            })
            .collect();
        let mut m = msg(Role::User, None);
        m.content = Some(Content::Parts(parts));
        m
    }

    fn wide_turn(blocks: usize) -> LlmRequest {
        req(vec![wide_message(blocks)])
    }

    fn marked_indexes(v: &serde_json::Value) -> Vec<usize> {
        v["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .flat_map(|t| t["content"].as_array().unwrap())
            .enumerate()
            .filter(|(_, b)| b.get("cache_control").is_some())
            .map(|(i, _)| i)
            .collect()
    }

    /// The one tool a test needs to spend the fourth breakpoint.
    fn one_tool() -> Vec<LlmTool> {
        vec![LlmTool {
            name: "a".into(),
            description: String::new(),
            input: None,
            output: None,
            defer: false,
        }]
    }

    #[test]
    fn the_transcript_is_marked_at_the_last_two_user_turns() {
        let v = request_to_wire(
            &req(vec![
                msg(Role::User, Some("one")),
                msg(Role::Assistant, Some("a")),
                msg(Role::User, Some("two")),
                msg(Role::Assistant, Some("b")),
                msg(Role::User, Some("three")),
            ]),
            DeferToolsStrategy::Search,
        );
        // One block per turn, so the marks land on turns three and five.
        assert_eq!(marked_indexes(&v), vec![2, 4]);

        // A transcript of one turn has only its own end to mark.
        let v = request_to_wire(&wide_turn(8), DeferToolsStrategy::Search);
        assert_eq!(marked_indexes(&v), vec![7]);
    }

    #[test]
    fn a_turn_of_any_width_leaves_the_older_mark_where_the_last_request_put_it() {
        // The request before this one ended at the first user turn, and marked
        // its last block. However wide the turn between them, this request
        // marks that same block again, and the provider reads the entry back.
        let mut assistant = msg(Role::Assistant, None);
        assistant.tool_calls = Some(
            (0..10)
                .map(|i| tool_call(&i.to_string(), "t", "{}"))
                .collect(),
        );
        let r = req(vec![
            msg(Role::User, Some("go")),
            assistant,
            wide_message(10),
        ]);
        let v = request_to_wire(&r, DeferToolsStrategy::Search);
        // user(1) assistant(10) user(10): 21 blocks, more than the provider
        // looks back, and the marks still name both ends.
        assert_eq!(marked_indexes(&v), vec![0, 20]);
    }

    #[test]
    fn a_request_carries_no_more_breakpoints_than_the_api_allows() {
        let mut r = req(vec![
            msg(Role::System, Some("be nice")),
            msg(Role::User, Some("one")),
            msg(Role::Assistant, Some("a")),
            msg(Role::User, Some("two")),
        ]);
        r.tools = Some(one_tool());
        let v = request_to_wire(&r, DeferToolsStrategy::Search);
        let marks = 1 + 1 + marked_indexes(&v).len();
        assert_eq!(marks, 4, "tools, system, and the last two user turns");
    }

    #[test]
    fn the_configured_life_rides_every_breakpoint() {
        let mut r = req(vec![
            msg(Role::System, Some("be nice")),
            msg(Role::User, Some("one")),
            msg(Role::Assistant, Some("a")),
            msg(Role::User, Some("two")),
        ]);
        r.tools = Some(one_tool());
        let v = serde_json::to_value(build_body(
            &r,
            4096,
            DeferToolsStrategy::Search,
            Some(false),
            CacheControl::with_ttl(Some("1h")),
        ))
        .unwrap();
        let hour = json!({ "type": "ephemeral", "ttl": "1h" });
        assert_eq!(v["tools"][0]["cache_control"], hour);
        assert_eq!(v["system"][0]["cache_control"], hour);
        assert_eq!(v["messages"][0]["content"][0]["cache_control"], hour);
        assert_eq!(v["messages"][2]["content"][0]["cache_control"], hour);

        // Anything else is the default five minutes, which sends no ttl.
        assert_eq!(
            serde_json::to_value(CacheControl::with_ttl(Some("5m"))).unwrap(),
            json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn a_streamed_call_reports_what_the_cache_did() {
        let mut parser = StreamParser::new();
        parser.parse_data(
            r#"{"type":"message_start","message":{"model":"claude-opus-4-8","usage":{"input_tokens":12,"cache_read_input_tokens":9000,"cache_creation_input_tokens":300}}}"#,
        );
        parser.parse_data(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
        );
        let usage = parser
            .into_response("claude-opus-4-8")
            .usage
            .expect("usage");
        assert_eq!(usage.uncached_input, 12);
        assert_eq!(usage.cache_read, 9000);
        assert_eq!(usage.cache_write, 300);
        assert_eq!(usage.output, 7);
        assert_eq!(usage.input, 9312, "every input token, cached or not");
        assert_eq!(usage.total, 9319);
        // The counts the provider sent stay whole, nested fields included.
        let raw = usage.provider.expect("the provider report");
        assert_eq!(raw["cache_read_input_tokens"], 9000);
    }

    /// The counts of two vendors add up only because the adapter states them
    /// the same way. Anthropic reports the uncached part of the prompt, OpenAI
    /// reports the whole prompt, and a tree that uses both adds them together.
    #[test]
    fn the_counts_of_two_vendors_add_up() {
        let anthropic = usage_from_value(Some(json!({
            "input_tokens": 1000,
            "cache_read_input_tokens": 9000,
            "output_tokens": 200
        })))
        .expect("usage");
        let openai = crate::providers::openai::usage_from_value(Some(json!({
            "prompt_tokens": 10000,
            "completion_tokens": 200,
            "prompt_tokens_details": { "cached_tokens": 9000 }
        })))
        .expect("usage");
        assert_eq!(anthropic.input, openai.input, "the same prompt");
        assert_eq!(anthropic.cache_read, openai.cache_read);

        let mut total = anthropic.clone();
        total.add(&openai);
        assert_eq!(total.input, 20000);
        assert_eq!(total.cache_read, 18000);
        assert_eq!(total.output, 400);
        assert!(total.provider.is_none(), "a sum reports for no one call");
    }

    #[test]
    fn a_stream_that_says_nothing_about_tokens_reports_no_usage() {
        let mut parser = StreamParser::new();
        parser.parse_data(r#"{"type":"message_start","message":{"model":"claude-opus-4-8"}}"#);
        assert!(parser.into_response("claude-opus-4-8").usage.is_none());
    }

    #[test]
    fn a_request_with_no_tools_and_no_system_still_caches_the_transcript() {
        let r = req(vec![msg(Role::User, Some("hello"))]);
        let v = request_to_wire(&r, DeferToolsStrategy::Search);
        assert!(v.get("system").is_none());
        assert!(v.get("tools").is_none());
        let block = v["messages"][0]["content"][0].clone();
        assert_eq!(
            block["cache_control"],
            serde_json::json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn maps_tools_effort_and_max_tokens() {
        let mut r = req(vec![msg(Role::User, Some("hi"))]);
        r.tools = Some(vec![
            LlmTool {
                name: "f".to_string(),
                description: "d".to_string(),
                input: Some(json!({"type": "object"})),
                output: None,
                defer: false,
            },
            LlmTool {
                name: "no_args".to_string(),
                description: "d".to_string(),
                input: None,
                output: None,
                defer: false,
            },
        ]);
        r.max_completion_tokens = Some(1000);
        r.reasoning = Some(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            max_tokens: None,
            exclude: None,
            enabled: None,
        });
        let v = serde_json::to_value(build_body(
            &r,
            4096,
            DeferToolsStrategy::Search,
            Some(true),
            CacheControl::EPHEMERAL,
        ))
        .unwrap();
        assert_eq!(v["max_tokens"], 1000);
        assert_eq!(v["tools"][0]["name"], "f");
        assert_eq!(v["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(
            v["tools"][1]["input_schema"],
            json!({"type": "object", "properties": {}}),
            "a no-input tool gets the empty object schema"
        );
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["output_config"]["effort"], "high");
        assert_eq!(v["stream"], true);
        // system omitted when absent
        assert!(v.get("system").is_none());
    }

    /// `(thinking, output_config)` as JSON, where an omitted field is null.
    fn reasoning(
        model: &str,
        effort: Option<ReasoningEffort>,
        max_tokens: u64,
    ) -> (serde_json::Value, serde_json::Value) {
        let (thinking, output_config) = reasoning_fields(model, effort, max_tokens);
        (
            serde_json::to_value(thinking).unwrap(),
            serde_json::to_value(output_config).unwrap(),
        )
    }

    const HIGH: Option<ReasoningEffort> = Some(ReasoningEffort::High);
    const XHIGH: Option<ReasoningEffort> = Some(ReasoningEffort::Xhigh);

    #[test]
    fn a_model_this_build_does_not_know_reads_as_the_current_generation() {
        let (thinking, config) = reasoning("claude-next-ultra", XHIGH, 64_000);
        assert_eq!(thinking["type"], "adaptive");
        assert_eq!(config["effort"], "xhigh");

        // Off stays off: `disabled` would be a 400 on a model that refuses it.
        let (thinking, config) = reasoning("claude-next-ultra", None, 64_000);
        assert!(thinking.is_null());
        assert!(config.is_null());
    }

    #[test]
    fn a_model_that_thinks_unasked_is_asked_to_stop() {
        for model in ["claude-opus-5", "claude-sonnet-5"] {
            let (thinking, config) = reasoning(model, None, 64_000);
            assert_eq!(thinking["type"], "disabled", "{model}");
            // No effort with it: `disabled` above `high` is a 400, and an
            // absent effort leaves the model on its own default.
            assert!(config.is_null(), "{model}");
        }
    }

    #[test]
    fn a_model_that_always_thinks_is_never_asked_to_stop() {
        for model in ["claude-fable-5", "claude-mythos-5"] {
            let (thinking, _) = reasoning(model, None, 64_000);
            assert!(thinking.is_null(), "{model} rejects an explicit disabled");
        }
    }

    #[test]
    fn xhigh_falls_to_high_on_a_model_that_predates_it() {
        assert_eq!(
            reasoning("claude-opus-4-8", XHIGH, 64_000).1["effort"],
            "xhigh"
        );
        assert_eq!(
            reasoning("claude-opus-4-7", XHIGH, 64_000).1["effort"],
            "xhigh"
        );
        assert_eq!(
            reasoning("claude-opus-4-6", XHIGH, 64_000).1["effort"],
            "high"
        );
        assert_eq!(
            reasoning("claude-sonnet-4-6", XHIGH, 64_000).1["effort"],
            "high"
        );
    }

    #[test]
    fn a_model_before_adaptive_thinking_takes_a_budget() {
        // Sonnet 4.5 reads neither effort nor adaptive.
        let (thinking, config) = reasoning("claude-sonnet-4-5", HIGH, 64_000);
        assert_eq!(thinking["type"], "enabled");
        assert_eq!(thinking["budget_tokens"], 24_000);
        assert!(config.is_null());

        // Opus 4.5 reads effort, and still thinks by budget.
        let (thinking, config) = reasoning("claude-opus-4-5", HIGH, 64_000);
        assert_eq!(thinking["type"], "enabled");
        assert_eq!(config["effort"], "high");

        // Haiku 4.5 is the same generation as Sonnet 4.5, not Opus 4.5.
        let (_, config) = reasoning("claude-haiku-4-5", HIGH, 64_000);
        assert!(config.is_null());
    }

    #[test]
    fn a_budget_leaves_the_answer_room() {
        // The budget comes out of max_tokens.
        let (thinking, _) = reasoning("claude-sonnet-4-5", HIGH, 4096);
        assert_eq!(thinking["budget_tokens"], 3072);

        // Below the floor there is no budget that leaves an answer.
        let (thinking, _) = reasoning("claude-sonnet-4-5", HIGH, 2000);
        assert!(thinking.is_null());
    }

    #[test]
    fn only_a_model_that_reads_temperature_is_sent_one() {
        let body = |model: &str| {
            let mut r = req(vec![msg(Role::User, Some("hi"))]);
            r.model = model.to_string();
            r.temperature = Some(0.7);
            serde_json::to_value(build_body(
                &r,
                4096,
                DeferToolsStrategy::Search,
                None,
                CacheControl::EPHEMERAL,
            ))
            .unwrap()
        };

        // Opus 4.7 and later refuse it, so it stays off the request.
        for model in ["claude-opus-5", "claude-opus-4-8", "claude-fable-5"] {
            assert!(body(model).get("temperature").is_none(), "{model}");
        }
        // Older models read it, and a request that set one meant it.
        for model in ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-6"] {
            assert_eq!(body(model)["temperature"], 0.7, "{model}");
        }
    }

    #[test]
    fn a_model_writes_no_more_than_it_can() {
        // Asking a model for more than its ceiling is a 400, so the ask is
        // held to the ceiling.
        assert_eq!(output_tokens("claude-opus-5", 128_000), 128_000);
        assert_eq!(output_tokens("claude-haiku-4-5", 128_000), 64_000);
        assert_eq!(output_tokens("claude-opus-4-1", 128_000), 32_000);
        // A smaller ask is the ask.
        assert_eq!(output_tokens("claude-opus-5", 4_096), 4_096);
        // An unplaced name has no known ceiling to hold it to.
        assert_eq!(output_tokens("claude-next-ultra", 200_000), 200_000);
    }

    #[test]
    fn a_held_ceiling_is_what_the_budget_comes_out_of() {
        let mut r = req(vec![msg(Role::User, Some("hi"))]);
        r.model = "claude-haiku-4-5".to_string();
        r.max_completion_tokens = Some(128_000);
        r.reasoning = Some(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            max_tokens: None,
            exclude: None,
            enabled: None,
        });
        let v = serde_json::to_value(build_body(
            &r,
            4096,
            DeferToolsStrategy::Search,
            None,
            CacheControl::EPHEMERAL,
        ))
        .unwrap();
        assert_eq!(v["max_tokens"], 64_000);
        // Haiku 4.5 predates adaptive thinking, and its budget has to fit the
        // held ceiling rather than the ask.
        assert_eq!(v["thinking"]["type"], "enabled");
        assert_eq!(v["thinking"]["budget_tokens"], 24_000);
    }

    #[test]
    fn a_date_stamp_is_not_a_minor_version() {
        assert_eq!(version("claude-opus-4-20250514", "opus"), Some((4, 0)));
        assert_eq!(version("claude-opus-4-8", "opus"), Some((4, 8)));
        assert_eq!(version("claude-opus-5", "opus"), Some((5, 0)));
        assert_eq!(version("claude-opus-4-6-latest", "opus"), Some((4, 6)));
        assert_eq!(version("claude-sonnet-4-6", "opus"), None);
        assert_eq!(
            version("anthropic/claude-opus-4-8", "opus"),
            Some((4, 8)),
            "a gateway prefix does not hide the version"
        );

        // Opus 4.0 predates every reasoning field, and a date stamp read as a
        // minor would have handed it all of them.
        let (thinking, config) = reasoning("claude-opus-4-20250514", HIGH, 64_000);
        assert_eq!(thinking["type"], "enabled");
        assert!(config.is_null());
    }

    #[test]
    fn parses_response_blocks_into_content_and_tool_calls() {
        let raw = json!({
            "model": "claude-opus-4-8",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "NYC"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })
        .to_string();

        let resp = serde_json::from_str::<MessagesResponse>(&raw)
            .unwrap()
            .into_llm_response();

        assert_eq!(resp.content.as_deref(), Some("Hello world"));
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "tu_1");
        assert_eq!(resp.tool_calls[0].function.name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&resp.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
        let usage = resp.usage.expect("usage");
        assert_eq!(usage.uncached_input, 10);
        assert_eq!(
            usage.input, 10,
            "nothing was cached, so the input is the whole prompt"
        );
        assert_eq!(usage.output, 5);
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("refusal"), "refusal");
    }

    #[test]
    fn deserializes_stream_events() {
        let start: StreamEvent = serde_json::from_str(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather","input":{}}}"#,
        )
        .unwrap();
        match start {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 1);
                assert!(matches!(content_block, StreamContentBlock::ToolUse { .. }));
            }
            _ => panic!("expected content_block_start"),
        }

        let delta: StreamEvent = serde_json::from_str(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
        )
        .unwrap();
        assert!(matches!(
            delta,
            StreamEvent::ContentBlockDelta {
                delta: StreamBlockDelta::InputJsonDelta { .. },
                ..
            }
        ));

        let msg_delta: StreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
        )
        .unwrap();
        match msg_delta {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason.as_deref(), Some("tool_use"));
                assert_eq!(usage.unwrap()["output_tokens"], 7);
            }
            _ => panic!("expected message_delta"),
        }

        assert!(matches!(
            serde_json::from_str::<StreamEvent>(r#"{"type":"ping"}"#).unwrap(),
            StreamEvent::Ping {}
        ));
        assert!(matches!(
            serde_json::from_str::<StreamEvent>(r#"{"type":"future_event","x":1}"#).unwrap(),
            StreamEvent::Unknown
        ));
    }
}
