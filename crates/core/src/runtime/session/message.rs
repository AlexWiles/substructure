use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

// ── Multimodal content parts (OpenAI/OpenRouter wire format) ─────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    pub filename: String,
    pub file_data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
    },
    File {
        file: FileData,
    },
    #[serde(rename = "input_audio")]
    InputAudio {
        input_audio: AudioData,
    },
    #[serde(rename = "video_url")]
    VideoUrl {
        video_url: VideoUrl,
    },
}

/// Message content: either a plain string or an array of typed parts.
/// Serializes as a raw string or array respectively (untagged).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Content {
    /// Extract the concatenated text from this content, ignoring non-text parts.
    pub fn text(&self) -> Option<&str> {
        match self {
            Content::Text(s) => Some(s.as_str()),
            Content::Parts(parts) => {
                // For single-text-part messages, return a direct reference.
                // For mixed content, callers should use text_owned().
                if parts.len() == 1 {
                    if let ContentPart::Text { text } = &parts[0] {
                        return Some(text.as_str());
                    }
                }
                None
            }
        }
    }

    /// Extract concatenated text from all text parts, allocating if needed.
    pub fn text_owned(&self) -> String {
        match self {
            Content::Text(s) => s.clone(),
            Content::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
