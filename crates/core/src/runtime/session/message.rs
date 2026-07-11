//! Behavior over the protocol message types; the types live in [`crate::protocol`].

use crate::protocol::{Content, ContentPart};

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
