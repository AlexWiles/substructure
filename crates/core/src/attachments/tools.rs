use std::borrow::Cow;

use serde_json::{json, Value};

use super::{Attachment, Attachments, Tool, READ, VIEW};
use crate::connectors::filter::engine_tool;
use crate::mime;
use crate::protocol::{ConnectorTool, ConnectorToolKind, StoredContent, StoredResult};
use crate::runtime::blob::{BlobRef, BlobStore};

const CHUNK: usize = 64 * 1024;
const MAX_LINE: usize = 2000;
const MAX_READ: u64 = 16 * 1024 * 1024;
/// Base64 is 4/3 the bytes, and a provider takes 10 MB encoded for one file.
const MAX_VIEW: u64 = 5 * 1024 * 1024;

pub struct Call {
    pub tool: Tool,
    pub arguments: String,
    pub attachments: Vec<Attachment>,
}

pub fn definitions(attachments: &Attachments) -> Vec<ConnectorTool> {
    let mut tools = attachments.tools.clone();
    tools.sort();
    tools.dedup();
    tools.into_iter().map(definition).collect()
}

fn definition(tool: Tool) -> ConnectorTool {
    let (description, input) = match tool {
        Tool::Read => (
            "Read a text attachment. Answers one chunk, with the total size and line count so \
             you can ask for the next. Name the attachment by the id in its `[attachment …]` \
             line."
                .to_string(),
            json!({
                "type": "object",
                "properties": {
                    "attachment": { "type": "string", "description": ID },
                    "from_line": { "type": "integer", "description": "First line to answer with, counting from 1." },
                    "lines": { "type": "integer", "description": "How many lines to answer with." }
                },
                "required": ["attachment"]
            }),
        ),
        Tool::View => (
            "Look at an image, audio, or video attachment. Answers with the file itself. Name \
             the attachment by the id in its `[attachment …]` line."
                .to_string(),
            json!({
                "type": "object",
                "properties": { "attachment": { "type": "string", "description": ID } },
                "required": ["attachment"]
            }),
        ),
    };
    engine_tool(
        tool.name(),
        description,
        input,
        ConnectorToolKind::Attachment,
    )
}

const ID: &str = "The attachment id, exactly as its line gives it.";

pub async fn answer(call: Call, blobs: &dyn BlobStore) -> StoredResult {
    let arguments: Value = serde_json::from_str(&call.arguments).unwrap_or_default();
    let named = arguments
        .get("attachment")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let Some(attachment) = call.attachments.iter().find(|a| a.id == named) else {
        return StoredResult::error(format!(
            "no attachment `{named}` in this conversation. Attachments: {}",
            crate::copy::declared(call.attachments.iter().map(|a| a.id.as_str()))
        ));
    };
    match call.tool {
        Tool::Read => read(attachment, &arguments, blobs).await,
        Tool::View => view(attachment),
    }
}

fn view(attachment: &Attachment) -> StoredResult {
    if !matches!(mime::essence(&attachment.mime), "image" | "audio" | "video") {
        return StoredResult::error(format!(
            "`{}` is {}; {VIEW} takes image, audio, and video attachments",
            attachment.id, attachment.mime
        ));
    }
    if attachment.size > MAX_VIEW {
        return StoredResult::error(too_big(attachment, VIEW, MAX_VIEW));
    }
    StoredResult {
        content: vec![StoredContent::Blob {
            uri: attachment.uri.clone(),
        }],
        ..Default::default()
    }
}

async fn read(attachment: &Attachment, arguments: &Value, blobs: &dyn BlobStore) -> StoredResult {
    if attachment.size > MAX_READ {
        return StoredResult::error(too_big(attachment, READ, MAX_READ));
    }
    let Some(stored) = BlobRef::parse(&attachment.uri) else {
        return StoredResult::error(format!("`{}` is not stored", attachment.id));
    };
    let bytes = match blobs.get(&stored).await {
        Ok(bytes) => bytes,
        Err(e) => return StoredResult::error(format!("`{}` is unreadable: {e}", attachment.id)),
    };
    let text = String::from_utf8_lossy(&bytes);
    if !text_enough(&text) {
        return StoredResult::error(format!(
            "`{}` is {}; {READ} takes text attachments",
            attachment.id, attachment.mime
        ));
    }
    let number = |key: &str| {
        arguments
            .get(key)
            .and_then(Value::as_u64)
            .map(|n| n as usize)
    };
    let lines: Vec<&str> = text.lines().collect();
    let (window, chunk) = by_line(&lines, number("from_line").unwrap_or(1), number("lines"));
    StoredResult::text(format!(
        "{} {} {}, {} lines\n{window}\n\n{chunk}",
        attachment.id,
        attachment.mime,
        crate::size::text(attachment.size),
        lines.len(),
    ))
}

fn too_big(attachment: &Attachment, tool: &str, cap: u64) -> String {
    format!(
        "`{}` is {}; {tool} takes up to {}",
        attachment.id,
        crate::size::text(attachment.size),
        crate::size::text(cap)
    )
}

fn text_enough(text: &str) -> bool {
    let head: Vec<char> = text.chars().take(CHUNK).collect();
    let replaced = head
        .iter()
        .filter(|c| **c == char::REPLACEMENT_CHARACTER)
        .count();
    replaced * 100 <= head.len()
}

fn by_line(lines: &[&str], from: usize, count: Option<usize>) -> (String, String) {
    let start = from.saturating_sub(1).min(lines.len());
    let mut chunk = String::new();
    let mut taken = 0;
    for raw in lines.iter().skip(start).take(count.unwrap_or(usize::MAX)) {
        let line = shorten(raw);
        if !chunk.is_empty() && chunk.len() + line.len() + 1 > CHUNK {
            break;
        }
        if !chunk.is_empty() {
            chunk.push('\n');
        }
        chunk.push_str(&line);
        taken += 1;
    }
    let window = format!("lines {}-{} of {}", start + 1, start + taken, lines.len());
    (window, chunk)
}

fn shorten(line: &str) -> Cow<'_, str> {
    match line.char_indices().nth(MAX_LINE) {
        None => Cow::Borrowed(line),
        Some((at, _)) => Cow::Owned(format!("{}… [truncated]", &line[..at])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::blob::{MemoryBlobStore, NewBlob};

    async fn stored(mime: &str, bytes: Vec<u8>) -> (MemoryBlobStore, Attachment) {
        let blobs = MemoryBlobStore::new();
        let stored = blobs
            .put(NewBlob {
                tenant_id: "t1".to_string(),
                mime: mime.to_string(),
                name: Some("notes.txt".to_string()),
                bytes,
            })
            .await
            .expect("stored");
        let attachment = Attachment {
            id: "notes.txt".to_string(),
            mime: mime.to_string(),
            size: stored.size,
            uri: stored.uri(),
        };
        (blobs, attachment)
    }

    fn call(tool: Tool, arguments: &str, attachment: &Attachment) -> Call {
        Call {
            tool,
            arguments: arguments.to_string(),
            attachments: vec![attachment.clone()],
        }
    }

    #[tokio::test]
    async fn read_answers_a_chunk_and_says_how_much_there_is() {
        let body = (1..=100)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (blobs, attachment) = stored("text/plain", body.into_bytes()).await;

        let all = answer(
            call(Tool::Read, "{\"attachment\":\"notes.txt\"}", &attachment),
            &blobs,
        )
        .await
        .as_text();
        assert!(all.contains("notes.txt text/plain"), "{all}");
        assert!(all.contains("100 lines"), "{all}");
        assert!(all.contains("line 100"), "{all}");

        let paged = answer(
            call(
                Tool::Read,
                "{\"attachment\":\"notes.txt\",\"from_line\":10,\"lines\":2}",
                &attachment,
            ),
            &blobs,
        )
        .await
        .as_text();
        assert!(paged.contains("lines 10-11 of 100"), "{paged}");
        assert!(paged.ends_with("line 10\nline 11"), "{paged}");
    }

    #[tokio::test]
    async fn read_cuts_a_long_line_and_says_it_cut_it() {
        let long = "x".repeat(MAX_LINE + 500);
        let (blobs, attachment) = stored("text/plain", format!("{long}\nshort").into_bytes()).await;
        let text = answer(
            call(Tool::Read, "{\"attachment\":\"notes.txt\"}", &attachment),
            &blobs,
        )
        .await
        .as_text();
        assert!(
            text.contains(&format!("{}… [truncated]", "x".repeat(MAX_LINE))),
            "{text}"
        );
        assert!(text.ends_with("\nshort"), "the next line still follows");
    }

    #[test]
    fn a_chunk_stays_within_its_bound_however_long_the_lines_are() {
        let long: Vec<String> = (0..500).map(|_| "\u{4e00}".repeat(MAX_LINE * 2)).collect();
        let lines: Vec<&str> = long.iter().map(String::as_str).collect();
        let (_, chunk) = by_line(&lines, 1, None);
        assert!(
            chunk.len() <= CHUNK,
            "{} bytes is over the {CHUNK} bound",
            chunk.len()
        );
    }

    #[test]
    fn a_line_at_the_cap_is_left_alone_and_one_over_it_is_cut() {
        assert_eq!(shorten("hello"), "hello");
        assert_eq!(shorten(&"é".repeat(MAX_LINE)), "é".repeat(MAX_LINE));
        assert_eq!(
            shorten(&"é".repeat(MAX_LINE + 1)),
            format!("{}… [truncated]", "é".repeat(MAX_LINE)),
            "the cap counts characters, not bytes"
        );
    }

    #[tokio::test]
    async fn view_refuses_a_file_too_big_for_the_wire() {
        let (blobs, attachment) = stored("image/png", vec![0; MAX_VIEW as usize + 1]).await;
        let answer = answer(
            call(Tool::View, "{\"attachment\":\"notes.txt\"}", &attachment),
            &blobs,
        )
        .await;
        assert!(answer.is_error);
        assert_eq!(
            answer.as_text(),
            "`notes.txt` is 5.0 MB; attachment_view takes up to 5.0 MB"
        );
    }

    #[tokio::test]
    async fn read_refuses_a_file_that_is_not_text() {
        let (blobs, attachment) = stored("application/zip", vec![0xff; 64]).await;
        let answer = answer(
            call(Tool::Read, "{\"attachment\":\"notes.txt\"}", &attachment),
            &blobs,
        )
        .await;
        assert!(answer.is_error);
        assert_eq!(
            answer.as_text(),
            "`notes.txt` is application/zip; attachment_read takes text attachments"
        );
    }

    #[tokio::test]
    async fn view_answers_with_the_stored_bytes() {
        let (blobs, attachment) = stored("image/png", vec![1, 2, 3]).await;
        let answer = answer(
            call(Tool::View, "{\"attachment\":\"notes.txt\"}", &attachment),
            &blobs,
        )
        .await;
        assert!(!answer.is_error);
        assert_eq!(
            answer.content,
            vec![StoredContent::Blob {
                uri: attachment.uri.clone()
            }]
        );
    }

    #[tokio::test]
    async fn view_refuses_what_it_cannot_show() {
        let (blobs, attachment) = stored("text/csv", vec![1, 2, 3]).await;
        let answer = answer(
            call(Tool::View, "{\"attachment\":\"notes.txt\"}", &attachment),
            &blobs,
        )
        .await;
        assert!(answer.is_error);
        assert_eq!(
            answer.as_text(),
            "`notes.txt` is text/csv; attachment_view takes image, audio, and video attachments"
        );
    }

    #[tokio::test]
    async fn an_id_that_is_not_on_the_path_is_refused() {
        let (blobs, attachment) = stored("text/plain", b"hi".to_vec()).await;
        let answer = answer(
            call(Tool::Read, "{\"attachment\":\"other.txt\"}", &attachment),
            &blobs,
        )
        .await;
        assert!(answer.is_error);
        assert!(answer.as_text().contains("no attachment `other.txt`"));
        assert!(answer.as_text().contains("notes.txt"));
    }

    #[test]
    fn the_tools_are_offered_in_name_order() {
        let attachments = Attachments {
            tools: vec![Tool::View, Tool::Read, Tool::View],
            ..Default::default()
        };
        let names: Vec<String> = definitions(&attachments)
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        assert_eq!(names, [READ, VIEW]);
    }
}
