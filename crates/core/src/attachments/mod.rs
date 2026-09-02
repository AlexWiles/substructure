pub mod tools;

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::mime;
use crate::protocol::{Content, DraftMessage, MessageTree, StoredContent};
use crate::runtime::blob::BlobRef;

pub const READ: &str = "attachment_read";
pub const VIEW: &str = "attachment_view";
const PREFIX: &str = "attachment_";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "Attachment")]
pub struct Attachment {
    pub id: String,
    pub mime: String,
    pub size: u64,
    pub uri: String,
}

impl Attachment {
    pub fn line(&self) -> String {
        format!(
            "[attachment {} {} {}]",
            self.id,
            self.mime,
            crate::size::text(self.size)
        )
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
#[schemars(rename = "AttachmentTool", title = "AttachmentTool")]
pub enum Tool {
    Read,
    View,
}

impl Tool {
    pub fn name(self) -> &'static str {
        match self {
            Self::Read => READ,
            Self::View => VIEW,
        }
    }

    pub fn of_name(name: &str) -> Option<Self> {
        match name.strip_prefix(PREFIX)? {
            "read" => Some(Self::Read),
            "view" => Some(Self::View),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(title = "Disposition")]
pub enum Disposition {
    Inline,
    Attachment,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
#[schemars(title = "Attachments")]
pub struct Attachments {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::size::de"
    )]
    #[schemars(with = "Option<crate::size::Wire>")]
    pub max_inline: Option<u64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub rules: BTreeMap<String, Disposition>,
}

impl Attachments {
    fn disposition(&self, mime: &str, size: u64) -> Option<Disposition> {
        let matched = *self
            .rules
            .get(mime::base(mime))
            .or_else(|| self.rules.get(&format!("{}/*", mime::parts(mime).0)))
            .or_else(|| self.rules.get("*/*"))?;
        match self.max_inline {
            Some(cap) if matched == Disposition::Inline && size > cap => {
                Some(Disposition::Attachment)
            }
            _ => Some(matched),
        }
    }
}

pub fn mark(messages: &mut [DraftMessage], rules: &Attachments, tree: &MessageTree) {
    let known = on_path(tree, None);
    let mut taken: BTreeSet<String> = known.iter().map(|a| a.id.clone()).collect();
    let mut seen: BTreeSet<String> = known.into_iter().map(|a| a.uri).collect();
    for message in messages {
        let Some(Content::Parts(parts)) = &mut message.content else {
            continue;
        };
        seen.extend(parts.iter().filter_map(|part| match part {
            StoredContent::Attachment(a) => Some(a.uri.clone()),
            _ => None,
        }));
        let mut marked = Vec::with_capacity(parts.len());
        for part in parts.drain(..) {
            let fresh = !matches!(&part, StoredContent::Blob { uri } if seen.contains(uri));
            match fresh.then(|| ruled(&part, rules, &taken)).flatten() {
                None => marked.push(part),
                Some((disposition, attachment)) => {
                    if disposition == Disposition::Inline {
                        marked.push(part);
                    }
                    taken.insert(attachment.id.clone());
                    seen.insert(attachment.uri.clone());
                    marked.push(StoredContent::Attachment(attachment));
                }
            }
        }
        *parts = marked;
    }
}

pub fn on_path(tree: &MessageTree, node: Option<&str>) -> Vec<Attachment> {
    let Some(leaf) = node.or(tree.head_id.as_deref()) else {
        return Vec::new();
    };
    tree.path_to(leaf)
        .iter()
        .filter_map(|message| match &message.content {
            Some(Content::Parts(parts)) => Some(parts),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            StoredContent::Attachment(attachment) => Some(attachment.clone()),
            _ => None,
        })
        .collect()
}

fn ruled(
    part: &StoredContent,
    rules: &Attachments,
    taken: &BTreeSet<String>,
) -> Option<(Disposition, Attachment)> {
    let StoredContent::Blob { uri } = part else {
        return None;
    };
    let stored = BlobRef::parse(uri)?;
    let disposition = rules.disposition(&stored.mime, stored.size)?;
    let attachment = Attachment {
        id: mint(stored.name.as_deref(), &stored.mime, taken),
        mime: mime::base(&stored.mime).to_string(),
        size: stored.size,
        uri: uri.clone(),
    };
    Some((disposition, attachment))
}

fn mint(name: Option<&str>, mime: &str, taken: &BTreeSet<String>) -> String {
    let named = name.filter(|name| !name.is_empty());
    let (stem, extension, first) = match named {
        Some(name) => {
            let (stem, extension) = split(name);
            (stem, extension, 2)
        }
        None => (mime::essence(mime).to_string(), extension_of(mime), 1),
    };
    named
        .map(str::to_string)
        .into_iter()
        .chain((first..).map(|n| format!("{stem}-{n}{extension}")))
        .find(|id| !taken.contains(id))
        .expect("the counter never runs out")
}

fn split(name: &str) -> (String, String) {
    match name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() => (stem.to_string(), format!(".{extension}")),
        _ => (name.to_string(), String::new()),
    }
}

fn extension_of(mime: &str) -> String {
    match mime::parts(mime).1.split('+').next().unwrap_or_default() {
        "" => String::new(),
        subtype => format!(".{subtype}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Message, NewMessage, Role};

    fn rules(toml: &str) -> Attachments {
        toml::from_str(toml).expect("parses")
    }

    fn blob(mime: &str, name: Option<&str>, size: u64) -> StoredContent {
        StoredContent::Blob {
            uri: BlobRef {
                tenant_id: "t1".to_string(),
                id: "0198b2a0-3c5d-7f00-8000-0123456789ab".to_string(),
                mime: mime.to_string(),
                name: name.map(str::to_string),
                size,
            }
            .uri(),
        }
    }

    fn message(parts: Vec<StoredContent>) -> DraftMessage {
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

    fn parts(message: &DraftMessage) -> &[StoredContent] {
        match &message.content {
            Some(Content::Parts(parts)) => parts,
            _ => panic!("parts"),
        }
    }

    #[test]
    fn the_line_names_the_file_its_type_and_its_size() {
        let attachment = Attachment {
            id: "sales.csv".to_string(),
            mime: "text/csv".to_string(),
            size: 2_202_009,
            uri: "blob://t1/x".to_string(),
        };
        assert_eq!(attachment.line(), "[attachment sales.csv text/csv 2.1 MB]");
    }

    #[test]
    fn the_most_specific_pattern_wins() {
        let rules = rules(
            r#"
[rules]
"application/pdf" = "inline"
"image/*" = "inline"
"*/*" = "attachment"
"#,
        );
        assert_eq!(
            rules.disposition("application/pdf", 1),
            Some(Disposition::Inline)
        );
        assert_eq!(rules.disposition("image/png", 1), Some(Disposition::Inline));
        assert_eq!(
            rules.disposition("text/csv", 1),
            Some(Disposition::Attachment)
        );
        assert_eq!(
            rules.disposition("text/csv; charset=utf-8", 1),
            Some(Disposition::Attachment)
        );
        assert_eq!(Attachments::default().disposition("image/png", 1), None);
    }

    #[test]
    fn a_file_over_max_inline_becomes_an_attachment() {
        let rules = rules(
            r#"
max_inline = "1kb"
[rules]
"image/*" = "inline"
"#,
        );
        assert_eq!(
            rules.disposition("image/png", 1024),
            Some(Disposition::Inline)
        );
        assert_eq!(
            rules.disposition("image/png", 1025),
            Some(Disposition::Attachment)
        );
    }

    #[test]
    fn a_section_names_only_the_tools_that_exist() {
        let rules = rules(r#"tools = ["view", "read"]"#);
        assert_eq!(rules.tools, vec![Tool::View, Tool::Read]);
        let err = toml::from_str::<Attachments>(r#"tools = ["grep"]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("grep"), "{err}");
    }

    #[test]
    fn max_inline_takes_bytes_or_a_size_word() {
        assert_eq!(rules(r#"max_inline = "20mb""#).max_inline, Some(20 << 20));
        assert_eq!(rules("max_inline = 512").max_inline, Some(512));
        let wire: Attachments = serde_json::from_value(
            serde_json::json!({ "max_inline": "2kb", "rules": { "*/*": "inline" } }),
        )
        .expect("parses");
        assert_eq!(wire.max_inline, Some(2048));
    }

    #[test]
    fn an_inline_file_keeps_its_bytes_and_gains_a_line() {
        let mut messages = vec![message(vec![
            StoredContent::Text {
                text: "look".to_string(),
            },
            blob("image/png", Some("shot.png"), 12),
        ])];
        mark(
            &mut messages,
            &rules(r#"rules = { "image/*" = "inline" }"#),
            &MessageTree::default(),
        );
        let parts = parts(&messages[0]);
        assert!(matches!(parts[1], StoredContent::Blob { .. }));
        let StoredContent::Attachment(attachment) = &parts[2] else {
            panic!("a line after the bytes, got {:?}", parts[2]);
        };
        assert_eq!(attachment.line(), "[attachment shot.png image/png 12 B]");
    }

    #[test]
    fn an_attachment_file_keeps_only_its_line() {
        let mut messages = vec![message(vec![blob("text/csv", Some("sales.csv"), 12)])];
        mark(
            &mut messages,
            &rules(r#"rules = { "*/*" = "attachment" }"#),
            &MessageTree::default(),
        );
        let parts = parts(&messages[0]);
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], StoredContent::Attachment(_)));
    }

    #[test]
    fn a_file_no_rule_matches_is_left_as_it_was() {
        let mut messages = vec![message(vec![blob("text/csv", Some("sales.csv"), 12)])];
        mark(
            &mut messages,
            &rules(r#"rules = { "image/*" = "inline" }"#),
            &MessageTree::default(),
        );
        assert_eq!(parts(&messages[0]).len(), 1);
        assert!(matches!(parts(&messages[0])[0], StoredContent::Blob { .. }));
    }

    #[test]
    fn a_blob_already_marked_is_not_marked_again() {
        let mut messages = vec![message(vec![blob("image/png", Some("shot.png"), 12)])];
        let rules = rules(r#"rules = { "image/*" = "inline" }"#);
        mark(&mut messages, &rules, &MessageTree::default());
        mark(&mut messages, &rules, &MessageTree::default());
        assert_eq!(
            parts(&messages[0]).len(),
            2,
            "one blob, one line, however often it is seen"
        );
    }

    #[test]
    fn ids_are_unique_along_the_conversation() {
        let mut messages = vec![message(vec![
            blob("text/csv", Some("sales.csv"), 1),
            blob("text/csv", Some("sales.csv"), 2),
            blob("image/jpg", None, 3),
            blob("image/jpg", None, 4),
        ])];
        let tree = MessageTree {
            head_id: Some("m1".to_string()),
            nodes: vec![NewMessage {
                parent_id: None,
                message: Message {
                    id: "m1".to_string(),
                    role: Role::User,
                    content: Some(Content::Parts(vec![StoredContent::Attachment(
                        Attachment {
                            id: "sales-2.csv".to_string(),
                            mime: "text/csv".to_string(),
                            size: 1,
                            uri: "blob://t1/old".to_string(),
                        },
                    )])),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    name: None,
                    reasoning: None,
                },
            }],
        };
        mark(
            &mut messages,
            &rules(r#"rules = { "*/*" = "attachment" }"#),
            &tree,
        );
        let ids: Vec<&str> = parts(&messages[0])
            .iter()
            .filter_map(|part| match part {
                StoredContent::Attachment(attachment) => Some(attachment.id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            ids,
            ["sales.csv", "sales-3.csv", "image-1.jpg", "image-2.jpg"]
        );
    }
}
