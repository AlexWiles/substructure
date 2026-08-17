//! Connections the engine holds and executes tools against, on the agent's
//! behalf. The agent config names a connection by id and never sees a URL or a
//! credential; the protocol below that id is an implementation detail, so a
//! non-MCP source is additive.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod credential;
pub mod filter;
pub mod mcp;
pub mod oauth;
pub mod registry;

use crate::protocol::{Audience, OwnerKind, SessionOwner};

/// Whose credential slot a call reads: the deployment's one credential, or
/// one person's.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Subject {
    Shared,
    Person(String),
}

impl std::fmt::Display for Subject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Subject::Shared => f.write_str("shared"),
            Subject::Person(id) => write!(f, "subject `{id}`"),
        }
    }
}

/// Who a call runs for, resolved from the session owner at execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Principal {
    /// No person: a schedule, an API key, or the system.
    Machine,
    /// One person, and the audience of the conversation they asked in.
    Person { subject: String, audience: Audience },
}

impl Principal {
    /// The deployment's owner→subject map. A single-deployment engine maps
    /// the surface-native owner id to itself; the cloud resolves through its
    /// identities instead.
    pub fn of_owner(owner: Option<&SessionOwner>) -> Self {
        match owner {
            Some(o) if o.kind == OwnerKind::Frontend => match &o.id {
                Some(id) => Principal::Person {
                    subject: id.clone(),
                    audience: o.audience,
                },
                None => Principal::Machine,
            },
            _ => Principal::Machine,
        }
    }
}

/// A tool as a connection offers it, before filtering or prefixing. The neutral
/// shape every protocol lowers to.
///
/// Stored verbatim on `connector.sync.completed`, unfiltered: filtering is pure,
/// so recording the offer lets a filter change re-derive the model's tool list
/// without another round trip, and lets the log answer what was filtered *out*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(default)]
    pub annotations: ToolAnnotations,
}

/// Behavioural hints a connection attaches to a tool. Every field is optional
/// because a server need not annotate anything, and absent is not false — a
/// capability filter treats an unannotated tool as failing, so a bare server
/// yields nothing under `read_only` instead of passing everything through.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    pub read_only: Option<bool>,
    pub destructive: Option<bool>,
    pub idempotent: Option<bool>,
    pub open_world: Option<bool>,
}

/// The credential headers for one connection, read for each request. A
/// credential replaced in the store thus reaches the next request, and the
/// connector session stays.
#[async_trait::async_trait]
pub trait CredentialSource: Send + Sync {
    async fn headers(&self) -> Result<reqwest::header::HeaderMap, ConnectorError>;
}

/// What a person must do before a connection operates again. The refresh path
/// corrects everything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthNeed {
    NeverAuthorized,
    Reauthorize,
    /// No consent flow can replace a static token. An operator sets a new one.
    TokenRejected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectorError {
    pub message: String,
    /// Whether another attempt could plausibly succeed. Transport faults and 5xx
    /// are retryable; a rejected credential or an unknown tool is not.
    pub retryable: bool,
    /// Set if the connection refused the credential.
    pub auth: Option<AuthNeed>,
}

impl ConnectorError {
    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            auth: None,
        }
    }

    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
            auth: None,
        }
    }

    pub fn unauthorized(need: AuthNeed, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            auth: Some(need),
        }
    }
}

impl std::fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConnectorError {}
