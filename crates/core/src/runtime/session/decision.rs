use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::connectors::registry::ConnectionPath;
use crate::protocol::{
    ClientContext, DeferToolsStrategy, DraftMessage, ErrorInfo, Handler, LlmFormat, LlmRequest,
    LlmResponse, RetryOverride, RetryPolicy, StoredResult, ToolResult, Usage,
};
use crate::runtime::retry::RetryTarget;

impl Handler {
    pub fn as_str(self) -> &'static str {
        match self {
            Handler::Server => "server",
            Handler::Worker => "worker",
            Handler::Client => "client",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHandler {
    #[default]
    Worker,
    Client,
    Server,
}

impl ToolHandler {
    pub fn declared(handler: Option<Handler>) -> Self {
        match handler {
            Some(Handler::Client) => Self::Client,
            _ => Self::Worker,
        }
    }

    pub fn retry_target(self) -> RetryTarget {
        match self {
            ToolHandler::Worker => RetryTarget::WorkerTool,
            ToolHandler::Client => RetryTarget::ClientTool,
            ToolHandler::Server => RetryTarget::ConnectorTool,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmHandler {
    #[default]
    Server,
    Worker,
}

impl From<ToolHandler> for Handler {
    fn from(h: ToolHandler) -> Self {
        match h {
            ToolHandler::Worker => Handler::Worker,
            ToolHandler::Client => Handler::Client,
            ToolHandler::Server => Handler::Server,
        }
    }
}

impl From<LlmHandler> for Handler {
    fn from(h: LlmHandler) -> Self {
        match h {
            LlmHandler::Server => Handler::Server,
            LlmHandler::Worker => Handler::Worker,
        }
    }
}

impl TryFrom<Handler> for ToolHandler {
    type Error = Handler;

    fn try_from(h: Handler) -> Result<Self, Handler> {
        match h {
            Handler::Worker => Ok(ToolHandler::Worker),
            Handler::Client => Ok(ToolHandler::Client),
            Handler::Server => Ok(ToolHandler::Server),
        }
    }
}

impl TryFrom<Handler> for LlmHandler {
    type Error = Handler;

    fn try_from(h: Handler) -> Result<Self, Handler> {
        match h {
            Handler::Server => Ok(LlmHandler::Server),
            Handler::Worker => Ok(LlmHandler::Worker),
            Handler::Client => Err(h),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Trigger {
    #[serde(rename = "session.start")]
    SessionStart,
    #[serde(rename = "client.message")]
    ClientMessage {
        messages: Vec<DraftMessage>,
        #[serde(default)]
        client: ClientContext,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
    },
    #[serde(rename = "client.messages")]
    ClientTranscript {
        messages: Vec<DraftMessage>,
        #[serde(default)]
        new_from: usize,
        #[serde(default)]
        client: ClientContext,
    },
    #[serde(rename = "client.action")]
    ClientAction {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<serde_json::Value>,
    },
    #[serde(rename = "tool.execute")]
    ToolExecute {
        id: String,
        name: String,
        arguments: String,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
    },
    #[serde(rename = "llm.execute")]
    LlmExecute {
        id: String,
        request: LlmRequest,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<LlmFormat>,
        #[serde(default)]
        defer_tools_strategy: DeferToolsStrategy,
        #[serde(default)]
        stream: bool,
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline: Option<DateTime<Utc>>,
    },
    #[serde(rename = "tool.finished")]
    ToolFinished {
        id: String,
        ok: bool,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<StoredResult>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<ErrorInfo>,
    },
    #[serde(rename = "sub_agent.finished")]
    SubAgentFinished {
        id: String,
        ok: bool,
        session_id: String,
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<ErrorInfo>,
    },
    #[serde(rename = "llm.finished")]
    LlmFinished {
        id: String,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<DraftMessage>,
        #[serde(default)]
        truncated: bool,
        #[serde(default)]
        refused: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost: Option<Decimal>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<ErrorInfo>,
    },
    #[serde(rename = "interrupt.resumed")]
    InterruptResumed {
        interrupt_id: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    #[serde(rename = "turn.finished")]
    TurnFinished {
        turn_id: String,
        #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
        data: serde_json::Value,
        #[serde(default)]
        cost: Decimal,
        #[serde(default)]
        usage: Usage,
    },
}

impl Trigger {
    pub fn deferred_turn_id(&self) -> Option<&str> {
        match self {
            Trigger::ClientMessage { turn_id, .. } => turn_id.as_deref(),
            _ => None,
        }
    }

    pub fn llm_ok(
        id: String,
        message: DraftMessage,
        truncated: bool,
        refused: bool,
        usage: Option<Usage>,
        cost: Option<Decimal>,
    ) -> Self {
        Trigger::LlmFinished {
            id,
            ok: true,
            message: Some(message),
            truncated,
            refused,
            usage,
            cost,
            error: None,
        }
    }

    pub fn llm_err(id: String, error: ErrorInfo) -> Self {
        Trigger::LlmFinished {
            id,
            ok: false,
            message: None,
            truncated: false,
            refused: false,
            usage: None,
            cost: None,
            error: Some(error),
        }
    }

    pub fn turn_finished(
        turn_id: String,
        data: serde_json::Value,
        cost: Decimal,
        usage: Usage,
    ) -> Self {
        Trigger::TurnFinished {
            turn_id,
            data,
            cost,
            usage,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkKind {
    ToolCall,
    LlmCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectResultPayload {
    ToolCall { result: ToolResult },
    LlmCall { response: Box<LlmResponse> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Action {
    #[serde(rename = "llm.call")]
    CallLlm {
        id: String,
        llm: String,
        request: LlmRequest,
        #[serde(default)]
        stream: bool,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
        handler: LlmHandler,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<LlmFormat>,
    },
    #[serde(rename = "tool.call")]
    CallTool {
        id: String,
        name: String,
        arguments: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryOverride>,
    },
    #[serde(rename = "tool.result")]
    ToolResult {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        result: StoredResult,
    },
    #[serde(rename = "llm.result")]
    LlmResult {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        response: LlmResponse,
    },
    #[serde(rename = "tool.error")]
    ToolError {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: ErrorInfo,
        retryable: bool,
    },
    #[serde(rename = "llm.error")]
    LlmError {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attempt: Option<u32>,
        error: ErrorInfo,
        retryable: bool,
    },
    #[serde(rename = "sub_agent.spawn")]
    SpawnSubAgent {
        session_id: String,
        agent_id: String,
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<DraftMessage>,
        #[serde(default = "RetryPolicy::no_retry")]
        retry: RetryPolicy,
    },
    #[serde(rename = "message.send")]
    SendMessage {
        session_id: String,
        message: DraftMessage,
    },
    #[serde(rename = "interrupt")]
    Interrupt {
        interrupt_id: String,
        reason: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    #[serde(rename = "interrupt.resolve")]
    ResolveInterrupt {
        interrupt_id: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
    #[serde(rename = "connector.sync")]
    SyncConnector { path: ConnectionPath },
    #[serde(rename = "done")]
    Done {
        #[serde(default)]
        data: serde_json::Value,
    },
}
