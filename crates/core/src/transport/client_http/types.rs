use serde::{Deserialize, Serialize};

use crate::protocol::{ClientInput, InterruptOrigin, Message, MessageTree};
use crate::session::state::{SessionState, SessionStatus};

/// The one client input request body. `input` is the tagged union of everything a client
/// can send, carrying its own addressing (a submit's `agent_id`/`turn_id`; a settle's effect
/// id). `session_id` is the one universal address — minted when absent — and rides the envelope.
#[derive(Debug, Deserialize)]
pub struct ClientInputRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    pub input: ClientInput,
}

#[derive(Debug, Serialize)]
pub struct ClientInputResponse {
    pub session_id: String,
    pub turn_id: String,
}

#[derive(Debug, Deserialize)]
pub struct InterruptSessionRequest {
    #[serde(default)]
    pub interrupt_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct InterruptSessionResponse {
    pub ok: bool,
    pub interrupt_id: String,
}

/// The client's view of a session: enough to rehydrate a UI. `messages` is the
/// head lineage (root→head); `message_tree` carries the full branch structure.
#[derive(Debug, Serialize)]
pub struct ClientSessionResponse {
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    pub status: &'static str,
    pub interrupts: Vec<ClientSessionInterrupt>,
    pub messages: Vec<Message>,
    pub message_tree: MessageTree,
}

#[derive(Debug, Serialize)]
pub struct ClientSessionInterrupt {
    pub interrupt_id: String,
    pub origin: InterruptOrigin,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

pub fn session_response(session_id: String, session: &SessionState) -> ClientSessionResponse {
    let status = match session.projected_status() {
        SessionStatus::Idle => "idle",
        SessionStatus::Done => "done",
        SessionStatus::Interrupted { .. } => "interrupted",
    };
    let message_tree = session.message_tree();
    let messages = message_tree
        .head_id
        .clone()
        .map(|h| message_tree.path_to(&h))
        .unwrap_or_default();
    ClientSessionResponse {
        session_id,
        agent_id: session.agent_id.clone(),
        status,
        interrupts: session
            .open_interrupts
            .iter()
            .map(|i| ClientSessionInterrupt {
                interrupt_id: i.interrupt_id.clone(),
                origin: i.origin,
                reason: i.reason.clone(),
                anchor: i.anchor.clone(),
            })
            .collect(),
        messages,
        message_tree,
    }
}

#[derive(Debug, Deserialize)]
pub struct StreamSessionEventsParams {
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub after_stream_version: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_input_accepts_a_tool_result() {
        let body = r#"{"input":{"type":"tool.result","id":"tc-1","attempt":0,"result":"x"}}"#;
        assert!(serde_json::from_str::<ClientInputRequest>(body).is_ok());
    }

    fn session_with_interrupt(anchor: Option<&str>) -> SessionState {
        let mut s = SessionState::new("s1".to_string());
        s.agent_id = Some("a1".to_string());
        s.status = SessionStatus::Idle;
        s.open_interrupts
            .push(crate::session::state::OpenInterrupt {
                interrupt_id: "int-1".to_string(),
                origin: InterruptOrigin::Frontend,
                reason: "confirmation".to_string(),
                payload: serde_json::Value::Null,
                anchor: anchor.map(str::to_string),
            });
        s
    }

    #[test]
    fn session_response_reports_interrupts_and_head_resolved_status() {
        // A global interrupt parks the (empty) head path.
        let json =
            serde_json::to_value(session_response("s1".into(), &session_with_interrupt(None)))
                .unwrap();
        assert_eq!(json["status"], "interrupted");
        assert_eq!(json["interrupts"][0]["interrupt_id"], "int-1");
        assert_eq!(json["interrupts"][0]["origin"], "frontend");
        assert!(json["interrupts"][0].get("anchor").is_none());
        assert_eq!(json["message_tree"]["nodes"], serde_json::json!([]));
    }

    #[test]
    fn session_response_reports_an_escaped_interrupt_without_parking_the_status() {
        // Anchored off the (empty) head path: listed, but the head is not parked.
        let json = serde_json::to_value(session_response(
            "s1".into(),
            &session_with_interrupt(Some("m-elsewhere")),
        ))
        .unwrap();
        assert_eq!(json["status"], "idle");
        assert_eq!(json["interrupts"][0]["anchor"], "m-elsewhere");
    }

    #[test]
    fn session_response_omits_nothing_when_idle() {
        let mut s = SessionState::new("s1".to_string());
        s.status = SessionStatus::Idle;
        let json = serde_json::to_value(session_response("s1".into(), &s)).unwrap();
        assert_eq!(json["status"], "idle");
        assert_eq!(json["interrupts"], serde_json::json!([]));
        assert!(json.get("agent_id").is_none());
    }

    #[test]
    fn session_response_messages_is_the_head_lineage() {
        use crate::protocol::{Message, NewMessage, Role};
        let node = |id: &str, role: Role, parent: Option<&str>| NewMessage {
            message: Message {
                id: id.to_string(),
                role,
                content: None,
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            parent_id: parent.map(str::to_string),
        };
        let mut s = SessionState::new("s1".to_string());
        s.status = SessionStatus::Idle;
        s.nodes.push(node("u1", Role::User, None));
        s.nodes.push(node("a1", Role::Assistant, Some("u1")));
        s.nodes.push(node("a2", Role::Assistant, Some("u1"))); // abandoned branch
        s.head_id = Some("a1".to_string());
        let json = serde_json::to_value(session_response("s1".into(), &s)).unwrap();
        let ids: Vec<&str> = json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["u1", "a1"]);
    }

    #[test]
    fn client_input_rejects_llm_result() {
        // The client surface answers client tools only; `llm.result` is not a
        // `ClientInput` variant, so it must not deserialize.
        let body =
            r#"{"input":{"type":"llm.result","id":"llm-1","attempt":0,"response":{"model":"m"}}}"#;
        assert!(
            serde_json::from_str::<ClientInputRequest>(body).is_err(),
            "the client surface answers client tools only — llm.result must not deserialize"
        );
    }
}
