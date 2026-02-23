use std::collections::HashMap;

use substructure::domain::agent::{AgentConfig, LlmConfig};
use substructure::domain::config::{EventStoreConfig, LlmClientConfig, SystemConfig};
use substructure::domain::event::{Role, SessionAuth, SpanContext};
use substructure::domain::session::{CommandPayload, SessionCommand};
use substructure::runtime::Runtime;

// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    println!("=== MCP Agentic Loop Demo ===\n");

    // 1. Build system config — agents and LLM clients defined in one place
    let config = SystemConfig {
        event_store: EventStoreConfig::InMemory,
        llm_clients: HashMap::from([("mock".into(), LlmClientConfig {
            client_type: "mock".into(),
            settings: serde_json::Map::new(),
        })]),
        agents: HashMap::from([(
            "weather-agent".into(),
            AgentConfig {
                id: Default::default(),
                name: "weather-agent".into(),
                llm: LlmConfig {
                    client: "mock".into(),
                    params: serde_json::Map::from_iter([
                        ("model".into(), serde_json::Value::String("mock-model".into())),
                    ]),
                },
                system_prompt: "You are a helpful weather assistant. Use the get_weather tool to answer weather questions.".into(),
                mcp_servers: vec![],
                strategy: Default::default(),
            },
        )]),
    };
    let runtime = Runtime::start(&config).await;

    // 2. Create session for a named agent
    let session = runtime
        .create_session_for(
            "weather-agent",
            SessionAuth {
                tenant_id: "tenant_acme".into(),
                client_id: "cli_v1".into(),
                sub: Some("user_abc123".into()),
            },
        )
        .await
        .unwrap();

    // 4. User sends a message — this triggers the full agentic loop:
    //    user msg → LLM (returns tool call) → MCP executes tool → tool result → LLM (returns text) → done
    println!("\n--- User sends: \"What's the weather in San Francisco?\" ---\n");
    let result = session
        .send_command(SessionCommand {
            span: SpanContext::root(),
            occurred_at: chrono::Utc::now(),
            payload: CommandPayload::SendUserMessage {
                content: "What's the weather in San Francisco?".into(),
                stream: true,
            },
        })
        .await;
    println!("send message result: {:?}\n", result.unwrap());

    // 5. Wait for the full agentic loop to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // 6. Query final state
    let state = session.get_state().await;

    println!("\n--- Final Session State ---");
    println!("Session: {}", state.session_id);
    if let Some(agent) = &state.agent {
        println!("Agent: {} (model: {})", agent.name, agent.llm.params.get("model").and_then(|v| v.as_str()).unwrap_or("unknown"));
    }
    println!("Messages ({}):", state.messages.len());
    for msg in &state.messages {
        let tool_info = if !msg.tool_calls.is_empty() {
            format!(
                " [tool_calls: {}]",
                msg.tool_calls
                    .iter()
                    .map(|tc| tc.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else if let Some(ref tcid) = msg.tool_call_id {
            format!(" [tool_call_id: {}]", tcid)
        } else {
            String::new()
        };
        println!(
            "  [{:?}]{} {}",
            msg.role,
            tool_info,
            msg.content.as_deref().unwrap_or("(no content)")
        );
    }
    println!("Tokens used: {}", state.tokens.used);

    // 7. Verification
    println!("\n--- Verification ---");
    println!(
        "Expected message count (4: user, assistant+tool_call, tool, assistant): {}",
        state.messages.len() == 4
    );
    println!(
        "Final message is assistant text: {}",
        state
            .messages
            .last()
            .map(|m| m.role == Role::Assistant && m.content.is_some())
            .unwrap_or(false)
    );

    // Clean shutdown
    session.shutdown().await;
    runtime.shutdown().await;

    println!("\nDone!");
}
