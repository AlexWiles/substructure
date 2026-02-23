use std::sync::Arc;

use async_trait::async_trait;
use ractor::{call_t, Actor};
use rand::RngExt;
use uuid::Uuid;

use substructure::domain::event::{
    AgentConfig, LlmRequest, LlmResponse, Role, SessionAuth, SpanContext,
};
use substructure::runtime::{
    CallToolResult, Content, InMemoryEventStore, LlmClient, McpClient, McpError,
    SessionActor, SessionActorState, SessionMessage, ToolDefinition,
    ClientMessage, SessionClientActor, SessionClientArgs,
    spawn_dispatcher,
};
use substructure::domain::openai;
use substructure::domain::session::SessionCommand;

// ---------------------------------------------------------------------------
// MockLlmClient — first call returns tool calls, second call returns text
// ---------------------------------------------------------------------------

struct MockLlmClient {
    call_count: std::sync::atomic::AtomicU32,
}

impl MockLlmClient {
    fn new() -> Self {
        Self {
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn call(&self, _request: &LlmRequest) -> Result<LlmResponse, String> {
        let n = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if n == 0 {
            // First call: LLM decides to use a tool
            println!("[mock-llm] call #{} -> returning tool call for 'get_weather'", n + 1);
            Ok(LlmResponse::OpenAi(openai::ChatCompletionResponse {
                id: "mock-completion-1".into(),
                model: "mock-model".into(),
                choices: vec![openai::Choice {
                    index: 0,
                    message: openai::ChatMessage {
                        role: openai::Role::Assistant,
                        content: None,
                        tool_calls: Some(vec![openai::ToolCall {
                            id: "call_001".into(),
                            call_type: "function".into(),
                            function: openai::FunctionCall {
                                name: "get_weather".into(),
                                arguments: r#"{"location":"San Francisco"}"#.into(),
                            },
                        }]),
                        tool_call_id: None,
                    },
                    finish_reason: Some("tool_calls".into()),
                }],
                usage: Some(openai::Usage {
                    prompt_tokens: 15,
                    completion_tokens: 20,
                    total_tokens: 35,
                }),
            }))
        } else {
            // Second call: LLM produces final text response
            println!("[mock-llm] call #{} -> returning final text response", n + 1);
            Ok(LlmResponse::OpenAi(openai::ChatCompletionResponse {
                id: "mock-completion-2".into(),
                model: "mock-model".into(),
                choices: vec![openai::Choice {
                    index: 0,
                    message: openai::ChatMessage {
                        role: openai::Role::Assistant,
                        content: Some(
                            "The weather in San Francisco is 72°F and sunny!".into(),
                        ),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: Some(openai::Usage {
                    prompt_tokens: 40,
                    completion_tokens: 15,
                    total_tokens: 55,
                }),
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// MockMcpClient — provides a "get_weather" tool
// ---------------------------------------------------------------------------

struct MockMcpClient {
    tools: Vec<ToolDefinition>,
}

impl MockMcpClient {
    fn new() -> Self {
        Self {
            tools: vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get the current weather for a location".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }),
            }],
        }
    }
}

#[async_trait]
impl McpClient for MockMcpClient {
    fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        println!(
            "[mock-mcp] tool '{}' called with args: {}",
            name, arguments
        );
        match name {
            "get_weather" => {
                let location = arguments
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Ok(CallToolResult {
                    content: vec![Content::Text {
                        text: format!("Weather in {location}: 72°F, sunny, humidity 45%"),
                    }],
                    is_error: false,
                })
            }
            _ => Ok(CallToolResult {
                content: vec![Content::Text {
                    text: format!("Unknown tool: {name}"),
                }],
                is_error: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------

fn span() -> SpanContext {
    SpanContext {
        trace_id: rand::rng().random(),
        span_id: rand::rng().random(),
        parent_span_id: None,
        trace_flags: 1,
        trace_state: None,
    }
}

#[tokio::main]
async fn main() {
    println!("=== MCP Agentic Loop Demo ===\n");

    // 1. Create store and dispatcher
    let store = Arc::new(InMemoryEventStore::new());
    let (dispatcher, dispatcher_handle) = spawn_dispatcher(store.clone()).await;

    // 2. Spawn session actor with MCP clients
    let session_id = Uuid::new_v4();
    let llm_client: Arc<dyn LlmClient> = Arc::new(MockLlmClient::new());
    let mcp_client: Arc<dyn McpClient> = Arc::new(MockMcpClient::new());

    let initial_state = SessionActorState::new(session_id, store.clone(), llm_client)
        .with_mcp_clients(vec![mcp_client]);

    let (actor, session_handle) = Actor::spawn(
        Some(format!("session-{session_id}")),
        SessionActor,
        initial_state,
    )
    .await
    .expect("failed to spawn actor");

    // 3. Spawn a SessionClient connected to the same session
    let (session_client, client_handle) = Actor::spawn(
        None,
        SessionClientActor,
        SessionClientArgs {
            session_id,
            session_actor: actor.clone(),
            store: store.clone(),
        },
    )
    .await
    .expect("failed to spawn session client");

    // 4. Create session via the SessionClient
    let result = call_t!(
        session_client,
        ClientMessage::SendCommand,
        5000,
        span(),
        SessionCommand::CreateSession {
            agent: AgentConfig {
                id: Uuid::new_v4(),
                name: "weather-agent".into(),
                model: "claude-sonnet-4-20250514".into(),
                provider: "anthropic".into(),
                system_prompt: "You are a helpful weather assistant. Use the get_weather tool to answer weather questions.".into(),
                mcp_servers: vec![],
            },
            auth: SessionAuth {
                tenant_id: "tenant_acme".into(),
                client_id: "cli_v1".into(),
                sub: Some("user_abc123".into()),
            },
        }
    )
    .expect("call failed");
    result.unwrap();

    // 5. User sends a message — this triggers the full agentic loop:
    //    user msg → LLM (returns tool call) → MCP executes tool → tool result → LLM (returns text) → done
    println!("\n--- User sends: \"What's the weather in San Francisco?\" ---\n");
    let result = call_t!(
        session_client,
        ClientMessage::SendCommand,
        5000,
        span(),
        SessionCommand::SendUserMessage {
            content: "What's the weather in San Francisco?".into(),
        }
    )
    .expect("call failed");
    println!("send message result: {:?}\n", result.unwrap());

    // 6. Wait for the full agentic loop to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // 7. Query final state
    let client_state =
        call_t!(session_client, ClientMessage::GetState, 5000).expect("call failed");
    let actor_state = call_t!(actor, SessionMessage::GetState, 5000).expect("call failed");

    println!("\n--- Final Session State ---");
    println!("Session: {}", client_state.session_id);
    if let Some(agent) = &client_state.agent {
        println!("Agent: {} (model: {})", agent.name, agent.model);
    }
    println!("Messages ({}):", client_state.messages.len());
    for msg in &client_state.messages {
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
    println!("Tokens used: {}", client_state.tokens.used);
    println!("Tool calls tracked: {}", client_state.tool_calls.len());
    for (id, tc) in &client_state.tool_calls {
        println!("  {} '{}' -> {:?}", id, tc.name, tc.status);
    }

    // 8. Verification
    println!("\n--- Verification ---");
    println!(
        "Messages match: {}",
        client_state.messages.len() == actor_state.messages.len()
    );
    println!(
        "Tokens match: {}",
        client_state.tokens.used == actor_state.tokens.used
    );
    println!(
        "Expected message count (4: user, assistant+tool_call, tool, assistant): {}",
        client_state.messages.len() == 4
    );
    println!(
        "Pending tool results: {} (expected 0)",
        client_state.pending_tool_results
    );
    println!(
        "Final message is assistant text: {}",
        client_state
            .messages
            .last()
            .map(|m| m.role == Role::Assistant && m.content.is_some())
            .unwrap_or(false)
    );

    // Clean shutdown
    session_client.stop(None);
    actor.stop(None);
    dispatcher.stop(None);
    client_handle.await.unwrap();
    session_handle.await.unwrap();
    dispatcher_handle.await.unwrap();

    println!("\nDone!");
}
