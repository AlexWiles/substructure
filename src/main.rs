use std::sync::Arc;

use async_trait::async_trait;
use ractor::{call_t, Actor};
use rand::RngExt;
use uuid::Uuid;

use substructure::client::{ClientMessage, SessionClientActor, SessionClientArgs};
use substructure::command::SessionCommand;
use substructure::event::*;
use substructure::openai;
use substructure::reactor::spawn_dispatcher;
use substructure::runtime::{LlmClient, SessionActor, SessionActorState, SessionMessage};
use substructure::store::InMemoryEventStore;

// ---------------------------------------------------------------------------
// MockLlmClient — returns a canned response for demo purposes
// ---------------------------------------------------------------------------

struct MockLlmClient;

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn call(&self, _request: &LlmRequest) -> Result<LlmResponse, String> {
        println!("[mock-llm] received call, returning canned response");
        Ok(LlmResponse::OpenAi(openai::ChatCompletionResponse {
            id: "mock-completion-1".into(),
            model: "mock-model".into(),
            choices: vec![openai::Choice {
                index: 0,
                message: openai::ChatMessage {
                    role: openai::Role::Assistant,
                    content: Some("I can help you with many tasks! (mock response)".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 12,
                total_tokens: 22,
            }),
        }))
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
    // 1. Create store and dispatcher
    let store = Arc::new(InMemoryEventStore::new());
    let (dispatcher, dispatcher_handle) = spawn_dispatcher(store.clone()).await;

    // 2. Spawn session actor
    let session_id = Uuid::new_v4();
    let client: Arc<dyn LlmClient> = Arc::new(MockLlmClient);
    let initial_state = SessionActorState::new(session_id, store.clone(), client);

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
                name: "demo-agent".into(),
                model: "claude-sonnet-4-20250514".into(),
                provider: "anthropic".into(),
                system_prompt: "You are a helpful assistant.".into(),
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

    // 5. User sends a message via SessionClient
    let result = call_t!(
        session_client,
        ClientMessage::SendCommand,
        5000,
        span(),
        SessionCommand::SendUserMessage {
            content: "Hello, what can you do?".into(),
        }
    )
    .expect("call failed");
    println!("send message result: {:?}", result.unwrap());

    // 6. Wait for the reactor event loop to complete the full chain
    // tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

    // 7. Query state from SessionClient — should match SessionActor's state
    let client_state = call_t!(session_client, ClientMessage::GetState, 5000).expect("call failed");
    let actor_state = call_t!(actor, SessionMessage::GetState, 5000).expect("call failed");

    println!("\n--- SessionClient State ---");
    println!("Session: {}", client_state.session_id);
    if let Some(agent) = &client_state.agent {
        println!("Agent: {} (model: {})", agent.name, agent.model);
    }
    if let Some(auth) = &client_state.auth {
        println!(
            "Tenant: {}, Client: {}, Sub: {:?}",
            auth.tenant_id, auth.client_id, auth.sub
        );
    }
    println!("Messages: {}", client_state.messages.len());
    for msg in &client_state.messages {
        println!(
            "  [{:?}] {}",
            msg.role,
            msg.content.as_deref().unwrap_or("")
        );
    }
    println!("Tokens used: {}", client_state.tokens.used);

    // 8. Verify client and actor states match
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
        "Agent match: {}",
        client_state.agent.as_ref().map(|a| &a.name) == actor_state.agent.as_ref().map(|a| &a.name)
    );

    // Clean shutdown
    session_client.stop(None);
    actor.stop(None);
    dispatcher.stop(None);
    client_handle.await.unwrap();
    session_handle.await.unwrap();
    dispatcher_handle.await.unwrap();
}
