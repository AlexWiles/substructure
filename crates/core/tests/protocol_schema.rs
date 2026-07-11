//! Generates `schemas/protocol.schema.json` from [`substructure_core::protocol`]
//! and fails if the checked-in file has drifted. Running `cargo test` rewrites
//! the file, so a failure means: commit the update.

use std::path::PathBuf;

use schemars::generate::SchemaSettings;
use substructure_core::protocol as p;

/// One draft 2020-12 schema with every protocol type under `$defs`.
fn protocol_schema() -> serde_json::Value {
    let mut generator = SchemaSettings::draft2020_12().into_generator();
    macro_rules! register {
        ($($t:ty),* $(,)?) => { $( generator.subschema_for::<$t>(); )* };
    }
    register!(
        p::Role,
        p::ToolCallFunction,
        p::ToolCall,
        p::ImageUrl,
        p::FileData,
        p::AudioData,
        p::VideoUrl,
        p::ContentPart,
        p::Content,
        p::Message,
        p::DraftMessage,
        p::NewMessage,
        p::NewControl,
        p::Control,
        p::ControlKind,
        p::InterruptOrigin,
        p::Node,
        p::MessageTree,
        p::ToolHandler,
        p::LlmHandler,
        p::RetryPolicy,
        p::SessionOwner,
        p::WorkerState,
        p::AgentConfig,
        p::AgentTool,
        p::SubAgent,
        p::LlmTool,
        p::LlmRequest,
        p::ReasoningConfig,
        p::ReasoningEffort,
        p::ResponseImage,
        p::LlmResponse,
        p::ErrorCode,
        p::ToolCallChunk,
        p::StreamDelta,
        p::TokenDelta,
        p::EffectStatus,
        p::Effect,
        p::EffectDetail,
        p::ToolInput,
        p::ClientMessage,
        p::ClientMessages,
        p::ClientAction,
        p::ClientPayload,
        p::ClientInput,
        p::DecisionTrigger,
        p::DecisionAction,
        p::DecisionProposal,
        p::DecisionResponse,
        p::DecisionRequest<'static>,
    );
    let defs = generator.take_definitions(true);
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Substructure protocol",
        "$defs": defs,
    })
}

fn schema_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schemas/protocol.schema.json")
}

#[test]
fn generated_schema_is_committed() {
    let generated = format!(
        "{}\n",
        serde_json::to_string_pretty(&protocol_schema()).expect("schema serializes")
    );
    let path = schema_path();
    let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
    if on_disk != generated {
        std::fs::create_dir_all(path.parent().unwrap()).expect("schemas dir");
        std::fs::write(&path, &generated).expect("write schema");
        panic!("schemas/protocol.schema.json was stale and has been regenerated — commit it");
    }
}

#[test]
fn every_protocol_type_lands_in_defs() {
    let schema = protocol_schema();
    let defs = schema["$defs"].as_object().expect("$defs object");
    for name in [
        "ClientInput",
        "DecisionRequest",
        "DecisionResponse",
        "DecisionTrigger",
        "DecisionAction",
        "DecisionProposal",
        "Message",
        "AgentConfig",
        "Effect",
        "TokenDelta",
    ] {
        assert!(defs.contains_key(name), "missing $defs entry: {name}");
    }
}

#[test]
fn lenient_fields_accept_any_json() {
    let schema = protocol_schema();
    let call_tool = schema["$defs"]["DecisionAction"]["oneOf"]
        .as_array()
        .expect("DecisionAction variants")
        .iter()
        .find(|v| v["properties"]["type"]["const"] == "tool.call")
        .expect("tool.call variant")
        .clone();
    let arguments = call_tool["properties"]["arguments"]
        .as_object()
        .expect("arguments schema");
    assert!(
        !arguments.contains_key("type") && !arguments.contains_key("$ref"),
        "arguments admits any JSON value; got {arguments:?}"
    );
}
