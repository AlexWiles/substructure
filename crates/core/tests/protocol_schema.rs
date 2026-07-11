//! Generates `schemas/protocol.schema.json` from [`substructure_core::protocol`]
//! and fails if the checked-in file has drifted. Running `cargo test` rewrites
//! the file, so a failure means: commit the update.

use std::path::PathBuf;

use schemars::generate::SchemaSettings;
use schemars::transform::RecursiveTransform;
use schemars::Schema;
use substructure_core::protocol as p;

/// Close plain object schemas (`additionalProperties: false`) so typo'd keys
/// fail validation and type-level TS importers get closed objects. A schema
/// mixing `properties` with a combinator (the flattened [`p::Effect`]) stays
/// open, as do its direct variants: closing either would reject the fields
/// the other half validates.
fn close_objects(value: &mut serde_json::Value, closable: bool) {
    match value {
        serde_json::Value::Object(map) => {
            let combined = map.contains_key("properties")
                && ["oneOf", "anyOf", "allOf"]
                    .iter()
                    .any(|k| map.contains_key(*k));
            if closable
                && !combined
                && map.contains_key("properties")
                && !map.contains_key("additionalProperties")
            {
                map.insert(
                    "additionalProperties".into(),
                    serde_json::Value::Bool(false),
                );
            }
            for (key, child) in map.iter_mut() {
                match child {
                    serde_json::Value::Array(items)
                        if combined && matches!(key.as_str(), "oneOf" | "anyOf" | "allOf") =>
                    {
                        for item in items {
                            close_objects(item, false);
                        }
                    }
                    _ => close_objects(child, true),
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                close_objects(item, true);
            }
        }
        _ => {}
    }
}

/// One draft 2020-12 schema with every protocol type under `$defs`.
fn protocol_schema() -> serde_json::Value {
    // Strip `default`: it is runtime serde behavior, not part of the wire
    // contract, and its presence makes type-level TS importers treat optional
    // fields as required.
    let mut generator = SchemaSettings::draft2020_12()
        .with_transform(RecursiveTransform(|s: &mut Schema| {
            s.remove("default");
        }))
        .into_generator();
    macro_rules! register {
        ($($t:ty),* $(,)?) => { $( generator.subschema_for::<$t>(); )* };
    }
    // `#[schemars(inline)]` types (Content, NewMessage, ToolCallFunction, media
    // data, Node, …) are omitted: they inline at each use site rather than
    // landing in `$defs`.
    register!(
        p::Role,
        p::ToolCall,
        p::ContentPart,
        p::Message,
        p::DraftMessage,
        p::Control,
        p::ControlKind,
        p::InterruptOrigin,
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
        p::ClientPayload,
        p::ClientInput,
        p::DecisionTrigger,
        p::DecisionAction,
        p::DecisionResponse,
        p::DecisionRequest<'static>,
    );
    let mut defs = serde_json::Value::Object(generator.take_definitions(true));
    close_objects(&mut defs, true);
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
        "Message",
        "AgentConfig",
        "Effect",
        "TokenDelta",
    ] {
        assert!(defs.contains_key(name), "missing $defs entry: {name}");
    }
    // `#[schemars(inline)]` types must NOT be named: they inline at each use site,
    // keeping ref-chains shallow for type-level TS importers.
    for name in ["Content", "NewMessage", "Node", "ToolCallFunction"] {
        assert!(
            !defs.contains_key(name),
            "expected inlined, found $defs entry: {name}"
        );
    }
}

#[test]
fn plain_objects_are_closed_but_flattened_effect_stays_open() {
    let schema = protocol_schema();
    let defs = &schema["$defs"];
    assert_eq!(defs["DecisionResponse"]["additionalProperties"], false);
    assert_eq!(defs["AgentConfig"]["additionalProperties"], false);
    // Effect flattens EffectDetail: its instances mix top-level and variant
    // fields, so neither half may be closed.
    assert!(defs["Effect"].get("additionalProperties").is_none());
    for variant in defs["Effect"]["oneOf"].as_array().expect("Effect variants") {
        assert!(variant.get("additionalProperties").is_none());
    }
    // Standalone tagged unions have self-contained variants: closed.
    for variant in defs["DecisionTrigger"]["oneOf"]
        .as_array()
        .expect("trigger variants")
    {
        assert_eq!(variant["additionalProperties"], false);
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
    let arguments = &call_tool["properties"]["arguments"];
    // A `Value` field is an unconstrained schema (`true`): any JSON value passes.
    let admits_any = arguments == &serde_json::json!(true)
        || arguments
            .as_object()
            .is_some_and(|o| !o.contains_key("type") && !o.contains_key("$ref"));
    assert!(
        admits_any,
        "arguments admits any JSON value; got {arguments:?}"
    );
}
