use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use schemars::generate::SchemaSettings;
use schemars::transform::RecursiveTransform;
use schemars::Schema;
use substructure_core::protocol as p;

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

fn protocol_schema() -> serde_json::Value {
    let mut generator = SchemaSettings::draft2020_12()
        .with_transform(RecursiveTransform(|s: &mut Schema| {
            s.ensure_object();
            s.remove("default");
        }))
        .into_generator();
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
        p::InterruptOrigin,
        p::InterruptPayload,
        p::InterruptOption,
        p::InterruptResolution,
        p::ResumeStatus,
        p::InterruptResponder,
        p::MessageTree,
        p::Handler,
        p::RetryPolicy,
        p::WorkerIdentity,
        p::WorkerState,
        p::AgentConfig,
        p::AgentTool,
        p::Subagent,
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
        p::EffectKind,
        p::Effect,
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
        "description": "One entry point per wire surface; every protocol type is named under $defs.",
        "type": "object",
        "properties": {
            "decision_request": { "$ref": "#/$defs/DecisionRequest" },
            "decision_response": { "$ref": "#/$defs/DecisionResponse" },
            "client_input": { "$ref": "#/$defs/ClientInput" },
            "client_payload": { "$ref": "#/$defs/ClientPayload" },
            "token_delta": { "$ref": "#/$defs/TokenDelta" },
            "stream_delta": { "$ref": "#/$defs/StreamDelta" },
            "interrupt_payload": { "$ref": "#/$defs/InterruptPayload" },
            "interrupt_resolution": { "$ref": "#/$defs/InterruptResolution" },
        },
        "$defs": defs,
    })
}

fn collect_refs(value: &serde_json::Value, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(r)) = map.get("$ref") {
                if let Some(name) = r.strip_prefix("#/$defs/") {
                    out.insert(name.to_string());
                }
            }
            for child in map.values() {
                collect_refs(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_refs(item, out);
            }
        }
        _ => {}
    }
}

fn rewrite_refs(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(r)) = map.get_mut("$ref") {
                if let Some(name) = r.strip_prefix("#/$defs/") {
                    *r = format!("#/components/schemas/{name}");
                }
            }
            for child in map.values_mut() {
                rewrite_refs(child);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                rewrite_refs(item);
            }
        }
        _ => {}
    }
}

fn worker_openapi() -> serde_json::Value {
    let schema = protocol_schema();
    let defs = schema["$defs"].as_object().expect("$defs object");

    let mut reachable = BTreeSet::from([
        "DecisionRequest".to_string(),
        "DecisionResponse".to_string(),
    ]);
    loop {
        let mut next = reachable.clone();
        for name in &reachable {
            collect_refs(&defs[name.as_str()], &mut next);
        }
        if next == reachable {
            break;
        }
        reachable = next;
    }

    let mut schemas: BTreeMap<&str, serde_json::Value> = reachable
        .iter()
        .map(|name| (name.as_str(), defs[name.as_str()].clone()))
        .collect();
    for schema in schemas.values_mut() {
        rewrite_refs(schema);
    }

    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Substructure worker",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "The decision protocol, from the worker's side: the engine POSTs a decision request describing what just happened; the worker replies with a decision — the actions to take and the conversation as it should now read. One request, one response, once per decision.",
            "license": { "name": "MIT", "url": "https://opensource.org/license/mit" },
        },
        "servers": [
            {
                "url": "http://localhost:4444",
                "description": "A local worker; the engine targets whatever `--worker-url` names.",
            },
        ],
        "paths": {
            "/": {
                "post": {
                    "operationId": "decide",
                    "summary": "Decide the next step for a session",
                    "description": "Switch on `trigger.type`. Echoing `proposed` (amended or verbatim) accepts the engine's default continuation; it is empty when the engine needs worker knowledge. An empty decision with nothing in flight parks the session.",
                    "security": [{}, { "signature": [] }],
                    "parameters": [
                        {
                            "name": "traceparent",
                            "in": "header",
                            "required": false,
                            "description": "W3C trace context.",
                            "schema": { "type": "string" },
                        },
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/DecisionRequest" },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "The worker's decision. A `null` body is accepted as the empty decision.",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/DecisionResponse" },
                                },
                            },
                        },
                        "4XX": {
                            "description": "The delivery attempt fails; the engine retries per its worker retry policy.",
                        },
                    },
                },
            },
        },
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "signature": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-Substructure-Signature",
                    "description": "`sha256=<hex>` HMAC of the raw body; sent when the app has a signing secret.",
                },
            },
        },
    })
}

fn assert_committed(rel: &str, doc: &serde_json::Value) {
    let generated = format!(
        "{}\n",
        serde_json::to_string_pretty(doc).expect("doc serializes")
    );
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas")
        .join(rel);
    let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
    if on_disk != generated {
        std::fs::create_dir_all(path.parent().unwrap()).expect("schemas dir");
        std::fs::write(&path, &generated).expect("write doc");
        panic!("schemas/{rel} was stale and has been regenerated — commit it");
    }
}

#[test]
fn generated_schema_is_committed() {
    assert_committed("protocol.schema.json", &protocol_schema());
}

#[test]
fn generated_openapi_is_committed() {
    assert_committed("worker.openapi.json", &worker_openapi());
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
        "EffectKind",
        "Handler",
        "TokenDelta",
        "Content",
        "MessageTree",
        "ToolCallFunction",
    ] {
        assert!(defs.contains_key(name), "missing $defs entry: {name}");
    }
    for name in ["NewMessage", "ClientMessage", "ClientAction"] {
        assert!(
            !defs.contains_key(name),
            "expected inlined, found $defs entry: {name}"
        );
    }
    for name in [
        "ToolHandler",
        "LlmHandler",
        "EffectDetail",
        "Control",
        "ControlKind",
        "Node",
    ] {
        assert!(!defs.contains_key(name), "retired $defs entry: {name}");
    }
}

#[test]
fn schema_root_names_every_wire_surface() {
    let schema = protocol_schema();
    let roots = schema["properties"].as_object().expect("root properties");
    for (name, def) in [
        ("decision_request", "DecisionRequest"),
        ("decision_response", "DecisionResponse"),
        ("client_input", "ClientInput"),
        ("client_payload", "ClientPayload"),
        ("token_delta", "TokenDelta"),
        ("stream_delta", "StreamDelta"),
    ] {
        assert_eq!(
            roots[name]["$ref"],
            format!("#/$defs/{def}"),
            "root {name} must point at {def}"
        );
    }
}

#[test]
fn plain_objects_are_closed_including_flat_effect() {
    let schema = protocol_schema();
    let defs = &schema["$defs"];
    assert_eq!(defs["DecisionResponse"]["additionalProperties"], false);
    assert_eq!(defs["AgentConfig"]["additionalProperties"], false);
    assert_eq!(defs["Effect"]["additionalProperties"], false);
    assert!(defs["Effect"].get("oneOf").is_none());
    for variant in defs["DecisionTrigger"]["oneOf"]
        .as_array()
        .expect("trigger variants")
    {
        assert_eq!(variant["additionalProperties"], false);
    }
}

#[test]
fn no_boolean_schemas_remain() {
    fn walk(v: &serde_json::Value, path: &str) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, child) in map {
                    if k == "properties" {
                        for (name, prop) in child.as_object().into_iter().flatten() {
                            assert!(
                                !prop.is_boolean(),
                                "boolean schema at {path}/properties/{name}"
                            );
                        }
                    }
                    walk(child, &format!("{path}/{k}"));
                }
            }
            serde_json::Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    walk(item, &format!("{path}/{i}"));
                }
            }
            _ => {}
        }
    }
    let schema = protocol_schema();
    walk(&schema, "");
}

#[test]
fn cost_is_a_decimal_string() {
    let schema = protocol_schema();
    let cost = &schema["$defs"]["LlmResponse"]["properties"]["cost"];
    assert_eq!(cost["type"], serde_json::json!(["string", "null"]));
}

#[test]
fn handler_is_one_wire_enum() {
    let schema = protocol_schema();
    let values: Vec<_> = schema["$defs"]["Handler"]["oneOf"]
        .as_array()
        .expect("Handler variants")
        .iter()
        .map(|v| v["const"].clone())
        .collect();
    assert_eq!(values, ["server", "worker", "client"]);
}

#[test]
fn openapi_describes_one_decide_call() {
    let doc = worker_openapi();
    let post = &doc["paths"]["/"]["post"];
    assert_eq!(post["operationId"], "decide");
    assert_eq!(
        post["requestBody"]["content"]["application/json"]["schema"]["$ref"],
        "#/components/schemas/DecisionRequest"
    );
    assert_eq!(
        post["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
        "#/components/schemas/DecisionResponse"
    );
}

#[test]
fn openapi_components_are_the_worker_closure() {
    let doc = worker_openapi();
    let schemas = doc["components"]["schemas"]
        .as_object()
        .expect("components.schemas");
    for name in [
        "DecisionRequest",
        "DecisionResponse",
        "DecisionTrigger",
        "DecisionAction",
        "Effect",
        "Handler",
        "AgentConfig",
        "Message",
        "Content",
    ] {
        assert!(schemas.contains_key(name), "missing component: {name}");
    }
    for name in ["ClientInput", "ClientPayload", "TokenDelta", "StreamDelta"] {
        assert!(!schemas.contains_key(name), "unexpected component: {name}");
    }
}

#[test]
fn openapi_refs_all_resolve_within_components() {
    fn walk(v: &serde_json::Value, schemas: &serde_json::Map<String, serde_json::Value>) {
        match v {
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(r)) = map.get("$ref") {
                    let name = r
                        .strip_prefix("#/components/schemas/")
                        .unwrap_or_else(|| panic!("ref outside components: {r}"));
                    assert!(schemas.contains_key(name), "dangling ref: {r}");
                }
                for child in map.values() {
                    walk(child, schemas);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, schemas);
                }
            }
            _ => {}
        }
    }
    let doc = worker_openapi();
    let schemas = doc["components"]["schemas"]
        .as_object()
        .expect("components.schemas");
    walk(&doc, schemas);
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
    let admits_any = arguments == &serde_json::json!(true)
        || arguments
            .as_object()
            .is_some_and(|o| !o.contains_key("type") && !o.contains_key("$ref"));
    assert!(
        admits_any,
        "arguments admits any JSON value; got {arguments:?}"
    );
}
