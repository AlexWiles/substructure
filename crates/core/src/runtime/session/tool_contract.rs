//! The tool contract: a tool's declared `input`/`output` JSON Schemas and the
//! engine's enforcement of them, in both directions.
//!
//! Inbound (model → tool), [`classify_arguments`] sorts a call's raw argument
//! string into the tagged [`ToolInput`] delivered on the `tool.execute`
//! trigger: `valid` (parsed, meets the declared schema), `invalid` (parsed,
//! violates it), or `malformed` (not a JSON object at all). Outbound (tool →
//! model), [`output_violation`] checks a settled result against the declared
//! `output` schema so the engine can refuse a result the tool's own contract
//! forbids.
//!
//! The engine validates, it never mutates: a `valid` value is exactly the
//! parsed argument string — no coercion, no default-filling. Real parsing
//! (typed bindings) belongs to the worker.
//!
//! Schemas are resolved by lineage ([`declared_tool`]): the executing call's
//! id appears in one assistant message on the active path, recorded under its
//! llm.call's id; that call's spec names the tool. A call with no resolvable
//! spec (worker-authored, off-path) is [`DeclaredTool::Unknowable`] — the
//! engine can't judge it and skips every check.

use super::state::LlmCallState;
use crate::protocol::{LlmTool, Message, ToolInput};

impl ToolInput {
    /// The error text for a non-valid classification.
    pub fn error(&self) -> Option<&str> {
        match self {
            ToolInput::Valid { .. } => None,
            ToolInput::Invalid { error, .. } | ToolInput::Malformed { error } => Some(error),
        }
    }
}

/// Classify a call's raw argument string against the tool's declared `input`
/// schema (`None` skips the schema check). Empty arguments normalize to `{}`.
pub fn classify_arguments(arguments: &str, schema: Option<&serde_json::Value>) -> ToolInput {
    let value = if arguments.trim().is_empty() {
        serde_json::json!({})
    } else {
        match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(value @ serde_json::Value::Object(_)) => value,
            Ok(value) => {
                return ToolInput::Malformed {
                    error: format!("arguments must be a JSON object, got {}", json_type(&value)),
                }
            }
            Err(e) => {
                return ToolInput::Malformed {
                    error: e.to_string(),
                }
            }
        }
    };
    match schema.and_then(|schema| schema_violations(schema, &value)) {
        Some(error) => ToolInput::Invalid { value, error },
        None => ToolInput::Valid { value },
    }
}

fn json_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// What the originating llm.call declared about the tool a call names.
#[derive(Debug)]
pub enum DeclaredTool<'a> {
    /// The call's spec declares this tool.
    Declared(&'a LlmTool),
    /// The call's spec declares tools, but not this name — a hallucinated call.
    Undeclared,
    /// No spec to judge against: a worker-authored call, an off-path call, or
    /// a request that declared no tools.
    Unknowable,
}

/// Resolve what the originating request declared about tool `name`, by
/// lineage: `tool_call_id` appears in one assistant message on the active
/// path, recorded under its llm.call's id; that call's spec lists the tools
/// the model was offered.
pub fn declared_tool<'a>(
    tool_call_id: &str,
    name: &str,
    active_path: &[Message],
    llm_call: impl Fn(&str) -> Option<&'a LlmCallState>,
) -> DeclaredTool<'a> {
    let assistant = active_path
        .iter()
        .rev()
        .find(|m| m.tool_calls.iter().any(|c| c.id == tool_call_id));
    let tools = assistant
        .and_then(|m| llm_call(&m.id))
        .and_then(|call| call.spec.tools.as_deref());
    match tools {
        None => DeclaredTool::Unknowable,
        Some(tools) => match tools.iter().find(|t| t.name == name) {
            Some(tool) => DeclaredTool::Declared(tool),
            None => DeclaredTool::Undeclared,
        },
    }
}

/// Check a settled result against the tool's declared `output` schema. The
/// result string is interpreted as JSON when it parses, else as a plain
/// string value.
pub fn output_violation(schema: &serde_json::Value, result: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(result)
        .unwrap_or_else(|_| serde_json::Value::String(result.to_string()));
    schema_violations(schema, &value)
}

/// Validate a value against a JSON Schema. An unbuildable schema skips
/// validation rather than failing the call.
fn schema_violations(schema: &serde_json::Value, value: &serde_json::Value) -> Option<String> {
    let validator = jsonschema::validator_for(schema).ok()?;
    let violations: Vec<String> = validator
        .iter_errors(value)
        .take(3)
        .map(|e| {
            let at = e.instance_path().to_string();
            if at.is_empty() {
                e.to_string()
            } else {
                format!("{at}: {e}")
            }
        })
        .collect();
    (!violations.is_empty()).then(|| violations.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{RetryPolicy, Role, ToolCall, ToolCallFunction};
    use crate::runtime::session::decision::LlmHandler;
    use crate::runtime::session::state::{EffectPayload, EffectState, EffectTracking, LlmCallSpec};

    fn city_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        })
    }

    #[test]
    fn classifies_the_three_argument_cases() {
        let schema = city_schema();

        match classify_arguments(r#"{"city":"NYC"}"#, Some(&schema)) {
            ToolInput::Valid { value } => assert_eq!(value, serde_json::json!({"city":"NYC"})),
            other => panic!("expected valid; got {other:?}"),
        }

        match classify_arguments(r#"{"city":5}"#, Some(&schema)) {
            ToolInput::Invalid { value, error } => {
                assert_eq!(
                    value,
                    serde_json::json!({"city":5}),
                    "the object is still delivered"
                );
                assert!(
                    error.contains("city"),
                    "the violation is reported; got {error}"
                );
            }
            other => panic!("expected invalid; got {other:?}"),
        }

        match classify_arguments("not json", Some(&schema)) {
            ToolInput::Malformed { error } => assert!(!error.is_empty()),
            other => panic!("expected malformed; got {other:?}"),
        }
    }

    #[test]
    fn empty_arguments_normalize_to_an_empty_object() {
        match classify_arguments("", None) {
            ToolInput::Valid { value } => assert_eq!(value, serde_json::json!({})),
            other => panic!("expected valid; got {other:?}"),
        }
    }

    #[test]
    fn a_non_object_value_is_malformed() {
        match classify_arguments("[1,2]", None) {
            ToolInput::Malformed { error } => {
                assert_eq!(error, "arguments must be a JSON object, got an array")
            }
            other => panic!("expected malformed; got {other:?}"),
        }
    }

    #[test]
    fn a_missing_required_field_is_invalid() {
        match classify_arguments("{}", Some(&city_schema())) {
            ToolInput::Invalid { error, .. } => assert!(error.contains("city"), "got {error}"),
            other => panic!("expected invalid; got {other:?}"),
        }
    }

    #[test]
    fn no_schema_means_any_object_is_valid() {
        assert!(matches!(
            classify_arguments(r#"{"anything": true}"#, None),
            ToolInput::Valid { .. }
        ));
    }

    #[test]
    fn an_unbuildable_schema_skips_validation() {
        let broken = serde_json::json!({"type": 12});
        assert!(matches!(
            classify_arguments(r#"{"city": 5}"#, Some(&broken)),
            ToolInput::Valid { .. }
        ));
    }

    #[test]
    fn the_input_statuses_serialize_tagged() {
        let v = serde_json::to_value(ToolInput::Valid {
            value: serde_json::json!({}),
        })
        .unwrap();
        assert_eq!(v, serde_json::json!({"status": "valid", "value": {}}));

        let v = serde_json::to_value(ToolInput::Malformed {
            error: "boom".to_string(),
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"status": "malformed", "error": "boom"})
        );
    }

    fn weather_call(
        tools: Option<Vec<LlmTool>>,
    ) -> (Vec<Message>, std::collections::HashMap<String, EffectState>) {
        let assistant = Message {
            id: "call-1".to_string(),
            role: Role::Assistant,
            content: None,
            tool_calls: vec![ToolCall {
                id: "tc-1".to_string(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                },
            }],
            tool_call_id: None,
            name: None,
        };
        let call = EffectState::new(
            "call-1",
            EffectTracking::new(RetryPolicy::no_retry(), chrono::Utc::now()),
            EffectPayload::LlmCall(LlmCallState {
                format: None,
                llm: "claude".to_string(),
                prompt: vec![],
                spec: LlmCallSpec {
                    model: "m".to_string(),
                    tools,
                    temperature: None,
                    max_completion_tokens: None,
                    reasoning: None,
                },
                stream: false,
                handler: LlmHandler::Server,
            }),
        );
        (
            vec![assistant],
            std::collections::HashMap::from([("call-1".to_string(), call)]),
        )
    }

    fn weather_tool() -> LlmTool {
        LlmTool {
            name: "get_weather".to_string(),
            description: "d".to_string(),
            input: Some(city_schema()),
            output: None,
        }
    }

    #[test]
    fn resolves_a_declared_tool_by_lineage() {
        let (path, calls) = weather_call(Some(vec![weather_tool()]));
        assert!(matches!(
            declared_tool("tc-1", "get_weather", &path, |id| calls.get(id).and_then(|e| e.llm())),
            DeclaredTool::Declared(t) if t.name == "get_weather"
        ));
    }

    #[test]
    fn a_name_outside_the_declared_list_is_undeclared() {
        let (path, calls) = weather_call(Some(vec![weather_tool()]));
        assert!(matches!(
            declared_tool("tc-1", "hallucinated", &path, |id| calls
                .get(id)
                .and_then(|e| e.llm())),
            DeclaredTool::Undeclared
        ));
    }

    #[test]
    fn no_resolvable_spec_is_unknowable() {
        let (path, calls) = weather_call(Some(vec![weather_tool()]));
        assert!(
            matches!(
                declared_tool("tc-other", "get_weather", &path, |id| calls
                    .get(id)
                    .and_then(|e| e.llm())),
                DeclaredTool::Unknowable
            ),
            "a call id absent from the path has no lineage"
        );
        let (path, calls) = weather_call(None);
        assert!(
            matches!(
                declared_tool("tc-1", "get_weather", &path, |id| calls
                    .get(id)
                    .and_then(|e| e.llm())),
                DeclaredTool::Unknowable
            ),
            "a request that declared no tools can't judge a name"
        );
    }

    #[test]
    fn output_violations_are_reported() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "temp_c": { "type": "number" } },
            "required": ["temp_c"],
        });
        assert_eq!(output_violation(&schema, r#"{"temp_c": 21}"#), None);
        assert!(output_violation(&schema, r#"{"temp_c": "warm"}"#)
            .is_some_and(|e| e.contains("temp_c")));
        assert!(
            output_violation(&schema, "just text").is_some(),
            "a non-JSON result is a plain string value"
        );
    }

    #[test]
    fn a_plain_string_result_satisfies_a_string_schema() {
        let schema = serde_json::json!({"type": "string"});
        assert_eq!(output_violation(&schema, "sunny"), None);
    }
}
