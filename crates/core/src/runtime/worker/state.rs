use std::ops::Deref;

use serde::{Deserialize, Serialize};

/// Opaque worker state: JSON the engine stores but never interprets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerState(pub serde_json::Value);

impl WorkerState {
    pub fn into_inner(self) -> serde_json::Value {
        self.0
    }
}

impl Deref for WorkerState {
    type Target = serde_json::Value;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<serde_json::Value> for WorkerState {
    fn from(value: serde_json::Value) -> Self {
        Self(value)
    }
}

impl From<WorkerState> for serde_json::Value {
    fn from(state: WorkerState) -> Self {
        state.0
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn serializes_as_raw_json() {
        let json = serde_json::to_string(&WorkerState(json!({ "n": 1 }))).unwrap();
        assert_eq!(json, r#"{"n":1}"#);
    }

    #[test]
    fn old_base64_string_passes_through() {
        let state: WorkerState = serde_json::from_str("\"eyJ9\"").unwrap();
        assert_eq!(state.0, json!("eyJ9"));
        assert_eq!(serde_json::to_string(&state).unwrap(), "\"eyJ9\"");
    }

    #[test]
    fn default_is_json_null() {
        assert_eq!(WorkerState::default().0, serde_json::Value::Null);
    }
}
