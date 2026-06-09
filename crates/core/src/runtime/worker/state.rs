use std::fmt;
use std::ops::Deref;

use base64::Engine;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Opaque worker state: bytes the engine stores but never interprets (workers
/// own the format). Serializes as base64; the encoding lives in the type, so a
/// field can't forget it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerState(pub Vec<u8>);

impl WorkerState {
    pub fn into_inner(self) -> Vec<u8> {
        self.0
    }
}

impl Serialize for WorkerState {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&base64::engine::general_purpose::STANDARD.encode(&self.0))
    }
}

impl<'de> Deserialize<'de> for WorkerState {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WorkerStateVisitor;

        impl<'de> Visitor<'de> for WorkerStateVisitor {
            type Value = WorkerState;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a base64 string")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<WorkerState, E> {
                base64::engine::general_purpose::STANDARD
                    .decode(v)
                    .map(WorkerState)
                    .map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(WorkerStateVisitor)
    }
}

impl Deref for WorkerState {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<u8>> for WorkerState {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl From<WorkerState> for Vec<u8> {
    fn from(state: WorkerState) -> Self {
        state.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_base64() {
        let json = serde_json::to_string(&WorkerState(vec![123, 34, 125])).unwrap();
        assert_eq!(json, "\"eyJ9\"");
    }

    #[test]
    fn deserializes_base64() {
        let from_b64: WorkerState = serde_json::from_str("\"eyJ9\"").unwrap();
        assert_eq!(from_b64.0, vec![123, 34, 125]);
    }

    #[test]
    fn rejects_byte_array() {
        assert!(serde_json::from_str::<WorkerState>("[123,34,125]").is_err());
    }
}
