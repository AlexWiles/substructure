use std::fmt;
use std::ops::Deref;

use base64::Engine;
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Opaque worker state: bytes the engine stores but never interprets (workers
/// own the format). Serializes as base64; the encoding lives in the type, so a
/// field can't forget it. Deserialize tolerates a legacy byte-array form so
/// already-persisted state still loads.
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
                f.write_str("a base64 string or a byte array")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<WorkerState, E> {
                base64::engine::general_purpose::STANDARD
                    .decode(v)
                    .map(WorkerState)
                    .map_err(de::Error::custom)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<WorkerState, A::Error> {
                let mut bytes = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(byte) = seq.next_element::<u8>()? {
                    bytes.push(byte);
                }
                Ok(WorkerState(bytes))
            }
        }

        deserializer.deserialize_any(WorkerStateVisitor)
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
    fn deserializes_base64_and_legacy_byte_array() {
        let from_b64: WorkerState = serde_json::from_str("\"eyJ9\"").unwrap();
        let from_array: WorkerState = serde_json::from_str("[123,34,125]").unwrap();
        assert_eq!(from_b64.0, vec![123, 34, 125]);
        assert_eq!(from_array.0, vec![123, 34, 125]);
    }
}
