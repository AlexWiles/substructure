use rand::RngExt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! hex_id {
    ($name:ident, $len:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name([u8; $len]);

        impl $name {
            pub fn random() -> Self {
                Self(rand::rng().random())
            }

            pub fn as_bytes(&self) -> [u8; $len] {
                self.0
            }

            pub fn from_bytes(bytes: [u8; $len]) -> Self {
                Self(bytes)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&hex::encode(self.0))
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let s = String::deserialize(deserializer)?;
                let bytes: [u8; $len] = hex::decode(s)
                    .map_err(serde::de::Error::custom)?
                    .try_into()
                    .map_err(|_| {
                        serde::de::Error::custom(concat!("expected ", stringify!($len), " bytes"))
                    })?;
                Ok(Self(bytes))
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&hex::encode(self.0))
            }
        }
    };
}

hex_id!(TraceId, 16);
hex_id!(SpanId, 8);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub trace_flags: u8,
    pub trace_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl SpanContext {
    pub fn root() -> Self {
        SpanContext {
            trace_id: TraceId::random(),
            span_id: SpanId::random(),
            parent_span_id: None,
            trace_flags: 1,
            trace_state: None,
            name: None,
        }
    }

    pub fn child(&self, name: impl Into<String>) -> Self {
        SpanContext {
            trace_id: self.trace_id,
            span_id: SpanId::random(),
            parent_span_id: Some(self.span_id),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
            name: Some(name.into()),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Create a new root-level span within an existing trace.
    ///
    /// Use this when continuing work on an aggregate whose trace_id is known
    /// but the original parent span is not (e.g. a new HTTP request to an
    /// existing session).
    pub fn in_trace(trace_id: TraceId, name: impl Into<String>) -> Self {
        SpanContext {
            trace_id,
            span_id: SpanId::random(),
            parent_span_id: None,
            trace_flags: 1,
            trace_state: None,
            name: Some(name.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_id_roundtrip() {
        let id = TraceId::random();
        let json = serde_json::to_string(&id).unwrap();
        // 32 hex chars + 2 quotes
        assert_eq!(json.len(), 34);
        let parsed: TraceId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn span_id_roundtrip() {
        let id = SpanId::random();
        let json = serde_json::to_string(&id).unwrap();
        // 16 hex chars + 2 quotes
        assert_eq!(json.len(), 18);
        let parsed: SpanId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn span_context_roundtrip() {
        let ctx = SpanContext::root();
        let json = serde_json::to_string(&ctx).unwrap();
        let parsed: SpanContext = serde_json::from_str(&json).unwrap();
        assert_eq!(ctx.trace_id, parsed.trace_id);
        assert_eq!(ctx.span_id, parsed.span_id);
    }

    #[test]
    fn display_matches_serde() {
        let id = TraceId::random();
        let display = id.to_string();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(format!("\"{display}\""), json);
    }
}
