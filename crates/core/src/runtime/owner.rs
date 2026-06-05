use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The end user a session belongs to — the subject work is performed on behalf
/// of. Stamped onto a session at creation (`SessionCreated.owner`) and persisted
/// as the durable owner record; a [`Caller::Frontend`](crate::Caller) is
/// authorized against it (its `user_id` must match `id`). Distinct from the
/// transient [`Caller`](crate::Caller), which is *who is acting* on a given
/// request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionOwner {
    pub tenant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}
