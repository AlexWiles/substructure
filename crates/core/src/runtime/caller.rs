use std::collections::HashMap;

use crate::protocol::{OwnerKind, SessionOwner};

/// Who is acting, in descending privilege. A program holds a key, a person logs
/// in. Only a worker answers a decision, so only `ApiKey` may.
#[derive(Debug, Clone)]
pub enum Caller {
    System {
        tenant_id: String,
    },
    ApiKey {
        tenant_id: String,
        key_id: String,
    },
    Operator {
        tenant_id: String,
        user_id: String,
    },
    Frontend {
        tenant_id: String,
        user_id: String,
        attrs: HashMap<String, String>,
    },
}

impl Caller {
    pub fn tenant_id(&self) -> &str {
        match self {
            Caller::System { tenant_id }
            | Caller::ApiKey { tenant_id, .. }
            | Caller::Operator { tenant_id, .. }
            | Caller::Frontend { tenant_id, .. } => tenant_id,
        }
    }

    pub fn subject(&self) -> Option<&str> {
        match self {
            Caller::System { .. } => None,
            Caller::ApiKey { key_id, .. } => Some(key_id),
            Caller::Operator { user_id, .. } | Caller::Frontend { user_id, .. } => Some(user_id),
        }
    }

    pub fn owner_kind(&self) -> OwnerKind {
        match self {
            Caller::System { .. } => OwnerKind::System,
            Caller::ApiKey { .. } => OwnerKind::ApiKey,
            Caller::Operator { .. } => OwnerKind::Operator,
            Caller::Frontend { .. } => OwnerKind::Frontend,
        }
    }

    /// This caller as the owner of a session it starts. The counterpart of
    /// [`owns`](Self::owns): what this writes, that matches.
    pub fn as_owner(&self) -> SessionOwner {
        SessionOwner {
            tenant_id: self.tenant_id().to_string(),
            id: self.subject().map(str::to_string),
            kind: self.owner_kind(),
            // A caller says nothing about who reads the answers, so the safe
            // value stands; a transport that knows sets it on its own owner.
            audience: Default::default(),
            metadata: HashMap::new(),
        }
    }

    /// The kind must agree as well as the name. Names come from different
    /// issuers, so the same name is not the same owner.
    pub fn owns(&self, owner: &SessionOwner) -> bool {
        let Some(subject) = self.subject() else {
            return false;
        };
        owner.kind == self.owner_kind()
            && owner.tenant_id == self.tenant_id()
            && owner.id.as_deref() == Some(subject)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(kind: OwnerKind, id: &str) -> SessionOwner {
        SessionOwner {
            tenant_id: "tenant-a".to_string(),
            id: Some(id.to_string()),
            kind,
            audience: Default::default(),
            metadata: HashMap::new(),
        }
    }

    fn frontend(user_id: &str) -> Caller {
        Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            user_id: user_id.to_string(),
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn one_kind_of_owner_does_not_match_another() {
        let name = "alex@example.test";
        assert!(frontend(name).owns(&owner(OwnerKind::Frontend, name)));
        assert!(!frontend(name).owns(&owner(OwnerKind::Operator, name)));
        assert!(!frontend(name).owns(&owner(OwnerKind::ApiKey, name)));
        assert!(!frontend(name).owns(&owner(OwnerKind::System, name)));
    }

    #[test]
    fn an_operator_owns_only_its_own_sessions() {
        let operator = Caller::Operator {
            tenant_id: "tenant-a".to_string(),
            user_id: "alex".to_string(),
        };
        assert!(operator.owns(&owner(OwnerKind::Operator, "alex")));
        assert!(!operator.owns(&owner(OwnerKind::Operator, "sam")));
        assert!(!operator.owns(&owner(OwnerKind::Frontend, "alex")));
    }

    #[test]
    fn an_owner_in_another_tenant_is_not_this_one() {
        let mut other = owner(OwnerKind::Frontend, "alex");
        other.tenant_id = "tenant-b".to_string();
        assert!(!frontend("alex").owns(&other));
    }

    /// What a caller writes as an owner is what it matches as one.
    #[test]
    fn a_caller_owns_the_sessions_it_starts() {
        for caller in [
            frontend("alex"),
            Caller::Operator {
                tenant_id: "tenant-a".to_string(),
                user_id: "alex".to_string(),
            },
            Caller::ApiKey {
                tenant_id: "tenant-a".to_string(),
                key_id: "alex".to_string(),
            },
        ] {
            assert!(
                caller.owns(&caller.as_owner()),
                "{caller:?} does not own what it starts"
            );
        }
    }

    /// The engine names nobody. Privilege lets it past a check, not a match.
    #[test]
    fn the_engine_owns_nothing() {
        let system = Caller::System {
            tenant_id: "tenant-a".to_string(),
        };
        assert!(!system.owns(&owner(OwnerKind::System, "anything")));
    }

    #[test]
    fn a_stored_owner_without_a_kind_reads_as_an_end_user() {
        let stored = r#"{"tenant_id":"tenant-a","id":"alex"}"#;
        let owner: SessionOwner = serde_json::from_str(stored).unwrap();
        assert_eq!(owner.kind, OwnerKind::Frontend);
        assert!(frontend("alex").owns(&owner));
    }
}
