use std::collections::HashMap;

use crate::protocol::{OwnerKind, SessionOwner};

/// Who is acting, in descending privilege.
///
/// `ApiKey` and `Admin` are both machine-authenticated, and they differ in what
/// holds the credential: a program holds a key, a person logs in. Only a worker
/// answers the calls the engine hands out, so only `ApiKey` may.
#[derive(Debug, Clone)]
pub enum Caller {
    /// The engine itself.
    System { tenant_id: String },
    /// A program with an API key. Workers submit decisions as this.
    ApiKey { tenant_id: String, key_id: String },
    /// A person who logged in, through the CLI or the dashboard.
    Admin { tenant_id: String, user_id: String },
    /// An end user, holding a client token.
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
            | Caller::Admin { tenant_id, .. }
            | Caller::Frontend { tenant_id, .. } => tenant_id,
        }
    }

    /// What the credential names: a key id, or a person.
    pub fn subject(&self) -> Option<&str> {
        match self {
            Caller::System { .. } => None,
            Caller::ApiKey { key_id, .. } => Some(key_id),
            Caller::Admin { user_id, .. } | Caller::Frontend { user_id, .. } => Some(user_id),
        }
    }

    pub fn owner_kind(&self) -> OwnerKind {
        match self {
            Caller::System { .. } => OwnerKind::System,
            Caller::ApiKey { .. } => OwnerKind::ApiKey,
            Caller::Admin { .. } => OwnerKind::Admin,
            Caller::Frontend { .. } => OwnerKind::Frontend,
        }
    }

    /// Whether this caller is the owner of a session.
    ///
    /// The kind must agree as well as the name. An admin and an end user called
    /// the same thing are different owners, and their names come from different
    /// issuers, so one must never open the other's session.
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

    /// The names come from different issuers, so the same string is not the
    /// same person. An end user must not open an admin's session with it.
    #[test]
    fn one_kind_of_owner_does_not_match_another() {
        let name = "alex@example.test";
        assert!(frontend(name).owns(&owner(OwnerKind::Frontend, name)));
        assert!(!frontend(name).owns(&owner(OwnerKind::Admin, name)));
        assert!(!frontend(name).owns(&owner(OwnerKind::ApiKey, name)));
        assert!(!frontend(name).owns(&owner(OwnerKind::System, name)));
    }

    #[test]
    fn an_admin_owns_only_its_own_sessions() {
        let admin = Caller::Admin {
            tenant_id: "tenant-a".to_string(),
            user_id: "alex".to_string(),
        };
        assert!(admin.owns(&owner(OwnerKind::Admin, "alex")));
        assert!(!admin.owns(&owner(OwnerKind::Admin, "sam")));
        assert!(!admin.owns(&owner(OwnerKind::Frontend, "alex")));
    }

    /// A name is only a name inside its tenant.
    #[test]
    fn an_owner_in_another_tenant_is_not_this_one() {
        let mut other = owner(OwnerKind::Frontend, "alex");
        other.tenant_id = "tenant-b".to_string();
        assert!(!frontend("alex").owns(&other));
    }

    /// The engine names nobody, so it owns nothing. It is allowed past an
    /// ownership check by privilege, not by matching one.
    #[test]
    fn the_engine_owns_nothing() {
        let system = Caller::System {
            tenant_id: "tenant-a".to_string(),
        };
        assert!(!system.owns(&owner(OwnerKind::System, "anything")));
    }

    /// An owner stored before the kind existed was an end user's.
    #[test]
    fn a_stored_owner_without_a_kind_reads_as_an_end_user() {
        let stored = r#"{"tenant_id":"tenant-a","id":"alex"}"#;
        let owner: SessionOwner = serde_json::from_str(stored).unwrap();
        assert_eq!(owner.kind, OwnerKind::Frontend);
        assert!(frontend("alex").owns(&owner));
    }
}
