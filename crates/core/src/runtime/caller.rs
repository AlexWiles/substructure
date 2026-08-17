use std::collections::HashMap;

use crate::protocol::{SessionOwner, Subject};

/// Who is acting, in descending privilege. A program holds a key, a person logs
/// in. Only a worker answers a decision, so only `ApiKey` may.
///
/// Separate from who a session is *for*: a key may act on behalf of one of the
/// project's own users, and a person acting on their own behalf is the special
/// case rather than the rule.
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
        subject: Subject,
    },
    Frontend {
        tenant_id: String,
        subject: Subject,
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

    /// The person this caller is, where it is one. A key is a program and
    /// names nobody, however it may name someone else on a request.
    pub fn subject(&self) -> Option<&Subject> {
        match self {
            Caller::System { .. } | Caller::ApiKey { .. } => None,
            Caller::Operator { subject, .. } | Caller::Frontend { subject, .. } => Some(subject),
        }
    }

    /// Whether this caller is the person a session belongs to. The issuer is
    /// half the comparison: one source's `bob` is not another's.
    pub fn owns(&self, owner: &SessionOwner) -> bool {
        let Some(subject) = self.subject() else {
            return false;
        };
        owner.tenant_id == self.tenant_id() && owner.requester.subject.as_ref() == Some(subject)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Issuer;
    use crate::protocol::Requester;

    fn owner(issuer: Issuer, id: &str) -> SessionOwner {
        SessionOwner {
            tenant_id: "tenant-a".to_string(),
            requester: Requester::private(Subject::new(issuer, id)),
            metadata: HashMap::new(),
        }
    }

    fn frontend(id: &str) -> Caller {
        Caller::Frontend {
            tenant_id: "tenant-a".to_string(),
            subject: Subject::new(Issuer::app(), id),
            attrs: HashMap::new(),
        }
    }

    /// The point of naming the source: one issuer's `alex` is not another's,
    /// so an end user cannot reach an operator's session by sharing an id.
    #[test]
    fn the_same_id_from_another_source_is_another_person() {
        assert!(frontend("alex").owns(&owner(Issuer::app(), "alex")));
        assert!(!frontend("alex").owns(&owner(Issuer::operator(), "alex")));
        assert!(!frontend("alex").owns(&owner(Issuer::slack(), "alex")));
        assert!(!frontend("alex").owns(&owner(Issuer::app(), "sam")));
    }

    #[test]
    fn a_session_in_another_tenant_is_never_ours() {
        let other = SessionOwner {
            tenant_id: "tenant-b".to_string(),
            requester: Requester::private(Subject::new(Issuer::app(), "alex")),
            metadata: HashMap::new(),
        };
        assert!(!frontend("alex").owns(&other));
    }

    /// A key is a program: it may act for someone, and is nobody itself.
    #[test]
    fn a_machine_owns_nothing() {
        let key = Caller::ApiKey {
            tenant_id: "tenant-a".to_string(),
            key_id: "key-1".to_string(),
        };
        let system = Caller::System {
            tenant_id: "tenant-a".to_string(),
        };
        assert!(key.subject().is_none());
        assert!(!key.owns(&owner(Issuer::app(), "key-1")));
        assert!(!system.owns(&owner(Issuer::app(), "anything")));
    }

    /// An owner naming no source names no person, so nobody owns it and no
    /// personal credential is within its reach.
    #[test]
    fn an_owner_without_a_subject_is_nobodys() {
        let stored = r#"{"tenant_id":"tenant-a"}"#;
        let owner: SessionOwner = serde_json::from_str(stored).unwrap();
        assert!(owner.requester.subject.is_none());
        assert!(!frontend("alex").owns(&owner));
    }
}
